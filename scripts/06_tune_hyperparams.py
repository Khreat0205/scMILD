#!/usr/bin/env python
"""
06_tune_hyperparams.py - Grid Search 하이퍼파라미터 튜닝 스크립트

LOOCV 기반 Grid Search를 수행하여 최적의 하이퍼파라미터 조합을 찾습니다.

Features:
    - 실시간 CSV 저장: 각 조합 완료 시마다 tuning_results.csv 업데이트
    - Top-K 모델 저장: 상위 K개 조합의 fold별 모델 저장
    - best_params.yaml: 최적 하이퍼파라미터 별도 저장

Usage:
    python scripts/06_tune_hyperparams.py --config config/skin3.yaml
    python scripts/06_tune_hyperparams.py --config config/scp1884.yaml --gpu 0
"""

import os
import sys
import argparse
import gc
from pathlib import Path
from datetime import datetime
from itertools import product
import copy
import heapq

# Add project root to path BEFORE other imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
import yaml
from torch.utils.data import DataLoader

from src.config import load_config, ScMILDConfig
from src.data import (
    load_adata_with_subset, print_adata_summary,
    LOOCVSplitter, get_sample_info_from_adata,
    MilDataset, InstanceDataset, collate_mil, create_instance_dataset_with_bag_labels
)
from src.models import (
    GatedAttentionModule, TeacherBranch, StudentBranch,
    VQEncoderWrapperConditional
)
from src.training import (
    MILTrainer, calculate_disease_ratio_from_dataloader
)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def create_dataloaders(
    adata,
    train_samples: list,
    test_samples: list,
    device: torch.device,
    config: ScMILDConfig,
    embedding_mapping: dict
) -> tuple:
    """Create train and test dataloaders for a fold."""

    sample_col = config.data.columns.sample_id
    label_col = config.data.columns.disease_label
    embedding_col = config.data.conditional_embedding.encoded_column
    embedding_source_col = config.data.conditional_embedding.column  # e.g., "study"
    embedding_mapping_path = config.data.conditional_embedding.mapping_path

    # If encoded column doesn't exist, create it from source column using pretrain mapping
    if embedding_col not in adata.obs.columns and embedding_source_col in adata.obs.columns:
        if embedding_mapping_path and Path(embedding_mapping_path).exists():
            # Load pretrain mapping (id -> name) and invert to (name -> id)
            from src.data import load_conditional_mapping
            id_to_name = load_conditional_mapping(embedding_mapping_path)
            name_to_id = {v: k for k, v in id_to_name.items()}
            print(f"Using pretrain {embedding_source_col} mapping from: {embedding_mapping_path}")
            print(f"  Mapping: {name_to_id}")
            adata.obs[embedding_col] = adata.obs[embedding_source_col].map(name_to_id)
        else:
            # Fallback: create from category codes (WARNING: may not match pretrain)
            print(f"WARNING: Creating '{embedding_col}' from '{embedding_source_col}' without pretrain mapping!")
            print(f"  This may cause inconsistency with pretrained encoder.")
            adata.obs[embedding_col] = adata.obs[embedding_source_col].astype('category').cat.codes

    def _create_datasets(sample_list):
        # Filter cells for these samples
        mask = adata.obs[sample_col].isin(sample_list)
        split_adata = adata[mask]

        # Extract data
        if hasattr(split_adata.X, 'toarray'):
            data = torch.tensor(split_adata.X.toarray(), dtype=torch.float32, device=device)
        else:
            data = torch.tensor(np.array(split_adata.X), dtype=torch.float32, device=device)

        # Sample IDs
        sample_ids = torch.tensor(
            split_adata.obs[sample_col].values, dtype=torch.long, device=device
        )

        # Labels (unique per sample)
        unique_samples = sorted(split_adata.obs[sample_col].unique())
        sample_labels = split_adata.obs.groupby(sample_col)[label_col].first()
        labels = torch.tensor(
            [sample_labels[s] for s in unique_samples],
            dtype=torch.long, device=device
        )

        # Instance labels
        instance_labels = torch.tensor(
            split_adata.obs[label_col].values, dtype=torch.long, device=device
        )

        # Embedding IDs (study or organ) - use direct column value
        embedding_ids = None
        if embedding_col in split_adata.obs.columns:
            embedding_ids = torch.tensor(
                split_adata.obs[embedding_col].values.astype(int),
                dtype=torch.long, device=device
            )

        return data, sample_ids, labels, instance_labels, embedding_ids

    # Create train datasets
    train_data, train_ids, train_labels, train_instance_labels, train_embedding_ids = \
        _create_datasets(train_samples)

    train_mil = MilDataset(
        train_data, train_ids, train_labels, train_instance_labels, train_embedding_ids
    )
    train_instance = InstanceDataset(
        train_data, train_ids, train_labels, train_instance_labels, train_embedding_ids
    )
    train_instance_with_bag = create_instance_dataset_with_bag_labels(train_instance, device)

    # Create test datasets
    test_data, test_ids, test_labels, test_instance_labels, test_embedding_ids = \
        _create_datasets(test_samples)

    test_mil = MilDataset(
        test_data, test_ids, test_labels, test_instance_labels, test_embedding_ids
    )

    # Create dataloaders
    train_bag_dl = DataLoader(
        train_mil,
        batch_size=config.mil.training.batch_size,
        shuffle=True,
        collate_fn=collate_mil
    )
    train_instance_dl = DataLoader(
        train_instance_with_bag,
        batch_size=256,
        shuffle=True
    )
    test_bag_dl = DataLoader(
        test_mil,
        batch_size=len(test_samples),
        shuffle=False,
        collate_fn=collate_mil
    )

    return train_bag_dl, train_instance_dl, test_bag_dl


def create_models(config: ScMILDConfig, device: torch.device, encoder_path: str):
    """Create MIL models."""

    # Load pretrained encoder
    checkpoint = torch.load(encoder_path, map_location=device)
    model_config = checkpoint.get('config', {})

    from src.models.autoencoder import VQ_AENB_Conditional

    encoder_model = VQ_AENB_Conditional(
        input_dim=model_config['input_dim'],
        latent_dim=model_config['latent_dim'],
        device=device,
        hidden_layers=model_config['hidden_layers'],
        n_studies=model_config['n_studies'],
        study_emb_dim=model_config.get('study_emb_dim', 16),
        num_codes=model_config.get('num_codes', 1024),
    )
    encoder_model.load_state_dict(checkpoint['model_state_dict'])
    encoder_model.to(device)

    # Wrap encoder
    model_encoder = VQEncoderWrapperConditional(
        encoder_model,
        use_projection=config.mil.use_projection,
        projection_dim=config.mil.projection_dim
    )
    model_encoder.freeze_encoder()
    model_encoder.to(device)

    # Attention module
    input_dim = model_encoder.input_dims
    attention_module = GatedAttentionModule(
        L=input_dim,
        D=config.mil.attention_dim,
        K=1
    ).to(device)

    # Teacher branch
    model_teacher = TeacherBranch(
        input_dims=input_dim,
        latent_dims=config.mil.latent_dim,
        attention_module=attention_module,
        num_classes=config.mil.num_classes
    ).to(device)

    # Student branch
    model_student = StudentBranch(
        input_dims=input_dim,
        latent_dims=config.mil.latent_dim,
        num_classes=config.mil.num_classes
    ).to(device)

    return model_teacher, model_student, model_encoder


def run_loocv_for_hyperparams(
    adata,
    config: ScMILDConfig,
    device: torch.device,
    embedding_mapping: dict,
    learning_rate: float,
    encoder_learning_rate: float,
    epochs: int,
    disease_ratio_lambda: float,
    verbose: bool = False,
    return_models: bool = False
) -> dict:
    """
    주어진 하이퍼파라미터로 전체 LOOCV를 수행하고 평균 메트릭 반환.

    Args:
        return_models: True이면 fold별 모델들도 반환

    Returns:
        metrics dict, 또는 return_models=True이면 (metrics, fold_models) 튜플
        fold_models: list of dict with keys 'teacher', 'student', 'encoder'
    """
    sample_col = config.data.columns.sample_id
    label_col = config.data.columns.disease_label
    sample_name_col = config.data.columns.sample_name

    # Get sample info
    sample_ids, labels, sample_names = get_sample_info_from_adata(
        adata,
        sample_col=sample_col,
        label_col=label_col,
        sample_name_col=sample_name_col
    )

    # Create splitter
    splitter = LOOCVSplitter(random_seed=config.splitting.random_seed)
    n_folds = splitter.get_n_splits(sample_ids)

    all_results = []
    fold_models = [] if return_models else None

    # Variables for cleanup
    model_teacher = None
    model_student = None
    model_encoder = None
    train_bag_dl = None
    train_instance_dl = None
    test_bag_dl = None

    for fold_info in splitter.split(sample_ids, labels, sample_names):
        fold_idx = fold_info.fold_idx

        # Clean up previous fold's GPU memory
        if fold_idx > 0:
            del model_teacher, model_student, model_encoder
            del train_bag_dl, train_instance_dl, test_bag_dl
            gc.collect()
            torch.cuda.empty_cache()

        # Create fresh models for each fold
        model_teacher, model_student, model_encoder = create_models(
            config, device, config.paths.pretrained_encoder
        )

        # Create dataloaders
        train_bag_dl, train_instance_dl, test_bag_dl = create_dataloaders(
            adata,
            fold_info.train_samples,
            fold_info.test_samples,
            device,
            config,
            embedding_mapping
        )

        # Calculate disease ratio if lambda > 0
        disease_ratio = None
        if disease_ratio_lambda > 0:
            disease_ratio = calculate_disease_ratio_from_dataloader(
                train_instance_dl,
                model_encoder,
                device,
                alpha=config.mil.loss.disease_ratio_reg.alpha,
                beta=config.mil.loss.disease_ratio_reg.beta,
                use_conditional=True
            )

        # Create trainer
        trainer = MILTrainer(
            model_teacher=model_teacher,
            model_student=model_student,
            model_encoder=model_encoder,
            device=device,
            use_conditional_ae=True,
            student_optimize_period=config.mil.student.optimize_period,
            student_loss_weight_neg=config.mil.loss.negative_weight,
            disease_ratio=disease_ratio,
            ratio_reg_lambda=disease_ratio_lambda
        )

        # Train fold (skip_fold_metrics=True for LOOCV)
        result = trainer.train_fold(
            train_bag_dl=train_bag_dl,
            train_instance_dl=train_instance_dl,
            test_bag_dl=test_bag_dl,
            n_epochs=epochs,
            learning_rate=learning_rate,
            encoder_learning_rate=encoder_learning_rate,
            use_early_stopping=False,
            fold_idx=fold_idx,
            test_sample_name=fold_info.test_sample_name or f"Sample_{fold_idx}",
            skip_fold_metrics=True,  # LOOCV: skip per-fold metrics
        )

        all_results.append(result)

        # Save models if requested
        if return_models:
            fold_models.append({
                'teacher': copy.deepcopy(model_teacher.state_dict()),
                'student': copy.deepcopy(model_student.state_dict()),
                'encoder': copy.deepcopy(model_encoder.state_dict()),
                'fold_idx': fold_idx,
                'test_sample': fold_info.test_sample_name or f"Sample_{fold_idx}",
            })

        if verbose:
            # For LOOCV, show prediction instead of meaningless fold AUC
            pred_label = "Disease" if result.y_pred_proba[0] >= 0.5 else "Control"
            true_label = "Disease" if result.y_true[0] == 1 else "Control"
            correct = "✓" if pred_label == true_label else "✗"
            print(f"  Fold {fold_idx + 1}/{n_folds}: prob={result.y_pred_proba[0]:.4f} "
                  f"(pred={pred_label}, true={true_label}) {correct}")

    # Final cleanup after all folds
    del model_teacher, model_student, model_encoder
    del train_bag_dl, train_instance_dl, test_bag_dl
    gc.collect()
    torch.cuda.empty_cache()

    # For LOOCV: concatenate all predictions and compute overall metrics
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    all_y_true = np.concatenate([r.y_true for r in all_results])
    all_y_pred_proba = np.concatenate([r.y_pred_proba for r in all_results])
    all_y_pred = (all_y_pred_proba >= 0.5).astype(int)

    # Compute overall metrics (this is the correct way for LOOCV)
    overall_metrics = {
        'mean_auc': roc_auc_score(all_y_true, all_y_pred_proba),
        'mean_accuracy': accuracy_score(all_y_true, all_y_pred),
        'mean_f1_score': f1_score(all_y_true, all_y_pred, zero_division=0),
        'std_auc': 0.0,  # No std for concatenated metrics
        'std_accuracy': 0.0,
        'std_f1_score': 0.0,
    }

    if return_models:
        return overall_metrics, fold_models
    return overall_metrics


class TopKTracker:
    """Top-K 조합을 추적하고 모델 저장을 관리하는 클래스"""

    def __init__(self, k: int, output_dir: Path, metric_key: str = "mean_auc"):
        self.k = k
        self.output_dir = output_dir
        self.metric_key = metric_key
        self.models_dir = output_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Min-heap for top-K tracking (score, config_id, params)
        self.top_k_heap = []
        self.saved_configs = {}  # config_id -> (params, score)

    def update(self, config_id: int, params: dict, score: float, fold_models: list):
        """새 결과로 Top-K 업데이트. 필요시 모델 저장/삭제."""
        if self.k <= 0:
            return

        if len(self.top_k_heap) < self.k:
            # 아직 K개 미만: 무조건 추가
            heapq.heappush(self.top_k_heap, (score, config_id))
            self._save_models(config_id, params, fold_models)
            self.saved_configs[config_id] = (params, score)
        elif score > self.top_k_heap[0][0]:
            # 현재 최소보다 높음: 최소 제거하고 새로 추가
            _, removed_id = heapq.heapreplace(self.top_k_heap, (score, config_id))
            self._delete_models(removed_id)
            del self.saved_configs[removed_id]

            self._save_models(config_id, params, fold_models)
            self.saved_configs[config_id] = (params, score)

    def _save_models(self, config_id: int, params: dict, fold_models: list):
        """fold별 모델 저장"""
        config_dir = self.models_dir / f"config_{config_id:03d}"
        config_dir.mkdir(parents=True, exist_ok=True)

        # params.yaml 저장
        with open(config_dir / "params.yaml", "w") as f:
            yaml.dump(params, f, default_flow_style=False)

        # 각 fold 모델 저장
        for fold_data in fold_models:
            fold_idx = fold_data['fold_idx']
            fold_dir = config_dir / f"fold_{fold_idx:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            torch.save(fold_data['teacher'], fold_dir / "teacher.pth")
            torch.save(fold_data['student'], fold_dir / "student.pth")
            torch.save(fold_data['encoder'], fold_dir / "encoder.pth")

            # fold 메타 정보
            with open(fold_dir / "info.yaml", "w") as f:
                yaml.dump({
                    'fold_idx': fold_idx,
                    'test_sample': fold_data['test_sample']
                }, f)

    def _delete_models(self, config_id: int):
        """모델 디렉토리 삭제"""
        import shutil
        config_dir = self.models_dir / f"config_{config_id:03d}"
        if config_dir.exists():
            shutil.rmtree(config_dir)

    def get_top_k_configs(self) -> list:
        """Top-K 설정 반환 (점수 내림차순)"""
        return sorted(
            [(cid, params, score) for cid, (params, score) in self.saved_configs.items()],
            key=lambda x: -x[2]
        )


def save_results_csv(results: list, output_path: Path):
    """결과를 CSV로 저장 (실시간 업데이트용)"""
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)


def save_best_params(best_params: dict, best_score: float, metric: str, output_path: Path):
    """최적 하이퍼파라미터를 YAML로 저장"""
    best_config = {
        'best_hyperparameters': best_params,
        'best_score': {
            'metric': metric,
            'value': float(best_score)
        },
        'timestamp': datetime.now().isoformat()
    }
    with open(output_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(description="Grid Search Hyperparameter Tuning for scMILD")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set seed
    set_seed(args.seed)

    # Setup device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.paths.output_root) / f"tuning_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load data
    print(f"\nLoading data...")
    adata = load_adata_with_subset(
        whole_adata_path=config.data.whole_adata_path,
        subset_enabled=config.data.subset.enabled,
        subset_column=config.data.subset.column,
        subset_values=config.data.subset.values,
        cache_dir=config.data.subset.cache_dir,
        use_cache=config.data.subset.use_cache,
    )
    print_adata_summary(adata, "Loaded Data")

    # Load embedding mapping
    embedding_col = config.data.conditional_embedding.encoded_column
    embedding_mapping = {}
    if embedding_col in adata.obs.columns:
        mapping_df = adata.obs[[config.data.columns.sample_id, embedding_col]].drop_duplicates()
        embedding_mapping = dict(zip(
            mapping_df[config.data.columns.sample_id].astype(int),
            mapping_df[embedding_col].astype(int)
        ))

    # Get search space from config
    tuning = config.tuning
    lr_values = tuning.learning_rate
    enc_lr_values = tuning.encoder_learning_rate
    epoch_values = tuning.epochs
    ratio_lambda_values = tuning.disease_ratio_lambda
    save_top_k = tuning.save_top_k

    # Generate all combinations
    param_grid = list(product(lr_values, enc_lr_values, epoch_values, ratio_lambda_values))
    n_combinations = len(param_grid)

    print(f"\n{'='*60}")
    print(f"Grid Search Hyperparameter Tuning")
    print(f"{'='*60}")
    print(f"Search space:")
    print(f"  - learning_rate: {lr_values}")
    print(f"  - encoder_learning_rate: {enc_lr_values}")
    print(f"  - epochs: {epoch_values}")
    print(f"  - disease_ratio_lambda: {ratio_lambda_values}")
    print(f"Total combinations: {n_combinations}")
    print(f"Evaluation metric: {tuning.metric}")
    print(f"Save top-K models: {save_top_k}")
    print(f"{'='*60}\n")

    # Initialize tracking
    results = []
    results_path = output_dir / tuning.results_file
    best_params_path = output_dir / "best_params.yaml"
    metric_key = f"mean_{tuning.metric}"

    # Top-K tracker for model saving
    top_k_tracker = TopKTracker(
        k=save_top_k,
        output_dir=output_dir,
        metric_key=metric_key
    )

    best_score = -float('inf')
    best_params = None

    for i, (lr, enc_lr, epochs, ratio_lambda) in enumerate(param_grid):
        print(f"\n[{i+1}/{n_combinations}] Testing: lr={lr}, enc_lr={enc_lr}, epochs={epochs}, ratio_lambda={ratio_lambda}")

        params = {
            'learning_rate': lr,
            'encoder_learning_rate': enc_lr,
            'epochs': epochs,
            'disease_ratio_lambda': ratio_lambda
        }

        try:
            # return_models=True if we need to save models
            need_models = save_top_k > 0
            result_data = run_loocv_for_hyperparams(
                adata=adata,
                config=config,
                device=device,
                embedding_mapping=embedding_mapping,
                learning_rate=lr,
                encoder_learning_rate=enc_lr,
                epochs=epochs,
                disease_ratio_lambda=ratio_lambda,
                verbose=args.verbose,
                return_models=need_models
            )

            if need_models:
                metrics, fold_models = result_data
            else:
                metrics = result_data
                fold_models = None

            # Store result
            result = {
                'config_id': i,
                **params,
                **metrics
            }
            results.append(result)

            # Check if best
            score = metrics.get(metric_key, 0)
            std_key = f"std_{tuning.metric}"
            std = metrics.get(std_key, 0)

            print(f"  Result: {tuning.metric}={score:.4f} ± {std:.4f}")

            if score > best_score:
                best_score = score
                best_params = params.copy()
                print(f"  *** New best! ***")

            # Update Top-K tracker
            if fold_models is not None:
                top_k_tracker.update(i, params, score, fold_models)

            # Save results CSV after each combination (실시간 저장)
            save_results_csv(results, results_path)
            print(f"  (Results saved to {results_path})")

        except Exception as e:
            import traceback
            print(f"  Error: {e}")
            traceback.print_exc()
            # 에러 발생해도 지금까지의 결과는 저장
            save_results_csv(results, results_path)
            continue

    # Save best params
    if best_params:
        save_best_params(best_params, best_score, tuning.metric, best_params_path)
        print(f"\nBest params saved to: {best_params_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Grid Search Results Summary")
    print(f"{'='*60}")

    if best_params:
        print(f"\nBest hyperparameters (by mean {tuning.metric}):")
        for k, v in best_params.items():
            print(f"  - {k}: {v}")
        print(f"  - Best {tuning.metric}: {best_score:.4f}")

    # Top 5 configurations
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        results_df_sorted = results_df.sort_values(metric_key, ascending=False)
        print(f"\nTop 5 configurations:")
        for rank, (_, row) in enumerate(results_df_sorted.head(5).iterrows(), 1):
            print(f"  {rank}. lr={row['learning_rate']}, enc_lr={row['encoder_learning_rate']}, "
                  f"epochs={int(row['epochs'])}, ratio_lambda={row['disease_ratio_lambda']} "
                  f"-> {tuning.metric}={row[metric_key]:.4f}")

    # Top-K saved models info
    if save_top_k > 0:
        print(f"\nSaved Top-{save_top_k} model configurations:")
        for config_id, params, score in top_k_tracker.get_top_k_configs():
            print(f"  - config_{config_id:03d}: {tuning.metric}={score:.4f}")
        print(f"  Models saved in: {top_k_tracker.models_dir}")

    print(f"\n{'='*60}")
    print(f"Output files:")
    print(f"  - {results_path}")
    print(f"  - {best_params_path}")
    if save_top_k > 0:
        print(f"  - {top_k_tracker.models_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
