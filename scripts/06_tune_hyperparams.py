#!/usr/bin/env python
"""
06_tune_hyperparams.py - Grid Search 하이퍼파라미터 튜닝 스크립트

LOOCV 기반 Grid Search를 수행하여 최적의 하이퍼파라미터 조합을 찾습니다.

Usage:
    python scripts/06_tune_hyperparams.py --config config/skin3.yaml
    python scripts/06_tune_hyperparams.py --config config/scp1884.yaml --gpu 0
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from itertools import product
import copy

# Add project root to path BEFORE other imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
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

        # Embedding IDs (study or organ)
        embedding_ids = None
        if embedding_mapping:
            embedding_ids = torch.tensor(
                [embedding_mapping.get(int(s), 0) for s in split_adata.obs[sample_col].values],
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
    verbose: bool = False
) -> dict:
    """
    주어진 하이퍼파라미터로 전체 LOOCV를 수행하고 평균 메트릭 반환.
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

    all_metrics = []

    for fold_info in splitter.split(sample_ids, labels, sample_names):
        fold_idx = fold_info.fold_idx

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

        # Train fold
        result = trainer.train_fold(
            train_bag_dl=train_bag_dl,
            train_instance_dl=train_instance_dl,
            test_bag_dl=test_bag_dl,
            n_epochs=epochs,
            learning_rate=learning_rate,
            encoder_learning_rate=encoder_learning_rate,
            use_early_stopping=False,
            fold_idx=fold_idx,
            test_sample_name=fold_info.test_sample_name or f"Sample_{fold_idx}"
        )

        all_metrics.append(result.metrics)

        if verbose:
            print(f"  Fold {fold_idx + 1}/{n_folds}: AUC={result.metrics['auc']:.4f}")

    # Aggregate metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[f"mean_{key}"] = np.mean(values)
        avg_metrics[f"std_{key}"] = np.std(values)

    return avg_metrics


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
    print(f"{'='*60}\n")

    # Run grid search
    results = []
    best_score = -float('inf')
    best_params = None

    for i, (lr, enc_lr, epochs, ratio_lambda) in enumerate(param_grid):
        print(f"\n[{i+1}/{n_combinations}] Testing: lr={lr}, enc_lr={enc_lr}, epochs={epochs}, ratio_lambda={ratio_lambda}")

        try:
            metrics = run_loocv_for_hyperparams(
                adata=adata,
                config=config,
                device=device,
                embedding_mapping=embedding_mapping,
                learning_rate=lr,
                encoder_learning_rate=enc_lr,
                epochs=epochs,
                disease_ratio_lambda=ratio_lambda,
                verbose=args.verbose
            )

            # Store result
            result = {
                'learning_rate': lr,
                'encoder_learning_rate': enc_lr,
                'epochs': epochs,
                'disease_ratio_lambda': ratio_lambda,
                **metrics
            }
            results.append(result)

            # Check if best
            metric_key = f"mean_{tuning.metric}"
            score = metrics.get(metric_key, 0)
            std_key = f"std_{tuning.metric}"
            std = metrics.get(std_key, 0)

            print(f"  Result: {tuning.metric}={score:.4f} ± {std:.4f}")

            if score > best_score:
                best_score = score
                best_params = (lr, enc_lr, epochs, ratio_lambda)
                print(f"  *** New best! ***")

        except Exception as e:
            import traceback
            print(f"  Error: {e}")
            traceback.print_exc()
            continue

    # Save results
    results_df = pd.DataFrame(results)
    results_path = output_dir / tuning.results_file
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Grid Search Results Summary")
    print(f"{'='*60}")

    if best_params:
        lr, enc_lr, epochs, ratio_lambda = best_params
        print(f"\nBest hyperparameters (by mean {tuning.metric}):")
        print(f"  - learning_rate: {lr}")
        print(f"  - encoder_learning_rate: {enc_lr}")
        print(f"  - epochs: {epochs}")
        print(f"  - disease_ratio_lambda: {ratio_lambda}")
        print(f"  - Best {tuning.metric}: {best_score:.4f}")

    # Top 5 configurations
    if len(results) > 0:
        results_df_sorted = results_df.sort_values(f"mean_{tuning.metric}", ascending=False)
        print(f"\nTop 5 configurations:")
        for idx, row in results_df_sorted.head(5).iterrows():
            print(f"  {idx+1}. lr={row['learning_rate']}, enc_lr={row['encoder_learning_rate']}, "
                  f"epochs={int(row['epochs'])}, ratio_lambda={row['disease_ratio_lambda']} "
                  f"-> {tuning.metric}={row[f'mean_{tuning.metric}']:.4f}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
