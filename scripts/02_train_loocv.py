#!/usr/bin/env python
"""
02_train_loocv.py - LOOCV 기반 MIL 학습 스크립트

Usage:
    python scripts/02_train_loocv.py --config config/skin3.yaml
    python scripts/02_train_loocv.py --config config/scp1884.yaml --gpu 0
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, ScMILDConfig
from src.data import (
    load_adata, load_adata_with_subset, preprocess_adata, print_adata_summary,
    LOOCVSplitter, get_sample_info_from_adata, print_split_summary,
    MilDataset, InstanceDataset, collate_mil, create_instance_dataset_with_bag_labels
)
from src.models import (
    GatedAttentionModule, TeacherBranch, StudentBranch,
    VQEncoderWrapperConditional
)
from src.training import (
    MILTrainer, compute_metrics, MetricsLogger,
    calculate_disease_ratio_from_dataloader, print_disease_ratio_summary
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
            # Load pretrain mapping (study_id -> study_name) and invert to (study_name -> study_id)
            from src.data import load_study_mapping
            id_to_name = load_study_mapping(embedding_mapping_path)
            name_to_id = {v: k for k, v in id_to_name.items()}
            print(f"Using pretrain study mapping from: {embedding_mapping_path}")
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


def main():
    parser = argparse.ArgumentParser(description="Train scMILD with LOOCV")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
    output_dir = Path(config.paths.output_root) / f"loocv_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load data (with optional subset)
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

    # Get sample info
    sample_ids, labels, sample_names = get_sample_info_from_adata(
        adata,
        sample_col=config.data.columns.sample_id,
        label_col=config.data.columns.disease_label,
        sample_name_col=config.data.columns.sample_name
    )

    # Load embedding mapping (for conditional encoder - study or organ)
    embedding_col = config.data.conditional_embedding.encoded_column
    embedding_mapping = {}
    if embedding_col in adata.obs.columns:
        mapping_df = adata.obs[[config.data.columns.sample_id, embedding_col]].drop_duplicates()
        embedding_mapping = dict(zip(
            mapping_df[config.data.columns.sample_id].astype(int),
            mapping_df[embedding_col].astype(int)
        ))
        print(f"Using conditional embedding: {config.data.conditional_embedding.column} -> {embedding_col}")

    # Create splitter
    splitter = LOOCVSplitter(random_seed=config.splitting.random_seed)
    print_split_summary(splitter, sample_ids, labels, sample_names)

    # Initialize metrics logger
    metrics_logger = MetricsLogger(str(output_dir / "results.csv"))

    # Run LOOCV
    all_results = []
    n_folds = splitter.get_n_splits(sample_ids)

    print(f"\n{'='*60}")
    print(f"Starting LOOCV Training ({n_folds} folds)")
    print(f"{'='*60}\n")

    for fold_info in splitter.split(sample_ids, labels, sample_names):
        fold_idx = fold_info.fold_idx
        test_sample_name = fold_info.test_sample_name or f"Sample_{fold_info.test_samples[0]}"

        print(f"\n[Fold {fold_idx + 1}/{n_folds}] Test sample: {test_sample_name}")
        print("-" * 40)

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

        # Calculate disease ratio if enabled (only once per fold)
        disease_ratio = None
        ratio_reg_lambda = 0.0

        if config.mil.loss.disease_ratio_reg.enabled:
            print("  Calculating disease ratio from training data...")
            disease_ratio = calculate_disease_ratio_from_dataloader(
                train_instance_dl,
                model_encoder,
                device,
                alpha=config.mil.loss.disease_ratio_reg.alpha,
                beta=config.mil.loss.disease_ratio_reg.beta,
                use_conditional=True
            )
            ratio_reg_lambda = config.mil.loss.disease_ratio_reg.lambda_weight

            if disease_ratio is not None and fold_idx == 0:
                print_disease_ratio_summary(disease_ratio, top_k=5)

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
            ratio_reg_lambda=ratio_reg_lambda
        )

        # Train fold (skip_fold_metrics=True for LOOCV)
        result = trainer.train_fold(
            train_bag_dl=train_bag_dl,
            train_instance_dl=train_instance_dl,
            test_bag_dl=test_bag_dl,
            n_epochs=config.mil.training.epochs,
            learning_rate=config.mil.training.learning_rate,
            encoder_learning_rate=config.mil.training.encoder_learning_rate,
            use_early_stopping=config.mil.training.use_early_stopping,
            patience=config.mil.training.patience,
            fold_idx=fold_idx,
            test_sample_name=test_sample_name,
            skip_fold_metrics=True,  # LOOCV: skip per-fold metrics
        )

        all_results.append(result)

        # Print fold result (for LOOCV: show prediction instead of metrics)
        pred_label = "Disease" if result.y_pred_proba[0] >= 0.5 else "Control"
        true_label = "Disease" if result.y_true[0] == 1 else "Control"
        correct = "✓" if pred_label == true_label else "✗"
        print(f"  prob={result.y_pred_proba[0]:.4f} (pred={pred_label}, true={true_label}) {correct}")

        # Save fold model
        if config.logging.save_checkpoints:
            trainer.save_models(str(output_dir / "models"), fold_idx)

    # Save final results
    metrics_logger.save()

    # Calculate overall AUROC by concatenating all fold predictions
    all_y_true = np.concatenate([r.y_true for r in all_results])
    all_y_pred_proba = np.concatenate([r.y_pred_proba for r in all_results])

    # Import metrics functions
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    # Calculate overall metrics
    overall_auc = roc_auc_score(all_y_true, all_y_pred_proba)

    # Find optimal threshold on concatenated predictions
    from src.training.metrics import find_optimal_threshold
    optimal_threshold, _ = find_optimal_threshold(all_y_true, all_y_pred_proba)
    all_y_pred = (all_y_pred_proba >= optimal_threshold).astype(int)

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)

    # Print summary
    print(f"\n{'='*60}")
    print("LOOCV Results Summary")
    print(f"{'='*60}")

    # Overall metrics (concatenated predictions - proper LOOCV evaluation)
    print(f"\n[Overall Metrics - Concatenated Predictions]")
    print(f"AUC:      {overall_auc:.4f}")
    print(f"Accuracy: {overall_acc:.4f}")
    print(f"F1 Score: {overall_f1:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")

    # Per-fold metrics (for reference)
    accs = [r.metrics['accuracy'] for r in all_results]
    print(f"\n[Per-Fold Accuracy]")
    print(f"Mean:     {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # Save overall metrics to a separate file
    overall_results = {
        'overall_auc': overall_auc,
        'overall_accuracy': overall_acc,
        'overall_f1': overall_f1,
        'optimal_threshold': optimal_threshold,
        'n_samples': len(all_y_true),
        'n_positive': int(all_y_true.sum()),
        'n_negative': int(len(all_y_true) - all_y_true.sum()),
    }
    pd.DataFrame([overall_results]).to_csv(output_dir / "overall_results.csv", index=False)

    # Save concatenated predictions for further analysis
    pred_df = pd.DataFrame({
        'sample_name': [r.test_sample for r in all_results],
        'y_true': all_y_true,
        'y_pred_proba': all_y_pred_proba,
        'y_pred': all_y_pred,
    })
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
