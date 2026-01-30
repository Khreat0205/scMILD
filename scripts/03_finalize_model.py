#!/usr/bin/env python
"""
03_finalize_model.py - Final Model 학습 스크립트

LOOCV에서 찾은 최적의 하이퍼파라미터로 전체 데이터를 사용하여
Final model을 학습합니다.

Usage:
    python scripts/03_finalize_model.py --config config/skin3.yaml
    python scripts/03_finalize_model.py --config config/scp1884.yaml --gpu 0

    # best_params.yaml 적용 (06_tune_hyperparams.py 결과물)
    python scripts/03_finalize_model.py --config config/skin3.yaml \\
        --best_params results/skin3/tuning_*/best_params.yaml --gpu 0
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, ScMILDConfig
from src.data import (
    load_adata,
    MilDataset, InstanceDataset, collate_mil, create_instance_dataset_with_bag_labels
)
from src.models import (
    GatedAttentionModule, TeacherBranch, StudentBranch,
    VQEncoderWrapperConditional
)
from src.training import MILTrainer


def load_best_params(best_params_path: str) -> dict:
    """Load best hyperparameters from YAML file.

    Args:
        best_params_path: Path to best_params.yaml from tuning script

    Returns:
        Dictionary of hyperparameters
    """
    with open(best_params_path, 'r') as f:
        data = yaml.safe_load(f)

    best_params = data.get('best_hyperparameters', {})
    best_score = data.get('best_score', {})

    print(f"Loaded best params from: {best_params_path}")
    print(f"  Best {best_score.get('metric', 'score')}: {best_score.get('value', 'N/A')}")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    return best_params


def apply_best_params(config: 'ScMILDConfig', best_params: dict) -> 'ScMILDConfig':
    """Apply best hyperparameters to config.

    Supported parameters:
        - learning_rate -> config.mil.training.learning_rate
        - encoder_learning_rate -> config.mil.training.encoder_learning_rate
        - attention_dim -> config.mil.attention_dim
        - latent_dim -> config.mil.latent_dim
        - projection_dim -> config.mil.projection_dim
        - negative_weight -> config.mil.loss.negative_weight
        - student_optimize_period -> config.mil.student.optimize_period
        - epochs -> config.mil.training.epochs
    """
    param_mapping = {
        'learning_rate': ('mil', 'training', 'learning_rate'),
        'encoder_learning_rate': ('mil', 'training', 'encoder_learning_rate'),
        'attention_dim': ('mil', 'attention_dim'),
        'latent_dim': ('mil', 'latent_dim'),
        'projection_dim': ('mil', 'projection_dim'),
        'negative_weight': ('mil', 'loss', 'negative_weight'),
        'student_optimize_period': ('mil', 'student', 'optimize_period'),
        'epochs': ('mil', 'training', 'epochs'),
    }

    for param_name, value in best_params.items():
        if param_name in param_mapping:
            path = param_mapping[param_name]
            obj = config
            for attr in path[:-1]:
                obj = getattr(obj, attr)
            setattr(obj, path[-1], value)
            print(f"  Applied: {'.'.join(path)} = {value}")
        else:
            print(f"  Warning: Unknown parameter '{param_name}', skipping")

    return config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def create_full_dataloaders(
    adata,
    device: torch.device,
    config: ScMILDConfig,
    embedding_mapping: dict
) -> tuple:
    """Create dataloaders using all data (no split)."""

    sample_col = config.data.columns.sample_id
    label_col = config.data.columns.disease_label

    # Extract data
    if hasattr(adata.X, 'toarray'):
        data = torch.tensor(adata.X.toarray(), dtype=torch.float32, device=device)
    else:
        data = torch.tensor(np.array(adata.X), dtype=torch.float32, device=device)

    # Sample IDs
    sample_ids = torch.tensor(
        adata.obs[sample_col].values, dtype=torch.long, device=device
    )

    # Labels (unique per sample)
    unique_samples = sorted(adata.obs[sample_col].unique())
    sample_labels = adata.obs.groupby(sample_col)[label_col].first()
    labels = torch.tensor(
        [sample_labels[s] for s in unique_samples],
        dtype=torch.long, device=device
    )

    # Instance labels
    instance_labels = torch.tensor(
        adata.obs[label_col].values, dtype=torch.long, device=device
    )

    # Embedding IDs
    embedding_ids = None
    if embedding_mapping:
        embedding_ids = torch.tensor(
            [embedding_mapping.get(int(s), 0) for s in adata.obs[sample_col].values],
            dtype=torch.long, device=device
        )

    # Create datasets
    mil_dataset = MilDataset(data, sample_ids, labels, instance_labels, embedding_ids)
    instance_dataset = InstanceDataset(data, sample_ids, labels, instance_labels, embedding_ids)
    instance_with_bag = create_instance_dataset_with_bag_labels(instance_dataset, device)

    # Create dataloaders
    bag_dl = DataLoader(
        mil_dataset,
        batch_size=config.mil.training.batch_size,
        shuffle=True,
        collate_fn=collate_mil
    )
    instance_dl = DataLoader(
        instance_with_bag,
        batch_size=256,
        shuffle=True
    )

    return bag_dl, instance_dl


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
    parser = argparse.ArgumentParser(description="Train final scMILD model on full data")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--best_params", type=str, default=None,
                        help="Path to best_params.yaml from tuning script")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--output_name", type=str, default=None, help="Custom output name")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply best params if provided
    if args.best_params:
        print(f"\n{'='*60}")
        print("Loading Best Hyperparameters")
        print(f"{'='*60}")
        best_params = load_best_params(args.best_params)
        config = apply_best_params(config, best_params)

    # Set seed
    set_seed(args.seed)

    # Setup device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = args.output_name or f"final_model_{timestamp}"
    output_dir = Path(config.paths.output_root) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load data
    print(f"\nLoading data from {config.data.adata_path}")
    adata = load_adata(config.data.adata_path)
    print(f"Data shape: {adata.n_obs} cells × {adata.n_vars} genes")

    n_samples = adata.obs[config.data.columns.sample_id].nunique()
    print(f"Number of samples: {n_samples}")

    # Load embedding mapping
    embedding_col = config.data.conditional_embedding.encoded_column
    embedding_mapping = {}
    if embedding_col in adata.obs.columns:
        mapping_df = adata.obs[[config.data.columns.sample_id, embedding_col]].drop_duplicates()
        embedding_mapping = dict(zip(
            mapping_df[config.data.columns.sample_id].astype(int),
            mapping_df[embedding_col].astype(int)
        ))
        print(f"Using conditional embedding: {config.data.conditional_embedding.column}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    bag_dl, instance_dl = create_full_dataloaders(
        adata, device, config, embedding_mapping
    )

    # Create models
    print("Creating models...")
    model_teacher, model_student, model_encoder = create_models(
        config, device, config.paths.pretrained_encoder
    )

    # Create trainer
    trainer = MILTrainer(
        model_teacher=model_teacher,
        model_student=model_student,
        model_encoder=model_encoder,
        device=device,
        use_conditional_ae=True,
        student_optimize_period=config.mil.student.optimize_period,
        student_loss_weight_neg=config.mil.loss.negative_weight
    )

    # Train
    n_epochs = args.epochs or config.mil.training.epochs
    print(f"\n{'='*60}")
    print(f"Training Final Model ({n_epochs} epochs)")
    print(f"{'='*60}\n")

    # Use the train_fold method but with same data for train/test
    # (we don't need test metrics, just training)
    result = trainer.train_fold(
        train_bag_dl=bag_dl,
        train_instance_dl=instance_dl,
        test_bag_dl=bag_dl,  # Same as train for final training
        n_epochs=n_epochs,
        learning_rate=config.mil.training.learning_rate,
        encoder_learning_rate=config.mil.training.encoder_learning_rate,
        use_early_stopping=False,  # No early stopping for final model
        fold_idx=0,
        test_sample_name="all_data"
    )

    # Save final model
    print("\nSaving final model...")
    trainer.save_models(str(output_dir), fold_idx=0)

    # Save model info
    import json
    model_info = {
        'config_path': args.config,
        'best_params_path': args.best_params,
        'n_samples': n_samples,
        'n_cells': adata.n_obs,
        'n_epochs': n_epochs,
        'hyperparameters': {
            'learning_rate': config.mil.training.learning_rate,
            'encoder_learning_rate': config.mil.training.encoder_learning_rate,
            'attention_dim': config.mil.attention_dim,
            'latent_dim': config.mil.latent_dim,
            'projection_dim': config.mil.projection_dim,
            'negative_weight': config.mil.loss.negative_weight,
            'student_optimize_period': config.mil.student.optimize_period,
        },
        'training_metrics': result.metrics,
        'embedding_column': config.data.conditional_embedding.column,
        'timestamp': timestamp
    }
    with open(output_dir / "model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"\n{'='*60}")
    print("Final Model Training Complete!")
    print(f"{'='*60}")
    print(f"Model saved to: {output_dir}")
    print(f"\nUse this model for cross-disease evaluation with:")
    print(f"  python scripts/04_cross_disease_eval.py --model_dir {output_dir} --test_data /path/to/other_dataset.h5ad")


if __name__ == "__main__":
    main()
