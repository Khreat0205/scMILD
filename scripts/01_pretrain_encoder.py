#!/usr/bin/env python
"""
01_pretrain_encoder.py - VQ-AENB-Conditional Encoder 사전학습 스크립트

전체 데이터셋을 사용하여 conditional encoder를 학습합니다.
이 encoder는 이후 MIL 학습에서 frozen 상태로 사용됩니다.

Usage:
    python scripts/01_pretrain_encoder.py --config config/default.yaml
    python scripts/01_pretrain_encoder.py --adata_path /path/to/data.h5ad --output_dir /path/to/output
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data import load_adata, preprocess_adata, encode_labels, print_adata_summary
from src.models.autoencoder import VQ_AENB_Conditional
from src.training import AETrainer


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloader(adata, device, batch_size=256, shuffle=True):
    """Create dataloader from AnnData."""
    # Extract data
    if hasattr(adata.X, 'toarray'):
        data = torch.tensor(adata.X.toarray(), dtype=torch.float32)
    else:
        data = torch.tensor(np.array(adata.X), dtype=torch.float32)

    # Extract study IDs
    study_ids = torch.tensor(
        adata.obs['study_id_numeric'].values, dtype=torch.long
    )

    # Create dataset
    dataset = TensorDataset(data, study_ids)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def main():
    parser = argparse.ArgumentParser(description="Pretrain VQ-AENB-Conditional encoder")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--adata_path", type=str, default=None, help="Path to AnnData file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model hyperparameters
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--num_codes", type=int, default=1024, help="Number of codebook entries")
    parser.add_argument("--study_emb_dim", type=int, default=16, help="Study embedding dimension")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        adata_path = args.adata_path or config.data.whole_adata_path
        output_dir = args.output_dir or config.paths.output_root
        latent_dim = args.latent_dim or config.encoder.latent_dim
        num_codes = args.num_codes or config.encoder.num_codes
    else:
        adata_path = args.adata_path
        output_dir = args.output_dir or "./results/pretrain"
        latent_dim = args.latent_dim
        num_codes = args.num_codes

    if not adata_path:
        print("Error: Please provide --adata_path or --config")
        sys.exit(1)

    # Set seed
    set_seed(args.seed)

    # Setup device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"pretrain_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    # Load data
    print(f"\nLoading data from {adata_path}")
    adata = load_adata(adata_path)
    print_adata_summary(adata, "Original Data")

    # Encode labels if not already done
    encoding_info = None
    if 'study_id_numeric' not in adata.obs.columns:
        print("Encoding labels...")
        adata, encoding_info = encode_labels(
            adata,
            sample_col='sample',
            label_col='Status',
            study_col='study'
        )

    # Get number of studies
    n_studies = adata.obs['study_id_numeric'].nunique()
    print(f"Number of studies: {n_studies}")

    # Build study mapping (study_id -> study_name) for MIL training with subset data
    if encoding_info and 'study' in encoding_info:
        # encoding_info['study']['mapping'] is {study_name: study_id}
        # We need {study_id: study_name}
        study_name_to_id = encoding_info['study']['mapping']
        study_id_to_name = {v: k for k, v in study_name_to_id.items()}
    else:
        # Build from adata if encoding was already done
        study_id_to_name = {}
        study_df = adata.obs[['study', 'study_id_numeric']].drop_duplicates()
        for _, row in study_df.iterrows():
            study_id_to_name[int(row['study_id_numeric'])] = row['study']

    print(f"Study ID mapping: {study_id_to_name}")

    # Get input dimension
    input_dim = adata.n_vars
    print(f"Input dimension: {input_dim}")

    # Hidden layers
    hidden_layers = [512, 256, 128]

    # Create dataloader
    print("\nCreating dataloader...")
    train_loader = create_dataloader(
        adata, device,
        batch_size=args.batch_size,
        shuffle=True
    )

    # Create model
    print("\nCreating VQ-AENB-Conditional model...")
    model = VQ_AENB_Conditional(
        input_dim=input_dim,
        latent_dim=latent_dim,
        device=device,
        hidden_layers=hidden_layers,
        n_studies=n_studies,
        study_emb_dim=args.study_emb_dim,
        num_codes=num_codes,
        commitment_weight=0.25
    )
    model.to(device)

    print(f"  Latent dim: {latent_dim}")
    print(f"  Num codes: {num_codes}")
    print(f"  Study embedding dim: {args.study_emb_dim}")
    print(f"  Hidden layers: {hidden_layers}")

    # Create trainer
    trainer = AETrainer(model, device, is_conditional=True)

    # Train
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")

    history = trainer.train(
        train_loader=train_loader,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        init_codebook=True,
        init_method="kmeans"
    )

    # Save model
    model_path = output_path / "vq_aenb_conditional.pth"
    config_to_save = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_layers': hidden_layers,
        'n_studies': n_studies,
        'study_emb_dim': args.study_emb_dim,
        'num_codes': num_codes,
    }
    trainer.save(str(model_path), config_to_save)

    # Print codebook usage
    usage = trainer.get_codebook_usage()
    print(f"\nCodebook usage:")
    print(f"  Active codes: {usage.get('num_active', 'N/A')} / {usage.get('total_codes', 'N/A')}")
    print(f"  Utilization: {usage.get('utilization', 0) * 100:.1f}%")

    # Save training history
    import json
    history_path = output_path / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)

    # Save study mapping (study_id -> study_name) for MIL training with subset data
    study_mapping_path = output_path / "study_mapping.json"
    with open(study_mapping_path, 'w') as f:
        json.dump({str(k): v for k, v in study_id_to_name.items()}, f, indent=2)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    print(f"Model saved to: {model_path}")
    print(f"History saved to: {history_path}")
    print(f"Study mapping saved to: {study_mapping_path}")


if __name__ == "__main__":
    main()
