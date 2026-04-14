#!/usr/bin/env python
"""
01_pretrain_encoder.py - VQ-AENB-Conditional Encoder 사전학습 스크립트

전체 데이터셋을 사용하여 conditional encoder를 학습합니다.
이 encoder는 이후 MIL 학습에서 frozen 상태로 사용됩니다.

Usage:
    python scripts/01_pretrain_encoder.py --config config/default.yaml
    python scripts/01_pretrain_encoder.py --config config/default.yaml --register  # 자동 등록
    python scripts/01_pretrain_encoder.py --adata_path /path/to/data.h5ad --output_dir /path/to/output
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime

import torch
torch.set_num_threads(16)
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data import load_adata, preprocess_adata, encode_labels, encode_celltype_labels, print_adata_summary
from src.models.autoencoder import VQ_AENB_Conditional
from src.models.celltype_classifier import CelltypeClassifier
from src.training import AETrainer


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloader(adata, device, conditional_col: str, batch_size=256, shuffle=True,
                      celltype_ids=None):
    """Create dataloader from AnnData.

    Args:
        adata: AnnData object
        device: torch device
        conditional_col: Column name for conditional IDs (e.g., 'study_id_numeric', 'Organ_id_numeric')
        batch_size: Batch size
        shuffle: Whether to shuffle
        celltype_ids: Optional numpy array of celltype labels (-1 for missing)
    """
    # Extract data
    if hasattr(adata.X, 'toarray'):
        data = torch.tensor(adata.X.toarray(), dtype=torch.float32)
    else:
        data = torch.tensor(np.array(adata.X), dtype=torch.float32)

    # Extract conditional IDs
    conditional_ids = torch.tensor(
        adata.obs[conditional_col].values, dtype=torch.long
    )

    # Create dataset (with optional celltype labels)
    if celltype_ids is not None:
        ct_tensor = torch.tensor(celltype_ids, dtype=torch.long)
        dataset = TensorDataset(data, conditional_ids, ct_tensor)
    else:
        dataset = TensorDataset(data, conditional_ids)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def main():
    parser = argparse.ArgumentParser(description="Pretrain VQ-AENB-Conditional encoder")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--adata_path", type=str, default=None, help="Path to AnnData file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model hyperparameters (None means use config value)
    parser.add_argument("--latent_dim", type=int, default=None, help="Latent dimension")
    parser.add_argument("--num_codes", type=int, default=None, help="Number of codebook entries")
    parser.add_argument("--conditional_emb_dim", type=int, default=None,
        help="Conditional embedding dimension (replaces --study_emb_dim)")
    # Backward compatibility alias
    parser.add_argument("--study_emb_dim", type=int, default=None,
        help="(Deprecated) Use --conditional_emb_dim instead")
    parser.add_argument("--hidden_layers", type=int, nargs='+', default=None, help="Hidden layer dimensions (e.g., --hidden_layers 512 128)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")

    # Celltype auxiliary task overrides
    parser.add_argument("--celltype_aux", action="store_true",
        help="Enable celltype auxiliary loss (overrides config)")
    parser.add_argument("--celltype_column", type=str, default=None,
        help="Column in adata.obs for celltype labels")
    parser.add_argument("--celltype_loss_weight", type=float, default=None,
        help="Weight for celltype auxiliary loss")

    # Auto-registration options
    parser.add_argument("--register", action="store_true",
        help="자동으로 pretrained 디렉토리에 등록 (기존 파일 백업)")
    parser.add_argument("--pretrained_dir", type=str, default=None,
        help="등록할 pretrained 디렉토리 경로 (기본: config에서 추론)")

    args = parser.parse_args()

    # Handle deprecated --study_emb_dim argument
    conditional_emb_dim_arg = args.conditional_emb_dim or args.study_emb_dim

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        adata_path = args.adata_path or config.data.whole_adata_path
        output_dir = args.output_dir or config.paths.output_root
        # Model hyperparameters: CLI args override config (None means use config)
        latent_dim = args.latent_dim if args.latent_dim is not None else config.encoder.latent_dim
        num_codes = args.num_codes if args.num_codes is not None else config.encoder.num_codes
        conditional_emb_dim = conditional_emb_dim_arg if conditional_emb_dim_arg is not None else config.encoder.conditional_emb_dim
        hidden_layers = args.hidden_layers if args.hidden_layers is not None else config.encoder.hidden_layers
        batch_size = args.batch_size if args.batch_size is not None else config.encoder.pretrain.batch_size
        epochs = args.epochs if args.epochs is not None else config.encoder.pretrain.epochs
        lr = args.lr if args.lr is not None else config.encoder.pretrain.learning_rate
        patience = args.patience if args.patience is not None else config.encoder.pretrain.patience
        # Conditional embedding settings from config
        conditional_col = config.data.conditional_embedding.column  # e.g., 'study' or 'Organ'
        conditional_encoded_col = config.data.conditional_embedding.encoded_column  # e.g., 'study_id_numeric' or 'Organ_id_numeric'
    else:
        adata_path = args.adata_path
        output_dir = args.output_dir or "./results/pretrain"
        # Default values when no config provided
        latent_dim = args.latent_dim if args.latent_dim is not None else 128
        num_codes = args.num_codes if args.num_codes is not None else 1024
        conditional_emb_dim = conditional_emb_dim_arg if conditional_emb_dim_arg is not None else 16
        hidden_layers = args.hidden_layers if args.hidden_layers is not None else [512, 256, 128]
        batch_size = args.batch_size if args.batch_size is not None else 256
        epochs = args.epochs if args.epochs is not None else 50
        lr = args.lr if args.lr is not None else 0.001
        patience = args.patience if args.patience is not None else 5
        # Default to 'study' for backward compatibility
        conditional_col = "study"
        conditional_encoded_col = "study_id_numeric"

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
    if conditional_encoded_col not in adata.obs.columns:
        print(f"Encoding labels (conditional column: {conditional_col})...")
        adata, encoding_info = encode_labels(
            adata,
            sample_col='sample',
            label_col='Status',
            conditional_col=conditional_col,
            conditional_encoded_col=conditional_encoded_col
        )

    # Get number of conditional categories (e.g., studies or organs)
    n_conditionals = adata.obs[conditional_encoded_col].nunique()
    print(f"Number of {conditional_col}s: {n_conditionals}")

    # Build conditional mapping (id -> name) for MIL training with subset data
    if encoding_info and 'conditional' in encoding_info:
        # encoding_info['conditional']['mapping'] is {name: id}
        # We need {id: name}
        name_to_id = encoding_info['conditional']['mapping']
        id_to_name = {v: k for k, v in name_to_id.items()}
    else:
        # Build from adata if encoding was already done
        id_to_name = {}
        cond_df = adata.obs[[conditional_col, conditional_encoded_col]].drop_duplicates()
        for _, row in cond_df.iterrows():
            id_to_name[int(row[conditional_encoded_col])] = row[conditional_col]

    print(f"{conditional_col} ID mapping: {id_to_name}")

    # Get input dimension
    input_dim = adata.n_vars
    print(f"Input dimension: {input_dim}")

    # === Celltype auxiliary task setup ===
    celltype_classifier = None
    celltype_ids = None
    ct_loss_weight = 0.0

    # Determine celltype aux settings (CLI overrides config)
    ct_aux_enabled = args.celltype_aux
    ct_aux_column = args.celltype_column
    ct_aux_loss_weight = args.celltype_loss_weight

    if args.config and hasattr(config.encoder.pretrain, 'celltype_aux'):
        ct_aux_cfg = config.encoder.pretrain.celltype_aux
        if not ct_aux_enabled:
            ct_aux_enabled = ct_aux_cfg.enabled
        if ct_aux_column is None:
            ct_aux_column = ct_aux_cfg.column
        if ct_aux_loss_weight is None:
            ct_aux_loss_weight = ct_aux_cfg.loss_weight
        ct_aux_hidden_dim = ct_aux_cfg.hidden_dim
        ct_aux_n_layers = ct_aux_cfg.n_layers
        ct_aux_missing_label = ct_aux_cfg.missing_label
        ct_aux_dropout = ct_aux_cfg.dropout
    else:
        # Defaults when no config
        if ct_aux_column is None:
            ct_aux_column = "celltype_lineage"
        if ct_aux_loss_weight is None:
            ct_aux_loss_weight = 0.1
        ct_aux_hidden_dim = 64
        ct_aux_n_layers = 1
        ct_aux_missing_label = "MISSING"
        ct_aux_dropout = 0.1

    if ct_aux_enabled:
        print(f"\n{'='*60}")
        print("Celltype auxiliary task")
        print(f"{'='*60}")
        print(f"  Column: {ct_aux_column}")
        celltype_ids, ct_mapping, n_ct_classes = encode_celltype_labels(
            adata,
            celltype_col=ct_aux_column,
            missing_label=ct_aux_missing_label,
        )
        n_valid = (celltype_ids >= 0).sum()
        n_missing = (celltype_ids < 0).sum()
        print(f"  Valid celltype labels: {n_valid} ({n_valid/len(celltype_ids)*100:.1f}%)")
        print(f"  Missing celltype labels: {n_missing} ({n_missing/len(celltype_ids)*100:.1f}%)")
        print(f"  Number of celltype classes: {n_ct_classes}")

        if n_ct_classes > 0 and n_valid > 0:
            ct_loss_weight = ct_aux_loss_weight
            celltype_classifier = CelltypeClassifier(
                input_dim=latent_dim,
                n_classes=n_ct_classes,
                hidden_dim=ct_aux_hidden_dim,
                n_layers=ct_aux_n_layers,
                dropout=ct_aux_dropout,
            ).to(device)
            print(f"  Classifier: {ct_aux_n_layers} hidden layer(s), hidden_dim={ct_aux_hidden_dim}")
            print(f"  Loss weight: {ct_loss_weight}")
        else:
            print("  WARNING: No valid celltype labels found. Aux loss disabled.")

    # Create dataloader
    print("\nCreating dataloader...")
    train_loader = create_dataloader(
        adata, device,
        conditional_col=conditional_encoded_col,
        batch_size=batch_size,
        shuffle=True,
        celltype_ids=celltype_ids,
    )

    # Create model
    print("\nCreating VQ-AENB-Conditional model...")
    qcfg = getattr(config.encoder, "quantizer", None)
    model = VQ_AENB_Conditional(
        input_dim=input_dim,
        latent_dim=latent_dim,
        device=device,
        hidden_layers=hidden_layers,
        n_conditionals=n_conditionals,
        conditional_emb_dim=conditional_emb_dim,
        num_codes=num_codes,
        commitment_weight=qcfg.commitment_weight if qcfg else 0.25,
        ema_update=qcfg.ema_update if qcfg else False,
        ema_decay=qcfg.ema_decay if qcfg else 0.99,
        ema_eps=qcfg.ema_eps if qcfg else 1e-5,
    )
    model.to(device)

    print(f"  Latent dim: {latent_dim}")
    print(f"  Num codes: {num_codes}")
    print(f"  Conditional ({conditional_col}) embedding dim: {conditional_emb_dim}")
    print(f"  Number of {conditional_col}s: {n_conditionals}")
    print(f"  Hidden layers: {hidden_layers}")

    # Create trainer
    trainer = AETrainer(
        model, device, is_conditional=True,
        celltype_classifier=celltype_classifier,
        celltype_loss_weight=ct_loss_weight,
    )

    # Train
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")

    history = trainer.train(
        train_loader=train_loader,
        n_epochs=epochs,
        learning_rate=lr,
        patience=patience,
        init_codebook=True,
        init_method="kmeans"
    )

    # Save model
    model_path = output_path / "vq_aenb_conditional.pth"
    config_to_save = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_layers': hidden_layers,
        'n_conditionals': n_conditionals,
        'n_studies': n_conditionals,  # Backward compatibility
        'conditional_column': conditional_col,
        'conditional_encoded_column': conditional_encoded_col,
        'conditional_emb_dim': conditional_emb_dim,
        'study_emb_dim': conditional_emb_dim,  # Backward compatibility
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

    # Save conditional mapping (id -> name) for MIL training with subset data
    # File name based on conditional column (e.g., 'study_mapping.json' or 'organ_mapping.json')
    mapping_filename = f"{conditional_col.lower()}_mapping.json"
    mapping_path = output_path / mapping_filename
    with open(mapping_path, 'w') as f:
        json.dump({str(k): v for k, v in id_to_name.items()}, f, indent=2)

    # Save celltype classifier and mapping (separate from encoder)
    if celltype_classifier is not None:
        ct_classifier_path = output_path / "celltype_classifier.pth"
        torch.save({
            'model_state_dict': celltype_classifier.state_dict(),
            'n_classes': n_ct_classes,
            'celltype_mapping': ct_mapping,
            'config': {
                'input_dim': latent_dim,
                'hidden_dim': ct_aux_hidden_dim,
                'n_layers': ct_aux_n_layers,
                'dropout': ct_aux_dropout,
                'loss_weight': ct_loss_weight,
                'column': ct_aux_column,
            }
        }, str(ct_classifier_path))
        print(f"Celltype classifier saved to: {ct_classifier_path}")

        ct_mapping_path = output_path / "celltype_mapping.json"
        with open(ct_mapping_path, 'w') as f:
            json.dump(ct_mapping, f, indent=2)
        print(f"Celltype mapping saved to: {ct_mapping_path}")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    print(f"Model saved to: {model_path}")
    print(f"History saved to: {history_path}")
    print(f"{conditional_col} mapping saved to: {mapping_path}")
    if celltype_classifier is not None:
        print(f"Celltype classifier saved to: {ct_classifier_path}")
        print(f"Celltype mapping saved to: {ct_mapping_path}")

    # Auto-registration to pretrained directory
    if args.register:
        print(f"\n{'='*60}")
        print("Registering to pretrained directory...")
        print(f"{'='*60}")

        # Determine pretrained directory
        if args.pretrained_dir:
            pretrained_dir = Path(args.pretrained_dir)
        elif args.config:
            # Infer from config's pretrained_encoder path
            pretrained_dir = Path(config.paths.pretrained_encoder).parent
        else:
            pretrained_dir = Path(output_dir) / "pretrained"

        pretrained_dir.mkdir(parents=True, exist_ok=True)

        # Target file paths
        target_model = pretrained_dir / "vq_aenb_conditional.pth"
        target_mapping = pretrained_dir / mapping_filename

        # Backup existing files if they exist
        if target_model.exists():
            backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_model = pretrained_dir / f"vq_aenb_conditional_backup_{backup_timestamp}.pth"
            shutil.move(str(target_model), str(backup_model))
            print(f"  Backed up existing model to: {backup_model.name}")

        if target_mapping.exists():
            backup_mapping = pretrained_dir / f"{mapping_filename.replace('.json', '')}_backup_{backup_timestamp}.json"
            shutil.move(str(target_mapping), str(backup_mapping))
            print(f"  Backed up existing mapping to: {backup_mapping.name}")

        # Copy new files
        shutil.copy(str(model_path), str(target_model))
        shutil.copy(str(mapping_path), str(target_mapping))

        print(f"\n  Registered model to: {target_model}")
        print(f"  Registered mapping to: {target_mapping}")
        print(f"\nPretrained directory: {pretrained_dir}")


if __name__ == "__main__":
    main()
