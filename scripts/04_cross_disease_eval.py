#!/usr/bin/env python
"""
04_cross_disease_eval.py - Cross-Disease 평가 스크립트

학습된 모델을 다른 질병 데이터셋에 적용하여 일반화 성능을 평가합니다.

예: Skin3 (HS)로 학습 → SCP1884 (CD)에 적용 (또는 그 반대)

Usage:
    python scripts/04_cross_disease_eval.py \\
        --model_dir results/final_model_xxx \\
        --test_config config/scp1884.yaml \\
        --gpu 0
"""

import os
import sys
import argparse
import json
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
from src.data import load_adata_with_subset, print_adata_summary, load_study_mapping, MilDataset, collate_mil
from src.models import (
    GatedAttentionModule, TeacherBranch, StudentBranch,
    VQEncoderWrapperConditional
)
from src.training import compute_metrics, find_optimal_threshold


def load_trained_models(model_dir: str, device: torch.device, config: ScMILDConfig):
    """Load trained models from directory."""
    model_dir = Path(model_dir)

    # Load pretrained encoder (from config path)
    encoder_path = config.paths.pretrained_encoder
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
    model_encoder.to(device)

    # Load encoder wrapper state (projection layer)
    encoder_state = torch.load(model_dir / "model_encoder_fold0.pt", map_location=device)
    model_encoder.load_state_dict(encoder_state)

    # Create and load teacher
    input_dim = model_encoder.input_dims
    attention_module = GatedAttentionModule(
        L=input_dim,
        D=config.mil.attention_dim,
        K=1
    ).to(device)

    model_teacher = TeacherBranch(
        input_dims=input_dim,
        latent_dims=config.mil.latent_dim,
        attention_module=attention_module,
        num_classes=config.mil.num_classes
    ).to(device)

    teacher_state = torch.load(model_dir / "model_teacher_fold0.pt", map_location=device)
    model_teacher.load_state_dict(teacher_state)

    # Create and load student
    model_student = StudentBranch(
        input_dims=input_dim,
        latent_dims=config.mil.latent_dim,
        num_classes=config.mil.num_classes
    ).to(device)

    student_state = torch.load(model_dir / "model_student_fold0.pt", map_location=device)
    model_student.load_state_dict(student_state)

    return model_teacher, model_student, model_encoder


def create_test_dataloader(
    adata,
    device: torch.device,
    config: ScMILDConfig,
) -> DataLoader:
    """Create test dataloader."""

    sample_col = config.data.columns.sample_id
    label_col = config.data.columns.disease_label
    embedding_col = config.data.conditional_embedding.encoded_column
    embedding_source_col = config.data.conditional_embedding.column
    embedding_mapping_path = config.data.conditional_embedding.mapping_path

    # If encoded column doesn't exist, create it from source column using pretrain mapping
    if embedding_col not in adata.obs.columns and embedding_source_col in adata.obs.columns:
        if embedding_mapping_path and Path(embedding_mapping_path).exists():
            id_to_name = load_study_mapping(embedding_mapping_path)
            name_to_id = {v: k for k, v in id_to_name.items()}
            print(f"Using pretrain study mapping from: {embedding_mapping_path}")
            print(f"  Mapping: {name_to_id}")
            adata.obs[embedding_col] = adata.obs[embedding_source_col].map(name_to_id)
        else:
            print(f"WARNING: Creating '{embedding_col}' from '{embedding_source_col}' without pretrain mapping!")
            adata.obs[embedding_col] = adata.obs[embedding_source_col].astype('category').cat.codes

    # Extract data
    if hasattr(adata.X, 'toarray'):
        data = torch.tensor(adata.X.toarray(), dtype=torch.float32, device=device)
    else:
        data = torch.tensor(np.array(adata.X), dtype=torch.float32, device=device)

    # Sample IDs
    sample_ids = torch.tensor(
        adata.obs[sample_col].values, dtype=torch.long, device=device
    )

    # Labels
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

    # Embedding IDs (study or organ) - use direct column value
    embedding_ids = None
    if embedding_col in adata.obs.columns:
        embedding_ids = torch.tensor(
            adata.obs[embedding_col].values.astype(int),
            dtype=torch.long, device=device
        )

    # Create dataset
    test_dataset = MilDataset(data, sample_ids, labels, instance_labels, embedding_ids)

    return DataLoader(
        test_dataset,
        batch_size=len(unique_samples),
        shuffle=False,
        collate_fn=collate_mil
    )


@torch.no_grad()
def evaluate(
    model_teacher,
    model_encoder,
    dataloader: DataLoader,
    device: torch.device,
    use_conditional: bool = True
):
    """Evaluate model on dataloader."""
    model_teacher.eval()
    model_encoder.eval()

    all_labels = []
    all_probs = []
    all_sample_names = []

    for batch in dataloader:
        t_data = batch[0].to(device)
        t_bagids = batch[1].to(device)
        t_labels = batch[2].to(device)
        t_study_ids = batch[3].to(device) if len(batch) == 4 else None

        # Encode
        if use_conditional and t_study_ids is not None:
            feat = model_encoder(t_data, t_study_ids)
        else:
            feat = model_encoder(t_data)

        # Get predictions for each bag
        inner_ids = t_bagids[-1]
        unique_bags = torch.unique(inner_ids)

        for i, bag in enumerate(unique_bags):
            bag_mask = inner_ids == bag
            bag_instances = feat[bag_mask]

            bag_pred = model_teacher(bag_instances)
            bag_prob = torch.softmax(bag_pred, dim=0)[1].item()

            all_probs.append(bag_prob)

        all_labels.extend(t_labels.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser(description="Cross-disease evaluation")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--test_config", type=str, required=True, help="Config for test dataset")
    parser.add_argument("--train_config", type=str, default=None, help="Config used for training (optional)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Setup device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model info
    model_dir = Path(args.model_dir)
    if (model_dir / "model_info.json").exists():
        with open(model_dir / "model_info.json") as f:
            model_info = json.load(f)
        print(f"Model trained on: {model_info.get('n_samples', 'unknown')} samples")

    # Load test config
    test_config = load_config(args.test_config)

    # Use train config if provided, otherwise use test config structure
    if args.train_config:
        train_config = load_config(args.train_config)
    else:
        train_config = test_config

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = model_dir / f"cross_eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data (with optional subset)
    print(f"\nLoading test data...")
    test_adata = load_adata_with_subset(
        whole_adata_path=test_config.data.whole_adata_path,
        subset_enabled=test_config.data.subset.enabled,
        subset_column=test_config.data.subset.column,
        subset_values=test_config.data.subset.values,
        cache_dir=test_config.data.subset.cache_dir,
        use_cache=test_config.data.subset.use_cache,
    )
    print_adata_summary(test_adata, "Test Data")

    n_test_samples = test_adata.obs[test_config.data.columns.sample_id].nunique()
    print(f"Number of test samples: {n_test_samples}")

    # Create test dataloader (study_id mapping is handled inside)
    test_dl = create_test_dataloader(test_adata, device, test_config)

    # Load models
    print("\nLoading trained models...")
    model_teacher, model_student, model_encoder = load_trained_models(
        model_dir, device, train_config
    )

    # Evaluate
    print("\nEvaluating on test data...")
    y_true, y_pred_proba = evaluate(
        model_teacher, model_encoder, test_dl, device, use_conditional=True
    )

    # Compute metrics
    threshold, metrics = find_optimal_threshold(y_true, y_pred_proba)
    metrics['threshold'] = threshold

    # Print results
    print(f"\n{'='*60}")
    print("Cross-Disease Evaluation Results")
    print(f"{'='*60}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"{'='*60}")

    # Save results
    results = {
        'model_dir': str(model_dir),
        'test_config': args.test_config,
        'n_test_samples': n_test_samples,
        'n_test_cells': test_adata.n_obs,
        **metrics
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv(output_dir / "cross_eval_results.csv", index=False)

    # Save per-sample predictions
    sample_col = test_config.data.columns.sample_id
    sample_name_col = test_config.data.columns.sample_name

    sample_info = test_adata.obs.groupby(sample_col).agg({
        sample_name_col: 'first',
        test_config.data.columns.disease_label: 'first'
    }).reset_index()

    sample_results = pd.DataFrame({
        'sample_id': sample_info[sample_col].values,
        'sample_name': sample_info[sample_name_col].values,
        'true_label': y_true,
        'pred_prob': y_pred_proba,
        'pred_label': (y_pred_proba >= threshold).astype(int)
    })
    sample_results.to_csv(output_dir / "sample_predictions.csv", index=False)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
