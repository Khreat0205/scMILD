#!/usr/bin/env python
"""
05_cell_scoring.py - Cell-level 점수 계산 스크립트

학습된 모델을 사용하여 각 세포의 attention score와 prediction을 계산합니다.
하위 분석 (subpopulation 발견 등)에 사용됩니다.

Usage:
    python scripts/05_cell_scoring.py \\
        --model_dir results/final_model_xxx \\
        --config config/skin3.yaml \\
        --gpu 0
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
torch.set_num_threads(16)
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, ScMILDConfig
from src.data import load_adata_with_subset, print_adata_summary, load_conditional_mapping, MilDataset, collate_mil
from src.models import (
    GatedAttentionModule, TeacherBranch, StudentBranch,
    VQEncoderWrapperConditional
)


def load_trained_models(model_dir: str, device: torch.device, config: ScMILDConfig):
    """Load trained models from directory."""
    model_dir = Path(model_dir)

    # Load pretrained encoder
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

    # Load encoder wrapper state
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


def ensure_embedding_column(adata, config: ScMILDConfig):
    """Ensure embedding column exists in adata.obs."""
    embedding_col = config.data.conditional_embedding.encoded_column
    embedding_source_col = config.data.conditional_embedding.column
    embedding_mapping_path = config.data.conditional_embedding.mapping_path

    if embedding_col not in adata.obs.columns and embedding_source_col in adata.obs.columns:
        if embedding_mapping_path and Path(embedding_mapping_path).exists():
            id_to_name = load_conditional_mapping(embedding_mapping_path)
            name_to_id = {v: k for k, v in id_to_name.items()}
            print(f"Using pretrain {embedding_source_col} mapping from: {embedding_mapping_path}")
            print(f"  Mapping: {name_to_id}")
            adata.obs[embedding_col] = adata.obs[embedding_source_col].map(name_to_id)
        else:
            print(f"WARNING: Creating '{embedding_col}' from '{embedding_source_col}' without pretrain mapping!")
            adata.obs[embedding_col] = adata.obs[embedding_source_col].astype('category').cat.codes
    return adata


@torch.no_grad()
def compute_cell_scores(
    adata,
    model_teacher,
    model_student,
    model_encoder,
    device: torch.device,
    config: ScMILDConfig,
    batch_size: int = 10000
):
    """
    Compute cell-level scores for all cells.

    Returns:
        DataFrame with cell_id, sample_id, attention_score, student_pred, etc.
    """
    model_teacher.eval()
    model_student.eval()
    model_encoder.eval()

    sample_col = config.data.columns.sample_id
    label_col = config.data.columns.disease_label
    embedding_col = config.data.conditional_embedding.encoded_column

    all_attention_scores = []
    all_student_preds = []
    all_sample_ids = []
    all_embeddings = []

    n_cells = adata.n_obs
    n_batches = (n_cells + batch_size - 1) // batch_size

    print(f"Processing {n_cells} cells in {n_batches} batches...")

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_cells)

        # Get batch data
        batch_adata = adata[start_idx:end_idx]

        if hasattr(batch_adata.X, 'toarray'):
            data = torch.tensor(batch_adata.X.toarray(), dtype=torch.float32, device=device)
        else:
            data = torch.tensor(np.array(batch_adata.X), dtype=torch.float32, device=device)

        sample_ids = batch_adata.obs[sample_col].values

        # Get embedding IDs from column directly
        if embedding_col in batch_adata.obs.columns:
            embedding_ids = torch.tensor(
                batch_adata.obs[embedding_col].values.astype(int),
                dtype=torch.long, device=device
            )
        else:
            embedding_ids = torch.zeros(len(sample_ids), dtype=torch.long, device=device)

        # Encode
        feat = model_encoder(data, embedding_ids)

        # Get attention scores
        attn_scores = model_teacher.attention_module(feat).squeeze()

        # Get student predictions
        student_out = model_student(feat)
        student_probs = torch.softmax(student_out, dim=1)[:, 1]

        all_attention_scores.append(attn_scores.cpu().numpy())
        all_student_preds.append(student_probs.cpu().numpy())
        all_sample_ids.extend(sample_ids)
        all_embeddings.append(feat.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {end_idx}/{n_cells} cells ({100*end_idx/n_cells:.1f}%)")

    # Concatenate results
    attention_scores = np.concatenate(all_attention_scores)
    student_preds = np.concatenate(all_student_preds)
    embeddings = np.concatenate(all_embeddings)

    # Normalize attention scores per sample
    attention_scores_norm = np.zeros_like(attention_scores)
    unique_samples = np.unique(all_sample_ids)

    for sample in unique_samples:
        mask = np.array(all_sample_ids) == sample
        scores = attention_scores[mask]
        if scores.max() - scores.min() > 1e-8:
            attention_scores_norm[mask] = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            attention_scores_norm[mask] = 0.5

    # Create results DataFrame
    results_df = pd.DataFrame({
        'cell_id': adata.obs.index,
        'sample_id': all_sample_ids,
        'sample_name': adata.obs[config.data.columns.sample_name].values,
        'disease_label': adata.obs[label_col].values,
        'attention_score_raw': attention_scores,
        'attention_score_norm': attention_scores_norm,
        'student_prediction': student_preds,
    })

    return results_df, embeddings


def main():
    parser = argparse.ArgumentParser(description="Compute cell-level scores")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--config", type=str, required=True, help="Config for dataset")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size for processing")
    parser.add_argument("--save_embeddings", action="store_true", help="Save cell embeddings (large file!)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Setup device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    config = load_config(args.config)

    # Output directory
    model_dir = Path(args.model_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = model_dir / f"cell_scores_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Ensure embedding column exists (apply study mapping if needed)
    adata = ensure_embedding_column(adata, config)

    # Load models
    print("\nLoading trained models...")
    model_teacher, model_student, model_encoder = load_trained_models(
        model_dir, device, config
    )

    # Compute cell scores
    print("\nComputing cell-level scores...")
    results_df, embeddings = compute_cell_scores(
        adata,
        model_teacher,
        model_student,
        model_encoder,
        device,
        config,
        batch_size=args.batch_size
    )

    # Save results
    print("\nSaving results...")
    results_df.to_csv(output_dir / "cell_scores.csv", index=False)

    if args.save_embeddings:
        np.save(output_dir / "cell_embeddings.npy", embeddings)
        print(f"  Embeddings saved: {embeddings.shape}")

    # Print summary statistics
    print(f"\n{'='*60}")
    print("Cell Scoring Complete!")
    print(f"{'='*60}")

    # Summary by sample
    sample_summary = results_df.groupby(['sample_name', 'disease_label']).agg({
        'attention_score_norm': ['mean', 'std', 'max'],
        'student_prediction': ['mean', 'std'],
        'cell_id': 'count'
    }).round(4)
    sample_summary.columns = ['attn_mean', 'attn_std', 'attn_max', 'student_mean', 'student_std', 'n_cells']
    sample_summary = sample_summary.reset_index()

    print("\nSample-level summary:")
    print(sample_summary.to_string(index=False))

    sample_summary.to_csv(output_dir / "sample_summary.csv", index=False)

    # Top cells by attention score
    print("\nTop 10 cells by attention score (per disease class):")
    for label in results_df['disease_label'].unique():
        subset = results_df[results_df['disease_label'] == label]
        top_cells = subset.nlargest(10, 'attention_score_norm')[
            ['cell_id', 'sample_name', 'attention_score_norm', 'student_prediction']
        ]
        print(f"\n  Disease label = {label}:")
        print(top_cells.to_string(index=False))

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
