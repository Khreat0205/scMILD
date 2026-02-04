#!/usr/bin/env python
"""
05_cell_scoring.py - Cell-level 점수 계산 스크립트

학습된 모델을 사용하여 각 세포의 attention score와 prediction을 계산합니다.
하위 분석 (subpopulation 발견 등)에 사용됩니다.

지원 모드:
    1. Final model 모드 (--model_dir): 단일 모델로 전체 데이터 scoring
    2. CV 모드 (--cv_dir): 각 fold 모델로 해당 test set만 scoring
    3. Tuning 모드 (--tuning_dir): Best params의 fold 모델 사용

출력:
    - scored_adata.h5ad: Cell-level scoring 결과가 추가된 AnnData
    - codebook_adata.h5ad: Codebook-level 통계 AnnData
    - cell_scores.csv: CSV 형식 백업

Usage:
    # Final model 모드
    python scripts/05_cell_scoring.py \\
        --model_dir results/final_model_xxx \\
        --config config/skin3.yaml \\
        --gpu 0

    # CV 모드
    python scripts/05_cell_scoring.py \\
        --cv_dir results/cv_stratified_kfold_xxx \\
        --config config/scp1884.yaml \\
        --output_dir results/cell_scores_cv \\
        --gpu 0
"""

import os
import sys
import argparse
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import torch
torch.set_num_threads(16)
import numpy as np
import pandas as pd
import scanpy as sc

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, ScMILDConfig
from src.data import load_adata_with_subset, print_adata_summary, load_conditional_mapping
from src.models import (
    GatedAttentionModule, TeacherBranch, StudentBranch,
    VQEncoderWrapperConditional
)
from src.models.autoencoder import VQ_AENB_Conditional


# ============================================================================
# Model Loading Functions
# ============================================================================

def load_pretrained_encoder(config: ScMILDConfig, device: torch.device):
    """Load pretrained VQ-AENB-Conditional encoder."""
    encoder_path = config.paths.pretrained_encoder
    checkpoint = torch.load(encoder_path, map_location=device)
    model_config = checkpoint.get('config', {})

    encoder_model = VQ_AENB_Conditional(
        input_dim=model_config['input_dim'],
        latent_dim=model_config['latent_dim'],
        device=device,
        hidden_layers=model_config['hidden_layers'],
        n_conditionals=model_config.get('n_conditionals', model_config.get('n_studies')),
        conditional_emb_dim=model_config.get('conditional_emb_dim', model_config.get('study_emb_dim', 16)),
        num_codes=model_config.get('num_codes', 1024),
    )
    encoder_model.load_state_dict(checkpoint['model_state_dict'])
    encoder_model.to(device)

    return encoder_model, model_config


def load_trained_models(
    model_dir: Path,
    device: torch.device,
    config: ScMILDConfig,
    fold_idx: int = 0
) -> Tuple:
    """Load trained models from directory for a specific fold."""

    # Load pretrained encoder
    encoder_model, model_config = load_pretrained_encoder(config, device)

    # Wrap encoder
    model_encoder = VQEncoderWrapperConditional(
        encoder_model,
        use_projection=config.mil.use_projection,
        projection_dim=config.mil.projection_dim
    )
    model_encoder.to(device)

    # Load encoder wrapper state
    encoder_state = torch.load(model_dir / f"model_encoder_fold{fold_idx}.pt", map_location=device)
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

    teacher_state = torch.load(model_dir / f"model_teacher_fold{fold_idx}.pt", map_location=device)
    model_teacher.load_state_dict(teacher_state)

    # Create and load student
    model_student = StudentBranch(
        input_dims=input_dim,
        latent_dims=config.mil.latent_dim,
        num_classes=config.mil.num_classes
    ).to(device)

    student_state = torch.load(model_dir / f"model_student_fold{fold_idx}.pt", map_location=device)
    model_student.load_state_dict(student_state)

    return model_teacher, model_student, model_encoder


def get_fold_model_paths(model_dir: Path) -> List[int]:
    """Get list of available fold indices from model directory."""
    fold_indices = []
    for f in model_dir.glob("model_encoder_fold*.pt"):
        fold_idx = int(f.stem.split("fold")[1])
        fold_indices.append(fold_idx)
    return sorted(fold_indices)


def load_trained_models_tuning(
    fold_dir: Path,
    device: torch.device,
    config: ScMILDConfig
) -> Tuple:
    """Load trained models from tuning fold directory.

    Tuning saves models as: fold_dir/encoder.pth, teacher.pth, student.pth
    These are state_dict only (not full model).
    """

    # Load pretrained encoder
    encoder_model, model_config = load_pretrained_encoder(config, device)

    # Wrap encoder
    model_encoder = VQEncoderWrapperConditional(
        encoder_model,
        use_projection=config.mil.use_projection,
        projection_dim=config.mil.projection_dim
    )
    model_encoder.to(device)

    # Load encoder wrapper state
    encoder_state = torch.load(fold_dir / "encoder.pth", map_location=device)
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

    teacher_state = torch.load(fold_dir / "teacher.pth", map_location=device)
    model_teacher.load_state_dict(teacher_state)

    # Create and load student
    model_student = StudentBranch(
        input_dims=input_dim,
        latent_dims=config.mil.latent_dim,
        num_classes=config.mil.num_classes
    ).to(device)

    student_state = torch.load(fold_dir / "student.pth", map_location=device)
    model_student.load_state_dict(student_state)

    return model_teacher, model_student, model_encoder


# ============================================================================
# Data Preparation Functions
# ============================================================================

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


def get_sample_fold_mapping(cv_dir: Path) -> Dict[str, int]:
    """Get sample_name -> fold mapping from CV results."""
    predictions_path = cv_dir / "predictions.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions.csv not found in {cv_dir}")

    pred_df = pd.read_csv(predictions_path)

    # predictions.csv에 fold 컬럼이 없을 수 있음 - sample 순서로 추론
    if 'fold' in pred_df.columns:
        return dict(zip(pred_df['sample_name'], pred_df['fold']))

    # fold 컬럼이 없으면 sample_name 등장 순서로 fold 할당
    # (CV 결과에서 각 sample은 한 번만 test로 등장)
    sample_to_fold = {}
    for idx, row in pred_df.iterrows():
        sample_name = row['sample_name']
        if sample_name not in sample_to_fold:
            sample_to_fold[sample_name] = len(sample_to_fold)

    return sample_to_fold


# ============================================================================
# Scoring Functions
# ============================================================================

@torch.no_grad()
def compute_cell_scores_batch(
    data: torch.Tensor,
    embedding_ids: torch.Tensor,
    model_teacher,
    model_student,
    model_encoder,
    device: torch.device
) -> Dict[str, np.ndarray]:
    """Compute scores for a batch of cells."""

    model_teacher.eval()
    model_student.eval()
    model_encoder.eval()

    data = data.to(device)
    embedding_ids = embedding_ids.to(device)

    # X_pretrained: projection 이전 (quantized latent)
    X_pretrained = model_encoder.vq_model.features(data, embedding_ids)

    # X_scmild: projection 이후 (학습된 MIL embedding)
    X_scmild = model_encoder(data, embedding_ids)

    # vq_code: codebook index
    vq_code = model_encoder.vq_model.get_codebook_indices(data, embedding_ids)

    # Attention scores
    attn_scores = model_teacher.attention_module(X_scmild).squeeze()

    # Student predictions
    student_out = model_student(X_scmild)
    student_probs = torch.softmax(student_out, dim=1)[:, 1]

    return {
        'X_pretrained': X_pretrained.cpu().numpy(),
        'X_scmild': X_scmild.cpu().numpy(),
        'vq_code': vq_code.cpu().numpy(),
        'attention_score_raw': attn_scores.cpu().numpy(),
        'student_prediction': student_probs.cpu().numpy(),
    }


@torch.no_grad()
def compute_codebook_direct_attention(
    model_encoder,
    model_teacher,
    device: torch.device
) -> np.ndarray:
    """Compute attention scores by passing codebook directly through attention module."""

    model_encoder.eval()
    model_teacher.eval()

    # Get codebook embeddings
    codebook = model_encoder.vq_model.quantizer.get_codebook()  # (num_codes, latent_dim)
    codebook = codebook.to(device)

    # Apply projection layer (if exists)
    if model_encoder.projection is not None:
        codebook_projected = model_encoder.projection(codebook)
    else:
        codebook_projected = codebook

    # Compute attention scores
    attn_direct = model_teacher.attention_module(codebook_projected)  # (num_codes, 1)

    return attn_direct.squeeze().cpu().numpy()


def normalize_attention_global(attention_scores: np.ndarray) -> np.ndarray:
    """Normalize attention scores globally (min-max across all cells)."""
    score_min = attention_scores.min()
    score_max = attention_scores.max()
    if score_max - score_min > 1e-8:
        return (attention_scores - score_min) / (score_max - score_min)
    else:
        return np.full_like(attention_scores, 0.5)


def normalize_attention_per_sample(
    attention_scores: np.ndarray,
    sample_ids: np.ndarray
) -> np.ndarray:
    """Normalize attention scores per sample (min-max within each sample)."""
    attention_scores_norm = np.zeros_like(attention_scores)
    unique_samples = np.unique(sample_ids)

    for sample in unique_samples:
        mask = sample_ids == sample
        scores = attention_scores[mask]
        if scores.max() - scores.min() > 1e-8:
            attention_scores_norm[mask] = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            attention_scores_norm[mask] = 0.5

    return attention_scores_norm


def compute_cell_scores_for_adata(
    adata,
    model_teacher,
    model_student,
    model_encoder,
    device: torch.device,
    config: ScMILDConfig,
    batch_size: int = 10000,
    fold_idx: Optional[int] = None
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cell-level scores for all cells in adata.

    Returns:
        results_df: DataFrame with scoring results
        X_pretrained: Pretrained embeddings
        X_scmild: MIL embeddings
        vq_codes: Codebook indices
    """
    sample_col = config.data.columns.sample_id
    sample_name_col = config.data.columns.sample_name
    label_col = config.data.columns.disease_label
    embedding_col = config.data.conditional_embedding.encoded_column

    n_cells = adata.n_obs
    n_batches = (n_cells + batch_size - 1) // batch_size

    all_results = {
        'X_pretrained': [],
        'X_scmild': [],
        'vq_code': [],
        'attention_score_raw': [],
        'student_prediction': [],
    }

    print(f"Processing {n_cells} cells in {n_batches} batches...")

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_cells)

        # Get batch data
        batch_adata = adata[start_idx:end_idx]

        if hasattr(batch_adata.X, 'toarray'):
            data = torch.tensor(batch_adata.X.toarray(), dtype=torch.float32)
        else:
            data = torch.tensor(np.array(batch_adata.X), dtype=torch.float32)

        # Get embedding IDs
        if embedding_col in batch_adata.obs.columns:
            embedding_ids = torch.tensor(
                batch_adata.obs[embedding_col].values.astype(int),
                dtype=torch.long
            )
        else:
            embedding_ids = torch.zeros(data.shape[0], dtype=torch.long)

        # Compute scores
        batch_results = compute_cell_scores_batch(
            data, embedding_ids,
            model_teacher, model_student, model_encoder,
            device
        )

        for key in all_results:
            all_results[key].append(batch_results[key])

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {end_idx}/{n_cells} cells ({100*end_idx/n_cells:.1f}%)")

    # Concatenate results
    X_pretrained = np.concatenate(all_results['X_pretrained'])
    X_scmild = np.concatenate(all_results['X_scmild'])
    vq_codes = np.concatenate(all_results['vq_code'])
    attention_scores_raw = np.concatenate(all_results['attention_score_raw'])
    student_predictions = np.concatenate(all_results['student_prediction'])

    # Normalize attention scores
    sample_ids = adata.obs[sample_col].values
    attention_scores_global = normalize_attention_global(attention_scores_raw)
    attention_scores_sample = normalize_attention_per_sample(attention_scores_raw, sample_ids)

    # Create results DataFrame
    results_df = pd.DataFrame({
        'cell_id': adata.obs.index,
        'sample_id': sample_ids,
        'sample_name': adata.obs[sample_name_col].values,
        'disease_label': adata.obs[label_col].values,
        'attention_score_raw': attention_scores_raw,
        'attention_score_global': attention_scores_global,  # 전체 기준 정규화
        'attention_score_sample': attention_scores_sample,  # sample 내 정규화
        'student_prediction': student_predictions,
        'vq_code': vq_codes,
    })

    if fold_idx is not None:
        results_df['fold'] = fold_idx

    return results_df, X_pretrained, X_scmild, vq_codes


# ============================================================================
# Codebook AnnData Functions
# ============================================================================

def create_codebook_adata(
    codebook: np.ndarray,
    cell_results_df: pd.DataFrame,
    attn_direct: np.ndarray,
    attn_direct_folds: Optional[Dict[int, np.ndarray]] = None,
) -> sc.AnnData:
    """
    Create codebook AnnData with statistics.

    Args:
        codebook: Codebook embeddings (num_codes, latent_dim)
        cell_results_df: Cell-level results DataFrame (must have 'disease_label' column)
        attn_direct: Direct attention scores from codebook
        attn_direct_folds: Dict of fold_idx -> attention scores (CV mode)

    Returns:
        AnnData for codebook
    """
    num_codes = codebook.shape[0]

    # Create AnnData
    adata_codebook = sc.AnnData(X=codebook.astype(np.float32))
    adata_codebook.obs_names = [f"code_{i}" for i in range(num_codes)]

    # Initialize obs columns
    adata_codebook.obs['code_idx'] = list(range(num_codes))
    adata_codebook.obs['n_cells'] = 0
    adata_codebook.obs['n_samples'] = 0
    adata_codebook.obs['disease_ratio'] = np.nan

    # 방식 1: Codebook 직접 통과 attention
    adata_codebook.obs['attn_direct'] = attn_direct

    # CV 모드: fold별 직접 통과 attention
    if attn_direct_folds is not None:
        for fold_idx, attn in attn_direct_folds.items():
            adata_codebook.obs[f'attn_direct_fold{fold_idx}'] = attn

    # 방식 2: Cell 기반 통계
    adata_codebook.obs['attn_cell_mean'] = np.nan
    adata_codebook.obs['attn_cell_std'] = np.nan
    adata_codebook.obs['attn_cell_median'] = np.nan
    adata_codebook.obs['attn_cell_max'] = np.nan
    adata_codebook.obs['attn_cell_n'] = 0

    # CV 모드: fold별 cell 기반 통계 컬럼 초기화
    if 'fold' in cell_results_df.columns:
        folds = cell_results_df['fold'].unique()
        for fold_idx in folds:
            adata_codebook.obs[f'attn_cell_mean_fold{fold_idx}'] = np.nan
            adata_codebook.obs[f'attn_cell_std_fold{fold_idx}'] = np.nan

    # Compute statistics per code
    print("Computing codebook statistics...")
    for code_idx in range(num_codes):
        code_name = f"code_{code_idx}"
        mask = cell_results_df['vq_code'] == code_idx
        n_cells = mask.sum()

        adata_codebook.obs.loc[code_name, 'n_cells'] = n_cells

        if n_cells > 0:
            cells = cell_results_df[mask]
            # Use global normalized scores for codebook statistics
            scores = cells['attention_score_global']

            adata_codebook.obs.loc[code_name, 'n_samples'] = cells['sample_id'].nunique()
            adata_codebook.obs.loc[code_name, 'disease_ratio'] = cells['disease_label'].mean()
            adata_codebook.obs.loc[code_name, 'attn_cell_mean'] = scores.mean()
            adata_codebook.obs.loc[code_name, 'attn_cell_std'] = scores.std()
            adata_codebook.obs.loc[code_name, 'attn_cell_median'] = scores.median()
            adata_codebook.obs.loc[code_name, 'attn_cell_max'] = scores.max()
            adata_codebook.obs.loc[code_name, 'attn_cell_n'] = n_cells

            # CV 모드: fold별 cell 기반 통계
            if 'fold' in cells.columns:
                for fold_idx in cells['fold'].unique():
                    fold_mask = cells['fold'] == fold_idx
                    if fold_mask.sum() > 0:
                        fold_scores = cells.loc[fold_mask, 'attention_score_global']
                        adata_codebook.obs.loc[code_name, f'attn_cell_mean_fold{fold_idx}'] = fold_scores.mean()
                        adata_codebook.obs.loc[code_name, f'attn_cell_std_fold{fold_idx}'] = fold_scores.std()

    # Convert dtypes
    int_cols = ['code_idx', 'n_cells', 'n_samples', 'attn_cell_n']
    for col in int_cols:
        if col in adata_codebook.obs.columns:
            adata_codebook.obs[col] = adata_codebook.obs[col].astype(int)

    return adata_codebook


def add_scores_to_adata(
    adata: sc.AnnData,
    results_df: pd.DataFrame,
    X_pretrained: np.ndarray,
    X_scmild: np.ndarray,
    codebook: np.ndarray,
    model_info: dict
) -> sc.AnnData:
    """Add scoring results to adata."""

    # Ensure same order
    assert list(adata.obs.index) == list(results_df['cell_id']), "Cell order mismatch!"

    # Add obs columns
    adata.obs['attention_score_raw'] = results_df['attention_score_raw'].values
    adata.obs['attention_score_global'] = results_df['attention_score_global'].values
    adata.obs['attention_score_sample'] = results_df['attention_score_sample'].values
    adata.obs['student_prediction'] = results_df['student_prediction'].values
    adata.obs['vq_code'] = results_df['vq_code'].values.astype(int)

    if 'fold' in results_df.columns:
        adata.obs['fold'] = results_df['fold'].values.astype(int)

    # Add obsm
    adata.obsm['X_pretrained'] = X_pretrained.astype(np.float32)
    adata.obsm['X_scmild'] = X_scmild.astype(np.float32)

    # Add uns
    adata.uns['codebook'] = codebook.astype(np.float32)
    adata.uns['model_info'] = model_info

    return adata


# ============================================================================
# Mode-specific Processing Functions
# ============================================================================

def process_final_model_mode(
    model_dir: Path,
    adata: sc.AnnData,
    config: ScMILDConfig,
    device: torch.device,
    batch_size: int
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """Process in final model mode."""

    print("\n[Mode: Final Model]")
    print(f"Model directory: {model_dir}")

    # Load models
    model_teacher, model_student, model_encoder = load_trained_models(
        model_dir, device, config, fold_idx=0
    )

    # Compute cell scores
    results_df, X_pretrained, X_scmild, vq_codes = compute_cell_scores_for_adata(
        adata, model_teacher, model_student, model_encoder,
        device, config, batch_size, fold_idx=None
    )

    # Compute codebook direct attention
    attn_direct = compute_codebook_direct_attention(model_encoder, model_teacher, device)

    # Get codebook
    codebook = model_encoder.vq_model.quantizer.get_codebook().cpu().numpy()

    return results_df, X_pretrained, X_scmild, codebook, attn_direct, None


def process_cv_mode(
    cv_dir: Path,
    adata: sc.AnnData,
    config: ScMILDConfig,
    device: torch.device,
    batch_size: int
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """Process in CV mode - each fold model scores its test samples."""

    print("\n[Mode: Cross-Validation]")
    print(f"CV directory: {cv_dir}")

    # Find models directory
    models_dir = cv_dir / "models"
    if not models_dir.exists():
        models_dir = cv_dir  # Models might be in root

    # Get available folds
    fold_indices = get_fold_model_paths(models_dir)
    n_folds = len(fold_indices)
    print(f"Found {n_folds} fold models: {fold_indices}")

    # Get sample -> fold mapping
    sample_to_fold = get_sample_fold_mapping(cv_dir)
    print(f"Sample-fold mapping: {len(sample_to_fold)} samples")

    # Verify all samples have a fold
    sample_name_col = config.data.columns.sample_name
    adata_samples = set(adata.obs[sample_name_col].unique())
    mapped_samples = set(sample_to_fold.keys())

    if not adata_samples.issubset(mapped_samples):
        missing = adata_samples - mapped_samples
        print(f"WARNING: {len(missing)} samples not in CV predictions: {missing}")
        # Assign missing samples to fold 0
        for sample in missing:
            sample_to_fold[sample] = 0

    # Process each fold
    all_results = []
    all_X_pretrained = []
    all_X_scmild = []
    all_cell_indices = []
    attn_direct_folds = {}
    codebook = None

    for fold_idx in fold_indices:
        print(f"\n--- Fold {fold_idx} ---")

        # Get samples for this fold
        fold_samples = [s for s, f in sample_to_fold.items() if f == fold_idx]
        print(f"  Test samples: {fold_samples}")

        # Filter adata for this fold's samples
        fold_mask = adata.obs[sample_name_col].isin(fold_samples)
        fold_adata = adata[fold_mask].copy()

        if fold_adata.n_obs == 0:
            print(f"  No cells for fold {fold_idx}, skipping...")
            continue

        print(f"  Cells: {fold_adata.n_obs}")

        # Load models for this fold
        model_teacher, model_student, model_encoder = load_trained_models(
            models_dir, device, config, fold_idx=fold_idx
        )

        # Compute cell scores
        results_df, X_pretrained, X_scmild, vq_codes = compute_cell_scores_for_adata(
            fold_adata, model_teacher, model_student, model_encoder,
            device, config, batch_size, fold_idx=fold_idx
        )

        # Compute codebook direct attention for this fold
        attn_direct_fold = compute_codebook_direct_attention(model_encoder, model_teacher, device)
        attn_direct_folds[fold_idx] = attn_direct_fold

        # Store codebook (same for all folds - from pretrained encoder)
        if codebook is None:
            codebook = model_encoder.vq_model.quantizer.get_codebook().cpu().numpy()

        # Store results
        all_results.append(results_df)
        all_X_pretrained.append(X_pretrained)
        all_X_scmild.append(X_scmild)
        all_cell_indices.extend(fold_adata.obs.index.tolist())

        # Clean up
        del model_teacher, model_student, model_encoder
        gc.collect()
        torch.cuda.empty_cache()

    # Merge results from all folds
    print("\nMerging results from all folds...")
    merged_results_df = pd.concat(all_results, ignore_index=True)
    merged_X_pretrained = np.concatenate(all_X_pretrained, axis=0)
    merged_X_scmild = np.concatenate(all_X_scmild, axis=0)

    # Reorder to match original adata order
    cell_order_map = {cell_id: idx for idx, cell_id in enumerate(all_cell_indices)}
    original_order = [cell_order_map[cell_id] for cell_id in adata.obs.index]

    merged_results_df = merged_results_df.iloc[original_order].reset_index(drop=True)
    merged_results_df['cell_id'] = adata.obs.index.tolist()
    merged_X_pretrained = merged_X_pretrained[original_order]
    merged_X_scmild = merged_X_scmild[original_order]

    # Average attn_direct across folds for overall score
    attn_direct = np.mean(list(attn_direct_folds.values()), axis=0)

    return merged_results_df, merged_X_pretrained, merged_X_scmild, codebook, attn_direct, attn_direct_folds


def process_tuning_mode(
    tuning_dir: Path,
    adata: sc.AnnData,
    config: ScMILDConfig,
    device: torch.device,
    batch_size: int
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """Process in tuning mode - use best params models."""

    print("\n[Mode: Tuning]")
    print(f"Tuning directory: {tuning_dir}")

    # Find best config from tuning_results.csv
    tuning_results_path = tuning_dir / "tuning_results.csv"
    if not tuning_results_path.exists():
        raise FileNotFoundError(f"tuning_results.csv not found in {tuning_dir}")

    tuning_df = pd.read_csv(tuning_results_path)
    best_idx = tuning_df['mean_auc'].idxmax()
    best_config_id = int(tuning_df.loc[best_idx, 'config_id'])
    best_auc = tuning_df.loc[best_idx, 'mean_auc']
    print(f"Best config: config_{best_config_id:03d} (AUC={best_auc:.4f})")

    # Find best model directory
    best_config_dir = tuning_dir / "models" / f"config_{best_config_id:03d}"
    if not best_config_dir.exists():
        raise FileNotFoundError(f"Best config directory not found: {best_config_dir}")

    # Get fold directories
    fold_dirs = sorted(best_config_dir.glob("fold_*"))
    n_folds = len(fold_dirs)
    print(f"Found {n_folds} folds in best config")

    # Recreate sample -> fold mapping using the same splitter
    # (Tuning used the same splitter config, so we regenerate the same splits)
    from src.data import create_splitter, get_sample_info_from_adata

    sample_col = config.data.columns.sample_id
    label_col = config.data.columns.disease_label
    sample_name_col = config.data.columns.sample_name

    sample_ids, labels, sample_names = get_sample_info_from_adata(
        adata,
        sample_col=sample_col,
        label_col=label_col,
        sample_name_col=sample_name_col
    )

    splitter = create_splitter(
        strategy=config.splitting.strategy,
        n_splits=config.splitting.n_splits,
        n_repeats=config.splitting.n_repeats,
        random_seed=config.splitting.random_seed
    )

    # Build sample_name -> fold mapping
    # fold_info.test_samples contains sample_ids, need to map to sample_names
    sample_id_to_name = dict(zip(sample_ids, sample_names))
    sample_to_fold = {}
    for fold_info in splitter.split(sample_ids, labels, sample_names):
        fold_idx = fold_info.fold_idx
        for sample_id in fold_info.test_samples:
            sample_name = sample_id_to_name[sample_id]
            sample_to_fold[sample_name] = fold_idx

    print(f"Recreated sample-fold mapping: {len(sample_to_fold)} samples")

    # Process each fold
    all_results = []
    all_X_pretrained = []
    all_X_scmild = []
    all_cell_indices = []
    attn_direct_folds = {}
    codebook = None

    for fold_idx in range(n_folds):
        fold_dir = best_config_dir / f"fold_{fold_idx:02d}"
        if not fold_dir.exists():
            print(f"  Fold {fold_idx} directory not found, skipping...")
            continue

        print(f"\n--- Fold {fold_idx} ---")

        # Get samples for this fold
        fold_samples = [s for s, f in sample_to_fold.items() if f == fold_idx]
        print(f"  Test samples: {fold_samples}")

        # Filter adata for this fold's samples
        fold_mask = adata.obs[sample_name_col].isin(fold_samples)
        fold_adata = adata[fold_mask].copy()

        if fold_adata.n_obs == 0:
            print(f"  No cells for fold {fold_idx}, skipping...")
            continue

        print(f"  Cells: {fold_adata.n_obs}")

        # Load models for this fold (tuning format)
        model_teacher, model_student, model_encoder = load_trained_models_tuning(
            fold_dir, device, config
        )

        # Compute cell scores
        results_df, X_pretrained, X_scmild, vq_codes = compute_cell_scores_for_adata(
            fold_adata, model_teacher, model_student, model_encoder,
            device, config, batch_size, fold_idx=fold_idx
        )

        # Compute codebook direct attention for this fold
        attn_direct_fold = compute_codebook_direct_attention(model_encoder, model_teacher, device)
        attn_direct_folds[fold_idx] = attn_direct_fold

        # Store codebook (same for all folds - from pretrained encoder)
        if codebook is None:
            codebook = model_encoder.vq_model.quantizer.get_codebook().cpu().numpy()

        # Store results
        all_results.append(results_df)
        all_X_pretrained.append(X_pretrained)
        all_X_scmild.append(X_scmild)
        all_cell_indices.extend(fold_adata.obs.index.tolist())

        # Clean up
        del model_teacher, model_student, model_encoder
        gc.collect()
        torch.cuda.empty_cache()

    # Merge results from all folds
    print("\nMerging results from all folds...")
    merged_results_df = pd.concat(all_results, ignore_index=True)
    merged_X_pretrained = np.concatenate(all_X_pretrained, axis=0)
    merged_X_scmild = np.concatenate(all_X_scmild, axis=0)

    # Reorder to match original adata order
    cell_order_map = {cell_id: idx for idx, cell_id in enumerate(all_cell_indices)}
    original_order = [cell_order_map[cell_id] for cell_id in adata.obs.index]

    merged_results_df = merged_results_df.iloc[original_order].reset_index(drop=True)
    merged_results_df['cell_id'] = adata.obs.index.tolist()
    merged_X_pretrained = merged_X_pretrained[original_order]
    merged_X_scmild = merged_X_scmild[original_order]

    # Average attn_direct across folds for overall score
    attn_direct = np.mean(list(attn_direct_folds.values()), axis=0)

    return merged_results_df, merged_X_pretrained, merged_X_scmild, codebook, attn_direct, attn_direct_folds


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute cell-level scores with trained scMILD models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--model_dir", type=str, help="Path to final model directory")
    mode_group.add_argument("--cv_dir", type=str, help="Path to CV results directory")
    mode_group.add_argument("--tuning_dir", type=str, help="Path to tuning results directory")

    # Required arguments
    parser.add_argument("--config", type=str, required=True, help="Config for dataset")

    # Optional arguments
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--output_format", type=str, default="h5ad",
                        choices=["h5ad", "csv"], help="Output format (default: h5ad)")
    parser.add_argument("--save_codebook_adata", action="store_true", default=True,
                        help="Save codebook AnnData (default: True)")
    parser.add_argument("--no_codebook_adata", action="store_true",
                        help="Do not save codebook AnnData")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size for processing")

    args = parser.parse_args()

    # Determine mode
    if args.model_dir:
        mode = "final"
        source_dir = Path(args.model_dir)
    elif args.cv_dir:
        mode = "cv"
        source_dir = Path(args.cv_dir)
    else:
        mode = "tuning"
        source_dir = Path(args.tuning_dir)

    # Setup device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    config = load_config(args.config)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = source_dir / f"cell_scores_{timestamp}"
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

    # Ensure embedding column exists
    adata = ensure_embedding_column(adata, config)

    # Process based on mode
    if mode == "final":
        results_df, X_pretrained, X_scmild, codebook, attn_direct, attn_direct_folds = \
            process_final_model_mode(source_dir, adata, config, device, args.batch_size)
    elif mode == "cv":
        results_df, X_pretrained, X_scmild, codebook, attn_direct, attn_direct_folds = \
            process_cv_mode(source_dir, adata, config, device, args.batch_size)
    else:  # tuning
        results_df, X_pretrained, X_scmild, codebook, attn_direct, attn_direct_folds = \
            process_tuning_mode(source_dir, adata, config, device, args.batch_size)

    # Model info for metadata
    model_info = {
        'mode': mode,
        'source_dir': str(source_dir),
        'config_path': args.config,
        'timestamp': datetime.now().isoformat(),
        'n_cells': len(results_df),
        'n_codes': codebook.shape[0],
        'latent_dim': codebook.shape[1],
    }

    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")

    # Always save CSV
    results_df.to_csv(output_dir / "cell_scores.csv", index=False)
    print(f"  Saved: cell_scores.csv")

    if args.output_format == "h5ad":
        # Add scores to adata and save
        scored_adata = add_scores_to_adata(
            adata.copy(), results_df, X_pretrained, X_scmild, codebook, model_info
        )
        scored_adata.write_h5ad(output_dir / "scored_adata.h5ad")
        print(f"  Saved: scored_adata.h5ad")

        # Save codebook adata
        if args.save_codebook_adata and not args.no_codebook_adata:
            codebook_adata = create_codebook_adata(
                codebook, results_df, attn_direct, attn_direct_folds
            )
            codebook_adata.write_h5ad(output_dir / "codebook_adata.h5ad")
            print(f"  Saved: codebook_adata.h5ad")

    # Save model info
    with open(output_dir / "model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"  Saved: model_info.json")

    # Print summary
    print(f"\n{'='*60}")
    print("Cell Scoring Complete!")
    print(f"{'='*60}")

    # Summary by sample
    sample_summary = results_df.groupby(['sample_name', 'disease_label'], observed=True).agg({
        'attention_score_global': ['mean', 'std', 'max'],
        'student_prediction': ['mean', 'std'],
        'cell_id': 'count'
    }).round(4)
    sample_summary.columns = ['attn_mean', 'attn_std', 'attn_max', 'student_mean', 'student_std', 'n_cells']
    sample_summary = sample_summary.reset_index()

    print("\nSample-level summary:")
    print(sample_summary.to_string(index=False))
    sample_summary.to_csv(output_dir / "sample_summary.csv", index=False)

    # Codebook usage summary
    vq_code_counts = results_df['vq_code'].value_counts()
    print(f"\nCodebook usage:")
    print(f"  Active codes: {len(vq_code_counts)} / {codebook.shape[0]}")
    print(f"  Most used codes: {vq_code_counts.head(5).to_dict()}")

    if 'fold' in results_df.columns:
        print(f"\nFold distribution:")
        print(results_df['fold'].value_counts().sort_index().to_string())

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
