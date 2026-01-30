"""
Preprocessing utilities for scMILD.

AnnData 전처리 및 로딩 함수들입니다.
"""

import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path


def load_adata(path: str):
    """
    AnnData 파일을 로드합니다.

    Args:
        path: Path to .h5ad file

    Returns:
        AnnData object
    """
    import scanpy as sc
    return sc.read_h5ad(path)


def load_adata_with_subset(
    whole_adata_path: str,
    subset_enabled: bool = False,
    subset_column: str = "study",
    subset_values: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    dataset_name: Optional[str] = None
):
    """
    AnnData를 로드하고 필요시 subset을 적용합니다.

    캐시가 활성화되면 subset된 adata를 저장/로드하여 재사용합니다.

    Args:
        whole_adata_path: Path to whole .h5ad file
        subset_enabled: Whether to apply subset filtering
        subset_column: Column to filter on
        subset_values: Values to include in subset
        cache_dir: Directory to cache subset adata
        use_cache: Whether to use cache
        dataset_name: Name for cache file (auto-generated if None)

    Returns:
        AnnData object (whole or subset)
    """
    import scanpy as sc
    import os

    # 캐시 파일 경로 결정
    cache_path = None
    if use_cache and cache_dir and subset_enabled and subset_values:
        os.makedirs(cache_dir, exist_ok=True)
        if dataset_name:
            cache_filename = f"{dataset_name}.h5ad"
        else:
            # subset_values로 파일명 생성
            values_str = "_".join(sorted(subset_values))[:50]  # 길이 제한
            cache_filename = f"subset_{subset_column}_{values_str}.h5ad"
        cache_path = os.path.join(cache_dir, cache_filename)

    # 캐시 로드 시도
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached subset from: {cache_path}")
        return sc.read_h5ad(cache_path)

    # 전체 데이터 로드
    print(f"Loading whole adata from: {whole_adata_path}")
    adata = sc.read_h5ad(whole_adata_path)

    # Subset 적용
    if subset_enabled and subset_values:
        if subset_column not in adata.obs.columns:
            raise ValueError(f"Subset column '{subset_column}' not found in adata.obs")

        print(f"Subsetting by {subset_column}: {subset_values}")
        mask = adata.obs[subset_column].isin(subset_values)
        adata = adata[mask].copy()
        print(f"Subset shape: {adata.n_obs} cells × {adata.n_vars} genes")

        # 캐시 저장
        if cache_path:
            print(f"Saving subset to cache: {cache_path}")
            adata.write_h5ad(cache_path)

    return adata


def get_cache_path(
    cache_dir: str,
    subset_column: str,
    subset_values: List[str],
    dataset_name: Optional[str] = None
) -> str:
    """캐시 파일 경로를 생성합니다."""
    import os

    if dataset_name:
        return os.path.join(cache_dir, f"{dataset_name}.h5ad")

    values_str = "_".join(sorted(subset_values))[:50]
    return os.path.join(cache_dir, f"subset_{subset_column}_{values_str}.h5ad")


def preprocess_adata(
    adata,
    n_top_genes: int = 6000,
    normalize_total: int = 10000,
    log_transform: bool = True,
    subset_hvg: bool = True,
    inplace: bool = False
):
    """
    단일 세포 RNA-seq 데이터 전처리.

    Args:
        adata: AnnData object
        n_top_genes: Number of highly variable genes to select
        normalize_total: Target sum for normalization
        log_transform: Whether to apply log1p transformation
        subset_hvg: Whether to subset to HVG only
        inplace: Whether to modify in place

    Returns:
        Preprocessed AnnData object
    """
    import scanpy as sc

    if not inplace:
        adata = adata.copy()

    # Normalize
    if normalize_total:
        sc.pp.normalize_total(adata, target_sum=normalize_total)

    # Log transform
    if log_transform:
        sc.pp.log1p(adata)

    # Highly variable genes
    if n_top_genes and n_top_genes < adata.n_vars:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        if subset_hvg:
            adata = adata[:, adata.var.highly_variable]

    return adata


def encode_labels(
    adata,
    sample_col: str = "sample",
    label_col: str = "Status",
    conditional_col: Optional[str] = "study",
    conditional_encoded_col: Optional[str] = None,
    inplace: bool = True
):
    """
    카테고리 변수를 숫자로 인코딩합니다.

    Args:
        adata: AnnData object
        sample_col: Column name for samples
        label_col: Column name for disease labels
        conditional_col: Column name for conditional embedding (e.g., 'study', 'Organ')
        conditional_encoded_col: Output column name for encoded conditional values
                                 (default: '{conditional_col}_id_numeric')
        inplace: Whether to modify in place

    Returns:
        AnnData with encoded columns
        encoding_info: Dictionary with encoding mappings
    """
    from sklearn.preprocessing import LabelEncoder

    if not inplace:
        adata = adata.copy()

    encoding_info = {}

    # Encode sample IDs
    le_sample = LabelEncoder()
    adata.obs['sample_id_numeric'] = le_sample.fit_transform(adata.obs[sample_col])
    encoding_info['sample'] = {
        'encoder': le_sample,
        'mapping': dict(zip(le_sample.classes_, range(len(le_sample.classes_))))
    }

    # Encode disease labels
    le_label = LabelEncoder()
    adata.obs['disease_numeric'] = le_label.fit_transform(adata.obs[label_col])
    encoding_info['label'] = {
        'encoder': le_label,
        'mapping': dict(zip(le_label.classes_, range(len(le_label.classes_))))
    }

    # Encode conditional column (e.g., study, Organ)
    if conditional_col and conditional_col in adata.obs.columns:
        le_conditional = LabelEncoder()
        # Determine output column name
        if conditional_encoded_col is None:
            conditional_encoded_col = f"{conditional_col}_id_numeric"
        adata.obs[conditional_encoded_col] = le_conditional.fit_transform(adata.obs[conditional_col])
        encoding_info['conditional'] = {
            'column': conditional_col,
            'encoded_column': conditional_encoded_col,
            'encoder': le_conditional,
            'mapping': dict(zip(le_conditional.classes_, range(len(le_conditional.classes_))))
        }

    return adata, encoding_info


def get_study_mapping(adata, sample_col: str = "sample_id_numeric", study_col: str = "study_id_numeric") -> dict:
    """
    Sample ID → Study ID 매핑을 생성합니다.

    Args:
        adata: AnnData object
        sample_col: Column name for sample IDs
        study_col: Column name for study IDs

    Returns:
        Dictionary mapping sample_id → study_id
    """
    if study_col not in adata.obs.columns:
        return {}

    mapping_df = adata.obs[[sample_col, study_col]].drop_duplicates()
    return dict(zip(
        mapping_df[sample_col].astype(int),
        mapping_df[study_col].astype(int)
    ))


def save_study_mapping(mapping: dict, path: str):
    """Save study mapping to JSON file."""
    import json
    with open(path, 'w') as f:
        # Convert keys to strings for JSON
        json.dump({str(k): v for k, v in mapping.items()}, f)


def load_study_mapping(path: str) -> dict:
    """Load study mapping from JSON file."""
    import json
    with open(path, 'r') as f:
        mapping = json.load(f)
    # Convert keys back to int
    return {int(k): v for k, v in mapping.items()}


def print_adata_summary(adata, title: str = "AnnData Summary"):
    """
    AnnData 요약 정보를 출력합니다.

    Args:
        adata: AnnData object
        title: Title for the summary
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"Shape: {adata.n_obs} cells × {adata.n_vars} genes")

    # Sample info
    if 'sample_id_numeric' in adata.obs.columns:
        n_samples = adata.obs['sample_id_numeric'].nunique()
        print(f"Samples: {n_samples}")

    # Disease distribution
    if 'disease_numeric' in adata.obs.columns:
        disease_counts = adata.obs['disease_numeric'].value_counts()
        print(f"Disease distribution (cells):")
        for label, count in disease_counts.items():
            print(f"  - Class {label}: {count}")

    # Sample-level disease distribution
    if 'sample_id_numeric' in adata.obs.columns and 'disease_numeric' in adata.obs.columns:
        sample_labels = adata.obs.groupby('sample_id_numeric')['disease_numeric'].first()
        print(f"Disease distribution (samples):")
        for label in sample_labels.unique():
            count = (sample_labels == label).sum()
            print(f"  - Class {label}: {count}")

    # Study distribution (if available)
    if 'study' in adata.obs.columns:
        study_counts = adata.obs['study'].value_counts()
        print(f"Studies: {len(study_counts)}")
        for study, count in study_counts.head(5).items():
            print(f"  - {study}: {count}")
        if len(study_counts) > 5:
            print(f"  ... and {len(study_counts) - 5} more")

    print(f"{'='*50}\n")


def subsample_adata(
    adata,
    max_cells_per_sample: int = 5000,
    sample_col: str = "sample_id_numeric",
    random_seed: int = 42
):
    """
    샘플당 세포 수를 제한합니다.

    세포 수가 많은 샘플에서 메모리/시간 절약을 위해 사용합니다.

    Args:
        adata: AnnData object
        max_cells_per_sample: Maximum cells per sample
        sample_col: Column name for sample IDs
        random_seed: Random seed

    Returns:
        Subsampled AnnData
    """
    np.random.seed(random_seed)

    indices_to_keep = []

    for sample_id in adata.obs[sample_col].unique():
        sample_mask = adata.obs[sample_col] == sample_id
        sample_indices = np.where(sample_mask)[0]

        if len(sample_indices) > max_cells_per_sample:
            # Randomly sample
            selected = np.random.choice(
                sample_indices, size=max_cells_per_sample, replace=False
            )
            indices_to_keep.extend(selected)
        else:
            indices_to_keep.extend(sample_indices)

    indices_to_keep = sorted(indices_to_keep)
    return adata[indices_to_keep].copy()
