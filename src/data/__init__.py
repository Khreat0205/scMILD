"""
scMILD Data module

데이터 로딩, 전처리, 분할을 위한 모듈입니다:
- Dataset classes (MilDataset, InstanceDataset)
- Data splitters (LOOCV, StratifiedKFold)
- Preprocessing utilities
"""

from .dataset import (
    MilDataset,
    InstanceDataset,
    InstanceDatasetWithBagLabel,
    collate_mil,
    create_instance_dataset_with_bag_labels,
    create_datasets_from_adata,
)
from .splitter import (
    LOOCVSplitter,
    StratifiedKFoldSplitter,
    create_splitter,
    get_sample_info_from_adata,
    print_split_summary,
    FoldInfo,
)
from .preprocessing import (
    preprocess_adata,
    load_adata,
    load_adata_with_subset,
    get_cache_path,
    print_adata_summary,
    subsample_adata,
    encode_labels,
    # Conditional mapping functions (general purpose)
    get_conditional_mapping,
    load_conditional_mapping,
    save_conditional_mapping,
    # Backward compatibility aliases
    get_study_mapping,
    load_study_mapping,
    save_study_mapping,
)

__all__ = [
    # Datasets
    "MilDataset",
    "InstanceDataset",
    "InstanceDatasetWithBagLabel",
    "collate_mil",
    "create_instance_dataset_with_bag_labels",
    "create_datasets_from_adata",
    # Splitters
    "LOOCVSplitter",
    "StratifiedKFoldSplitter",
    "create_splitter",
    "get_sample_info_from_adata",
    "print_split_summary",
    "FoldInfo",
    # Preprocessing
    "preprocess_adata",
    "load_adata",
    "load_adata_with_subset",
    "get_cache_path",
    "print_adata_summary",
    "subsample_adata",
    "encode_labels",
    # Conditional mapping functions (general purpose)
    "get_conditional_mapping",
    "load_conditional_mapping",
    "save_conditional_mapping",
    # Backward compatibility aliases
    "get_study_mapping",
    "load_study_mapping",
    "save_study_mapping",
]
