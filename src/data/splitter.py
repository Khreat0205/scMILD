"""
Data splitting strategies for scMILD.

샘플 레벨의 데이터 분할 전략들입니다:
- LOOCVSplitter: Leave-One-Out Cross Validation
- StratifiedKFoldSplitter: Stratified K-Fold Cross Validation
"""

import numpy as np
from typing import List, Tuple, Iterator, Optional, Literal
from dataclasses import dataclass


@dataclass
class FoldInfo:
    """Information about a single fold."""
    fold_idx: int
    train_samples: List[int]
    test_samples: List[int]
    test_sample_name: Optional[str] = None  # For LOOCV: name of left-out sample


class BaseSplitter:
    """Base class for data splitters."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

    def split(
        self,
        sample_ids: np.ndarray,
        labels: np.ndarray,
        sample_names: Optional[np.ndarray] = None
    ) -> Iterator[FoldInfo]:
        """
        Generate train/test splits.

        Args:
            sample_ids: Array of sample IDs
            labels: Array of labels for each sample
            sample_names: Optional array of sample names

        Yields:
            FoldInfo for each fold
        """
        raise NotImplementedError

    def get_n_splits(self, sample_ids: np.ndarray) -> int:
        """Return number of splits."""
        raise NotImplementedError


class LOOCVSplitter(BaseSplitter):
    """
    Leave-One-Out Cross Validation splitter.

    각 샘플을 한 번씩 테스트 세트로 사용합니다.
    샘플 수가 적을 때 (< 30) 권장됩니다.

    Note: LOOCV에서는 early stopping을 사용하지 않습니다.
    """

    def __init__(self, random_seed: int = 42, shuffle: bool = False):
        """
        Args:
            random_seed: Random seed for shuffling
            shuffle: Whether to shuffle samples before splitting
        """
        super().__init__(random_seed)
        self.shuffle = shuffle

    def split(
        self,
        sample_ids: np.ndarray,
        labels: np.ndarray,
        sample_names: Optional[np.ndarray] = None
    ) -> Iterator[FoldInfo]:
        """
        Generate LOOCV splits.

        Args:
            sample_ids: Array of unique sample IDs
            labels: Array of labels for each sample
            sample_names: Optional array of sample names

        Yields:
            FoldInfo for each fold (one per sample)
        """
        n_samples = len(sample_ids)
        indices = np.arange(n_samples)

        if self.shuffle:
            self.rng.shuffle(indices)

        for fold_idx, test_idx in enumerate(indices):
            train_indices = [i for i in indices if i != test_idx]

            test_sample_name = None
            if sample_names is not None:
                test_sample_name = sample_names[test_idx]

            yield FoldInfo(
                fold_idx=fold_idx,
                train_samples=list(sample_ids[train_indices]),
                test_samples=[sample_ids[test_idx]],
                test_sample_name=test_sample_name
            )

    def get_n_splits(self, sample_ids: np.ndarray) -> int:
        """Return number of samples (= number of folds)."""
        return len(sample_ids)


class StratifiedKFoldSplitter(BaseSplitter):
    """
    Stratified K-Fold Cross Validation splitter.

    각 fold에서 클래스 비율을 유지합니다.

    Args:
        n_splits: Number of folds
        n_repeats: Number of repetitions (for Repeated Stratified K-Fold)
        random_seed: Random seed
        shuffle: Whether to shuffle before splitting
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 1,
        random_seed: int = 42,
        shuffle: bool = True
    ):
        super().__init__(random_seed)
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.shuffle = shuffle

    def split(
        self,
        sample_ids: np.ndarray,
        labels: np.ndarray,
        sample_names: Optional[np.ndarray] = None
    ) -> Iterator[FoldInfo]:
        """
        Generate Stratified K-Fold splits.

        Args:
            sample_ids: Array of unique sample IDs
            labels: Array of labels for each sample
            sample_names: Optional array of sample names

        Yields:
            FoldInfo for each fold
        """
        from sklearn.model_selection import StratifiedKFold

        fold_idx = 0

        for repeat in range(self.n_repeats):
            # Different random state for each repeat
            skf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_seed + repeat
            )

            for train_idx, test_idx in skf.split(sample_ids, labels):
                yield FoldInfo(
                    fold_idx=fold_idx,
                    train_samples=list(sample_ids[train_idx]),
                    test_samples=list(sample_ids[test_idx]),
                    test_sample_name=None
                )
                fold_idx += 1

    def get_n_splits(self, sample_ids: np.ndarray) -> int:
        """Return total number of folds (n_splits * n_repeats)."""
        return self.n_splits * self.n_repeats


def create_splitter(
    strategy: Literal["loocv", "stratified_kfold", "repeated_stratified_kfold"],
    n_splits: int = 5,
    n_repeats: int = 1,
    random_seed: int = 42,
    shuffle: bool = True
) -> BaseSplitter:
    """
    Factory function to create appropriate splitter.

    Args:
        strategy: Splitting strategy name
        n_splits: Number of folds (for kfold strategies)
        n_repeats: Number of repetitions (for repeated kfold)
        random_seed: Random seed
        shuffle: Whether to shuffle

    Returns:
        Splitter instance
    """
    if strategy == "loocv":
        return LOOCVSplitter(random_seed=random_seed, shuffle=shuffle)

    elif strategy == "stratified_kfold":
        return StratifiedKFoldSplitter(
            n_splits=n_splits,
            n_repeats=1,
            random_seed=random_seed,
            shuffle=shuffle
        )

    elif strategy == "repeated_stratified_kfold":
        return StratifiedKFoldSplitter(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_seed=random_seed,
            shuffle=shuffle
        )

    else:
        raise ValueError(f"Unknown splitting strategy: {strategy}")


def get_sample_info_from_adata(
    adata,
    sample_col: str = "sample_id_numeric",
    label_col: str = "disease_numeric",
    sample_name_col: str = "sample"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    AnnData에서 샘플 정보를 추출합니다.

    Args:
        adata: AnnData object
        sample_col: Column name for sample IDs
        label_col: Column name for labels
        sample_name_col: Column name for sample names

    Returns:
        sample_ids: Unique sample IDs
        labels: Label for each sample
        sample_names: Name for each sample
    """
    # Get unique samples with their labels
    sample_info = adata.obs.groupby(sample_col).agg({
        label_col: 'first',
        sample_name_col: 'first'
    }).reset_index()

    sample_ids = sample_info[sample_col].values
    labels = sample_info[label_col].values
    sample_names = sample_info[sample_name_col].values

    return sample_ids, labels, sample_names


def print_split_summary(
    splitter: BaseSplitter,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    sample_names: Optional[np.ndarray] = None
):
    """
    분할 결과 요약을 출력합니다.

    Args:
        splitter: Splitter instance
        sample_ids: Sample IDs
        labels: Labels
        sample_names: Sample names
    """
    n_splits = splitter.get_n_splits(sample_ids)
    n_class_0 = (labels == 0).sum()
    n_class_1 = (labels == 1).sum()

    print(f"=" * 50)
    print(f"Data Split Summary")
    print(f"=" * 50)
    print(f"Splitter: {splitter.__class__.__name__}")
    print(f"Total samples: {len(sample_ids)}")
    print(f"  - Class 0 (Control): {n_class_0}")
    print(f"  - Class 1 (Disease): {n_class_1}")
    print(f"Number of folds: {n_splits}")
    print(f"=" * 50)

    # Print first few folds
    print("\nFirst 3 folds preview:")
    for i, fold in enumerate(splitter.split(sample_ids, labels, sample_names)):
        if i >= 3:
            print("...")
            break

        train_labels = labels[np.isin(sample_ids, fold.train_samples)]
        test_labels = labels[np.isin(sample_ids, fold.test_samples)]

        print(f"\nFold {fold.fold_idx}:")
        print(f"  Train: {len(fold.train_samples)} samples "
              f"(C0: {(train_labels==0).sum()}, C1: {(train_labels==1).sum()})")
        print(f"  Test:  {len(fold.test_samples)} samples "
              f"(C0: {(test_labels==0).sum()}, C1: {(test_labels==1).sum()})")

        if fold.test_sample_name:
            print(f"  Test sample: {fold.test_sample_name}")
