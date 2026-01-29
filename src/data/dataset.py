"""
Dataset classes for scMILD.

MIL (Multiple Instance Learning) 구조의 데이터셋 클래스들입니다:
- MilDataset: 샘플(Bag) 단위 데이터셋
- InstanceDataset: 세포(Instance) 단위 데이터셋
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, Union, List


class MilDataset(Dataset):
    """
    MIL Dataset for sample-level learning.

    각 샘플(Bag)은 여러 세포(Instance)들의 집합입니다.

    Args:
        data: Cell feature data (n_cells, n_features)
        ids: Sample ID for each cell (n_cells,)
        labels: Sample labels (n_samples,)
        instance_labels: Cell labels (n_cells,) - 보통 bag label과 동일
        study_ids: Study ID for each cell (n_cells,) - Conditional model용
    """

    def __init__(
        self,
        data: torch.Tensor,
        ids: torch.Tensor,
        labels: torch.Tensor,
        instance_labels: torch.Tensor,
        study_ids: Optional[torch.Tensor] = None
    ):
        self.data = data
        self.labels = labels
        self.ids = ids
        self.instance_labels = instance_labels
        self.study_ids = study_ids

        # Ensure ids is 2D for compatibility
        if len(ids.shape) == 1:
            self.ids = ids.unsqueeze(0)
        else:
            self.ids = ids

        self.bags = torch.unique(self.ids[0])

    def __len__(self) -> int:
        """Return number of samples (bags)."""
        return len(self.bags)

    def __getitem__(self, index: int) -> Tuple:
        """
        Get all cells belonging to a sample.

        Returns:
            data: Cell features for this sample
            bagids: Sample ID tensor
            label: Sample label
            instance_labels: Cell labels
            study_ids: Study IDs (if available)
        """
        bag_id = self.bags[index]
        bagids = self.ids[:, self.ids[0] == bag_id]
        label = self.labels[index]
        indices = torch.where(self.ids[0] == bag_id)[0]
        data = self.data[indices]
        bag_instance_labels = self.instance_labels[indices]

        if self.study_ids is not None:
            bag_study_ids = self.study_ids[indices]
            return data, bagids, label, bag_instance_labels, bag_study_ids

        return data, bagids, label, bag_instance_labels

    def n_features(self) -> int:
        """Return number of features per cell."""
        return self.data.size(1)

    def get_sample_info(self, index: int) -> dict:
        """Get detailed information about a sample."""
        bag_id = self.bags[index]
        indices = torch.where(self.ids[0] == bag_id)[0]
        return {
            'bag_id': bag_id.item(),
            'n_cells': len(indices),
            'label': self.labels[index].item()
        }


class InstanceDataset(Dataset):
    """
    Instance-level Dataset for cell-level learning.

    개별 세포 단위의 데이터셋입니다.

    Args:
        data: Cell feature data (n_cells, n_features)
        ids: Sample ID for each cell (n_cells,)
        labels: Sample labels (n_samples,)
        instance_labels: Cell labels (n_cells,)
        study_ids: Study ID for each cell (n_cells,)
        normalize: Whether to normalize data
    """

    def __init__(
        self,
        data: torch.Tensor,
        ids: torch.Tensor,
        labels: torch.Tensor,
        instance_labels: torch.Tensor,
        study_ids: Optional[torch.Tensor] = None,
        normalize: bool = False
    ):
        self.data = data
        self.labels = labels
        self.ids = ids
        self.instance_labels = instance_labels
        self.study_ids = study_ids

        # For compatibility with MIL operations
        self.mil_ids = ids.clone()
        if len(self.mil_ids.shape) == 1:
            self.mil_ids = self.mil_ids.unsqueeze(0)
        self.bags = torch.unique(self.mil_ids[0])

        if normalize:
            std = self.data.std(dim=0)
            mean = self.data.mean(dim=0)
            std[std == 0] = 1  # Avoid division by zero
            self.data = (self.data - mean) / std

    def __len__(self) -> int:
        """Return number of cells."""
        return self.data.size(0)

    def __getitem__(self, index: int) -> Tuple:
        """
        Get a single cell's data.

        Returns:
            data: Cell features
            bag_id: Sample ID this cell belongs to
            instance_label: Cell label
            study_id: Study ID (if available)
        """
        data = self.data[index]
        bag_id = self.ids[index]
        instance_label = self.instance_labels[index]

        if self.study_ids is not None:
            study_id = self.study_ids[index]
            return data, bag_id, instance_label, study_id

        return data, bag_id, instance_label


class InstanceDatasetWithBagLabel(Dataset):
    """
    Instance-level Dataset with bag (sample) labels.

    Student branch 학습을 위해 각 세포에 해당 샘플의 레이블을 추가합니다.

    Args:
        data: Cell feature data (n_cells, n_features)
        ids: Sample ID for each cell (n_cells,)
        labels: Sample labels (n_samples,)
        instance_labels: Cell labels (n_cells,)
        bag_labels: Sample label for each cell (n_cells,)
        study_ids: Study ID for each cell (n_cells,)
    """

    def __init__(
        self,
        data: torch.Tensor,
        ids: torch.Tensor,
        labels: torch.Tensor,
        instance_labels: torch.Tensor,
        bag_labels: torch.Tensor,
        study_ids: Optional[torch.Tensor] = None
    ):
        self.data = data
        self.labels = labels
        self.ids = ids
        self.instance_labels = instance_labels
        self.bag_labels = bag_labels
        self.study_ids = study_ids

        self.mil_ids = ids.clone()
        if len(self.mil_ids.shape) == 1:
            self.mil_ids = self.mil_ids.unsqueeze(0)
        self.bags = torch.unique(self.mil_ids[0])

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, index: int) -> Tuple:
        """
        Returns:
            data: Cell features
            bag_id: Sample ID
            instance_label: Cell label
            bag_label: Sample label
            study_id: Study ID (if available)
        """
        data = self.data[index]
        bag_id = self.ids[index]
        instance_label = self.instance_labels[index]
        bag_label = self.bag_labels[index]

        if self.study_ids is not None:
            study_id = self.study_ids[index]
            return data, bag_id, instance_label, bag_label, study_id

        return data, bag_id, instance_label, bag_label


def collate_mil(batch: List[Tuple]) -> Tuple:
    """
    Collate function for MilDataset.

    여러 샘플의 세포들을 하나의 배치로 결합합니다.

    Args:
        batch: List of samples from MilDataset

    Returns:
        data: All cells concatenated (total_cells, n_features)
        bagids: Sample IDs (1, total_cells)
        labels: Sample labels (batch_size,)
        study_ids: Study IDs (total_cells,) if available
    """
    batch_data = []
    batch_bagids = []
    batch_labels = []
    batch_study_ids = []

    # Check if batch contains study_ids (length 5)
    has_study_ids = len(batch[0]) == 5

    for sample in batch:
        batch_data.append(sample[0])
        batch_bagids.append(sample[1])
        batch_labels.append(sample[2])
        if has_study_ids:
            batch_study_ids.append(sample[4])

    out_data = torch.cat(batch_data, dim=0)
    out_bagids = torch.cat(batch_bagids, dim=1)
    out_labels = torch.stack(batch_labels)

    if has_study_ids:
        out_study_ids = torch.cat(batch_study_ids, dim=0)
        return out_data, out_bagids, out_labels, out_study_ids

    return out_data, out_bagids, out_labels


def create_instance_dataset_with_bag_labels(
    instance_dataset: InstanceDataset,
    device: torch.device
) -> InstanceDatasetWithBagLabel:
    """
    InstanceDataset에 bag label을 추가한 새로운 데이터셋을 생성합니다.

    Args:
        instance_dataset: Original InstanceDataset
        device: Device for tensors

    Returns:
        InstanceDatasetWithBagLabel with bag labels for each cell
    """
    n_cells = len(instance_dataset)
    bag_labels = torch.empty(n_cells, dtype=torch.long, device=device)

    for i in range(n_cells):
        item = instance_dataset[i]
        bag_id = item[1]  # bag_id is always second element

        # Find bag index and get its label
        bag_index = (instance_dataset.bags == bag_id).nonzero(as_tuple=True)[0][0]
        bag_label = instance_dataset.labels[bag_index]
        bag_labels[i] = bag_label

    return InstanceDatasetWithBagLabel(
        data=instance_dataset.data,
        ids=instance_dataset.ids,
        labels=instance_dataset.labels,
        instance_labels=instance_dataset.instance_labels,
        bag_labels=bag_labels,
        study_ids=instance_dataset.study_ids
    )


def create_datasets_from_adata(
    adata,
    train_indices: List[int],
    test_indices: List[int],
    device: torch.device,
    sample_col: str = "sample_id_numeric",
    label_col: str = "disease_numeric",
    study_col: Optional[str] = "study_id_numeric"
) -> Tuple[MilDataset, InstanceDataset, MilDataset, InstanceDataset]:
    """
    AnnData에서 train/test 데이터셋을 생성합니다.

    Args:
        adata: AnnData object
        train_indices: Sample indices for training
        test_indices: Sample indices for testing
        device: Device for tensors
        sample_col: Column name for sample IDs
        label_col: Column name for labels
        study_col: Column name for study IDs (optional)

    Returns:
        train_mil, train_instance, test_mil, test_instance datasets
    """
    import numpy as np

    def _create_split_data(indices):
        # Get cells belonging to these samples
        mask = adata.obs[sample_col].isin(indices)
        split_adata = adata[mask]

        # Extract data
        if hasattr(split_adata.X, 'toarray'):
            data = torch.tensor(split_adata.X.toarray(), dtype=torch.float32, device=device)
        else:
            data = torch.tensor(np.array(split_adata.X), dtype=torch.float32, device=device)

        # Sample IDs and labels
        sample_ids = torch.tensor(
            split_adata.obs[sample_col].values, dtype=torch.long, device=device
        )

        # Get unique samples and their labels
        unique_samples = split_adata.obs.groupby(sample_col)[label_col].first()
        labels = torch.tensor(
            [unique_samples[s] for s in sorted(unique_samples.index)],
            dtype=torch.long, device=device
        )

        # Instance labels (same as sample labels for each cell)
        instance_labels = torch.tensor(
            split_adata.obs[label_col].values, dtype=torch.long, device=device
        )

        # Study IDs (optional)
        study_ids = None
        if study_col and study_col in split_adata.obs.columns:
            study_ids = torch.tensor(
                split_adata.obs[study_col].values, dtype=torch.long, device=device
            )

        return data, sample_ids, labels, instance_labels, study_ids

    # Create train datasets
    train_data, train_ids, train_labels, train_instance_labels, train_study_ids = \
        _create_split_data(train_indices)
    train_mil = MilDataset(
        train_data, train_ids, train_labels, train_instance_labels, train_study_ids
    )
    train_instance = InstanceDataset(
        train_data, train_ids, train_labels, train_instance_labels, train_study_ids
    )

    # Create test datasets
    test_data, test_ids, test_labels, test_instance_labels, test_study_ids = \
        _create_split_data(test_indices)
    test_mil = MilDataset(
        test_data, test_ids, test_labels, test_instance_labels, test_study_ids
    )
    test_instance = InstanceDataset(
        test_data, test_ids, test_labels, test_instance_labels, test_study_ids
    )

    return train_mil, train_instance, test_mil, test_instance
