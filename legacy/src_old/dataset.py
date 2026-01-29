
import torch
from torch.utils.data import Dataset
import pandas
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import modin.pandas as pd
import numpy as np

# class MilDataset(Dataset):
#     '''
#     Subclass of torch.utils.data.Dataset. 

#     Args:
#         data:
#         ids:
#         labels:
#         instance_labels:
#         normalize:
#     '''
#     def __init__(self, data, ids, labels, instance_labels):
#         self.data = data
#         self.labels = labels
#         self.ids = ids
#         self.instance_labels = instance_labels  # Add instance labels
         
#         # Modify shape of bagids if only 1d tensor
#         if (len(ids.shape) == 1):
#             ids = ids.unsqueeze(0)
#             #ids.resize_(1, len(ids))
    
#         self.bags = torch.unique(self.ids[0])

#     def __len__(self):
#         return len(self.bags)
    
#     def __getitem__(self, index):
#         # data = self.data[self.ids[0] == self.bags[index]]
#         bag_id = self.bags[index]
#         bagids = self.ids[:, self.ids[0] == bag_id]
#         labels = self.labels[index]
#         indices = torch.where(self.ids[0] == bag_id)[0]
#         data = self.data[indices]
#         bag_instance_labels = self.instance_labels[indices]
        
#         return data, bagids, labels, bag_instance_labels
    
#     def n_features(self):
#         return self.data.size(1)


# def collate(batch):
#     '''

#     '''
#     batch_data = []
#     batch_bagids = []
#     batch_labels = []
    
#     for sample in batch:
#         batch_data.append(sample[0])
#         batch_bagids.append(sample[1])
#         batch_labels.append(sample[2])
    
#     out_data = torch.cat(batch_data, dim = 0)
#     out_bagids = torch.cat(batch_bagids, dim = 1)
#     out_labels = torch.stack(batch_labels)
    
#     return out_data, out_bagids, out_labels


def collate_np(batch):
    '''

    '''
    batch_data = []
    batch_bagids = []
    batch_labels = []
    
    for sample in batch:
        batch_data.append(sample[0])
        batch_bagids.append(sample[1])
        batch_labels.append(sample[2])
    
    out_data = torch.cat(batch_data, dim = 0)
    out_bagids = torch.cat(batch_bagids, dim = 1)
    out_labels = torch.tensor(batch_labels)
    
    return out_data, out_bagids, out_labels




############################################################
############################################################

# class InstanceDataset(Dataset):
#     '''
    
#     instance-level dataset class.
#     similar to MilDataset. return labels and data of each instance

#     Args:
#         data (Tensor): feature data
#         ids (Tensor): Bag id of each instance
#         labels (Tensor): bag labels
#         instance_labels (Tensor)
#     '''
#     def __init__(self, data, ids, labels, instance_labels, normalize=False):
#         self.data = data
#         self.labels = labels
#         self.ids = ids
#         self.mil_ids = ids.clone()
#         self.instance_labels = instance_labels
        
#         # print(f"ids {ids.shape}")
#         # print(f"self ids {self.ids.shape}")
#         # print(f"self_mil ids {self.mil_ids.shape}")
#         # Modify shape of bagids if only 1d tensor
        
#         if (len(self.mil_ids.shape) == 1):
#             self.mil_ids.resize_(1, len(self.mil_ids))
#         # print("after")
#         # print(f"ids {ids.shape}")
#         # print(f"self ids {self.ids.shape}")
#         # print(f"self_mil ids {self.mil_ids.shape}")
#         self.bags = torch.unique(self.mil_ids[0])
        
#         if normalize:
#             std = self.data.std(dim=0)
#             mean = self.data.mean(dim=0)
#             self.data = (self.data - mean)/std
#     def __len__(self):
#         return self.data.size(0)
    
#     def __getitem__(self, index):
#         # 각 인스턴스에 대한 데이터와 레이블을 반환
#         data = self.data[index]
#         bag_id = self.ids[index] 
#         # indices = torch.where(self.ids[0] == bag_id)
#         #label = self.labels[index]  # 해당 인스턴스의 백 레이블
#         instance_label = self.instance_labels[index]
        
#         return data, bag_id, instance_label
    

# class InstanceDataset2(Dataset):
#     '''
#     Instance-level Dataset class.
#     similar to MilDataset. return labels and data of each instance

#     Args:
#         data (Tensor): 특성 데이터
#         ids (Tensor): 각 인스턴스에 대응하는 백의 ID
#         labels (Tensor): 각 백에 대한 레이블
#         instance_labels (Tensor): 각 인스턴스에 대한 레이블
#     '''
#     def __init__(self, data, ids, labels, instance_labels, bag_labels):
#         self.data = data
#         self.labels = labels
#         self.ids = ids
#         self.mil_ids = ids.clone()
#         self.instance_labels = instance_labels
#         self.bag_labels = bag_labels
#         if (len(self.mil_ids.shape) == 1):
#             self.mil_ids.resize_(1, len(self.mil_ids))
#         self.bags = torch.unique(self.mil_ids[0])
#     def __len__(self):
#         return self.data.size(0)
#     def __getitem__(self, index):
#         # 각 인스턴스에 대한 데이터와 레이블을 반환
#         data = self.data[index]
#         bag_id = self.ids[index] 
        
#         instance_label = self.instance_labels[index]
#         bag_label = self.bag_labels[index]
#         return data, bag_id, instance_label, bag_label
    

# def update_instance_labels_with_bag_labels(instance_dataset, device):
#     """
#     Updates the instance labels in the InstanceDataset with the corresponding bag labels.
    
#     Args:
#     instance_dataset (InstanceDataset): The dataset whose instance labels are to be updated.
    
#     Note: This function modifies the instance_dataset in-place.
#     """
#     combined_labels = torch.empty(len(instance_dataset), dtype=torch.long, device=device)
#     for i in range(len(instance_dataset)):
#         _, bag_id, instance_label = instance_dataset[i]
#         bag_index = (instance_dataset.bags == bag_id).nonzero(as_tuple=True)[0][0]

#         bag_label = instance_dataset.labels[bag_index]

#         combined_label = instance_label * 0 + bag_label
#         combined_labels[i] = combined_label 
        
#     return InstanceDataset2(instance_dataset.data, instance_dataset.ids, instance_dataset.labels, instance_dataset.instance_labels, combined_labels)
################ Conditional


class MilDataset(Dataset):
    '''
    Subclass of torch.utils.data.Dataset. 
    Args:
        data:
        ids:
        labels:
        instance_labels:
        study_ids: [ADDED] Tensor of study IDs per instance
    '''
    def __init__(self, data, ids, labels, instance_labels, study_ids=None): # [MODIFIED]
        self.data = data
        self.labels = labels
        self.ids = ids
        self.instance_labels = instance_labels
        self.study_ids = study_ids # [ADDED]
        
        # Modify shape of bagids if only 1d tensor
        if (len(ids.shape) == 1):
            ids = ids.unsqueeze(0)
            
        self.bags = torch.unique(self.ids[0])

    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, index):
        bag_id = self.bags[index]
        bagids = self.ids[:, self.ids[0] == bag_id]
        labels = self.labels[index]
        indices = torch.where(self.ids[0] == bag_id)[0]
        data = self.data[indices]
        bag_instance_labels = self.instance_labels[indices]
        
        # [ADDED] Bag에 속한 instance들의 study_id 추출
        if self.study_ids is not None:
            bag_study_ids = self.study_ids[indices]
            return data, bagids, labels, bag_instance_labels, bag_study_ids
        
        return data, bagids, labels, bag_instance_labels
    
    def n_features(self):
        return self.data.size(1)


def collate(batch):
    '''
    Modified to handle optional study_ids
    '''
    batch_data = []
    batch_bagids = []
    batch_labels = []
    batch_study_ids = [] # [ADDED]
    
    # Check if batch contains study_ids (length 5)
    has_study_ids = len(batch[0]) == 5
    
    for sample in batch:
        batch_data.append(sample[0])
        batch_bagids.append(sample[1])
        batch_labels.append(sample[2])
        if has_study_ids:
            batch_study_ids.append(sample[4]) # [ADDED]
    
    out_data = torch.cat(batch_data, dim=0)
    out_bagids = torch.cat(batch_bagids, dim=1)
    out_labels = torch.stack(batch_labels)
    
    if has_study_ids:
        out_study_ids = torch.cat(batch_study_ids, dim=0) # [ADDED]
        return out_data, out_bagids, out_labels, out_study_ids # Return 4 items if study_ids exist
    
    return out_data, out_bagids, out_labels


class InstanceDataset(Dataset):
    '''
    Args:
        data (Tensor): feature data
        ids (Tensor): Bag id of each instance
        labels (Tensor): bag labels
        instance_labels (Tensor)
        normalize (bool)
        study_ids (Tensor): [ADDED]
    '''
    def __init__(self, data, ids, labels, instance_labels, normalize=False, study_ids=None): # [MODIFIED]
        self.data = data
        self.labels = labels
        self.ids = ids
        self.mil_ids = ids.clone()
        self.instance_labels = instance_labels
        self.study_ids = study_ids # [ADDED]
        
        if (len(self.mil_ids.shape) == 1):
            self.mil_ids.resize_(1, len(self.mil_ids))
        self.bags = torch.unique(self.mil_ids[0])
        
        if normalize:
            std = self.data.std(dim=0)
            mean = self.data.mean(dim=0)
            self.data = (self.data - mean)/std

    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, index):
        data = self.data[index]
        bag_id = self.ids[index] 
        instance_label = self.instance_labels[index]
        
        # [ADDED]
        if self.study_ids is not None:
            study_id = self.study_ids[index]
            return data, bag_id, instance_label, study_id
            
        return data, bag_id, instance_label

def update_instance_labels_with_bag_labels(instance_dataset, device):
    """
    Updates the instance labels in the InstanceDataset with the corresponding bag labels.
    
    Args:
        instance_dataset (InstanceDataset): The dataset whose instance labels are to be updated.
    
    Note: This function modifies the instance_dataset in-place.
    """
    combined_labels = torch.empty(len(instance_dataset), dtype=torch.long, device=device)
    
    for i in range(len(instance_dataset)):
        item = instance_dataset[i]
        
        # study_ids 유무에 따라 반환값 개수 다름
        if len(item) == 4:
            _, bag_id, instance_label, _ = item  # (data, bag_id, instance_label, study_id)
        else:
            _, bag_id, instance_label = item  # (data, bag_id, instance_label)
        
        bag_index = (instance_dataset.bags == bag_id).nonzero(as_tuple=True)[0][0]
        bag_label = instance_dataset.labels[bag_index]
        combined_label = instance_label * 0 + bag_label
        combined_labels[i] = combined_label
    
    # study_ids 보존하여 새 dataset 생성
    new_dataset = InstanceDataset2(
        instance_dataset.data, 
        instance_dataset.ids, 
        instance_dataset.labels, 
        instance_dataset.instance_labels, 
        combined_labels
    )
    
    # study_ids 복사
    if hasattr(instance_dataset, 'study_ids') and instance_dataset.study_ids is not None:
        new_dataset.study_ids = instance_dataset.study_ids
    
    return new_dataset


class InstanceDataset2(Dataset):
    '''
    Instance-level Dataset class with bag_labels.
    '''
    def __init__(self, data, ids, labels, instance_labels, bag_labels, study_ids=None):
        self.data = data
        self.labels = labels
        self.ids = ids
        self.mil_ids = ids.clone()
        self.instance_labels = instance_labels
        self.bag_labels = bag_labels
        self.study_ids = study_ids  # [ADDED]
        
        if (len(self.mil_ids.shape) == 1):
            self.mil_ids.resize_(1, len(self.mil_ids))
        self.bags = torch.unique(self.mil_ids[0])
        
    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, index):
        data = self.data[index]
        bag_id = self.ids[index]
        instance_label = self.instance_labels[index]
        bag_label = self.bag_labels[index]
        
        # [ADDED] study_ids 반환
        if self.study_ids is not None:
            study_id = self.study_ids[index]
            return data, bag_id, instance_label, bag_label, study_id
        
        return data, bag_id, instance_label, bag_label