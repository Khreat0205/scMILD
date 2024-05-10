
import torch
from torch.utils.data import Dataset
import pandas
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import modin.pandas as pd
import numpy as np

class MilDataset(Dataset):
    '''
    Subclass of torch.utils.data.Dataset. 

    Args:
        data:
        ids:
        labels:
        instance_labels:
        normalize:
    '''
    def __init__(self, data, ids, labels, instance_labels):
        self.data = data
        self.labels = labels
        self.ids = ids
        self.instance_labels = instance_labels  # Add instance labels
         
        # Modify shape of bagids if only 1d tensor
        if (len(ids.shape) == 1):
            ids = ids.unsqueeze(0)
            #ids.resize_(1, len(ids))
    
        self.bags = torch.unique(self.ids[0])

    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, index):
        # data = self.data[self.ids[0] == self.bags[index]]
        bag_id = self.bags[index]
        bagids = self.ids[:, self.ids[0] == bag_id]
        labels = self.labels[index]
        indices = torch.where(self.ids[0] == bag_id)[0]
        data = self.data[indices]
        bag_instance_labels = self.instance_labels[indices]
        
        return data, bagids, labels, bag_instance_labels
    
    def n_features(self):
        return self.data.size(1)


def collate(batch):
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
    out_labels = torch.stack(batch_labels)
    
    return out_data, out_bagids, out_labels


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

def load_dataset_and_preprocessors(base_path, exp):
    train_dataset = torch.load(f"{base_path}/train_dataset_exp{exp}_HVG_count.pt")
    val_dataset = torch.load(f"{base_path}/val_dataset_exp{exp}_HVG_count.pt")
    test_dataset = torch.load(f"{base_path}/test_dataset_exp{exp}_HVG_count.pt")
    
    with open(f"{base_path}/label_encoder_exp{exp}_HVG_count.pkl", 'rb') as f:
        label_encoder = pickle.load(f)
    # with open(f"{base_path}/scaler_exp{exp}_HVG_count.pkl", 'rb') as f:
        # scaler = pickle.load(f)

    return train_dataset, val_dataset, test_dataset, label_encoder # , scaler



class InstanceDataset(Dataset):
    '''
    인스턴스 단위로 데이터를 반환하는 Dataset 클래스.
    MilDataset과 유사하지만, 각 인스턴스에 대한 데이터와 레이블을 반환합니다.

    Args:
        data (Tensor): 특성 데이터
        ids (Tensor): 각 인스턴스에 대응하는 백의 ID
        labels (Tensor): 각 백에 대한 레이블
        instance_labels (Tensor): 각 인스턴스에 대한 레이블
        normalize (bool): 데이터 정규화 여부
    '''
    def __init__(self, data, ids, labels, instance_labels, normalize=False):
        self.data = data
        self.labels = labels
        self.ids = ids
        self.mil_ids = ids.clone()
        self.instance_labels = instance_labels
        
        # print(f"ids {ids.shape}")
        # print(f"self ids {self.ids.shape}")
        # print(f"self_mil ids {self.mil_ids.shape}")
        # Modify shape of bagids if only 1d tensor
        
        if (len(self.mil_ids.shape) == 1):
            self.mil_ids.resize_(1, len(self.mil_ids))
        # print("after")
        # print(f"ids {ids.shape}")
        # print(f"self ids {self.ids.shape}")
        # print(f"self_mil ids {self.mil_ids.shape}")
        self.bags = torch.unique(self.mil_ids[0])
        
        if normalize:
            std = self.data.std(dim=0)
            mean = self.data.mean(dim=0)
            self.data = (self.data - mean)/std
    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, index):
        # 각 인스턴스에 대한 데이터와 레이블을 반환
        data = self.data[index]
        bag_id = self.ids[index] 
        # indices = torch.where(self.ids[0] == bag_id)
        #label = self.labels[index]  # 해당 인스턴스의 백 레이블
        instance_label = self.instance_labels[index]
        
        return data, bag_id, instance_label
    

class InstanceDataset2(Dataset):
    '''
    인스턴스 단위로 데이터를 반환하는 Dataset 클래스.
    MilDataset과 유사하지만, 각 인스턴스에 대한 데이터와 레이블을 반환합니다.

    Args:
        data (Tensor): 특성 데이터
        ids (Tensor): 각 인스턴스에 대응하는 백의 ID
        labels (Tensor): 각 백에 대한 레이블
        instance_labels (Tensor): 각 인스턴스에 대한 레이블
    '''
    def __init__(self, data, ids, labels, instance_labels, bag_labels):
        self.data = data
        self.labels = labels
        self.ids = ids
        self.mil_ids = ids.clone()
        self.instance_labels = instance_labels
        self.bag_labels = bag_labels
        if (len(self.mil_ids.shape) == 1):
            self.mil_ids.resize_(1, len(self.mil_ids))
        self.bags = torch.unique(self.mil_ids[0])
    def __len__(self):
        return self.data.size(0)
    def __getitem__(self, index):
        # 각 인스턴스에 대한 데이터와 레이블을 반환
        data = self.data[index]
        bag_id = self.ids[index] 
        
        instance_label = self.instance_labels[index]
        bag_label = self.bag_labels[index]
        return data, bag_id, instance_label, bag_label
    

def update_instance_labels_with_bag_labels(instance_dataset):
    """
    Updates the instance labels in the InstanceDataset with the corresponding bag labels.
    
    Args:
    instance_dataset (InstanceDataset): The dataset whose instance labels are to be updated.
    
    Note: This function modifies the instance_dataset in-place.
    """
    combined_labels = torch.empty(len(instance_dataset), dtype=torch.long, device=device)
    for i in range(len(instance_dataset)):
        _, bag_id, instance_label = instance_dataset[i]
        bag_index = (instance_dataset.bags == bag_id).nonzero(as_tuple=True)[0][0]

        bag_label = instance_dataset.labels[bag_index]

        combined_label = instance_label * 0 + bag_label
        combined_labels[i] = combined_label 
        
    return InstanceDataset2(instance_dataset.data, instance_dataset.ids, instance_dataset.labels, instance_dataset.instance_labels, combined_labels)
