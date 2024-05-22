import torch
import pickle
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import issparse
from src.dataset import InstanceDataset
from sklearn.model_selection import train_test_split

def load_mil_dataset_from_adata(adata, label_encoder=None, is_train=True, device='cpu'):
    instance_labels = adata.obs['cell_type'].values
    features = adata.X

    if is_train:
        if label_encoder is None:
            label_encoder = LabelEncoder()

        instance_labels = label_encoder.fit_transform(instance_labels)
        features = features.toarray() if issparse(features) else features
    else:
        if label_encoder is None:
            raise ValueError("label_encoder must be provided for test set")
        try: 
            instance_labels = label_encoder.transform(instance_labels)
        except ValueError:
            valid_labels = [label for label in instance_labels if label in label_encoder.classes_]
            adata = adata[adata.obs['cell_type'].isin(valid_labels)]
            instance_labels = label_encoder.transform(adata.obs['ct_cov'].values)
            features = adata.X
        features = features.toarray() if issparse(features) else features

    sample_ids = adata.obs['sample_id_numeric'].values
    labels = adata.obs['disease_numeric'].values

    # PyTorch Tensor
    tensor_data = torch.tensor(features, dtype=torch.float).to(device)
    tensor_ids = torch.tensor(sample_ids, dtype=torch.long).to(device)
    tensor_labels = torch.tensor(labels, dtype=torch.long).to(device)
    tensor_instance_labels = torch.tensor(instance_labels, dtype=torch.long).to(device)

    
    unique_bag_ids = torch.unique(tensor_ids)
    bag_labels = torch.stack([max(tensor_labels[tensor_ids == i]) for i in unique_bag_ids]).long()
    bag_labels = bag_labels.cpu().long()
    
    instance_dataset = InstanceDataset(tensor_data, tensor_ids, bag_labels, tensor_instance_labels)
    return instance_dataset, label_encoder


def save_preprocessors(base_path, label_encoder, exp):
    with open(f"{base_path}/label_encoder_exp{exp}.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)

def load_and_save_datasets_adata(base_path, exps, device, adata):
    '''
    base_path: str
        path to save the datasets
    exps: list
        list of integers for experiment numbers
    device: str
        device to save the datasets
    adata: AnnData
        AnnData object to load the datasets
        
    '''
    for exp in exps:
        label_encoder = LabelEncoder()
        
        print(f"Experiment {exp}")
        sample_labels = adata.obs[['disease_numeric', 'sample_id_numeric']].drop_duplicates()
        split_ratio = [0.5, 0.25, 0.25]
        
        train_val_set, test_set = train_test_split(sample_labels, test_size=split_ratio[2], random_state=exp, stratify=sample_labels['disease_numeric'])
        train_set, val_set = train_test_split(train_val_set, test_size=split_ratio[1] / (1 - split_ratio[2]), random_state=exp,stratify=train_val_set['disease_numeric'])
        print(f"Experiment {exp} Train-Val-Test Split Complete!")
        
        train_data = adata[adata.obs['sample_id_numeric'].isin(train_set['sample_id_numeric'])]
        val_data = adata[adata.obs['sample_id_numeric'].isin(val_set['sample_id_numeric'])]
        test_data = adata[adata.obs['sample_id_numeric'].isin(test_set['sample_id_numeric'])]

        train_dataset, label_encoder = load_mil_dataset_from_adata(adata=train_data, device=device, is_train=True, label_encoder=label_encoder)
        save_preprocessors(base_path, label_encoder, exp)
        torch.save(train_dataset, f"{base_path}/train_dataset_exp{exp}.pt")
        print(f"Experiment {exp} Train Dataset Saved!")
        
        val_dataset, _ = load_mil_dataset_from_adata(adata=val_data, is_train=False, label_encoder=label_encoder)
        torch.save(val_dataset, f"{base_path}/val_dataset_exp{exp}.pt")
        print(f"Experiment {exp} Val Dataset Saved!")
        
        test_dataset, _ = load_mil_dataset_from_adata(adata=test_data, is_train=False, label_encoder=label_encoder)
        torch.save(test_dataset, f"{base_path}/test_dataset_exp{exp}.pt")
        print(f"Experiment {exp} Test Dataset Saved!")
        
        del train_dataset, label_encoder, val_dataset, test_dataset, train_data, val_data, test_data
        

def load_dataset_and_preprocessors(base_path, exp, device):
    train_dataset = torch.load(f"{base_path}/train_dataset_exp{exp}.pt", map_location= device)
    val_dataset = torch.load(f"{base_path}/val_dataset_exp{exp}.pt", map_location= device)
    test_dataset = torch.load(f"{base_path}/test_dataset_exp{exp}.pt",map_location = device)
    
    with open(f"{base_path}/label_encoder_exp{exp}.pkl", 'rb') as f:
        label_encoder = pickle.load(f)

    return train_dataset, val_dataset, test_dataset, label_encoder


def set_random_seed(exp):
    torch.manual_seed(exp)
    torch.cuda.manual_seed(exp)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(exp)
    random.seed(exp)
    torch.cuda.manual_seed_all(exp)
# git test