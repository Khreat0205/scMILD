import os 
import torch
import pickle
import numpy as np
import random
import json
import argparse
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import issparse
from src.dataset import MilDataset, InstanceDataset, collate, update_instance_labels_with_bag_labels
from src.model import AENB, VQ_AENB, AttentionModule, TeacherBranch, StudentBranch
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler


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



def load_ae_hyperparameters(ae_dir):
    hyperparameter_file_path = os.path.join(ae_dir, f'hyperparameters_pretraining_autoencoder.json')
    with open(hyperparameter_file_path, 'r') as file:
        loaded_args_dict = json.load(file)
        
    loaded_args = argparse.Namespace(**loaded_args_dict)
    
    # Return basic parameters plus model type info if available
    model_type = getattr(loaded_args, 'model_type', 'AENB')  # Default to AENB for backward compatibility
    vq_num_codes = getattr(loaded_args, 'vq_num_codes', 256)
    vq_commitment_weight = getattr(loaded_args, 'vq_commitment_weight', 0.25)
    
    return loaded_args.ae_latent_dim, loaded_args.ae_hidden_layers, model_type, vq_num_codes, vq_commitment_weight

def load_and_process_datasets(data_dir, exp, device, student_batch_size):
    train_dataset, val_dataset, test_dataset, _ = load_dataset_and_preprocessors(data_dir, exp, device=torch.device('cpu'))
    instance_train_dataset = update_instance_labels_with_bag_labels(train_dataset, device=torch.device('cpu'))
    
    set_random_seed(exp)
    
    #### add 06 04 _ sampler
    # Calculate weights for each combined label
    label_counts = instance_train_dataset.bag_labels.bincount()
    label_weights = 1.0 / label_counts
    instance_weights = label_weights[instance_train_dataset.bag_labels]
    # Define your DataLoader
    num_workers = 1
    pin_memory = True
    prefetch_factor_student=4
    # Create a WeightedRandomSampler
    sampler = WeightedRandomSampler(instance_weights, len(instance_train_dataset))
    
    instance_train_dl = DataLoader(instance_train_dataset, batch_size=student_batch_size, sampler=sampler, drop_last=False, pin_memory=pin_memory, num_workers=num_workers, prefetch_factor=prefetch_factor_student) 
    
    bag_train = MilDataset(train_dataset.data.to(device), train_dataset.ids.unsqueeze(0).to(device), train_dataset.labels.to(device), train_dataset.instance_labels.to(device))
    bag_val = MilDataset(val_dataset.data.to(device), val_dataset.ids.unsqueeze(0).to(device), val_dataset.labels.to(device), val_dataset.instance_labels.to(device))
    bag_test = MilDataset(test_dataset.data.to(device), test_dataset.ids.unsqueeze(0).to(device), test_dataset.labels.to(device), test_dataset.instance_labels.to(device))
    
    return instance_train_dl, bag_train, bag_val, bag_test

def load_dataloaders(bag_train, bag_val, bag_test):
    # bag_train_dl = DataLoader(bag_train,batch_size = 14, shuffle=False, drop_last=False,collate_fn=collate)
    bag_train_dl = DataLoader(bag_train,batch_size = 28, shuffle=False, drop_last=False,collate_fn=collate)
    bag_val_dl = DataLoader(bag_val,batch_size = 15, shuffle=False, drop_last=False,collate_fn=collate)
    bag_test_dl = DataLoader(bag_test,batch_size = 15, shuffle=False, drop_last=False,collate_fn=collate)
    return bag_train_dl, bag_val_dl, bag_test_dl


def load_model_and_optimizer(data_dim, ae_latent_dim, ae_hidden_layers, device, ae_dir, exp, mil_latent_dim,
                            teacher_learning_rate, student_learning_rate, encoder_learning_rate,
                            model_type='AENB', vq_num_codes=256, vq_commitment_weight=0.25):
    """
    Load pretrained autoencoder and create MIL models and optimizers.
    Automatically detects model type from saved file or uses provided model_type.
    """
    
    # Try to load VQ-AENB first if model_type suggests it
    if model_type == 'VQ-AENB':
        vq_model_path = f"{ae_dir}/vq_aenb_{exp}.pth"
        if os.path.exists(vq_model_path):
            ae = VQ_AENB(input_dim=data_dim, latent_dim=ae_latent_dim,
                        device=device, hidden_layers=ae_hidden_layers,
                        num_codes=vq_num_codes, commitment_weight=vq_commitment_weight,
                        activation_function=nn.Sigmoid).to(device)
            ae.load_state_dict(torch.load(vq_model_path, map_location=device))
            print(f"Loaded VQ-AENB model from {vq_model_path}")
            model_encoder = ae  # Use the entire VQ-AENB model
        else:
            # Fallback to AENB if VQ-AENB file doesn't exist
            print(f"VQ-AENB model file not found at {vq_model_path}, falling back to AENB")
            model_type = 'AENB'
    
    # Load AENB model (either as primary choice or fallback)
    if model_type == 'AENB':
        aenb_model_path = f"{ae_dir}/aenb_{exp}.pth"
        if os.path.exists(aenb_model_path):
            ae = AENB(input_dim=data_dim, latent_dim=ae_latent_dim,
                     device=device, hidden_layers=ae_hidden_layers,
                     activation_function=nn.Sigmoid).to(device)
            ae.load_state_dict(torch.load(aenb_model_path, map_location=device))
            print(f"Loaded AENB model from {aenb_model_path}")
            model_encoder = ae.features  # Use only the encoder part
        else:
            raise FileNotFoundError(f"No model file found at {aenb_model_path}")
    
    encoder_dim = ae_latent_dim
    attention_module = AttentionModule(L=encoder_dim, D=encoder_dim, K=1).to(device)
    model_teacher = TeacherBranch(input_dims=encoder_dim, latent_dims=mil_latent_dim,
                                 attention_module=attention_module, num_classes=2, activation_function=nn.Tanh)

    model_student = StudentBranch(input_dims=encoder_dim, latent_dims=mil_latent_dim,
                                 num_classes=2, activation_function=nn.Tanh)
    
    model_teacher.to(device)
    model_student.to(device)
    
    optimizer_teacher = torch.optim.Adam(model_teacher.parameters(), lr=teacher_learning_rate)
    optimizer_student = torch.optim.Adam(model_student.parameters(), lr=student_learning_rate)
    
    # For VQ-AENB, we optimize the entire model; for AENB, just the encoder features
    if model_type == 'VQ-AENB':
        # Only optimize the encoder part of VQ-AENB, not the quantizer or decoder
        optimizer_encoder = torch.optim.Adam(ae.encoder.parameters(), lr=encoder_learning_rate)
    else:
        optimizer_encoder = torch.optim.Adam(model_encoder.parameters(), lr=encoder_learning_rate)
    
    return model_teacher, model_student, model_encoder, optimizer_teacher, optimizer_student, optimizer_encoder