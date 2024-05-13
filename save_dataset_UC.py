import sys
import os
import scanpy as sc
from scipy import sparse
# import ray
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import modin.pandas as pd
from sklearn.model_selection import train_test_split
from utils import *
from dataset import *
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.sparse import issparse


data_dir = "data/UC/"


adata = sc.read_mtx(f"{data_dir}/Fib.matrix.mtx")
barcodes = open(os.path.join(data_dir, "Fib.barcodes.tsv")).read().strip().split("\n")
genes = open(os.path.join(data_dir, "Fib.genes.tsv")).read().strip().split("\n")
meta = pd.read_csv(f'{data_dir}/all.meta2.txt',sep="\t").drop(axis=0, index=0).reset_index(drop=True)
meta[['NAME']].isin(barcodes)
meta_subset =meta[meta['NAME'].isin(barcodes)].reset_index(drop=True)

adata = adata.X.transpose()
adata = sc.AnnData(adata,obs = barcodes, var=genes)
meta_subset.set_index('NAME', inplace=True)

adata.obs['Location'] = pd.Categorical(meta_subset['Location'])
adata.obs['Health'] = pd.Categorical(meta_subset['Health'])
adata.obs['Subject'] = meta_subset['Subject'].values
adata.obs['Cluster'] = pd.Categorical(meta_subset['Cluster'])
adata.obs['Sample'] = meta_subset['Sample'].values

print(adata.shape)
sc.pp.filter_genes(adata, min_cells=5)
adata_raw = adata.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
print("Preprocessing Complete!")
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

adata = adata_raw[:, adata.var.highly_variable]
print(adata.shape)

adata.obs.rename(columns={0: 'cell.names'}, inplace=True)
adata.var.rename(columns={0: 'gene.names'}, inplace=True)

# adata.write(f'{data_dir}/Fib.h5ad')

adata_subset = adata[adata.obs['Health'].isin(('Healthy','Inflamed'))]

#print(adata_subset.obs['Subject'].value_counts())
# print(adata_subset.obs)

mapping = {'Healthy': 0, 'Inflamed': 1}
adata_subset.obs['disease_numeric'] = adata_subset.obs['Health'].map(mapping)
adata_subset.obs['sample_id_numeric'], _ = pd.factorize(adata_subset.obs['Subject'])



device_num = 4
device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')

print("INFO: Using device: {}".format(device))

def load_mil_dataset_from_adata(adata, label_encoder=None, is_train=True, device='cpu'):
    instance_labels = adata.obs['Cluster'].values
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
            adata = adata[adata.obs['Cluster'].isin(valid_labels)]
            instance_labels = label_encoder.transform(adata.obs['Cluster'].values)
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
    for exp in exps:
        label_encoder = LabelEncoder()
        
        print(f"Experiment {exp} Start")
        sample_labels = adata.obs[['disease_numeric', 'sample_id_numeric']].drop_duplicates()
        split_ratio = [0.5, 0.25, 0.25]
        
        train_val_set, test_set = train_test_split(sample_labels, test_size=split_ratio[2], random_state=exp, stratify=sample_labels['disease_numeric'])
        train_set, val_set = train_test_split(train_val_set, test_size=split_ratio[1] / (1 - split_ratio[2]), random_state=exp,stratify=train_val_set['disease_numeric'])
        
        train_data = adata[adata.obs['sample_id_numeric'].isin(train_set['sample_id_numeric'])]
        val_data = adata[adata.obs['sample_id_numeric'].isin(val_set['sample_id_numeric'])]
        test_data = adata[adata.obs['sample_id_numeric'].isin(test_set['sample_id_numeric'])]

        train_dataset, label_encoder = load_mil_dataset_from_adata(adata=train_data, device=device, is_train=True, label_encoder=label_encoder)
        save_preprocessors(base_path, label_encoder, exp)
        torch.save(train_dataset, f"{base_path}/train_dataset_exp{exp}.pt")

        val_dataset, _ = load_mil_dataset_from_adata(adata=val_data, device=device, is_train=False, label_encoder=label_encoder)
        torch.save(val_dataset, f"{base_path}/val_dataset_exp{exp}.pt")
        
        test_dataset, _ = load_mil_dataset_from_adata(adata=test_data, device=device, is_train=False, label_encoder=label_encoder)
        torch.save(test_dataset, f"{base_path}/test_dataset_exp{exp}.pt")
        
        print(f'Experiment {exp} dataset saved')
        
        
        del train_dataset, label_encoder, val_dataset, test_dataset, train_data, val_data, test_data


exps = range(1, 9) # 1부터 8까지의 exp
load_and_save_datasets_adata(data_dir, exps, device, adata_subset)
