import sys
import os
import scanpy as sc
from scipy import sparse
import tqdm
import ray
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import modin.pandas as pd
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.dataset import InstanceDataset
import pickle
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import issparse


data_dir = "NS/"

dat = sparse.load_npz(os.path.join(data_dir, "RawCounts.npz"))
genes = open(os.path.join(data_dir, "genes.txt")).read().strip().split("\n")
barcodes = open(os.path.join(data_dir, "barcodes.txt")).read().strip().split("\n")
meta = pd.read_csv(os.path.join(data_dir, "20210701_NasalSwab_MetaData.txt"), sep="\t").drop(axis=0,index=0).reset_index(drop=True)

cell_types = pd.read_csv(os.path.join(data_dir, "20210220_NasalSwab_UMAP.txt"), sep="\t").drop(axis=0,index=0).reset_index(drop=True)["Category"]
ct_id = sorted(set(cell_types))
mapping_ct = {c:idx for idx, c in enumerate(ct_id)}

X = []
y = []
ct = []

adata = sc.AnnData(dat.astype(np.float32), obs=barcodes, var=genes)
print(adata.shape)
barcodes = adata.obs[0].tolist()

meta_subset = meta[meta['NAME'].isin(barcodes)]
meta_subset.set_index('NAME', inplace=True)
meta_subset = meta_subset.reindex(adata.obs[0])

adata.obs['ind_cov'] = meta_subset['donor_id'].values
adata.obs['ct_cov'] = meta_subset['Coarse_Cell_Annotations'].values
adata.obs['disease_cov'] = meta_subset['disease__ontology_label'].values
adata = adata[adata.obs['disease_cov'].isin(['normal', 'COVID-19'])]


sc.pp.filter_genes(adata, min_cells=5)
adata_raw = adata.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print("Preprocessing Complete!")
print(adata.shape)

sc.pp.highly_variable_genes(adata, n_top_genes=3000)
adata = adata_raw[:, adata.var.highly_variable]
print(adata.shape)

mapping = {'normal': 0, 'COVID-19': 1}
adata.obs['disease_numeric'] = adata.obs['disease_cov'].map(mapping)
adata.obs['sample_id_numeric'], _ = pd.factorize(adata.obs['ind_cov'])

device_num = 6
device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')

print("INFO: Using device: {}".format(device))



def load_mil_dataset_from_adata(adata, label_encoder=None, is_train=True, device='cpu'):
    instance_labels = adata.obs['ct_cov'].values
    features = adata.X

    if is_train:
        if label_encoder is None:
            label_encoder = LabelEncoder()

        instance_labels = label_encoder.fit_transform(instance_labels)
        print("label encoder fit complete")
        features = features.toarray() if issparse(features) else features
        print("scaler fit complete")
    else:
        if label_encoder is None:
            raise ValueError("label_encoder must be provided for test set")
        try: 
            instance_labels = label_encoder.transform(instance_labels)
        except ValueError:
            valid_labels = [label for label in instance_labels if label in label_encoder.classes_]
            adata = adata[adata.obs['ct_cov'].isin(valid_labels)]
            instance_labels = label_encoder.transform(adata.obs['ct_cov'].values)
            features = adata.X
        print("label encoder transform complete")    
        features = features.toarray() if issparse(features) else features
        print("scaler transform complete")

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
        print("고유한 샘플 레이블 추출 및 분할 완료")
        
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


exps = range(1, 9) # 1부터 8까지의 exp
load_and_save_datasets_adata(data_dir, exps, device, adata)
