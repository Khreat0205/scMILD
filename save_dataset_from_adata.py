
import scanpy as sc
import torch
import modin.pandas as pd
from src.utils import load_and_save_datasets_adata

### Set the dataset name, directory, and other parameters
dataset="MyData"
data_dir = f"data/{dataset}"
data_dim = 2000
obs_name_sample_label = 'Health'
obs_name_sample_id = 'Subject'
obs_name_cell_type = 'Cluster'
sample_label_negative = 'Healthy'
sample_label_positive = 'Inflamed'
device_num = 6
n_exp = 8 

### preprocsessing adata
adata = sc.read_h5ad(f'{data_dir}/adata.h5ad')
adata_raw = adata.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
print(f'Normalization Complete: {adata.shape}')
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=data_dim)
adata = adata_raw[:, adata.var.highly_variable]
print(f'Highly variable gene selection Complete: {adata.shape}')

adata = adata[adata.obs[obs_name_sample_label].isin((sample_label_negative, sample_label_positive))]
print(f'Subset: {adata.shape}')

### One-hot encoding for sample labels
mapping = {sample_label_negative: 0, sample_label_positive: 1}

### Obs columns: disease_numeric, sample_id_numeric, cell_type
adata.obs['disease_numeric'] = adata.obs[obs_name_sample_label].map(mapping)
adata.obs['sample_id_numeric'], _ = pd.factorize(adata.obs[obs_name_sample_id])
adata.obs['cell_type'] = adata.obs[obs_name_cell_type]

device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
print("INFO: Using device: {}".format(device))

### Save split datasets
exps = range(1, n_exp + 1) # 
load_and_save_datasets_adata(data_dir, exps, device, adata)
