import argparse
import scanpy as sc
import torch
import modin.pandas as pd
from src.utils import load_and_save_datasets_adata

def preprocess_adata(data_dir, data_dim, obs_name_sample_label, obs_name_sample_id, obs_name_cell_type, 
                    sample_label_negative, sample_label_positive, device_num, n_exp):
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
    exps = range(1, n_exp + 1)
    load_and_save_datasets_adata(data_dir, exps, device, adata)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data and save split datasets.')
    parser.add_argument('--data_dir', type=str, default="data/MyData", help='Directory containing adata.h5ad')
    parser.add_argument('--data_dim', type=int, default=2000, help='Number of top genes to select')
    parser.add_argument('--obs_name_sample_label', type=str, default='Health', help='Obs column name for sample labels')
    parser.add_argument('--obs_name_sample_id', type=str, default='Subject', help='Obs column name for sample IDs')
    parser.add_argument('--obs_name_cell_type', type=str, default='Cluster', help='Obs column name for cell types')
    parser.add_argument('--sample_label_negative', type=str, default='Healthy', help='Negative sample label')
    parser.add_argument('--sample_label_positive', type=str, default='Inflamed', help='Positive sample label')
    parser.add_argument('--device_num', type=int, default=6, help='CUDA device number')
    parser.add_argument('--n_exp', type=int, default=8, help='Number of experiments')

    args = parser.parse_args()

    preprocess_adata(args)