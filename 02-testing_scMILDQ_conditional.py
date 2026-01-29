# 02-testing_scMILDQ_conditional.py
"""
Conditional scMILD Cross-dataset Evaluation

Usage:
    python 02-testing_scMILDQ_conditional.py --trained SCP1884 --device 0
    python 02-testing_scMILDQ_conditional.py --trained Skin3 --device 1
    python 02-testing_scMILDQ_conditional.py --trained SCP1884 --device 0 --conditional_ae_dir data_conditional/
    # SCP1884로 학습한 모델 평가 (Skin 데이터셋들에서)
    python 02-testing_scMILDQ_conditional.py --trained SCP1884 --device 0 --use_conditional

    # Skin3로 학습한 모델 평가 (Colon 데이터셋들에서)
    python 02-testing_scMILDQ_conditional.py --trained Skin3 --device 4 --use_conditional
    python 02-testing_scMILDQ_conditional.py --trained SCP1884 --device 1 --use_conditional
    python 02-testing_scMILDQ_conditional.py --trained Skin3 --device 1 --use_conditional
    python 02-testing_scMILDQ_conditional.py --trained SCP1884 --device 1 --use_conditional
    
    # Non-conditional 모드 (기존 방식)
    python 02-testing_scMILDQ_conditional.py --trained SCP1884 --device 0
"""

import sys, os
sys.path.append(r"src")

import torch 
torch.set_num_threads(32)
import numpy as np
import scanpy as sc
import pandas as pd
import argparse
import warnings
import gc
import json
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from src.utils import load_mil_dataset_from_adata, dataset_to_dl
from src.dataset import MilDataset, collate


def get_subfolders(root_path):
    """Get model folders that have completed training (exp8 exists)"""
    p = Path(root_path).expanduser()
    if not p.exists():
        return []
    return sorted(str(x.resolve()) for x in p.iterdir() 
                  if x.is_dir() 
                  and not x.name.startswith('.')
                  and (x / 'model_teacher_exp8.pt').exists())


def load_study_mapping(conditional_ae_dir):
    """Load study name -> study_id mapping"""
    mapping_path = f"{conditional_ae_dir}/study_mapping.json"
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            # {id: name} -> {name: id}
            mapping = json.load(f)
            return {v: int(k) for k, v in mapping.items()}
    return None


def get_study_ids_for_adata(adata, study_to_id, study_col='study'):
    """Get study_ids tensor for adata based on mapping"""
    if study_to_id is None:
        return None
    
    study_ids = []
    for study in adata.obs[study_col]:
        study_ids.append(study_to_id.get(study, 0))
    return torch.tensor(study_ids, dtype=torch.long)


def dataset_to_dl_conditional(dataset, device, study_ids=None):
    """Create dataloader with optional study_ids"""
    bag_data = MilDataset(
        dataset[0].data.to(device), 
        dataset[0].ids.unsqueeze(0).to(device), 
        dataset[0].labels.to(device), 
        dataset[0].instance_labels.to(device),
        study_ids=study_ids.to(device) if study_ids is not None else None
    )
    dl = DataLoader(bag_data, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate)
    return dl


def evaluate_model(model_teacher, model_encoder, loader, device, use_conditional=False):
    """Evaluate model on dataloader"""
    model_teacher.eval()
    model_encoder.eval()
    
    bag_label_gt = []
    bag_label_pred = []
    
    with torch.no_grad():
        for batch in loader:
            t_data = batch[0].to(device)
            t_bagids = batch[1].to(device)
            t_labels = batch[2].to(device)
            t_study_ids = batch[3].to(device) if len(batch) == 4 else None
            
            inner_ids = t_bagids[-1]
            unique, inverse, count = torch.unique(inner_ids, return_inverse=True, return_counts=True)
            bag_idx = torch.cat([(inverse == x).nonzero()[0] for x in range(len(unique))]).sort()[1]
            bags = unique[bag_idx]
            
            # Encode features
            if use_conditional and t_study_ids is not None:
                feat = model_encoder(t_data, t_study_ids)[:, :model_teacher.input_dims]
            else:
                feat = model_encoder(t_data)[:, :model_teacher.input_dims]
            
            batch_preds = torch.empty((len(bags), 2), dtype=torch.float, device=device)
            for b, bag in enumerate(bags):
                bag_instances = feat[inner_ids == bag]
                bag_pred = model_teacher(bag_instances, replaceAS=None)
                bag_pred = torch.softmax(bag_pred, dim=0)
                batch_preds[b] = bag_pred
            
            bag_label_gt.append(t_labels)
            bag_label_pred.append(batch_preds)
    
    bag_label_gt = torch.cat(bag_label_gt, dim=0).cpu().numpy()
    bag_label_pred = torch.cat(bag_label_pred, dim=0).cpu().numpy()[:, 1]
    
    auc = roc_auc_score(bag_label_gt, bag_label_pred)
    auprc = average_precision_score(bag_label_gt, bag_label_pred)
    
    return auc, auprc


def build_dataset_dict(adata_whole, trained_on):
    """Build test dataset dictionary based on what model was trained on"""
    
    adata_whole.obs['sample_id_numeric'] = adata_whole.obs['sample_id_numeric'].astype(int)
    adata_whole.obs['disease_numeric'] = adata_whole.obs['disease_numeric'].astype(int)
    
    adata_skin = adata_whole[adata_whole.obs['Organ'] == 'Skin'].copy()
    adata_colon = adata_whole[adata_whole.obs['Organ'] == 'Colon'].copy()
    adata_colon_scp1884 = adata_whole[adata_whole.obs['study'] == 'SCP1884'].copy()
    # adata_colon_pcd = adata_colon[adata_colon.obs['study'] != 'SCP1884'].copy()
    # adata_colon_gse225199 = adata_colon_pcd[adata_colon_pcd.obs['study'] == 'GSE225199'].copy()
    
    # skin2_studies = ['GSE154775', 'GSE175990', 'GSE220116']
    skin3_studies = ['GSE175990', 'GSE220116']
    
    # adata_skin_2 = adata_skin[adata_skin.obs['study'].isin(skin2_studies)].copy()
    adata_skin_3 = adata_skin[adata_skin.obs['study'].isin(skin3_studies)].copy()
    
    # skin2_rest_studies = [s for s in adata_skin.obs['study'].unique() if s not in skin2_studies]
    # skin3_rest_studies = [s for s in adata_skin.obs['study'].unique() if s not in skin3_studies]
    
    # adata_skin_2_rest = adata_skin[adata_skin.obs['study'].isin(skin2_rest_studies)].copy()
    # adata_skin_3_rest = adata_skin[adata_skin.obs['study'].isin(skin3_rest_studies)].copy()
    
    # Return both adata and dataset
    dataset_info = {}
    
    if trained_on == 'SCP1884':
        # Trained on Colon(SCP1884) -> Test on Skin datasets
        test_adatas = {
            # 'Skin': adata_skin,
            # 'Skin2': adata_skin_2,
            'Skin3': adata_skin_3,
            # 'Skin2_rest': adata_skin_2_rest,
            # 'Skin3_rest': adata_skin_3_rest,
            # 'PCD': adata_colon_pcd,
            # 'Colon_GSE225199': adata_colon_gse225199,
        }
        # Add individual skin studies
        for study in adata_skin.obs['study'].cat.categories:
            test_adatas[f'Skin_{study}'] = adata_skin[adata_skin.obs['study'] == study].copy()
            
    elif trained_on == 'Skin3':
        # Trained on Skin3 -> Test on Colon datasets
        test_adatas = {
            'SCP1884': adata_colon_scp1884,
            # 'PCD': adata_colon_pcd,
            # 'Skin2_rest': adata_skin_2_rest,
            # 'Skin3_rest': adata_skin_3_rest,
        }
        # Add individual colon studies
        for study in adata_colon.obs['study'].cat.categories:
            test_adatas[f'Colon_{study}'] = adata_colon[adata_colon.obs['study'] == study].copy()
    else:
        raise ValueError(f"Unknown trained_on: {trained_on}")
    
    return test_adatas


def main():
    parser = argparse.ArgumentParser(description='Conditional scMILD Cross-dataset Evaluation')
    parser.add_argument('--trained', type=str, required=True, choices=['SCP1884', 'Skin3'],
                       help='Dataset the model was trained on')
    parser.add_argument('--device', type=int, default=0, help='CUDA device number')
    parser.add_argument('--n_exp', type=int, default=8, help='Number of experiments')
    parser.add_argument('--res_dir', type=str, default=None, 
                       help='Results directory (default: auto-generated based on trained)')
    parser.add_argument('--conditional_ae_dir', type=str, default='data_conditional/',
                       help='Directory with conditional AE files')
    parser.add_argument('--h5ad_path', type=str, 
                       default='/home/bmi-user/workspace/data/HSvsCD/data/Whole_SCP_PCD_Skin_805k_6k.h5ad',
                       help='Path to h5ad file')
    parser.add_argument('--use_conditional', action='store_true',
                       help='Use conditional encoder (requires study_ids)')
    args = parser.parse_args()
    
    # Setup
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"INFO: Using device: {device}")
    
    trained_on = args.trained
    exps = range(1, args.n_exp + 1)
    
    # Directories
    if args.res_dir:
        res_dir = args.res_dir
    else:
        res_dir = f'/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_conditional_v3_{trained_on}/'
    
    csv_dir = f'./cross_perfs_conditional_v3_{trained_on}/'
    os.makedirs(csv_dir, exist_ok=True)
    
    csv_name_each = f'{csv_dir}/perfs_exps.csv'
    csv_name_grouped = f'{csv_dir}/perfs_grouped.csv'
    
    print(f"="*60)
    print(f"Conditional scMILD Evaluation")
    print(f"="*60)
    print(f"Trained on: {trained_on}")
    print(f"Results dir: {res_dir}")
    print(f"Output dir: {csv_dir}")
    print(f"Use conditional: {args.use_conditional}")
    print(f"="*60)
    
    # Load study mapping for conditional encoder
    study_to_id = None
    if args.use_conditional:
        study_to_id = load_study_mapping(args.conditional_ae_dir)
        if study_to_id:
            print(f"Loaded study mapping: {len(study_to_id)} studies")
        else:
            print("WARNING: Study mapping not found, conditional mode may not work properly")
    
    # Load data
    print("Loading h5ad...")
    adata_whole = sc.read_h5ad(args.h5ad_path)
    
    # Build test datasets
    print("Building test datasets...")
    test_adatas = build_dataset_dict(adata_whole, trained_on)
    print(f"Test datasets: {list(test_adatas.keys())}")
    
    # Preload datasets with study_ids
    dataset_dict = {}
    for key, adata in test_adatas.items():
        dataset = load_mil_dataset_from_adata(adata, device=device)
        study_ids = get_study_ids_for_adata(adata, study_to_id) if args.use_conditional else None
        dataset_dict[key] = (dataset, study_ids, adata)
    
    # Get model paths
    saved_model_paths = get_subfolders(res_dir)
    print(f"Found {len(saved_model_paths)} model folders")
    
    if len(saved_model_paths) == 0:
        print(f"ERROR: No trained models found in {res_dir}")
        return
    
    # Evaluate
    perform_df = pd.DataFrame()
    warnings.filterwarnings("ignore", category=UserWarning)
    
    for saved_model_path in saved_model_paths:
        model_name = os.path.basename(saved_model_path)
        dir_name = os.path.basename(os.path.dirname(saved_model_path))
        print(f"\nProcessing: {model_name}")
        
        for exp in exps:
            model_teacher = torch.load(f'{saved_model_path}/model_teacher_exp{exp}.pt', 
                                       map_location=device, weights_only=False)
            model_encoder = torch.load(f'{saved_model_path}/model_encoder_exp{exp}.pt', 
                                       map_location=device, weights_only=False)
            
            for dataset_key, (dataset, study_ids, adata) in dataset_dict.items():
                loader = dataset_to_dl_conditional(dataset, device, study_ids)
                
                try:
                    auc, auprc = evaluate_model(
                        model_teacher, model_encoder, loader, device,
                        use_conditional=args.use_conditional
                    )
                    
                    perform_df = pd.concat([perform_df, pd.DataFrame({
                        'dir': [dir_name],
                        'model': [model_name],
                        'trained': [trained_on],
                        'tested': [dataset_key],
                        'exp': [exp],
                        'auc': [auc],
                        'auprc': [auprc]
                    })])
                    
                except Exception as e:
                    print(f"  ERROR on {dataset_key} exp{exp}: {e}")
                    continue
        
        print(f"  ✓ Completed all exps")
    
    # Save results
    perform_df_grouped = perform_df.groupby(['model', 'trained', 'tested', 'dir']).agg({
        'auc': ['mean', 'std'], 
        'auprc': ['mean', 'std']
    }).reset_index()
    perform_df_grouped.columns = ['model', 'trained', 'tested', 'dir', 
                                   'auc_mean', 'auc_std', 'auprc_mean', 'auprc_std']
    
    perform_df.to_csv(csv_name_each, index=False)
    perform_df_grouped.to_csv(csv_name_grouped, index=False)
    
    print(f"\n✅ Results saved:")
    print(f"  - {csv_name_each}")
    print(f"  - {csv_name_grouped}")
    
    # Summary
    print(f"\n📊 Summary (mean AUC across models):")
    summary = perform_df.groupby('tested')['auc'].mean().sort_values(ascending=False)
    for tested, auc in summary.items():
        print(f"  {tested}: {auc:.4f}")
    
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()