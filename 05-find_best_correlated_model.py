#!/usr/bin/env python
"""
05-find_best_correlated_model.py

기존 학습된 모델들 중 disease ratio - attention score correlation이 높은 모델 찾기
Conditional / Non-conditional 모두 지원

Usage:
    python 05-find_best_correlated_model.py --mode all
    python 05-find_best_correlated_model.py --mode conditional --device 2
    python 05-find_best_correlated_model.py --mode non_conditional --device 1
"""
import sys
import os
sys.path.append(r"src")

import torch
import numpy as np
import pandas as pd
import scanpy as sc
import argparse
import json
from pathlib import Path
from scipy.stats import spearmanr
from tqdm import tqdm
import torch.nn as nn

from src.model import VQ_AENB, VQ_AENB_Conditional
from src.utils import load_ae_hyperparameters, load_mil_dataset_from_adata, VQEncoderWrapper, VQEncoderWrapperConditional


def get_model_folders(base_dir):
    """하위 모델 폴더들 찾기"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    folders = []
    for f in base_path.iterdir():
        if f.is_dir() and (f / 'model_teacher_exp1.pt').exists():
            folders.append(f)
    return sorted(folders)


def load_study_mapping(conditional_ae_dir):
    """Load study name -> study_id mapping"""
    mapping_path = f"{conditional_ae_dir}/study_mapping.json"
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
            return {v: int(k) for k, v in mapping.items()}
    return None


def get_study_ids_for_adata(adata, study_to_id, study_col='study'):
    """Get study_ids tensor for adata based on mapping"""
    if study_to_id is None:
        return None
    study_ids = [study_to_id.get(study, 0) for study in adata.obs[study_col]]
    return torch.tensor(study_ids, dtype=torch.long)


def calculate_correlation(adata, model_encoder, model_teacher, device, code_disease_ratio,
                          study_ids=None, use_conditional=False):
    """단일 모델의 disease ratio - attention score correlation 계산"""
    model_encoder.eval()
    model_teacher.eval()
    
    with torch.no_grad():
        data = torch.tensor(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X, 
                           dtype=torch.float32).to(device)
        
        if use_conditional and study_ids is not None:
            features = model_encoder(data, study_ids.to(device))[:, :model_teacher.input_dims]
        else:
            features = model_encoder(data)[:, :model_teacher.input_dims]
        
        attention_scores = model_teacher.attention_module(features).squeeze().cpu().numpy()
    
    # Code별 평균 attention score
    code_scores = pd.Series(attention_scores, index=adata.obs.index).groupby(adata.obs['vq_code']).mean()
    
    # Merge with disease ratio
    merged = pd.DataFrame({
        'disease_ratio': code_disease_ratio,
        'attention': code_scores
    }).dropna()
    
    if len(merged) < 10:
        return np.nan, np.nan
    
    r, p = spearmanr(merged['disease_ratio'], merged['attention'])
    return r, p


def load_adata_with_codes(dataset_name, device, pretrained_dir, use_conditional=False, 
                          conditional_ae_dir=None, study_to_id=None):
    """adata 로드 및 vq_code 할당"""
    h5ad_dir = '/home/bmi-user/workspace/data/HSvsCD/data/'
    adata_whole = sc.read_h5ad(f'{h5ad_dir}/Whole_SCP_PCD_Skin_805k_6k.h5ad')
    
    if dataset_name == 'SCP1884':
        adata = adata_whole[adata_whole.obs['study'] == 'SCP1884'].copy()
    elif dataset_name == 'Skin3':
        adata = adata_whole[adata_whole.obs['study'].isin(['GSE175990', 'GSE220116'])].copy()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    adata.obs['disease_numeric'] = adata.obs['disease_numeric'].astype(int)
    
    # Load pretrained encoder for code assignment
    ae_latent_dim, ae_hidden_layers, model_type, vq_num_codes, vq_commitment_weight = load_ae_hyperparameters(pretrained_dir)
    
    if use_conditional and model_type == 'VQ-AENB-Conditional':
        # Load conditional metadata
        metadata_path = f"{conditional_ae_dir}/conditional_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            n_studies = metadata.get('n_studies', 11)
        else:
            n_studies = 11
        
        ae = VQ_AENB_Conditional(
            input_dim=6000, latent_dim=ae_latent_dim,
            n_studies=n_studies, device=device,
            hidden_layers=ae_hidden_layers,
            num_codes=vq_num_codes, commitment_weight=vq_commitment_weight,
            activation_function=nn.Sigmoid
        ).to(device)
        ae.load_state_dict(torch.load(f'{pretrained_dir}/vq_aenb_conditional_whole.pth', 
                                      map_location=device, weights_only=False))
        
        # Get study_ids for code assignment
        study_ids = get_study_ids_for_adata(adata, study_to_id)
        
        with torch.no_grad():
            data = torch.tensor(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
                               dtype=torch.float32).to(device)
            codes = ae.get_codebook_indices(data, study_ids.to(device)).cpu().numpy()
    else:
        ae = VQ_AENB(
            input_dim=6000, latent_dim=ae_latent_dim,
            device=device, hidden_layers=ae_hidden_layers,
            num_codes=vq_num_codes, commitment_weight=vq_commitment_weight,
            activation_function=nn.Sigmoid
        ).to(device)
        ae.load_state_dict(torch.load(f'{pretrained_dir}/vq_aenb_whole.pth', 
                                      map_location=device, weights_only=False))
        study_ids = None
        
        with torch.no_grad():
            data = torch.tensor(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
                               dtype=torch.float32).to(device)
            codes = ae.get_codebook_indices(data).cpu().numpy()
    
    adata.obs['vq_code'] = codes
    
    # Calculate disease ratio per code
    code_disease = adata.obs.groupby('vq_code')['disease_numeric'].agg(['sum', 'count'])
    code_disease_ratio = code_disease['sum'] / code_disease['count']
    
    return adata, code_disease_ratio, study_ids


def analyze_models(model_base_dirs, device, use_conditional=False, conditional_ae_dir=None):
    """모델들의 correlation 분석"""
    n_exps = 8
    
    # Load study mapping for conditional
    study_to_id = None
    if use_conditional and conditional_ae_dir:
        study_to_id = load_study_mapping(conditional_ae_dir)
        if study_to_id:
            print(f"Loaded study mapping: {len(study_to_id)} studies")
    
    all_results = []
    
    for model_type, base_dir in model_base_dirs.items():
        dataset_name = 'SCP1884' if 'SCP1884' in model_type else 'Skin3'
        print(f"\n{'='*60}")
        print(f"Analyzing: {model_type}")
        print(f"Dataset: {dataset_name}")
        print('='*60)
        
        # Determine pretrained dir
        if use_conditional:
            pretrained_dir = f'{conditional_ae_dir}/AE/'
        else:
            pretrained_dir = 'data_quantized_all_datasets/AE/'
        
        # Load adata once per dataset
        print("Loading adata...")
        adata, code_disease_ratio, study_ids = load_adata_with_codes(
            dataset_name, device, pretrained_dir,
            use_conditional=use_conditional,
            conditional_ae_dir=conditional_ae_dir,
            study_to_id=study_to_id
        )
        print(f"  Loaded {adata.shape[0]} cells, {len(code_disease_ratio)} codes")
        
        # Get all model folders
        model_folders = get_model_folders(base_dir)
        print(f"  Found {len(model_folders)} model configurations")
        
        for model_folder in tqdm(model_folders, desc=f"Processing {model_type}"):
            model_name = model_folder.name
            
            exp_correlations = []
            
            for exp in range(1, n_exps + 1):
                try:
                    model_teacher = torch.load(model_folder / f'model_teacher_exp{exp}.pt',
                                              map_location=device, weights_only=False)
                    model_encoder = torch.load(model_folder / f'model_encoder_exp{exp}.pt',
                                              map_location=device, weights_only=False)
                    
                    model_teacher.to(device)
                    model_encoder.to(device)
                    
                    r, p = calculate_correlation(
                        adata, model_encoder, model_teacher, device, code_disease_ratio,
                        study_ids=study_ids, use_conditional=use_conditional
                    )
                    
                    exp_correlations.append({
                        'model_type': model_type,
                        'model_name': model_name,
                        'exp': exp,
                        'spearman_r': r,
                        'spearman_p': p
                    })
                    
                except Exception as e:
                    print(f"    Error exp{exp}: {e}")
                    continue
            
            if exp_correlations:
                df_exp = pd.DataFrame(exp_correlations)
                all_results.append({
                    'model_type': model_type,
                    'dataset': dataset_name,
                    'model_name': model_name,
                    'mean_correlation': df_exp['spearman_r'].mean(),
                    'std_correlation': df_exp['spearman_r'].std(),
                    'min_correlation': df_exp['spearman_r'].min(),
                    'max_correlation': df_exp['spearman_r'].max(),
                    'n_positive_exps': (df_exp['spearman_r'] > 0).sum(),
                })
        
        torch.cuda.empty_cache()
    
    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser(description='Find models with best disease ratio correlation')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'conditional', 'non_conditional'],
                        help='Which models to analyze')
    parser.add_argument('--device', type=int, default=1, help='CUDA device')
    parser.add_argument('--conditional_ae_dir', type=str, default='data_conditional/',
                        help='Conditional AE directory')
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path('./model_correlation_analysis_v3/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Non-conditional models
    if args.mode in ['all', 'non_conditional']:
        print("\n" + "="*60)
        print("🔬 Analyzing Non-Conditional Models")
        print("="*60)
        
        non_cond_dirs = {
            '0B_SCP1884': '/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_all_whole_encoder_freeze_SCP1884/',
            '0B_Skin3': '/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_all_whole_encoder_freeze_Skin3/',
            '0C_SCP1884': '/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_all_freeze_and_projection_SCP1884/',
            '0C_Skin3': '/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_all_freeze_and_projection_Skin3/',
        }
        
        results_nc = analyze_models(non_cond_dirs, device, use_conditional=False)
        if not results_nc.empty:
            results_nc['conditional'] = False
            all_results.append(results_nc)
    
    # Conditional models
    if args.mode in ['all', 'conditional']:
        print("\n" + "="*60)
        print("🔬 Analyzing Conditional Models")
        print("="*60)
        
        cond_dirs = {
            'Cond_SCP1884': '/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_conditional_v3_SCP1884/',
            'Cond_Skin3': '/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_conditional_v3_Skin3/',
        }
        
        results_c = analyze_models(cond_dirs, device, use_conditional=True, 
                                   conditional_ae_dir=args.conditional_ae_dir)
        if not results_c.empty:
            results_c['conditional'] = True
            all_results.append(results_c)
    
    # Combine results
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        results_df = results_df.sort_values('mean_correlation', ascending=False)
        
        # Save
        results_df.to_csv(output_dir / 'all_model_correlations.csv', index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 Top 10 Models by Correlation")
        print("="*60)
        print(results_df.head(10)[['model_type', 'model_name', 'mean_correlation', 'std_correlation']].to_string(index=False))
        
        # Best per model_type
        print("\n" + "="*60)
        print("📊 Best Model per Type")
        print("="*60)
        best_per_type = results_df.groupby('model_type').first().reset_index()
        print(best_per_type[['model_type', 'model_name', 'mean_correlation', 'std_correlation']].to_string(index=False))
        best_per_type.to_csv(output_dir / 'best_models_per_type.csv', index=False)
        
        # High correlation models
        high_corr = results_df[results_df['mean_correlation'] > 0.3]
        if len(high_corr) > 0:
            print(f"\n✅ Models with correlation > 0.3: {len(high_corr)}")
            high_corr.to_csv(output_dir / 'high_correlation_models.csv', index=False)
        
        print(f"\n📁 Results saved to: {output_dir}")
    else:
        print("No results found!")


if __name__ == '__main__':
    main()