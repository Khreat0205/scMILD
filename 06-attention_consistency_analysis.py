#!/usr/bin/env python
"""
06-attention_consistency_analysis.py

Seed별 attention score 일관성 분석
- Code별 attention score의 seed간 std
- Disease ratio와 attention의 correlation
# Non-conditional
python 06-attention_consistency_analysis.py --device 0

# Conditional
python 06-attention_consistency_analysis.py --device 3 --use_conditional --output_dir ./attention_consistency_analysis_v3 --all_models
"""
import sys, os
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
from src.utils import load_ae_hyperparameters, VQEncoderWrapper, VQEncoderWrapperConditional


def load_study_mapping(conditional_ae_dir):
    mapping_path = f"{conditional_ae_dir}/study_mapping.json"
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            return {v: int(k) for k, v in json.load(f).items()}
    return None


def get_study_ids_for_adata(adata, study_to_id):
    if study_to_id is None:
        return None
    return torch.tensor([study_to_id.get(s, 0) for s in adata.obs['study']], dtype=torch.long)


def get_all_model_folders(base_dir):
    """모든 하이퍼파라미터 조합 폴더"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return {}
    return {f.name: str(f) for f in base_path.iterdir() 
            if f.is_dir() and (f / 'model_teacher_exp1.pt').exists()}


def get_code_attention_per_seed(model_path, adata, codes, device, n_exps=8,
                                 study_ids=None, use_conditional=False):
    """
    각 seed별로 code의 평균 attention score 계산
    Returns: DataFrame (code_id x exp)
    """
    code_attention_dict = {f'exp{i}': {} for i in range(1, n_exps + 1)}
    
    for exp in range(1, n_exps + 1):
        try:
            model_teacher = torch.load(f'{model_path}/model_teacher_exp{exp}.pt',
                                       map_location=device, weights_only=False)
            model_encoder = torch.load(f'{model_path}/model_encoder_exp{exp}.pt',
                                       map_location=device, weights_only=False)
            
            model_teacher.to(device).eval()
            model_encoder.to(device).eval()
            
            with torch.no_grad():
                data = torch.tensor(
                    adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
                    dtype=torch.float32
                ).to(device)
                
                if use_conditional and study_ids is not None:
                    features = model_encoder(data, study_ids.to(device))[:, :model_teacher.input_dims]
                else:
                    features = model_encoder(data)[:, :model_teacher.input_dims]
                
                attention = model_teacher.attention_module(features).squeeze().cpu().numpy()
            
            # Code별 평균 attention
            for code in np.unique(codes):
                mask = codes == code
                code_attention_dict[f'exp{exp}'][code] = attention[mask].mean()
                
        except Exception as e:
            print(f"  Error exp{exp}: {e}")
            continue
    
    # DataFrame으로 변환 (code_id x exp)
    df = pd.DataFrame(code_attention_dict)
    df.index.name = 'code_id'
    return df


def analyze_code_consistency(code_attention_df, code_disease_ratio):
    """
    Code 수준 일관성 분석
    
    Returns:
        summary: dict with overall metrics
        code_stats: DataFrame with per-code stats
    """
    # 각 code의 seed간 통계
    code_stats = pd.DataFrame({
        'mean_attention': code_attention_df.mean(axis=1),
        'std_across_seeds': code_attention_df.std(axis=1),
        'min_attention': code_attention_df.min(axis=1),
        'max_attention': code_attention_df.max(axis=1),
        'range': code_attention_df.max(axis=1) - code_attention_df.min(axis=1),
    })
    
    # CV (Coefficient of Variation) - 평균 대비 변동
    code_stats['cv'] = code_stats['std_across_seeds'] / (code_stats['mean_attention'].abs() + 1e-8)
    
    # Disease ratio merge
    code_stats['disease_ratio'] = code_disease_ratio
    
    # Seed간 pairwise correlation (code attention 기준)
    exp_cols = code_attention_df.columns
    seed_correlations = []
    for i, col1 in enumerate(exp_cols):
        for col2 in exp_cols[i+1:]:
            valid = code_attention_df[[col1, col2]].dropna()
            if len(valid) > 10:
                r, _ = spearmanr(valid[col1], valid[col2])
                seed_correlations.append(r)
    
    # 각 seed별 disease ratio - attention correlation
    disease_corrs = []
    for col in exp_cols:
        merged = pd.DataFrame({
            'attention': code_attention_df[col],
            'disease_ratio': code_disease_ratio
        }).dropna()
        if len(merged) > 10:
            r, p = spearmanr(merged['attention'], merged['disease_ratio'])
            disease_corrs.append({'seed': col, 'spearman_r': r, 'p_value': p})
    
    disease_corr_df = pd.DataFrame(disease_corrs)
    
    # Summary
    summary = {
        'n_codes': len(code_stats),
        'seed_corr_mean': np.mean(seed_correlations) if seed_correlations else np.nan,
        'seed_corr_min': np.min(seed_correlations) if seed_correlations else np.nan,
        'seed_corr_std': np.std(seed_correlations) if seed_correlations else np.nan,
        'code_std_mean': code_stats['std_across_seeds'].mean(),
        'code_std_median': code_stats['std_across_seeds'].median(),
        'code_cv_mean': code_stats['cv'].mean(),
        'code_range_mean': code_stats['range'].mean(),
        'disease_corr_mean': disease_corr_df['spearman_r'].mean() if len(disease_corr_df) > 0 else np.nan,
        'disease_corr_std': disease_corr_df['spearman_r'].std() if len(disease_corr_df) > 0 else np.nan,
    }
    
    return summary, code_stats, disease_corr_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_conditional', action='store_true')
    parser.add_argument('--conditional_ae_dir', type=str, default='data_conditional/')
    parser.add_argument('--output_dir', type=str, default='./attention_consistency_analysis/')
    parser.add_argument('--all_models', action='store_true', help='Analyze all hyperparameter combinations')
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model paths
    if args.use_conditional:
        base_dirs = {
            'SCP1884': '/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_conditional_v3_SCP1884/',
            'Skin3': '/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_conditional_v3_Skin3/',
        }
        pretrained_dir = f'{args.conditional_ae_dir}/AE/'
    else:
        base_dirs = {
            'SCP1884': '/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_all_freeze_and_projection_SCP1884/',
            'Skin3': '/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_all_freeze_and_projection_Skin3/',
        }
        pretrained_dir = 'data_quantized_all_datasets/AE/'
    
    # Load study mapping
    study_to_id = load_study_mapping(args.conditional_ae_dir) if args.use_conditional else None
    
    # Load data
    print("Loading h5ad...")
    h5ad_path = '/home/bmi-user/workspace/data/HSvsCD/data/Whole_SCP_PCD_Skin_805k_6k.h5ad'
    adata_whole = sc.read_h5ad(h5ad_path)
    adata_whole.obs['disease_numeric'] = adata_whole.obs['disease_numeric'].astype(int)
    
    datasets = {
        'SCP1884': adata_whole[adata_whole.obs['study'] == 'SCP1884'].copy(),
        'Skin3': adata_whole[adata_whole.obs['study'].isin(['GSE175990', 'GSE220116'])].copy(),
    }
    
    # Load AE for code assignment
    ae_latent_dim, ae_hidden_layers, model_type, vq_num_codes, vq_commitment_weight = load_ae_hyperparameters(pretrained_dir)
    
    if args.use_conditional:
        n_studies = 11
        metadata_path = f"{args.conditional_ae_dir}/conditional_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                n_studies = json.load(f).get('n_studies', 11)
        
        ae = VQ_AENB_Conditional(
            input_dim=6000, latent_dim=ae_latent_dim, n_studies=n_studies,
            device=device, hidden_layers=ae_hidden_layers,
            num_codes=vq_num_codes, commitment_weight=vq_commitment_weight,
            activation_function=nn.Sigmoid
        ).to(device)
        ae.load_state_dict(torch.load(f'{pretrained_dir}/vq_aenb_conditional_whole.pth', map_location=device, weights_only=False))
    else:
        ae = VQ_AENB(
            input_dim=6000, latent_dim=ae_latent_dim, device=device,
            hidden_layers=ae_hidden_layers, num_codes=vq_num_codes,
            commitment_weight=vq_commitment_weight, activation_function=nn.Sigmoid
        ).to(device)
        ae.load_state_dict(torch.load(f'{pretrained_dir}/vq_aenb_whole.pth', map_location=device, weights_only=False))
    
    # Assign codes & calculate disease ratio
    print("Assigning codes and calculating disease ratio...")
    dataset_info = {}
    for name, adata in datasets.items():
        study_ids = get_study_ids_for_adata(adata, study_to_id) if args.use_conditional else None
        
        with torch.no_grad():
            data = torch.tensor(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
                               dtype=torch.float32).to(device)
            if args.use_conditional and study_ids is not None:
                codes = ae.get_codebook_indices(data, study_ids.to(device)).cpu().numpy()
            else:
                codes = ae.get_codebook_indices(data).cpu().numpy()
        
        # Disease ratio per code
        code_disease = pd.DataFrame({'code': codes, 'disease': adata.obs['disease_numeric'].values})
        code_disease_ratio = code_disease.groupby('code')['disease'].mean()
        
        dataset_info[name] = {
            'adata': adata,
            'codes': codes,
            'disease_ratio': code_disease_ratio,
            'study_ids': study_ids,
            'n_unique_codes': len(np.unique(codes))
        }
        print(f"  {name}: {len(np.unique(codes))} codes, {len(adata)} cells")
    
    # Analyze models
    all_summaries = []
    all_code_stats = []
    
    for dataset_name, base_dir in base_dirs.items():
        info = dataset_info[dataset_name]
        
        if args.all_models:
            model_folders = get_all_model_folders(base_dir)
        else:
            # Best model only (첫 번째)
            model_folders = get_all_model_folders(base_dir)
            if model_folders:
                first_key = list(model_folders.keys())[0]
                model_folders = {first_key: model_folders[first_key]}
        
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} | Models: {len(model_folders)}")
        print('='*60)
        
        for model_name, model_path in tqdm(model_folders.items(), desc=dataset_name):
            # Get code attention per seed
            code_attention_df = get_code_attention_per_seed(
                model_path, info['adata'], info['codes'], device,
                study_ids=info['study_ids'], use_conditional=args.use_conditional
            )
            
            if code_attention_df.empty:
                continue
            
            # Analyze consistency
            summary, code_stats, disease_corr_df = analyze_code_consistency(
                code_attention_df, info['disease_ratio']
            )
            
            summary['model'] = model_name
            summary['dataset'] = dataset_name
            all_summaries.append(summary)
            
            code_stats['model'] = model_name
            code_stats['dataset'] = dataset_name
            code_stats = code_stats.reset_index()
            all_code_stats.append(code_stats)
    
    # Combine and save
    summary_df = pd.DataFrame(all_summaries)
    summary_df = summary_df.sort_values(['dataset', 'seed_corr_mean'], ascending=[True, False])
    summary_df.to_csv(output_dir / 'code_consistency_summary.csv', index=False)
    
    if all_code_stats:
        code_stats_df = pd.concat(all_code_stats, ignore_index=True)
        code_stats_df.to_csv(output_dir / 'per_code_stats.csv', index=False)
    
    # Print results
    print("\n" + "="*70)
    print("📊 Code-Level Consistency Summary (sorted by seed_corr_mean)")
    print("="*70)
    
    display_cols = ['dataset', 'model', 'seed_corr_mean', 'seed_corr_min', 
                    'code_std_mean', 'disease_corr_mean']
    display_cols = [c for c in display_cols if c in summary_df.columns]
    
    # 짧은 모델명 표시
    summary_display = summary_df[display_cols].copy()
    summary_display['model'] = summary_display['model'].str.replace('scMILDQ_all_model_ae_ed128_', '').str.replace('_reported_useoplTrue', '')
    
    print(summary_display.to_string(index=False))
    
    # Best per dataset
    print("\n📊 Best Model per Dataset:")
    for dataset in summary_df['dataset'].unique():
        best = summary_df[summary_df['dataset'] == dataset].iloc[0]
        print(f"\n  {dataset}:")
        print(f"    Model: {best['model']}")
        print(f"    Seed Corr: {best['seed_corr_mean']:.4f} (min: {best['seed_corr_min']:.4f})")
        print(f"    Disease Corr: {best['disease_corr_mean']:.4f}")
    
    print(f"\n📁 Results saved to: {output_dir}")


if __name__ == '__main__':
    main()