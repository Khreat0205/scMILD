# 04-cell_score_scMILDQ_conditional.py
"""
Cell-level scoring with Conditional VQ-AENB

Usage:
    python 04-cell_score_scMILDQ_conditional.py --device 0
    python 04-cell_score_scMILDQ_conditional.py --device 0 --use_conditional
    # Non-conditional (기존 방식)
    python 04-cell_score_scMILDQ_conditional.py --device 0

    # Conditional
    python 04-cell_score_scMILDQ_conditional.py --device 3 --use_conditional
    python 04-cell_score_scMILDQ_conditional.py --device 6 --use_conditional --output_dir adata_with_scores_v3_Skin3
"""

import sys, os
sys.path.append(r"src")
import torch 
torch.set_num_threads(32)
import numpy as np
import scanpy as sc
import pandas as pd
import argparse
import json
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from src.model import VQ_AENB, VQ_AENB_Conditional
from src.utils import load_ae_hyperparameters, load_mil_dataset_from_adata, add_cell_scores, VQEncoderWrapper, VQEncoderWrapperConditional


####################
# Utility functions
####################

def calculate_mean_scores(adata, model_key=None, score_types=['cell_score', 'cell_score_minmax', 'cell_score_cb', 'cell_score_cb_minmax']):
    new_cols = {}
    for score_type in score_types:
        exp_cols = [col for col in adata.obs.columns if col.startswith(f'{score_type}_{model_key}_exp')]
        if exp_cols:
            new_cols[f'{score_type}_{model_key}_mean'] = adata.obs[exp_cols].mean(axis=1).values
            print(f"  ✓ Calculated mean for {score_type} from {len(exp_cols)} experiments")
    
    if new_cols:
        new_df = pd.DataFrame(new_cols, index=adata.obs.index)
        adata.obs = pd.concat([adata.obs, new_df], axis=1)
    
    return adata


def refined_label_colors(adata):
    if 'disease_cat' not in adata.obs.columns:
        adata.obs['disease_cat'] = pd.Categorical(adata.obs['disease_numeric'].map({0: 'Control', 1: 'Case'}))
    disease_colors = {'Case': '#FF6B6B', 'Control': '#4ECDC4'}
    adata.uns['disease_cat_colors'] = [disease_colors[cat] for cat in adata.obs['disease_cat'].cat.categories]
    return adata


def perform_gmm_analysis(adata, model_key=None, exp=7):
    score_col = f'cell_score_minmax_{model_key}_exp{exp}'
    
    if score_col not in adata.obs.columns:
        return adata
    
    gmm = GaussianMixture(n_components=2, random_state=42,
                         init_params='kmeans',
                         means_init=np.array([[0.1], [0.9]]),
                         weights_init=np.array([0.5, 0.5]),
                         precisions_init=np.array([[[10.0]], [[10.0]]]))
    scores = adata.obs[score_col].values.reshape(-1, 1)
    labels = gmm.fit_predict(scores)
    
    mean0 = adata.obs.loc[labels == 0, score_col].mean()
    mean1 = adata.obs.loc[labels == 1, score_col].mean()
    
    high_label, low_label = (0, 1) if mean0 > mean1 else (1, 0)
    
    gmm_col = f'gmm_{model_key}_exp{exp}'
    adata.obs[gmm_col] = np.where(labels == high_label, 'high_scored', 'low_scored')
    
    assoc_col = f'association_{model_key}_exp{exp}'
    conditions = [
        (adata.obs['disease_cat'] == 'Case') & (adata.obs[gmm_col] == 'high_scored'),
        (adata.obs['disease_cat'] == 'Case') & (adata.obs[gmm_col] == 'low_scored'),
        (adata.obs['disease_cat'] == 'Control')
    ]
    choices = ['Case_associated', 'Case_independent', 'Control']
    adata.obs[assoc_col] = np.select(conditions, choices, default='Unknown')
    
    adata.obs[gmm_col] = pd.Categorical(adata.obs[gmm_col])
    adata.obs[assoc_col] = pd.Categorical(adata.obs[assoc_col])
    
    adata.uns[f'{gmm_col}_colors'] = ['#9B59B6', '#F39C12']
    adata.uns[f'{assoc_col}_colors'] = ['#E74C3C', '#F39C12', '#5DADE2']
    
    return adata


def get_code_assignments(adata, vq_model, dataset, device, study_ids=None, use_conditional=False):
    """Get VQ code assignments for each cell."""
    vq_model.eval()
    with torch.no_grad():
        data = dataset.data.clone().detach().to(device)
        if use_conditional and study_ids is not None:
            study_ids = study_ids.to(device)
            code_indices = vq_model.get_codebook_indices(data, study_ids)
        else:
            code_indices = vq_model.get_codebook_indices(data)
    return code_indices.cpu().numpy()


def load_study_mapping(conditional_ae_dir):
    """Load study name -> study_id mapping"""
    mapping_path = f"{conditional_ae_dir}/study_mapping.json"
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
            return {v: int(k) for k, v in mapping.items()}
    
    # Try alternate locations
    for suffix in ['SCP1884', 'Skin3']:
        alt_path = f"{conditional_ae_dir}/study_mapping_{suffix}.json"
        if os.path.exists(alt_path):
            with open(alt_path, 'r') as f:
                mapping = json.load(f)
                return {v: int(k) for k, v in mapping.items()}
    
    return None


def get_study_ids_for_adata(adata, study_to_id, study_col='study'):
    """Get study_ids tensor for adata based on mapping"""
    if study_to_id is None:
        return None
    
    study_ids = [study_to_id.get(study, 0) for study in adata.obs[study_col]]
    return torch.tensor(study_ids, dtype=torch.long)


def add_cell_scores_conditional(adata, model_encoder, model_teacher, model_student, device, 
                                 model_key=None, exp=None, dataset=None, study_ids=None, use_conditional=False):
    """Add cell scores with optional conditional encoding."""
    if dataset is None:
        dataset, _ = load_mil_dataset_from_adata(adata, device=device)
    
    model_encoder.eval() 
    model_teacher.eval()
    model_student.eval()
    
    # Define column suffix
    parts = []
    if model_key is not None:
        parts.append(model_key)
    if exp is not None:
        parts.append(f'exp{exp}')
    suffix = '_' + '_'.join(parts) if parts else ''    
    
    with torch.no_grad():
        data = dataset.data.clone().detach().to(device)
        
        if use_conditional and study_ids is not None:
            study_ids_device = study_ids.to(device)
            features = model_encoder(data, study_ids_device)[:, :model_teacher.input_dims]
        else:
            features = model_encoder(data)[:, :model_teacher.input_dims]
        
        cell_score_teacher = model_teacher.attention_module(features).squeeze(0)
        cell_score_student = model_student(features).squeeze(0)
        cell_score_student = torch.softmax(cell_score_student, dim=1)
    
    cell_score_student = cell_score_student[:, 1]
    features_np = features.cpu().numpy()
    cell_score_teacher_np = cell_score_teacher.cpu().detach().numpy()
    cell_score_teacher_np_minmax = (cell_score_teacher_np - cell_score_teacher_np.min()) / (cell_score_teacher_np.max() - cell_score_teacher_np.min())
    cell_score_student_np = cell_score_student.cpu().detach().numpy()
    cell_score_student_np_minmax = (cell_score_student_np - cell_score_student_np.min()) / (cell_score_student_np.max() - cell_score_student_np.min())

    adata.obs[f'cell_score{suffix}'] = cell_score_teacher_np
    adata.obs[f'cell_score_minmax{suffix}'] = cell_score_teacher_np_minmax
    adata.obs[f'cell_score_cb{suffix}'] = cell_score_student_np
    adata.obs[f'cell_score_cb_minmax{suffix}'] = cell_score_student_np_minmax
    adata.obsm[f'X_scMILD{suffix}'] = features_np

    return adata


####################
# Main
####################

def main():
    parser = argparse.ArgumentParser(description='Cell-level scoring with Conditional VQ-AENB')
    parser.add_argument('--device', type=int, default=0, help='CUDA device number')
    parser.add_argument('--n_exp', type=int, default=8, help='Number of experiments')
    parser.add_argument('--use_conditional', action='store_true', help='Use conditional encoder')
    parser.add_argument('--conditional_ae_dir', type=str, default='data_conditional/',
                       help='Directory with conditional AE files')
    parser.add_argument('--data_dir', type=str, default='data_quantized_all_datasets/',
                       help='Directory with quantized data')
    parser.add_argument('--h5ad_path', type=str,
                       default='/home/bmi-user/workspace/data/HSvsCD/data/Whole_SCP_PCD_Skin_805k_6k.h5ad')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (auto-generated if not specified)')
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"INFO: Using device: {device}")
    
    n_exp = args.n_exp
    exps = range(1, n_exp + 1)
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = 'conditional' if args.use_conditional else 'freeze_projection'
        output_dir = Path(f'./adata_with_scores_{suffix}/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"="*60)
    print(f"Cell Scoring - {'Conditional' if args.use_conditional else 'Non-conditional'}")
    print(f"="*60)
    print(f"Output: {output_dir}")
    print(f"="*60)
    
    ########### Model paths
    if args.use_conditional:
        saved_model_paths = {
            # 'Cond_SCP1884': '/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_conditional_SCP1884/',
            'Cond_Skin3': '/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_conditional_v3_Skin3/scMILDQ_all_model_ae_ed128_md128_cb1024_lr0.0001_elr0.0005_epoch100_1_15_reported_useoplTrue_lam0.2/',
        }
        pretrained_model_dir = f'{args.conditional_ae_dir}/AE/'
    else:
        saved_model_paths = {
            '0B_SCP1884': '/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_all_whole_encoder_freeze_SCP1884/scMILDQ_all_model_ae_ed128_md64_cb1024_lr0.001_elr0.0005_epoch30_1_5_reported_useoplTrue/',
            '0B_Skin3': '/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_all_whole_encoder_freeze_Skin3/scMILDQ_all_model_ae_ed128_md32_cb1024_lr0.001_elr0.0005_epoch30_1_5_reported_useoplTrue/',
            '0C_SCP1884': '/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_all_freeze_and_projection_SCP1884/scMILDQ_all_model_ae_ed128_md32_cb1024_lr0.0005_elr0.001_epoch30_1_5_reported_useoplTrue/',
            '0C_Skin3': '/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_all_freeze_and_projection_Skin3/scMILDQ_all_model_ae_ed128_md128_cb1024_lr0.001_elr0.001_epoch30_1_5_reported_useoplTrue/',
        }
        pretrained_model_dir = f'{args.data_dir}/AE/'
    
    # For conditional models, find best model in directory
    if args.use_conditional:
        resolved_paths = {}
        for key, base_dir in saved_model_paths.items():
            p = Path(base_dir)
            # Full path가 이미 model 파일 포함하면 그대로 사용
            if (p / 'model_teacher_exp8.pt').exists():
                resolved_paths[key] = str(p)
                print(f"  {key}: {p.name}")
            elif p.exists():
                # 아니면 하위 폴더 탐색
                subdirs = [x for x in p.iterdir() if x.is_dir() and (x / 'model_teacher_exp8.pt').exists()]
                if subdirs:
                    resolved_paths[key] = str(subdirs[0])
                    print(f"  {key}: {subdirs[0].name}")
        saved_model_paths = resolved_paths
    
    # Load AE hyperparameters
    ae_latent_dim, ae_hidden_layers, model_type, vq_num_codes, vq_commitment_weight = load_ae_hyperparameters(pretrained_model_dir)
    print(f"Model type: {model_type}, Latent dim: {ae_latent_dim}, Codebook: {vq_num_codes}")
    
    # Load study mapping for conditional
    study_to_id = None
    if args.use_conditional:
        study_to_id = load_study_mapping(args.conditional_ae_dir)
        if study_to_id:
            print(f"Loaded study mapping: {len(study_to_id)} studies")
        else:
            print("WARNING: Study mapping not found")
    
    # Load pretrained encoder
    print("\nLoading pretrained encoder...")
    if args.use_conditional and model_type == 'VQ-AENB-Conditional':
        # Load conditional metadata
        metadata_path = f"{args.conditional_ae_dir}/conditional_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            n_studies = metadata.get('n_studies', 11)
        else:
            n_studies = 11
        
        pretrained_path = f'{pretrained_model_dir}/vq_aenb_conditional_whole.pth'
        ae_model = VQ_AENB_Conditional(
            input_dim=6000, latent_dim=ae_latent_dim,
            n_studies=n_studies, device=device,
            hidden_layers=ae_hidden_layers,
            num_codes=vq_num_codes, commitment_weight=vq_commitment_weight,
            activation_function=nn.Sigmoid
        ).to(device)
        ae_model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=False))
        pretrained_encoder = VQEncoderWrapperConditional(ae_model)
        print(f"Loaded Conditional VQ-AENB (n_studies={n_studies})")
    else:
        pretrained_path = f'{pretrained_model_dir}/vq_aenb_whole.pth'
        ae_model = VQ_AENB(
            input_dim=6000, latent_dim=ae_latent_dim,
            device=device, hidden_layers=ae_hidden_layers,
            num_codes=vq_num_codes, commitment_weight=vq_commitment_weight,
            activation_function=nn.Sigmoid
        ).to(device)
        ae_model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=False))
        pretrained_encoder = VQEncoderWrapper(ae_model)
        print(f"Loaded VQ-AENB")
    
    pretrained_encoder.eval()
    
    # Load data
    print("\nLoading h5ad...")
    adata_whole = sc.read_h5ad(args.h5ad_path)
    adata_whole.obs['sample_id_numeric'] = adata_whole.obs['sample_id_numeric'].astype(int)
    adata_whole.obs['disease_numeric'] = adata_whole.obs['disease_numeric'].astype(int)
    refined_label_colors(adata_whole)
    
    # Build adata dict
    adata_skin3 = adata_whole[adata_whole.obs['study'].isin(['GSE175990', 'GSE220116'])].copy()
    adata_scp1884 = adata_whole[adata_whole.obs['study'] == 'SCP1884'].copy()
    
    adata_dict = {
        'Skin3': adata_skin3,
        'SCP1884': adata_scp1884,
    }
    print(f"Datasets: {list(adata_dict.keys())}")
    
    #############
    # Add pretrained embeddings
    print("\n[Step 1] Adding pretrained embeddings...")
    for adata_key, adata in tqdm(adata_dict.items(), desc="Pretrained embeddings"):
        dataset, _ = load_mil_dataset_from_adata(adata, device=device)
        study_ids = get_study_ids_for_adata(adata, study_to_id) if args.use_conditional else None
        
        with torch.no_grad():
            data = dataset.data.clone().detach().to(device)
            if args.use_conditional and study_ids is not None:
                pretrained_features = pretrained_encoder(data, study_ids.to(device))
            else:
                pretrained_features = pretrained_encoder(data)
            pretrained_features = pretrained_features[:, :ae_latent_dim]
        
        adata.obsm['X_Pretrained_whole'] = pretrained_features.cpu().numpy()
        
        # Code assignments
        code_indices = get_code_assignments(adata, ae_model, dataset, device, 
                                            study_ids=study_ids, use_conditional=args.use_conditional)
        adata.obs['vq_code'] = code_indices
    
    print("  ✓ Pretrained embeddings added")
    
    #############
    # Process each model
    print("\n[Step 2] Processing models...")
    
    checkpoint_file = output_dir / 'checkpoint.txt'
    completed_models = set()
    if checkpoint_file.exists():
        completed_models = set(checkpoint_file.read_text().strip().split('\n'))
        print(f"  Resuming... Completed: {completed_models}")
    
    for model_key, saved_model_path in saved_model_paths.items():
        if model_key in completed_models:
            print(f"\n[SKIP] {model_key} already processed")
            continue
        
        print(f"\n{'='*50}")
        print(f"Model: {model_key}")
        print(f"Path: {saved_model_path}")
        print(f"{'='*50}")
        
        total_iterations = len(exps) * len(adata_dict)
        with tqdm(total=total_iterations, desc=f"Processing {model_key}") as pbar:
            for exp in exps:
                model_teacher = torch.load(f'{saved_model_path}/model_teacher_exp{exp}.pt',
                                           map_location=device, weights_only=False)
                model_encoder = torch.load(f'{saved_model_path}/model_encoder_exp{exp}.pt',
                                           map_location=device, weights_only=False)
                model_student = torch.load(f'{saved_model_path}/model_student_exp{exp}.pt',
                                           map_location=device, weights_only=False)
                
                model_teacher.to(device)
                model_encoder.to(device)
                model_student.to(device)
                
                for adata_key, adata in adata_dict.items():
                    pbar.set_description(f"{model_key} | Exp {exp} | {adata_key}")
                    
                    dataset, _ = load_mil_dataset_from_adata(adata, device=device)
                    study_ids = get_study_ids_for_adata(adata, study_to_id) if args.use_conditional else None
                    
                    # Add cell scores
                    add_cell_scores_conditional(
                        adata, model_encoder, model_teacher, model_student, device,
                        model_key=model_key, exp=exp, dataset=dataset,
                        study_ids=study_ids, use_conditional=args.use_conditional
                    )
                    
                    # GMM analysis
                    perform_gmm_analysis(adata, model_key=model_key, exp=exp)
                    
                    pbar.update(1)
        
        # Calculate mean scores
        print(f"\nCalculating mean scores for {model_key}...")
        for adata_key, adata in adata_dict.items():
            calculate_mean_scores(adata, model_key=model_key)
        
        # Checkpoint
        with open(checkpoint_file, 'a') as f:
            f.write(f"{model_key}\n")
        print(f"  ✓ Checkpoint saved")
    
    torch.cuda.empty_cache()
    print("\n✅ All model processing complete!")
    
    #############
    # Save outputs
    print("\n[Step 3] Saving outputs...")
    
    # Save adata files
    for adata_key, adata in adata_dict.items():
        save_path = output_dir / f'adata_{adata_key}.h5ad'
        adata.write_h5ad(save_path)
        print(f"  ✓ Saved: {save_path.name} ({adata.shape[0]} cells)")
    
    # Save codebook
    codebook_weights = ae_model.quantizer.codebook.weight.detach().cpu().numpy()
    codebook_df = pd.DataFrame(codebook_weights,
                               columns=[f'dim_{i}' for i in range(codebook_weights.shape[1])])
    codebook_df.index.name = 'code_id'
    codebook_df.to_csv(output_dir / 'codebook_weights.csv')
    print(f"  ✓ Saved codebook: {codebook_weights.shape}")
    
    # Code statistics
    print("\n[Step 4] Calculating code statistics...")
    code_stats_list = []
    
    for model_key in saved_model_paths.keys():
        for adata_key, adata in adata_dict.items():
            if 'vq_code' not in adata.obs.columns:
                continue
            
            score_col = f'cell_score_{model_key}_mean'
            score_minmax_col = f'cell_score_minmax_{model_key}_mean'
            
            if score_col not in adata.obs.columns:
                continue
            
            for code in adata.obs['vq_code'].unique():
                code_mask = adata.obs['vq_code'] == code
                n_cells = code_mask.sum()
                
                if n_cells == 0:
                    continue
                
                n_case = (code_mask & (adata.obs['disease_numeric'] == 1)).sum()
                n_control = (code_mask & (adata.obs['disease_numeric'] == 0)).sum()
                
                scores = adata.obs.loc[code_mask, score_col]
                scores_minmax = adata.obs.loc[code_mask, score_minmax_col]
                
                code_stats_list.append({
                    'model': model_key,
                    'dataset': adata_key,
                    'code_id': code,
                    'n_cells': n_cells,
                    'n_case': n_case,
                    'n_control': n_control,
                    'case_ratio': n_case / n_cells,
                    'attn_score_mean': scores.mean(),
                    'attn_score_std': scores.std(),
                    'attn_score_minmax_mean': scores_minmax.mean(),
                    'attn_score_minmax_std': scores_minmax.std(),
                })
    
    if code_stats_list:
        code_stats_df = pd.DataFrame(code_stats_list)
        code_stats_df.to_csv(output_dir / 'code_statistics_with_attention.csv', index=False)
        print(f"  ✓ Saved code statistics: {len(code_stats_df)} rows")
    
    # Model info
    model_info = f"""Model paths:
{chr(10).join([f'{k}: {v}' for k, v in saved_model_paths.items()])}

Pretrained encoder: {pretrained_path}
Model type: {model_type}
Codebook size: {vq_num_codes}
Latent dim: {ae_latent_dim}
Conditional: {args.use_conditional}
Processed: {datetime.now()}
"""
    (output_dir / 'model_info.txt').write_text(model_info)
    
    print(f"\n🎉 Done! Output: {output_dir}")


if __name__ == '__main__':
    main()