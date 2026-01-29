"""
sample_id_numeric -> study_id 매핑 파일 생성
scMILD 학습 전에 한 번 실행

Usage:
    python generate_sample_to_study_mapping.py --adata_path data/adata_combined.h5ad \
        --output_dir data_conditional --n_exp 8 --suffix SCP1884
"""

import json
import argparse
import scanpy as sc
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def generate_mapping(adata_path, output_dir, n_exp=8, suffix=None, 
                     study_col='study', sample_col='sample_id_numeric', 
                     disease_col='disease_numeric', split_ratio=[0.5, 0.25, 0.25]):
    """
    각 exp별 train/val/test split에 맞춰 sample_id -> study_id 매핑 생성
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load adata
    print(f"Loading {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    
    # Study encoding
    study_encoder = LabelEncoder()
    adata.obs['study_id_numeric'] = study_encoder.fit_transform(adata.obs[study_col])
    
    # Save study mapping (study_id -> study_name)
    study_mapping = {str(i): name for i, name in enumerate(study_encoder.classes_)}
    with open(f"{output_dir}/study_mapping.json", 'w') as f:
        json.dump(study_mapping, f, indent=2)
    print(f"Study mapping: {study_mapping}")
    
    # sample_id -> study_id 매핑 (전체)
    sample_study_df = adata.obs[[sample_col, 'study_id_numeric']].drop_duplicates()
    full_mapping = dict(zip(
        sample_study_df[sample_col].astype(int), 
        sample_study_df['study_id_numeric'].astype(int)
    ))
    
    # 전체 매핑 저장 (exp 무관하게 사용 가능)
    fname = f"sample_to_study_full{'_' + suffix if suffix else ''}.json"
    with open(f"{output_dir}/{fname}", 'w') as f:
        json.dump({str(k): v for k, v in full_mapping.items()}, f, indent=2)
    print(f"Saved: {fname} ({len(full_mapping)} samples)")
    
    # Exp별 저장 (split 재현용 - 선택사항)
    sample_labels = adata.obs[[disease_col, sample_col]].drop_duplicates()
    
    for exp in range(1, n_exp + 1):
        # 동일한 split 로직 (utils.py의 load_and_save_datasets_adata와 동일)
        train_val_set, test_set = train_test_split(
            sample_labels, test_size=split_ratio[2], 
            random_state=exp, stratify=sample_labels[disease_col]
        )
        train_set, val_set = train_test_split(
            train_val_set, test_size=split_ratio[1] / (1 - split_ratio[2]),
            random_state=exp, stratify=train_val_set[disease_col]
        )
        
        # 해당 exp의 sample들만 추출
        all_samples = set(train_set[sample_col]) | set(val_set[sample_col]) | set(test_set[sample_col])
        exp_mapping = {str(int(s)): full_mapping[int(s)] for s in all_samples if int(s) in full_mapping}
        
        fname = f"sample_to_study_exp{exp}{'_' + suffix if suffix else ''}.json"
        with open(f"{output_dir}/{fname}", 'w') as f:
            json.dump(exp_mapping, f, indent=2)
        print(f"Saved: {fname} ({len(exp_mapping)} samples)")
    
    print(f"\n✅ 매핑 파일 생성 완료: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--adata_path', type=str, required=True, help='Path to combined adata h5ad')
    parser.add_argument('--output_dir', type=str, default='data_conditional')
    parser.add_argument('--n_exp', type=int, default=8)
    parser.add_argument('--suffix', type=str, default=None, help='Dataset suffix (e.g., SCP1884, Skin3)')
    parser.add_argument('--study_col', type=str, default='study')
    parser.add_argument('--sample_col', type=str, default='sample_id_numeric')
    parser.add_argument('--disease_col', type=str, default='disease_numeric')
    args = parser.parse_args()
    
    generate_mapping(
        adata_path=args.adata_path,
        output_dir=args.output_dir,
        n_exp=args.n_exp,
        suffix=args.suffix,
        study_col=args.study_col,
        sample_col=args.sample_col,
        disease_col=args.disease_col
    )