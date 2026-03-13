#!/usr/bin/env python
"""
07_evaluate_codebook.py - Pretrained Encoder Codebook 품질 평가

사전학습된 VQ-AENB-Conditional encoder의 codebook 품질을 평가합니다.
MIL 학습 없이, pretrain 결과만으로 다음 지표를 산출합니다:

1. Code Usage: 각 code에 할당된 세포 수 및 비율
2. Celltype Purity: 각 code 내 celltype 순도 (entropy, dominant fraction)
3. Study Mixing: 각 code 내 study 혼합 정도 (entropy, effective #studies)

Usage:
    python scripts/07_evaluate_codebook.py --config config/pretrain_celltype_aux.yaml --gpu 0
    python scripts/07_evaluate_codebook.py --config config/pretrain_celltype_aux.yaml --celltype_col cell_type --gpu 0
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from scipy.stats import entropy

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data import load_adata, load_adata_with_subset, preprocess_adata, encode_labels


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pretrained_model(model_path: str, device: torch.device):
    """Pretrained VQ-AENB-Conditional 모델 로드."""
    from src.models.autoencoder import VQ_AENB_Conditional

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint['config']

    # Backward compatibility: n_studies -> n_conditionals
    n_conditionals = cfg.get('n_conditionals', cfg.get('n_studies', 1))
    conditional_emb_dim = cfg.get('conditional_emb_dim', cfg.get('study_emb_dim', 16))

    model = VQ_AENB_Conditional(
        input_dim=cfg['input_dim'],
        latent_dim=cfg['latent_dim'],
        device=device,
        hidden_layers=cfg.get('hidden_layers', [512, 256, 128]),
        n_conditionals=n_conditionals,
        conditional_emb_dim=conditional_emb_dim,
        num_codes=cfg['num_codes'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"  Model loaded: input_dim={cfg['input_dim']}, latent_dim={cfg['latent_dim']}, "
          f"num_codes={cfg['num_codes']}, n_conditionals={n_conditionals}")
    return model


def assign_vq_codes(model, adata, conditional_col: str, device: torch.device,
                    batch_size: int = 512):
    """전체 데이터에 대해 VQ code 할당."""
    if hasattr(adata.X, 'toarray'):
        data = torch.tensor(adata.X.toarray(), dtype=torch.float32)
    else:
        data = torch.tensor(np.array(adata.X), dtype=torch.float32)

    conditional_ids = torch.tensor(
        adata.obs[conditional_col].values, dtype=torch.long
    )

    all_codes = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch_x = data[i:i+batch_size].to(device)
            batch_cond = conditional_ids[i:i+batch_size].to(device)
            codes = model.get_codebook_indices(batch_x, batch_cond)
            all_codes.append(codes.cpu().numpy())

    return np.concatenate(all_codes)


def compute_code_stats(vq_codes: np.ndarray, celltype_labels: np.ndarray,
                       conditional_labels: np.ndarray, num_codes: int):
    """Code별 usage, celltype purity, study mixing 통계 계산.

    Returns:
        code_stats: DataFrame with per-code statistics
        summary: dict with global summary metrics
    """
    df = pd.DataFrame({
        'vq_code': vq_codes,
        'celltype': celltype_labels,
        'conditional': conditional_labels,
    })

    records = []
    for code_id in range(num_codes):
        mask = df['vq_code'] == code_id
        subset = df[mask]
        n_cells = len(subset)

        if n_cells == 0:
            records.append({
                'code': code_id,
                'n_cells': 0,
                'usage_frac': 0.0,
                'dominant_celltype': 'N/A',
                'celltype_purity': 0.0,
                'celltype_entropy': 0.0,
                'n_celltypes': 0,
                'dominant_conditional': 'N/A',
                'conditional_entropy': 0.0,
                'n_conditionals': 0,
                'effective_n_conditionals': 0.0,
            })
            continue

        # --- Celltype purity ---
        ct_counts = subset['celltype'].value_counts()
        ct_probs = ct_counts.values / ct_counts.values.sum()
        ct_ent = entropy(ct_probs, base=2)
        dominant_ct = ct_counts.index[0]
        purity = ct_counts.values[0] / n_cells

        # --- Study/Conditional mixing ---
        cond_counts = subset['conditional'].value_counts()
        cond_probs = cond_counts.values / cond_counts.values.sum()
        cond_ent = entropy(cond_probs, base=2)
        # Effective number of conditionals (2^entropy)
        eff_n_cond = 2 ** cond_ent if cond_ent > 0 else 1.0

        records.append({
            'code': code_id,
            'n_cells': n_cells,
            'usage_frac': n_cells / len(df),
            'dominant_celltype': dominant_ct,
            'celltype_purity': purity,
            'celltype_entropy': ct_ent,
            'n_celltypes': len(ct_counts),
            'dominant_conditional': cond_counts.index[0],
            'conditional_entropy': cond_ent,
            'n_conditionals': len(cond_counts),
            'effective_n_conditionals': eff_n_cond,
        })

    code_stats = pd.DataFrame(records)

    # --- Global summary ---
    used_codes = code_stats[code_stats['n_cells'] > 0]
    n_total_conditionals = df['conditional'].nunique()

    summary = {
        'total_cells': len(df),
        'num_codes': num_codes,
        'used_codes': len(used_codes),
        'dead_codes': num_codes - len(used_codes),
        'usage_rate': len(used_codes) / num_codes,
        'mean_celltype_purity': used_codes['celltype_purity'].mean(),
        'median_celltype_purity': used_codes['celltype_purity'].median(),
        'mean_celltype_entropy': used_codes['celltype_entropy'].mean(),
        'mean_conditional_entropy': used_codes['conditional_entropy'].mean(),
        'mean_effective_n_conditionals': used_codes['effective_n_conditionals'].mean(),
        'ideal_conditional_entropy': np.log2(n_total_conditionals) if n_total_conditionals > 1 else 0.0,
        'n_total_celltypes': df['celltype'].nunique(),
        'n_total_conditionals': n_total_conditionals,
    }

    return code_stats, summary


def compute_crosstabs(vq_codes, celltype_labels, conditional_labels):
    """Code × Celltype, Code × Conditional 교차표."""
    df = pd.DataFrame({
        'vq_code': vq_codes,
        'celltype': celltype_labels,
        'conditional': conditional_labels,
    })
    ct_cross = pd.crosstab(df['vq_code'], df['celltype'], margins=True)
    cond_cross = pd.crosstab(df['vq_code'], df['conditional'], margins=True)
    return ct_cross, cond_cross


def print_summary(summary: dict, code_stats: pd.DataFrame):
    """콘솔에 요약 출력."""
    print("\n" + "=" * 70)
    print("  CODEBOOK QUALITY EVALUATION")
    print("=" * 70)

    print(f"\n[Overview]")
    print(f"  Total cells:        {summary['total_cells']:,}")
    print(f"  Codebook size:      {summary['num_codes']}")
    print(f"  Used codes:         {summary['used_codes']} / {summary['num_codes']} "
          f"({summary['usage_rate']:.1%})")
    print(f"  Dead codes:         {summary['dead_codes']}")

    print(f"\n[Celltype Purity] (higher = code captures single celltype)")
    print(f"  Mean purity:        {summary['mean_celltype_purity']:.3f}")
    print(f"  Median purity:      {summary['median_celltype_purity']:.3f}")
    print(f"  Mean entropy:       {summary['mean_celltype_entropy']:.3f} bits")
    print(f"  Total celltypes:    {summary['n_total_celltypes']}")

    print(f"\n[Study/Conditional Mixing] (higher entropy = better batch correction)")
    print(f"  Mean entropy:       {summary['mean_conditional_entropy']:.3f} bits")
    print(f"  Ideal entropy:      {summary['ideal_conditional_entropy']:.3f} bits "
          f"(uniform over {summary['n_total_conditionals']} groups)")
    print(f"  Mean eff. #groups:  {summary['mean_effective_n_conditionals']:.2f} "
          f"/ {summary['n_total_conditionals']}")

    # Top-10 most used codes
    used = code_stats[code_stats['n_cells'] > 0].sort_values('n_cells', ascending=False)
    print(f"\n[Top 10 Most Used Codes]")
    print(f"  {'Code':>6}  {'Cells':>8}  {'Usage%':>7}  {'Purity':>7}  {'CT_Ent':>7}  "
          f"{'Cond_Ent':>8}  {'Dominant Celltype'}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*20}")
    for _, row in used.head(10).iterrows():
        print(f"  {int(row['code']):>6}  {int(row['n_cells']):>8}  "
              f"{row['usage_frac']:>6.2%}  {row['celltype_purity']:>7.3f}  "
              f"{row['celltype_entropy']:>7.3f}  {row['conditional_entropy']:>8.3f}  "
              f"{row['dominant_celltype']}")

    # Purity distribution
    print(f"\n[Purity Distribution]")
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    for lo, hi in bins:
        count = ((used['celltype_purity'] >= lo) & (used['celltype_purity'] < hi)).sum()
        label = f"  [{lo:.1f}, {hi:.1f})" if hi < 1.01 else f"  [{lo:.1f}, 1.0]"
        print(f"{label:>16}: {count:>4} codes")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained codebook quality")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override pretrained model path (default: from config)")
    parser.add_argument("--celltype_col", type=str, default=None,
                        help="Override celltype column (default: from config celltype_aux.column)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for inference")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results/codebook_eval_<timestamp>)")
    args = parser.parse_args()

    set_seed(42)

    # --- Config ---
    config = load_config(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model_path = args.model_path or config.paths.pretrained_encoder
    if not model_path or not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    celltype_col = args.celltype_col or config.encoder.pretrain.celltype_aux.column
    conditional_col = config.data.conditional_embedding.column
    conditional_encoded_col = config.data.conditional_embedding.encoded_column

    # --- Output dir ---
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.paths.output_root) / f"codebook_eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config:         {args.config}")
    print(f"Model:          {model_path}")
    print(f"Celltype col:   {celltype_col}")
    print(f"Conditional:    {conditional_col} (encoded: {conditional_encoded_col})")
    print(f"Output:         {output_dir}")
    print(f"Device:         {device}")

    # --- Load data ---
    print("\n[1/4] Loading data...")
    adata = load_adata_with_subset(
        whole_adata_path=config.data.whole_adata_path,
        subset_enabled=config.data.subset.enabled,
        subset_column=config.data.subset.column,
        subset_values=config.data.subset.values,
        cache_dir=config.data.subset.cache_dir,
        use_cache=config.data.subset.use_cache,
    )
    print(f"  Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # --- Preprocess ---
    print("\n[2/4] Preprocessing...")
    pp = config.data.preprocessing
    adata = preprocess_adata(adata, n_top_genes=pp.n_top_genes,
                             normalize_total=pp.normalize_total,
                             log_transform=pp.log_transform)

    # Encode conditional labels
    adata, _ = encode_labels(
        adata,
        sample_col=config.data.columns.sample_name,
        label_col=config.data.columns.status,
        conditional_col=conditional_col,
        conditional_encoded_col=conditional_encoded_col,
    )

    # Validate celltype column
    if celltype_col not in adata.obs.columns:
        print(f"  WARNING: celltype column '{celltype_col}' not found. "
              f"Available: {list(adata.obs.columns)}")
        print(f"  Celltype purity will use 'unknown' for all cells.")
        celltype_labels = np.array(['unknown'] * adata.n_obs)
    else:
        celltype_labels = adata.obs[celltype_col].astype(str).fillna('unknown').values
        n_types = len(np.unique(celltype_labels))
        print(f"  Celltypes: {n_types} types from '{celltype_col}'")

    conditional_labels = adata.obs[conditional_col].astype(str).values
    print(f"  Conditionals: {len(np.unique(conditional_labels))} groups from '{conditional_col}'")
    print(f"  Final shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # --- Load model & assign codes ---
    print("\n[3/4] Loading model & assigning VQ codes...")
    model = load_pretrained_model(model_path, device)
    vq_codes = assign_vq_codes(model, adata, conditional_encoded_col, device,
                                batch_size=args.batch_size)
    num_codes = model.quantizer.num_codes
    print(f"  Unique codes used: {len(np.unique(vq_codes))} / {num_codes}")

    # --- Compute statistics ---
    print("\n[4/4] Computing statistics...")
    code_stats, summary = compute_code_stats(vq_codes, celltype_labels,
                                              conditional_labels, num_codes)

    # --- Print summary ---
    print_summary(summary, code_stats)

    # --- Save outputs ---
    code_stats.to_csv(output_dir / "code_stats.csv", index=False)

    ct_cross, cond_cross = compute_crosstabs(vq_codes, celltype_labels, conditional_labels)
    ct_cross.to_csv(output_dir / "crosstab_code_celltype.csv")
    cond_cross.to_csv(output_dir / "crosstab_code_conditional.csv")

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {output_dir}/")
    print(f"  - code_stats.csv                 (per-code statistics)")
    print(f"  - crosstab_code_celltype.csv     (code × celltype)")
    print(f"  - crosstab_code_conditional.csv  (code × conditional)")
    print(f"  - summary.json                   (global metrics)")


if __name__ == "__main__":
    main()
