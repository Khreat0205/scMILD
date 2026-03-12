#!/usr/bin/env python
"""
08_code_deg_overlap.py
======================
Code-level DEG cross-dataset overlap analysis.

각 VQ code에 대해:
  1) Skin3, SCP1884 각각에서 DEG 계산 (code cells vs same CT ctrl cells)
  2) Full DEG table을 CSV로 저장
  3) Cross-dataset overlap 통계 (Jaccard, overlap_coef) 산출
  4) code_label별 요약

Usage:
    python scripts/08_code_deg_overlap.py
    python scripts/08_code_deg_overlap.py --min_cells 30 --exclude_ctrl
    python scripts/08_code_deg_overlap.py --resume  # 체크포인트에서 재개
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG — 경로 수정 필요
# =============================================================================
BASE_DIR = '/home/bmi-user/workspace/BMI_BI_Open/Atlas_single_cell'

# Scored adata (vq_code, code_label, celltype, Status 포함된 h5ad)
# TODO: 실제 경로로 수정
ADATA_SKIN_PATH = f'{BASE_DIR}/adata_Skin_18kGene_flt_code.h5ad'
ADATA_SCP_PATH = f'{BASE_DIR}/adata_SCP1884_18kGene_code.h5ad'
CB_PATH = f'/home/bmi-user/workspace/BMI_BI_Open/Atlas_single_cell/nb03_results/codebook_adata_centroid_umap.h5ad'

# Output
OUTPUT_DIR = f'{BASE_DIR}/deg_overlap_results'

# =============================================================================
# CONSTANTS
# =============================================================================
# dominant_celltype (Skin3 기준) → SCP1884 celltype name
CT_NAME_MAP_TO_SCP = {
    'T cells':           'T cells',
    'Myeloid cells':     'Dendritic cells',
    'Keratinocytes':     'Epithelial cells',
    'B cells':           'B cells',
    'Plasma cells':      'Plasma cells',
    'Myofibroblasts':    'Myofibroblasts',
    'Endothelial cells': 'Endothelial cells',
    'Stromal cells':     'Stromal cells',
    'Mast cells':        'Mast cells',
    'Epithelial cells':  'Epithelial cells',
    'Dendritic cells':   'Dendritic cells',
}

STATUS_MAP = {
    'Skin3':   {'disease': 'HS',  'ctrl': 'ctrl_skin'},
    'SCP1884': {'disease': 'CD',  'ctrl': 'ctrl_colon'},
}

DEG_PARAMS = {
    'method': 'wilcoxon',
    'fdr_threshold': 0.01,
    'logfc_threshold': 1.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Code-level DEG cross-dataset overlap')
    parser.add_argument('--min_cells', type=int, default=10,
                        help='Minimum cells per group for DEG (default: 10)')
    parser.add_argument('--exclude_ctrl', action='store_true',
                        help='Exclude Control_associated codes')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--checkpoint_every', type=int, default=50,
                        help='Save checkpoint every N codes (default: 50)')
    return parser.parse_args()


def load_data():
    """Load adata and codebook."""
    print("Loading data...")
    t0 = time.time()

    adata_skin = sc.read_h5ad(ADATA_SKIN_PATH)
    print(f"  Skin3: {adata_skin.shape}")

    adata_scp = sc.read_h5ad(ADATA_SCP_PATH)
    print(f"  SCP1884: {adata_scp.shape}")

    cb = sc.read_h5ad(CB_PATH)
    print(f"  Codebook: {cb.shape}")
    print(f"  Loaded in {time.time() - t0:.0f}s")

    return adata_skin, adata_scp, cb


def precompute_cell_counts(adata_skin, adata_scp, cb):
    """
    각 code별 양쪽 dataset의 code cell 수 + ctrl cell 수를 미리 계산.
    실제 DEG를 안 돌려도 될 code를 사전에 걸러냄.
    """
    print("\nPre-computing cell counts per code...")
    cb_info = cb.obs[['code_idx', 'code_label', 'dominant_celltype']].copy()
    cb_info['code_idx'] = cb_info['code_idx'].astype(int)

    records = []
    for _, row in cb_info.iterrows():
        code = int(row['code_idx'])
        dom_ct = row['dominant_celltype']

        rec = {
            'code': code,
            'code_label': row['code_label'],
            'dominant_celltype': dom_ct,
        }

        for ds_name, adata in [('Skin3', adata_skin), ('SCP1884', adata_scp)]:
            ct_val = dom_ct if ds_name == 'Skin3' else CT_NAME_MAP_TO_SCP.get(dom_ct, dom_ct)
            ctrl_val = STATUS_MAP[ds_name]['ctrl']

            n_code = int(((adata.obs['vq_code'] == code) &
                          (adata.obs['celltype'] == ct_val)).sum())
            n_ctrl = int(((adata.obs['celltype'] == ct_val) &
                          (adata.obs['Status'] == ctrl_val) &
                          (adata.obs['vq_code'] != code)).sum())

            rec[f'n_code_{ds_name}'] = n_code
            rec[f'n_ctrl_{ds_name}'] = n_ctrl

        records.append(rec)

    counts_df = pd.DataFrame(records)
    return counts_df


def run_deg(adata, code, ct_val, ctrl_val):
    """
    Single code에 대해 DEG 실행.
    Returns: full DEG DataFrame (all genes, not just significant)
    """
    mask_code = (adata.obs['vq_code'] == code) & (adata.obs['celltype'] == ct_val)
    mask_ctrl = ((adata.obs['celltype'] == ct_val) &
                 (adata.obs['Status'] == ctrl_val) &
                 (adata.obs['vq_code'] != code))

    sub = adata[mask_code | mask_ctrl].copy()
    sub.obs['_group'] = 'ctrl'
    sub.obs.loc[mask_code[mask_code].index.intersection(sub.obs.index), '_group'] = 'code'

    sc.tl.rank_genes_groups(sub, groupby='_group', groups=['code'],
                            reference='ctrl', method=DEG_PARAMS['method'],
                            use_raw=False)
    result_df = sc.get.rank_genes_groups_df(sub, group='code')
    return result_df


def overlap_metrics(set_a, set_b):
    """Jaccard index and overlap coefficient."""
    if len(set_a) == 0 or len(set_b) == 0:
        return 0.0, 0.0, 0
    inter = set_a & set_b
    jaccard = len(inter) / len(set_a | set_b)
    ov_coef = len(inter) / min(len(set_a), len(set_b))
    return jaccard, ov_coef, len(inter)


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    deg_dir = os.path.join(OUTPUT_DIR, 'deg_per_code')
    os.makedirs(deg_dir, exist_ok=True)

    # --- Load ---
    adata_skin, adata_scp, cb = load_data()
    datasets = {'Skin3': adata_skin, 'SCP1884': adata_scp}

    # --- Pre-filter ---
    counts_df = precompute_cell_counts(adata_skin, adata_scp, cb)
    counts_df.to_csv(os.path.join(OUTPUT_DIR, 'code_cell_counts.csv'), index=False)

    # Filter: 양쪽 모두 code cells >= min_cells AND ctrl cells >= min_cells
    eligible = counts_df[
        (counts_df['n_code_Skin3'] >= args.min_cells) &
        (counts_df['n_ctrl_Skin3'] >= args.min_cells) &
        (counts_df['n_code_SCP1884'] >= args.min_cells) &
        (counts_df['n_ctrl_SCP1884'] >= args.min_cells)
    ].copy()

    if args.exclude_ctrl:
        eligible = eligible[eligible['code_label'] != 'Control_associated']

    eligible_codes = sorted(eligible['code'].tolist())
    total = len(eligible_codes)

    print(f"\n{'='*70}")
    print(f"Total codes: 512 → Eligible (min_cells={args.min_cells}): {total}")
    if args.exclude_ctrl:
        print(f"  (Control_associated excluded)")
    print(f"Label distribution:")
    print(eligible['code_label'].value_counts().to_string())
    print(f"{'='*70}\n")

    # --- Resume from checkpoint ---
    checkpoint_path = os.path.join(OUTPUT_DIR, 'overlap_checkpoint.csv')
    done_codes = set()
    results = []

    if args.resume and os.path.exists(checkpoint_path):
        prev = pd.read_csv(checkpoint_path)
        results = prev.to_dict('records')
        done_codes = set(prev['code'].tolist())
        print(f"Resuming: {len(done_codes)} codes already done\n")

    # --- Main loop ---
    print(f"{'#':>4} | {'Code':>5} | {'Label':>18} | {'dom_CT':>15} | "
          f"{'Skin3_DEG':>9} | {'SCP_DEG':>9} | {'Overlap':>7} | "
          f"{'Jaccard':>7} | {'OvCoef':>7} | {'Time':>5}")
    print("-" * 115)

    start_time = time.time()

    for idx, code in enumerate(eligible_codes):
        if code in done_codes:
            continue

        row_info = counts_df[counts_df['code'] == code].iloc[0]
        code_label = row_info['code_label']
        dom_ct = row_info['dominant_celltype']

        t_code = time.time()
        degs = {}
        n_sig = {}

        for ds_name, adata in datasets.items():
            ct_val = dom_ct if ds_name == 'Skin3' else CT_NAME_MAP_TO_SCP.get(dom_ct, dom_ct)
            ctrl_val = STATUS_MAP[ds_name]['ctrl']

            # DEG 실행
            full_deg = run_deg(adata, code, ct_val, ctrl_val)

            # Full DEG 저장
            deg_path = os.path.join(deg_dir, f'code{code}_{ds_name}.csv')
            full_deg.to_csv(deg_path, index=False)

            # Significant genes
            sig = full_deg[
                (full_deg['pvals_adj'] < DEG_PARAMS['fdr_threshold']) &
                (full_deg['logfoldchanges'] > DEG_PARAMS['logfc_threshold'])
            ]
            degs[ds_name] = set(sig['names'].values)
            n_sig[ds_name] = len(degs[ds_name])

        # Overlap
        jaccard, ov_coef, n_overlap = overlap_metrics(degs['Skin3'], degs['SCP1884'])

        elapsed = time.time() - t_code

        rec = {
            'code': code,
            'code_label': code_label,
            'dominant_celltype': dom_ct,
            'n_code_Skin3': int(row_info['n_code_Skin3']),
            'n_code_SCP1884': int(row_info['n_code_SCP1884']),
            'n_ctrl_Skin3': int(row_info['n_ctrl_Skin3']),
            'n_ctrl_SCP1884': int(row_info['n_ctrl_SCP1884']),
            'n_deg_Skin3': n_sig['Skin3'],
            'n_deg_SCP1884': n_sig['SCP1884'],
            'n_overlap': n_overlap,
            'jaccard': jaccard,
            'overlap_coef': ov_coef,
        }
        results.append(rec)

        # 출력
        progress = len(results)
        print(f"{progress:>4} | {code:>5} | {code_label:>18} | {dom_ct:>15} | "
              f"{n_sig['Skin3']:>9} | {n_sig['SCP1884']:>9} | {n_overlap:>7} | "
              f"{jaccard:>7.3f} | {ov_coef:>7.3f} | {elapsed:>4.1f}s")

        # Checkpoint
        if progress % args.checkpoint_every == 0:
            _save_checkpoint(results, checkpoint_path)
            elapsed_total = time.time() - start_time
            remaining = (total - progress) * (elapsed_total / progress)
            print(f"  >>> Checkpoint saved ({progress}/{total}), "
                  f"ETA: {remaining/60:.1f}min")

    # --- Final save ---
    overlap_df = pd.DataFrame(results)
    final_path = os.path.join(OUTPUT_DIR, 'code_overlap_summary.csv')
    overlap_df.to_csv(final_path, index=False)

    # Remove checkpoint (completed)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"DONE — {len(overlap_df)} codes processed in {(time.time()-start_time)/60:.1f}min")
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  code_overlap_summary.csv  — overlap statistics")
    print(f"  code_cell_counts.csv      — pre-filter cell counts (all 512)")
    print(f"  deg_per_code/             — full DEG tables ({len(overlap_df)*2} files)")
    print(f"\n{'='*70}")

    print(f"\n=== code_label별 Valid codes ===")
    print(overlap_df['code_label'].value_counts().to_string())

    print(f"\n=== code_label별 Jaccard 요약 ===")
    summary = overlap_df.groupby('code_label')['jaccard'].agg(
        ['count', 'mean', 'std', 'min', 'median', 'max']
    ).round(3)
    print(summary.to_string())

    print(f"\n=== code_label별 Overlap Coefficient 요약 ===")
    summary_ov = overlap_df.groupby('code_label')['overlap_coef'].agg(
        ['count', 'mean', 'std', 'min', 'median', 'max']
    ).round(3)
    print(summary_ov.to_string())


def _save_checkpoint(results, path):
    pd.DataFrame(results).to_csv(path, index=False)


if __name__ == '__main__':
    main()
