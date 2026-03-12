#!/usr/bin/env python
"""
09_deg_overlap_topN.py
======================
저장된 code별 DEG CSV에서 top N gene 기준 overlap 통계를 계산.

08_code_deg_overlap.py의 후속 스크립트.
Full DEG table이 이미 저장되어 있으므로 scanpy 재실행 없이 빠르게 수행.

Usage:
    python scripts/09_deg_overlap_topN.py
    python scripts/09_deg_overlap_topN.py --top_ns 10 20 50 100 200
    python scripts/09_deg_overlap_topN.py --rank_by pvals_adj --ascending
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = './'
DEG_DIR = os.path.join(BASE_DIR, 'deg_overlap_results', 'deg_per_code')
SUMMARY_PATH = os.path.join(BASE_DIR, 'deg_overlap_results', 'code_overlap_summary.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'deg_overlap_results')


def parse_args():
    parser = argparse.ArgumentParser(description='Top-N DEG overlap analysis')
    parser.add_argument('--top_ns', type=int, nargs='+', default=[10, 50, 100, 200],
                        help='Top N values to compute (default: 10 50 100 200)')
    parser.add_argument('--rank_by', type=str, default='logfoldchanges',
                        choices=['logfoldchanges', 'scores', 'pvals_adj'],
                        help='Column to rank genes by (default: logfoldchanges)')
    parser.add_argument('--ascending', action='store_true',
                        help='Rank ascending (use for pvals_adj)')
    parser.add_argument('--sig_only', action='store_true',
                        help='Only consider significant genes (FDR<0.01, logFC>1) before ranking')
    return parser.parse_args()


def load_deg(code, dataset, deg_dir=DEG_DIR):
    """Load full DEG table for a code-dataset pair."""
    path = os.path.join(deg_dir, f'code{code}_{dataset}.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def get_top_genes(deg_df, n, rank_by='logfoldchanges', ascending=False, sig_only=False):
    """
    DEG table에서 top N gene names 추출.
    
    sig_only=True: significant genes 중에서만 top N
    sig_only=False: 전체 gene 중 top N (logFC > 0인 것만)
    """
    if deg_df is None or len(deg_df) == 0:
        return set()
    
    df = deg_df.copy()
    
    if sig_only:
        df = df[(df['pvals_adj'] < 0.01) & (df['logfoldchanges'] > 1.0)]
    else:
        # 최소한 upregulated만
        df = df[df['logfoldchanges'] > 0]
    
    if len(df) == 0:
        return set()
    
    df = df.sort_values(rank_by, ascending=ascending)
    top = df.head(n)
    return set(top['names'].values)


def overlap_metrics(set_a, set_b):
    if len(set_a) == 0 or len(set_b) == 0:
        return 0.0, 0.0, 0
    inter = set_a & set_b
    jaccard = len(inter) / len(set_a | set_b)
    ov_coef = len(inter) / min(len(set_a), len(set_b))
    return jaccard, ov_coef, len(inter)


def main():
    args = parse_args()
    top_ns = sorted(args.top_ns)
    
    # Load summary (code list + metadata)
    summary = pd.read_csv(SUMMARY_PATH)
    codes = summary['code'].tolist()
    
    print(f"Codes: {len(codes)}")
    print(f"Top Ns: {top_ns}")
    print(f"Rank by: {args.rank_by} ({'ascending' if args.ascending else 'descending'})")
    print(f"Sig only: {args.sig_only}")
    print(f"{'='*90}\n")
    
    # Header
    n_cols = ''.join([f' | J_{n:>3} OC_{n:>3}' for n in top_ns])
    print(f"{'Code':>5} | {'Label':>18}{n_cols}")
    print("-" * (30 + 16 * len(top_ns)))
    
    # Compute
    results = []
    for code in codes:
        deg_skin = load_deg(code, 'Skin3')
        deg_scp = load_deg(code, 'SCP1884')
        
        row_info = summary[summary['code'] == code].iloc[0]
        rec = {'code': code}
        
        vals_str = ""
        for n in top_ns:
            genes_skin = get_top_genes(deg_skin, n, args.rank_by, args.ascending, args.sig_only)
            genes_scp = get_top_genes(deg_scp, n, args.rank_by, args.ascending, args.sig_only)
            
            j, oc, n_ov = overlap_metrics(genes_skin, genes_scp)
            
            rec[f'jaccard_top{n}'] = j
            rec[f'overlap_coef_top{n}'] = oc
            rec[f'n_overlap_top{n}'] = n_ov
            rec[f'n_skin3_top{n}'] = len(genes_skin)
            rec[f'n_scp_top{n}'] = len(genes_scp)
            
            vals_str += f" | {j:>.3f} {oc:>.3f}"
        
        results.append(rec)
        print(f"{code:>5} | {row_info['code_label']:>18}{vals_str}")
    
    # Merge with original summary
    topn_df = pd.DataFrame(results)
    merged = summary.merge(topn_df, on='code', how='left')
    
    # Save
    suffix = f"_rank_{args.rank_by}"
    if args.sig_only:
        suffix += "_sigonly"
    out_path = os.path.join(OUTPUT_DIR, f'code_overlap_topN{suffix}.csv')
    merged.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    
    # === Summary by code_label ===
    print(f"\n{'='*90}")
    for n in top_ns:
        print(f"\n--- Top {n} (Jaccard) ---")
        print(merged.groupby('code_label')[f'jaccard_top{n}'].agg(
            ['count', 'mean', 'std', 'min', 'median', 'max']
        ).round(3).to_string())
    
    print(f"\n{'='*90}")
    for n in top_ns:
        print(f"\n--- Top {n} (Overlap Coefficient) ---")
        print(merged.groupby('code_label')[f'overlap_coef_top{n}'].agg(
            ['count', 'mean', 'std', 'min', 'median', 'max']
        ).round(3).to_string())
    
    # === Quick comparison: threshold-based vs top-N ===
    print(f"\n{'='*90}")
    print("\n=== Median Jaccard: Threshold-based vs Top-N ===")
    header = f"{'Label':>18} | {'Thresh':>7}"
    for n in top_ns:
        header += f" | {'Top'+str(n):>7}"
    print(header)
    print("-" * (30 + 10 * len(top_ns)))
    
    for label in ['Shared', 'CD_enriched', 'HS_enriched']:
        sub = merged[merged['code_label'] == label]
        line = f"{label:>18} | {sub['jaccard'].median():>7.3f}"
        for n in top_ns:
            line += f" | {sub[f'jaccard_top{n}'].median():>7.3f}"
        print(line)
    
    print(f"\n=== Median Overlap Coef: Threshold-based vs Top-N ===")
    header = f"{'Label':>18} | {'Thresh':>7}"
    for n in top_ns:
        header += f" | {'Top'+str(n):>7}"
    print(header)
    print("-" * (30 + 10 * len(top_ns)))
    
    for label in ['Shared', 'CD_enriched', 'HS_enriched']:
        sub = merged[merged['code_label'] == label]
        line = f"{label:>18} | {sub['overlap_coef'].median():>7.3f}"
        for n in top_ns:
            line += f" | {sub[f'overlap_coef_top{n}'].median():>7.3f}"
        print(line)


if __name__ == '__main__':
    main()
