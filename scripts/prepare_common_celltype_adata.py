#!/usr/bin/env python
"""
Common cell type subset adata 생성 스크립트
===========================================
1. SCP1884 + Skin3 adata 로드
2. celltype_unified 기준으로 unified_v2에 매핑되는 cell type만 필터링
3. 두 adata concat
4. HVG 6k 선택 (raw count 보존)
5. 저장

Usage:
    python scripts/prepare_common_celltype_adata.py
    python scripts/prepare_common_celltype_adata.py --n_top_genes 6000 --dry_run
"""

import argparse
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# celltype_unified → unified_v2 매핑
# "—" = 제외 대상
# ============================================================================
UNIFIED_TO_V2 = {
    "T cells CD4": "T cells CD4",
    "T cells CD8": "T cells CD8",
    "Tregs": "Tregs",
    "NK cells": "NK cells",
    "ILCs": "ILCs",
    "B cells": "B cells",
    "Plasma cells": "Plasma cells",
    "Macrophages": "Macrophages",
    "Monocytes": "Monocytes",
    "DC2": "DC",
    "MigDC": "DC",
    "cDC1": "DC",
    "pDC": "DC",
    "Mast cells": "Mast cells",
    "Fibroblasts": "Fibroblasts",
    "Myofibroblasts": "Fibroblasts",
    "Endothelial cells": "Endothelial cells",
    "Lymphatic endothelial": "Lymphatic endothelial",
    "Pericytes": "Pericytes",
    "Keratinocyte": "Epithelial cells",
    "Enterocytes": "Epithelial cells",
}

# 제외되는 cell types (unified_v2 = "—")
EXCLUDED = [
    "Melanocyte", "Stem cells", "Goblet cells", "Enteroendocrine",
    "Tuft cells", "Paneth cells", "Epithelial cycling",
    "Glial cells", "Immune cycling", "Low_quality",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scp1884", type=str,
                        default="/home/bmi-user/workspace/data/HSvsCD/scMILDQ_Cond/data/adata_SCP1884_18kGene_unified_2603_v2.h5ad")
    parser.add_argument("--skin3", type=str,
                        default="/home/bmi-user/workspace/data/HSvsCD/scMILDQ_Cond/data/adata_Skin_18kGene_unified_2603_v2.h5ad")
    parser.add_argument("--output_dir", type=str,
                        default="/home/bmi-user/workspace/data/HSvsCD/scMILDQ_Cond/data")
    parser.add_argument("--n_top_genes", type=int, default=6000)
    parser.add_argument("--celltype_col", type=str, default="celltype_unified")
    parser.add_argument("--dry_run", action="store_true",
                        help="필터링 통계만 출력하고 저장하지 않음")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ct_col = args.celltype_col

    # --- 1. Load ---
    print("=" * 60)
    print("1. Loading adata files")
    print("=" * 60)
    adata_scp = sc.read_h5ad(args.scp1884)
    adata_skin = sc.read_h5ad(args.skin3)
    print(f"  SCP1884: {adata_scp.shape}")
    print(f"  Skin3:   {adata_skin.shape}")

    # --- 2. 필터링 전 통계 ---
    print()
    print("=" * 60)
    print("2. Cell type distribution BEFORE filtering")
    print("=" * 60)

    valid_types = set(UNIFIED_TO_V2.keys())

    for name, ad in [("SCP1884", adata_scp), ("Skin3", adata_skin)]:
        print(f"\n--- {name} ({ct_col}) ---")
        vc = ad.obs[ct_col].astype(str).value_counts()
        for ct, n in vc.items():
            status = "KEEP" if ct in valid_types else "DROP"
            v2 = UNIFIED_TO_V2.get(ct, "—")
            print(f"  {status}  {ct:30s} → {v2:25s}  ({n:,} cells)")

        mask = ad.obs[ct_col].astype(str).isin(valid_types)
        print(f"  Total: {len(ad):,} → Keep: {mask.sum():,} ({mask.mean()*100:.1f}%) | Drop: {(~mask).sum():,}")

    # --- 3. 필터링 ---
    print()
    print("=" * 60)
    print("3. Filtering to common cell types")
    print("=" * 60)

    mask_scp = adata_scp.obs[ct_col].astype(str).isin(valid_types)
    mask_skin = adata_skin.obs[ct_col].astype(str).isin(valid_types)

    adata_scp_f = adata_scp[mask_scp].copy()
    adata_skin_f = adata_skin[mask_skin].copy()

    print(f"  SCP1884: {adata_scp.shape[0]:,} → {adata_scp_f.shape[0]:,}")
    print(f"  Skin3:   {adata_skin.shape[0]:,} → {adata_skin_f.shape[0]:,}")

    # unified_v2 컬럼 추가
    adata_scp_f.obs["celltype_unified_v2"] = (
        adata_scp_f.obs[ct_col].astype(str).map(UNIFIED_TO_V2).astype("category")
    )
    adata_skin_f.obs["celltype_unified_v2"] = (
        adata_skin_f.obs[ct_col].astype(str).map(UNIFIED_TO_V2).astype("category")
    )

    # --- 4. Concat ---
    print()
    print("=" * 60)
    print("4. Concatenating")
    print("=" * 60)

    # 공통 obs 컬럼만 유지
    common_obs_cols = sorted(
        set(adata_scp_f.obs.columns) & set(adata_skin_f.obs.columns)
    )
    adata_scp_f.obs = adata_scp_f.obs[common_obs_cols]
    adata_skin_f.obs = adata_skin_f.obs[common_obs_cols]

    # 공통 유전자 (이미 같은 18k gene set일 것이나 안전하게)
    common_genes = sorted(set(adata_scp_f.var_names) & set(adata_skin_f.var_names))
    print(f"  Common genes: {len(common_genes)}")
    adata_scp_f = adata_scp_f[:, common_genes]
    adata_skin_f = adata_skin_f[:, common_genes]

    adata = sc.concat([adata_scp_f, adata_skin_f], merge="same")
    print(f"  Merged: {adata.shape}")

    # --- 5. unified_v2 분포 ---
    print()
    print("=" * 60)
    print("5. unified_v2 cell type distribution (merged)")
    print("=" * 60)
    vc2 = adata.obs["celltype_unified_v2"].value_counts()
    for ct, n in vc2.items():
        print(f"  {ct:30s}  {n:>8,}")
    print(f"  {'TOTAL':30s}  {len(adata):>8,}")

    # sample 수 확인
    n_samples = adata.obs["sample"].nunique()
    print(f"\n  Unique samples: {n_samples}")

    # study별 sample/cell 수
    print("\n  Per-study breakdown:")
    for study, grp in adata.obs.groupby("study"):
        n_s = grp["sample"].nunique()
        print(f"    {study}: {n_s} samples, {len(grp):,} cells")

    if args.dry_run:
        print("\n[DRY RUN] Stopping here. No files saved.")
        return

    # --- 6. HVG selection + save ---
    print()
    print("=" * 60)
    print(f"6. HVG selection (n_top_genes={args.n_top_genes})")
    print("=" * 60)

    # raw count 보존
    adata.layers["raw_counts"] = adata.X.copy()

    # normalize → log → HVG
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_top_genes, flavor="seurat_v3",
                                layer="raw_counts")
    n_hvg = adata.var["highly_variable"].sum()
    print(f"  HVGs selected: {n_hvg}")

    # HVG subset
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    # raw count를 X로 복원
    adata_hvg.X = adata_hvg.layers["raw_counts"].copy()
    del adata_hvg.layers["raw_counts"]

    print(f"  Final shape: {adata_hvg.shape}")

    # --- 7. Save ---
    out_name = f"Whole_SCP_PCD_Skin_common_celltype_{args.n_top_genes // 1000}k.h5ad"
    out_path = output_dir / out_name
    print()
    print(f"  Saving to: {out_path}")
    adata_hvg.write_h5ad(out_path)
    print(f"  Done! File size: {out_path.stat().st_size / 1024**2:.1f} MB")

    # 요약
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Original cells: SCP1884={adata_scp.shape[0]:,} + Skin3={adata_skin.shape[0]:,} = {adata_scp.shape[0]+adata_skin.shape[0]:,}")
    print(f"  Filtered cells: {adata_hvg.shape[0]:,} ({adata_hvg.shape[0]/(adata_scp.shape[0]+adata_skin.shape[0])*100:.1f}%)")
    print(f"  Genes: {adata_scp.shape[1]:,} → {adata_hvg.shape[1]:,} (HVG)")
    print(f"  Cell types (unified_v2): {adata_hvg.obs['celltype_unified_v2'].nunique()}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
