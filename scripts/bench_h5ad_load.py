#!/usr/bin/env python -u
"""
h5ad 로딩 벤치마크 — IO 경합 상황에서 소요 시간 측정
Usage:
    python -u scripts/bench_h5ad_load.py
    python -u scripts/bench_h5ad_load.py --file /path/to/other.h5ad
    python -u scripts/bench_h5ad_load.py --subset   # skin3 subset + toarray 포함
"""
import argparse, time, sys

def ts():
    return time.strftime("%H:%M:%S")

def bench(label, fn):
    print(f"[{ts()}] START  {label}", flush=True)
    t0 = time.time()
    result = fn()
    dt = time.time() - t0
    print(f"[{ts()}] DONE   {label}  ({dt:.1f}s)", flush=True)
    return result, dt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="/home/bmi-user/workspace/data/HSvsCD/scMILDQ_Cond/data/Whole_SCP_PCD_Skin_805k_6k_unified_v2.h5ad")
    parser.add_argument("--subset", action="store_true", help="skin3 subset + toarray 시뮬레이션")
    parser.add_argument("--common", action="store_true", help="common celltype h5ad 로드 테스트")
    args = parser.parse_args()

    import scanpy as sc
    import numpy as np

    timings = {}

    # 1. h5ad 로드
    adata, t = bench("sc.read_h5ad", lambda: sc.read_h5ad(args.file))
    timings["read_h5ad"] = t
    print(f"         shape={adata.shape}, sparse={hasattr(adata.X, 'toarray')}", flush=True)

    if args.common:
        f2 = "/home/bmi-user/workspace/data/HSvsCD/scMILDQ_Cond/data/Whole_SCP_PCD_Skin_common_celltype_6k.h5ad"
        adata2, t = bench("read common h5ad", lambda: sc.read_h5ad(f2))
        timings["read_common"] = t
        print(f"         shape={adata2.shape}", flush=True)

    if args.subset:
        # 2. Skin3 subset
        _, t = bench("subset skin3", lambda: adata[adata.obs["study"].isin(["GSE175990", "GSE220116"])])
        timings["subset_skin3"] = t
        skin = adata[adata.obs["study"].isin(["GSE175990", "GSE220116"])]
        print(f"         skin3 shape={skin.shape}", flush=True)

        # 3. toarray (sparse → dense) — skin3
        if hasattr(skin.X, 'toarray'):
            import torch
            _, t = bench("toarray+tensor skin3", lambda: torch.tensor(skin.X.toarray(), dtype=torch.float32))
            timings["toarray_skin3"] = t

        # 4. SCP1884 subset
        scp = adata[adata.obs["study"] == "SCP1884"]
        print(f"         scp1884 shape={scp.shape}", flush=True)

        if hasattr(scp.X, 'toarray'):
            import torch
            _, t = bench("toarray+tensor scp1884", lambda: torch.tensor(scp.X.toarray(), dtype=torch.float32))
            timings["toarray_scp1884"] = t

    # Summary
    print(f"\n{'='*50}", flush=True)
    print("Summary:", flush=True)
    total = 0
    for k, v in timings.items():
        print(f"  {k:30s}  {v:7.1f}s", flush=True)
        total += v
    print(f"  {'TOTAL':30s}  {total:7.1f}s", flush=True)

if __name__ == "__main__":
    main()
