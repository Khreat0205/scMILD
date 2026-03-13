#!/usr/bin/env python
"""
summarize_sweep.py - Pretrain sweep 결과 요약 및 비교

Usage:
    python scripts/summarize_sweep.py --sweep_dir results/pretrain_sweep
    python scripts/summarize_sweep.py --sweep_dir results/pretrain_sweep --sort ct_loss
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def parse_experiment_name(name: str) -> Dict[str, str]:
    """Parse experiment directory name into components.

    Examples:
        sub_c256_w005 → {data_scope: subset_326k, num_codes: 256, loss_weight: 0.05}
        whole_c512_w010 → {data_scope: whole_805k, num_codes: 512, loss_weight: 0.1}
    """
    parts = name.split("_")
    info = {}

    # Data scope
    if parts[0] == "sub":
        info["data_scope"] = "subset_326k"
    elif parts[0] == "whole":
        info["data_scope"] = "whole_805k"
    else:
        info["data_scope"] = parts[0]

    # Parse remaining parts
    for part in parts[1:]:
        if part.startswith("c") and part[1:].isdigit():
            info["num_codes"] = int(part[1:])
        elif part.startswith("w") and part[1:].isdigit():
            # w005 → 0.05, w010 → 0.1, w020 → 0.2
            info["loss_weight"] = int(part[1:]) / 100.0

    return info


def load_experiment_results(exp_dir: Path) -> Optional[Dict]:
    """Load results from a single experiment directory."""
    # Find the pretrain_* subdirectory
    pretrain_dirs = sorted(exp_dir.glob("pretrain_*"))
    if not pretrain_dirs:
        return None

    # Use the latest one
    run_dir = pretrain_dirs[-1]

    result = {"run_dir": str(run_dir)}

    # Load training history
    history_path = run_dir / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        # Final and best losses
        train_losses = history.get("train_loss", [])
        ct_losses = history.get("celltype_loss", [])
        commit_losses = history.get("commitment_loss", [])

        if train_losses:
            result["final_train_loss"] = train_losses[-1]
            result["best_train_loss"] = min(train_losses)
            result["best_epoch"] = int(np.argmin(train_losses)) + 1
            result["total_epochs"] = len(train_losses)

        if ct_losses:
            result["final_ct_loss"] = ct_losses[-1]
            result["best_ct_loss"] = min(ct_losses)

        if commit_losses:
            result["final_commit_loss"] = commit_losses[-1]

        # Check for NaN
        result["has_nan"] = any(
            np.isnan(v) for v in train_losses + ct_losses + commit_losses
        )
    else:
        return None

    # Check if model was saved
    model_path = run_dir / "vq_aenb_conditional.pth"
    result["model_saved"] = model_path.exists()

    # Check celltype classifier
    ct_path = run_dir / "celltype_classifier.pth"
    result["ct_classifier_saved"] = ct_path.exists()

    return result


def main():
    parser = argparse.ArgumentParser(description="Summarize pretrain sweep results")
    parser.add_argument("--sweep_dir", type=str, required=True,
                        help="Sweep results directory")
    parser.add_argument("--sort", type=str, default="train_loss",
                        choices=["train_loss", "ct_loss", "name", "codes", "weight"],
                        help="Sort by metric")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save summary to CSV file")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Error: {sweep_dir} does not exist")
        sys.exit(1)

    # Collect all experiment directories
    exp_dirs = sorted([
        d for d in sweep_dir.iterdir()
        if d.is_dir() and (d.name.startswith("sub_") or d.name.startswith("whole_"))
    ])

    if not exp_dirs:
        print(f"No experiment directories found in {sweep_dir}")
        sys.exit(1)

    # Load results
    results = []
    for exp_dir in exp_dirs:
        info = parse_experiment_name(exp_dir.name)
        exp_result = load_experiment_results(exp_dir)

        if exp_result is None:
            info["status"] = "NOT_FOUND"
            results.append(info)
            continue

        info.update(exp_result)
        if info.get("has_nan"):
            info["status"] = "NaN"
        elif not info.get("model_saved"):
            info["status"] = "INCOMPLETE"
        else:
            info["status"] = "OK"

        results.append(info)

    # Sort
    def sort_key(r):
        if args.sort == "train_loss":
            return r.get("best_train_loss", float("inf"))
        elif args.sort == "ct_loss":
            return r.get("best_ct_loss", float("inf"))
        elif args.sort == "codes":
            return r.get("num_codes", 0)
        elif args.sort == "weight":
            return r.get("loss_weight", 0)
        else:
            return r.get("data_scope", "") + str(r.get("num_codes", 0))

    results.sort(key=sort_key)

    # Print summary table
    print("")
    print("=" * 100)
    print("  Pretrain Sweep Summary")
    print("=" * 100)
    print("")

    # Header
    header = f"{'Experiment':<25} {'Status':<10} {'Codes':>5} {'Weight':>6} {'BestLoss':>9} {'CTLoss':>9} {'Epoch':>6} {'Total':>6}"
    print(header)
    print("-" * 100)

    # Group by data_scope
    for scope in ["subset_326k", "whole_805k"]:
        scope_results = [r for r in results if r.get("data_scope") == scope]
        if not scope_results:
            continue

        print(f"\n  [{scope}]")
        for r in scope_results:
            name = f"{r['data_scope'][:3]}_c{r.get('num_codes', '?')}_w{r.get('loss_weight', '?')}"
            status = r.get("status", "?")
            codes = r.get("num_codes", "")
            weight = r.get("loss_weight", "")
            best_loss = r.get("best_train_loss", "")
            ct_loss = r.get("best_ct_loss", "")
            best_epoch = r.get("best_epoch", "")
            total_epochs = r.get("total_epochs", "")

            best_loss_str = f"{best_loss:.4f}" if isinstance(best_loss, float) else str(best_loss)
            ct_loss_str = f"{ct_loss:.4f}" if isinstance(ct_loss, float) else str(ct_loss)
            weight_str = f"{weight:.2f}" if isinstance(weight, float) else str(weight)

            status_mark = {"OK": "  OK", "NaN": " NaN", "NOT_FOUND": "  --", "INCOMPLETE": " INC"}.get(status, "  ??")

            print(f"  {name:<23} {status_mark:<10} {codes:>5} {weight_str:>6} {best_loss_str:>9} {ct_loss_str:>9} {best_epoch:>6} {total_epochs:>6}")

    print("")
    print("=" * 100)

    # Statistics
    ok_results = [r for r in results if r.get("status") == "OK"]
    nan_results = [r for r in results if r.get("status") == "NaN"]
    missing_results = [r for r in results if r.get("status") == "NOT_FOUND"]

    print(f"  Completed: {len(ok_results)}  |  NaN: {len(nan_results)}  |  Not found: {len(missing_results)}  |  Total: {len(results)}")

    if ok_results:
        best = min(ok_results, key=lambda r: r.get("best_train_loss", float("inf")))
        best_name = f"{best['data_scope'][:3]}_c{best.get('num_codes')}_w{best.get('loss_weight')}"
        print(f"\n  Best train loss: {best_name} → {best['best_train_loss']:.4f} (epoch {best['best_epoch']})")

        # Best by CT loss
        ct_results = [r for r in ok_results if r.get("best_ct_loss") is not None and r.get("best_ct_loss", 0) > 0]
        if ct_results:
            best_ct = min(ct_results, key=lambda r: r.get("best_ct_loss", float("inf")))
            best_ct_name = f"{best_ct['data_scope'][:3]}_c{best_ct.get('num_codes')}_w{best_ct.get('loss_weight')}"
            print(f"  Best CT loss:    {best_ct_name} → {best_ct['best_ct_loss']:.4f}")

    print("")
    print("=" * 100)

    # Save to CSV if requested
    if args.csv:
        import csv
        csv_path = Path(args.csv)
        fieldnames = [
            "data_scope", "num_codes", "loss_weight", "status",
            "best_train_loss", "final_train_loss", "best_ct_loss", "final_ct_loss",
            "final_commit_loss", "best_epoch", "total_epochs",
            "has_nan", "model_saved", "run_dir"
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        print(f"\n  CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
