#!/usr/bin/env python
"""
07_multi_model_scoring.py - 다중 모델 Cell-level 점수 계산 스크립트

복수의 학습된 final model을 사용하여 각 세포의 attention score와 prediction을 계산합니다.
동일한 pretrained encoder를 공유하는 여러 모델(예: CD, HS)의 score를
하나의 AnnData에 suffix 구분으로 함께 저장합니다.

출력:
    - scored_adata.h5ad: Cell-level scoring 결과가 추가된 AnnData (모델별 suffix)
    - codebook_adata.h5ad: Codebook-level 통계 AnnData (모델별 suffix)
    - cell_scores.csv: CSV 형식 백업

Usage:
    # 두 모델로 SCP1884 subset scoring
    python scripts/07_multi_model_scoring.py \\
        --models CD:results/scp1884_skfold_512/final_model_xxx:config/scp1884.yaml \\
                 HS:results/skin3_skfold_512/final_model_xxx:config/skin3.yaml \\
        --data_config config/scp1884.yaml \\
        --output_dir results/multi_scores/scp1884/ \\
        --gpu 0

    # 임의의 adata에 두 모델 scoring
    python scripts/07_multi_model_scoring.py \\
        --models CD:results/.../final_model_xxx:config/scp1884.yaml \\
                 HS:results/.../final_model_xxx:config/skin3.yaml \\
        --data_path /path/to/custom.h5ad \\
        --data_config config/default.yaml \\
        --output_dir results/multi_scores/custom/ \\
        --gpu 0
"""

import sys
import argparse
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple, NamedTuple

import torch
torch.set_num_threads(16)
import numpy as np
import pandas as pd
import scanpy as sc

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, ScMILDConfig
from src.data import load_adata_with_subset, print_adata_summary, load_conditional_mapping

# Import reusable functions from 05_cell_scoring.py
# (guarded by if __name__ == "__main__", so import is safe)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "cell_scoring", PROJECT_ROOT / "scripts" / "05_cell_scoring.py"
)
_scoring = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scoring)

load_pretrained_encoder = _scoring.load_pretrained_encoder
load_trained_models = _scoring.load_trained_models
compute_codebook_direct_attention = _scoring.compute_codebook_direct_attention
ensure_embedding_column = _scoring.ensure_embedding_column


# ============================================================================
# Helper Functions
# ============================================================================

@torch.no_grad()
def compute_codebook_direct_student(
    model_encoder, model_student, device: torch.device
) -> np.ndarray:
    """Compute student predictions by passing codebook directly through student branch.

    codebook → projection → student → softmax[:, 1]
    """
    model_encoder.eval()
    model_student.eval()

    codebook = model_encoder.vq_model.quantizer.get_codebook().to(device)
    if model_encoder.projection is not None:
        codebook_projected = model_encoder.projection(codebook)
    else:
        codebook_projected = codebook

    student_out = model_student(codebook_projected)
    student_probs = torch.softmax(student_out, dim=1)[:, 1]
    return student_probs.cpu().numpy()


def extract_data_name(
    config_path: str, cli_data_name: Optional[str] = None
) -> str:
    """Extract data.info.name from raw YAML config, with CLI override."""
    if cli_data_name:
        return cli_data_name

    import yaml as _yaml
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = _yaml.safe_load(f)

    # _base_ 상속 처리
    if "_base_" in raw:
        base_path = config_path.parent / raw["_base_"]
        with open(base_path, "r", encoding="utf-8") as f:
            base = _yaml.safe_load(f)
        del raw["_base_"]
        # deep merge: base에 raw overlay
        def _deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    _deep_update(d[k], v)
                else:
                    d[k] = v
            return d
        raw = _deep_update(base, raw)

    name = raw.get("data", {}).get("info", {}).get("name")
    if name:
        return name

    # fallback: subset values에서 추론
    subset_vals = raw.get("data", {}).get("subset", {}).get("values", [])
    if subset_vals:
        return "_".join(subset_vals)

    return "unknown"


# ============================================================================
# Data Types
# ============================================================================

class ModelSpec(NamedTuple):
    suffix: str
    model_dir: Path
    config_path: str


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_model_spec(spec_str: str) -> ModelSpec:
    """Parse 'SUFFIX:MODEL_DIR:CONFIG' format."""
    parts = spec_str.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid model spec: '{spec_str}'. "
            f"Expected format: SUFFIX:MODEL_DIR:CONFIG_PATH"
        )
    return ModelSpec(
        suffix=parts[0],
        model_dir=Path(parts[1]),
        config_path=parts[2]
    )


# ============================================================================
# Shared Encoder Functions
# ============================================================================

@torch.no_grad()
def encode_all_cells_shared(
    adata: sc.AnnData,
    encoder_model,
    config: ScMILDConfig,
    device: torch.device,
    batch_size: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    VQ encode all cells using the shared pretrained encoder (without projection).

    Returns:
        X_pretrained: Pre-projection embeddings (n_cells, latent_dim)
        vq_codes: Codebook indices (n_cells,)
    """
    embedding_col = config.data.conditional_embedding.encoded_column
    n_cells = adata.n_obs
    n_batches = (n_cells + batch_size - 1) // batch_size

    all_X_pretrained = []
    all_vq_codes = []

    encoder_model.eval()
    print(f"Encoding {n_cells} cells in {n_batches} batches (shared encoder)...")

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_cells)

        batch_adata = adata[start_idx:end_idx]

        if hasattr(batch_adata.X, 'toarray'):
            data = torch.tensor(batch_adata.X.toarray(), dtype=torch.float32).to(device)
        else:
            data = torch.tensor(np.array(batch_adata.X), dtype=torch.float32).to(device)

        if embedding_col in batch_adata.obs.columns:
            embedding_ids = torch.tensor(
                batch_adata.obs[embedding_col].values.astype(int),
                dtype=torch.long
            ).to(device)
        else:
            embedding_ids = torch.zeros(data.shape[0], dtype=torch.long, device=device)

        # Pre-projection features (quantized latent)
        X_pretrained = encoder_model.features(data, embedding_ids)
        vq_codes = encoder_model.get_codebook_indices(data, embedding_ids)

        all_X_pretrained.append(X_pretrained.cpu().numpy())
        all_vq_codes.append(vq_codes.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Encoded {end_idx}/{n_cells} cells ({100*end_idx/n_cells:.1f}%)")

    return np.concatenate(all_X_pretrained), np.concatenate(all_vq_codes)


@torch.no_grad()
def compute_model_specific_scores(
    X_pretrained: np.ndarray,
    model_encoder,
    model_teacher,
    model_student,
    device: torch.device,
    batch_size: int = 10000,
) -> Dict[str, np.ndarray]:
    """
    Compute model-specific scores using pre-computed X_pretrained.

    Only applies the model-specific projection layer, teacher attention,
    and student prediction (skips the expensive VQ encoding).

    Returns dict with: X_scmild, attention_score_raw, student_prediction
    """
    model_encoder.eval()
    model_teacher.eval()
    model_student.eval()

    n_cells = X_pretrained.shape[0]
    n_batches = (n_cells + batch_size - 1) // batch_size

    all_X_scmild = []
    all_attn_raw = []
    all_student_pred = []

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_cells)

        X_pre_batch = torch.tensor(
            X_pretrained[start_idx:end_idx], dtype=torch.float32, device=device
        )

        # Apply model-specific projection
        if model_encoder.projection is not None:
            X_scmild = model_encoder.projection(X_pre_batch)
        else:
            X_scmild = X_pre_batch

        # Teacher attention
        attn_scores = model_teacher.attention_module(X_scmild).squeeze()

        # Student prediction
        student_out = model_student(X_scmild)
        student_probs = torch.softmax(student_out, dim=1)[:, 1]

        all_X_scmild.append(X_scmild.cpu().numpy())
        all_attn_raw.append(attn_scores.cpu().numpy())
        all_student_pred.append(student_probs.cpu().numpy())

    return {
        'X_scmild': np.concatenate(all_X_scmild),
        'attention_score_raw': np.concatenate(all_attn_raw),
        'student_prediction': np.concatenate(all_student_pred),
    }


# ============================================================================
# Output Building Functions
# ============================================================================

def build_multi_model_cell_adata(
    adata: sc.AnnData,
    X_pretrained: np.ndarray,
    vq_codes: np.ndarray,
    codebook: np.ndarray,
    model_results: Dict[str, Dict[str, np.ndarray]],
    sample_col: str,
    model_infos: Dict[str, dict],
) -> sc.AnnData:
    """Build cell-level AnnData with multi-model scores."""
    scored_adata = adata.copy()

    # Shared columns
    scored_adata.obs['vq_code'] = vq_codes.astype(int)
    scored_adata.obsm['X_pretrained'] = X_pretrained.astype(np.float32)
    scored_adata.uns['codebook'] = codebook.astype(np.float32)
    scored_adata.uns['model_info'] = model_infos

    # Per-model columns
    for suffix, results in model_results.items():
        scored_adata.obs[f'attn_raw_{suffix}'] = results['attention_score_raw']
        scored_adata.obs[f'student_prediction_{suffix}'] = results['student_prediction']
        scored_adata.obsm[f'X_scmild_{suffix}'] = results['X_scmild'].astype(np.float32)

    return scored_adata


def build_multi_model_codebook_adata(
    codebook: np.ndarray,
    scored_adata: sc.AnnData,
    model_results: Dict[str, Dict[str, np.ndarray]],
    sample_col: str,
    data_name: str,
    label_col: Optional[str] = None,
) -> sc.AnnData:
    """Build codebook-level AnnData with multi-model statistics.

    컬럼 네이밍은 cell-level adata와 통일:
    - attn_raw_{suffix}: codebook 직접 통과 attention (raw logit)
    - student_prediction_{suffix}: codebook 직접 통과 student prediction
    - n_cells_{data_name}, n_samples_{data_name}, disease_ratio_{data_name}: 데이터셋별
    """
    num_codes = codebook.shape[0]

    adata_cb = sc.AnnData(X=codebook.astype(np.float32))
    adata_cb.obs_names = [f"code_{i}" for i in range(num_codes)]
    adata_cb.obs['code_idx'] = list(range(num_codes))

    vq_codes = scored_adata.obs['vq_code'].values
    sample_ids = scored_adata.obs[sample_col].values

    # Dataset-specific code statistics
    adata_cb.obs[f'n_cells_{data_name}'] = 0
    adata_cb.obs[f'n_samples_{data_name}'] = 0

    if label_col and label_col in scored_adata.obs.columns:
        adata_cb.obs[f'disease_ratio_{data_name}'] = np.nan
        disease_labels = scored_adata.obs[label_col].values
    else:
        disease_labels = None

    # Per-model codebook columns: direct scores
    for suffix in model_results:
        adata_cb.obs[f'attn_raw_{suffix}'] = model_results[suffix]['attn_direct']
        adata_cb.obs[f'student_prediction_{suffix}'] = model_results[suffix]['student_direct']

    # Compute per-code statistics
    print("Computing codebook statistics...")
    for code_idx in range(num_codes):
        code_name = f"code_{code_idx}"
        mask = vq_codes == code_idx
        n_cells = mask.sum()

        adata_cb.obs.loc[code_name, f'n_cells_{data_name}'] = n_cells
        adata_cb.obs.loc[code_name, f'n_samples_{data_name}'] = (
            len(np.unique(sample_ids[mask])) if n_cells > 0 else 0
        )

        if n_cells > 0:
            if disease_labels is not None:
                adata_cb.obs.loc[code_name, f'disease_ratio_{data_name}'] = disease_labels[mask].mean()

    # Convert dtypes
    adata_cb.obs['code_idx'] = adata_cb.obs['code_idx'].astype(int)
    adata_cb.obs[f'n_cells_{data_name}'] = adata_cb.obs[f'n_cells_{data_name}'].astype(int)
    adata_cb.obs[f'n_samples_{data_name}'] = adata_cb.obs[f'n_samples_{data_name}'].astype(int)

    return adata_cb


# ============================================================================
# CSV Building
# ============================================================================

def build_cell_scores_csv(
    adata: sc.AnnData,
    vq_codes: np.ndarray,
    model_results: Dict[str, Dict[str, np.ndarray]],
    sample_col: str,
    sample_name_col: str,
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    """Build cell_scores CSV DataFrame."""
    df = pd.DataFrame({
        'cell_id': adata.obs.index,
        'sample_id': adata.obs[sample_col].values,
        'sample_name': adata.obs[sample_name_col].values,
        'vq_code': vq_codes,
    })

    if label_col and label_col in adata.obs.columns:
        df['disease_label'] = adata.obs[label_col].values

    for suffix, results in model_results.items():
        df[f'attn_raw_{suffix}'] = results['attention_score_raw']
        df[f'student_prediction_{suffix}'] = results['student_prediction']

    return df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model cell scoring for scMILD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Model specs in SUFFIX:MODEL_DIR:CONFIG format. "
             "Example: CD:results/final_model_xxx:config/scp1884.yaml"
    )
    parser.add_argument(
        "--data_config", type=str, required=True,
        help="Config for data loading (columns, conditional embedding, subset settings)"
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Override data path (instead of config's whole_adata_path)"
    )
    parser.add_argument(
        "--data_name", type=str, default=None,
        help="Dataset name for codebook statistics suffix "
             "(default: auto-detect from data_config's data.info.name)"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size")
    parser.add_argument(
        "--no_codebook_adata", action="store_true",
        help="Skip saving codebook AnnData"
    )

    args = parser.parse_args()

    # Parse model specs
    model_specs = [parse_model_spec(s) for s in args.models]
    suffixes = [m.suffix for m in model_specs]
    if len(set(suffixes)) != len(suffixes):
        raise ValueError(f"Duplicate model suffixes: {suffixes}")

    # Extract data_name
    data_name = extract_data_name(args.data_config, args.data_name)

    print(f"Models to score ({len(model_specs)}):")
    for ms in model_specs:
        print(f"  [{ms.suffix}] dir={ms.model_dir}, config={ms.config_path}")
    print(f"Data name: {data_name}")

    # Setup
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data config
    data_config = load_config(args.data_config)

    # Load data
    print("\nLoading data...")
    if args.data_path:
        adata = sc.read_h5ad(args.data_path)
        print(f"Loaded from --data_path: {args.data_path}")
    else:
        adata = load_adata_with_subset(
            whole_adata_path=data_config.data.whole_adata_path,
            subset_enabled=data_config.data.subset.enabled,
            subset_column=data_config.data.subset.column,
            subset_values=data_config.data.subset.values,
            cache_dir=data_config.data.subset.cache_dir,
            use_cache=data_config.data.subset.use_cache,
        )
    print_adata_summary(adata, "Loaded Data")

    # Ensure embedding column
    adata = ensure_embedding_column(adata, data_config)

    # ---------------------------------------------------------------
    # Step 1: Shared VQ encoding (once)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 1: Shared VQ Encoding")
    print("=" * 60)

    # Load pretrained encoder from first model's config (shared)
    first_config = load_config(model_specs[0].config_path)
    encoder_model, model_config = load_pretrained_encoder(first_config, device)

    X_pretrained, vq_codes = encode_all_cells_shared(
        adata, encoder_model, data_config, device, args.batch_size
    )
    codebook = encoder_model.quantizer.get_codebook().cpu().numpy()
    print(f"  X_pretrained: {X_pretrained.shape}")
    print(f"  vq_codes: {vq_codes.shape}, unique: {len(np.unique(vq_codes))}")
    print(f"  codebook: {codebook.shape}")

    del model_config
    gc.collect()

    # ---------------------------------------------------------------
    # Step 2: Per-model scoring
    # ---------------------------------------------------------------
    model_results = {}  # suffix -> dict of arrays
    model_infos = {}

    for ms in model_specs:
        print(f"\n{'=' * 60}")
        print(f"Step 2: Scoring with model [{ms.suffix}]")
        print(f"  model_dir: {ms.model_dir}")
        print(f"  config: {ms.config_path}")
        print(f"{'=' * 60}")

        model_config = load_config(ms.config_path)

        # Load trained models (encoder wrapper + teacher + student)
        model_teacher, model_student, model_encoder = load_trained_models(
            ms.model_dir, device, model_config, fold_idx=0
        )

        # Compute model-specific scores using pre-computed X_pretrained
        results = compute_model_specific_scores(
            X_pretrained, model_encoder, model_teacher, model_student,
            device, args.batch_size
        )

        # Codebook direct scores
        attn_direct = compute_codebook_direct_attention(model_encoder, model_teacher, device)
        student_direct = compute_codebook_direct_student(model_encoder, model_student, device)
        results['attn_direct'] = attn_direct
        results['student_direct'] = student_direct

        model_results[ms.suffix] = results

        model_infos[ms.suffix] = {
            'model_dir': str(ms.model_dir),
            'config_path': ms.config_path,
            'n_cells': X_pretrained.shape[0],
            'n_codes': codebook.shape[0],
            'latent_dim': codebook.shape[1],
        }

        print(f"  attn_raw range: [{results['attention_score_raw'].min():.4f}, {results['attention_score_raw'].max():.4f}]")
        print(f"  student_prediction range: [{results['student_prediction'].min():.4f}, {results['student_prediction'].max():.4f}]")
        print(f"  codebook attn_direct range: [{attn_direct.min():.4f}, {attn_direct.max():.4f}]")
        print(f"  codebook student_direct range: [{student_direct.min():.4f}, {student_direct.max():.4f}]")

        # Cleanup
        del model_teacher, model_student, model_encoder, model_config
        gc.collect()
        torch.cuda.empty_cache()

    # Free shared encoder
    del encoder_model
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Step 3: Build and save outputs
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Step 3: Saving results")
    print(f"{'=' * 60}")

    sample_col = data_config.data.columns.sample_id
    sample_name_col = data_config.data.columns.sample_name
    label_col = data_config.data.columns.disease_label

    # Metadata
    timestamp = datetime.now().isoformat()
    combined_info = {
        'timestamp': timestamp,
        'data_config': args.data_config,
        'data_path': args.data_path,
        'data_name': data_name,
        'models': model_infos,
    }

    # Cell-level AnnData
    scored_adata = build_multi_model_cell_adata(
        adata, X_pretrained, vq_codes, codebook,
        model_results, sample_col, combined_info
    )
    scored_adata.write_h5ad(output_dir / "scored_adata.h5ad")
    print(f"  Saved: scored_adata.h5ad ({scored_adata.n_obs} cells)")

    # Codebook AnnData
    if not args.no_codebook_adata:
        codebook_adata = build_multi_model_codebook_adata(
            codebook, scored_adata, model_results,
            sample_col, data_name, label_col
        )
        codebook_adata.write_h5ad(output_dir / "codebook_adata.h5ad")
        print(f"  Saved: codebook_adata.h5ad ({codebook_adata.n_obs} codes)")

    # CSV
    cell_scores_df = build_cell_scores_csv(
        adata, vq_codes, model_results,
        sample_col, sample_name_col, label_col
    )
    cell_scores_df.to_csv(output_dir / "cell_scores.csv", index=False)
    print(f"  Saved: cell_scores.csv")

    # Model info JSON
    with open(output_dir / "model_info.json", 'w') as f:
        json.dump(combined_info, f, indent=2)
    print(f"  Saved: model_info.json")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Multi-Model Scoring Complete!")
    print(f"{'=' * 60}")

    # Per-model sample summary
    for suffix in model_results:
        attn_col = f'attn_raw_{suffix}'
        student_col = f'student_prediction_{suffix}'

        agg_dict = {
            attn_col: ['mean', 'std', 'max'],
            student_col: ['mean', 'std'],
        }

        group_cols = [sample_name_col]
        if label_col in scored_adata.obs.columns:
            group_cols.append(label_col)

        summary = scored_adata.obs.groupby(group_cols, observed=True).agg(agg_dict).round(4)
        summary.columns = ['attn_mean', 'attn_std', 'attn_max', 'student_mean', 'student_std']
        summary = summary.reset_index()

        print(f"\n[{suffix}] Sample-level summary:")
        print(summary.to_string(index=False))

        summary.to_csv(output_dir / f"sample_summary_{suffix}.csv", index=False)

    # Codebook usage
    vq_code_counts = pd.Series(vq_codes).value_counts()
    print(f"\nCodebook usage:")
    print(f"  Active codes: {len(vq_code_counts)} / {codebook.shape[0]}")
    print(f"  Most used: {vq_code_counts.head(5).to_dict()}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
