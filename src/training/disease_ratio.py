"""
Disease Ratio Regularization for scMILD.

VQ 코드북의 각 코드별 질병 비율을 계산하고, 이를 attention score의 타겟으로 사용합니다.
"""

import torch
import numpy as np
from typing import Dict, Optional
from collections import defaultdict
from torch.utils.data import DataLoader


def calculate_disease_ratio_from_dataloader(
    dataloader: DataLoader,
    model_encoder,
    device: torch.device,
    alpha: float = 1.0,
    beta: float = 1.0,
    use_conditional: bool = True
) -> Optional[Dict[int, float]]:
    """
    Training 데이터로더에서 VQ 코드별 질병 비율을 계산합니다.

    Bayesian smoothing을 사용하여 데이터가 적은 코드에도 안정적인 비율을 제공합니다.
    smoothed_ratio = (n_case + alpha) / (n_total + alpha + beta)

    Args:
        dataloader: Training instance dataloader
        model_encoder: VQ encoder (get_codebook_indices 메서드 필요)
        device: torch device
        alpha, beta: Beta prior parameters (기본값: 균일 prior)
        use_conditional: Conditional encoder 사용 여부

    Returns:
        dict: {code_id: smoothed_ratio} 또는 None (코드북 인덱스를 얻을 수 없는 경우)
    """
    # Check if encoder has codebook
    if not hasattr(model_encoder, 'vq_model') and not hasattr(model_encoder, 'get_codebook_indices'):
        print("WARNING: Encoder does not have VQ codebook, disease ratio calculation skipped")
        return None

    code_case_count = defaultdict(int)
    code_total_count = defaultdict(int)

    model_encoder.eval()

    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch - InstanceDatasetWithBagLabel 형식
            # (data, bag_id, instance_label, bag_label, [study_id])
            data = batch[0].to(device)

            # Instance label (disease label for each cell)
            if len(batch) >= 4:
                instance_labels = batch[2].to(device)  # instance_label
            else:
                continue

            # Study IDs for conditional encoder
            study_ids = None
            if use_conditional and len(batch) == 5:
                study_ids = batch[4].to(device)

            # Get codebook indices
            try:
                if hasattr(model_encoder, 'vq_model'):
                    if study_ids is not None:
                        code_indices = model_encoder.vq_model.get_codebook_indices(data, study_ids)
                    else:
                        code_indices = model_encoder.vq_model.get_codebook_indices(data)
                elif hasattr(model_encoder, 'get_codebook_indices'):
                    if study_ids is not None:
                        code_indices = model_encoder.get_codebook_indices(data, study_ids)
                    else:
                        code_indices = model_encoder.get_codebook_indices(data)
                else:
                    continue
            except Exception as e:
                print(f"WARNING: Failed to get codebook indices: {e}")
                continue

            # Count per code
            for code, label in zip(code_indices.cpu().numpy(), instance_labels.cpu().numpy()):
                code_idx = int(code)
                code_total_count[code_idx] += 1
                if label == 1:  # Disease class
                    code_case_count[code_idx] += 1

    if not code_total_count:
        print("WARNING: No codes found, returning None")
        return None

    # Bayesian smoothing
    smoothed_ratio = {}
    for code in code_total_count:
        n_case = code_case_count[code]
        n_total = code_total_count[code]
        smoothed_ratio[code] = (n_case + alpha) / (n_total + alpha + beta)

    return smoothed_ratio


def calculate_disease_ratio_from_adata(
    adata,
    model_encoder,
    device: torch.device,
    sample_col: str = "sample_id_numeric",
    label_col: str = "disease_numeric",
    study_col: str = "study_id_numeric",
    alpha: float = 1.0,
    beta: float = 1.0,
    batch_size: int = 10000,
    use_conditional: bool = True
) -> Optional[Dict[int, float]]:
    """
    AnnData에서 직접 VQ 코드별 질병 비율을 계산합니다.

    대용량 데이터셋의 경우 배치 단위로 처리합니다.

    Args:
        adata: AnnData object
        model_encoder: VQ encoder
        device: torch device
        sample_col: Sample ID column
        label_col: Disease label column
        study_col: Study ID column (conditional encoder용)
        alpha, beta: Beta prior parameters
        batch_size: Processing batch size
        use_conditional: Conditional encoder 사용 여부

    Returns:
        dict: {code_id: smoothed_ratio}
    """
    import numpy as np

    if not hasattr(model_encoder, 'vq_model') and not hasattr(model_encoder, 'get_codebook_indices'):
        print("WARNING: Encoder does not have VQ codebook")
        return None

    code_case_count = defaultdict(int)
    code_total_count = defaultdict(int)

    model_encoder.eval()
    n_cells = adata.n_obs

    with torch.no_grad():
        for start_idx in range(0, n_cells, batch_size):
            end_idx = min(start_idx + batch_size, n_cells)
            batch_adata = adata[start_idx:end_idx]

            # Extract data
            if hasattr(batch_adata.X, 'toarray'):
                data = torch.tensor(batch_adata.X.toarray(), dtype=torch.float32, device=device)
            else:
                data = torch.tensor(np.array(batch_adata.X), dtype=torch.float32, device=device)

            labels = batch_adata.obs[label_col].values

            # Study IDs
            study_ids = None
            if use_conditional and study_col in batch_adata.obs.columns:
                study_ids = torch.tensor(
                    batch_adata.obs[study_col].values.astype(int),
                    dtype=torch.long, device=device
                )

            # Get codebook indices
            try:
                if hasattr(model_encoder, 'vq_model'):
                    if study_ids is not None:
                        code_indices = model_encoder.vq_model.get_codebook_indices(data, study_ids)
                    else:
                        code_indices = model_encoder.vq_model.get_codebook_indices(data)
                else:
                    if study_ids is not None:
                        code_indices = model_encoder.get_codebook_indices(data, study_ids)
                    else:
                        code_indices = model_encoder.get_codebook_indices(data)
            except Exception as e:
                print(f"WARNING: Failed to get codebook indices: {e}")
                continue

            # Count
            for code, label in zip(code_indices.cpu().numpy(), labels):
                code_idx = int(code)
                code_total_count[code_idx] += 1
                if label == 1:
                    code_case_count[code_idx] += 1

    if not code_total_count:
        return None

    # Bayesian smoothing
    smoothed_ratio = {}
    for code in code_total_count:
        n_case = code_case_count[code]
        n_total = code_total_count[code]
        smoothed_ratio[code] = (n_case + alpha) / (n_total + alpha + beta)

    return smoothed_ratio


def compute_ratio_regularization_loss(
    attention_scores: torch.Tensor,
    code_indices: torch.Tensor,
    disease_ratio: Dict[int, float],
    device: torch.device
) -> torch.Tensor:
    """
    Attention score와 질병 비율 간의 MSE loss를 계산합니다.

    Attention score가 해당 코드의 질병 비율과 유사하도록 유도합니다.

    Args:
        attention_scores: Normalized attention scores (n_cells,)
        code_indices: VQ codebook indices (n_cells,)
        disease_ratio: {code_id: disease_ratio} mapping
        device: torch device

    Returns:
        MSE loss between attention scores and target disease ratios
    """
    # Get target ratios for each cell
    target_ratios = torch.tensor(
        [disease_ratio.get(int(c), 0.5) for c in code_indices.cpu().numpy()],
        dtype=torch.float32, device=device
    )

    # Normalize attention scores to [0, 1]
    attn_min = attention_scores.min()
    attn_max = attention_scores.max()
    attn_norm = (attention_scores - attn_min) / (attn_max - attn_min + 1e-8)

    # MSE loss
    loss = torch.nn.functional.mse_loss(attn_norm, target_ratios)

    return loss


def print_disease_ratio_summary(disease_ratio: Dict[int, float], top_k: int = 10):
    """질병 비율 요약 출력."""
    if disease_ratio is None:
        print("Disease ratio not available")
        return

    ratios = list(disease_ratio.values())
    print(f"\n{'='*50}")
    print("Disease Ratio Summary")
    print(f"{'='*50}")
    print(f"  Total codes: {len(disease_ratio)}")
    print(f"  Mean ratio: {np.mean(ratios):.4f}")
    print(f"  Std ratio: {np.std(ratios):.4f}")
    print(f"  Min ratio: {np.min(ratios):.4f}")
    print(f"  Max ratio: {np.max(ratios):.4f}")

    # Top disease-associated codes
    sorted_codes = sorted(disease_ratio.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top {top_k} disease-associated codes:")
    for code, ratio in sorted_codes[:top_k]:
        print(f"    Code {code}: {ratio:.4f}")

    print(f"{'='*50}\n")
