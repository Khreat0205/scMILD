#!/bin/bash
# ============================================================================
# Pretrain Encoder Hyperparameter Sweep
# ============================================================================
# Grid: loss_weight x num_codes x data_scope
#
# Fixed: lr=0.0001, hidden_dim=16, batch_size=2048
#
# Usage:
#   bash scripts/sweep_pretrain.sh --gpu 1                    # 전체 (subset → whole)
#   bash scripts/sweep_pretrain.sh --gpu 1 --phase subset     # subset만
#   bash scripts/sweep_pretrain.sh --gpu 1 --phase whole      # whole만
#   bash scripts/sweep_pretrain.sh --gpu 0 --skip_existing    # 완료된 실험 건너뛰기
#
# 주의:
#   - subset (326k): n_conditionals=3 (3 studies만 포함)
#     → downstream MIL은 동일 3 studies에서만 평가 가능
#   - whole (805k): n_conditionals=11 (전체 studies)
#     → celltype aux는 40.5% valid cells에서만 계산
#   - num_codes=1024 + whole: k-means init 시 샘플 부족 가능
#     → NaN 발생 시 batch_size 추가 증가 필요
# ============================================================================

# ===========================
# Configuration
# ===========================
BASE_CONFIG="config/pretrain_celltype_aux_v3.yaml"

# Data paths (서버 경로)
PROJECT_ROOT="/home/bmi-user/workspace/data/HSvsCD/scMILDQ_Cond"
SUBSET_DATA="${PROJECT_ROOT}/data/SCP_Skin_326k_6k_unified_v2.h5ad"
OUTPUT_BASE="${PROJECT_ROOT}/results/pretrain_sweep"

# Hyperparameter grid
LOSS_WEIGHTS=(0.05 0.1 0.2)
NUM_CODES=(256 512 1024)

# ===========================
# Parse arguments
# ===========================
GPU=1
PHASE="all"        # all | subset | whole
SKIP_EXISTING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU=$2; shift 2;;
        --phase) PHASE=$2; shift 2;;
        --skip_existing) SKIP_EXISTING=true; shift;;
        --config) BASE_CONFIG=$2; shift 2;;
        --output) OUTPUT_BASE=$2; shift 2;;
        -h|--help)
            echo "Usage: bash scripts/sweep_pretrain.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu N            GPU ID (default: 1)"
            echo "  --phase PHASE      all|subset|whole (default: all)"
            echo "  --skip_existing    Skip experiments with existing results"
            echo "  --config PATH      Base config file"
            echo "  --output PATH      Output base directory"
            exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# ===========================
# Counters
# ===========================
TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0
FAILED_LIST=""

# ===========================
# Helper functions
# ===========================
format_weight() {
    # 0.05 → 005, 0.1 → 010, 0.2 → 020
    # bc 없이 awk 사용
    echo "$1" | awk '{printf "%03d", $1 * 100}'
}

check_existing() {
    local dir="$1"
    # Check if any pretrain_* subdirectory has a saved model
    if ls "${dir}"/pretrain_*/vq_aenb_conditional.pth 1>/dev/null 2>&1; then
        return 0  # exists
    fi
    return 1  # not exists
}

run_experiment() {
    local name=$1
    shift
    local extra_args=("$@")
    local out_dir="${OUTPUT_BASE}/${name}"

    ((TOTAL++))

    # Skip if already completed
    if $SKIP_EXISTING && check_existing "$out_dir"; then
        echo "[SKIP] ${name} (already completed)"
        ((SKIPPED++))
        return 0
    fi

    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${name} (${TOTAL})"
    echo "============================================================"

    python scripts/01_pretrain_encoder.py \
        --config "${BASE_CONFIG}" \
        --output_dir "${out_dir}" \
        --gpu "${GPU}" \
        "${extra_args[@]}"

    local status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: ${name}"
        ((SUCCESS++))
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: ${name} (exit code: ${status})"
        ((FAILED++))
        FAILED_LIST="${FAILED_LIST}\n  - ${name}"
    fi

    # GPU memory cleanup
    sleep 5
    return $status
}

# ===========================
# Print sweep plan
# ===========================
N_SUBSET=$(( ${#NUM_CODES[@]} * ${#LOSS_WEIGHTS[@]} ))
N_WHOLE=$N_SUBSET
case $PHASE in
    all)     N_TOTAL=$(( N_SUBSET + N_WHOLE ));;
    subset)  N_TOTAL=$N_SUBSET;;
    whole)   N_TOTAL=$N_WHOLE;;
esac

echo "============================================================"
echo "  Pretrain Encoder Hyperparameter Sweep"
echo "============================================================"
echo "  GPU:          ${GPU}"
echo "  Phase:        ${PHASE}"
echo "  Base config:  ${BASE_CONFIG}"
echo "  Output:       ${OUTPUT_BASE}"
echo "  Skip exist:   ${SKIP_EXISTING}"
echo ""
echo "  Grid:"
echo "    loss_weight = [${LOSS_WEIGHTS[*]}]"
echo "    num_codes   = [${NUM_CODES[*]}]"
echo "    data_scope  = [subset_326k, whole_805k]"
echo ""
echo "  Total experiments: ${N_TOTAL}"
echo "============================================================"
echo ""

# ===========================
# Phase 1: Subset (326k, 3 annotated studies)
# ===========================
if [[ "$PHASE" == "all" || "$PHASE" == "subset" ]]; then
    echo ""
    echo "############################################################"
    echo "  Phase 1: Subset 326k (3 annotated studies)"
    echo "  Data: ${SUBSET_DATA}"
    echo "############################################################"
    echo ""

    for codes in "${NUM_CODES[@]}"; do
        for weight in "${LOSS_WEIGHTS[@]}"; do
            w_name=$(format_weight "$weight")
            name="sub_c${codes}_w${w_name}"

            run_experiment "$name" \
                --adata_path "$SUBSET_DATA" \
                --num_codes "$codes" \
                --celltype_loss_weight "$weight" \
                || true  # Continue on failure
        done
    done
fi

# ===========================
# Phase 2: Whole (805k, 11 studies)
# ===========================
if [[ "$PHASE" == "all" || "$PHASE" == "whole" ]]; then
    echo ""
    echo "############################################################"
    echo "  Phase 2: Whole 805k (11 studies, 40.5% annotated)"
    echo "  Data: config default (Whole_SCP_PCD_Skin_805k_6k_unified_v2.h5ad)"
    echo "############################################################"
    echo ""

    for codes in "${NUM_CODES[@]}"; do
        for weight in "${LOSS_WEIGHTS[@]}"; do
            w_name=$(format_weight "$weight")
            name="whole_c${codes}_w${w_name}"

            run_experiment "$name" \
                --num_codes "$codes" \
                --celltype_loss_weight "$weight" \
                || true  # Continue on failure
        done
    done
fi

# ===========================
# Summary
# ===========================
echo ""
echo "============================================================"
echo "  Sweep Complete"
echo "============================================================"
echo "  Total:   ${TOTAL}"
echo "  Success: ${SUCCESS}"
echo "  Failed:  ${FAILED}"
echo "  Skipped: ${SKIPPED}"
if [ -n "$FAILED_LIST" ]; then
    echo ""
    echo "  Failed experiments:${FAILED_LIST}"
fi
echo ""
echo "  Results: ${OUTPUT_BASE}/"
echo "============================================================"
