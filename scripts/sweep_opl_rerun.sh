#!/bin/bash
# ============================================================================
# OPL 적용 Sweep 재실행 스크립트
# ============================================================================
# 기존 sweep의 pretrain 결과를 symlink로 재활용하고,
# MIL 학습(CV/finalize/cross-eval)만 OPL 적용하여 재실행합니다.
#
# OPL은 MIL training에만 영향 (pretrain은 동일) → pretrain 재활용으로 시간 절약
#
# Usage:
#   # Extended sweep (OPL 재실행, 기존 pretrain 재활용)
#   bash scripts/sweep_opl_rerun.sh \
#       --type ext \
#       --prev_sweep /path/to/results/ema_small/ema_sweep/sweep_ext_YYYYMMDD_HHMMSS \
#       --gpus 0 2 3 --jobs_per_gpu 2
#
#   # Common sweep (OPL 재실행)
#   bash scripts/sweep_opl_rerun.sh \
#       --type common \
#       --prev_sweep /path/to/results/ema_common/ema_sweep/sweep_common_YYYYMMDD_HHMMSS \
#       --gpus 0 2 3 --jobs_per_gpu 2
#
#   # pretrain 없이 처음부터 전체 실행 (기존 sweep 결과 없을 때)
#   bash scripts/sweep_opl_rerun.sh --type ext --gpus 0 2 3 --jobs_per_gpu 2
#
#   # OPL lambda 값 변경
#   bash scripts/sweep_opl_rerun.sh --type ext --opl_lambda 0.3 \
#       --prev_sweep /path/to/prev_sweep --gpus 0 2 3 --jobs_per_gpu 2
#
#   # Dry run
#   bash scripts/sweep_opl_rerun.sh --type ext --dry_run
# ============================================================================

set -u

PROJECT_ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT_DIR}"

# --- defaults ---------------------------------------------------------------
SWEEP_TYPE=""           # ext | common
PREV_SWEEP_DIR=""       # 기존 sweep 디렉토리 (pretrain 재활용)
OPL_LAMBDA=""           # 비어있으면 config 기본값(0.2) 사용
DRY_RUN=false
EXTRA_ARGS=()          # sweep_ema_extended.sh 에 전달할 추가 인자

while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            SWEEP_TYPE=$2; shift 2;;
        --prev_sweep)
            PREV_SWEEP_DIR=$2; shift 2;;
        --opl_lambda)
            OPL_LAMBDA=$2; shift 2;;
        --dry_run)
            DRY_RUN=true
            EXTRA_ARGS+=(--dry_run)
            shift;;
        --gpus|--gpu|--jobs_per_gpu|--tag|--decays|--codes|--commits|--epochs)
            # Accumulate args for sweep_ema_extended.sh
            EXTRA_ARGS+=("$1")
            shift
            while [[ $# -gt 0 && ! $1 == --* ]]; do
                EXTRA_ARGS+=("$1")
                shift
            done
            ;;
        --skip_existing)
            EXTRA_ARGS+=(--skip_existing)
            shift;;
        -h|--help)
            sed -n '2,35p' "$0"; exit 0;;
        *)
            echo "Unknown option: $1"; exit 1;;
    esac
done

# --- validate ---------------------------------------------------------------
if [ -z "${SWEEP_TYPE}" ]; then
    echo "[ERROR] --type 을 지정해주세요 (ext 또는 common)"
    exit 1
fi

if [[ "${SWEEP_TYPE}" != "ext" && "${SWEEP_TYPE}" != "common" ]]; then
    echo "[ERROR] --type 은 ext 또는 common 이어야 합니다 (입력: ${SWEEP_TYPE})"
    exit 1
fi

# --- determine tag ----------------------------------------------------------
# Check if --tag was already provided in EXTRA_ARGS
HAS_TAG=false
for arg in "${EXTRA_ARGS[@]:-}"; do
    [ "$arg" = "--tag" ] && HAS_TAG=true
done

if ! $HAS_TAG; then
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    if [ -n "${OPL_LAMBDA}" ]; then
        opl_s=$(printf "%s" "${OPL_LAMBDA}" | tr '.' '_')
        TAG="sweep_${SWEEP_TYPE}_opl${opl_s}_${TIMESTAMP}"
    else
        TAG="sweep_${SWEEP_TYPE}_opl_${TIMESTAMP}"
    fi
    EXTRA_ARGS+=(--tag "${TAG}")
fi

echo "============================================================"
echo "  OPL Sweep Rerun"
echo "============================================================"
echo "  Type:         ${SWEEP_TYPE}"
echo "  Prev sweep:   ${PREV_SWEEP_DIR:-<none — full run>}"
echo "  OPL lambda:   ${OPL_LAMBDA:-<config default (0.2)>}"
echo "  Extra args:   ${EXTRA_ARGS[*]:-}"
echo "============================================================"

# --- symlink pretrain results from previous sweep ---------------------------
if [ -n "${PREV_SWEEP_DIR}" ]; then
    if [ ! -d "${PREV_SWEEP_DIR}/runs" ]; then
        echo "[ERROR] 이전 sweep 의 runs 디렉토리를 찾을 수 없습니다: ${PREV_SWEEP_DIR}/runs"
        exit 1
    fi

    # Resolve sweep output parent for new sweep
    if [ "${SWEEP_TYPE}" = "ext" ]; then
        BASE_CFG="${PROJECT_ROOT_DIR}/config/ema_small_pretrain.yaml"
    else
        BASE_CFG="${PROJECT_ROOT_DIR}/config/ema_common_pretrain.yaml"
    fi

    SWEEP_PARENT=$(python - <<PY
import os
from src.config import load_config
out = load_config("${BASE_CFG}").paths.output_root
print(os.path.dirname(out.rstrip("/")))
PY
    )

    # Get the tag from EXTRA_ARGS
    NEW_TAG=""
    for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
        if [ "${EXTRA_ARGS[$i]}" = "--tag" ] && [ $((i+1)) -lt ${#EXTRA_ARGS[@]} ]; then
            NEW_TAG="${EXTRA_ARGS[$((i+1))]}"
            break
        fi
    done

    if [ -z "${NEW_TAG}" ]; then
        echo "[ERROR] Could not determine new sweep tag"
        exit 1
    fi

    NEW_RUN_DIR="${SWEEP_PARENT}/ema_sweep/${NEW_TAG}/runs"

    if ! $DRY_RUN; then
        mkdir -p "${NEW_RUN_DIR}"
    fi

    # Count and link pretrain results
    N_LINKED=0
    N_TOTAL=0
    for exp_dir in "${PREV_SWEEP_DIR}/runs"/d*; do
        [ ! -d "${exp_dir}" ] && continue
        exp_tag=$(basename "${exp_dir}")
        N_TOTAL=$((N_TOTAL + 1))

        pretrain_src="${exp_dir}/pretrained"
        if [ -d "${pretrain_src}" ] && [ -f "${pretrain_src}/vq_aenb_conditional.pth" ]; then
            target_exp="${NEW_RUN_DIR}/${exp_tag}"
            if ! $DRY_RUN; then
                mkdir -p "${target_exp}"
                # Symlink the pretrained directory
                if [ ! -e "${target_exp}/pretrained" ]; then
                    ln -s "$(realpath "${pretrain_src}")" "${target_exp}/pretrained"
                    N_LINKED=$((N_LINKED + 1))
                else
                    echo "  [EXISTS] ${exp_tag}/pretrained already exists, skipping"
                fi
            else
                N_LINKED=$((N_LINKED + 1))
            fi
        fi
    done

    echo ""
    echo "[SYMLINK] ${N_LINKED}/${N_TOTAL} pretrain results linked"
    echo "  From: ${PREV_SWEEP_DIR}/runs/"
    echo "  To:   ${NEW_RUN_DIR}/"

    # Add --skip_existing to reuse pretrain (status=1 → rerun downstream only)
    EXTRA_ARGS+=(--skip_existing)
fi

# --- OPL lambda override in base configs -----------------------------------
# 임시로 base config를 복사하고 OPL lambda를 오버라이드
CLEANUP_FILES=()

if [ -n "${OPL_LAMBDA}" ]; then
    echo ""
    echo "[CONFIG] OPL lambda override: ${OPL_LAMBDA}"

    if [ "${SWEEP_TYPE}" = "ext" ]; then
        CONFIGS=("ema_small_pretrain.yaml" "ema_small_skin3.yaml" "ema_small_scp1884.yaml")
    else
        CONFIGS=("ema_common_pretrain.yaml" "ema_common_skin3.yaml" "ema_common_scp1884.yaml")
    fi

    TEMP_DIR=$(mktemp -d "${PROJECT_ROOT_DIR}/config/.opl_override_XXXXXX")
    CLEANUP_FILES+=("${TEMP_DIR}")

    for cfg_name in "${CONFIGS[@]}"; do
        src="${PROJECT_ROOT_DIR}/config/${cfg_name}"
        dst="${TEMP_DIR}/${cfg_name}"
        cp "${src}" "${dst}"

        # Replace orthogonal_projection_lambda value
        if grep -q "orthogonal_projection_lambda:" "${dst}"; then
            sed -i.bak "s/orthogonal_projection_lambda:.*/orthogonal_projection_lambda: ${OPL_LAMBDA}/" "${dst}"
            rm -f "${dst}.bak"
        else
            # Append to mil.loss section
            cat >> "${dst}" <<EOF

mil:
  loss:
    orthogonal_projection_lambda: ${OPL_LAMBDA}
EOF
        fi
        echo "  ${cfg_name} → orthogonal_projection_lambda: ${OPL_LAMBDA}"
    done

    # Override base configs in EXTRA_ARGS
    if [ "${SWEEP_TYPE}" = "ext" ]; then
        EXTRA_ARGS+=(--base_pretrain "${TEMP_DIR}/ema_small_pretrain.yaml")
        EXTRA_ARGS+=(--base_skin "${TEMP_DIR}/ema_small_skin3.yaml")
        EXTRA_ARGS+=(--base_scp "${TEMP_DIR}/ema_small_scp1884.yaml")
    else
        EXTRA_ARGS+=(--base_pretrain "${TEMP_DIR}/ema_common_pretrain.yaml")
        EXTRA_ARGS+=(--base_skin "${TEMP_DIR}/ema_common_skin3.yaml")
        EXTRA_ARGS+=(--base_scp "${TEMP_DIR}/ema_common_scp1884.yaml")
    fi
fi

# --- cleanup trap -----------------------------------------------------------
cleanup() {
    for f in "${CLEANUP_FILES[@]:-}"; do
        [ -d "$f" ] && rm -rf "$f"
    done
}
trap cleanup EXIT

# --- launch sweep -----------------------------------------------------------
echo ""
echo "[LAUNCH] Running sweep_ema_${SWEEP_TYPE} with OPL …"
echo ""

if [ "${SWEEP_TYPE}" = "ext" ]; then
    exec bash "${PROJECT_ROOT_DIR}/scripts/sweep_ema_extended.sh" "${EXTRA_ARGS[@]}"
else
    # For common type, we need to pass base configs through sweep_ema_common.sh
    # But sweep_ema_common.sh sets its own base configs.
    # So for common + OPL lambda override, call sweep_ema_extended.sh directly with common configs.
    if [ -n "${OPL_LAMBDA}" ]; then
        # Already have --base_pretrain/skin/scp in EXTRA_ARGS from OPL override
        exec bash "${PROJECT_ROOT_DIR}/scripts/sweep_ema_extended.sh" "${EXTRA_ARGS[@]}"
    else
        exec bash "${PROJECT_ROOT_DIR}/scripts/sweep_ema_common.sh" "${EXTRA_ARGS[@]}"
    fi
fi
