#!/bin/bash
# ============================================================================
# EMA Sweep — Common Cell Type subset
# ============================================================================
# sweep_ema_extended.sh 를 common cell type adata 로 실행하는 래퍼.
# Grid 는 sweep_ema_extended.sh 와 동일하여 whole cell type 결과와 1:1 비교 가능.
#
# Usage
#   bash scripts/sweep_ema_common.sh --gpu 0
#   bash scripts/sweep_ema_common.sh --gpu 0 --skip_existing
#   bash scripts/sweep_ema_common.sh --gpu 0 --dry_run
#
# 모든 추가 옵션은 sweep_ema_extended.sh 로 그대로 전달됩니다.
# (--decays, --codes, --commits, --epochs, --existing_sweep 등)
# ============================================================================

set -u

PROJECT_ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

exec bash "${PROJECT_ROOT_DIR}/scripts/sweep_ema_extended.sh" \
    --base_pretrain "${PROJECT_ROOT_DIR}/config/ema_common_pretrain.yaml" \
    --base_skin    "${PROJECT_ROOT_DIR}/config/ema_common_skin3.yaml" \
    --base_scp     "${PROJECT_ROOT_DIR}/config/ema_common_scp1884.yaml" \
    --tag "sweep_common_$(date '+%Y%m%d_%H%M%S')" \
    "$@"
