#!/bin/bash
# ============================================================================
# EMA Quantizer Small Benchmark
# ============================================================================
# 목적
#   - Quantizer 에 EMA codebook update (Oord 2017 App. A.1) 를 켜고
#     small config 으로 pretrain → within CV → finalize → cross-disease eval
#     을 skin3 / scp1884 양쪽 모두 돌린다.
#
# 실행
#   bash scripts/run_ema_small.sh --gpu 0
#   bash scripts/run_ema_small.sh --gpu 0 --skip_pretrain   # 이미 pretrain 완료된 경우
#
# 결과 / 로그
#   - 모든 산출물:  ${output_root} (= config/ema_small_pretrain.yaml 의 output_root)
#   - 실행 로그:    ${output_root}/logs/run_YYYYMMDD_HHMMSS/
#       ├─ pipeline.log                 (tee 된 전체 stdout+stderr)
#       ├─ pretrain.log                 (phase 별 stdout+stderr)
#       ├─ cv_skin3.log  /  cv_scp1884.log
#       ├─ final_skin3.log  /  final_scp1884.log
#       ├─ cross_skin3_to_scp1884.log
#       ├─ cross_scp1884_to_skin3.log
#       ├─ summary.md                   (각 단계 CSV 에서 추출한 핵심 메트릭)
#       └─ manifest.txt                 (생성된 주요 디렉토리 경로)
# ============================================================================

set -u  # undefined var error (pipefail 은 tee 때문에 끔)

# --- defaults ---------------------------------------------------------------
GPU=0
SKIP_PRETRAIN=false
PROJECT_ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

PRETRAIN_CFG="${PROJECT_ROOT_DIR}/config/ema_small_pretrain.yaml"
SKIN3_CFG="${PROJECT_ROOT_DIR}/config/ema_small_skin3.yaml"
SCP_CFG="${PROJECT_ROOT_DIR}/config/ema_small_scp1884.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU=$2; shift 2;;
        --skip_pretrain) SKIP_PRETRAIN=true; shift;;
        --pretrain_cfg) PRETRAIN_CFG=$2; shift 2;;
        --skin3_cfg) SKIN3_CFG=$2; shift 2;;
        --scp_cfg) SCP_CFG=$2; shift 2;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# --- resolve output_root from pretrain config ------------------------------
# pretrain config 의 ${paths.output_root} 을 python 으로 resolve (변수 치환 포함).
OUTPUT_ROOT=$(python - <<PY
from src.config import load_config
c = load_config("${PRETRAIN_CFG}")
print(c.paths.output_root)
PY
)
if [ -z "${OUTPUT_ROOT}" ]; then
    echo "[FATAL] Could not resolve output_root from ${PRETRAIN_CFG}"
    exit 2
fi

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR="${OUTPUT_ROOT}/logs/run_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

PIPELINE_LOG="${LOG_DIR}/pipeline.log"
MANIFEST="${LOG_DIR}/manifest.txt"
SUMMARY="${LOG_DIR}/summary.md"

# All stdout/stderr from this script also goes to pipeline.log.
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

echo "============================================================"
echo "  EMA small benchmark"
echo "============================================================"
echo "  GPU:             ${GPU}"
echo "  Pretrain config: ${PRETRAIN_CFG}"
echo "  Skin3 config:    ${SKIN3_CFG}"
echo "  SCP1884 config:  ${SCP_CFG}"
echo "  Output root:     ${OUTPUT_ROOT}"
echo "  Log dir:         ${LOG_DIR}"
echo "  Skip pretrain:   ${SKIP_PRETRAIN}"
echo "============================================================"

cd "${PROJECT_ROOT_DIR}"

# --- helpers ---------------------------------------------------------------
STATUS_OK=()
STATUS_FAIL=()

run_step() {
    local name=$1; shift
    local log="${LOG_DIR}/${name}.log"
    echo ""
    echo "[$(date '+%F %T')] >>> ${name}"
    echo "    cmd: $*"
    echo "    log: ${log}"
    if "$@" >"${log}" 2>&1; then
        echo "[$(date '+%F %T')] <<< ${name}  [OK]"
        STATUS_OK+=("${name}")
        return 0
    else
        local rc=$?
        echo "[$(date '+%F %T')] <<< ${name}  [FAIL rc=${rc}]"
        echo "    tail:"
        tail -n 20 "${log}" | sed 's/^/      /'
        STATUS_FAIL+=("${name}")
        return $rc
    fi
}

# Latest directory matching a glob under a parent dir.
latest_dir() {
    local parent=$1; local pattern=$2
    ls -1dt "${parent}/"${pattern} 2>/dev/null | head -n 1
}

echo "${LOG_DIR}" > "${MANIFEST}"

# --- 1. Pretrain (EMA on) ---------------------------------------------------
if [ "${SKIP_PRETRAIN}" = "false" ]; then
    run_step "pretrain" \
        python scripts/01_pretrain_encoder.py \
            --config "${PRETRAIN_CFG}" \
            --gpu "${GPU}" \
            --register \
        || { echo "[FATAL] pretrain failed; aborting."; exit 3; }
else
    echo "[SKIP] pretrain (--skip_pretrain)"
    STATUS_OK+=("pretrain(skipped)")
fi

PRETRAINED_PATH=$(python - <<PY
from src.config import load_config
print(load_config("${PRETRAIN_CFG}").paths.pretrained_encoder)
PY
)
echo "pretrained_encoder=${PRETRAINED_PATH}" >> "${MANIFEST}"

# --- 2. Within: CV train + finalize ---------------------------------------
train_and_finalize() {
    local tag=$1        # skin3 | scp1884
    local cfg=$2
    local disease_root
    disease_root=$(python - <<PY
from src.config import load_config
print(load_config("${cfg}").paths.output_root)
PY
    )
    echo "${tag}.output_root=${disease_root}" >> "${MANIFEST}"

    run_step "cv_${tag}" \
        python scripts/02_train_cv.py \
            --config "${cfg}" \
            --gpu "${GPU}"
    local cv_dir
    cv_dir=$(latest_dir "${disease_root}" "cv_*")
    echo "${tag}.cv_dir=${cv_dir}" >> "${MANIFEST}"

    run_step "final_${tag}" \
        python scripts/03_finalize_model.py \
            --config "${cfg}" \
            --gpu "${GPU}"
    local final_dir
    final_dir=$(latest_dir "${disease_root}" "final_model_*")
    echo "${tag}.final_dir=${final_dir}" >> "${MANIFEST}"

    # Export so caller can pick up.
    eval "${tag^^}_FINAL_DIR=\"${final_dir}\""
}

train_and_finalize skin3   "${SKIN3_CFG}"
train_and_finalize scp1884 "${SCP_CFG}"

# --- 3. Cross-disease eval (pair-wise) -------------------------------------
# SKIN3 final  evaluated on SCP1884 config, and vice versa.
run_step "cross_skin3_to_scp1884" \
    python scripts/04_cross_disease_eval.py \
        --model_dir "${SKIN3_FINAL_DIR}" \
        --test_config "${SCP_CFG}" \
        --gpu "${GPU}"
CROSS_SKIN3_DIR=$(latest_dir "${SKIN3_FINAL_DIR}" "cross_eval_*")
echo "cross.skin3_to_scp1884=${CROSS_SKIN3_DIR}" >> "${MANIFEST}"

run_step "cross_scp1884_to_skin3" \
    python scripts/04_cross_disease_eval.py \
        --model_dir "${SCP1884_FINAL_DIR}" \
        --test_config "${SKIN3_CFG}" \
        --gpu "${GPU}"
CROSS_SCP_DIR=$(latest_dir "${SCP1884_FINAL_DIR}" "cross_eval_*")
echo "cross.scp1884_to_skin3=${CROSS_SCP_DIR}" >> "${MANIFEST}"

# --- 4. Summary --------------------------------------------------------------
{
    echo "# EMA small benchmark — ${TIMESTAMP}"
    echo
    echo "- pretrain config: \`${PRETRAIN_CFG}\`"
    echo "- pretrained encoder: \`${PRETRAINED_PATH}\`"
    echo "- log dir: \`${LOG_DIR}\`"
    echo
    echo "## Step status"
    for s in "${STATUS_OK[@]}";   do echo "- OK   ${s}"; done
    for s in "${STATUS_FAIL[@]}"; do echo "- FAIL ${s}"; done
    echo
    echo "## Within-disease CV (overall_results.csv)"
    for tag in skin3 scp1884; do
        cv_d=$(grep "^${tag}.cv_dir=" "${MANIFEST}" | cut -d= -f2-)
        echo
        echo "### ${tag}"
        if [ -n "${cv_d}" ] && [ -f "${cv_d}/overall_results.csv" ]; then
            echo '```'
            cat "${cv_d}/overall_results.csv"
            echo '```'
        else
            echo "_missing_"
        fi
    done
    echo
    echo "## Cross-disease eval (cross_eval_results.csv)"
    for pair in "skin3_to_scp1884:${CROSS_SKIN3_DIR}" "scp1884_to_skin3:${CROSS_SCP_DIR}"; do
        name=${pair%%:*}; dir=${pair#*:}
        echo
        echo "### ${name}"
        if [ -n "${dir}" ] && [ -f "${dir}/cross_eval_results.csv" ]; then
            echo '```'
            cat "${dir}/cross_eval_results.csv"
            echo '```'
        else
            echo "_missing_"
        fi
    done
} > "${SUMMARY}"

echo ""
echo "============================================================"
echo "  Done"
echo "============================================================"
echo "  OK steps:   ${#STATUS_OK[@]}  (${STATUS_OK[*]:-})"
echo "  FAIL steps: ${#STATUS_FAIL[@]} (${STATUS_FAIL[*]:-})"
echo "  Log dir:    ${LOG_DIR}"
echo "  Summary:    ${SUMMARY}"
echo "  Manifest:   ${MANIFEST}"
echo "============================================================"

[ "${#STATUS_FAIL[@]}" -eq 0 ] || exit 1
