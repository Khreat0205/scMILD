#!/bin/bash
# ============================================================================
# EMA Quantizer Sweep
# ============================================================================
# 각 HP 조합별로:
#   pretrain → CV skin3 → CV scp1884 → finalize skin3/scp1884 → cross 양방향
# 을 run_ema_small.sh 로 실행하고, 조합별 성능을 master_results.csv 로 집계.
#
# 기본 grid: ema_decay × num_codes × commitment_weight.
# 다른 축 (loss_type/lr/epochs 등) 을 sweep 하려면 밑의 배열만 추가하면 됨.
#
# Usage
#   bash scripts/sweep_ema.sh --gpu 0
#   bash scripts/sweep_ema.sh --gpu 0 --decays 0.99 --codes 256 512
#   bash scripts/sweep_ema.sh --gpu 0 --skip_existing
#
# 출력
#   ${project_root}/results/ema_sweep/<TAG>/
#     ├─ configs/<exp>/{pretrain,skin3,scp1884}.yaml
#     ├─ runs/<exp>/…  (run_ema_small.sh 의 logs + 모델 산출물)
#     ├─ master_results.csv       # 핵심 메트릭 한 행/exp
#     └─ sweep.log                # 전체 진행 tee
# ============================================================================

set -u

PROJECT_ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT_DIR}"

# --- defaults ---------------------------------------------------------------
GPU=0
SKIP_EXISTING=false
BASE_PRETRAIN="${PROJECT_ROOT_DIR}/config/ema_small_pretrain.yaml"
BASE_SKIN="${PROJECT_ROOT_DIR}/config/ema_small_skin3.yaml"
BASE_SCP="${PROJECT_ROOT_DIR}/config/ema_small_scp1884.yaml"
DECAYS=(0.99 0.999)
NUM_CODES_LIST=(256 512)
COMMIT_WEIGHTS=(0.25)
LOSS_TYPE=""          # empty → base 의 값을 그대로 사용
INPUT_TRANSFORM=""
EPOCHS=""             # empty → base 의 값을 그대로 사용
SWEEP_TAG="sweep_$(date '+%Y%m%d_%H%M%S')"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU=$2; shift 2;;
        --skip_existing) SKIP_EXISTING=true; shift;;
        --decays) shift; DECAYS=(); while [[ $# -gt 0 && ! $1 == --* ]]; do DECAYS+=("$1"); shift; done;;
        --codes) shift; NUM_CODES_LIST=(); while [[ $# -gt 0 && ! $1 == --* ]]; do NUM_CODES_LIST+=("$1"); shift; done;;
        --commits) shift; COMMIT_WEIGHTS=(); while [[ $# -gt 0 && ! $1 == --* ]]; do COMMIT_WEIGHTS+=("$1"); shift; done;;
        --loss_type) LOSS_TYPE=$2; shift 2;;
        --input_transform) INPUT_TRANSFORM=$2; shift 2;;
        --epochs) EPOCHS=$2; shift 2;;
        --tag) SWEEP_TAG=$2; shift 2;;
        --base_pretrain) BASE_PRETRAIN=$2; shift 2;;
        --base_skin) BASE_SKIN=$2; shift 2;;
        --base_scp) BASE_SCP=$2; shift 2;;
        -h|--help) sed -n '2,30p' "$0"; exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# --- resolve project_root / sweep dir --------------------------------------
PROJECT_ROOT=$(python - <<PY
from src.config import load_config
print(load_config("${BASE_PRETRAIN}").paths.project_root)
PY
)
SWEEP_DIR="${PROJECT_ROOT}/results/ema_sweep/${SWEEP_TAG}"
CFG_DIR="${SWEEP_DIR}/configs"
RUN_DIR="${SWEEP_DIR}/runs"
MASTER_CSV="${SWEEP_DIR}/master_results.csv"
SWEEP_LOG="${SWEEP_DIR}/sweep.log"
mkdir -p "${CFG_DIR}" "${RUN_DIR}"

exec > >(tee -a "${SWEEP_LOG}") 2>&1

echo "============================================================"
echo "  EMA sweep: ${SWEEP_TAG}"
echo "============================================================"
echo "  GPU:            ${GPU}"
echo "  Base pretrain:  ${BASE_PRETRAIN}"
echo "  Base skin3:     ${BASE_SKIN}"
echo "  Base scp1884:   ${BASE_SCP}"
echo "  Sweep dir:      ${SWEEP_DIR}"
echo "  Grid:"
echo "    ema_decay          = [${DECAYS[*]}]"
echo "    num_codes          = [${NUM_CODES_LIST[*]}]"
echo "    commitment_weight  = [${COMMIT_WEIGHTS[*]}]"
[ -n "${LOSS_TYPE}" ] && echo "    loss_type override = ${LOSS_TYPE}"
[ -n "${INPUT_TRANSFORM}" ] && echo "    input_xform override = ${INPUT_TRANSFORM}"
[ -n "${EPOCHS}" ] && echo "    epochs override    = ${EPOCHS}"
N_COMBOS=$(( ${#DECAYS[@]} * ${#NUM_CODES_LIST[@]} * ${#COMMIT_WEIGHTS[@]} ))
echo "  Total combos:   ${N_COMBOS}"
echo "============================================================"

# Master CSV header
if [ ! -f "${MASTER_CSV}" ]; then
    echo "exp_tag,ema_decay,num_codes,commit_weight,loss_type,input_transform,epochs,status,skin3_auc,skin3_f1,scp1884_auc,scp1884_f1,cross_s2c_auc,cross_s2c_f1,cross_c2s_auc,cross_c2s_f1,run_dir" \
        > "${MASTER_CSV}"
fi

# --- helpers ----------------------------------------------------------------
format_tag() {
    # Safe filename tag from decay/codes/commit.
    local d=$1 c=$2 w=$3
    local d_s c_s w_s
    d_s=$(printf "%s" "$d" | tr '.' '_')
    c_s=$(printf "%d" "$c")
    w_s=$(printf "%s" "$w" | tr '.' '_')
    echo "d${d_s}_c${c_s}_w${w_s}"
}

gen_configs() {
    # Generate pretrain/skin3/scp1884 yaml for this exp.
    # We don't use _base_ here — emit fully-resolved "overlay" files that
    # still extend the project's BASE_* via _base_, only overriding the
    # knobs we sweep and the output paths (so runs are isolated).
    local exp_tag=$1 decay=$2 codes=$3 commit=$4
    local exp_cfg_dir="${CFG_DIR}/${exp_tag}"
    local exp_out_root="${RUN_DIR}/${exp_tag}"
    mkdir -p "${exp_cfg_dir}"

    local loss_line=""
    local xform_line=""
    local epochs_line=""
    [ -n "${LOSS_TYPE}" ] && loss_line="  loss_type: \"${LOSS_TYPE}\""
    [ -n "${INPUT_TRANSFORM}" ] && xform_line="  input_transform: \"${INPUT_TRANSFORM}\""
    [ -n "${EPOCHS}" ] && epochs_line="    epochs: ${EPOCHS}"

    cat > "${exp_cfg_dir}/pretrain.yaml" <<EOF
# Auto-generated by sweep_ema.sh — exp ${exp_tag}
_base_: "${BASE_PRETRAIN}"

paths:
  pretrained_encoder: "${exp_out_root}/pretrained/vq_aenb_conditional.pth"
  output_root: "${exp_out_root}"

data:
  conditional_embedding:
    mapping_path: "${exp_out_root}/pretrained/study_mapping.json"

encoder:
  num_codes: ${codes}
${loss_line}
${xform_line}
  quantizer:
    commitment_weight: ${commit}
    ema_decay: ${decay}
  pretrain:
${epochs_line}
EOF

    cat > "${exp_cfg_dir}/skin3.yaml" <<EOF
# Auto-generated by sweep_ema.sh — exp ${exp_tag} / skin3
_base_: "${exp_cfg_dir}/pretrain.yaml"

paths:
  output_root: "${exp_out_root}/skin3"

data:
  subset:
    enabled: true
    column: "study"
    values:
      - "GSE175990"
      - "GSE220116"

splitting:
  strategy: "stratified_kfold"
  n_splits: 5
  n_repeats: 1
  random_seed: 42
EOF

    cat > "${exp_cfg_dir}/scp1884.yaml" <<EOF
# Auto-generated by sweep_ema.sh — exp ${exp_tag} / scp1884
_base_: "${exp_cfg_dir}/pretrain.yaml"

paths:
  output_root: "${exp_out_root}/scp1884"

data:
  subset:
    enabled: true
    column: "study"
    values:
      - "SCP1884"

splitting:
  strategy: "stratified_kfold"
  n_splits: 5
  n_repeats: 1
  random_seed: 42

mil:
  subsampling:
    enabled: true
    max_cells_per_sample: 5000
EOF
}

# Read auc/f1 from an overall_results.csv (CV) — "auc,f1" or ",".
read_cv_metrics() {
    local csv=$1
    python - "$csv" <<'PY'
import sys, pandas as pd, math
p = sys.argv[1]
try:
    df = pd.read_csv(p)
    auc = float(df.iloc[0].get("overall_auc", float("nan")))
    f1 = float(df.iloc[0].get("overall_f1", float("nan")))
    print(f"{auc if math.isfinite(auc) else ''},{f1 if math.isfinite(f1) else ''}")
except Exception:
    print(",")
PY
}

# Read auc/f1 from a cross_eval_results.csv.
read_cross_metrics() {
    local csv=$1
    python - "$csv" <<'PY'
import sys, pandas as pd, math
p = sys.argv[1]
try:
    df = pd.read_csv(p)
    # schema: single row aggregate OR per-sample; prefer aggregate row.
    if "auc" in df.columns:
        auc = float(df["auc"].iloc[0])
    elif "overall_auc" in df.columns:
        auc = float(df["overall_auc"].iloc[0])
    else:
        auc = float("nan")
    if "f1_score" in df.columns:
        f1 = float(df["f1_score"].iloc[0])
    elif "overall_f1" in df.columns:
        f1 = float(df["overall_f1"].iloc[0])
    else:
        f1 = float("nan")
    print(f"{auc if math.isfinite(auc) else ''},{f1 if math.isfinite(f1) else ''}")
except Exception:
    print(",")
PY
}

latest_dir() { ls -1dt "$1/"$2 2>/dev/null | head -n 1; }

# --- main sweep loop --------------------------------------------------------
TOTAL=0; OK=0; FAIL=0; SKIP=0
for decay in "${DECAYS[@]}"; do
  for codes in "${NUM_CODES_LIST[@]}"; do
    for commit in "${COMMIT_WEIGHTS[@]}"; do
      TOTAL=$((TOTAL+1))
      exp_tag=$(format_tag "$decay" "$codes" "$commit")
      exp_out_root="${RUN_DIR}/${exp_tag}"

      echo ""
      echo "### [${TOTAL}/${N_COMBOS}] ${exp_tag}"
      echo "    decay=${decay}  codes=${codes}  commit=${commit}"

      if $SKIP_EXISTING && [ -f "${exp_out_root}/pretrained/vq_aenb_conditional.pth" ]; then
          # Only skips the whole exp if pretrained already exists; downstream
          # CV/finalize/cross outputs live alongside and will be picked up if
          # present, re-run otherwise (run_ema_small.sh has --skip_pretrain).
          echo "    [SKIP] pretrained/ already present"
          SKIP=$((SKIP+1))
          continue
      fi

      gen_configs "$exp_tag" "$decay" "$codes" "$commit"
      exp_cfg_dir="${CFG_DIR}/${exp_tag}"

      if bash scripts/run_ema_small.sh \
            --gpu "${GPU}" \
            --pretrain_cfg "${exp_cfg_dir}/pretrain.yaml" \
            --skin3_cfg    "${exp_cfg_dir}/skin3.yaml" \
            --scp_cfg      "${exp_cfg_dir}/scp1884.yaml"; then
          status="OK"; OK=$((OK+1))
      else
          status="FAIL"; FAIL=$((FAIL+1))
      fi

      # Scrape metrics (best-effort; missing files → empty cols).
      skin_cv=$(latest_dir "${exp_out_root}/skin3" "cv_*")
      scp_cv=$(latest_dir "${exp_out_root}/scp1884" "cv_*")
      skin_fin=$(latest_dir "${exp_out_root}/skin3" "final_model_*")
      scp_fin=$(latest_dir "${exp_out_root}/scp1884" "final_model_*")
      s2c_dir=""
      c2s_dir=""
      [ -n "${skin_fin}" ] && s2c_dir=$(latest_dir "${skin_fin}" "cross_eval_*")
      [ -n "${scp_fin}" ]  && c2s_dir=$(latest_dir "${scp_fin}" "cross_eval_*")

      skin_m=",";  [ -n "${skin_cv}"  ] && [ -f "${skin_cv}/overall_results.csv"  ] && skin_m=$(read_cv_metrics "${skin_cv}/overall_results.csv")
      scp_m=",";   [ -n "${scp_cv}"   ] && [ -f "${scp_cv}/overall_results.csv"   ] && scp_m=$(read_cv_metrics "${scp_cv}/overall_results.csv")
      s2c_m=",";   [ -n "${s2c_dir}"  ] && [ -f "${s2c_dir}/cross_eval_results.csv" ] && s2c_m=$(read_cross_metrics "${s2c_dir}/cross_eval_results.csv")
      c2s_m=",";   [ -n "${c2s_dir}"  ] && [ -f "${c2s_dir}/cross_eval_results.csv" ] && c2s_m=$(read_cross_metrics "${c2s_dir}/cross_eval_results.csv")

      # Emit one master row.
      echo "${exp_tag},${decay},${codes},${commit},${LOSS_TYPE:-inherit},${INPUT_TRANSFORM:-inherit},${EPOCHS:-inherit},${status},${skin_m},${scp_m},${s2c_m},${c2s_m},${exp_out_root}" \
          >> "${MASTER_CSV}"

      echo "    status=${status}  skin=(${skin_m})  scp=(${scp_m})  s2c=(${s2c_m})  c2s=(${c2s_m})"
    done
  done
done

echo ""
echo "============================================================"
echo "  Sweep done"
echo "============================================================"
echo "  Total: ${TOTAL}   OK: ${OK}   FAIL: ${FAIL}   SKIP: ${SKIP}"
echo "  Master CSV: ${MASTER_CSV}"
echo "  Sweep log:  ${SWEEP_LOG}"
echo "============================================================"

[ ${FAIL} -eq 0 ] || exit 1
