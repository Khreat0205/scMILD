#!/bin/bash
# ============================================================================
# EMA Quantizer Extended Sweep (Parallel)
# ============================================================================
# sweep_ema.sh 의 확장 버전. 다중 GPU × 다중 job 병렬 실행 지원.
#
# 확장 Grid:
#   ema_decay:         0.99, 0.995, 0.999, 0.9999
#   num_codes:         64, 128, 256, 512, 1024
#   commitment_weight: 0.1, 0.25, 0.5
#   → 총 60 조합
#
# Usage
#   # GPU 3장, 각 2개 job → 6 동시 실행
#   bash scripts/sweep_ema_extended.sh --gpus 0 1 2 --jobs_per_gpu 2
#
#   # 기존 결과 건너뛰기
#   bash scripts/sweep_ema_extended.sh --gpus 0 1 2 --jobs_per_gpu 2 --skip_existing
#
#   # 기존 호환: GPU 1장 순차 실행
#   bash scripts/sweep_ema_extended.sh --gpu 0
#
#   # dry run
#   bash scripts/sweep_ema_extended.sh --dry_run
#
# 출력
#   ${SWEEP_DIR}/
#     ├─ configs/<exp>/
#     ├─ runs/<exp>/
#     ├─ master_results.csv
#     ├─ sweep.log
#     └─ workers/worker_<N>.log
# ============================================================================

set -u

PROJECT_ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT_DIR}"

# --- defaults ---------------------------------------------------------------
GPUS=(0)
JOBS_PER_GPU=1
SKIP_EXISTING=false
DRY_RUN=false
BASE_PRETRAIN="${PROJECT_ROOT_DIR}/config/ema_small_pretrain.yaml"
BASE_SKIN="${PROJECT_ROOT_DIR}/config/ema_small_skin3.yaml"
BASE_SCP="${PROJECT_ROOT_DIR}/config/ema_small_scp1884.yaml"

# Extended grid
DECAYS=(0.99 0.995 0.999 0.9999)
NUM_CODES_LIST=(64 128 256 512 1024)
COMMIT_WEIGHTS=(0.1 0.25 0.5)

LOSS_TYPE=""
INPUT_TRANSFORM=""
EPOCHS=""
SWEEP_TAG="sweep_ext_$(date '+%Y%m%d_%H%M%S')"
EXISTING_SWEEP_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPUS=("$2"); shift 2;;
        --gpus) shift; GPUS=(); while [[ $# -gt 0 && ! $1 == --* ]]; do GPUS+=("$1"); shift; done;;
        --jobs_per_gpu) JOBS_PER_GPU=$2; shift 2;;
        --skip_existing) SKIP_EXISTING=true; shift;;
        --dry_run) DRY_RUN=true; shift;;
        --existing_sweep) EXISTING_SWEEP_DIR=$2; shift 2;;
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
        -h|--help) sed -n '2,40p' "$0"; exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

N_WORKERS=$(( ${#GPUS[@]} * JOBS_PER_GPU ))

# --- resolve sweep dir ------------------------------------------------------
SWEEP_PARENT=$(python - <<PY
import os
from src.config import load_config
out = load_config("${BASE_PRETRAIN}").paths.output_root
print(os.path.dirname(out.rstrip("/")))
PY
)
if [ -z "${SWEEP_PARENT}" ]; then
    echo "[FATAL] Could not resolve sweep parent dir from ${BASE_PRETRAIN}"
    exit 2
fi
SWEEP_DIR="${SWEEP_PARENT}/ema_sweep/${SWEEP_TAG}"
CFG_DIR="${SWEEP_DIR}/configs"
RUN_DIR="${SWEEP_DIR}/runs"
MASTER_CSV="${SWEEP_DIR}/master_results.csv"
SWEEP_LOG="${SWEEP_DIR}/sweep.log"
WORKER_DIR="${SWEEP_DIR}/workers"

if ! $DRY_RUN; then
    mkdir -p "${CFG_DIR}" "${RUN_DIR}" "${WORKER_DIR}"
    exec > >(tee -a "${SWEEP_LOG}") 2>&1
fi

N_COMBOS=$(( ${#DECAYS[@]} * ${#NUM_CODES_LIST[@]} * ${#COMMIT_WEIGHTS[@]} ))

echo "============================================================"
echo "  EMA Extended Sweep: ${SWEEP_TAG}"
echo "============================================================"
echo "  GPUs:           [${GPUS[*]}]"
echo "  Jobs per GPU:   ${JOBS_PER_GPU}"
echo "  Total workers:  ${N_WORKERS}"
echo "  Base pretrain:  ${BASE_PRETRAIN}"
echo "  Base skin3:     ${BASE_SKIN}"
echo "  Base scp1884:   ${BASE_SCP}"
echo "  Sweep dir:      ${SWEEP_DIR}"
[ -n "${EXISTING_SWEEP_DIR}" ] && echo "  Existing sweep:  ${EXISTING_SWEEP_DIR}"
echo "  Grid:"
echo "    ema_decay          = [${DECAYS[*]}]"
echo "    num_codes          = [${NUM_CODES_LIST[*]}]"
echo "    commitment_weight  = [${COMMIT_WEIGHTS[*]}]"
[ -n "${LOSS_TYPE}" ] && echo "    loss_type override = ${LOSS_TYPE}"
[ -n "${INPUT_TRANSFORM}" ] && echo "    input_xform override = ${INPUT_TRANSFORM}"
[ -n "${EPOCHS}" ] && echo "    epochs override    = ${EPOCHS}"
echo "  Total combos:   ${N_COMBOS}"
$DRY_RUN && echo "  *** DRY RUN — no experiments will be executed ***"
echo "============================================================"

# --- dry run: list grid and exit --------------------------------------------
if $DRY_RUN; then
    echo ""
    echo "Experiment grid (${N_COMBOS} combos → ${N_WORKERS} workers):"
    idx=0
    for decay in "${DECAYS[@]}"; do
      for codes in "${NUM_CODES_LIST[@]}"; do
        for commit in "${COMMIT_WEIGHTS[@]}"; do
          idx=$((idx+1))
          w_id=$(( (idx - 1) % N_WORKERS ))
          gpu_id=${GPUS[$(( w_id / JOBS_PER_GPU ))]}
          d_s=$(printf "%s" "$decay" | tr '.' '_')
          c_s=$(printf "%d" "$codes")
          w_s=$(printf "%s" "$commit" | tr '.' '_')
          tag="d${d_s}_c${c_s}_w${w_s}"
          echo "  [${idx}/${N_COMBOS}] ${tag}  gpu=${gpu_id}"
        done
      done
    done
    echo ""
    echo "Done (dry run). Use without --dry_run to execute."
    exit 0
fi

# --- Master CSV header ------------------------------------------------------
CSV_HEADER="exp_tag,ema_decay,num_codes,commit_weight,loss_type,input_transform,epochs,status,skin3_auc,skin3_f1,scp1884_auc,scp1884_f1,cross_s2c_auc,cross_s2c_f1,cross_c2s_auc,cross_c2s_f1,run_dir"
if [ ! -f "${MASTER_CSV}" ]; then
    echo "${CSV_HEADER}" > "${MASTER_CSV}"
fi

# Import existing sweep results
if [ -n "${EXISTING_SWEEP_DIR}" ] && [ -f "${EXISTING_SWEEP_DIR}/master_results.csv" ]; then
    echo "[INFO] Importing previous results from ${EXISTING_SWEEP_DIR}/master_results.csv"
    tail -n +2 "${EXISTING_SWEEP_DIR}/master_results.csv" >> "${MASTER_CSV}"
    N_IMPORTED=$(tail -n +2 "${EXISTING_SWEEP_DIR}/master_results.csv" | wc -l | tr -d ' ')
    echo "[INFO] Imported ${N_IMPORTED} existing results"
fi

# --- helpers ----------------------------------------------------------------
format_tag() {
    local d=$1 c=$2 w=$3
    d_s=$(printf "%s" "$d" | tr '.' '_')
    c_s=$(printf "%d" "$c")
    w_s=$(printf "%s" "$w" | tr '.' '_')
    echo "d${d_s}_c${c_s}_w${w_s}"
}

is_already_done() {
    local exp_tag=$1
    local exp_out_root="${RUN_DIR}/${exp_tag}"
    # Done in current sweep
    if [ -f "${exp_out_root}/pretrained/vq_aenb_conditional.pth" ]; then
        return 0
    fi
    # Done in existing sweep
    if [ -n "${EXISTING_SWEEP_DIR}" ] && \
       [ -f "${EXISTING_SWEEP_DIR}/runs/${exp_tag}/pretrained/vq_aenb_conditional.pth" ]; then
        return 0
    fi
    return 1
}

gen_configs() {
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

    emit_encoder_overlay() {
        local out_path=$1 out_root=$2
        cat >> "${out_path}" <<EOF
paths:
  pretrained_encoder: "${exp_out_root}/pretrained/vq_aenb_conditional.pth"
  output_root: "${out_root}"

data:
  conditional_embedding:
    mapping_path: "${exp_out_root}/pretrained/study_mapping.json"

encoder:
  num_codes: ${codes}
EOF
        [ -n "${loss_line}" ] && echo "${loss_line}" >> "${out_path}"
        [ -n "${xform_line}" ] && echo "${xform_line}" >> "${out_path}"
        cat >> "${out_path}" <<EOF
  quantizer:
    commitment_weight: ${commit}
    ema_decay: ${decay}
EOF
        if [ -n "${epochs_line}" ]; then
            cat >> "${out_path}" <<EOF
  pretrain:
${epochs_line}
EOF
        fi
    }

    # pretrain.yaml
    {
        echo "# Auto-generated by sweep_ema_extended.sh — exp ${exp_tag} / pretrain"
        echo "_base_: \"${BASE_PRETRAIN}\""
        echo ""
    } > "${exp_cfg_dir}/pretrain.yaml"
    emit_encoder_overlay "${exp_cfg_dir}/pretrain.yaml" "${exp_out_root}"

    # skin3.yaml
    {
        echo "# Auto-generated by sweep_ema_extended.sh — exp ${exp_tag} / skin3"
        echo "_base_: \"${BASE_PRETRAIN}\""
        echo ""
    } > "${exp_cfg_dir}/skin3.yaml"
    emit_encoder_overlay "${exp_cfg_dir}/skin3.yaml" "${exp_out_root}/skin3"
    cat >> "${exp_cfg_dir}/skin3.yaml" <<EOF

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

    # scp1884.yaml
    {
        echo "# Auto-generated by sweep_ema_extended.sh — exp ${exp_tag} / scp1884"
        echo "_base_: \"${BASE_PRETRAIN}\""
        echo ""
    } > "${exp_cfg_dir}/scp1884.yaml"
    emit_encoder_overlay "${exp_cfg_dir}/scp1884.yaml" "${exp_out_root}/scp1884"
    cat >> "${exp_cfg_dir}/scp1884.yaml" <<EOF

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

read_cross_metrics() {
    local csv=$1
    python - "$csv" <<'PY'
import sys, pandas as pd, math
p = sys.argv[1]
try:
    df = pd.read_csv(p)
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

# ============================================================================
# Build task list & distribute to workers (round-robin)
# ============================================================================
echo ""
echo "[PREP] Building task list and distributing to ${N_WORKERS} workers …"

# Collect all tasks into an array
ALL_TASKS=()
TOTAL=0; SKIP=0
for decay in "${DECAYS[@]}"; do
  for codes in "${NUM_CODES_LIST[@]}"; do
    for commit in "${COMMIT_WEIGHTS[@]}"; do
      TOTAL=$((TOTAL+1))
      exp_tag=$(format_tag "$decay" "$codes" "$commit")

      if $SKIP_EXISTING && is_already_done "$exp_tag"; then
          echo "  [SKIP] ${exp_tag}"
          SKIP=$((SKIP+1))
          continue
      fi

      # Generate configs eagerly (fast, no GPU needed)
      gen_configs "$exp_tag" "$decay" "$codes" "$commit"

      ALL_TASKS+=("${exp_tag} ${decay} ${codes} ${commit}")
    done
  done
done

N_TASKS=${#ALL_TASKS[@]}
echo "[PREP] ${N_TASKS} tasks to run (${SKIP} skipped of ${TOTAL} total)"

if [ "${N_TASKS}" -eq 0 ]; then
    echo "[DONE] All experiments already completed."
    exit 0
fi

# Write per-worker task files (round-robin distribution)
# Build GPU assignment array: [gpu0, gpu0, gpu1, gpu1, gpu2, gpu2, ...]
WORKER_GPUS=()
for gpu in "${GPUS[@]}"; do
    for ((j=0; j<JOBS_PER_GPU; j++)); do
        WORKER_GPUS+=("${gpu}")
    done
done

for ((w=0; w<N_WORKERS; w++)); do
    > "${WORKER_DIR}/tasks_${w}.txt"
done

for ((i=0; i<N_TASKS; i++)); do
    w=$((i % N_WORKERS))
    echo "${ALL_TASKS[$i]}" >> "${WORKER_DIR}/tasks_${w}.txt"
done

echo ""
echo "[PREP] Task distribution:"
for ((w=0; w<N_WORKERS; w++)); do
    n=$(wc -l < "${WORKER_DIR}/tasks_${w}.txt" | tr -d ' ')
    echo "  Worker ${w} (GPU ${WORKER_GPUS[$w]}): ${n} tasks"
done

# ============================================================================
# Worker function — processes its own pre-assigned task file
# ============================================================================
run_worker() {
    local worker_id=$1
    local gpu_id=$2
    local task_file="${WORKER_DIR}/tasks_${worker_id}.txt"
    local worker_log="${WORKER_DIR}/worker_${worker_id}_gpu${gpu_id}.log"

    echo "[Worker ${worker_id}] Started on GPU ${gpu_id}, log: ${worker_log}"

    local ok=0 fail=0 done_count=0

    while IFS=' ' read -r exp_tag decay codes commit; do
        [ -z "${exp_tag}" ] && continue
        done_count=$((done_count+1))
        local exp_cfg_dir="${CFG_DIR}/${exp_tag}"
        local exp_out_root="${RUN_DIR}/${exp_tag}"

        echo "[Worker ${worker_id}] [${done_count}] ${exp_tag} (GPU ${gpu_id})" | tee -a "${worker_log}"

        # Run experiment with CUDA_VISIBLE_DEVICES
        local status
        if CUDA_VISIBLE_DEVICES="${gpu_id}" bash scripts/run_ema_small.sh \
                --gpu 0 \
                --pretrain_cfg "${exp_cfg_dir}/pretrain.yaml" \
                --skin3_cfg    "${exp_cfg_dir}/skin3.yaml" \
                --scp_cfg      "${exp_cfg_dir}/scp1884.yaml" \
                >> "${worker_log}" 2>&1; then
            status="OK"; ok=$((ok+1))
        else
            status="FAIL"; fail=$((fail+1))
        fi

        # Scrape metrics
        local skin_cv scp_cv skin_fin scp_fin s2c_dir c2s_dir
        skin_cv=$(latest_dir "${exp_out_root}/skin3" "cv_*")
        scp_cv=$(latest_dir "${exp_out_root}/scp1884" "cv_*")
        skin_fin=$(latest_dir "${exp_out_root}/skin3" "final_model_*")
        scp_fin=$(latest_dir "${exp_out_root}/scp1884" "final_model_*")
        s2c_dir=""
        c2s_dir=""
        [ -n "${skin_fin}" ] && s2c_dir=$(latest_dir "${skin_fin}" "cross_eval_*")
        [ -n "${scp_fin}" ]  && c2s_dir=$(latest_dir "${scp_fin}" "cross_eval_*")

        local skin_m="," scp_m="," s2c_m="," c2s_m=","
        [ -n "${skin_cv}"  ] && [ -f "${skin_cv}/overall_results.csv"  ] && skin_m=$(read_cv_metrics "${skin_cv}/overall_results.csv")
        [ -n "${scp_cv}"   ] && [ -f "${scp_cv}/overall_results.csv"   ] && scp_m=$(read_cv_metrics "${scp_cv}/overall_results.csv")
        [ -n "${s2c_dir}"  ] && [ -f "${s2c_dir}/cross_eval_results.csv" ] && s2c_m=$(read_cross_metrics "${s2c_dir}/cross_eval_results.csv")
        [ -n "${c2s_dir}"  ] && [ -f "${c2s_dir}/cross_eval_results.csv" ] && c2s_m=$(read_cross_metrics "${c2s_dir}/cross_eval_results.csv")

        # Append to master CSV (>> is atomic for lines < PIPE_BUF on Linux)
        echo "${exp_tag},${decay},${codes},${commit},${LOSS_TYPE:-inherit},${INPUT_TRANSFORM:-inherit},${EPOCHS:-inherit},${status},${skin_m},${scp_m},${s2c_m},${c2s_m},${exp_out_root}" \
            >> "${MASTER_CSV}"

        echo "[Worker ${worker_id}] ${exp_tag} → ${status}  skin=(${skin_m})  scp=(${scp_m})" | tee -a "${worker_log}"
    done < "${task_file}"

    echo "[Worker ${worker_id}] Finished. OK=${ok} FAIL=${fail}" | tee -a "${worker_log}"
    return ${fail}
}

# ============================================================================
# Launch workers
# ============================================================================
echo ""
echo "[LAUNCH] Starting ${N_WORKERS} workers across GPUs [${GPUS[*]}] (${JOBS_PER_GPU} per GPU) …"
echo ""

WORKER_PIDS=()
for ((w=0; w<N_WORKERS; w++)); do
    run_worker "${w}" "${WORKER_GPUS[$w]}" &
    WORKER_PIDS+=($!)
done

# Wait for all workers; track failures
TOTAL_FAIL=0
for pid in "${WORKER_PIDS[@]}"; do
    if ! wait "${pid}"; then
        TOTAL_FAIL=$((TOTAL_FAIL+1))
    fi
done

# Count results from master CSV
N_OK=$(grep -c ",OK," "${MASTER_CSV}" 2>/dev/null || echo 0)
N_FAIL=$(grep -c ",FAIL," "${MASTER_CSV}" 2>/dev/null || echo 0)

echo ""
echo "============================================================"
echo "  Extended Sweep done"
echo "============================================================"
echo "  Total combos: ${TOTAL}   Skipped: ${SKIP}   Ran: ${N_TASKS}"
echo "  OK: ${N_OK}   FAIL: ${N_FAIL}"
echo "  Workers: ${N_WORKERS} (GPUs: [${GPUS[*]}] × ${JOBS_PER_GPU}/GPU)"
echo "  Master CSV: ${MASTER_CSV}"
echo "  Sweep log:  ${SWEEP_LOG}"
echo "  Worker logs: ${WORKER_DIR}/"
echo "============================================================"

[ "${N_FAIL}" -eq 0 ] || exit 1
