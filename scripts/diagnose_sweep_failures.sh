#!/bin/bash
# ============================================================================
# Sweep 실패 원인 진단
# ============================================================================
# Usage:
#   bash scripts/diagnose_sweep_failures.sh /path/to/sweep_dir
#   bash scripts/diagnose_sweep_failures.sh /path/to/sweep_dir --verbose
# ============================================================================

set -u

SWEEP_DIR="${1:?Usage: $0 <sweep_dir> [--verbose]}"
VERBOSE=false
[ "${2:-}" = "--verbose" ] && VERBOSE=true

MASTER_CSV="${SWEEP_DIR}/master_results.csv"
RUN_DIR="${SWEEP_DIR}/runs"

if [ ! -f "${MASTER_CSV}" ]; then
    echo "[ERROR] master_results.csv not found: ${MASTER_CSV}"
    exit 1
fi

# Count stats
TOTAL=$(tail -n +2 "${MASTER_CSV}" | wc -l | tr -d ' ')
N_OK=$(grep -c ",OK," "${MASTER_CSV}" || echo 0)
N_FAIL=$(grep -c ",FAIL," "${MASTER_CSV}" || echo 0)

echo "============================================================"
echo "  Sweep Failure Diagnosis"
echo "============================================================"
echo "  Sweep dir: ${SWEEP_DIR}"
echo "  Total: ${TOTAL}   OK: ${N_OK}   FAIL: ${N_FAIL}"
echo "============================================================"
echo ""

# Extract FAIL experiments
FAIL_TAGS=$(grep ",FAIL," "${MASTER_CSV}" | cut -d',' -f1)

if [ -z "${FAIL_TAGS}" ]; then
    echo "No failures found!"
    exit 0
fi

# Categorize failures
declare -A ERROR_CATEGORIES
declare -A ERROR_EXAMPLES

for tag in ${FAIL_TAGS}; do
    exp_run="${RUN_DIR}/${tag}"

    # Find the latest log dir
    log_dir=$(ls -1dt "${exp_run}/logs/run_"* 2>/dev/null | head -1)
    if [ -z "${log_dir}" ]; then
        echo "[${tag}] No log directory found"
        continue
    fi

    # Check each step's status from pipeline.log
    pipeline_log="${log_dir}/pipeline.log"
    failed_steps=""
    error_msg=""

    for step in pretrain cv_skin3 cv_scp1884 final_skin3 final_scp1884 cross_skin3_to_scp1884 cross_scp1884_to_skin3; do
        step_log="${log_dir}/${step}.log"
        [ ! -f "${step_log}" ] && continue

        # Check if step failed (look for common error patterns)
        if grep -q "Error\|Exception\|FATAL\|Traceback" "${step_log}" 2>/dev/null; then
            failed_steps="${failed_steps} ${step}"

            # Extract the last error line
            last_error=$(grep -E "Error:|Exception:|FATAL" "${step_log}" | tail -1 | sed 's/^[[:space:]]*//')
            if [ -z "${error_msg}" ]; then
                error_msg="${last_error}"
            fi
        fi
    done

    # Also check pipeline.log for FAIL steps
    if [ -f "${pipeline_log}" ]; then
        pipeline_fails=$(grep "FAIL" "${pipeline_log}" | grep -oP '>>> \K\w+' | tr '\n' ' ')
        if [ -n "${pipeline_fails}" ]; then
            failed_steps="${pipeline_fails}"
        fi
        # Get fail summary from pipeline.log
        fail_summary=$(grep "FAIL steps:" "${pipeline_log}" | tail -1)
    fi

    # Categorize the error
    category="unknown"
    if echo "${error_msg}" | grep -qi "OutOfMemory\|CUDA out of memory"; then
        category="CUDA_OOM"
    elif echo "${error_msg}" | grep -qi "FileNotFoundError\|No such file"; then
        category="FILE_NOT_FOUND"
    elif echo "${error_msg}" | grep -qi "NaN\|nan"; then
        category="NAN"
    elif echo "${error_msg}" | grep -qi "KeyboardInterrupt"; then
        category="INTERRUPTED"
    elif echo "${error_msg}" | grep -qi "RuntimeError"; then
        category="RUNTIME_ERROR"
    elif [ -n "${error_msg}" ]; then
        category="OTHER"
    fi

    ERROR_CATEGORIES["${category}"]=$(( ${ERROR_CATEGORIES["${category}"]:-0} + 1 ))
    ERROR_EXAMPLES["${category}"]="${tag}"

    # Print per-experiment summary
    echo "--- ${tag} ---"
    echo "  Failed steps: ${failed_steps:-unknown}"
    echo "  Category:     ${category}"
    echo "  Error:        ${error_msg:-<no error message found>}"

    if $VERBOSE; then
        # Show tail of each failed step log
        for step in pretrain cv_skin3 cv_scp1884 final_skin3 final_scp1884 cross_skin3_to_scp1884 cross_scp1884_to_skin3; do
            step_log="${log_dir}/${step}.log"
            if [ -f "${step_log}" ] && grep -q "Error\|Exception\|Traceback" "${step_log}" 2>/dev/null; then
                echo ""
                echo "  [${step}] last 10 lines:"
                tail -n 10 "${step_log}" | sed 's/^/    /'
            fi
        done
    fi
    echo ""
done

# Summary by category
echo "============================================================"
echo "  Failure Summary by Category"
echo "============================================================"
for cat in "${!ERROR_CATEGORIES[@]}"; do
    count=${ERROR_CATEGORIES[$cat]}
    example=${ERROR_EXAMPLES[$cat]}
    echo "  ${cat}: ${count} failures (e.g. ${example})"
done
echo ""

# Show which metrics are missing (partially successful runs)
echo "============================================================"
echo "  Partial Success (some metrics missing)"
echo "============================================================"
tail -n +2 "${MASTER_CSV}" | while IFS=',' read -r tag decay codes commit loss xform epochs status s_auc s_f1 c_auc c_f1 s2c_auc s2c_f1 c2s_auc c2s_f1 run_dir; do
    missing=""
    [ -z "${s_auc}" ] && missing="${missing} skin3_auc"
    [ -z "${c_auc}" ] && missing="${missing} scp1884_auc"
    [ -z "${s2c_auc}" ] && missing="${missing} cross_s2c"
    [ -z "${c2s_auc}" ] && missing="${missing} cross_c2s"
    if [ -n "${missing}" ]; then
        echo "  ${tag} (${status}): missing${missing}"
    fi
done

echo ""
echo "============================================================"
echo "  Quick fix suggestions"
echo "============================================================"
if [ "${ERROR_CATEGORIES["CUDA_OOM"]:-0}" -gt 0 ]; then
    echo "  CUDA_OOM: Reduce --jobs_per_gpu to 1, or data is already on CPU (check scripts)"
fi
if [ "${ERROR_CATEGORIES["FILE_NOT_FOUND"]:-0}" -gt 0 ]; then
    echo "  FILE_NOT_FOUND: Likely cascading failure from upstream step (OOM/NaN). Fix upstream first."
fi
if [ "${ERROR_CATEGORIES["NAN"]:-0}" -gt 0 ]; then
    echo "  NAN: Increase batch_size or reduce learning_rate. Check dead code revival."
fi
if [ "${ERROR_CATEGORIES["INTERRUPTED"]:-0}" -gt 0 ]; then
    echo "  INTERRUPTED: Previous run was killed. Re-run with --skip_existing."
fi
