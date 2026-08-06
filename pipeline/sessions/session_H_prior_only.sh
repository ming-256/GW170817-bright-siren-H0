#!/usr/bin/env bash
# Session H — Prior-only q-diagnostic. CPU only. No GPU needed. Runs in <1 min.

. "$(dirname "$0")/_common.sh"
init_session "session_H_prior_only"

RUN_ID="sH__gw170817__prior_only_q__seed0000"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"
write_config_json "${RUN_ID}" \
    "pipeline/sessions/prior_only_q_diagnostic.py" \
    "n/a" "prior_only" 5000 0 0 0 ""

echo "--- [${RUN_ID}] $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"

${PYTHON} "${TEST_SUITE_ROOT}/prior_only_q_diagnostic.py" \
    --n-samples 200000 \
    --out-dir "${RUN_DIR}" \
    >> "${RUN_DIR}/sampler.log" 2>&1
RC=$?
finalise_run "${RUN_ID}" "${RC}"

finalise_session
