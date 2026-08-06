#!/usr/bin/env bash
# Session 03 — Unheterodyned TaylorF2 scaling (500, 1500, 2500 live points).
# Estimated wall-clock: ~50 + ~150 + ~250 min = ~7.5 h.
# Fits comfortably in one 12-hour session with margin.

. "$(dirname "$0")/_common.sh"
init_session "session_03_unhet_tf2_scaling"

SCRIPT="${REPO_ROOT}/pipeline/GW170817_unheterodyned_1.py"
WAVEFORM="TaylorF2"

BUDGET_SECONDS=39600  # 11 hours

for N_LIVE in 500 1500 2500; do
    bank_check "${BUDGET_SECONDS}" || break
    RUN_ID="s03__gw170817__taylorf2__unheterodyned__nlive$(printf '%05d' ${N_LIVE})__seed0000"
    RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
    NUM_DELETE=$(( N_LIVE / 2 ))

    echo "--- [${RUN_ID}] n_live=${N_LIVE} --- $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
    mkdir -p "${RUN_DIR}"
    write_config_json "${RUN_ID}" "${SCRIPT}" "${WAVEFORM}" "unheterodyned" \
        "${N_LIVE}" "${NUM_DELETE}" 259201 0 ""

    ${PYTHON} "${SCRIPT}" \
        --waveform "${WAVEFORM}" \
        --data-source local \
        --psd-source gwtc1 \
        --phase-marginalization \
        --nlive "${N_LIVE}" \
        --output-dir "${RUN_DIR}" \
        >> "${RUN_DIR}/sampler.log" 2>&1
    RC=$?

    canonicalise_csv "${RUN_ID}" "${RUN_DIR}/PhaseMarg_Unheterodyned_*TaylorF2*.csv" || RC=1
    finalise_run "${RUN_ID}" "${RC}"
done

finalise_session
