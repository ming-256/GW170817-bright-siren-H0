#!/usr/bin/env bash
# Session 01 — TaylorF2 heterodyned scaling (500 → 20000 live points).
# Requires no code changes. Estimated wall-clock: ~2.0 h on A100.

. "$(dirname "$0")/_common.sh"
init_session "session_01_tf2_scaling"

SCRIPT="${REPO_ROOT}/pipeline/GW170817_heterodyned_1.py"
WAVEFORM="TaylorF2"

# Time bank: 3 hours — we expect ~2h and leave margin.
BUDGET_SECONDS=10800

for N_LIVE in 500 1000 2500 5000 10000 20000; do
    bank_check "${BUDGET_SECONDS}" || break
    RUN_ID="s01__gw170817__taylorf2__nlive$(printf '%05d' ${N_LIVE})__seed0000"
    RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
    NUM_DELETE=$(( N_LIVE / 2 ))

    echo "--- [${RUN_ID}] n_live=${N_LIVE} --- $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"

    mkdir -p "${RUN_DIR}"
    write_config_json "${RUN_ID}" "${SCRIPT}" "${WAVEFORM}" "baseline" \
        "${N_LIVE}" "${NUM_DELETE}" 501 0 ""

    ${PYTHON} "${SCRIPT}" \
        --waveform "${WAVEFORM}" \
        --data-source local \
        --psd-source gwtc1 \
        --ref-params gwtc1 \
        --phase-marginalization \
        --n-live "${N_LIVE}" \
        --output-dir "${RUN_DIR}" \
        >> "${RUN_DIR}/sampler.log" 2>&1
    RC=$?

    # The sampler writes PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1.csv into RUN_DIR.
    canonicalise_csv "${RUN_ID}" "${RUN_DIR}/PhaseMarg_*TaylorF2*.csv" || RC=1
    finalise_run "${RUN_ID}" "${RC}"
done

finalise_session
