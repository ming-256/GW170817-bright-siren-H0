#!/usr/bin/env bash
# Session 08 — num_delete sweep at fixed n_live=5000.
# PREREQUISITE: patch P-NDELETE (see CODE_CHANGES_NEEDED.md §3).
# Estimated wall-clock: ~1.5 h.

. "$(dirname "$0")/_common.sh"
init_session "session_08_num_delete_sweep"

SCRIPT="${REPO_ROOT}/pipeline/GW170817_heterodyned_1.py"
BUDGET_SECONDS=10800  # 3 hours

# Guard
if ! grep -q "num-delete" "${SCRIPT}" 2>/dev/null; then
    echo "!! Patch P-NDELETE not applied. See CODE_CHANGES_NEEDED.md §3." | tee -a "${SESSION_LOG}"
    exit 2
fi

for ND in 500 1250 2500 3750; do
    bank_check "${BUDGET_SECONDS}" || break
    RUN_ID="s08__gw170817__imrphenomd_nrtidalv2__baseline__ndelete$(printf '%05d' ${ND})__seed0000"
    RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
    echo "--- [${RUN_ID}] num_delete=${ND} --- $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
    mkdir -p "${RUN_DIR}"
    write_config_json "${RUN_ID}" "${SCRIPT}" "IMRPhenomD_NRTidalv2" "baseline_nd${ND}" 5000 "${ND}" 501 0 "--num-delete ${ND}"

    ${PYTHON} "${SCRIPT}" \
        --waveform IMRPhenomD_NRTidalv2 \
        --data-source local --psd-source gwtc1 --ref-params gwtc1 \
        --phase-marginalization --n-live 5000 --num-delete "${ND}" \
        --output-dir "${RUN_DIR}" \
        >> "${RUN_DIR}/sampler.log" 2>&1
    RC=$?
    canonicalise_csv "${RUN_ID}" "${RUN_DIR}/PhaseMarg_*IMRPhenomD_NRTidalv2*.csv" || RC=1
    finalise_run "${RUN_ID}" "${RC}"
done

finalise_session
