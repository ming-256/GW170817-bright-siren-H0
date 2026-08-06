#!/usr/bin/env bash
# Session 11 — n_live=20000 scaling-anomaly diagnostic.
# Re-runs n_live=20000 with a tighter termination tolerance to test whether the
# low dead-point count observed in scaling_summary.csv is driven by early termination.
# PREREQUISITE: patch P-TERM (see CODE_CHANGES_NEEDED.md §6).
# Estimated wall-clock: ~1 h.

. "$(dirname "$0")/_common.sh"
init_session "session_11_scaling_20k_diagnostic"

SCRIPT="${REPO_ROOT}/pipeline/GW170817_heterodyned_1.py"
BUDGET_SECONDS=7200

if ! grep -q "tolerance" "${SCRIPT}" 2>/dev/null; then
    echo "!! Patch P-TERM not applied. See CODE_CHANGES_NEEDED.md §6." | tee -a "${SESSION_LOG}"
    exit 2
fi

RUN_ID="s11__gw170817__imrphenomd_nrtidalv2__baseline__nlive20000__tol1e-4__seed0000"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
echo "--- [${RUN_ID}] n_live=20000 tol=1e-4 --- $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
mkdir -p "${RUN_DIR}"
write_config_json "${RUN_ID}" "${SCRIPT}" "IMRPhenomD_NRTidalv2" "baseline_tol1e-4" 20000 10000 501 0 "--tolerance 1e-4"

bank_check "${BUDGET_SECONDS}" && {
    ${PYTHON} "${SCRIPT}" --waveform IMRPhenomD_NRTidalv2 \
        --data-source local --psd-source gwtc1 --ref-params gwtc1 \
        --phase-marginalization --n-live 20000 --tolerance 1e-4 \
        --output-dir "${RUN_DIR}" >> "${RUN_DIR}/sampler.log" 2>&1
    RC=$?
    canonicalise_csv "${RUN_ID}" "${RUN_DIR}/PhaseMarg_*.csv" || RC=1
    finalise_run "${RUN_ID}" "${RC}"
}

finalise_session
