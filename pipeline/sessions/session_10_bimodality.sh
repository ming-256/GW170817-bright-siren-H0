#!/usr/bin/env bash
# Session 10 — d_L–iota bimodality characterisation.
# PREREQUISITE: patch P-MODEB (see CODE_CHANGES_NEEDED.md §5).
# Estimated wall-clock: ~45 min (3 × ~15 min).

. "$(dirname "$0")/_common.sh"
init_session "session_10_bimodality"

SCRIPT="${REPO_ROOT}/pipeline/GW170817_heterodyned_2.py"   # flat-in-z
BUDGET_SECONDS=7200

if ! grep -q "dl-min\|dl-max" "${SCRIPT}" 2>/dev/null; then
    echo "!! Patch P-MODEB not applied. See CODE_CHANGES_NEEDED.md §5." | tee -a "${SESSION_LOG}"
    exit 2
fi

# Mode B: narrow prior on d_L = [10, 30] Mpc
RUN_ID="s10__gw170817__imrphenomd_nrtidalv2__flatz__dL10-30__refGWTC1__seed0000"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
echo "--- [${RUN_ID}] Mode-B targeted --- $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
mkdir -p "${RUN_DIR}"
write_config_json "${RUN_ID}" "${SCRIPT}" "IMRPhenomD_NRTidalv2" "flatZ_modeB" 5000 2500 501 0 "--dl-min 10 --dl-max 30"
bank_check "${BUDGET_SECONDS}" && {
    ${PYTHON} "${SCRIPT}" --waveform IMRPhenomD_NRTidalv2 \
        --data-source local --psd-source gwtc1 --ref-params gwtc1 \
        --phase-marginalization --n-live 5000 \
        --dl-min 10 --dl-max 30 \
        --output-dir "${RUN_DIR}" >> "${RUN_DIR}/sampler.log" 2>&1
    RC=$?
    canonicalise_csv "${RUN_ID}" "${RUN_DIR}/PhaseMarg_*.csv" || RC=1
    finalise_run "${RUN_ID}" "${RC}"
}

# Mode A: narrow prior on d_L = [30, 75] Mpc
RUN_ID="s10__gw170817__imrphenomd_nrtidalv2__flatz__dL30-75__refGWTC1__seed0000"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
echo "--- [${RUN_ID}] Mode-A targeted --- $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
mkdir -p "${RUN_DIR}"
write_config_json "${RUN_ID}" "${SCRIPT}" "IMRPhenomD_NRTidalv2" "flatZ_modeA" 5000 2500 501 0 "--dl-min 30 --dl-max 75"
bank_check "${BUDGET_SECONDS}" && {
    ${PYTHON} "${SCRIPT}" --waveform IMRPhenomD_NRTidalv2 \
        --data-source local --psd-source gwtc1 --ref-params gwtc1 \
        --phase-marginalization --n-live 5000 \
        --dl-min 30 --dl-max 75 \
        --output-dir "${RUN_DIR}" >> "${RUN_DIR}/sampler.log" 2>&1
    RC=$?
    canonicalise_csv "${RUN_ID}" "${RUN_DIR}/PhaseMarg_*.csv" || RC=1
    finalise_run "${RUN_ID}" "${RC}"
}

# Reference-parameter swap: full d_L range, but heterodyne reference anchored in Mode B.
RUN_ID="s10__gw170817__imrphenomd_nrtidalv2__flatz__dL10-75__refModeB__seed0000"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
echo "--- [${RUN_ID}] Mode-B reference swap --- $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
mkdir -p "${RUN_DIR}"
write_config_json "${RUN_ID}" "${SCRIPT}" "IMRPhenomD_NRTidalv2" "flatZ_refModeB" 5000 2500 501 0 "--ref-dl 20 --ref-iota 2.0"
bank_check "${BUDGET_SECONDS}" && {
    ${PYTHON} "${SCRIPT}" --waveform IMRPhenomD_NRTidalv2 \
        --data-source local --psd-source gwtc1 --ref-params gwtc1 \
        --phase-marginalization --n-live 5000 \
        --ref-dl 20.0 --ref-iota 2.0 \
        --output-dir "${RUN_DIR}" >> "${RUN_DIR}/sampler.log" 2>&1
    RC=$?
    canonicalise_csv "${RUN_ID}" "${RUN_DIR}/PhaseMarg_*.csv" || RC=1
    finalise_run "${RUN_ID}" "${RC}"
}

finalise_session
