#!/usr/bin/env bash
# Session 04 — Unheterodyned IMRPhenomD_NRTidalv2 at n_live=2500.
# This is the headline single-run unheterodyned cross-check at the session maximum.
# Estimated wall-clock: ~9.5 h. Runs alone, fills the session.

. "$(dirname "$0")/_common.sh"
init_session "session_04_unhet_imr_nrt_2500"

SCRIPT="${REPO_ROOT}/pipeline/GW170817_unheterodyned_1.py"
WAVEFORM="IMRPhenomD_NRTidalv2"
N_LIVE=2500
NUM_DELETE=1250

BUDGET_SECONDS=39600  # 11 hours

RUN_ID="s04__gw170817__imrphenomd_nrtidalv2__unheterodyned__nlive$(printf '%05d' ${N_LIVE})__seed0000"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"

echo "--- [${RUN_ID}] n_live=${N_LIVE} --- $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
mkdir -p "${RUN_DIR}"
write_config_json "${RUN_ID}" "${SCRIPT}" "${WAVEFORM}" "unheterodyned" \
    "${N_LIVE}" "${NUM_DELETE}" 259201 0 ""

bank_check "${BUDGET_SECONDS}" || { echo "Session already over budget; aborting." | tee -a "${SESSION_LOG}"; exit 1; }

${PYTHON} "${SCRIPT}" \
    --waveform "${WAVEFORM}" \
    --data-source local \
    --psd-source gwtc1 \
    --phase-marginalization \
    --nlive "${N_LIVE}" \
    --output-dir "${RUN_DIR}" \
    >> "${RUN_DIR}/sampler.log" 2>&1
RC=$?

canonicalise_csv "${RUN_ID}" "${RUN_DIR}/PhaseMarg_Unheterodyned_*IMRPhenomD_NRTidalv2*.csv" || RC=1
finalise_run "${RUN_ID}" "${RC}"

finalise_session
