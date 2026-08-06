#!/usr/bin/env bash
# Session 05 — Unheterodyned IMRPhenomD_NRTidalv2 at n_live=500 plus short supplementary heterodyned runs.
# Budget: ~2.5 h (IMR unhet 500 ~1.8 h, plus three heterodyned runs totalling ~30 min).

. "$(dirname "$0")/_common.sh"
init_session "session_05_unhet_imr_nrt_500_plus"

UNHET_SCRIPT="${REPO_ROOT}/pipeline/GW170817_unheterodyned_1.py"
HET_SCRIPT="${REPO_ROOT}/pipeline/GW170817_heterodyned_1.py"

BUDGET_SECONDS=14400  # 4 hours

# ---- Run 1: unheterodyned IMRPhenomD_NRTidalv2 at n_live=500 ----
RUN_ID="s05__gw170817__imrphenomd_nrtidalv2__unheterodyned__nlive00500__seed0000"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
echo "--- [${RUN_ID}] unhet IMR_NRT n_live=500 --- $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
mkdir -p "${RUN_DIR}"
write_config_json "${RUN_ID}" "${UNHET_SCRIPT}" "IMRPhenomD_NRTidalv2" "unheterodyned" 500 250 259201 0 ""

bank_check "${BUDGET_SECONDS}" || exit 1
${PYTHON} "${UNHET_SCRIPT}" \
    --waveform IMRPhenomD_NRTidalv2 \
    --data-source local --psd-source gwtc1 \
    --phase-marginalization --nlive 500 \
    --output-dir "${RUN_DIR}" >> "${RUN_DIR}/sampler.log" 2>&1
RC=$?
canonicalise_csv "${RUN_ID}" "${RUN_DIR}/PhaseMarg_Unheterodyned_*IMRPhenomD_NRTidalv2*.csv" || RC=1
finalise_run "${RUN_ID}" "${RC}"

# ---- Run 2: heterodyned IMR baseline with ref-params=optimize (tests reference-parameter dependence) ----
RUN_ID="s05__gw170817__imrphenomd_nrtidalv2__baseline__refOptimize__seed0000"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
echo "--- [${RUN_ID}] heterodyned IMR baseline ref-params=optimize --- $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
mkdir -p "${RUN_DIR}"
write_config_json "${RUN_ID}" "${HET_SCRIPT}" "IMRPhenomD_NRTidalv2" "baseline_refOptimize" 5000 2500 501 0 "--ref-params optimize"

bank_check "${BUDGET_SECONDS}" || exit 0
${PYTHON} "${HET_SCRIPT}" \
    --waveform IMRPhenomD_NRTidalv2 \
    --data-source local --psd-source gwtc1 --ref-params optimize \
    --phase-marginalization --n-live 5000 \
    --output-dir "${RUN_DIR}" >> "${RUN_DIR}/sampler.log" 2>&1
RC=$?
canonicalise_csv "${RUN_ID}" "${RUN_DIR}/PhaseMarg_*IMRPhenomD_NRTidalv2*.csv" || RC=1
finalise_run "${RUN_ID}" "${RC}"

# ---- Run 3: TaylorF2 baseline with PSD source = kazewong (tests PSD sensitivity) ----
RUN_ID="s05__gw170817__taylorf2__baseline__psdKazewong__seed0000"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
echo "--- [${RUN_ID}] TF2 baseline PSD=kazewong --- $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
mkdir -p "${RUN_DIR}"
write_config_json "${RUN_ID}" "${HET_SCRIPT}" "TaylorF2" "baseline_psdKazewong" 5000 2500 501 0 "--psd-source kazewong"

bank_check "${BUDGET_SECONDS}" || exit 0
${PYTHON} "${HET_SCRIPT}" \
    --waveform TaylorF2 \
    --data-source local --psd-source kazewong --ref-params gwtc1 \
    --phase-marginalization --n-live 5000 \
    --output-dir "${RUN_DIR}" >> "${RUN_DIR}/sampler.log" 2>&1
RC=$?
canonicalise_csv "${RUN_ID}" "${RUN_DIR}/PhaseMarg_*TaylorF2*.csv" || RC=1
finalise_run "${RUN_ID}" "${RC}"

# ---- Run 4: TaylorF2 baseline with PSD source = bilby ----
RUN_ID="s05__gw170817__taylorf2__baseline__psdBilby__seed0000"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
echo "--- [${RUN_ID}] TF2 baseline PSD=bilby --- $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
mkdir -p "${RUN_DIR}"
write_config_json "${RUN_ID}" "${HET_SCRIPT}" "TaylorF2" "baseline_psdBilby" 5000 2500 501 0 "--psd-source bilby"

bank_check "${BUDGET_SECONDS}" || exit 0
${PYTHON} "${HET_SCRIPT}" \
    --waveform TaylorF2 \
    --data-source local --psd-source bilby --ref-params gwtc1 \
    --phase-marginalization --n-live 5000 \
    --output-dir "${RUN_DIR}" >> "${RUN_DIR}/sampler.log" 2>&1
RC=$?
canonicalise_csv "${RUN_ID}" "${RUN_DIR}/PhaseMarg_*TaylorF2*.csv" || RC=1
finalise_run "${RUN_ID}" "${RC}"

finalise_session
