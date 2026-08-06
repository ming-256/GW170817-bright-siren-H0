#!/usr/bin/env bash
# Session 17 — GW150914 IMRPhenomXPHM with larger live points and MCMC steps.
#
# Two runs for convergence comparison against s06 (n_live=5000, mcmc_steps=112):
#   s17a: n_live=8000,  n_mcmc_steps=160  (reduced from 210 — runtime too long)
#   s17b: n_live=10000, n_mcmc_steps=210  (reduced from 280)
#
# All other settings match s06: LVK mass bounds [5, 100], gwtc2p1 PSD, gwtc1 ref,
# local data, phase-marginalization ON.

. "$(dirname "$0")/_common.sh"
init_session "session_17_gw150914_xphm_larger"

SCRIPT="${REPO_ROOT}/pipeline/GW150914_heterodyned.py"
WAVEFORM="IMRPhenomXPHM"
LVK_M_LO=5.0
LVK_M_HI=100.0

# --- Run A: n_live=8000, n_mcmc_steps=160 ---
RUN_ID_A="s17a__gw150914__imrphenomxphm__nlive8000_mcmc160__seed0000"
RUN_DIR_A="${RESULTS_ROOT}/${RUN_ID_A}"
echo "--- [${RUN_ID_A}] $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
mkdir -p "${RUN_DIR_A}"
write_config_json "${RUN_ID_A}" "${SCRIPT}" "${WAVEFORM}" "nlive8000_mcmc160" 8000 2400 501 0 \
    "--m-comp-lo ${LVK_M_LO} --m-comp-hi ${LVK_M_HI} --n-mcmc-steps 160"

${PYTHON} "${SCRIPT}" \
    --waveform "${WAVEFORM}" \
    --data-source local --psd-source gwtc2p1 --ref-params gwtc1 \
    --phase-marginalization \
    --n-live 8000 --n-mcmc-steps 160 \
    --m-comp-lo "${LVK_M_LO}" --m-comp-hi "${LVK_M_HI}" \
    --output-dir "${RUN_DIR_A}" \
    2>&1 | tee "${RUN_DIR_A}/sampler.log"
RC_A=$?
canonicalise_csv "${RUN_ID_A}" "${RUN_DIR_A}/GW15_*.csv" || RC_A=1
finalise_run "${RUN_ID_A}" "${RC_A}"

# --- Run B: n_live=10000, n_mcmc_steps=210 ---
RUN_ID_B="s17b__gw150914__imrphenomxphm__nlive10000_mcmc210__seed0000"
RUN_DIR_B="${RESULTS_ROOT}/${RUN_ID_B}"
echo "--- [${RUN_ID_B}] $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
mkdir -p "${RUN_DIR_B}"
write_config_json "${RUN_ID_B}" "${SCRIPT}" "${WAVEFORM}" "nlive10000_mcmc210" 10000 3000 501 0 \
    "--m-comp-lo ${LVK_M_LO} --m-comp-hi ${LVK_M_HI} --n-mcmc-steps 210"

${PYTHON} "${SCRIPT}" \
    --waveform "${WAVEFORM}" \
    --data-source local --psd-source gwtc2p1 --ref-params gwtc1 \
    --phase-marginalization \
    --n-live 10000 --n-mcmc-steps 210 \
    --m-comp-lo "${LVK_M_LO}" --m-comp-hi "${LVK_M_HI}" \
    --output-dir "${RUN_DIR_B}" \
    2>&1 | tee "${RUN_DIR_B}/sampler.log"
RC_B=$?
canonicalise_csv "${RUN_ID_B}" "${RUN_DIR_B}/GW15_*.csv" || RC_B=1
finalise_run "${RUN_ID_B}" "${RC_B}"

finalise_session
