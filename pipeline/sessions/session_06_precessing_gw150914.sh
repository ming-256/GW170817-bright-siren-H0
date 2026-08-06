#!/usr/bin/env bash
# Session 06 — GW150914 with IMRPhenomXPHM and LVK-matched mass bounds.
#
# Goal: test whether matching LVK-canonical mass bounds (m1, m2 in [5, 100])
# AND switching to the LVK production waveform (IMRPhenomXPHM) resolves the
# d_L offset (our IMRPhenomD posterior peaks ~30 Mpc below LVK).
#
# PREREQUISITES:
#   * patch P-WAV-GW150914  (CODE_CHANGES_NEEDED.md §1)
#   * patch P-MASSBOUNDS    (CODE_CHANGES_NEEDED.md §7)
#
# Estimated wall-clock: ~20 min on A100.

. "$(dirname "$0")/_common.sh"
init_session "session_06_precessing_gw150914"

SCRIPT="${REPO_ROOT}/pipeline/GW150914_heterodyned.py"
BUDGET_SECONDS=5400  # 1.5 hours

if ! grep -q "IMRPhenomXPHM" "${SCRIPT}" 2>/dev/null; then
    echo "!! Patch P-WAV-GW150914 not applied. See CODE_CHANGES_NEEDED.md §1." | tee -a "${SESSION_LOG}"
    exit 2
fi
if ! grep -q "m-comp-lo" "${SCRIPT}" 2>/dev/null; then
    echo "!! Patch P-MASSBOUNDS not applied. See CODE_CHANGES_NEEDED.md §7." | tee -a "${SESSION_LOG}"
    exit 2
fi

# LVK GW150914 BBH mass bounds (approximate canonical range).
LVK_M_LO=5.0
LVK_M_HI=100.0

WAVEFORM="IMRPhenomXPHM"
RUN_ID="s06__gw150914__imrphenomxphm__lvkbounds__seed0000"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
echo "--- [${RUN_ID}] $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
mkdir -p "${RUN_DIR}"
write_config_json "${RUN_ID}" "${SCRIPT}" "${WAVEFORM}" "baseline_lvkbounds" 5000 2500 501 0 \
    "--m-comp-lo ${LVK_M_LO} --m-comp-hi ${LVK_M_HI}"

bank_check "${BUDGET_SECONDS}" && {
    ${PYTHON} "${SCRIPT}" \
        --waveform "${WAVEFORM}" \
        --data-source local --psd-source gwtc2p1 --ref-params gwtc1 \
        --phase-marginalization --n-live 5000 \
        --m-comp-lo "${LVK_M_LO}" --m-comp-hi "${LVK_M_HI}" \
        --output-dir "${RUN_DIR}" \
        >> "${RUN_DIR}/sampler.log" 2>&1
    RC=$?
    canonicalise_csv "${RUN_ID}" "${RUN_DIR}/GW15_*.csv" || RC=1
    finalise_run "${RUN_ID}" "${RC}"
}

finalise_session
