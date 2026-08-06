#!/usr/bin/env bash
# Session 07b — GW170817 TaylorF2 on the LVK-bounds prior set.
#
# Adds the missing TF2 sibling to the s07 family (IMRPhenomD_NRTidalv2,
# IMRPhenomXAS_NRTidalv3, IMRPhenomPv2 already at LVK-bounds; TF2 was only
# at default-mass). This is required so that Figure 2 (H0_waveform_comparison)
# and Table 4 (tab:waveform-h0) report TF2 on the same prior set as XAS/IMR
# rather than on the default-mass prior.
#
# Settings match session_07 exactly:
#   n_live=5000, num_delete=2500, n_bins=501, seed=0
#   --m-comp-lo 0.87 --m-comp-hi 1.74  (LVK low-spin BNS bounds)
#   --data-source local --psd-source gwtc1 --ref-params gwtc1 --phase-marginalization
#
# Estimated wall-clock: ~9 min on A100 (matches §4.1 TF2 figure).
# Idempotent: skips if Results/test_suite/<run_id>/samples.csv already exists.

. "$(dirname "$0")/_common.sh"
init_session "session_07b_tf2_lvkbounds"

HET_BASELINE="${REPO_ROOT}/pipeline/GW170817_heterodyned_1.py"

# Guard: same patches as session_07.
if ! grep -q "m-comp-lo" "${HET_BASELINE}" 2>/dev/null; then
    echo "!! Patch P-MASSBOUNDS not applied. See CODE_CHANGES_NEEDED.md §7." | tee -a "${SESSION_LOG}"
    exit 2
fi

LVK_M_LO=0.87
LVK_M_HI=1.74

WAVEFORM="TaylorF2"
TAG="taylorf2"
RUN_ID="s07__gw170817__${TAG}__baseline_lvkbounds__seed0000"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"

if [[ -f "${RUN_DIR}/samples.csv" ]]; then
    echo "--- [${RUN_ID}] already complete (samples.csv exists) — skipping" | tee -a "${SESSION_LOG}"
    finalise_session
    exit 0
fi

echo "--- [${RUN_ID}] $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
mkdir -p "${RUN_DIR}"
write_config_json "${RUN_ID}" "${HET_BASELINE}" "${WAVEFORM}" "baseline_lvkbounds" 5000 2500 501 0 \
    "--m-comp-lo ${LVK_M_LO} --m-comp-hi ${LVK_M_HI}"

${PYTHON} "${HET_BASELINE}" \
    --waveform "${WAVEFORM}" \
    --data-source local --psd-source gwtc1 --ref-params gwtc1 \
    --phase-marginalization --n-live 5000 \
    --m-comp-lo "${LVK_M_LO}" --m-comp-hi "${LVK_M_HI}" \
    --output-dir "${RUN_DIR}" \
    >> "${RUN_DIR}/sampler.log" 2>&1
RC=$?
canonicalise_csv "${RUN_ID}" "${RUN_DIR}/PhaseMarg_*${WAVEFORM}*.csv" || RC=1
finalise_run "${RUN_ID}" "${RC}"

finalise_session
