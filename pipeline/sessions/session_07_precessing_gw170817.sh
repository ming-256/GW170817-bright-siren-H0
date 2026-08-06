#!/usr/bin/env bash
# Session 07 — GW170817 q-resolution comparison.
#
# Goal: test whether matching LVK-canonical mass bounds (m1, m2 in [0.87, 1.74])
# resolves the q-posterior discrepancy seen in the corner_combined_waveforms.pdf
# figure. Then layer waveform comparisons on top.
#
# Strategy: each waveform is run ONCE in baseline (volumetric d_L) only — no
# flatZ, no vp250. Prior sensitivity has been studied separately on
# IMRPhenomD_NRTidalv2 already.
#
# PREREQUISITES:
#   * patch P-WAV-GW170817 (CODE_CHANGES_NEEDED.md §2)
#   * patch P-MASSBOUNDS  (CODE_CHANGES_NEEDED.md §7)
#
# Estimated wall-clock: 4 × ~15 min (~1.0 h) on A100.
# Each run uses LVK-matched component-mass bounds [0.87, 1.74] M_sun (low-spin).

. "$(dirname "$0")/_common.sh"
init_session "session_07_precessing_gw170817"

HET_BASELINE="${REPO_ROOT}/pipeline/GW170817_heterodyned_1.py"
BUDGET_SECONDS=10800  # 3 hours

# Guard: verify both patches are applied.
if ! grep -q -E "IMRPhenomXAS_NRTidalv3|IMRPhenomPv2" "${HET_BASELINE}" 2>/dev/null; then
    echo "!! Patch P-WAV-GW170817 not applied. See CODE_CHANGES_NEEDED.md §2." | tee -a "${SESSION_LOG}"
    exit 2
fi
if ! grep -q "m-comp-lo" "${HET_BASELINE}" 2>/dev/null; then
    echo "!! Patch P-MASSBOUNDS not applied. See CODE_CHANGES_NEEDED.md §7." | tee -a "${SESSION_LOG}"
    exit 2
fi

# LVK low-spin component-mass bounds (Abbott 2017b H0 paper).
LVK_M_LO=0.87
LVK_M_HI=1.74

run_gw170817() {
    local waveform="$1" tag="$2"
    local run_id="s07__gw170817__${tag}__baseline_lvkbounds__seed0000"
    local run_dir="${RESULTS_ROOT}/${run_id}"
    echo "--- [${run_id}] $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
    mkdir -p "${run_dir}"
    write_config_json "${run_id}" "${HET_BASELINE}" "${waveform}" "baseline_lvkbounds" 5000 2500 501 0 \
        "--m-comp-lo ${LVK_M_LO} --m-comp-hi ${LVK_M_HI}"

    ${PYTHON} "${HET_BASELINE}" \
        --waveform "${waveform}" \
        --data-source local --psd-source gwtc1 --ref-params gwtc1 \
        --phase-marginalization --n-live 5000 \
        --m-comp-lo "${LVK_M_LO}" --m-comp-hi "${LVK_M_HI}" \
        --output-dir "${run_dir}" \
        >> "${run_dir}/sampler.log" 2>&1
    local rc=$?
    canonicalise_csv "${run_id}" "${run_dir}/PhaseMarg_*${waveform}*.csv" || rc=1
    finalise_run "${run_id}" "${rc}"
}

# Run 1: CONTROL — IMRPhenomD_NRTidalv2 with LVK-matched bounds.
# This is the diagnostic: does just changing the bounds resolve the q gap?
bank_check "${BUDGET_SECONDS}" && run_gw170817 IMRPhenomD_NRTidalv2    imrphenomd_nrtidalv2

# Run 2: BEST tidal waveform — IMRPhenomXAS_NRTidalv3 (newer base + newer tides).
bank_check "${BUDGET_SECONDS}" && run_gw170817 IMRPhenomXAS_NRTidalv3  imrphenomxas_nrtidalv3

# Run 3: PRECESSION-ONLY — IMRPhenomPv2 (BBH; no tides).
bank_check "${BUDGET_SECONDS}" && run_gw170817 IMRPhenomPv2            imrphenompv2

finalise_session
