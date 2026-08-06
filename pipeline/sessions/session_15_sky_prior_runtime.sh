#!/usr/bin/env bash
# Session 15 — Sky-prior runtime comparison (heterodyned).
#
# Goal: produce direct full-sky vs narrow-sky (LVK-style EM-localised) wall-
# clock measurements for the heterodyned baseline, so the paper can quantify
# how much of the heterodyned speedup comes from the sky-prior choice.
#
# All conditions match s07 except for the new --narrow-sky flag, so the
# matched-pair comparison is one-variable:
#
#                 |  full-sky          | narrow-sky (this script)
#   IMRPhenomD_NRTv2 |  s07__…__imrphenomd_nrtidalv2__baseline_lvkbounds        | s15__…__imrphenomd_nrtidalv2__baseline_lvkbounds_narrow
#   IMRPhenomXAS_NRTv3 |  s07__…__imrphenomxas_nrtidalv3__baseline_lvkbounds    | s15__…__imrphenomxas_nrtidalv3__baseline_lvkbounds_narrow
#
# Mass bounds: LVK low-spin BNS [0.87, 1.74] M_sun (matches s07).
# n_live=5000, n_bins=501, GWTC-1 PSD/reference, phase-marginalised.
# The unheterodyned full-sky vs narrow-sky pair already exists in
# Results/gwtc1_phasemarg/PhaseMarg_Unheterodyned_*.csv (n_live=1500), so
# we don't need any new unheterodyned runs.
#
# PREREQUISITE: GW170817_heterodyned_1.py must support --narrow-sky.
#
# Estimated wall-clock: 2 × ~15 min (~30 min) on A100.

. "$(dirname "$0")/_common.sh"
init_session "session_15_sky_prior_runtime"

HET_BASELINE="${REPO_ROOT}/pipeline/GW170817_heterodyned_1.py"
BUDGET_SECONDS=7200

if ! grep -q -- "--narrow-sky" "${HET_BASELINE}" 2>/dev/null; then
    echo "!! GW170817_heterodyned_1.py is missing --narrow-sky. Pull latest main and retry." | tee -a "${SESSION_LOG}"
    exit 2
fi

LVK_M_LO=0.87
LVK_M_HI=1.74

run_narrow_sky() {
    local waveform="$1" tag="$2"
    local run_id="s15__gw170817__${tag}__baseline_lvkbounds_narrow__seed0000"
    local run_dir="${RESULTS_ROOT}/${run_id}"
    echo "--- [${run_id}] $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
    mkdir -p "${run_dir}"
    write_config_json "${run_id}" "${HET_BASELINE}" "${waveform}" "baseline_lvkbounds_narrow" 5000 2500 501 0 \
        "--m-comp-lo ${LVK_M_LO} --m-comp-hi ${LVK_M_HI} --narrow-sky"

    ${PYTHON} "${HET_BASELINE}" \
        --waveform "${waveform}" \
        --data-source local --psd-source gwtc1 --ref-params gwtc1 \
        --phase-marginalization --n-live 5000 \
        --m-comp-lo "${LVK_M_LO}" --m-comp-hi "${LVK_M_HI}" \
        --narrow-sky \
        --output-dir "${run_dir}" \
        >> "${run_dir}/sampler.log" 2>&1
    local rc=$?
    canonicalise_csv "${run_id}" "${run_dir}/PhaseMarg_*${waveform}*.csv" || rc=1
    finalise_run "${run_id}" "${rc}"
}

# Matched pair 1: IMR baseline LVK-bounds, narrow sky.
bank_check "${BUDGET_SECONDS}" && run_narrow_sky IMRPhenomD_NRTidalv2    imrphenomd_nrtidalv2

# Matched pair 2: XAS baseline LVK-bounds, narrow sky (matches the
# locked-primary waveform).
bank_check "${BUDGET_SECONDS}" && run_narrow_sky IMRPhenomXAS_NRTidalv3  imrphenomxas_nrtidalv3

finalise_session

# After this completes, on the analysis box:
#   python Plots/build_scaling_table.py        # picks up s15 narrow-sky rows
#   python Plots/plot_sky_prior_runtime.py     # produces the comparison plot
