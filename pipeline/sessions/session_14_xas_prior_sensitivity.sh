#!/usr/bin/env bash
# Session 14 — IMRPhenomXAS_NRTidalv3 prior-sensitivity sweep (host-localised).
#
# Goal: extend the prior-sensitivity story (baseline / flat-in-z / vp250) to
# the nominated primary GW170817 waveform, mirroring the existing
# IMRPhenomD_NRTidalv2 + TaylorF2 host-localised suite under
# Results/gwtc1_phasemarg/. Required to make the IMRPhenomXAS_NRTidalv3 row
# in the prior-sensitivity figure scientifically self-contained, instead of
# relying on the IMR analogue.
#
# All runs: host-localised sky (NGC 4993, hard-coded in the heterodyned
# scripts), default mass bounds [0.5, 7.7] M_sun, n_live=5000, n_bins=501,
# phase-marginalised, GWTC-1 PSD, GWTC-1 reference parameters.
#
# PREREQUISITES: none — IMRPhenomXAS_NRTidalv3 is already supported by
# heterodyned_1.py / _2.py / _3.py.
#
# Estimated wall-clock: 3 × ~15 min (~45 min) on A100. Adjust BUDGET_SECONDS
# below if running on a slower GPU.

. "$(dirname "$0")/_common.sh"
init_session "session_14_xas_prior_sensitivity"

HET_BASELINE="${REPO_ROOT}/pipeline/GW170817_heterodyned_1.py"
HET_FLATZ="${REPO_ROOT}/pipeline/GW170817_heterodyned_2.py"
HET_VP250="${REPO_ROOT}/pipeline/GW170817_heterodyned_3.py"
WAVEFORM="IMRPhenomXAS_NRTidalv3"
WAVEFORM_TAG="imrphenomxas_nrtidalv3"
BUDGET_SECONDS=10800   # 3 hours hard cap

COMMON_ARGS=(
    --waveform "${WAVEFORM}"
    --data-source local
    --psd-source gwtc1
    --ref-params gwtc1
    --phase-marginalization
    --n-live 5000
)

run_xas() {
    local script="$1" tag="$2"
    local run_id="s14__gw170817__${WAVEFORM_TAG}__${tag}__seed0000"
    local run_dir="${RESULTS_ROOT}/${run_id}"
    echo "--- [${run_id}] $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
    mkdir -p "${run_dir}"
    write_config_json "${run_id}" "${script}" "${WAVEFORM}" "${tag}" 5000 2500 501 0 ""

    ${PYTHON} "${script}" "${COMMON_ARGS[@]}" \
        --output-dir "${run_dir}" \
        >> "${run_dir}/sampler.log" 2>&1
    local rc=$?
    canonicalise_csv "${run_id}" "${run_dir}/PhaseMarg_*${WAVEFORM}*.csv" || rc=1
    finalise_run "${run_id}" "${rc}"
}

# 1: Baseline — Beta(3,1) on d_L (∝ d_L^2 in volume; LVK convention).
bank_check "${BUDGET_SECONDS}" && run_xas "${HET_BASELINE}" baseline

# 2: Flat-in-z d_L prior — direct sampling, not reweighted.
bank_check "${BUDGET_SECONDS}" && run_xas "${HET_FLATZ}" flatz

# 3: Increased peculiar-velocity sigma (sigma_vp = 250 km/s).
bank_check "${BUDGET_SECONDS}" && run_xas "${HET_VP250}" vp250

finalise_session

# After this script completes, run locally on the analysis box (CPU only).
# The input CSV is positional; only the output takes a flag:
#
#   python pipeline/reweight_dL_to_flat_z.py \
#       results/test_suite/s14__gw170817__imrphenomxas_nrtidalv3__baseline__seed0000/samples.csv \
#       --output results/test_suite/s14__gw170817__imrphenomxas_nrtidalv3__reweighted_flatz__seed0000/samples.csv
#
# That gives the importance-sampled flat-z variant (no extra GPU time) so the
# XAS row in the prior-sensitivity figure has the same four columns as IMR.
