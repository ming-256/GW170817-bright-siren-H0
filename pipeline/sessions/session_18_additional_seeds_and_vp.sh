#!/usr/bin/env bash
# Session 18 — Additional GPU runs needed before MNRAS submission.
#
# Goal:
#   (a) Second seed for the bimodality runs (s10), so the ln B_B/A = -0.66
#       claim is not a single-seed result.
#   (b) Peculiar-velocity centre sweep on the IMRPhenomXAS_NRTidalv3 baseline:
#       <v_p> in {215, 310, 405} km/s, sigma_vp = 150 km/s. Closes the
#       hedge in §3.1/§3.2 about the original Abbott+2017 v_p choice
#       (some references quote 310, some 215) and lets the paper say "the
#       H0 posterior is robust to the v_p prior centre over the historical
#       range".
#
# All runs use the LVK-style host setup: full sky, default mass bounds,
# n_live=5000, n_bins=501, GWTC-1 PSD/reference, phase-marginalised.
#
# Estimated wall-clock:
#   3 × ~15 min (bimodality second seed)   = ~45 min
#   3 × ~15 min (vp_mean sweep on s14 IMRX) = ~45 min
#   total ≈ 1.5 h on a single A100.
#
# Dependencies:
#   - GW170817_heterodyned_1.py must support --vp-mean and --sigma-vp
#     (added in this commit; default values 310.0 / 150.0 reproduce earlier
#     behaviour, so existing runs are unaffected).
#
# Output: new run directories under Results/test_suite/ named
#   s18__... — one per added run.

. "$(dirname "$0")/_common.sh"
init_session "session_18_additional_seeds_and_vp"

HET_BASELINE="${REPO_ROOT}/pipeline/GW170817_heterodyned_1.py"
HET_FLATZ="${REPO_ROOT}/pipeline/GW170817_heterodyned_2.py"
BUDGET_SECONDS=10800   # 3 hours hard cap

WAVE_IMR="IMRPhenomD_NRTidalv2"
WAVE_IMRX="IMRPhenomXAS_NRTidalv3"
TAG_IMR="imrphenomd_nrtidalv2"
TAG_IMRX="imrphenomxas_nrtidalv3"

# --------------------------------------------------------------------------- #
# (a) Bimodality second seed                                                   #
# --------------------------------------------------------------------------- #
# Mirrors session_10 (Mode A / Mode B / Unrestricted) but with seed=1 instead
# of seed=0. The flat-z d_L sub-range is set inside heterodyned_2.py via the
# --dL-lo / --dL-hi flags (these flags must already be supported by the
# script — they were used in session_10).

run_bimodality_seed1() {
    local tag="$1" dL_lo="$2" dL_hi="$3"
    local run_id="s18__gw170817__${TAG_IMR}__flatz__dL${dL_lo}-${dL_hi}__refGWTC1__seed0001"
    local run_dir="${RESULTS_ROOT}/${run_id}"
    echo "--- [${run_id}] $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
    mkdir -p "${run_dir}"
    write_config_json "${run_id}" "${HET_FLATZ}" "${WAVE_IMR}" "${tag}" 5000 2500 501 1 \
        "--dL-lo ${dL_lo} --dL-hi ${dL_hi} --seed 1"

    ${PYTHON} "${HET_FLATZ}" \
        --waveform "${WAVE_IMR}" \
        --data-source local --psd-source gwtc1 --ref-params gwtc1 \
        --phase-marginalization --n-live 5000 \
        --dL-lo "${dL_lo}" --dL-hi "${dL_hi}" \
        --seed 1 \
        --output-dir "${run_dir}" \
        >> "${run_dir}/sampler.log" 2>&1
    local rc=$?
    canonicalise_csv "${run_id}" "${run_dir}/PhaseMarg_*${WAVE_IMR}*.csv" || rc=1
    finalise_run "${run_id}" "${rc}"
}

bank_check "${BUDGET_SECONDS}" && run_bimodality_seed1 flatZ_modeA  30 75
bank_check "${BUDGET_SECONDS}" && run_bimodality_seed1 flatZ_modeB  10 30
# Unrestricted with the Mode-B-anchored heterodyne reference (matches s10).
run_bimodality_unrestricted_seed1() {
    local run_id="s18__gw170817__${TAG_IMR}__flatz__dL10-75__refModeB__seed0001"
    local run_dir="${RESULTS_ROOT}/${run_id}"
    echo "--- [${run_id}] $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
    mkdir -p "${run_dir}"
    write_config_json "${run_id}" "${HET_FLATZ}" "${WAVE_IMR}" flatZ_refModeB 5000 2500 501 1 \
        "--dL-lo 10 --dL-hi 75 --ref-modeB --seed 1"

    ${PYTHON} "${HET_FLATZ}" \
        --waveform "${WAVE_IMR}" \
        --data-source local --psd-source gwtc1 --ref-params gwtc1 \
        --phase-marginalization --n-live 5000 \
        --dL-lo 10 --dL-hi 75 --ref-modeB \
        --seed 1 \
        --output-dir "${run_dir}" \
        >> "${run_dir}/sampler.log" 2>&1
    local rc=$?
    canonicalise_csv "${run_id}" "${run_dir}/PhaseMarg_*${WAVE_IMR}*.csv" || rc=1
    finalise_run "${run_id}" "${rc}"
}
bank_check "${BUDGET_SECONDS}" && run_bimodality_unrestricted_seed1

# --------------------------------------------------------------------------- #
# (b) <v_p> centre sweep on the IMRX baseline                                  #
# --------------------------------------------------------------------------- #

run_vp_mean_variant() {
    local vp_mean="$1"
    local run_id="s18__gw170817__${TAG_IMRX}__baseline__vpmean${vp_mean}__seed0000"
    local run_dir="${RESULTS_ROOT}/${run_id}"
    echo "--- [${run_id}] $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
    mkdir -p "${run_dir}"
    write_config_json "${run_id}" "${HET_BASELINE}" "${WAVE_IMRX}" "baseline_vp${vp_mean}" 5000 2500 501 0 \
        "--vp-mean ${vp_mean} --sigma-vp 150"

    ${PYTHON} "${HET_BASELINE}" \
        --waveform "${WAVE_IMRX}" \
        --data-source local --psd-source gwtc1 --ref-params gwtc1 \
        --phase-marginalization --n-live 5000 \
        --vp-mean "${vp_mean}" --sigma-vp 150 \
        --output-dir "${run_dir}" \
        >> "${run_dir}/sampler.log" 2>&1
    local rc=$?
    canonicalise_csv "${run_id}" "${run_dir}/PhaseMarg_*${WAVE_IMRX}*.csv" || rc=1
    finalise_run "${run_id}" "${rc}"
}

# Three v_p centres bracketing the historical literature range.
# 310 km/s reproduces the existing s14 baseline as a consistency check.
for VP in 215 310 405; do
    bank_check "${BUDGET_SECONDS}" && run_vp_mean_variant "${VP}"
done

finalise_session

# --------------------------------------------------------------------------- #
# Post-processing (run locally on the analysis box):                          #
#                                                                             #
#   python Plots/build_paper_tables.py                                         #
#                                                                             #
# This regenerates Tables 4/5/6 + paper_diagnostics.csv. To incorporate the    #
# new s18 runs into the published numbers, append them to                      #
# TABLE5_PRIOR_SENSITIVITY (vp_mean variants) and TABLE6_BIMODALITY            #
# (seed=1 rows) inside Plots/build_paper_tables.py and rerun.                  #
# --------------------------------------------------------------------------- #
