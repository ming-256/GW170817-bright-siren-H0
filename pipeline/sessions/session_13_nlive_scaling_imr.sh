#!/usr/bin/env bash
# Session 13 — IMRPhenomD_NRTidalv2 n_live convergence scaling.
#
# Goal: produce a proper n_live convergence curve for the production waveform.
# Existing points: n_live=5000 (s07, standard termination) and n_live=20000
# (s11, tight-tol diagnostic — not standard termination). This session fills
# the gaps: 500, 1000, 2500, 10000, and a clean standard-termination 20000.
#
# All runs use the same configuration as s07:
#   IMRPhenomD_NRTidalv2, LVK mass bounds [0.87,1.74], phase-marg,
#   n_bins=501, psd=gwtc1, ref=gwtc1.
# num_delete = n_live / 2 throughout (standard 50% replacement).
#
# Estimated wall-clock: ~75 min total on A100.

. "$(dirname "$0")/_common.sh"
init_session "session_13_nlive_scaling_imr"

HET_SCRIPT="${REPO_ROOT}/pipeline/GW170817_heterodyned_1.py"
BUDGET_SECONDS=18000  # 5 hours

LVK_M_LO=0.87
LVK_M_HI=1.74

run_nlive() {
    local n_live="$1" n_delete="$2" tag="$3"
    local run_id="s13__gw170817__imrphenomd_nrtidalv2__baseline__${tag}__seed0000"
    local run_dir="${RESULTS_ROOT}/${run_id}"
    echo "--- [${run_id}] n_live=${n_live} --- $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
    mkdir -p "${run_dir}"
    write_config_json "${run_id}" "${HET_SCRIPT}" "IMRPhenomD_NRTidalv2" "baseline_lvkbounds" \
        "${n_live}" "${n_delete}" 501 0 "--m-comp-lo ${LVK_M_LO} --m-comp-hi ${LVK_M_HI}"

    bank_check "${BUDGET_SECONDS}" || return 1
    ${PYTHON} "${HET_SCRIPT}" \
        --waveform IMRPhenomD_NRTidalv2 \
        --data-source local --psd-source gwtc1 --ref-params gwtc1 \
        --phase-marginalization --n-live "${n_live}" --num-delete "${n_delete}" \
        --m-comp-lo "${LVK_M_LO}" --m-comp-hi "${LVK_M_HI}" \
        --output-dir "${run_dir}" \
        >> "${run_dir}/sampler.log" 2>&1
    local rc=$?
    canonicalise_csv "${run_id}" "${run_dir}/PhaseMarg_*IMRPhenomD_NRTidalv2*.csv" || rc=1
    finalise_run "${run_id}" "${rc}"
}

bank_check "${BUDGET_SECONDS}" && run_nlive  500    250  nlive00500
bank_check "${BUDGET_SECONDS}" && run_nlive 1000    500  nlive01000
bank_check "${BUDGET_SECONDS}" && run_nlive 2500   1250  nlive02500
bank_check "${BUDGET_SECONDS}" && run_nlive 10000  5000  nlive10000
bank_check "${BUDGET_SECONDS}" && run_nlive 20000 10000  nlive20000

finalise_session
