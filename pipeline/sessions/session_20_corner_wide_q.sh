#!/usr/bin/env bash
# Session 20 — GW170817 corner plot: wide-q baseline (XAS + TF2).
#
# Motivation: s07_lvkbounds runs used --m-comp-lo 0.87 --m-comp-hi 1.74,
# which hard-cuts q at ~0.62 in the posterior. This session re-runs the
# same setup (gwtc1 PSD, gwtc1 ref, phase-marg) without any component-mass
# restriction, giving q down to 0.125 for a clean corner plot.
#
# Runs:
#   s20__gw170817__imrphenomxas_nrtidalv3__baseline__seed0000
#   s20__gw170817__taylorf2__baseline__seed0000

. "$(dirname "$0")/_common.sh"
init_session "session_20_corner_wide_q"

HET_BASELINE="${REPO_ROOT}/pipeline/GW170817_heterodyned_1.py"
BUDGET_SECONDS=7200   # 2 hours

COMMON_ARGS=(
    --data-source local
    --psd-source gwtc1
    --ref-params gwtc1
    --phase-marginalization
    --n-live 5000
)

run_one() {
    local waveform="$1" wf_tag="$2"
    local run_id="s20__gw170817__${wf_tag}__baseline__seed0000"
    local run_dir="${RESULTS_ROOT}/${run_id}"
    echo "--- [${run_id}] $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
    mkdir -p "${run_dir}"
    write_config_json "${run_id}" "${HET_BASELINE}" "${waveform}" "baseline" 5000 2500 501 0 ""

    ${PYTHON} "${HET_BASELINE}" "${COMMON_ARGS[@]}" \
        --waveform "${waveform}" \
        --output-dir "${run_dir}" \
        >> "${run_dir}/sampler.log" 2>&1
    local rc=$?
    canonicalise_csv "${run_id}" "${run_dir}/PhaseMarg_*${waveform}*.csv" || rc=1
    finalise_run "${run_id}" "${rc}"
}

bank_check "${BUDGET_SECONDS}" && run_one "IMRPhenomXAS_NRTidalv3" "imrphenomxas_nrtidalv3"
bank_check "${BUDGET_SECONDS}" && run_one "TaylorF2"               "taylorf2"

finalise_session
