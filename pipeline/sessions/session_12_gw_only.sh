#!/usr/bin/env bash
# Session 12 — GW170817 GW-only (full-sky) runs.
#
# Goal: remove the EM/standard-siren constraint (H0, v_p) and use an
# uninformative volumetric d_L prior [10, 300] Mpc (p(d_L) ∝ d_L^2).
# Compares posteriors and runtime against the standard-siren baseline
# from session 07 (same waveforms, same LVK mass bounds).
#
# Two waveforms:
#   1. IMRPhenomD_NRTidalv2  — direct comparison with s07 siren baseline
#   2. IMRPhenomXAS_NRTidalv3 — best-evidence tidal waveform from s07
#
# Estimated wall-clock: 2 × ~15 min (~30 min) on A100.

. "$(dirname "$0")/_common.sh"
init_session "session_12_gw_only"

HET_SCRIPT="${REPO_ROOT}/pipeline/GW170817_heterodyned_1.py"
BUDGET_SECONDS=7200  # 2 hours

LVK_M_LO=0.87
LVK_M_HI=1.74

run_gw_only() {
    local waveform="$1" tag="$2"
    local run_id="s12__gw170817__${tag}__gw_only__seed0000"
    local run_dir="${RESULTS_ROOT}/${run_id}"
    echo "--- [${run_id}] $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
    mkdir -p "${run_dir}"
    write_config_json "${run_id}" "${HET_SCRIPT}" "${waveform}" "gw_only" 5000 2500 501 0 \
        "--gw-only --m-comp-lo ${LVK_M_LO} --m-comp-hi ${LVK_M_HI}"

    bank_check "${BUDGET_SECONDS}" || return 1
    ${PYTHON} "${HET_SCRIPT}" \
        --waveform "${waveform}" \
        --data-source local --psd-source gwtc1 --ref-params gwtc1 \
        --phase-marginalization --n-live 5000 \
        --m-comp-lo "${LVK_M_LO}" --m-comp-hi "${LVK_M_HI}" \
        --gw-only \
        --output-dir "${run_dir}" \
        >> "${run_dir}/sampler.log" 2>&1
    local rc=$?
    canonicalise_csv "${run_id}" "${run_dir}/PhaseMarg_*${waveform}*_gw_only.csv" || rc=1
    finalise_run "${run_id}" "${rc}"
}

bank_check "${BUDGET_SECONDS}" && run_gw_only IMRPhenomD_NRTidalv2   imrphenomd_nrtidalv2
bank_check "${BUDGET_SECONDS}" && run_gw_only IMRPhenomXAS_NRTidalv3 imrphenomxas_nrtidalv3

finalise_session
