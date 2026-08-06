#!/usr/bin/env bash
# Session 09 — Heterodyne bin-count sweep at fixed n_live=5000.
# PREREQUISITE: patch P-NBINS (see CODE_CHANGES_NEEDED.md §4).
# Estimated wall-clock: ~1.5 h (runtime depends weakly on n_bins on A100).

. "$(dirname "$0")/_common.sh"
init_session "session_09_het_bins_sweep"

SCRIPT="${REPO_ROOT}/pipeline/GW170817_heterodyned_1.py"
BUDGET_SECONDS=10800

if ! grep -q "n-bins" "${SCRIPT}" 2>/dev/null; then
    echo "!! Patch P-NBINS not applied. See CODE_CHANGES_NEEDED.md §4." | tee -a "${SESSION_LOG}"
    exit 2
fi

for NBINS in 251 501 1001; do
    bank_check "${BUDGET_SECONDS}" || break
    RUN_ID="s09__gw170817__imrphenomd_nrtidalv2__baseline__nbins$(printf '%05d' ${NBINS})__seed0000"
    RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
    echo "--- [${RUN_ID}] n_bins=${NBINS} --- $(date -u +%H:%M:%S)" | tee -a "${SESSION_LOG}"
    mkdir -p "${RUN_DIR}"
    write_config_json "${RUN_ID}" "${SCRIPT}" "IMRPhenomD_NRTidalv2" "baseline_bins${NBINS}" 5000 2500 "${NBINS}" 0 "--n-bins ${NBINS}"

    ${PYTHON} "${SCRIPT}" \
        --waveform IMRPhenomD_NRTidalv2 \
        --data-source local --psd-source gwtc1 --ref-params gwtc1 \
        --phase-marginalization --n-live 5000 --n-bins "${NBINS}" \
        --output-dir "${RUN_DIR}" \
        >> "${RUN_DIR}/sampler.log" 2>&1
    RC=$?
    canonicalise_csv "${RUN_ID}" "${RUN_DIR}/PhaseMarg_*IMRPhenomD_NRTidalv2*.csv" || RC=1
    finalise_run "${RUN_ID}" "${RC}"
done

finalise_session
