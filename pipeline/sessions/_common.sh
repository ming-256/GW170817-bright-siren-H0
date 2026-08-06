#!/usr/bin/env bash
# Shared helpers for test_suite session scripts.
# Source this from each session_NN_*.sh via:  . "$(dirname "$0")/_common.sh"

set -euo pipefail

# Resolve repo root regardless of where the script is invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_SUITE_ROOT="${REPO_ROOT}/pipeline/sessions"
RESULTS_ROOT="${REPO_ROOT}/results/test_suite"
CATALOG="${RESULTS_ROOT}/run_catalog.csv"

# Override with PYTHON=... env var; otherwise resolved in priority order below.
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "${REPO_ROOT}/venv/bin/python" ]]; then
        PYTHON="${REPO_ROOT}/venv/bin/python"
    elif [[ -x "/opt/miniconda3/envs/jax/bin/python" ]]; then
        PYTHON="/opt/miniconda3/envs/jax/bin/python"
    else
        PYTHON="$(command -v python3)"
    fi
fi
export PYTHON

mkdir -p "${RESULTS_ROOT}" "${RESULTS_ROOT}/logs"

# Record session metadata once.
init_session() {
    SESSION_TAG="$1"
    SESSION_START_EPOCH="$(date +%s)"
    SESSION_START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    SESSION_LOG="${RESULTS_ROOT}/logs/${SESSION_TAG}.${SESSION_START_EPOCH}.log"
    GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo 'unknown')"
    echo "=== ${SESSION_TAG} start: ${SESSION_START_ISO}  git ${GIT_SHA} ===" | tee -a "${SESSION_LOG}"
}

# Write the per-run config.json that sits next to samples.csv.
# Args: run_id, script, waveform, variant, n_live, num_delete, n_bins, seed, extra_args (string)
write_config_json() {
    local run_id="$1" script="$2" waveform="$3" variant="$4"
    local n_live="$5" num_delete="$6" n_bins="$7" seed="$8" extra_args="$9"
    local run_dir="${RESULTS_ROOT}/${run_id}"
    mkdir -p "${run_dir}"
    cat > "${run_dir}/config.json" <<JSON
{
    "run_id":      "${run_id}",
    "script":      "${script}",
    "waveform":    "${waveform}",
    "variant":     "${variant}",
    "n_live":      ${n_live},
    "num_delete":  ${num_delete},
    "n_bins":      ${n_bins},
    "seed":        ${seed},
    "extra_args":  "${extra_args}",
    "git_sha":     "${GIT_SHA}",
    "started":     "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
}

# Call after a run finishes; annotates config.json with finish metadata.
finalise_run() {
    local run_id="$1" exit_status="$2"
    local run_dir="${RESULTS_ROOT}/${run_id}"
    local cfg="${run_dir}/config.json"
    local finished
    finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    # Append a tiny finish record rather than rewriting JSON.
    cat >> "${run_dir}/finish.json" <<JSON
{"finished": "${finished}", "exit_status": ${exit_status}}
JSON
    # Update catalog: set status=done or status=failed for this run_id.
    if [[ ${exit_status} -eq 0 ]]; then
        ${PYTHON} "${TEST_SUITE_ROOT}/_update_catalog_status.py" "${run_id}" done "${CATALOG}" || true
    else
        ${PYTHON} "${TEST_SUITE_ROOT}/_update_catalog_status.py" "${run_id}" failed "${CATALOG}" || true
    fi
}

# Copy the produced CSV (whatever the sampler named it) to samples.csv in the run_dir.
# First argument is run_id, second is glob of produced CSVs.
canonicalise_csv() {
    local run_id="$1"
    local pattern="$2"
    local run_dir="${RESULTS_ROOT}/${run_id}"
    local produced
    produced="$(ls -t ${pattern} 2>/dev/null | head -1 || true)"
    if [[ -n "${produced}" && -f "${produced}" ]]; then
        cp -f "${produced}" "${run_dir}/samples.csv"
        echo "  -> canonicalised CSV: ${produced} -> ${run_dir}/samples.csv" | tee -a "${SESSION_LOG}"
    else
        echo "  !! no CSV matching pattern ${pattern}" | tee -a "${SESSION_LOG}"
        return 1
    fi
}

# Time-banked runner. Stops launching runs if elapsed budget would exceed budget_seconds.
# Usage:  bank_check budget_seconds
bank_check() {
    local budget_seconds="$1"
    local now elapsed
    now="$(date +%s)"
    elapsed=$(( now - SESSION_START_EPOCH ))
    if [[ ${elapsed} -ge ${budget_seconds} ]]; then
        echo "!! time bank exhausted: ${elapsed}s >= ${budget_seconds}s. Skipping remaining runs." | tee -a "${SESSION_LOG}"
        return 1
    fi
    return 0
}

finalise_session() {
    local end_epoch end_iso elapsed
    end_epoch="$(date +%s)"
    end_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    elapsed=$(( end_epoch - SESSION_START_EPOCH ))
    echo "=== ${SESSION_TAG} end: ${end_iso}  elapsed: ${elapsed}s ===" | tee -a "${SESSION_LOG}"
}
