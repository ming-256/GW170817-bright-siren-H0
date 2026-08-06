#!/usr/bin/env bash
# Yang et al. (2026) MNRAS — GPU chain regeneration.
#
# You do NOT need this to reproduce the paper. The chains this release is
# built from are published in the Zenodo deposit: `bash fetch_data.sh chains`
# downloads them and `bash regenerate.sh` does the rest on a CPU. Use this
# script when you want to reproduce the sampling itself rather than trust
# ours.
#
# Each session script re-runs one group of runs from the paper, writing
# results/test_suite/<run_id>/{samples.csv,sampler.log,config.json} and
# updating results/test_suite/run_catalog.csv. Re-running overwrites
# whatever is in that directory, so keep a copy of the published chain if
# you want to diff yours against ours.
#
# Hardware: a single NVIDIA A100 (40 GB SXM4 or PCIe). Other CUDA-12
# GPUs with >= 24 GB HBM should work but are not benchmarked.
#
# Usage:
#   bash run_chains.sh list                 # show every session and what it produces
#   bash run_chains.sh session_14_xas_prior_sensitivity
#   bash run_chains.sh all                  # ~12-15 h on an A100
#
# Prerequisites (see docs/chain_regeneration.md for versions and install):
#   - the GPU stack: jax[cuda12], jimgw, ripplegw, flowMC, blackjax@nested_sampling
#   - GW170817 strain + PSDs under $GWOSC_GW170817_DIR (default data/GWOSC/GW170817)
#   - GW150914 strain under $GWOSC_GW150914_DIR (default data/GWOSC/GW150914),
#     for the session_06 / session_17 validation runs only
#   Both strain sets are LVK public data; `bash fetch_data.sh` downloads them.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SESSION_DIR="pipeline/sessions"
MODE="${1:-list}"

list_sessions() {
    echo "Session scripts in ${SESSION_DIR}/ (run one with: bash run_chains.sh <name>)"
    echo
    printf '  %-38s %s\n' "SESSION" "PURPOSE"
    for s in "${SESSION_DIR}"/session_*.sh; do
        name="$(basename "$s" .sh)"
        # The one-line purpose is the second comment line of each session script.
        desc="$(sed -n '2s/^# *//p' "$s")"
        printf '  %-38s %s\n' "$name" "${desc:-—}"
    done
    echo
    echo "Per-run wall-clock estimates and sampler settings: docs/chain_regeneration.md"
    echo "Run IDs, sampler settings and status:              results/test_suite/run_catalog.csv"
}

require_gpu_stack() {
    local py="${PYTHON:-python}"
    if ! "$py" -c 'import jax, jimgw, blackjax' 2>/dev/null; then
        echo "error: the GPU sampling stack is not importable with '${py}'." >&2
        echo "       jax, jimgw and blackjax (nested_sampling branch) are all required." >&2
        echo "       See docs/chain_regeneration.md for the install recipe." >&2
        echo "       Note: you do NOT need this stack to reproduce the paper —" >&2
        echo "       run 'bash fetch_data.sh chains' for the published chains," >&2
        echo "       then 'bash regenerate.sh'." >&2
        exit 1
    fi
}

run_session() {
    local name="$1"
    local script="${SESSION_DIR}/${name}.sh"
    if [[ ! -f "$script" ]]; then
        echo "error: no such session '${name}'." >&2
        echo "       Run 'bash run_chains.sh list' to see the available sessions." >&2
        exit 1
    fi
    require_gpu_stack
    echo "=== ${name} ==="
    bash "$script"
}

case "$MODE" in
    list|--list|-l)
        list_sessions
        ;;
    all)
        require_gpu_stack
        for s in "${SESSION_DIR}"/session_*.sh; do
            echo "=== $(basename "$s" .sh) ==="
            bash "$s"
        done
        ;;
    session_*)
        run_session "$MODE"
        ;;
    *)
        echo "Unknown argument: ${MODE}" >&2
        echo "Use: bash run_chains.sh [list | all | session_<name>]" >&2
        exit 1
        ;;
esac
