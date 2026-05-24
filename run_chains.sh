#!/usr/bin/env bash
# Yang et al. (2026) MNRAS — GPU-only chain regeneration (stub).
#
# Each per-run chain regenerates a samples.csv (~100 MB) and a sampler.log.
# Full set runs to several GB.  See docs/chain_regeneration.md for the
# per-run BlackJAX-NS invocation, the heterodyned-likelihood kernel from
# Prathaban et al. (2025), and the strain/PSD inputs.
#
# Hardware: a single NVIDIA A100 (40 GB SXM4 or PCIe).
# Wall-clock estimates:
#   - GW170817 IMRX  baseline (n_live=5000):                   ~13 min
#   - GW170817 TF2   baseline (n_live=5000):                    ~4 min
#   - GW150914 XPHM  validation (n_live=8000, n_mcmc=160):      ~5 h
#   - Full prior-sensitivity 4-variant suite:                   ~1 h
#   - Full bimodality 6-run suite (2 seeds):                    ~1.5 h
#   - All Appendix-A robustness sweeps (~10 runs):              ~6 h
#   - All 17 cited runs in one batch:                           ~12-15 h
#
# This script is a stub.  The chain CSVs are also available pre-baked on
# the companion Zenodo deposit (DOI: TODO); for read-only reproduction
# of the figures/tables, download and unpack them into
# results/test_suite/ and then run regenerate.sh.

set -euo pipefail
echo "Chain regeneration is GPU-bound and not redistributed in-band."
echo "See docs/chain_regeneration.md for the per-run BlackJAX-NS invocation,"
echo "the Zenodo bundle URL, and per-run wall-clock budgets."
exit 1
