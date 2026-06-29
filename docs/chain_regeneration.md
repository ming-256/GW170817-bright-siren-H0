# Chain regeneration on a GPU

The 17 nested-sampling chains the paper cites can be regenerated end-to-
end on a single NVIDIA A100 (40 GB) GPU using the public stack:

- [BlackJAX-NS](https://github.com/handley-lab/blackjax) — the slice
  sampler that drives nested sampling on the GPU.
- The Prathaban et al. (2025) heterodyned-likelihood kernel — a JAX
  implementation of relative binning for compact-binary mergers.
- [ripple](https://github.com/GW-JAX-Team/ripple) (GW-JAX-Team fork, v0.0.9) — the JAX
  waveform library (IMRPhenomD\_NRTidalv2, IMRPhenomXAS\_NRTidalv3,
  IMRPhenomXPHM, TaylorF2). The IMRPhenomXAS\_NRTidalv3 implementation
  is by Robin Chan. Accessed via jim's waveform-wrapper classes.
- [jim](https://github.com/GW-JAX-Team/jim) (`jimgw` v0.3.0) — waveform-wrapper
  classes (`RippleIMRPhenomXAS_NRTidalv3`, etc.), detector response
  (`get_H1/L1/V1`, `fd_response`), and data/PSD ingestion
  (`Data`, `PowerSpectrum`). jim's likelihood and flowMC sampler are
  not used; the nested-sampling engine is BlackJAX-NS.

## Software versions

The table below records the versions used to produce the paper's 17 chains.
Commit hashes for the GW-JAX-Team packages are not pinned in this repo
(no lockfile was captured at run time); the version tags are as declared in
the `GPU-Accelerated-Bayesian-Inference-of-Gravitational-Waves` README.
If you need exact provenance, check the git tags `v0.0.9` and `v0.3.0` in
the respective repos.

| Package | Source | Version used for paper | Recommended for new runs |
|---------|--------|------------------------|--------------------------|
| ripple | https://github.com/GW-JAX-Team/ripple | v0.0.9 | latest stable release |
| jim (`jimgw`) | https://github.com/GW-JAX-Team/jim | v0.3.0 | latest stable release |
| BlackJAX-NS | https://github.com/handley-lab/blackjax (`nested_sampling` branch) | commit at time of runs (hash not recorded) | latest `nested_sampling` or stable release |
| flowMC | https://github.com/GW-JAX-Team/flowMC | v0.4.5 (jim dependency; not directly used) | as required by jim |

**For new runs:** install the latest stable releases of ripple and jim first;
the GW-JAX-Team packages evolve quickly and newer versions are likely to be
faster and more accurate. Fall back to the tagged versions above only if
you encounter API incompatibilities with the run scripts.

## Reproducing figures and tables only

The chains are not redistributed as a pre-baked download. To rebuild the
paper's figures and tables, regenerate the chains with the steps above
(or request them from the authors), then run `regenerate.sh`.

## Per-run wall-clock

| Run set | Hardware | Wall-clock |
|---------|----------|------------|
| GW170817 IMRX baseline (n_live=5000) | A100-40GB SXM4 | ~13 min |
| GW170817 TaylorF2 baseline (n_live=5000) | A100-40GB SXM4 | ~4 min |
| GW150914 XPHM validation (n_live=8000, n_mcmc=160) | A100-40GB SXM4 | ~5 h |
| IMRX prior-sensitivity 4-variant suite (s14) | A100-40GB SXM4 | ~1 h |
| IMR bimodality 6-run suite (s10 + s18 seed=1) | A100-40GB SXM4 | ~1.5 h |
| Appendix-A robustness sweeps (s08 / s09 / s05) | A100-40GB SXM4 | ~6 h |
| **All 17 cited runs** | A100-40GB SXM4 | **~12–15 h** |

Other CUDA-12-capable GPUs with ≥ 24 GB HBM should also work but are not
benchmarked.

## Sampler hyperparameters used in the paper

All heterodyned GW170817 science runs:

| Parameter | Value |
|-----------|-------|
| n_live | 5000 |
| n_delete | 2500 (n_live / 2) |
| n_mcmc | 8 × n_dim = 112 |
| n_dim (after phase marginalisation) | 14 |
| Termination | fractional evidence increment < 10⁻³ |
| Heterodyne bins | 501 (GW170817), 383 (GW150914) |
| LVK strain band | 20 Hz – 2048 Hz, Δf = 1/128 Hz |

GW150914 XPHM validation:

| Parameter | Value |
|-----------|-------|
| n_live | 8000 |
| n_mcmc | 160 (16 × n_dim) |
| n_dim | 10 |

## Per-run invocation (schematic)

The exact launch scripts that drove the runs are preserved in the
main project repository under
`mnras_paper/test_suite/session_plans/session_NN_*.sh`. The schematic
form is:

```bash
python -m blackjax_ns.cli \
    --event gw170817 \
    --waveform IMRPhenomXAS_NRTidalv3 \
    --prior uniform-in-dL \
    --n-live 5000 --n-delete 2500 --n-mcmc 112 \
    --heterodyne --het-bins 501 \
    --phase-marginalise \
    --output-dir results/test_suite/s14__gw170817__imrphenomxas_nrtidalv3__flatz__seed0000 \
    --seed 0
```

The exact CLI may differ between BlackJAX-NS releases; consult the
sampler's documentation. The full set of run IDs that need to be
populated is in `results/test_suite/run_catalog.csv`.

## After the chains are in place

```bash
bash regenerate.sh
```

regenerates everything downstream (tables, figures, PDF).
