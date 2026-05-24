# Chain regeneration on a GPU

The 17 nested-sampling chains the paper cites can be regenerated end-to-
end on a single NVIDIA A100 (40 GB) GPU using the public stack:

- [BlackJAX-NS](https://github.com/handley-lab/blackjax) — the slice
  sampler that drives nested sampling on the GPU.
- The Prathaban et al. (2025) heterodyned-likelihood kernel — a JAX
  implementation of relative binning for compact-binary mergers.
- [Ripple](https://github.com/tedwards2412/ripple) — the JAX waveform
  library (IMRPhenomD_NRTidalv2, IMRPhenomXAS_NRTidalv3,
  IMRPhenomXPHM, TaylorF2 family).

## Pre-baked alternative

For reproduction of figures and tables only, the pre-baked chains live
on Zenodo (DOI: TODO). Download once and skip this entire document.

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
