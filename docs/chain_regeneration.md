# Chain regeneration on a GPU

**You probably do not need this.** The chains are published in the Zenodo
deposit; `bash fetch_data.sh chains` downloads them and `bash regenerate.sh`
then rebuilds every table and figure on a CPU in about three minutes. This
document is for reproducing the *sampling* rather than the paper.

The sampler is in `pipeline/`, and `pipeline/sessions/session_*.sh` are
the launch scripts that drove the paper's runs. `bash run_chains.sh list`
enumerates them. Re-running a session overwrites whatever is in
`results/test_suite/<run_id>/`, so keep a copy of the published chain if
you want to diff yours against ours.

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

## Input data

The sampler reads LVK strain and PSDs, which are not committed here.
Fetch them once:

```bash
bash fetch_data.sh strain
```

That writes `data/GWOSC/GW170817/` (H1, L1, V1 cleaned strain plus the
GWTC-1 BayesWave PSDs from LIGO-P1900011) and `data/GWOSC/GW150914/`
(H1, L1 strain from the GWOSC O1 archive). The five strain files are
byte-identical by sha256 to the ones used for the paper. Override the
locations with `GWOSC_GW170817_DIR` and `GWOSC_GW150914_DIR`.

The GWTC-1 GW170817 reference posterior that supplies the heterodyne
reference parameters is already committed at
`results/GW170817_GWTC-1.hdf5` (override with `GWTC1_HDF5`).

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

## Per-run invocation

Each run is one invocation of a `pipeline/` script. The d_L prior is
selected by *which script* you run, not by a flag:

| Script | d_L prior / variant |
|--------|---------------------|
| `pipeline/GW170817_heterodyned_1.py` | baseline, Beta(3,1) — volumetric, LVK convention |
| `pipeline/GW170817_heterodyned_2.py` | flat-in-z, sampled directly |
| `pipeline/GW170817_heterodyned_3.py` | baseline prior with sigma_vp = 250 km/s |
| `pipeline/GW170817_unheterodyned_1.py` | baseline, unheterodyned likelihood |
| `pipeline/GW150914_heterodyned.py` | GW150914 XPHM validation |

The s14 flat-in-z IMRX run in the table above is exactly:

```bash
python pipeline/GW170817_heterodyned_2.py \
    --waveform IMRPhenomXAS_NRTidalv3 \
    --data-source local \
    --psd-source gwtc1 \
    --ref-params gwtc1 \
    --phase-marginalization \
    --n-live 5000 \
    --output-dir results/test_suite/s14__gw170817__imrphenomxas_nrtidalv3__flatz__seed0000
```

Run `python pipeline/GW170817_heterodyned_1.py --help` for the full flag
set (`--num-delete`, `--n-bins`, `--seed`, `--m-comp-lo/--m-comp-hi`,
`--fixed-sky`, and so on).

In practice you want the session wrapper rather than the bare command,
because it also writes the `config.json` provenance record, canonicalises
the output to `samples.csv`, and updates `run_catalog.csv`:

```bash
bash run_chains.sh session_14_xas_prior_sensitivity
```

`results/test_suite/run_catalog.csv` lists every run ID with its sampler
settings and status, and each existing `results/test_suite/*/config.json`
records the script, waveform, settings, seed and git SHA behind that
chain.

The one exception is `s14__..._reweighted_flatz`, which is not a sampling
run: it is produced on a CPU from the baseline chain by
`pipeline/reweight_dL_to_flat_z.py`, as noted at the end of
`pipeline/sessions/session_14_xas_prior_sensitivity.sh`.

## After the chains are in place

```bash
bash regenerate.sh
```

regenerates everything downstream (tables, figures, PDF).
