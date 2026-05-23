# GW170817 GPU-accelerated bright-siren H₀ — data and analysis release

This repository accompanies

> **Yang M., Prathaban M., Yallup D., Handley W.** (2026), *Rapid Hubble
> constant inference from GW170817 using GPU-accelerated nested sampling:
> prior sensitivity and the limits of post-hoc reweighting*, MNRAS
> (submitted).

It contains the analysis scripts, run catalogue, derived summary
tables, and figure- and table-generation code needed to reproduce
every numerical claim and figure in the paper. The nested-sampling
**chains themselves are not committed here** — each individual
`samples.csv` is ~100 MB and the full set runs to several GB. They
can be regenerated from the public strain data using the BlackJAX-NS
sampler and the custom GW-likelihood kernel cited below.

If you use this release, please cite the paper and this repository.

## What's in here

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── Plots/                             # figure-regeneration scripts (the
│   ├── _plot_utils.py                 # production set; the dir also
│   ├── build_paper_tables.py          # ships exploratory variants from
│   ├── plot_H0_prior_sensitivity.py   # the research workflow)
│   ├── plot_bimodality.py
│   ├── plot_GW170817_waveform_corner.py
│   ├── plot_GW150914_waveform_comparison.py
│   └── plot_H0_GW170817_waveform_comparison.py
├── analysis/                          # per-sweep aggregators + response analyses
│   ├── _helpers.py                    # shared chain/config loaders
│   ├── run_catalog.csv                # catalogue of canonical runs
│   ├── analyze_selection_term.py      # M1: H₀-independence of N_s(H₀)
│   ├── analyze_seed_ensemble.py       # M2: per-mode σ(ln Z) ensemble
│   ├── analyze_bimodality.py          # IMR Mode-A/B Bayes factor
│   ├── analyze_bimodality_imrx.py     # IMRX Mode-A/B Bayes factor (queued)
│   ├── analyze_nmcmc_sweep.py         # M7: slice-step convergence (queued)
│   └── compare_bimodality_waveforms.py # IMR-vs-IMRX (d_L, ι) cross-check
└── Results/test_suite/                # *NOT* in git — populate locally:
    └── sNN__<event>__<wf>__<variant>__seed####/
        ├── samples.csv                # nested-sampling chain (anesthetic format)
        ├── config.json                # run metadata (waveform, prior, seed, ...)
        └── sampler.log                # ln Z, n_eff, runtime
```

## How to reproduce

### Prerequisites

- Python 3.11+ with the packages in `requirements.txt`
  (`numpy`, `scipy`, `pandas`, `matplotlib`, `anesthetic`).
- LaTeX (`latexmk`, MNRAS class file `mnras.cls` + `mnras.bst`) only if you
  want to rebuild the manuscript PDF; the figure/table regeneration does
  not need it.
- The BlackJAX-NS sampler + GW-likelihood kernel (see below) are required
  only to *generate* the nested-sampling chains. Once `Results/test_suite/`
  is populated, all the scripts in this repo are pure CPU and need only the
  Python stack above.

The original analysis used the conda env at
`/opt/miniconda3/envs/PhD` on the author's macOS workstation; any
equivalent environment with the packages above will do.

### Step 0 — regenerate the chains (one-time, requires a GPU)

The nested-sampling chains are NOT in this repository. Each individual
`samples.csv` is ~100 MB and the full set runs to several GB, so we ship
the analysis stack and rely on the BlackJAX-NS pipeline to regenerate
the chains from the public strain data. Instructions and example driver
scripts are in the kernel paper repository linked under
"Generating new chains" below; expected wall-clock is ≈13 min per
GW170817 IMRPhenomXAS\_NRTidalv3 run on a single NVIDIA A100.

After running, populate `Results/test_suite/sNN__...__seed####/` with
each run's `samples.csv`, `config.json`, and `sampler.log`. The expected
run names are listed in `analysis/run_catalog.csv`.

### Step 1 — regenerate the paper tables

```bash
python Plots/build_paper_tables.py
```

Writes the `.tex` table fragments that the manuscript `\input`s,
into `Results/gwtc1_phasemarg/`.

### Step 2 — regenerate the paper figures

```bash
python Plots/plot_H0_prior_sensitivity.py
python Plots/plot_bimodality.py
python Plots/plot_GW170817_waveform_corner.py
python Plots/plot_GW150914_waveform_comparison.py
python Plots/plot_H0_GW170817_waveform_comparison.py
```

Each writes its output to `Results/gwtc1_phasemarg/plots/`.

### Step 3 — run the response-letter analyses

```bash
python analysis/analyze_selection_term.py       # M1 selection-term cancellation
python analysis/analyze_seed_ensemble.py        # M2 ln Z ensemble across seeds
python analysis/compare_bimodality_waveforms.py # M4 IMR-vs-IMRX cross-check
```

## Generating new chains (advanced)

The parallel-slice BlackJAX-NS sampler is at
[Yallup et al. (2025), arXiv:2509.24949](https://arxiv.org/abs/2509.24949)
and the bilby-like nested-sampling kernel that wraps it for
gravitational-wave inference is at
[Prathaban et al. (2025), arXiv:2509.04336](https://arxiv.org/abs/2509.04336).
The GW170817-specific driver scripts that produced the chains in this
release live in the parent project and are not duplicated here; reach
out if you want a frozen snapshot.

## Strain data and reference posteriors

This work uses publicly available strain data and reference posteriors
from the LIGO–Virgo–KAGRA Collaboration data releases — *not* from
GWOSC directly:

- **GW170817 strain** and host-galaxy peculiar-velocity inputs:
  LIGO Document P1800061
  (<https://dcc.ligo.org/LIGO-P1800061/public>) and the H₀
  measurement release P1700296
  (<https://dcc.ligo.org/LIGO-P1700296/public>).
- **GW150914 strain** and the GWTC-2.1 IMRPhenomXPHM
  parameter-estimation reference posterior: Zenodo deposit
  [10.5281/zenodo.6513631](https://doi.org/10.5281/zenodo.6513631).

These sources are not redistributed here; download them directly.

## Author contact

Ming Yang — Cavendish Laboratory, University of Cambridge
(GitHub: [@ming-256](https://github.com/ming-256))

## License

Code released under the MIT License (see `LICENSE`). Chain CSVs and
summary tables are released under CC BY 4.0. If you redistribute or
build on this release, please cite the paper.
