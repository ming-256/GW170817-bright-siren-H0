# GW170817 GPU-accelerated bright-siren H₀ — data and analysis release

This repository accompanies

> **Yang M., Prathaban M., Yallup D., Handley W.** (2026), *Rapid Hubble
> constant inference from GW170817 using GPU-accelerated nested sampling:
> prior sensitivity and the limits of post-hoc reweighting*, MNRAS
> (submitted).

It contains the nested-sampling chains, derived summary tables, and the
figure- and table-generation scripts needed to reproduce every numerical
claim and figure in the paper.

If you use this release, please cite the paper and this repository.

## What's in here

```
.
├── README.md
├── LICENSE
├── Results/
│   └── test_suite/                    # one directory per nested-sampling run
│       └── sNN__<event>__<wf>__<variant>__seed####/
│           ├── samples.csv            # nested-sampling chain (anesthetic format)
│           ├── config.json            # run metadata (waveform, prior, seed, ...)
│           └── sampler.log            # ln Z, n_eff, runtime
├── Plots/                             # figure-regeneration scripts
│   ├── build_paper_tables.py          # regenerates Table 1, 2, 3, 4 entries
│   ├── plot_H0_prior_sensitivity.py
│   ├── plot_bimodality.py
│   ├── plot_GW170817_waveform_corner.py
│   ├── plot_H0_GW170817_waveform_comparison.py
│   └── _plot_utils.py                 # shared utilities
└── analysis/                          # response-time and convergence analyses
    ├── _helpers.py                    # shared chain/config loaders
    ├── analyze_selection_term.py      # M1: H₀-independence of N_s(H₀)
    ├── analyze_seed_ensemble.py       # M2: per-mode σ(ln Z) ensemble
    ├── analyze_bimodality.py          # IMR Mode-A/B Bayes factor
    ├── analyze_bimodality_imrx.py     # IMRX Mode-A/B Bayes factor (queued runs)
    ├── analyze_nmcmc_sweep.py         # M7: slice-step convergence sweep (queued runs)
    └── compare_bimodality_waveforms.py # IMR-vs-IMRX (d_L, ι) cross-check
```

Some subdirectories are still being populated; this README is the index.

## How to reproduce

### Prerequisites

- Python 3.11+ with `numpy`, `scipy`, `pandas`, `matplotlib`, `anesthetic`.
- LaTeX (`latexmk`, MNRAS class file `mnras.cls` + `mnras.bst`) only if you
  want to rebuild the manuscript PDF; the figure/table regeneration does
  not need it.
- The `blackjax-ns` GW-likelihood kernel is required only to *generate*
  new chains; consuming the released chains needs only the Python stack
  above.

The original analysis used the conda env at
`/opt/miniconda3/envs/PhD` on the author's macOS workstation; any
equivalent environment with the packages above will do.

### Regenerate the paper tables

```bash
python Plots/build_paper_tables.py
```

This writes the `.tex` table fragments that the manuscript `\input`s.

### Regenerate the paper figures

```bash
python Plots/plot_H0_prior_sensitivity.py
python Plots/plot_bimodality.py
python Plots/plot_GW170817_waveform_corner.py
python Plots/plot_H0_GW170817_waveform_comparison.py
```

Each writes its output to `Results/gwtc1_phasemarg/plots/`.

### Run the response-letter analyses

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
