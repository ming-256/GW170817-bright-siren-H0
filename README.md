# GPU-accelerated bright-siren H₀ from GW170817 — data and analysis release

This repository accompanies

> **Yang M. H., Prathaban M., Yallup D., Handley W.** (2026).
> *Rapid Hubble constant inference from GW170817 using GPU-accelerated
> nested sampling: prior sensitivity and the limits of post-hoc
> reweighting.* MNRAS (submitted).
> [arXiv:2606.30504](https://arxiv.org/abs/2606.30504)

It contains the analysis scripts, run catalogue, derived summary tables,
and figure- and table-generation code needed to reproduce every numerical
claim and every figure in the paper. The nested-sampling **chains
themselves are not committed here** — each individual `samples.csv` is
~100 MB and the full set runs to several GB; they are not redistributed,
but can be regenerated from the public LVK strain data with the
[BlackJAX-NS](https://github.com/handley-lab/blackjax) sampler and the
heterodyned-likelihood kernel of Prathaban et al. (2025). An archival
snapshot of this repository is on Zenodo:
[10.5281/zenodo.21038511](https://doi.org/10.5281/zenodo.21038511).

## Headline result

Under the modern aligned-spin tidal waveform IMRPhenomXAS_NRTidalv3,
switching the luminosity-distance prior from volumetric (π(d_L) ∝ d_L²)
to uniform-in-d_L by **direct sampling** raises P(H₀ > 120 km/s/Mpc)
from **0.017 → 0.159**, while the binned MAP stays at 70.5 km/s/Mpc.
Post-hoc **reweighting** of the same baseline draws recovers only
P = 0.041 — *17 % of the directly-sampled shift*. A 4 000-draw bootstrap
on the reweighted estimator gives a 95 % CI of [0.037, 0.042] that
excludes the directly-sampled 0.159 by ~100 binomial standard errors: the
reweighting deficit is *bias*, not high variance. The mechanism is a
(d_L, ι) bimodality whose high-H₀ / low-d_L branch (Mode B) carries
appreciable likelihood but negligible volumetric-prior mass.

The full GPU pipeline completes the n_live = 5000 IMRX analysis in
≈13 min on a single NVIDIA A100; the four-variant prior-sensitivity suite
fits inside an hour. This makes per-event prior-sensitivity reruns the
*default* robustness tool for bright-siren cosmology, replacing post-hoc
reweighting.

## Citation

```bibtex
@misc{Yang2026DataRelease,
  author = {{Yang}, M.~H. and {Prathaban}, M. and {Yallup}, D. and {Handley}, W.},
  title  = {{GW170817 bright-siren H_0: data and analysis release}},
  year   = {2026},
  howpublished = {\url{https://github.com/ming-256/GW170817-bright-siren-H0}},
  doi    = {10.5281/zenodo.21038511},
  note   = {GitHub repository plus Zenodo archival snapshot containing the
            derived CSV summaries, run catalogue, and figure/table-
            generation scripts. Nested-sampling chains are regenerable
            from the public strain data using the BlackJAX-NS sampler.}
}
```

A GitHub Citation widget is configured via `CITATION.cff`.

## Quick start (CPU only, ≈ 3 min)

```bash
git clone https://github.com/ming-256/GW170817-bright-siren-H0
cd GW170817-bright-siren-H0

conda env create -f environment.yml
conda activate gw170817-bright-siren-H0

# Per-run chains are not redistributed: regenerate them (GPU; see
# docs/chain_regeneration.md) or request them from the authors, then
# unpack under results/test_suite/.  Layout in docs/data_provenance.md.

bash regenerate.sh
```

`regenerate.sh` produces

- 4 table `.tex` files in `results/gwtc1_phasemarg/` (mirrored to `paper/tables/`)
- 7 paper-figure PDFs, plus the supplementary `scaling_study_full.pdf`, in `results/gwtc1_phasemarg/plots/` (mirrored to `paper/figures/`)
- `paper/main.pdf` — the 12-page submitted MNRAS manuscript

## Chain regeneration (GPU only)

The nested-sampling chains are reproducible on a single NVIDIA A100
(40 GB) GPU using the BlackJAX-NS sampler and the heterodyned-likelihood
kernel. See `docs/chain_regeneration.md` for the per-run invocations and
the expected wall-clock per run. As a budget guide:

| Run set | Wall-clock |
|---------|-----------|
| IMRX prior-sensitivity sweep (4 variants) | ~1 h |
| Bimodality 6-run suite (2 seeds) | ~1.5 h |
| GW150914 XPHM validation (n_live=8000, n_mcmc=160) | ~5 h |
| All 17 cited runs in one batch | ~12–15 h |

## Repository layout

```
.
├── README.md                # this file
├── MANIFEST.md              # file-by-file provenance table
├── LICENSE                  # MIT (code) + CC-BY-4.0 (data)
├── CITATION.cff             # GitHub citation widget
├── environment.yml          # conda environment
├── requirements.txt         # pip-only mirror of the env's pip section
├── regenerate.sh            # CPU-only rebuild of tables, figures, PDF
├── run_chains.sh            # GPU-only chain regeneration (stub; see docs/)
├── paper/                   # LaTeX source + figures + tables + PDF
├── scripts/                 # the 9 production plot/table scripts
├── analysis/                # the 9 per-sweep aggregators + the M-series referee diagnostics
├── results/                 # derived CSVs + .tex tables + plot PDFs
└── docs/                    # reproducibility / chain_regeneration / data_provenance
```

## Data sources

- GW170817 strain + PSD + reference PE — [LIGO P1800061](https://dcc.ligo.org/LIGO-P1800061/public) (LVK, 2018)
- GW170817 H₀ analysis — [LIGO P1700296](https://dcc.ligo.org/LIGO-P1700296/public) (LVK, 2017)
- GW150914 PE data release — [Zenodo 10.5281/zenodo.6513631](https://doi.org/10.5281/zenodo.6513631) (LVK GWTC-2.1)
- This repository (archival snapshot) — [Zenodo 10.5281/zenodo.21038511](https://doi.org/10.5281/zenodo.21038511)

## Software

This analysis builds on the public JAX gravitational-wave stack. If you
reuse this code, please link/cite the upstream packages:

- [ripple](https://github.com/GW-JAX-Team/ripple) (GW-JAX-Team fork) —
  differentiable waveforms (IMRPhenomXAS_NRTidalv3, IMRPhenomD_NRTidalv2,
  TaylorF2, IMRPhenomXPHM); the IMRPhenomXAS_NRTidalv3 implementation is
  by Robin Chan.
- [jim](https://github.com/GW-JAX-Team/jim) (`jimgw`) — waveform-wrapper,
  detector-response, and data-handling utilities (our likelihood, priors,
  and nested sampler are our own; see the paper).
- [BlackJAX-NS](https://github.com/handley-lab/blackjax) — the GPU-native
  nested-sampling kernel.

Pinned versions are listed in `docs/chain_regeneration.md`.

## Hardware requirements

- **Tables + figures + PDF build (CPU only):** any modern laptop;
  tested on macOS / Apple M2 with Python 3.12, numpy ≥ 2, anesthetic ≥ 2.8.
- **Chain regeneration (optional, GPU only):** a single NVIDIA A100
  (40 GB SXM4 or PCIe). Other CUDA-12-capable GPUs with ≥ 24 GB HBM
  should also work but are not benchmarked.

## Licence

Code: MIT.  Data files (CSV) and figure PDFs: CC BY 4.0. See `LICENSE`.

## Acknowledgements

This work was supported by the research environment of the Handley Lab
at the University of Cambridge. MP is supported by the Harding
Distinguished Postgraduate Scholars Programme (HDPSP). This material is
based upon work supported by the Google Cloud research credits program
with the award GCP397499138. We acknowledge the LIGO–Virgo–KAGRA
Collaboration for the public strain data and reference posteriors used
here.
