# GPU-accelerated bright-siren H₀ from GW170817 — data and analysis release

This repository accompanies

> **Yang M. H., Prathaban M., Yallup D., Handley W.** (2026).
> *Rapid Hubble constant inference from GW170817 using GPU-accelerated
> nested sampling: prior sensitivity and the limits of post-hoc
> reweighting.* MNRAS (submitted).
> [arXiv:2606.30504](https://arxiv.org/abs/2606.30504)

It is a complete release: the sampling pipeline, the nested-sampling
chains it produced, the analysis and plotting code, the derived summary
tables, and the manuscript source.

The release spans two places, split by size rather than by importance:

| What | Where | Size |
|------|-------|------|
| **Nested-sampling chains, all 58 files** | **Zenodo** — `bash fetch_data.sh chains` | **3.8 GB** |
| Sampler that produced them | git — `pipeline/` | — |
| Launch scripts for each run group | git — `pipeline/sessions/` | — |
| Figure and table generators | git — `scripts/`, `analysis/` | — |
| Per-run provenance and sampler logs | git — `results/test_suite/<run_id>/` | 0.3 MB |
| Run catalogue and derived summaries | git — `results/test_suite/*.csv` | — |
| Chain checksums | git — `results/CHAIN_MANIFEST.csv` | — |
| LVK GWTC-1 GW170817 reference posterior | git — `results/GW170817_GWTC-1.hdf5` | 2.5 MB |

The chains are on Zenodo and not in git because one of them is 1.2 GB, past
GitHub's 100 MB per-file hard limit. Splitting them — most in git, that one
elsewhere — would leave the repository holding an arbitrary subset, so the
whole set is published together in the deposit instead. Every chain's
sha256 is in `results/CHAIN_MANIFEST.csv`, so a download can be checked
against what git says it should be.

What stays in git is the part that makes the chains interpretable: for each
run, the `config.json` recording the exact script, waveform, sampler
settings, seed and git SHA, and the `sampler.log` the evidences are parsed
from.

Two further inputs belong to the LVK rather than to us and are fetched, not
redistributed: the 287 MB GWTC-2.1 GW150914 PE release (Figure 1 only), and
the strain and PSD inputs (only needed to re-run the sampler). The strain
files `fetch_data.sh` downloads are byte-identical by sha256 to the ones
that produced these chains.

The Zenodo deposit sits under the concept DOI
[10.5281/zenodo.21038511](https://doi.org/10.5281/zenodo.21038511), which
always resolves to the newest version — that is the DOI to cite.

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
            nested-sampling chains, the sampling pipeline, the derived CSV
            summaries, the run catalogue, and the figure/table-generation
            scripts for this paper.}
}
```

A GitHub Citation widget is configured via `CITATION.cff`.

## Quick start (CPU only)

```bash
git clone https://github.com/ming-256/GW170817-bright-siren-H0
cd GW170817-bright-siren-H0

conda env create -f environment.yml
conda activate gw170817-bright-siren-H0

bash fetch_data.sh chains     # 3.8 GB from Zenodo, once
bash fetch_data.sh verify     # check every file against CHAIN_MANIFEST.csv

bash regenerate.sh            # ~3 min, no GPU
```

`regenerate.sh` produces

- 4 table `.tex` files in `results/gwtc1_phasemarg/` (mirrored to `paper/tables/`)
- 7 paper-figure PDFs, plus the supplementary `scaling_study_full.pdf`, in `results/gwtc1_phasemarg/plots/` (mirrored to `paper/figures/`)
- `paper/main.pdf` — the 12-page submitted MNRAS manuscript

Figure 1 overlays the LVK GWTC-2.1 GW150914 posterior, which is 287 MB of
LVK data we do not redistribute. Fetch it once if you want that figure:

```bash
bash fetch_data.sh figures      # 287 MB, from Zenodo 10.5281/zenodo.6513631
```

Without it `regenerate.sh` still builds the other seven figures, all four
tables and the PDF.

## Re-running the sampler (GPU)

You do not need this to reproduce the paper — `fetch_data.sh chains` gets
you the published chains. Use it to reproduce the sampling itself.

```bash
bash fetch_data.sh strain       # LVK strain + PSDs, ~440 MB
bash run_chains.sh list         # what each session produces
bash run_chains.sh session_14_xas_prior_sensitivity
```

The sampler lives in `pipeline/`; `pipeline/sessions/session_*.sh` are the
scripts that drove the paper's runs, and they write straight back into
`results/test_suite/<run_id>/`. `docs/chain_regeneration.md` has the
sampler settings, the software versions and the per-run wall-clock. As a
budget guide on a single NVIDIA A100 (40 GB):

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
├── run_chains.sh            # GPU re-run of the sampler, one session at a time
├── fetch_data.sh            # chains from Zenodo, LVK strain/PSD/PE inputs; and `verify`
├── make_chain_bundle.sh     # builds the chain tarball for a new Zenodo version
├── make_chain_manifest.py   # regenerates the chain checksum manifest
├── paper/                   # LaTeX source + figures + tables + PDF
├── pipeline/                # the nested sampler that produced the chains
│   ├── GW170817_heterodyned_{1,2,3}.py   # baseline / flat-in-z / sigma_vp=250
│   ├── GW170817_unheterodyned_1.py       # unheterodyned reference
│   ├── GW150914_heterodyned.py           # GW150914 XPHM validation
│   ├── reweight_dL_to_flat_z.py          # post-hoc reweighting
│   └── sessions/                         # the 19 launch scripts, one per run group
├── scripts/                 # the 11 production plot/table generators
├── analysis/                # the 11 per-sweep aggregators + referee diagnostics
├── results/                 # provenance + derived CSVs + .tex tables + plot PDFs
│   ├── CHAIN_MANIFEST.csv   # sha256 + size of all 58 chains (they live on Zenodo)
│   ├── test_suite/<run_id>/ # sampler.log + config.json per run (chains fetched here)
│   └── gwtc1_phasemarg/     # host-localised chains, tables, figure PDFs
└── docs/                    # reproducibility / chain_regeneration / data_provenance
```

## Data sources

- GW170817 strain + PSD + reference PE — [LIGO P1800061](https://dcc.ligo.org/LIGO-P1800061/public) (LVK, 2018)
- GW170817 H₀ analysis — [LIGO P1700296](https://dcc.ligo.org/LIGO-P1700296/public) (LVK, 2017)
- GW150914 PE data release — [Zenodo 10.5281/zenodo.6513631](https://doi.org/10.5281/zenodo.6513631) (LVK GWTC-2.1)
- GW170817 and GW150914 strain — [GWOSC](https://gwosc.org); `fetch_data.sh` has the exact URLs
- This repository (archival snapshot) — [Zenodo 10.5281/zenodo.21038511](https://doi.org/10.5281/zenodo.21038511) (concept DOI, resolves to the latest version)

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
