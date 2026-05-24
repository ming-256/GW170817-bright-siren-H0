# Reproducibility — fresh-clone → main.pdf in ~3 min

A clean reproduction of every numerical claim, table, and figure in the
paper, from this repository alone (plus a Zenodo download of the chain
CSVs), on a CPU-only laptop.

## 1. Environment

```bash
git clone https://github.com/ming-256/GW170817-bright-siren-H0
cd GW170817-bright-siren-H0

conda env create -f environment.yml
conda activate gw170817-bright-siren-H0
```

The environment pins Python 3.12 and the package versions tested at
submission time (numpy ≥ 2.0, scipy ≥ 1.13, pandas ≥ 2.2, matplotlib ≥ 3.9,
h5py ≥ 3.10, anesthetic ≥ 2.8).

## 2. Chain bundle

The 17 nested-sampling chains the paper cites are ~5 GB combined and
live on Zenodo (DOI: TODO).  Download the bundle, unzip, and place
the per-run directories under `results/test_suite/` so that the
layout becomes:

```
results/test_suite/
├── run_catalog.csv                        # already in git
├── bimodality_summary.csv                 # already in git
├── ...                                    # already in git
├── s07__gw170817__imrphenomxas_nrtidalv3__baseline_lvkbounds__seed0000/
│   ├── samples.csv                        # from Zenodo
│   ├── sampler.log                        # from Zenodo
│   └── config.json                        # from Zenodo
├── s10__gw170817__imrphenomd_nrtidalv2__flatz__dL30-75__refGWTC1__seed0000/
│   └── ...
├── s14__gw170817__imrphenomxas_nrtidalv3__baseline__seed0000/
│   └── ...
└── ...
```

The required run IDs are listed in `MANIFEST.md` under "results/" and
in `results/test_suite/run_catalog.csv`.

You will also need the LVK GW170817 GWTC-1 reference HDF5
(used by `scripts/plot_GW170817_waveform_corner.py`):

```bash
# Place anywhere; tell the pipeline via GWTC1_HDF5 environment variable.
curl -L -o results/GW170817_GWTC-1.hdf5 \
  "https://dcc.ligo.org/public/0156/P1800061/.../GW170817_GWTC-1.hdf5"
export GWTC1_HDF5="$(pwd)/results/GW170817_GWTC-1.hdf5"
```

## 3. Regenerate

```bash
bash regenerate.sh
```

This runs (in order)

1. `scripts/build_paper_tables.py` — emits 4 table `.tex` files +
   `paper_tables.csv`, `paper_diagnostics.csv`, `evidence_table.csv`.
   Tables are mirrored from `results/gwtc1_phasemarg/` to
   `paper/tables/`.
2. The 7 figure scripts, in order. PDFs land in
   `results/gwtc1_phasemarg/plots/` and are then mirrored to
   `paper/figures/`. Figure 3
   (`scripts/compare_bimodality_waveforms.py`) writes its output
   directly into `paper/figures/`.
3. `latexmk -pdf paper/main.tex` — builds the 11-page MNRAS PDF.

Wall-clock ~3 min on a 2024-vintage laptop. No GPU required.

## 4. Verifying the output

Spot-check that the headline numbers in `paper/tables/table5_prior_sensitivity.tex`
match the abstract:

| Row | MAP | median | P(H₀ > 120) |
|-----|-----|--------|-------------|
| Baseline (volumetric) | 70.5 | 77.6 | 0.017 |
| Uniform-in-d_L, direct | 70.5 | 87.6 | 0.159 |
| Uniform-in-d_L, reweighted | 73.5 | 82.9 | 0.041 |
| σ_vp = 250 km/s | 73.5 | 78.3 | 0.069 |

The PSIS k̂ + bootstrap diagnostic referenced in §4.1 is appended to
`results/gwtc1_phasemarg/paper_diagnostics.csv` by
`analysis/analyze_psis_khat.py`. To recompute it:

```bash
python analysis/analyze_psis_khat.py
```

Expected output (on the IMRX reweighted draw):
`PSIS k̂ = 0.683`, reweighted bootstrap 95 % CI = [0.0374, 0.0419].

## 5. Building the PDF alone

If you have only modified `paper/main.tex` and want a fresh PDF without
re-running the pipeline:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
```

## 6. Reading the chains directly

The CSV chains use the `anesthetic` format. To load one in Python:

```python
import sys
sys.path.append("analysis")
from _helpers import load_run, weighted_median, weighted_tail_prob

run = load_run("s14__gw170817__imrphenomxas_nrtidalv3__baseline__seed0000")
h0 = run.param("H_0")
w  = run.weights
print(f"weighted median H_0 = {weighted_median(h0, w):.2f}")
print(f"P(H_0 > 120)        = {weighted_tail_prob(h0, w, 120):.4f}")
```

Or open the CSV in any tool that understands columnar data:

```bash
head -5 results/test_suite/s14__gw170817__imrphenomxas_nrtidalv3__baseline__seed0000/samples.csv
```
