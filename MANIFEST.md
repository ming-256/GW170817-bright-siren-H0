# Manifest — Yang et al. (2026) data release

Every artefact in this repository, with its scientific role and source.
For the per-file classification companion (564 rows including
exploratory files NOT shipped in this release), see the audit's
inventory CSV in the main project repository.

**A note on that inventory:** it predates the consolidation of every
numerical claim into `scripts/build_paper_tables.py`, and still marks
three superseded scripts as "include" —
`compute_prior_sensitivity.py`, `compute_summary_stats.py` and
`compute_waveform_systematics.py`, together with their outputs
(`prior_sensitivity*.csv`, `summary_stats*.csv`,
`waveform_systematics.csv`). They are deliberately not shipped. Their
divergence statistics (KL, JSD, Hellinger, Wasserstein between prior
variants and between waveforms) appear nowhere in the manuscript, and
they use a different MAP estimator from the one the paper reports —
69.7 / 72.1 against the paper's binned 70.5 km/s/Mpc. The paper's one
Wasserstein figure is the heterodyne-bin sweep, which *is* released as
`results/test_suite/het_bins_sweep_wasserstein.csv`. Treat the inventory
as stale on this point rather than the release as incomplete.

## Top-level

| Path | Role |
|------|------|
| `README.md` | quick-start, citation, headline result |
| `MANIFEST.md` | this file |
| `LICENSE` | MIT (code) + CC-BY-4.0 (data) |
| `CITATION.cff` | GitHub citation widget |
| `environment.yml` | conda env spec |
| `requirements.txt` | pip-only mirror |
| `regenerate.sh` | CPU-only rebuild of tables, figures, PDF |
| `run_chains.sh` | GPU re-run of the sampler, one session at a time |
| `fetch_data.sh` | `chains` (from Zenodo), `verify` (against CHAIN_MANIFEST.csv), `figures`, `strain` |
| `make_chain_bundle.sh` | builds the chain tarball to upload as a new Zenodo version |
| `make_chain_manifest.py` | regenerates `results/CHAIN_MANIFEST.csv`; refuses symlinks, LFS pointers and undersized files |

## `paper/` — manuscript

| Path | Role | Source / generator |
|------|------|-------------------|
| `paper/main.tex` | manuscript source (409 lines) | authored in the working repository (ming-256/GPU-Accelerated-Bayesian-Inference-of-Gravitational-Waves); this is the canonical copy |
| `paper/references.bib` | bibliography (45 entries) | authored in the working repository; this is the canonical copy |
| `paper/main.pdf` | built artefact (12 pages, 249-word abstract) | `latexmk -pdf paper/main.tex` |
| `paper/figures/corner_GW150914_waveform_comparison.pdf` | Figure 1 | `scripts/plot_GW150914_waveform_comparison.py` + the s17a chain |
| `paper/figures/corner_IMRPhenomD_hetero_vs_unhetero.pdf` | Figure 2 | `scripts/plot_corner_IMRPhenomD_hetero_vs_unhetero.py` + IMRPhenomD heterodyned + unheterodyned baseline + LVK GWTC-1 HDF5 |
| `paper/figures/H0_prior_sensitivity.pdf` | Figure 3 | `scripts/plot_H0_prior_sensitivity.py` + s14 IMRX × 4 + s18 vpmean × 3 |
| `paper/figures/bimodality_imr_vs_imrx.pdf` | Figure 4 | `scripts/compare_bimodality_waveforms.py` + s10 IMR refModeB + s14 IMRX flatz |
| `paper/figures/bimodality.pdf` | Figure 5 | `scripts/plot_bimodality.py` + s10 IMR dL30-75 / dL10-30 / dL10-75-refModeB |
| `paper/figures/H0_waveform_comparison.pdf` | Figure 6 | `scripts/plot_H0_GW170817_waveform_comparison.py` + s14 IMRX + TF2 baseline |
| `paper/figures/corner_GW170817_waveform_comparison.pdf` | Figure 7 | `scripts/plot_GW170817_waveform_corner.py` + s14 IMRX + TF2 baseline + LVK GWTC-1 HDF5 |
| `paper/figures/scaling_study_full.pdf` | supplementary scaling plot (not a paper figure; retained for reproducibility) | `scripts/plot_scaling_full.py` + s13 n_live sweep + s07 LVK-bounds anchor |
| `paper/tables/table1_gw150914.tex` | Table 1 (GW150914 validation) | `scripts/build_paper_tables.py` |
| `paper/tables/tableW_waveform.tex` | Table 4 (cross-waveform H₀) | `scripts/build_waveform_table.py` |
| `paper/tables/table5_prior_sensitivity.tex` | Table 5 (prior-sensitivity sweep) | `scripts/build_paper_tables.py` |
| `paper/tables/table6_bimodality.tex` | Table 6 (bimodality) | `scripts/build_paper_tables.py` |

## `pipeline/` — the nested sampler that produced the chains

Every `results/test_suite/*/config.json` names one of these scripts in its
`script` field. The d_L prior is chosen by which script you run.

| Path | Role |
|------|------|
| `pipeline/GW170817_heterodyned_1.py` | baseline GW170817 run: Beta(3,1) d_L prior (volumetric, LVK convention) |
| `pipeline/GW170817_heterodyned_2.py` | flat-in-z d_L prior, sampled directly |
| `pipeline/GW170817_heterodyned_3.py` | baseline prior with sigma_vp = 250 km/s |
| `pipeline/GW170817_unheterodyned_1.py` | unheterodyned reference likelihood (the speed-up denominator) |
| `pipeline/GW150914_heterodyned.py` | GW150914 XPHM validation run |
| `pipeline/reweight_dL_to_flat_z.py` | CPU post-hoc reweighting; produces the s14 `reweighted_flatz` run |
| `pipeline/sessions/_common.sh` | writes each run's `config.json`, canonicalises the CSV to `samples.csv`, updates `run_catalog.csv` |
| `pipeline/sessions/session_*.sh` | the 19 launch scripts that drove the paper's runs, one per run group |
| `pipeline/sessions/_update_catalog_status.py` | marks a run done/failed in `run_catalog.csv` |
| `pipeline/sessions/prior_only_q_diagnostic.py` | prior-only q diagnostic (session H; CPU, <1 min) |

Data paths are read from the environment with defaults matching this
repository: `GWOSC_GW170817_DIR`, `GWOSC_GW150914_DIR`, `GWTC1_HDF5`,
`GWTC2P1_GW150914_HDF5`.

## `scripts/` — production figure and table generators

| Path | Role |
|------|------|
| `scripts/_plot_utils.py` | shared plotting helpers (LaTeX setup, palette, HPD/MAP, LVK HDF5 loader) |
| `scripts/build_paper_tables.py` | canonical table & summary generator (Tables 1/5/6); emits `paper/tables/*.tex` + `results/gwtc1_phasemarg/{paper_tables,paper_diagnostics}.csv` |
| `scripts/build_waveform_table.py` | cross-waveform table (Table 4); emits `results/gwtc1_phasemarg/tableW_waveform.tex` |
| `scripts/plot_GW150914_waveform_comparison.py` | Figure 1 |
| `scripts/plot_corner_IMRPhenomD_hetero_vs_unhetero.py` | Figure 2 |
| `scripts/plot_H0_prior_sensitivity.py` | Figure 3 |
| `scripts/compare_bimodality_waveforms.py` | Figure 4 |
| `scripts/plot_bimodality.py` | Figure 5 |
| `scripts/plot_H0_GW170817_waveform_comparison.py` | Figure 6 |
| `scripts/plot_GW170817_waveform_corner.py` | Figure 7 |
| `scripts/plot_scaling_full.py` | supplementary scaling plot (not a paper figure) |

## `analysis/` — per-sweep aggregators and referee-response diagnostics

| Path | Role |
|------|------|
| `analysis/_helpers.py` | shared chain/config loaders + weighted-statistic helpers |
| `analysis/analyze_bimodality.py` | IMR Mode-A / Mode-B Bayes factor (Table 6 upper block) |
| `analysis/analyze_bimodality_imrx.py` | IMRX Mode-A / Mode-B Bayes factor (queued s19 follow-up) |
| `analysis/analyze_het_bins_sweep.py` | Appendix A — heterodyne-bin sweep summary |
| `analysis/analyze_num_delete_sweep.py` | Appendix A — n_delete sweep summary |
| `analysis/analyze_psd_sensitivity.py` | Appendix A — PSD-source sensitivity |
| `analysis/analyze_ref_params.py` | Appendix A — heterodyne reference (gwtc1 vs optimize) |
| `analysis/analyze_seed_ensemble.py` | Bimodality seed-ensemble ln Z scatter aggregator |
| `analysis/analyze_selection_term.py` | Selection-term N_s(H₀) verification (footnote at §2.4) |
| `analysis/analyze_psis_khat.py` | PSIS k̂ + bootstrap-bias diagnostic (§4.1) |
| `analysis/compile_test_suite_report.py` | end-to-end test-suite report builder |

## `results/` — provenance, derived summaries, and where the chains go

| Path | Role |
|------|------|
| `results/gwtc1_phasemarg/evidence_table.csv` | per-variant ln Z ± σ + n_eff (machine-readable) |
| `results/gwtc1_phasemarg/paper_diagnostics.csv` | per-variant n_eff + efficiency + PSIS k̂ + bootstrap CI (reweighted row) |
| `results/gwtc1_phasemarg/paper_tables.csv` | per-row summary statistics underlying Tables 4–6 |
| `results/gwtc1_phasemarg/table{1,4,5,6}*.tex` | LaTeX include fragments (mirror of `paper/tables/`) |
| `results/gwtc1_phasemarg/plots/<7 PDFs + PNG>` | canonical figure PDFs (mirror of `paper/figures/` plus PNG previews) |
| `results/test_suite/run_catalog.csv` | one row per run directory (58): sampler settings, seed, `status`, `chain_published`, and the scripts that consume it |
| `results/test_suite/bimodality_summary.csv` | per-mode ln Z, MAP, P(H₀>120) for s10 |
| `results/test_suite/bimodality_imrx_summary.csv` | same for the queued IMRX bimodality set |
| `results/test_suite/bimodality_waveform_check.csv` | IMR/IMRX cross-waveform Mode-B weight |
| `results/test_suite/gw150914_waveform_comparison.csv` | GW150914 source-param medians + HPDs |
| `results/test_suite/gw170817_waveform_comparison.csv` | GW170817 cross-waveform summaries |
| `results/test_suite/het_bins_sweep_summary.csv` | n_bins ∈ {251,501,1001} sweep |
| `results/test_suite/het_bins_sweep_wasserstein.csv` | pairwise W₁ on H₀ across the bin counts |
| `results/test_suite/num_delete_sweep_summary.csv` | n_delete/n_live ∈ {0.10,…,0.75} sweep |
| `results/test_suite/psd_sensitivity_summary.csv` | GWTC-1 / kazewong / bilby PSD sweep |
| `results/test_suite/seed_ensemble_summary.csv` | per-seed ln Z scatter for the bimodality runs |
| `results/test_suite/seed_ensemble_bayes_factor.csv` | per-seed ln 𝓑(B/A) with the ln(20/45) correction |
| `results/test_suite/selection_term_Ns.csv` | N_s(H₀) for the as-implemented and the hypothetical flat-in-z priors |
| `results/CHAIN_MANIFEST.csv` | path, size and sha256 of all 58 chains; the contract between git and the Zenodo deposit |
| `results/test_suite/sNN__*/samples.csv` | the nested-sampling chains, in anesthetic format. **Not in git** — 3.8 GB across 58 files; `bash fetch_data.sh chains` |
| `results/test_suite/sNN__*/sampler.log` | per-run sampler log; every ln Z in Tables 1/5/6 is parsed from these |
| `results/test_suite/sNN__*/config.json` | per-run provenance: script, waveform, n_live, n_delete, n_bins, seed, git SHA |
| `results/GW170817_GWTC-1.hdf5` | LVK GWTC-1 GW170817 reference posterior (2.5 MB); heterodyne reference params + GWTC-1 contours |

### Where the chains live

The chains are published in the Zenodo deposit rather than in git: 58
files, 3.8 GB, one of them 1.2 GB — past GitHub's 100 MB per-file hard
limit. Publishing most of them here and that one elsewhere would leave
this repository holding an arbitrary subset, so the whole set ships
together in the deposit. `bash fetch_data.sh chains` downloads them,
`bash fetch_data.sh verify` checks each one against
`results/CHAIN_MANIFEST.csv`.

What git keeps is everything that makes a chain interpretable without
downloading it: the per-run `config.json` and `sampler.log`, the run
catalogue, and the derived summary CSVs.

### Runs without a chain at all

41 of the 58 catalogued runs produced a chain; `chain_published` in
`run_catalog.csv` says which. The rest break down as:

- **9 `skipped`** — planned in a session script but never run. No output of
  any kind.
- **5 `s16__*` q-diagnostics** — exploratory runs kept for their
  `sampler.log` only; the chains were not retained.
- **3 completed runs whose chains were not kept**, all of them the largest
  configurations:
  `s11__…__nlive20000__tol1e-4`, `s13__…__nlive10000`, `s13__…__nlive20000`.
  These are marked `status=done, chain_published=no` and are absent from
  the Zenodo deposit too. Their `sampler.log` survives, and it is the log —
  not the chain — that the scaling results use: `plot_scaling_full.py`
  reads `results/scaling_study/scaling_summary_full.csv`, so nothing in
  the paper depends on the missing CSVs. Regenerate them with
  `bash run_chains.sh session_13_nlive_scaling_imr` (~1.5 h on an A100) if
  you want the samples themselves.

No figure or table in the paper reads a run whose chain is absent.

## `docs/`

| Path | Role |
|------|------|
| `docs/reproducibility.md` | fresh-clone → main.pdf recipe |
| `docs/chain_regeneration.md` | per-run `pipeline/` invocation, sampler settings, software versions, expected wall-clock |
| `docs/data_provenance.md` | which paper claim each summary CSV underwrites |
