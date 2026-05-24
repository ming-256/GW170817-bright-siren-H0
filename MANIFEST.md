# Manifest — Yang et al. (2026) data release

Every artefact in this repository, with its scientific role and source.
For the per-file classification companion (564 rows including
exploratory files NOT shipped in this release), see the audit's
inventory CSV in the main project repository.

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
| `run_chains.sh` | GPU-only chain regeneration stub |

## `paper/` — manuscript

| Path | Role | Source / generator |
|------|------|-------------------|
| `paper/main.tex` | manuscript source (412 lines) | canonical version of `mnras_paper/main.tex` |
| `paper/references.bib` | bibliography (45 entries) | canonical version of `mnras_paper/references.bib` |
| `paper/main.pdf` | built artefact (11 pages, 249-word abstract) | `latexmk -pdf paper/main.tex` |
| `paper/figures/corner_GW150914_waveform_comparison.pdf` | Figure 1 | `scripts/plot_GW150914_waveform_comparison.py` + the s17a chain |
| `paper/figures/H0_prior_sensitivity.pdf` | Figure 2 | `scripts/plot_H0_prior_sensitivity.py` + s14 IMRX × 4 + s18 vpmean × 3 |
| `paper/figures/bimodality_imr_vs_imrx.pdf` | Figure 3 | `scripts/compare_bimodality_waveforms.py` + s10 IMR refModeB + s14 IMRX flatz |
| `paper/figures/bimodality.pdf` | Figure 4 | `scripts/plot_bimodality.py` + s10 IMR dL30-75 / dL10-30 / dL10-75-refModeB |
| `paper/figures/H0_waveform_comparison.pdf` | Figure 5 | `scripts/plot_H0_GW170817_waveform_comparison.py` + s14 IMRX + TF2 baseline |
| `paper/figures/corner_GW170817_waveform_comparison.pdf` | Figure 6 | `scripts/plot_GW170817_waveform_corner.py` + s14 IMRX + TF2 baseline + LVK GWTC-1 HDF5 |
| `paper/figures/scaling_study_full.pdf` | Figure 7 | `scripts/plot_scaling_full.py` + s13 n_live sweep + s07 LVK-bounds anchor |
| `paper/tables/table1_gw150914.tex` | Table 1 (GW150914 validation) | `scripts/build_paper_tables.py` |
| `paper/tables/table4_cross_waveform.tex` | Table 4 (cross-waveform H₀) | `scripts/build_paper_tables.py` |
| `paper/tables/table5_prior_sensitivity.tex` | Table 5 (prior-sensitivity sweep) | `scripts/build_paper_tables.py` |
| `paper/tables/table6_bimodality.tex` | Table 6 (bimodality) | `scripts/build_paper_tables.py` |

## `scripts/` — production figure and table generators

| Path | Role |
|------|------|
| `scripts/_plot_utils.py` | shared plotting helpers (LaTeX setup, palette, HPD/MAP, LVK HDF5 loader) |
| `scripts/build_paper_tables.py` | canonical table & summary generator; emits `paper/tables/*.tex` + `results/gwtc1_phasemarg/{paper_tables,paper_diagnostics}.csv` |
| `scripts/plot_GW150914_waveform_comparison.py` | Figure 1 |
| `scripts/plot_H0_prior_sensitivity.py` | Figure 2 |
| `scripts/compare_bimodality_waveforms.py` | Figure 3 |
| `scripts/plot_bimodality.py` | Figure 4 |
| `scripts/plot_H0_GW170817_waveform_comparison.py` | Figure 5 |
| `scripts/plot_GW170817_waveform_corner.py` | Figure 6 |
| `scripts/plot_scaling_full.py` | Figure 7 |

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

## `results/` — derived summaries (chains live on Zenodo)

| Path | Role |
|------|------|
| `results/gwtc1_phasemarg/evidence_table.csv` | per-variant ln Z ± σ + n_eff (machine-readable) |
| `results/gwtc1_phasemarg/paper_diagnostics.csv` | per-variant n_eff + efficiency + PSIS k̂ + bootstrap CI (reweighted row) |
| `results/gwtc1_phasemarg/paper_tables.csv` | per-row summary statistics underlying Tables 4–6 |
| `results/gwtc1_phasemarg/table{1,4,5,6}*.tex` | LaTeX include fragments (mirror of `paper/tables/`) |
| `results/gwtc1_phasemarg/plots/<7 PDFs + PNG>` | canonical figure PDFs (mirror of `paper/figures/` plus PNG previews) |
| `results/test_suite/run_catalog.csv` | sN__* metadata (one row per chain) |
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
| `results/test_suite/sNN__*/samples.csv` | NS chains — NOT redistributed; deposit on Zenodo. See `docs/chain_regeneration.md`. |

## `docs/`

| Path | Role |
|------|------|
| `docs/reproducibility.md` | fresh-clone → main.pdf recipe |
| `docs/chain_regeneration.md` | per-run BlackJAX-NS invocation, expected wall-clock |
| `docs/data_provenance.md` | which paper claim each summary CSV underwrites |
