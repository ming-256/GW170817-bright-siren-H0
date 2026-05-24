# Data provenance

Where each derived summary CSV in this release came from, and which
paper claim it underwrites.

## `results/gwtc1_phasemarg/`

### `paper_tables.csv`

One row per Table 4 / 5 / 6 entry, with MAP, weighted median, 68 %/95 % HPDs,
P(H₀ > 120/150 km/s/Mpc), n_eff, ln Z ± σ.

- Generator: `scripts/build_paper_tables.py`
- Source chains: `results/test_suite/{s07__*_lvkbounds, s14__*xas*, s18__*xas*_vpmean*, s10__*_dL*, s18__*_dL*, s17a__gw150914}/samples.csv`
- Underwrites: Tables 4, 5, 6 in the paper; the headline numbers in the abstract

### `paper_diagnostics.csv`

Per-variant `n_eff`, sampling efficiency, plus PSIS k̂ and the
reweighted-bootstrap 95 % CI for the reweighted IMRX row.

- Generator: `scripts/build_paper_tables.py` (n_eff columns) and
  `analysis/analyze_psis_khat.py` (PSIS k̂ + bootstrap columns)
- Source chains: same as `paper_tables.csv`
- Underwrites: §4.1 "Effective-sample-size diagnostic" and "Reweighting bias versus variance"

### `evidence_table.csv`

Per-run ln Z ± σ, D_KL, N_dead, n_eff. Machine-readable companion to the
ln Z values quoted in the body. Includes both this work's runs and a
small set of historical TF2/IMR unheterodyned baselines for cross-check.

- Generator: `scripts/build_paper_tables.py` (subset) plus a hand-curated
  block for the unheterodyned reference baselines.
- Underwrites: §3.2 heterodyned-vs-unheterodyned consistency check;
  Appendix A.

### `table{1,4,5,6}*.tex`

LaTeX `\input` fragments built by `build_paper_tables.py`. Mirrored to
`paper/tables/` for the LaTeX build.

### `plots/*.pdf` (and `.png`)

Mirror of `paper/figures/`. Both the canonical (`paper/figures/`) and
the `results/gwtc1_phasemarg/plots/` copies are byte-identical for the
seven figures cited by the paper.

## `results/test_suite/`

### `run_catalog.csv`

One row per chain. Columns: `run_id`, `session`, `event`, `waveform`,
`prior`, `variant`, `seed`, `n_live`, `status`, `started_at`,
`finished_at`. Maintained by hand.

### `bimodality_summary.csv` — IMR Mode-A / Mode-B

- Generator: `analysis/analyze_bimodality.py`
- Underwrites: Table 6 (upper block, seed=0); §5 ln 𝓑(B/A) = −0.66 claim

### `bimodality_imrx_summary.csv` — IMRX Mode-A / Mode-B (queued)

- Generator: `analysis/analyze_bimodality_imrx.py`
- Underwrites: queued IMRX cross-check; cited as follow-up

### `bimodality_waveform_check.csv` — IMR/IMRX cross-waveform

- Generator: `scripts/compare_bimodality_waveforms.py`
- Underwrites: Figure 3 (`bimodality_imr_vs_imrx.pdf`); §5 "the bimodality is robust across the NRTidalv2 → NRTidalv3 calibration"

### `gw150914_waveform_comparison.csv`

- Generator: incidental output of `scripts/build_paper_tables.py` (the
  GW150914 row builder writes a per-param median + HPD)
- Underwrites: Table 1

### `gw170817_waveform_comparison.csv`

- Generator: `scripts/build_paper_tables.py`
- Underwrites: Table 4

### `het_bins_sweep_summary.csv` + `het_bins_sweep_wasserstein.csv`

- Generator: `analysis/analyze_het_bins_sweep.py`
- Underwrites: Appendix A — heterodyne-bin count sweep

### `num_delete_sweep_summary.csv`

- Generator: `analysis/analyze_num_delete_sweep.py`
- Underwrites: Appendix A — n_delete sweep

### `psd_sensitivity_summary.csv`

- Generator: `analysis/analyze_psd_sensitivity.py`
- Underwrites: Appendix A — PSD source sensitivity

### `seed_ensemble_summary.csv` + `seed_ensemble_bayes_factor.csv`

- Generator: `analysis/analyze_seed_ensemble.py`
- Underwrites: §5 "the run-to-run ln Z scatter is larger than the nominal ±0.1 per-run uncertainty…the conclusion that survives this scatter is the sign-independent one"

### `selection_term_Ns.csv`

- Generator: `analysis/analyze_selection_term.py`
- Underwrites: footnote at §2.4 — selection-term cancellation
