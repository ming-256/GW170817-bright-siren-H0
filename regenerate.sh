#!/usr/bin/env bash
# Yang et al. (2026) MNRAS — CPU-only regeneration pipeline.
#
# Inputs:  results/test_suite/sNN__*/samples.csv  (the nested-sampling chains) and their sampler.log
#          files, plus results/gwtc1_phasemarg/*.csv.  The chains are NOT in git -- they are 3.7 GB
#          and live in the Zenodo deposit.  Fetch them once with `bash fetch_data.sh chains`
#          (and check them with `bash fetch_data.sh verify`) before running this.
#          The LVK GWTC-1 GW170817 reference posterior is at $GWTC1_HDF5 (default: results/GW170817_GWTC-1.hdf5).
#          Figure 1 additionally needs the 287 MB LVK GWTC-2.1 GW150914 PE release, which is the one
#          input not committed here: run `bash fetch_data.sh figures` once to get it.  Without it the
#          other seven figures, all four tables and the PDF still build.
# Outputs: results/gwtc1_phasemarg/{table1,4,5,6}*.tex  +  paper_tables.csv + paper_diagnostics.csv + evidence_table.csv
#          results/gwtc1_phasemarg/plots/<7 PDFs>      +  PNG companions
#          paper/figures/<7 PDFs>                       (copies for the LaTeX include path)
#          paper/main.pdf
#
# Env:  conda env create -f environment.yml && conda activate gw170817-bright-siren-H0
# Hardware: CPU only.  Wall-clock ~ 3 min on an M2 MacBook.
#
# Usage:
#   bash regenerate.sh           # full pipeline (tables, figures, PDF)
#   bash regenerate.sh tables    # just the tables and summary CSVs
#   bash regenerate.sh figures   # just the seven figure PDFs
#   bash regenerate.sh pdf       # just the LaTeX build

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY=${PY:-python}
MODE="${1:-all}"

mkdir -p paper/figures paper/tables
mkdir -p results/gwtc1_phasemarg/plots

run_tables() {
    echo "--- Tables (build_paper_tables.py + build_waveform_table.py) ---"
    $PY scripts/build_paper_tables.py
    $PY scripts/build_waveform_table.py
    # Mirror the .tex includes from results/ to paper/tables/
    for t in table1_gw150914 table5_prior_sensitivity table6_bimodality tableW_waveform; do
        cp -f "results/gwtc1_phasemarg/${t}.tex"  "paper/tables/${t}.tex"
    done
}

run_figures() {
    echo "--- Figures ---"
    # Figure 1 overlays the LVK GWTC-2.1 GW150914 posterior, the one input we
    # do not redistribute. Skip it rather than aborting the whole run, so a
    # clone without that 287 MB download still gets the other seven figures.
    GWTC2P1="${GWTC2P1_GW150914_HDF5:-results/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_nocosmo.h5}"
    if [[ -s "$GWTC2P1" ]]; then
        $PY scripts/plot_GW150914_waveform_comparison.py   # Fig 1
        FIG1_OK=1
    else
        echo "  SKIP Fig 1: ${GWTC2P1} not found."
        echo "       Run 'bash fetch_data.sh figures' to download it (287 MB), then re-run."
        FIG1_OK=0
    fi
    $PY scripts/plot_H0_prior_sensitivity.py           # Fig 2
    $PY scripts/compare_bimodality_waveforms.py        # Fig 3
    $PY scripts/plot_bimodality.py                     # Fig 4
    $PY scripts/plot_H0_GW170817_waveform_comparison.py # Fig 5
    $PY scripts/plot_GW170817_waveform_corner.py       # Fig 6
    $PY scripts/plot_corner_IMRPhenomD_hetero_vs_unhetero.py  # Fig 7
    $PY scripts/plot_scaling_full.py                   # supplementary scaling plot (not a paper figure)
    # Mirror PDFs to paper/figures/
    MIRROR=(H0_prior_sensitivity bimodality H0_waveform_comparison corner_GW170817_waveform_comparison corner_IMRPhenomD_hetero_vs_unhetero scaling_study_full)
    [[ "$FIG1_OK" == 1 ]] && MIRROR+=(corner_GW150914_waveform_comparison)
    for f in "${MIRROR[@]}"; do
        cp -f "results/gwtc1_phasemarg/plots/${f}.pdf"  "paper/figures/${f}.pdf"
    done
    # Fig 3 is written by compare_bimodality_waveforms.py directly to paper/figures/.
    cp -f "paper/figures/bimodality_imr_vs_imrx.pdf"  "results/gwtc1_phasemarg/plots/" || true
}

run_pdf() {
    echo "--- LaTeX build ---"
    ( cd paper && latexmk -pdf -interaction=nonstopmode main.tex )
}

case "$MODE" in
    tables)   run_tables ;;
    figures)  run_figures ;;
    pdf)      run_pdf ;;
    all)
        run_tables
        run_figures
        run_pdf
        ;;
    *)
        echo "Unknown mode: $MODE.  Use one of: all | tables | figures | pdf"
        exit 1
        ;;
esac

echo ""
echo "=== regenerate.sh: complete ==="
ls -1 paper/main.pdf 2>/dev/null && echo "  paper/main.pdf is up to date"
echo "  results/gwtc1_phasemarg/plots/ has $(ls results/gwtc1_phasemarg/plots/*.pdf 2>/dev/null | wc -l | tr -d ' ') figure PDFs"
echo "  results/gwtc1_phasemarg/  has $(ls results/gwtc1_phasemarg/table*.tex 2>/dev/null | wc -l | tr -d ' ') table .tex files"
