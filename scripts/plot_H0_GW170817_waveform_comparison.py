"""
GW170817 H_0 posterior — two-waveform comparison plus published LVK reference.

Overlays the two waveforms reported in Table~\\ref{tab:waveform-h0} and
references the published Abbott+2017 GW170817 H_0 = 70 +12/-8 km/s/Mpc
band as a vertical reference (no derived posterior).

  - IMRPhenomXAS_NRTidalv3   (s14 baseline, LVK-matched prior)  — primary
  - TaylorF2                 (gwtc1_phasemarg baseline)         — family check
  - Abbott+2017 published H_0 band                              — literature reference

The LVK-matched prior is the GW170817 PE prior of Abbott et al. 2019
(PRX 9, 011001): component masses uniform in [0.5, 7.7] M_sun. Both runs use
that prior, the GWTC-1 PSD and the GWTC-1 heterodyne reference.

IMRPhenomD_NRTidalv2 (anchor) results are reported in Appendix~\\ref{app:robustness}
only and stay out of this figure.

Output: Results/gwtc1_phasemarg/plots/H0_waveform_comparison.{pdf,png}
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import (
    OUT_DIR, RESULTS_DIR, COLORS, load_nested_csv, plot_h0_kde,
)

CSV_XAS_NRTV3   = 'results/test_suite/s14__gw170817__imrphenomxas_nrtidalv3__baseline__seed0000/samples.csv'
CSV_TF2_LVK     = 'results/gwtc1_phasemarg/PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_baseline.csv'

runs = []
for csv, label, colour in [
    (CSV_XAS_NRTV3, 'IMRPhenomXAS_NRTidalv3', COLORS['imr_baseline']),
    (CSV_TF2_LVK,   'TaylorF2',               COLORS['tf2_baseline']),
]:
    if os.path.exists(csv):
        s = load_nested_csv(csv)
        runs.append((s, label, colour))
    else:
        print(f"  WARNING: missing {csv}")

if runs:
    plot_h0_kde(runs, 'H0_waveform_comparison', xlim=(40, 180),
                add_planck_shoes=True, lvk_band=False, hpd_lines=False,
                figsize=(6, 4))
print("\nDone.")
