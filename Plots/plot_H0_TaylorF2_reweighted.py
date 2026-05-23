"""
H_0 posterior: TaylorF2 — baseline / reweighted flat-in-z / vp250.
Uses the post-hoc reweighted flat-in-z samples instead of the sampled flat-in-z.
With MAP, HPD 1σ/2σ intervals, SHoES and Planck bands.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *

BASELINE_CSV = os.path.join(RESULTS_DIR,
    'gwtc1_phasemarg/PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_baseline.csv')
REWEIGHTED_CSV = os.path.join(RESULTS_DIR,
    'gwtc1_phasemarg/PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_reweighted_flatZ.csv')
VP250_CSV = os.path.join(RESULTS_DIR,
    'gwtc1_phasemarg/PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_vp250.csv')

runs = []

if os.path.exists(BASELINE_CSV):
    runs.append((load_nested_csv(BASELINE_CSV),
                 r'TaylorF2 baseline ($\sigma_{v_p}$=150)', COLORS['tf2_baseline']))

if os.path.exists(REWEIGHTED_CSV):
    runs.append((load_reweighted_csv(REWEIGHTED_CSV),
                 r'TaylorF2 reweighted flat-in-$z$', COLORS['reweighted']))

if os.path.exists(VP250_CSV):
    runs.append((load_nested_csv(VP250_CSV),
                 r'TaylorF2 $\sigma_{v_p}$=250', COLORS['vp250']))

if runs:
    plot_h0(runs, 'H0_TaylorF2_reweighted')
else:
    print("  No data found.")

print("\nDone.")
