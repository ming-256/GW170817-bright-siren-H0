"""
H_0 posterior: IMRPhenomD_NRTidalv2 baseline only.
With MAP, HPD 1σ/2σ intervals, SHoES and Planck bands.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *

BASELINE_CSV = os.path.join(RESULTS_DIR,
    'gwtc1_phasemarg/PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_baseline.csv')

runs = []
if os.path.exists(BASELINE_CSV):
    runs.append((load_nested_csv(BASELINE_CSV),
                 'IMRPhenomD (this work)', COLORS['imr_baseline']))
else:
    print(f"  WARNING: {BASELINE_CSV} not found")

if runs:
    plot_h0(runs, 'H0_baseline_IMRPhenomD')
else:
    print("  No data found.")

print("\nDone.")
