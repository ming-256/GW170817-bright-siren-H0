"""
H_0 posterior: IMRPhenomD_NRTidalv2 — baseline / flat-in-z / vp250.
With MAP, HPD 1σ/2σ intervals, SHoES and Planck bands.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *

CSVS = [
    (os.path.join(RESULTS_DIR,
        'gwtc1_phasemarg/PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_baseline.csv'),
     r'IMRPhenomD baseline ($\sigma_{v_p}$=150)', COLORS['imr_baseline']),
    (os.path.join(RESULTS_DIR,
        'gwtc1_phasemarg/PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_flatZ.csv'),
     r'IMRPhenomD flat-in-$z$', COLORS['flatZ']),
    (os.path.join(RESULTS_DIR,
        'gwtc1_phasemarg/PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_vp250.csv'),
     r'IMRPhenomD $\sigma_{v_p}$=250', COLORS['vp250']),
]

runs = []
for csv_path, label, color in CSVS:
    if os.path.exists(csv_path):
        runs.append((load_nested_csv(csv_path), label, color))
    else:
        print(f"  WARNING: {csv_path} not found — skipping")

if runs:
    plot_h0(runs, 'H0_IMRPhenomD_variants')
else:
    print("  No data found.")

print("\nDone.")
