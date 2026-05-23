"""
Corner plot: GW170817 — baseline TaylorF2 + baseline IMRPhenomD + GWTC-1.

Parameters: M_c, q, d_L, iota
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *

IMR_CSV = os.path.join(RESULTS_DIR,
    'gwtc1_phasemarg/PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_baseline.csv')
TF2_CSV = os.path.join(RESULTS_DIR,
    'gwtc1_phasemarg/PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_baseline.csv')

plot_params = ['M_c', 'q', 'd_L', 'iota']

# GWTC-1 reference
gwtc = load_gwtc1_gw170817(columns=plot_params)
datasets = [(gwtc, 'LVK (GWTC-1)', COLORS['gwtc'])]

if os.path.exists(IMR_CSV):
    datasets.append((load_nested_csv(IMR_CSV),
                     'IMRPhenomD (this work)', COLORS['imr_baseline']))

if os.path.exists(TF2_CSV):
    datasets.append((load_nested_csv(TF2_CSV),
                     'TaylorF2 (this work)', COLORS['tf2_baseline']))

make_corner(datasets, plot_params, 'corner_combined_waveforms')
print("\nDone.")
