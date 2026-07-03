"""
Corner plot: GW170817 — baseline IMRPhenomD (heterodyned) vs
unheterodyned IMRPhenomD vs GWTC-1.

Parameters: M_c, q, d_L, iota
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *

BASELINE_CSV = os.path.join(RESULTS_DIR,
    'gwtc1_phasemarg/PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_baseline.csv')
UNHETERO_CSV = os.path.join(RESULTS_DIR,
    'gwtc1_phasemarg/PhaseMarg_Unheterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1.csv')

plot_params = ['M_c', 'q', 'd_L', 'iota']

# GWTC-1 reference
gwtc = load_gwtc1_gw170817(columns=plot_params)
datasets = [(gwtc, 'LVK (GWTC-1)', COLORS['gwtc'])]

if os.path.exists(BASELINE_CSV):
    datasets.append((load_nested_csv(BASELINE_CSV),
                     'IMRPhenomD heterodyned', COLORS['imr_baseline']))

if os.path.exists(UNHETERO_CSV):
    datasets.append((load_nested_csv(UNHETERO_CSV),
                     'IMRPhenomD unheterodyned', COLORS['unhetero_imr']))

make_corner(datasets, plot_params, 'corner_IMRPhenomD_hetero_vs_unhetero')
print("\nDone.")
