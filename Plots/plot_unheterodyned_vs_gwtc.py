"""
Corner + H_0 plots: unheterodyned normal samples vs GWTC-1.

Compares the standard unheterodyned runs (narrow sky prior, focused on
GW170817 localization) against the LVK GWTC-1 posteriors.
Both IMRPhenomD_NRTidalv2 and TaylorF2 waveform models are shown.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *

RESULTS_PHASEMARG = os.path.join(RESULTS_DIR, 'gwtc1_phasemarg')

UNHETERO_IMR_CSV = os.path.join(RESULTS_PHASEMARG,
    'PhaseMarg_Unheterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1.csv')
UNHETERO_TF2_CSV = os.path.join(RESULTS_PHASEMARG,
    'PhaseMarg_Unheterodyned_TaylorF2_local_psd-gwtc1.csv')

# ----------------------------------------------------------------------- #
# Corner plot: M_c, q, d_L, iota
# ----------------------------------------------------------------------- #
plot_params = ['M_c', 'q', 'd_L', 'iota']

gwtc = load_gwtc1_gw170817(columns=plot_params)
datasets = [(gwtc, 'LVK (GWTC-1)', COLORS['gwtc'])]

if os.path.exists(UNHETERO_IMR_CSV):
    datasets.append((load_nested_csv(UNHETERO_IMR_CSV),
                     'IMRPhenomD unheterodyned', COLORS['unhetero_imr']))

if os.path.exists(UNHETERO_TF2_CSV):
    datasets.append((load_nested_csv(UNHETERO_TF2_CSV),
                     'TaylorF2 unheterodyned', COLORS['unhetero_tf2']))

make_corner(datasets, plot_params, 'corner_unheterodyned_vs_gwtc')

# ----------------------------------------------------------------------- #
# H_0 posterior
# ----------------------------------------------------------------------- #
h0_runs = []
for ds, label, color in datasets[1:]:  # skip GWTC (no H_0)
    if 'H_0' in ds.columns.get_level_values(0):
        h0_runs.append((ds, label, color))

if h0_runs:
    plot_h0(h0_runs, 'H0_unheterodyned_vs_gwtc')
else:
    print("  (no H_0 column — skipping H_0 plot)")

print("\nDone.")
