"""
Corner + H_0 plots: small H_0 prior vs normal prior (both unheterodyned).

Both runs use narrow sky priors focused on GW170817 localization.
The 'small_h0_prior' restricts the H_0 prior range, while the 'normal'
run uses the standard wider H_0 prior.  This comparison demonstrates
the sensitivity of H_0 inference to the prior volume.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *

RESULTS_PHASEMARG = os.path.join(RESULTS_DIR, 'gwtc1_phasemarg')

# Normal (standard H_0 prior)
NORMAL_IMR_CSV = os.path.join(RESULTS_PHASEMARG,
    'PhaseMarg_Unheterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1.csv')
NORMAL_TF2_CSV = os.path.join(RESULTS_PHASEMARG,
    'PhaseMarg_Unheterodyned_TaylorF2_local_psd-gwtc1.csv')

# Small H_0 prior
SMALL_IMR_CSV = os.path.join(RESULTS_PHASEMARG,
    'PhaseMarg_Unheterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_small_h0_prior.csv')
SMALL_TF2_CSV = os.path.join(RESULTS_PHASEMARG,
    'PhaseMarg_Unheterodyned_TaylorF2_local_psd-gwtc1_small_h0_prior.csv')

COLORS_PRIOR = {
    'normal_imr':  COLORS['imr_baseline'],
    'small_imr':   COLORS['small_h0_imr'],
    'normal_tf2':  COLORS['tf2_baseline'],
    'small_tf2':   COLORS['small_h0_tf2'],
}

# ----------------------------------------------------------------------- #
# Corner plot: IMRPhenomD normal vs small H_0 prior
# ----------------------------------------------------------------------- #
plot_params = ['M_c', 'q', 'd_L', 'iota']

datasets_imr = []
if os.path.exists(NORMAL_IMR_CSV):
    datasets_imr.append((load_nested_csv(NORMAL_IMR_CSV),
                         r'IMRPhenomD (standard $H_0$ prior)',
                         COLORS_PRIOR['normal_imr']))
if os.path.exists(SMALL_IMR_CSV):
    datasets_imr.append((load_nested_csv(SMALL_IMR_CSV),
                         r'IMRPhenomD (narrow $H_0$ prior)',
                         COLORS_PRIOR['small_imr']))

if datasets_imr:
    make_corner(datasets_imr, plot_params, 'corner_h0_prior_IMRPhenomD')

# ----------------------------------------------------------------------- #
# Corner plot: TaylorF2 normal vs small H_0 prior
# ----------------------------------------------------------------------- #
datasets_tf2 = []
if os.path.exists(NORMAL_TF2_CSV):
    datasets_tf2.append((load_nested_csv(NORMAL_TF2_CSV),
                         r'TaylorF2 (standard $H_0$ prior)',
                         COLORS_PRIOR['normal_tf2']))
if os.path.exists(SMALL_TF2_CSV):
    datasets_tf2.append((load_nested_csv(SMALL_TF2_CSV),
                         r'TaylorF2 (narrow $H_0$ prior)',
                         COLORS_PRIOR['small_tf2']))

if datasets_tf2:
    make_corner(datasets_tf2, plot_params, 'corner_h0_prior_TaylorF2')

# ----------------------------------------------------------------------- #
# H_0 posterior comparison: all four runs overlaid
# ----------------------------------------------------------------------- #
h0_runs = []
for datasets, pairs in [
    (datasets_imr, [('normal_imr', 'small_imr')]),
    (datasets_tf2, [('normal_tf2', 'small_tf2')]),
]:
    for ds, label, color in datasets:
        if 'H_0' in ds.columns.get_level_values(0):
            h0_runs.append((ds, label, color))

if h0_runs:
    plot_h0(h0_runs, 'H0_prior_comparison')
else:
    print("  (no H_0 column — skipping H_0 plot)")

print("\nDone.")
