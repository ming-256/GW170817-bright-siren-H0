"""
Corner + H_0 plots: full-sky vs narrow-sky (event localization) priors.

Compares unheterodyned runs with wide sky prior ('full_sky') against
the standard narrow sky prior focused on the GW170817 localization,
both overlaid with LVK GWTC-1 posteriors.

Both IMRPhenomD_NRTidalv2 and TaylorF2 waveform models are shown.

NOTE: This script expects *full_sky* CSV files in Results/gwtc1_phasemarg/.
      It will skip gracefully if these are not yet available.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *

RESULTS_PHASEMARG = os.path.join(RESULTS_DIR, 'gwtc1_phasemarg')

# Full sky
FULL_SKY_IMR_CSV = os.path.join(RESULTS_PHASEMARG,
    'PhaseMarg_Unheterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_full_sky.csv')
FULL_SKY_TF2_CSV = os.path.join(RESULTS_PHASEMARG,
    'PhaseMarg_Unheterodyned_TaylorF2_local_psd-gwtc1_full_sky.csv')

# Narrow sky (standard unheterodyned — event localization)
NARROW_IMR_CSV = os.path.join(RESULTS_PHASEMARG,
    'PhaseMarg_Unheterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1.csv')
NARROW_TF2_CSV = os.path.join(RESULTS_PHASEMARG,
    'PhaseMarg_Unheterodyned_TaylorF2_local_psd-gwtc1.csv')

COLORS_SKY = {
    'narrow_imr': COLORS['imr_baseline'],
    'full_imr':   COLORS['unhetero_imr'],
    'narrow_tf2': COLORS['tf2_baseline'],
    'full_tf2':   COLORS['unhetero_tf2'],
}

plot_params = ['M_c', 'q', 'd_L', 'iota']

# ----------------------------------------------------------------------- #
# Check for full_sky files — warn if not present
# ----------------------------------------------------------------------- #
full_sky_found = os.path.exists(FULL_SKY_IMR_CSV) or os.path.exists(FULL_SKY_TF2_CSV)
if not full_sky_found:
    print("  WARNING: No *full_sky* CSV files found yet.")
    print("           Expected at:")
    print(f"             {FULL_SKY_IMR_CSV}")
    print(f"             {FULL_SKY_TF2_CSV}")
    print("           Skipping full-sky plots — re-run once sampling completes.")
    sys.exit(0)

# ----------------------------------------------------------------------- #
# Corner plot: full sky vs narrow sky vs GWTC-1
# ----------------------------------------------------------------------- #
gwtc = load_gwtc1_gw170817(columns=plot_params)
datasets = [(gwtc, 'LVK (GWTC-1)', COLORS['gwtc'])]

if os.path.exists(NARROW_IMR_CSV):
    datasets.append((load_nested_csv(NARROW_IMR_CSV),
                     'IMRPhenomD narrow sky', COLORS_SKY['narrow_imr']))
if os.path.exists(FULL_SKY_IMR_CSV):
    datasets.append((load_nested_csv(FULL_SKY_IMR_CSV),
                     'IMRPhenomD full sky', COLORS_SKY['full_imr']))

if os.path.exists(NARROW_TF2_CSV):
    datasets.append((load_nested_csv(NARROW_TF2_CSV),
                     'TaylorF2 narrow sky', COLORS_SKY['narrow_tf2']))
if os.path.exists(FULL_SKY_TF2_CSV):
    datasets.append((load_nested_csv(FULL_SKY_TF2_CSV),
                     'TaylorF2 full sky', COLORS_SKY['full_tf2']))

make_corner(datasets, plot_params, 'corner_full_sky_vs_narrow')

# ----------------------------------------------------------------------- #
# Sky localization corner: ra, dec comparison
# ----------------------------------------------------------------------- #
sky_params = ['ra', 'dec', 'd_L', 'iota']
sky_datasets = []

if os.path.exists(NARROW_IMR_CSV):
    sky_datasets.append((load_nested_csv(NARROW_IMR_CSV),
                         'IMRPhenomD narrow sky', COLORS_SKY['narrow_imr']))
if os.path.exists(FULL_SKY_IMR_CSV):
    sky_datasets.append((load_nested_csv(FULL_SKY_IMR_CSV),
                         'IMRPhenomD full sky', COLORS_SKY['full_imr']))
if os.path.exists(NARROW_TF2_CSV):
    sky_datasets.append((load_nested_csv(NARROW_TF2_CSV),
                         'TaylorF2 narrow sky', COLORS_SKY['narrow_tf2']))
if os.path.exists(FULL_SKY_TF2_CSV):
    sky_datasets.append((load_nested_csv(FULL_SKY_TF2_CSV),
                         'TaylorF2 full sky', COLORS_SKY['full_tf2']))

if sky_datasets:
    make_corner(sky_datasets, sky_params, 'corner_sky_localization',
                figsize=(10, 10))

# ----------------------------------------------------------------------- #
# H_0 posterior: full sky vs narrow sky
# ----------------------------------------------------------------------- #
h0_runs = []
for ds, label, color in datasets[1:]:  # skip GWTC (no H_0)
    try:
        cols = ds.columns.get_level_values(0)
    except AttributeError:
        cols = ds.columns
    if 'H_0' in cols:
        h0_runs.append((ds, label, color))

if h0_runs:
    plot_h0(h0_runs, 'H0_full_sky_vs_narrow')
else:
    print("  (no H_0 column — skipping H_0 plot)")

print("\nDone.")
