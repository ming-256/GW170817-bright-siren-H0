"""
Corner plot: reweighted flat-in-z vs sampled flat-in-z posteriors (XAS only).

Identifies systematic differences between post-hoc reweighting of the
baseline posterior to a flat-in-z prior vs directly sampling with the
flat-in-z prior, for the locked primary waveform IMRPhenomXAS_NRTidalv3.

Output: Results/gwtc1_phasemarg/plots/
  corner_reweighted_vs_sampled_flatZ_XAS.{pdf,png}
  H0_reweighted_vs_sampled_flatZ.{pdf,png}
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *

XAS_BASE = 'Results/test_suite/s14__gw170817__imrphenomxas_nrtidalv3'
SAMPLED_CSV    = f'{XAS_BASE}__flatz__seed0000/samples.csv'
REWEIGHTED_CSV = f'{XAS_BASE}__reweighted_flatz__seed0000/samples.csv'

plot_params = ['M_c', 'q', 's1_z', 's2_z', 'd_L', 'iota', 'H_0']

datasets = []
if os.path.exists(SAMPLED_CSV):
    datasets.append((load_nested_csv(SAMPLED_CSV),
                     r'XAS_NRTv3 sampled flat-in-$z$', COLORS['flatZ']))
else:
    print(f"  WARNING: missing {SAMPLED_CSV}")

if os.path.exists(REWEIGHTED_CSV):
    # The s14 reweighted_flatz run is a fresh nested-sampling run with the
    # reweighting applied — same loader as a normal nested-sampling CSV.
    datasets.append((load_nested_csv(REWEIGHTED_CSV),
                     r'XAS_NRTv3 reweighted flat-in-$z$', COLORS['reweighted']))
else:
    print(f"  WARNING: missing {REWEIGHTED_CSV}")

if datasets:
    make_corner(datasets, plot_params,
                'corner_reweighted_vs_sampled_flatZ_XAS',
                figsize=(14, 14))

    h0_runs = [(ds, lab, col) for (ds, lab, col) in datasets
               if 'H_0' in (ds.columns.get_level_values(0)
                            if hasattr(ds.columns, 'get_level_values')
                            else ds.columns)]
    if h0_runs:
        plot_h0(h0_runs, 'H0_reweighted_vs_sampled_flatZ')

print("\nDone.")
