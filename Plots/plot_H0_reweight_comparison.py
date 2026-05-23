"""
H_0 posterior comparison: reweighted flat-in-z vs sampled flat-in-z.
Both waveforms (IMRPhenomD and TaylorF2) overlaid.
With MAP, HPD 1σ/2σ intervals, SHoES and Planck bands.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *

RESULTS_PHASEMARG = os.path.join(RESULTS_DIR, 'gwtc1_phasemarg')

# Sampled flat-in-z runs
IMR_FLATZ_CSV = os.path.join(RESULTS_PHASEMARG,
    'PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_flatZ.csv')
#TF2_FLATZ_CSV = os.path.join(RESULTS_PHASEMARG,
    #'PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_flatZ.csv')

# Reweighted flat-in-z runs
IMR_REWEIGHTED_CSV = os.path.join(RESULTS_PHASEMARG,
    'PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_reweighted_flatZ.csv')
#TF2_REWEIGHTED_CSV = os.path.join(RESULTS_PHASEMARG,
    #'PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_reweighted_flatZ.csv')

runs = []

if os.path.exists(IMR_FLATZ_CSV):
    runs.append((load_nested_csv(IMR_FLATZ_CSV),
                 r'IMRPhenomD sampled flat-in-$z$', COLORS['flatZ']))

if os.path.exists(IMR_REWEIGHTED_CSV):
    runs.append((load_reweighted_csv(IMR_REWEIGHTED_CSV),
                 r'IMRPhenomD reweighted flat-in-$z$', COLORS['reweighted']))

#if os.path.exists(TF2_FLATZ_CSV):
 #   runs.append((load_nested_csv(TF2_FLATZ_CSV),
  #               r'TaylorF2 sampled flat-in-$z$', 'tab:green'))

#if os.path.exists(TF2_REWEIGHTED_CSV):
#    runs.append((load_reweighted_csv(TF2_REWEIGHTED_CSV),
#                 r'TaylorF2 reweighted flat-in-$z$', 'tab:purple'))

if runs:
    plot_h0(runs, 'H0_reweight_comparison')
else:
    print("  No data found.")

print("\nDone.")
