"""
GW170817 multi-waveform corner plot: M_c, q, chi_eff, d_L, iota, H_0.

Overlays our four-waveform suite against the GWTC-1 reference for context.
Output: Results/gwtc1_phasemarg/plots/corner_GW170817_waveform_comparison.{pdf,png}
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *

PLOT_COLS = [r'$\mathcal{M}_c$', r'$q$', r'$d_L$', r'$\iota$']

def _to_samples(s):
    return MCMCSamples(
        np.column_stack([s['M_c'].to_numpy(), s['q'].to_numpy(),
                         s['d_L'].to_numpy(), s['iota'].to_numpy()]),
        columns=PLOT_COLS,
        weights=np.asarray(s.get_weights()),
    )

# GWTC-1 reference (PhenomPv2_NRTidal low-spin posterior)
gwtc1 = load_gwtc1_gw170817(columns=['M_c', 'q', 'd_L', 'iota'])
gwtc1_4d = MCMCSamples(
    np.column_stack([gwtc1['M_c'].to_numpy(), gwtc1['q'].to_numpy(),
                     gwtc1['d_L'].to_numpy(), gwtc1['iota'].to_numpy()]),
    columns=PLOT_COLS,
)

datasets = [(gwtc1_4d, 'GWTC-1 IMRPhenomPv2_NRTidal', COLORS['gwtc'])]

# LVK-matched prior (m_comp in [0.5, 7.7] M_sun, Mc_det in [1.184, 2.168]
# M_sun — the GW170817 PE prior of Abbott et al. 2019, PRX 9, 011001).
# IMRX: s14 baseline.  TF2: gwtc1_phasemarg baseline (GWTC-1 PSD, ref-gwtc1).
for csv, label, colour in [
    (os.path.join(RESULTS_DIR, 'test_suite',
                  's14__gw170817__imrphenomxas_nrtidalv3__baseline__seed0000',
                  'samples.csv'),
     'this work (IMRPhenomXAS_NRTidalv3)', COLORS['imr_baseline']),
    (os.path.join(RESULTS_DIR, 'gwtc1_phasemarg',
                  'PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_baseline.csv'),
     'this work (TaylorF2)', COLORS['tf2_baseline']),
]:
    if os.path.exists(csv):
        s = load_nested_csv(csv)
        datasets.append((_to_samples(s), label, colour))
    else:
        print(f"  WARNING: missing {csv}")

make_corner(datasets, PLOT_COLS, 'corner_GW170817_waveform_comparison',
            figsize=(13, 13),
            lims={r'$\iota$': (np.pi/2, np.pi)})
print("\nDone.")
