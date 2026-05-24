"""
GW150914 validation overlay.

Two-curve overlay (LVK reference vs this work; both IMRPhenomXPHM):
  - LVK GWTC-2.1 IMRPhenomXPHM PE samples
  - Our IMRPhenomXPHM heterodyned run, n_live=8000, n_mcmc=160 (s17a)

Output: Results/gwtc1_phasemarg/plots/corner_GW150914_waveform_comparison.{pdf,png}
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *

PLOT_COLS = [r'$\mathcal{M}_c$', r'$q$', r'$\chi_{\rm eff}$',
             r'$d_L$', r'$\iota$']

datasets = []

# --- LVK GWTC-2.1 reference (IMRPhenomXPHM) ---
gwtc = load_gwtc2p1_gw150914()
datasets.append((gwtc, 'LVK', COLORS['gwtc']))

def _load_GW150914(csv, has_inplane=False):
    """Load a GW150914 nested-sampling CSV and synthesise (M_c, q, chi_eff, d_L, iota)."""
    s = load_nested_csv(csv)
    Mc = s['M_c'].to_numpy()
    q  = s['q'].to_numpy()
    if has_inplane:
        a1 = s['a_1'].to_numpy(); a2 = s['a_2'].to_numpy()
        s1z = a1 * s['cost_1'].to_numpy()
        s2z = a2 * s['cost_2'].to_numpy()
    else:
        s1z = s['s1_z'].to_numpy()
        s2z = s['s2_z'].to_numpy()
    dL = s['d_L'].to_numpy()
    iota = s['iota'].to_numpy()
    chi_eff = (s1z + q * s2z) / (1.0 + q)
    return MCMCSamples(
        np.column_stack([Mc, q, chi_eff, dL, iota]),
        columns=PLOT_COLS,
        weights=np.asarray(s.get_weights()),
    )

# --- Our IMRPhenomXPHM heterodyned, n_live=8000, n_mcmc=160 (s17a) ---
s17a_csv = 'results/test_suite/s17a__gw150914__imrphenomxphm__nlive8000_mcmc160__seed0000/samples.csv'
if os.path.exists(s17a_csv):
    datasets.append((_load_GW150914(s17a_csv, has_inplane=True),
                     'this work (IMRPhenomXPHM)', 'tab:green'))
else:
    raise SystemExit(f"  Missing s17a XPHM CSV: {s17a_csv}")

make_corner(datasets, PLOT_COLS, 'corner_GW150914_waveform_comparison',
            figsize=(12, 12), lims={r'$q$': (0.3, 1.0)})
print("\nDone.")
