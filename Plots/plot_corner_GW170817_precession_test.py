"""
Precession-isolation corner plot for GW170817.

Compares three posteriors that share identical LVK BNS mass bounds
[0.87, 1.74] M_sun so that the ONLY variable between the heterodyned runs
is the spin treatment (aligned vs precessing):

  1. GWTC-1  IMRPhenomPv2_NRTidal   (LVK reference, precessing + tides)
  2. s07 XAS IMRPhenomXAS_NRTidalv3 (aligned spin + tides, LVK bounds)
  3. s07 Pv2 IMRPhenomPv2           (precessing, NO tides, LVK bounds)

Comparing (2) vs (3) isolates the precession effect on the q posterior.
Comparing (1) vs (3) isolates the tidal effect (both precessing).

Output: Results/gwtc1_phasemarg/plots/corner_GW170817_precession_test.{pdf,png}
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *

PLOT_COLS = [r'$\mathcal{M}_c$', r'$q$', r'$\chi_{\rm eff}$',
             r'$d_L$', r'$\iota$']

def _to_chi_eff(s, has_inplane=False):
    Mc  = s['M_c'].to_numpy()
    q   = s['q'].to_numpy()
    if has_inplane:
        s1z = s['a_1'].to_numpy() * s['cost_1'].to_numpy()
        s2z = s['a_2'].to_numpy() * s['cost_2'].to_numpy()
    else:
        s1z = s['s1_z'].to_numpy()
        s2z = s['s2_z'].to_numpy()
    dL   = s['d_L'].to_numpy()
    iota = s['iota'].to_numpy()
    chi_eff = (s1z + q * s2z) / (1.0 + q)
    return MCMCSamples(
        np.column_stack([Mc, q, chi_eff, dL, iota]),
        columns=PLOT_COLS,
        weights=np.asarray(s.get_weights()),
    )

# GWTC-1 reference
gwtc1 = load_gwtc1_gw170817(columns=['M_c', 'q', 's1_z', 's2_z', 'd_L', 'iota'])
q_g   = gwtc1['q'].to_numpy()
chi_g = (gwtc1['s1_z'].to_numpy() + q_g * gwtc1['s2_z'].to_numpy()) / (1.0 + q_g)
gwtc1_5d = MCMCSamples(
    np.column_stack([gwtc1['M_c'].to_numpy(), q_g, chi_g,
                     gwtc1['d_L'].to_numpy(), gwtc1['iota'].to_numpy()]),
    columns=PLOT_COLS,
)

datasets = [(gwtc1_5d, 'GWTC-1 IMRPhenomPv2_NRTidal', COLORS['gwtc'])]

for csv, label, colour, has_inplane in [
    (os.path.join(RESULTS_DIR, 'test_suite',
                  's07__gw170817__imrphenomxas_nrtidalv3__baseline_lvkbounds__seed0000',
                  'samples.csv'),
     'IMRPhenomXAS_NRTidalv3 (aligned, LVK bounds)', COLORS['imr_baseline'], False),
    (os.path.join(RESULTS_DIR, 'test_suite',
                  's07__gw170817__imrphenompv2__baseline_lvkbounds__seed0000',
                  'samples.csv'),
     'IMRPhenomPv2 (precessing, no tides, LVK bounds)', COLORS['tf2_baseline'], True),
]:
    if os.path.exists(csv):
        s = load_nested_csv(csv)
        datasets.append((_to_chi_eff(s, has_inplane), label, colour))
    else:
        print(f"  WARNING: missing {csv}")

make_corner(datasets, PLOT_COLS, 'corner_GW170817_precession_test',
            figsize=(13, 13))
print("\nDone.")
