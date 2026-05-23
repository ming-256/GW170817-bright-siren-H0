"""
Corner plot: GW150914 — our IMRPhenomD results vs GWTC-2p1.

Parameters: M_c, q, chi_eff, d_L, iota
  (chi_eff derived from s1_z, s2_z, M_c, q)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
OUR_CSV = os.path.join(RESULTS_DIR,
    'gwtc1_phasemarg/GW15_PhaseMarg_Heterodyned_IMRPhenomD_local_psd-gwtc2p1_ref-gwtc1.csv')

plot_columns = [r'$\mathcal{M}_c$', r'$q$', r'$\chi_{\rm eff}$',
                r'$d_L$', r'$\iota$']

# --------------------------------------------------------------------------- #
# Load GWTC-2p1 reference
# --------------------------------------------------------------------------- #
gwtc = load_gwtc2p1_gw150914()

# --------------------------------------------------------------------------- #
# Load our results
# --------------------------------------------------------------------------- #
datasets = [(gwtc, 'LVK (GWTC-2p1)', COLORS['gwtc'])]

if os.path.exists(OUR_CSV):
    s = load_nested_csv(OUR_CSV)
    Mc = s['M_c'].to_numpy()
    q = s['q'].to_numpy()
    s1z = s['s1_z'].to_numpy()
    s2z = s['s2_z'].to_numpy()
    dL = s['d_L'].to_numpy()
    iota = s['iota'].to_numpy()
    chi_eff = (s1z + q * s2z) / (1.0 + q)

    our_samples = MCMCSamples(
        np.column_stack([Mc, q, chi_eff, dL, iota]),
        columns=plot_columns,
        weights=np.asarray(s.get_weights()),
    )
    datasets.append((our_samples, 'IMRPhenomD (this work)', COLORS['imr_baseline']))
else:
    print(f"  WARNING: {OUR_CSV} not found — plotting GWTC-2p1 only")

# --------------------------------------------------------------------------- #
# Corner plot
# --------------------------------------------------------------------------- #
make_corner(datasets, plot_columns, 'corner_GW150914', figsize=(12, 12))
print("\nDone.")
