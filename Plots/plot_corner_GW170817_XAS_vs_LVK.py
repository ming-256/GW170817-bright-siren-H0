"""
GW170817 full-parameter corner: this work's IMRPhenomXAS_NRTidalv3 vs LVK GWTC-1.

Overlays seven parameters common to both posteriors:
  M_c, q, chi_eff, d_L, iota, Lambda_tilde, ra, dec

Output: Results/gwtc1_phasemarg/plots/corner_GW170817_XAS_vs_LVK.{pdf,png}
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *
import numpy as np
import h5py

PLOT_COLS = [r'$\mathcal{M}_c$', r'$q$', r'$\chi_{\rm eff}$',
             r'$d_L$', r'$\iota$', r'$\tilde\Lambda$']

def lambda_tilde(q, l1, l2):
    """Favata 2014 effective tidal deformability (m1 ≥ m2 convention; q = m2/m1 ≤ 1)."""
    m1, m2 = 1.0, q
    return (16.0/13.0) * (
        (m1 + 12*m2) * m1**4 * l1 + (m2 + 12*m1) * m2**4 * l2
    ) / (m1 + m2)**5

# --- this work: XAS s07 baseline_lvkbounds ---
xas_csv = 'Results/test_suite/s07__gw170817__imrphenomxas_nrtidalv3__baseline_lvkbounds__seed0000/samples.csv'
s = load_nested_csv(xas_csv)
Mc = s['M_c'].to_numpy(); q = s['q'].to_numpy()
s1z = s['s1_z'].to_numpy(); s2z = s['s2_z'].to_numpy()
chi_eff = (s1z + q * s2z) / (1.0 + q)
dL = s['d_L'].to_numpy(); iota = s['iota'].to_numpy()
l1 = s['lambda_1'].to_numpy(); l2 = s['lambda_2'].to_numpy()
lt = lambda_tilde(q, l1, l2)
xas_arr = np.column_stack([Mc, q, chi_eff, dL, iota, lt])
xas = MCMCSamples(xas_arr, columns=PLOT_COLS,
                  weights=np.asarray(s.get_weights()))

# --- LVK GWTC-1 IMRPhenomPv2_NRTidal lowSpin ---
with h5py.File('Results/GW170817_GWTC-1.hdf5', 'r') as f:
    d = f['IMRPhenomPv2NRT_lowSpin_posterior'][:]
m1L = d['m1_detector_frame_Msun']; m2L = d['m2_detector_frame_Msun']
McL = (m1L * m2L)**0.6 / (m1L + m2L)**0.2
qL = m2L / m1L
s1zL = d['spin1'] * d['costilt1']; s2zL = d['spin2'] * d['costilt2']
chi_effL = (s1zL + qL * s2zL) / (1.0 + qL)
dLL = d['luminosity_distance_Mpc']
iotaL = np.arccos(d['costheta_jn'])
ltL = lambda_tilde(qL, d['lambda1'], d['lambda2'])
lvk_arr = np.column_stack([McL, qL, chi_effL, dLL, iotaL, ltL])
lvk = MCMCSamples(lvk_arr, columns=PLOT_COLS)

datasets = [
    (lvk, 'LVK GWTC-1 (PhenomPv2\\_NRTidal)', COLORS['gwtc']),
    (xas, 'IMRPhenomXAS\\_NRTidalv3 (this work)', COLORS['flatZ']),
]

make_corner(datasets, PLOT_COLS, 'corner_GW170817_XAS_vs_LVK',
            figsize=(15, 15))
print("\nDone.")
