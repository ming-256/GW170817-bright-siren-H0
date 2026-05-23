"""
Compare IMRPhenomD_NRTidalv2 vs TaylorF2 (local, GWTC PSD only).

Plots all four combinations (marginalised / non-marginalised) x (IMRPhenomD / TaylorF2)
on a single corner plot and a single H_0 posterior, with LVK (GWTC-1) as reference.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import shutil
from anesthetic import MCMCSamples, read_chains, make_2d_axes, make_1d_axes
from scipy.stats import gaussian_kde

# --------------------------------------------------------------------------- #
# Rendering setup
# --------------------------------------------------------------------------- #
if shutil.which('pdflatex') or shutil.which('latex'):
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern']
else:
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['mathtext.fontset'] = 'cm'

# --------------------------------------------------------------------------- #
# Files to plot: (csv_path, label, color)
# --------------------------------------------------------------------------- #
RESULTS_DIR = 'Results'
RUNS = [
    (
        'NoMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1.csv',
        'IMRPhenomD — NoMarg',
        'purple',
    ),
    (
        'PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1.csv',
        'IMRPhenomD — PhaseMarg',
        'r',
    ),
    (
        'NoMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1.csv',
        'TaylorF2 — NoMarg',
        'g',
    ),
    (
        'PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1.csv',
        'TaylorF2 — PhaseMarg',
        'darkorange',
    ),
]

# --------------------------------------------------------------------------- #
# 1. Load LVK (GWTC-1) reference samples
# --------------------------------------------------------------------------- #
hdf5_path = os.path.join(RESULTS_DIR, 'GW170817_GWTC-1.hdf5')
dataset_name = 'IMRPhenomPv2NRT_lowSpin_posterior'

with h5py.File(hdf5_path, 'r') as f:
    data = f[dataset_name][:]

m1 = data['m1_detector_frame_Msun']
m2 = data['m2_detector_frame_Msun']
M_c_lvk = (m1 * m2)**0.6 / (m1 + m2)**0.2
q_lvk   = m2 / m1
d_L_lvk = data['luminosity_distance_Mpc']
iota_lvk = np.arccos(data['costheta_jn'])
s1_z_lvk = data['spin1'] * data['costilt1']
s2_z_lvk = data['spin2'] * data['costilt2']

lvk_columns = ['M_c', 'q', 's1_z', 's2_z', 'd_L', 'iota']
LVK_samples = MCMCSamples(
    np.column_stack([M_c_lvk, q_lvk, s1_z_lvk, s2_z_lvk, d_L_lvk, iota_lvk]),
    columns=lvk_columns,
)

# --------------------------------------------------------------------------- #
# 2. Load run samples
# --------------------------------------------------------------------------- #
samples_list = []
for csv_name, label, color in RUNS:
    csv_path = os.path.join(RESULTS_DIR, csv_name)
    if not os.path.exists(csv_path):
        print(f"  WARNING: {csv_path} not found – skipping")
        continue
    s = read_chains(csv_path)
    samples_list.append((s, label, color))
    print(f"  Loaded {csv_name}  ({len(s)} samples)")

if not samples_list:
    raise SystemExit("No data files found.")

os.makedirs('Plots/Results', exist_ok=True)

# --------------------------------------------------------------------------- #
# 3. Corner plot: M_c, q, d_L, iota
# --------------------------------------------------------------------------- #
plot_params = ['M_c', 'q', 'd_L', 'iota']

fig, axes = make_2d_axes(params=plot_params, upper=False, figsize=(10, 10))

LVK_samples.plot_2d(
    axes,
    kinds=dict(diagonal='kde_1d', lower='kde_2d'),
    lower_kwargs=dict(levels=[0.99730, 0.95450, 0.68269]),
    color='b', alpha=0.65, label='LVK (GWTC-1)',
)

for samples, label, color in samples_list:
    samples.plot_2d(
        axes,
        kinds=dict(diagonal='kde_1d', lower='kde_2d'),
        lower_kwargs=dict(levels=[0.99730, 0.95450, 0.68269]),
        color=color, alpha=0.65, label=label,
    )

axes['M_c']['M_c'].set_xlabel(r'$M_c^{\mathrm{det}}$', labelpad=10)
axes['M_c']['M_c'].set_ylabel(r'$M_c^{\mathrm{det}}$')
axes['d_L']['d_L'].set_xlabel('$d_L$ (Mpc)', labelpad=10)
axes['d_L']['d_L'].set_ylabel(r'$P(d_L)$ (Mpc$^{-1}$)')

axes.iloc[-1, 0].legend(
    bbox_to_anchor=(len(axes) * 0.85, len(axes) * 0.8),
    loc='lower center', fontsize=8,
)
fig.suptitle('IMRPhenomD vs TaylorF2 — GWTC PSD (local)', fontsize=14, y=1.01)
fig.tight_layout()
axes.tick_params(grid_alpha=0)

corner_path = 'Plots/Results/corner_waveform_comparison_gwtc'
plt.savefig(f'{corner_path}.pdf', bbox_inches='tight')
plt.savefig(f'{corner_path}.png', dpi=150, bbox_inches='tight')
print(f"  -> Saved {corner_path}.pdf / .png")
plt.close(fig)

# --------------------------------------------------------------------------- #
# 4. H_0 1D posterior
# --------------------------------------------------------------------------- #
h0_runs = [(s, l, c) for s, l, c in samples_list
           if 'H_0' in s.columns.get_level_values(0)]

if not h0_runs:
    print("  (no H_0 column found – skipping H_0 plot)")
else:
    fig2, axes2 = make_1d_axes(params='H_0', figsize=(6, 6))

    for samples, label, color in h0_runs:
        samples.plot_1d(axes2, kind='kde_1d', color=color, alpha=0.8, label=label)

        h0_vals = samples['H_0'].to_numpy()
        weights = np.asarray(samples.get_weights())
        weights = weights / weights.sum()
        kde = gaussian_kde(h0_vals, weights=weights)
        x_eval = np.linspace(h0_vals.min() - 1, h0_vals.max() + 1, 10000)
        map_value = x_eval[np.argmax(kde(x_eval))]
        print(f"  {label}: H_0 MAP = {map_value:.1f} km/s/Mpc")

    ax = axes2['H_0']
    ax.axvspan(66.93 - 0.62, 66.93 + 0.62,
               edgecolor='none', alpha=0.3, color='#0CDE79', label='Planck')
    ax.axvspan(73.24 - 1.74, 73.24 + 1.74,
               edgecolor='none', alpha=0.3, color='#E87317', label='SHoES')
    ax.set_ylabel(r'$P(H_0)$ (km s$^{-1}$ Mpc$^{-1}$)')
    ax.set_xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)')
    ax.set_xlim(20, 140)
    ax.legend(fontsize=8)
    axes2.tick_params(grid_alpha=0)
    fig2.suptitle('IMRPhenomD vs TaylorF2 — GWTC PSD (local)', fontsize=14)
    fig2.tight_layout()

    h0_path = 'Plots/Results/H0_waveform_comparison_gwtc'
    plt.savefig(f'{h0_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{h0_path}.png', dpi=150, bbox_inches='tight')
    print(f"  -> Saved {h0_path}.pdf / .png")
    plt.close(fig2)

print("\nDone.")
