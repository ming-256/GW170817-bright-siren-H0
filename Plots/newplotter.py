"""
Comparison plot: JAX nested sampling vs LVK (GWTC-1) posteriors for GW170817.

Auto-discovers all Results/*.csv files and overlays them against LVK samples.
Produces:
  - Corner plot (M_c, q, d_L, iota) for each result vs LVK
  - H_0 1D posterior overlay for all results
"""

import warnings
warnings.filterwarnings('ignore')

import glob
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import shutil
from anesthetic import NestedSamples, MCMCSamples, read_chains, make_2d_axes, make_1d_axes

# LaTeX rendering if available, otherwise mathtext fallback
if shutil.which('pdflatex') or shutil.which('latex'):
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern']
else:
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['mathtext.fontset'] = 'cm'

COLORS = ['purple', 'r', 'g', 'darkorange', 'teal', 'brown', 'olive', 'm']

# ============================================================================
# 1. Load LVK (GWTC-1) samples
# ============================================================================
hdf5_path = 'Results/GW170817_GWTC-1.hdf5'
dataset_name = 'IMRPhenomPv2NRT_lowSpin_posterior'

with h5py.File(hdf5_path, 'r') as f:
    data = f[dataset_name][:]

# Derive parameters in the same convention as the inference script
m1 = data['m1_detector_frame_Msun']
m2 = data['m2_detector_frame_Msun']
M_c_lvk = (m1 * m2)**0.6 / (m1 + m2)**0.2
q_lvk = m2 / m1
d_L_lvk = data['luminosity_distance_Mpc']
iota_lvk = np.arccos(data['costheta_jn'])
s1_z_lvk = data['spin1'] * data['costilt1']
s2_z_lvk = data['spin2'] * data['costilt2']

lvk_columns = ['M_c', 'q', 's1_z', 's2_z', 'd_L', 'iota']
LVK_samples = MCMCSamples(
    np.column_stack([M_c_lvk, q_lvk, s1_z_lvk, s2_z_lvk, d_L_lvk, iota_lvk]),
    columns=lvk_columns,
)

# ============================================================================
# 2. Auto-discover all result CSVs
# ============================================================================
csv_files = sorted(glob.glob('Results/*.csv'))
if not csv_files:
    print("No CSV files found in Results/")
    exit()

print(f"Found {len(csv_files)} result file(s):")
for f in csv_files:
    print(f"  {f}")

all_samples = []
for csv_path in csv_files:
    try:
        s = read_chains(csv_path)
        label = os.path.splitext(os.path.basename(csv_path))[0]
        all_samples.append((s, label))
    except Exception as e:
        print(f"Skipping {csv_path}: {e}")

# ============================================================================
# 3. Corner plot: M_c, q, d_L, iota (all results vs LVK)
# ============================================================================
plot_params = ['M_c', 'q', 'd_L', 'iota']

fig, axes = make_2d_axes(params=plot_params, upper=False, figsize=(10, 10))

LVK_samples.plot_2d(
    axes,
    kinds=dict(diagonal='hist_1d', lower='hist_2d'),
    lower_kwargs=dict(levels=[0.99730, 0.95450, 0.68269]),
    color='b', alpha=0.65, label='LVK (GWTC-1)',
)

for i, (samples, label) in enumerate(all_samples):
    color = COLORS[i % len(COLORS)]
    samples.plot_2d(
        axes,
        kinds=dict(diagonal='hist_1d', lower='hist_2d'),
        lower_kwargs=dict(levels=[0.99730, 0.95450, 0.68269]),
        color=color, alpha=0.65, label=label,
    )

# Axis labels
axes['M_c']['M_c'].set_xlabel(r'$M_c^{\mathrm{det}}$', labelpad=10)
axes['M_c']['M_c'].set_ylabel(r'$M_c^{\mathrm{det}}$')
axes['d_L']['d_L'].set_xlabel('$d_L$ (Mpc)', labelpad=10)
axes['d_L']['d_L'].set_ylabel(r'$P(d_L)$ (Mpc$^{-1}$)')

axes.iloc[-1, 0].legend(
    bbox_to_anchor=(len(axes) * 0.85, len(axes) * 0.8), loc='lower center'
)
fig.tight_layout()
axes.tick_params(grid_alpha=0)

corner_label = 'Plots/Results/JAX_vs_LVK_GW170817'
plt.savefig(f'{corner_label}.pdf')
plt.savefig(f'{corner_label}.png', dpi=150)
print(f"Saved corner plot to {corner_label}.pdf and {corner_label}.png")

# ============================================================================
# 4. H_0 1D posterior (all results that have H_0)
# ============================================================================
from scipy.stats import gaussian_kde

h0_runs = [(s, l) for s, l in all_samples if 'H_0' in s.columns.get_level_values(0)]

if h0_runs:
    fig2, axes2 = make_1d_axes(params='H_0', figsize=(6, 6))

    for i, (samples, label) in enumerate(h0_runs):
        color = COLORS[i % len(COLORS)]
        samples.plot_1d(axes2, kind='kde_1d', color=color, alpha=0.8, label=label)

        h0_vals = samples['H_0'].to_numpy()
        weights = np.asarray(samples.get_weights())
        weights = weights / weights.sum()
        kde = gaussian_kde(h0_vals, weights=weights)
        x_eval = np.linspace(h0_vals.min() - 1, h0_vals.max() + 1, 10000)
        map_value = x_eval[np.argmax(kde(x_eval))]
        print(f"{label}: H_0 MAP = {map_value:.1f} km/s/Mpc")

    # Planck & SHoES reference bands (draw vertical span on the H_0 axis)
    ax = axes2['H_0']
    ax.axvspan(66.93 - 0.62, 66.93 + 0.62, edgecolor='none', alpha=0.3, color='#0CDE79', label='Planck')
    ax.axvspan(73.24 - 1.74, 73.24 + 1.74, edgecolor='none', alpha=0.3, color='#E87317', label='SHoES')
    ax.set_ylabel(r'$P(H_0)$ (km s$^{-1}$ Mpc$^{-1}$)')
    ax.set_xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)')
    ax.set_xlim(20, 140)
    ax.legend()
    axes2.tick_params(grid_alpha=0)
    fig2.tight_layout()

    h0_label = 'Plots/Results/H0_posterior_GW170817'
    plt.savefig(f'{h0_label}.pdf')
    plt.savefig(f'{h0_label}.png', dpi=150)
    print(f"Saved H_0 plot to {h0_label}.pdf and {h0_label}.png")

plt.show()
