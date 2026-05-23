"""
H_0 posterior comparison plot for GW170817.

Auto-discovers all Results/*.csv files and creates two comparison plots:
  1. TaylorF2 runs (all data/PSD/phase-marg variants)
  2. IMRPhenomD_NRTidalv2 runs (all data/PSD/phase-marg variants)

This allows comparing which data source / PSD / phase-marginalization
combination gives the best H_0 constraint within each waveform family.
"""

import warnings
warnings.filterwarnings('ignore')

import glob
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import shutil
from scipy.stats import gaussian_kde
from anesthetic import read_chains

# LaTeX rendering
if shutil.which('pdflatex') or shutil.which('latex'):
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern']
else:
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['mathtext.fontset'] = 'cm'


# ============================================================================
# Configuration
# ============================================================================
H0_MIN = 20
H0_MAX = 140

COLORS = ['b', 'r', 'g', 'm', 'darkorange', 'teal', 'brown', 'olive',
          'navy', 'crimson', 'darkgreen', 'indigo', 'sienna', 'darkslategray']

# Auto-discover all result CSVs
csv_files = sorted(glob.glob('Results/*.csv'))
if not csv_files:
    print("No CSV files found in Results/")
    exit()

print(f"Found {len(csv_files)} result file(s):")
for f in csv_files:
    print(f"  {f}")

# Split CSVs by waveform family
waveform_groups = {
    'TaylorF2': [f for f in csv_files if 'TaylorF2' in os.path.basename(f)],
    'IMRPhenomD_NRTidalv2': [f for f in csv_files if 'IMRPhenomD' in os.path.basename(f)],
}

# Catch any that don't match either pattern
unmatched = [f for f in csv_files
             if 'TaylorF2' not in os.path.basename(f) and 'IMRPhenomD' not in os.path.basename(f)]
if unmatched:
    waveform_groups['Other'] = unmatched


# ============================================================================
# Plotting helper
# ============================================================================
def plot_h0_comparison(csv_list, title, out_label):
    """Plot H_0 posteriors for a list of CSV result files."""
    if not csv_list:
        print(f"No files for {title}, skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0

    for csv_path in csv_list:
        try:
            samples = read_chains(csv_path)
        except Exception as e:
            print(f"Skipping {csv_path}: {e}")
            continue

        if 'H_0' not in samples.columns.get_level_values(0):
            print(f"Skipping {csv_path}: no H_0 column")
            continue

        label = os.path.splitext(os.path.basename(csv_path))[0]
        color = COLORS[plotted % len(COLORS)]

        h0 = samples['H_0'].to_numpy()
        weights = np.asarray(samples.get_weights())

        mask = (h0 >= H0_MIN) & (h0 <= H0_MAX)
        h0 = h0[mask]
        weights = weights[mask]
        if len(h0) < 10:
            print(f"Skipping {csv_path}: too few samples in H_0 range")
            continue
        weights = weights / weights.sum()

        kde = gaussian_kde(h0, weights=weights)
        x_eval = np.linspace(H0_MIN, H0_MAX, 1000)
        pdf = kde(x_eval)

        map_val = x_eval[np.argmax(pdf)]
        print(f"  {label}: H_0 MAP = {map_val:.1f} km/s/Mpc, median = {np.median(h0):.1f}")

        ax.plot(x_eval, pdf, color=color, lw=2, label=label)
        plotted += 1

    if plotted == 0:
        print(f"No valid results for {title}.")
        plt.close(fig)
        return

    # Planck & SHoES reference bands
    ax.axvspan(66.93 - 0.62, 66.93 + 0.62, alpha=0.3, color='#0CDE79', edgecolor='none', label='Planck')
    ax.axvspan(73.24 - 1.74, 73.24 + 1.74, alpha=0.3, color='#E87317', edgecolor='none', label='SHoES')

    ax.set_xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=14)
    ax.set_ylabel(r'$P(H_0)$', fontsize=14)
    ax.set_xlim(20, 250)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=15)
    ax.legend(fontsize=12, loc='upper right')
    ax.tick_params(labelsize=12)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_label), exist_ok=True)
    plt.savefig(f'{out_label}.pdf')
    plt.savefig(f'{out_label}.png', dpi=150)
    print(f"Saved to {out_label}.pdf and {out_label}.png")
    plt.close(fig)


# ============================================================================
# Generate plots
# ============================================================================
for waveform_name, csv_list in waveform_groups.items():
    if not csv_list:
        continue
    safe_name = waveform_name.replace(' ', '_')
    plot_h0_comparison(
        csv_list,
        title=f'$H_0$ Posterior Comparison: {waveform_name}',
        out_label=f'Plots/Results/H0_{safe_name}',
    )

print("\nDone.")
