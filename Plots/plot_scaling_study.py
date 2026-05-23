"""
Scaling study: runtime vs number of live points.

Creates a multi-panel figure showing:
  1. Total sampling time vs n_live (with linear-scaling reference)
  2. Sampling time per 1000 live points vs n_live (identifies saturation)
  3. Dead points / second vs n_live (throughput metric)

Can use either:
  - Results/scaling_study/scaling_summary.csv  (from run_scaling_study.sh)
  - Hardcoded data from existing A100 and L4 runs

Also overlays Bilby/pBilby reference points when available.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ----------------------------------------------------------------------- #
# Data sources
# ----------------------------------------------------------------------- #

SCALING_CSV = os.path.join(RESULTS_DIR, 'scaling_study', 'scaling_summary.csv')

# Hardcoded reference data from existing runs
# (used if scaling_summary.csv doesn't exist or to augment it)

# A100 runs (heterodyned, IMRPhenomD_NRTidalv2, from a100_run_data.md)
A100_IMR = {
    'n_live': np.array([5000]),
    'sampling_s': np.array([772.6]),
    'total_s': np.array([884.6]),
    'dead_pts': np.array([198000]),
    'label': 'A100 (IMRPhenomD\_NRTidalv2)',
}

A100_TF2 = {
    'n_live': np.array([5000]),
    'sampling_s': np.array([161.6]),
    'total_s': np.array([235.3]),
    'dead_pts': np.array([199500]),
    'label': 'A100 (TaylorF2)',
}

# L4 runs (from thesis Table 5, heterodyned, IMRPhenomD_NRTidalv2, Model A)
L4_IMR = {
    'n_live': np.array([2500, 7500, 10000, 15000]),
    'sampling_s': np.array([504, 1355, 1886, 2729]),  # estimated from runtime/K
    'total_s': np.array([504, 1358, 1886, 2729]),
    'dead_pts': np.array([np.nan, np.nan, np.nan, np.nan]),
    'label': 'L4 (IMRPhenomD\_NRTidalv2, thesis)',
}

# Bilby reference (CSD3, standard bilby)
BILBY_REF = {
    'n_live': np.array([1216, 1596]),
    'total_s': np.array([4*3600 + 37*60 + 32, 3*3600 + 49*60 + 17]),
    'cores': np.array([456, 532]),
    'label': 'Bilby (CSD3, standard)',
}


def load_scaling_csv():
    """Load scaling study results if available."""
    if not os.path.exists(SCALING_CSV):
        return None
    df = pd.read_csv(SCALING_CSV)
    return df


def plot_scaling_study():
    """Main scaling study figure."""
    df = load_scaling_csv()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --------------------------------------------------------------- #
    # Collect all GPU data points
    # --------------------------------------------------------------- #
    gpu_sets = []

    if df is not None and len(df) > 0:
        # Group by waveform
        for wf, grp in df.groupby('waveform'):
            grp = grp.sort_values('n_live')
            n = grp['n_live'].to_numpy()
            samp = grp['sampling_s'].to_numpy().astype(float)
            total = grp['total_s'].to_numpy().astype(float)
            dead = grp['dead_points'].to_numpy().astype(float)
            lbl = f'A100 ({wf}, scaling study)'
            color = COLORS['imr_baseline'] if 'IMRPhen' in wf else COLORS['tf2_baseline']
            gpu_sets.append({
                'n_live': n, 'sampling_s': samp, 'total_s': total,
                'dead_pts': dead, 'label': lbl, 'color': color,
                'marker': 'o', 'ms': 8,
            })
    else:
        # Fall back to hardcoded data
        gpu_sets.append({**A100_IMR, 'color': COLORS['imr_baseline'],
                         'marker': 'D', 'ms': 10})
        gpu_sets.append({**A100_TF2, 'color': COLORS['tf2_baseline'],
                         'marker': 'D', 'ms': 10})

    # Always add L4 reference
    gpu_sets.append({**L4_IMR, 'color': '#cc6600', 'marker': 's', 'ms': 7})

    # --------------------------------------------------------------- #
    # Panel 1: Total sampling time vs n_live
    # --------------------------------------------------------------- #
    ax = axes[0]
    for d in gpu_sets:
        ax.plot(d['n_live'], d['sampling_s'], marker=d['marker'],
                ms=d['ms'], color=d['color'], label=d['label'],
                lw=2, zorder=5)

    # Linear reference line (through L4 data where scaling is known)
    if len(L4_IMR['n_live']) > 1:
        n_ref = np.linspace(500, 20000, 100)
        # Use rate from L4 linear regime (~3 min per 1K = 180s/K)
        rate = 180.0  # seconds per 1000 live points
        ax.plot(n_ref, rate * n_ref / 1000, 'k--', alpha=0.4, lw=1.5,
                label=f'Linear ref ({rate:.0f}s / 1K)')

    ax.set_xlabel(r'$n_{\rm live}$')
    ax.set_ylabel('Sampling time (s)')
    ax.set_title('Runtime scaling')
    ax.legend(frameon=False, fontsize=8, loc='upper left')

    # --------------------------------------------------------------- #
    # Panel 2: Time per 1000 live points
    # --------------------------------------------------------------- #
    ax = axes[1]
    for d in gpu_sets:
        per_k = d['sampling_s'] / (d['n_live'] / 1000)
        ax.plot(d['n_live'], per_k, marker=d['marker'],
                ms=d['ms'], color=d['color'], label=d['label'], lw=2, zorder=5)

    ax.axhline(180, color='k', ls='--', alpha=0.4, lw=1.5,
               label='Linear regime (180s/K)')
    ax.set_xlabel(r'$n_{\rm live}$')
    ax.set_ylabel('Sampling time per 1K live points (s)')
    ax.set_title('Per-live-point efficiency')
    ax.legend(frameon=False, fontsize=8, loc='upper right')

    # Shade the saturation region
    ax.axvspan(0, 2000, color='grey', alpha=0.08, zorder=0)
    ax.text(1000, ax.get_ylim()[1] * 0.95, 'GPU\nsaturation',
            ha='center', va='top', fontsize=9, color='grey', style='italic')

    # --------------------------------------------------------------- #
    # Panel 3: Throughput (dead points / second)
    # --------------------------------------------------------------- #
    ax = axes[2]
    for d in gpu_sets:
        if not np.all(np.isnan(d['dead_pts'])):
            mask = ~np.isnan(d['dead_pts'])
            throughput = d['dead_pts'][mask] / d['sampling_s'][mask]
            ax.plot(d['n_live'][mask], throughput, marker=d['marker'],
                    ms=d['ms'], color=d['color'], label=d['label'], lw=2, zorder=5)

    # Add Bilby reference as horizontal band
    bilby_throughput = []
    for i in range(len(BILBY_REF['n_live'])):
        # Rough estimate: assume ~30 dead points per live point for BNS
        est_dead = BILBY_REF['n_live'][i] * 30
        t = est_dead / BILBY_REF['total_s'][i]
        bilby_throughput.append(t)
    if bilby_throughput:
        mean_bt = np.mean(bilby_throughput)
        ax.axhline(mean_bt, color='tab:green', ls=':', lw=2, alpha=0.8,
                   label=f'Bilby CSD3 ~{mean_bt:.1f} dead/s\n({int(np.mean(BILBY_REF["cores"]))} cores)')

    ax.set_xlabel(r'$n_{\rm live}$')
    ax.set_ylabel('Dead points / second')
    ax.set_title('Sampling throughput')
    ax.legend(frameon=False, fontsize=8, loc='upper left')

    # --------------------------------------------------------------- #
    # Global formatting
    # --------------------------------------------------------------- #
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'scaling_study')
    plt.savefig(f'{path}.pdf', bbox_inches='tight')
    plt.savefig(f'{path}.png', dpi=150, bbox_inches='tight')
    print(f"  -> Saved {path}.pdf / .png")
    plt.close(fig)


def plot_gpu_vs_cpu_projection():
    """
    Projected runtime comparison: GPU (single A100) vs CPU cluster.

    Shows the regime where GPU becomes unambiguously faster.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    n_live = np.array([500, 1000, 2500, 5000, 10000, 15000, 20000])

    # GPU projection: linear scaling from A100 IMRPhenomD_NRTidalv2 data
    # 5000 live points -> 772.6s sampling + ~112s overhead
    gpu_rate_imr = 772.6 / 5000  # s per live point
    gpu_overhead = 112  # s (data load + het setup + init + JIT)
    gpu_total_imr = gpu_rate_imr * n_live + gpu_overhead

    gpu_rate_tf2 = 161.6 / 5000
    gpu_total_tf2 = gpu_rate_tf2 * n_live + gpu_overhead * 0.5  # TF2 has less JIT overhead

    # CPU reference: Bilby CSD3 (scale from 1596 pts / 532 cores / 3h49m)
    # At 532 cores: 13757s for 1596 live points
    # CPU scaling: roughly linear in n_live, but core count is fixed
    bilby_rate = 13757 / 1596  # s per live point at 532 cores
    cpu_total_532 = bilby_rate * n_live

    # At higher n_live, CPU would need more cores or more time
    # Show 532-core projection
    ax.plot(n_live, gpu_total_imr / 60, 'o-', color=COLORS['imr_baseline'],
            lw=2.5, ms=7, label='A100 (IMRPhenomD\_NRTidalv2)', zorder=5)
    ax.plot(n_live, gpu_total_tf2 / 60, 's-', color=COLORS['tf2_baseline'],
            lw=2.5, ms=7, label='A100 (TaylorF2)', zorder=5)
    ax.plot(n_live, cpu_total_532 / 60, 'D--', color='tab:green',
            lw=2, ms=7, label='Bilby CSD3 (532 cores, projected)', zorder=4)

    # Mark actual data points
    ax.scatter([5000], [884.6 / 60], color=COLORS['imr_baseline'], s=150,
               zorder=10, edgecolors='black', linewidths=1.5, marker='*',
               label='A100 measured')
    ax.scatter([1216, 1596], [16652 / 60, 13757 / 60], color='tab:green', s=100,
               zorder=10, edgecolors='black', linewidths=1.5, marker='*',
               label='Bilby measured')

    # Annotate the crossover regime
    ax.fill_between([0, 2500], 0, 1500, color='grey', alpha=0.05, zorder=0)
    ax.text(1250, ax.get_ylim()[0] + 5, 'CPU competitive\n(few live points)',
            ha='center', fontsize=9, color='grey', style='italic')

    ax.set_xlabel(r'$n_{\rm live}$', fontsize=13)
    ax.set_ylabel('Wall-clock time (minutes)', fontsize=13)
    ax.set_title('GPU vs CPU: Projected runtime scaling (heterodyned GW170817)')
    ax.set_yscale('log')
    ax.legend(frameon=False, fontsize=10)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'gpu_vs_cpu_projection')
    plt.savefig(f'{path}.pdf', bbox_inches='tight')
    plt.savefig(f'{path}.png', dpi=150, bbox_inches='tight')
    print(f"  -> Saved {path}.pdf / .png")
    plt.close(fig)


if __name__ == '__main__':
    print("=== Scaling Study Plots ===\n")
    plot_scaling_study()
    print()
    plot_gpu_vs_cpu_projection()
    print("\nDone.")
