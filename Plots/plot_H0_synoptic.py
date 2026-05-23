"""
Synoptic H_0 forest plot for GW170817 — sample-derived HPD intervals.

Each row is a separate H_0 measurement plotted as MAP + 68%/95% HPD.
HPD intervals are computed *directly from the weighted samples* (no KDE),
matching the user's preference for histogram-honest summaries.

LVK Abbott+2017 enters as a real row using the GWTC-1 IMRPhenomPv2_NRTidal
posterior samples mapped through this work's standard-siren model — *not*
as a shaded reference band. This is the apples-to-apples comparison the
paper needs.

IMRPhenomPv2 (no tides) is dropped from the main figure; it remains in the
test_suite for the open-source reproducibility appendix.

Output: Results/gwtc1_phasemarg/plots/H0_synoptic.{pdf,png}
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import (
    OUT_DIR, RESULTS_DIR, COLORS, load_nested_csv,
    load_gwtc1_gw170817, derive_lvk_h0_samples,
    compute_hpd_samples, map_from_hist,
)
import numpy as np
import matplotlib.pyplot as plt

BIN_EDGES = np.linspace(40, 230, 191)   # 1 km/s/Mpc bins for MAP estimation


def sample_summary(h0, w):
    map_ = map_from_hist(h0, w, BIN_EDGES)
    lo68, hi68 = compute_hpd_samples(h0, w, 0.68269)
    lo95, hi95 = compute_hpd_samples(h0, w, 0.95450)
    return map_, lo68, hi68, lo95, hi95


def siren_summary(samples_csv):
    s = load_nested_csv(samples_csv)
    x = s['H_0'].to_numpy().astype(float)
    w = np.asarray(s.get_weights(), dtype=float); w /= w.sum()
    return sample_summary(x, w)


def gw_only_summary(samples_csv, v_rec=3017.0):
    """Standard-siren H_0 = v_rec / d_L on a uniform-in-d_L sample (s12 GW-only runs)."""
    s = load_nested_csv(samples_csv)
    dL = s['d_L'].to_numpy().astype(float)
    w = np.asarray(s.get_weights(), dtype=float); w /= w.sum()
    h0 = v_rec / dL
    return sample_summary(h0, w)


def lvk_summary():
    lvk = load_gwtc1_gw170817(columns=['d_L'])
    h0 = derive_lvk_h0_samples(lvk['d_L'].to_numpy(), rng=np.random.default_rng(170817))
    w = np.ones_like(h0) / len(h0)
    return sample_summary(h0, w)


# (label, csv path, color, marker, kind)
ENTRIES = [
    (r'IMRPhenomXAS\_NRTidalv3 (siren, primary)',
     'Results/test_suite/s07__gw170817__imrphenomxas_nrtidalv3__baseline_lvkbounds__seed0000/samples.csv',
     COLORS['flatZ'], 's', 'siren'),
    (r'IMRPhenomD\_NRTidalv2 (siren, anchor)',
     'Results/test_suite/s07__gw170817__imrphenomd_nrtidalv2__baseline_lvkbounds__seed0000/samples.csv',
     COLORS['imr_baseline'], 'o', 'siren'),
    (r'TaylorF2 (siren, host-loc)',
     os.path.join(RESULTS_DIR,
                  'gwtc1_phasemarg/PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_baseline.csv'),
     COLORS['tf2_baseline'], 'D', 'siren'),
    (r'IMRPhenomXAS\_NRTidalv3 (GW-only)',
     'Results/test_suite/s12__gw170817__imrphenomxas_nrtidalv3__gw_only__seed0000/samples.csv',
     COLORS['flatZ'], 's', 'gw_only'),
    (r'IMRPhenomD\_NRTidalv2 (GW-only)',
     'Results/test_suite/s12__gw170817__imrphenomd_nrtidalv2__gw_only__seed0000/samples.csv',
     COLORS['imr_baseline'], 'o', 'gw_only'),
    (r'LVK GWTC-1 (Abbott+2017, this work\'s $v_p$ model)',
     None, '0.25', '*', 'lvk'),
]

rows = []
for lab, csv, col, mk, kind in ENTRIES:
    if kind == 'lvk':
        m, lo68, hi68, lo95, hi95 = lvk_summary()
    elif kind == 'siren':
        if not os.path.exists(csv):
            print(f"  WARNING: missing {csv}")
            continue
        m, lo68, hi68, lo95, hi95 = siren_summary(csv)
    else:
        if not os.path.exists(csv):
            print(f"  WARNING: missing {csv}")
            continue
        m, lo68, hi68, lo95, hi95 = gw_only_summary(csv)
    print(f"  {lab}: MAP={m:.1f}, 68% HPD=[{lo68:.1f},{hi68:.1f}], 95% HPD=[{lo95:.1f},{hi95:.1f}]")
    rows.append((lab, m, lo68, hi68, lo95, hi95, col, mk))

# Reverse so first entry sits at the top of the forest.
rows = rows[::-1]

fig, ax = plt.subplots(figsize=(11, 6))

# Reference cosmology bands (kept — these are population-level priors, not GW170817 measurements).
ax.axvspan(66.93 - 0.62, 66.93 + 0.62,
           color=COLORS['planck_inner'], alpha=0.3, zorder=0)
ax.axvspan(73.24 - 1.74, 73.24 + 1.74,
           color=COLORS['shoes_inner'], alpha=0.3, zorder=0)

for i, (lab, m, lo68, hi68, lo95, hi95, col, mk) in enumerate(rows):
    ax.hlines(i, lo95, hi95, color=col, lw=1.2, alpha=0.5, zorder=4)   # 95% HPD
    ax.hlines(i, lo68, hi68, color=col, lw=3.0, zorder=5)               # 68% HPD
    ax.scatter([m], [i], color=col, marker=mk, s=70, zorder=6,
               edgecolors='black', linewidths=0.6)
    ax.text(232, i, lab, ha='right', va='center', fontsize=10)

# Custom legend — note no LVK shaded band; LVK is a row.
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles = [
    Patch(color=COLORS['planck_inner'], alpha=0.4, label=r'Planck CMB ($H_0 = 66.93\pm0.62$)'),
    Patch(color=COLORS['shoes_inner'], alpha=0.4, label=r'SH0ES ($H_0 = 73.24\pm1.74$)'),
    Line2D([], [], color='k', lw=3.0, label=r'68\% HPD (sample-derived)'),
    Line2D([], [], color='k', lw=1.2, alpha=0.6, label=r'95\% HPD (sample-derived)'),
]
ax.legend(handles=handles, fontsize=10, loc='upper left',
          bbox_to_anchor=(0.005, 0.99), frameon=False)

ax.set_xlim(40, 235)
ax.set_yticks([]); ax.set_ylim(-0.7, len(rows) - 0.3)
ax.set_xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=13)
for spine in ax.spines.values():
    spine.set_edgecolor('black'); spine.set_linewidth(1.5)

fig.tight_layout()
p = os.path.join(OUT_DIR, 'H0_synoptic')
plt.savefig(f'{p}.pdf', bbox_inches='tight')
plt.savefig(f'{p}.png', dpi=150, bbox_inches='tight')
print(f"  -> Saved {p}.pdf / .png")
plt.close(fig)
