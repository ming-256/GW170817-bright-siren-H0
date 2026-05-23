"""
GW170817 d_L–iota bimodality figure (existing Plots/ style).

Two panels:
  (a) The unrestricted uniform-in-d_L run shown in (d_L, iota) plane,
      with the prior-restricted Mode-A and Mode-B contours overlaid.
  (b) 1D H_0 marginal under each variant (showing why direct flat-z
      sampling places ~28 per cent of the posterior mass at H_0 > 120,
      which the volumetric prior suppresses).

Output: Results/gwtc1_phasemarg/plots/bimodality.{pdf,png}
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

CSV_A    = 'Results/test_suite/s10__gw170817__imrphenomd_nrtidalv2__flatz__dL30-75__refGWTC1__seed0000/samples.csv'
CSV_B    = 'Results/test_suite/s10__gw170817__imrphenomd_nrtidalv2__flatz__dL10-30__refGWTC1__seed0000/samples.csv'
CSV_FULL = 'Results/test_suite/s10__gw170817__imrphenomd_nrtidalv2__flatz__dL10-75__refModeB__seed0000/samples.csv'
# seed=1 verification (s18)
CSV_A1   = 'Results/test_suite/s18__gw170817__imrphenomd_nrtidalv2__flatz__dL30-75__refGWTC1__seed0001/samples.csv'
CSV_B1   = 'Results/test_suite/s18__gw170817__imrphenomd_nrtidalv2__flatz__dL10-30__refGWTC1__seed0001/samples.csv'
CSV_FULL1= 'Results/test_suite/s18__gw170817__imrphenomd_nrtidalv2__flatz__dL10-75__refModeB__seed0001/samples.csv'

def load_dl_iota_h0(csv):
    s = load_nested_csv(csv)
    return (s['d_L'].to_numpy(), s['iota'].to_numpy(),
            s['H_0'].to_numpy() if 'H_0' in s.columns else None,
            np.asarray(s.get_weights()))

dlA, iA, h0A, wA = load_dl_iota_h0(CSV_A)
dlB, iB, h0B, wB = load_dl_iota_h0(CSV_B)
dlF, iF, h0F, wF = load_dl_iota_h0(CSV_FULL)
_, _, h0A1, wA1 = load_dl_iota_h0(CSV_A1)
_, _, h0B1, wB1 = load_dl_iota_h0(CSV_B1)
_, _, h0F1, wF1 = load_dl_iota_h0(CSV_FULL1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Two distinct colour families for Mode A and Mode B (each is a separate
# prior-restricted run with a different d_L range; the shared `Combined`
# unrestricted run is in a third neutral colour).
MODE_A_COL = '#2C7FB8'    # blue  (Mode A peaks left in H0; avoids SHoES orange)
MODE_B_COL = '#d62728'    # red   (Mode B peaks right in H0)
COMBINED_COL = '#111111'  # black (unrestricted)

# ----- Panel (a): d_L vs iota -----
ax = axes[0]
xi = np.linspace(8, 80, 200)
yi = np.linspace(0, np.pi, 200)
XX, YY = np.meshgrid(xi, yi)
ZA = gaussian_kde(np.vstack([dlA, iA]), weights=wA/wA.sum())(
    np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
ZB = gaussian_kde(np.vstack([dlB, iB]), weights=wB/wB.sum())(
    np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)

ZA_n = ZA / ZA.max()
ZB_n = ZB / ZB.max()

# 5 iso-density levels — Mode B (red) then Mode A (blue), no outline rings
_levels = np.linspace(0.10, 0.90, 5)
ax.contourf(XX, YY, ZB_n, levels=_levels, cmap='Reds',  alpha=0.70, extend='max')
ax.contourf(XX, YY, ZA_n, levels=_levels, cmap='Blues', alpha=0.70, extend='max')
# Mode boundary at d_L = 30 Mpc (the dividing line of the two restricted
# priors): make explicit that A and B come from runs with different priors.
ax.axvline(30, color='black', ls=':', lw=1.0, alpha=0.7)
ax.text(30.5, 3.05, r'$d_L=30\,\rm Mpc$ (prior boundary)',
        rotation=90, ha='left', va='top', fontsize=9, color='0.3')
ax.set_xlim(10, 50); ax.set_ylim(1.8, np.pi)
ax.set_xlabel(r'$d_L$ (Mpc)', fontsize=13)
ax.set_ylabel(r'$\iota$ (rad)', fontsize=13)
ax.set_title(r'(a) Joint $(d_L,\iota)$ posterior, uniform-in-$d_L$')
# Mode B: right of the blue distribution at the bottom edge
ax.text(28, 1.83, r'Mode B run   $d_L\in[10,30]$',
        color=MODE_B_COL, fontsize=10, weight='bold', ha='right', va='bottom',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=2))
# Mode A: just beneath the bottom edge of the orange distribution
ax.text(40, 2.18, r'Mode A run   $d_L\in[30,75]$',
        color=MODE_A_COL, fontsize=10, weight='bold', ha='center', va='top',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=2))

# ----- Panel (b): H_0 1D marginals -----
ax = axes[1]
_h0_eval = np.linspace(40, 230, 1000)
_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
for x, w, label, col in [
    (h0A, wA, r'Mode A ($d_L\in[30,75]$\,Mpc)', MODE_A_COL),
    (h0B, wB, r'Mode B ($d_L\in[10,30]$\,Mpc)', MODE_B_COL),
    (h0F, wF, r'Unrestricted',                   COMBINED_COL),
]:
    if x is None: continue
    w = w / w.sum()
    kde = gaussian_kde(x, weights=w, bw_method='silverman')
    pdf = kde(_h0_eval)
    pdf /= _trapz(pdf, _h0_eval)
    ax.plot(_h0_eval, pdf, color=col, lw=2.0, ls='-', label=label)
# Cosmological reference bands — Planck CMB and SH0ES distance-ladder.
# (LVK GW170817 band intentionally dropped: this work is itself a
# GW170817 reanalysis, so the LVK band is uninformative here.)
ax.axvspan(65.7, 68.2, color=COLORS['planck_outer'], alpha=0.3, zorder=0)
ax.axvspan(66.93 - 0.62, 66.93 + 0.62, color=COLORS['planck_inner'],
           alpha=0.3, zorder=0, label='Planck')
ax.axvspan(69.76, 76.72, color=COLORS['shoes_outer'], alpha=0.3, zorder=0)
ax.axvspan(73.24 - 1.74, 73.24 + 1.74, color=COLORS['shoes_inner'],
           alpha=0.3, zorder=0, label='SH0ES')
ax.set_xlim(40, 230); ax.set_ylim(bottom=0)
ax.set_xlabel(r'$H_0$ (km s$^{-1}$ Mpc$^{-1}$)', fontsize=13)
ax.set_ylabel(r'$P(H_0)$ (km$^{-1}$ s Mpc)', fontsize=13)
ax.set_title(r'(b) $H_0$ marginal by mode')
ax.legend(frameon=False, fontsize=10, loc='upper right')

for a in axes:
    for sp in a.spines.values():
        sp.set_edgecolor('black'); sp.set_linewidth(1.5)

fig.tight_layout()
p = os.path.join(OUT_DIR, 'bimodality')
plt.savefig(f'{p}.pdf', bbox_inches='tight')
plt.savefig(f'{p}.png', dpi=150, bbox_inches='tight')
print(f"  -> Saved {p}.pdf / .png")
plt.close(fig)
