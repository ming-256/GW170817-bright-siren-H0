"""
Abbott+2017 reproduction study — H_0 sky-restriction comparison.

Compares the H_0 posterior width progression as we tighten the sky prior
to emulate Abbott+2017's fixed-sky (EM-counterpart) analysis:

  Full-sky (s07) → narrow-sky ±0.05 rad (s15) → fixed-sky ±0.001 rad (s19)

Also overlays the IMRPhenomPv2 full-sky run (waveform-family reference) to
separate sky-restriction and waveform contributions to the width difference.

s19 fixed-sky runs are loaded if present; missing files are skipped with a
warning so the plot can be generated immediately from s15 results alone.

Output: Results/gwtc1_phasemarg/plots/H0_abbott_reproduction.{pdf,png}
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import (
    OUT_DIR, RESULTS_DIR, COLORS, load_nested_csv, compute_hpd_samples,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

XLIM = (40, 180)
BINS = np.linspace(XLIM[0], XLIM[1], 71)   # ~2 km/s bins

TS = 'Results/test_suite'

RUNS = [
    # (csv, label, color, linestyle)
    ('Figure1.csv',
     r'Abbott+2017 (IMRPhenomPv2\_NRTidal, fixed sky)',
     '0.40', '--', True),                           # True = uniform weights

    (f'{TS}/s07__gw170817__imrphenompv2__baseline_lvkbounds__seed0000/samples.csv',
     r'IMRPhenomPv2 — full-sky (this work, no tides)',
     'tab:orange', ':', False),

    (f'{TS}/s07__gw170817__imrphenomxas_nrtidalv3__baseline_lvkbounds__seed0000/samples.csv',
     r'IMRPhenomXAS\_NRTidalv3 — full-sky',
     COLORS['imr_baseline'], '-', False),

    (f'{TS}/s15__gw170817__imrphenomxas_nrtidalv3__baseline_lvkbounds_narrow__seed0000/samples.csv',
     r'IMRPhenomXAS\_NRTidalv3 — narrow-sky $\pm$0.05 rad',
     COLORS['imr_baseline'], '--', False),

    (f'{TS}/s19__gw170817__imrphenomxas_nrtidalv3__baseline_lvkbounds_fixedsky__seed0000/samples.csv',
     r'IMRPhenomXAS\_NRTidalv3 — fixed-sky $\pm$0.001 rad',
     COLORS['imr_baseline'], '-.', False),

    (f'{TS}/s07__gw170817__imrphenomd_nrtidalv2__baseline_lvkbounds__seed0000/samples.csv',
     r'IMRPhenomD\_NRTidalv2 — full-sky',
     COLORS['tf2_baseline'], '-', False),

    (f'{TS}/s15__gw170817__imrphenomd_nrtidalv2__baseline_lvkbounds_narrow__seed0000/samples.csv',
     r'IMRPhenomD\_NRTidalv2 — narrow-sky $\pm$0.05 rad',
     COLORS['tf2_baseline'], '--', False),

    (f'{TS}/s19__gw170817__imrphenomd_nrtidalv2__baseline_lvkbounds_fixedsky__seed0000/samples.csv',
     r'IMRPhenomD\_NRTidalv2 — fixed-sky $\pm$0.001 rad',
     COLORS['tf2_baseline'], '-.', False),
]


def load_xw(csv, uniform_weights):
    if uniform_weights:
        df = pd.read_csv(csv)
        x = df['H0_samples'].to_numpy().astype(float)
        w = np.ones(len(x)) / len(x)
    else:
        s = load_nested_csv(csv)
        x = s['H_0'].to_numpy().astype(float)
        w = np.asarray(s.get_weights(), dtype=float)
    finite = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[finite], w[finite]
    return x, w / w.sum()


fig, ax = plt.subplots(figsize=(9, 5))

centres = 0.5 * (BINS[:-1] + BINS[1:])
print("Run                                               MAP    68% HPD")

for csv, label, color, ls, uniform in RUNS:
    if not os.path.exists(csv):
        print(f"  SKIP (not found): {csv}")
        continue
    x, w = load_xw(csv, uniform)
    counts, _ = np.histogram(x, bins=BINS, weights=w, density=True)
    x_step = np.r_[BINS[0], np.repeat(BINS[1:-1], 2), BINS[-1]]
    y_step = np.r_[np.repeat(counts, 2)]
    ax.plot(x_step, y_step, color=color, lw=1.8, ls=ls, label=label)
    lo68, hi68 = compute_hpd_samples(x, w, 0.6827)
    map_ = float(centres[np.argmax(counts)])
    print(f"  {label[:50]:<50}  {map_:5.1f}  [{lo68:.1f}, {hi68:.1f}] w={hi68-lo68:.1f}")

# Planck / SH0ES bands
ax.axvspan(65.7, 68.2, color=COLORS['planck_outer'], alpha=0.25, zorder=0)
ax.axvspan(66.93 - 0.62, 66.93 + 0.62, color=COLORS['planck_inner'],
           alpha=0.25, zorder=0, label='Planck')
ax.axvspan(69.76, 76.72, color=COLORS['shoes_outer'], alpha=0.25, zorder=0)
ax.axvspan(73.24 - 1.74, 73.24 + 1.74, color=COLORS['shoes_inner'],
           alpha=0.25, zorder=0, label='SH0ES')

ax.set_xlim(XLIM)
ax.set_ylim(bottom=0)
ax.set_xlabel(r'$H_0\;(\mathrm{km\,s^{-1}\,Mpc^{-1}})$')
ax.set_ylabel(r'$P(H_0)$')
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)
ax.legend(frameon=False, fontsize=8.5, loc='upper right')
ax.set_title('Abbott+2017 reproduction study: sky-restriction and waveform effects on $H_0$',
             fontsize=10)

fig.tight_layout()
path = os.path.join(OUT_DIR, 'H0_abbott_reproduction')
plt.savefig(f'{path}.pdf', bbox_inches='tight')
plt.savefig(f'{path}.png', dpi=150, bbox_inches='tight')
print(f"\n-> Saved {path}.pdf / .png")
plt.close(fig)
