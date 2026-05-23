"""
Sky-prior runtime comparison for GW170817 — full-sky vs narrow-sky.

Two panels (matching the rest of the paper's _plot_utils style):

  Left: bar chart of wall-clock seconds per (waveform, sky-prior) cell at
        matched n_live, separately for heterodyned and unheterodyned.
  Right: ratio narrow / full-sky per cell (the "what does sky-prior buy?"
        number used in the runtime discussion section).

Heterodyned data:
  - Full-sky:   s07 (IMR + XAS + Pv2 baseline_lvkbounds, n_live=5000)
  - Narrow-sky: s15 (IMR + XAS baseline_lvkbounds_narrow, n_live=5000)
Unheterodyned data (no new runs needed — pair already in tree):
  - Full-sky:   gwtc1_phasemarg/PhaseMarg_Unheterodyned_*_local_psd-gwtc1_full_sky.csv
  - Narrow-sky: gwtc1_phasemarg/PhaseMarg_Unheterodyned_*_local_psd-gwtc1.csv
                (the unhetero `--wide-prior` flag = ±0.05 rad of NGC 4993; same
                 sky restriction as the hetero --narrow-sky flag)

Output: Results/gwtc1_phasemarg/plots/sky_prior_runtime.{pdf,png}
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import OUT_DIR, COLORS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCALING_CSV = os.path.join(REPO, 'Results', 'scaling_study', 'scaling_summary_full.csv')

df = pd.read_csv(SCALING_CSV)

# Heterodyned matched pairs at n_live=5000 (LVK-bounds).
het_full = df[(df['kind'] == 'heterodyned') &
              (df['priors'] == 'lvk-bounds') &
              (df['n_live'] == 5000)]
het_narrow = df[(df['kind'] == 'heterodyned') &
                (df['priors'] == 'lvk-bounds, narrow-sky') &
                (df['n_live'] == 5000)]

# Unheterodyned matched pairs at n_live=1500 (gwtc1_phasemarg row data).
unhet_full = df[(df['kind'] == 'unheterodyned') &
                (df['priors'] == 'full-sky') &
                (df['n_live'] == 1500)]
unhet_narrow = df[(df['kind'] == 'unheterodyned') &
                  (df['priors'] == 'host-localised') &
                  (df['n_live'] == 1500)]

def merge_pair(full, narrow, kind_label):
    pairs = []
    for wf in full['waveform'].unique():
        f = full[full['waveform'] == wf]
        n = narrow[narrow['waveform'] == wf]
        if len(f) == 0 or len(n) == 0:
            print(f"  WARNING: no match for {kind_label} {wf}: full={len(f)} narrow={len(n)}")
            continue
        t_full = float(f['total_s'].iloc[0])
        t_narrow = float(n['total_s'].iloc[0])
        pairs.append((wf, kind_label, t_full, t_narrow, t_narrow / t_full))
        print(f"  {kind_label:14s} {wf:24s}  full={t_full:8.1f}s  narrow={t_narrow:8.1f}s  ratio={t_narrow/t_full:.3f}")
    return pairs

print("Sky-prior runtime pairs:")
pairs = []
pairs += merge_pair(het_full, het_narrow, 'heterodyned')
pairs += merge_pair(unhet_full, unhet_narrow, 'unheterodyned')

if not pairs:
    print("\n  No matched pairs available — narrow-sky s15 runs are still pending.")
    print(f"  Run: bash mnras_paper/test_suite/session_plans/session_15_sky_prior_runtime.sh")
    sys.exit(0)

# ---------------- plotting ----------------
fig, (ax_bar, ax_ratio) = plt.subplots(1, 2, figsize=(13, 5.5),
                                       gridspec_kw={'width_ratios': [1.6, 1.0]})

def _esc(wf):
    return wf.replace('_', r'\_')
labels = ['%s\n(%s)' % (_esc(wf), kind) for wf, kind, _, _, _ in pairs]
x = np.arange(len(pairs))
w = 0.38
full_times   = [p[2] for p in pairs]
narrow_times = [p[3] for p in pairs]
ratios       = [p[4] for p in pairs]

# Per-row colour by waveform; alpha encodes sky-prior.
def wf_color(wf):
    if 'XAS' in wf:   return COLORS['flatZ']
    if 'TaylorF2' in wf: return COLORS['tf2_baseline']
    return COLORS['imr_baseline']

bar_full   = ax_bar.bar(x - w/2, full_times,   w, label='Full-sky',
                        color=[wf_color(wf) for wf, *_ in pairs], alpha=0.4,
                        edgecolor='black', linewidth=1.0)
bar_narrow = ax_bar.bar(x + w/2, narrow_times, w, label='Narrow-sky (LVK-style)',
                        color=[wf_color(wf) for wf, *_ in pairs], alpha=1.0,
                        edgecolor='black', linewidth=1.0)
for b, t in zip(bar_full, full_times):
    ax_bar.text(b.get_x() + b.get_width()/2, t, f' {t:.0f}s',
                ha='center', va='bottom', fontsize=9, rotation=0)
for b, t in zip(bar_narrow, narrow_times):
    ax_bar.text(b.get_x() + b.get_width()/2, t, f' {t:.0f}s',
                ha='center', va='bottom', fontsize=9, rotation=0)

ax_bar.set_xticks(x); ax_bar.set_xticklabels(labels, fontsize=9)
ax_bar.set_ylabel('Wall-clock total (s)', fontsize=12)
ax_bar.set_yscale('log')
ax_bar.set_title(r'GW170817 — wall-clock by sky prior', fontsize=12)
ax_bar.legend(frameon=False, fontsize=10)
for spine in ax_bar.spines.values():
    spine.set_edgecolor('black'); spine.set_linewidth(1.2)

# Ratio panel
bar_ratio = ax_ratio.bar(x, ratios, 0.6,
                          color=[wf_color(wf) for wf, *_ in pairs],
                          alpha=0.85, edgecolor='black', linewidth=1.0)
for b, r in zip(bar_ratio, ratios):
    ax_ratio.text(b.get_x() + b.get_width()/2, r, f' {r:.2f}',
                  ha='center', va='bottom', fontsize=10)
ax_ratio.axhline(1.0, color='0.3', ls='--', lw=1.0)
ax_ratio.set_xticks(x); ax_ratio.set_xticklabels(labels, fontsize=9)
ax_ratio.set_ylabel(r'$t_{\rm narrow} / t_{\rm full-sky}$', fontsize=12)
ax_ratio.set_ylim(0, max(1.1, max(ratios) * 1.15))
ax_ratio.set_title('Speedup from EM-localised sky', fontsize=12)
for spine in ax_ratio.spines.values():
    spine.set_edgecolor('black'); spine.set_linewidth(1.2)

fig.tight_layout()
p = os.path.join(OUT_DIR, 'sky_prior_runtime')
plt.savefig(f'{p}.pdf', bbox_inches='tight')
plt.savefig(f'{p}.png', dpi=150, bbox_inches='tight')
print(f"\n  -> Saved {p}.pdf / .png")
plt.close(fig)
