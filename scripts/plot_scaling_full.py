"""
Comprehensive scaling-study plot: heterodyned vs unheterodyned.

Single panel runtime-vs-n_live; unheterodyned IMR (host-loc) annotated with
speedup factor against the heterodyned LVK-bounds series at matched n_live.
TaylorF2 unhetero and full-sky unhetero are excluded.

Output: Results/gwtc1_phasemarg/plots/scaling_study_full.{pdf,png}
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

CSV = os.path.join(RESULTS_DIR, 'scaling_study', 'scaling_summary_full.csv')
df = pd.read_csv(CSV)

# Two series: heterodyned IMR (host-loc, A100) vs unheterodyned IMR (host-loc).
# LVK-bounds series intentionally dropped — the central scientific point is
# heterodyned-vs-unheterodyned, not the choice of mass-prior bounds.
GROUPS = [
    (((df['kind']=='heterodyned') & (df['waveform']=='IMRPhenomD_NRTidalv2')
     & (df['priors']=='host-localised')),
     'IMRPhenomD_NRTidalv2 heterodyned',
     COLORS['imr_baseline'], 'o'),
    (((df['kind']=='unheterodyned') & (df['waveform']=='IMRPhenomD_NRTidalv2')
      & (df['priors']=='host-localised')),
     'IMRPhenomD_NRTidalv2 unheterodyned',
     'tab:red', '^'),
]

fig, ax = plt.subplots(figsize=(8.0, 6.0))

for mask, lab, col, mk in GROUPS:
    sub = df[mask].sort_values('n_live')
    if len(sub) == 0:
        continue
    ax.plot(sub['n_live'], sub['total_s'], color=col, lw=1.6,
            marker=mk, ms=8, label=lab, zorder=5)

# Linear reference
n_ref = np.array([200, 200000])
rate = 858.0 / 5000.0
ax.plot(n_ref, rate * n_ref, 'k--', lw=1.0, alpha=0.4,
        label=r'Linear ref $\propto n_{\rm live}$')

# Calculate gradient for heterodyned LVK-bounds
df_het_lvk = df[(df['kind']=='heterodyned') & (df['waveform']=='IMRPhenomD_NRTidalv2') & (df['priors'] == 'lvk-bounds')]
if len(df_het_lvk) > 1:
    log_n = np.log10(df_het_lvk['n_live'])
    log_t = np.log10(df_het_lvk['total_s'])
    slope, intercept = np.polyfit(log_n, log_t, 1)
    print(f"  Heterodyned LVK-bounds log-log gradient: {slope:.2f}")

# Speedup annotation: unhet IMR host-loc vs hetero host-loc, matched n_live.
het_hl = df[(df['kind']=='heterodyned') & (df['waveform']=='IMRPhenomD_NRTidalv2')
            & (df['priors']=='host-localised')].set_index('n_live')['total_s']
unhet_hl = df[(df['kind']=='unheterodyned') & (df['waveform']=='IMRPhenomD_NRTidalv2')
              & (df['priors']=='host-localised')].set_index('n_live')['total_s']
nl_arr = np.sort(het_hl.index.to_numpy())
t_arr = het_hl.loc[nl_arr].to_numpy()
for nl, t_un in unhet_hl.items():
    if nl in het_hl.index:
        t_h = het_hl.loc[nl]
    else:
        t_h = np.exp(np.interp(np.log(nl), np.log(nl_arr), np.log(t_arr)))
    speedup = t_un / t_h
    ax.annotate(f'{speedup:.0f}'+r'$\times$ slower',
                xy=(nl, t_un), xytext=(nl*1.3, t_un*0.78),
                fontsize=10, color='tab:red',
                arrowprops=dict(arrowstyle='-', color='tab:red', lw=0.6, alpha=0.6))

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$n_{\rm live}$', fontsize=13)
ax.set_ylabel('Total wall-clock (s)', fontsize=13)
ax.set_title('Runtime scaling, all configurations')
ax.legend(frameon=False, fontsize=9, loc='lower right')
ax.grid(True, which='both', alpha=0.2)

for spine in ax.spines.values():
    spine.set_edgecolor('black'); spine.set_linewidth(1.5)

fig.tight_layout()
p = os.path.join(OUT_DIR, 'scaling_study_full')
plt.savefig(f'{p}.pdf', bbox_inches='tight')
plt.savefig(f'{p}.png', dpi=150, bbox_inches='tight')
print(f"  -> Saved {p}.pdf / .png")
plt.close(fig)

# Speedup table
print("\nSpeedup at matched n_live (host-loc):")
print(f"  {'n_live':>8}  {'hetero (s)':>10}  {'unhetero (s)':>13}  {'speedup':>8}")
for nl, t_un in unhet_hl.items():
    if nl in het_hl.index:
        t_h = het_hl.loc[nl]
    else:
        t_h = float(np.exp(np.interp(np.log(nl), np.log(nl_arr), np.log(t_arr))))
    print(f"  {nl:>8}  {t_h:>10.0f}  {t_un:>13.0f}  {t_un/t_h:>7.1f}x")

# Verify n_live=20000 is plotted in LVK-bounds series
lvk20k = df[(df['kind']=='heterodyned') & (df['waveform']=='IMRPhenomD_NRTidalv2')
            & (df['priors'] == 'lvk-bounds')
            & (df['n_live']==20000)]
print(f"\nLVK-bounds n_live=20000 rows plotted: {len(lvk20k)}")
print(lvk20k[['source','priors','total_s']].to_string(index=False))
