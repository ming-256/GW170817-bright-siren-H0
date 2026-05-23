"""Build a complete H0 / d_L summary stats CSV across all primary GW170817 runs.

Computes for each posterior:
  - MAP, KDE-based 68% / 95% HPD intervals
  - Weighted median / 68% / 95% quantile intervals
  - tail probabilities P(H0 > 120) and P(H0 > 150)

Output: Results/gwtc1_phasemarg/summary_stats_full.csv
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mnras_paper', 'test_suite', 'analysis'))
from _helpers import load_run, weighted_median, weighted_quantiles, weighted_tail_prob, RESULTS_ROOT
sys.path.insert(0, os.path.dirname(__file__))
from _plot_utils import compute_hpd
from scipy.stats import gaussian_kde
from anesthetic import read_chains

def hpd_summary(x, w, xmin, xmax, level):
    grid = np.linspace(xmin, xmax, 4000)
    w = np.asarray(w); w = w / w.sum()
    kde = gaussian_kde(x, weights=w)
    pdf = kde(grid)
    map_ = grid[np.argmax(pdf)]
    lo, hi = compute_hpd(grid, pdf, level)
    return map_, lo, hi

def gw170817_h0_stats(samples_csv, label):
    if 'reweighted' in samples_csv.lower() or 'reweight' in os.path.basename(samples_csv).lower():
        # Reweighted CSVs use plain CSV with explicit 'weight' column
        df = pd.read_csv(samples_csv, low_memory=False)
        if 'H_0' not in df.columns:
            return None
        x = df['H_0'].to_numpy().astype(float)
        w = df['weight'].to_numpy().astype(float)
    else:
        s = read_chains(samples_csv)
        if 'H_0' not in s.columns:
            return None
        x = s['H_0'].to_numpy(); w = np.asarray(s.get_weights())
    map_, hpd68lo, hpd68hi = hpd_summary(x, w, 40, 250, 0.68269)
    _,    hpd95lo, hpd95hi = hpd_summary(x, w, 40, 250, 0.95450)
    q05, q16, q50, q84, q95 = weighted_quantiles(x, w, [0.025, 0.15865, 0.5, 0.84135, 0.975])
    pgt120 = weighted_tail_prob(x, w, 120.0)
    pgt150 = weighted_tail_prob(x, w, 150.0)
    return dict(label=label, MAP=map_,
                HPD68_lo=hpd68lo, HPD68_hi=hpd68hi,
                HPD95_lo=hpd95lo, HPD95_hi=hpd95hi,
                q05=q05, q16=q16, q50=q50, q84=q84, q95=q95,
                P_gt_120=pgt120, P_gt_150=pgt150,
                n_samples=len(x))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

ENTRIES = [
    # primary IMRPhenomD_NRTidalv2 host-localised suite (existing)
    ('IMR baseline (host-loc)',
     f'{REPO}/Results/gwtc1_phasemarg/PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_baseline.csv'),
    ('IMR flat-z direct (host-loc)',
     f'{REPO}/Results/gwtc1_phasemarg/PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_flatZ.csv'),
    ('IMR flat-z reweighted (host-loc)',
     f'{REPO}/Results/gwtc1_phasemarg/PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_reweighted_flatZ.csv'),
    ('IMR vp250 (host-loc)',
     f'{REPO}/Results/gwtc1_phasemarg/PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_vp250.csv'),
    # TaylorF2
    ('TF2 baseline (host-loc)',
     f'{REPO}/Results/gwtc1_phasemarg/PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_baseline.csv'),
    ('TF2 flat-z direct (host-loc)',
     f'{REPO}/Results/gwtc1_phasemarg/PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_flatZ.csv'),
    ('TF2 flat-z reweighted (host-loc)',
     f'{REPO}/Results/gwtc1_phasemarg/PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_reweighted_flatZ.csv'),
    ('TF2 vp250 (host-loc)',
     f'{REPO}/Results/gwtc1_phasemarg/PhaseMarg_Heterodyned_TaylorF2_local_psd-gwtc1_ref-gwtc1_vp250.csv'),
    # s07 LVK-bounds 4-waveform suite
    ('IMR baseline (LVK-bounds)',
     f'{REPO}/Results/test_suite/s07__gw170817__imrphenomd_nrtidalv2__baseline_lvkbounds__seed0000/samples.csv'),
    ('XAS_NRTv3 baseline (LVK-bounds)',
     f'{REPO}/Results/test_suite/s07__gw170817__imrphenomxas_nrtidalv3__baseline_lvkbounds__seed0000/samples.csv'),
    ('Pv2 baseline (LVK-bounds)',
     f'{REPO}/Results/test_suite/s07__gw170817__imrphenompv2__baseline_lvkbounds__seed0000/samples.csv'),
    # bimodality
    ('IMR Mode A (flat-z, dL [30,75])',
     f'{REPO}/Results/test_suite/s10__gw170817__imrphenomd_nrtidalv2__flatz__dL30-75__refGWTC1__seed0000/samples.csv'),
    ('IMR Mode B (flat-z, dL [10,30])',
     f'{REPO}/Results/test_suite/s10__gw170817__imrphenomd_nrtidalv2__flatz__dL10-30__refGWTC1__seed0000/samples.csv'),
    ('IMR refModeB (flat-z, dL [10,75])',
     f'{REPO}/Results/test_suite/s10__gw170817__imrphenomd_nrtidalv2__flatz__dL10-75__refModeB__seed0000/samples.csv'),
    # s14 IMRPhenomXAS_NRTidalv3 prior-sensitivity sweep (full-sky, default mass bounds)
    ('XAS_NRTv3 baseline (full-sky)',
     f'{REPO}/Results/test_suite/s14__gw170817__imrphenomxas_nrtidalv3__baseline__seed0000/samples.csv'),
    ('XAS_NRTv3 flat-z direct (full-sky)',
     f'{REPO}/Results/test_suite/s14__gw170817__imrphenomxas_nrtidalv3__flatz__seed0000/samples.csv'),
    ('XAS_NRTv3 flat-z reweighted (full-sky)',
     f'{REPO}/Results/test_suite/s14__gw170817__imrphenomxas_nrtidalv3__reweighted_flatz__seed0000/samples.csv'),
    ('XAS_NRTv3 vp250 (full-sky)',
     f'{REPO}/Results/test_suite/s14__gw170817__imrphenomxas_nrtidalv3__vp250__seed0000/samples.csv'),
]

rows = []
for label, csv in ENTRIES:
    if not os.path.exists(csv):
        print(f"  SKIP missing {label}: {csv}")
        continue
    r = gw170817_h0_stats(csv, label)
    if r is None:
        print(f"  no H_0 column in {label}")
        continue
    rows.append(r)

df = pd.DataFrame(rows)
out = f'{REPO}/Results/gwtc1_phasemarg/summary_stats_full.csv'
df.to_csv(out, index=False, float_format='%.4f')

# Print human-friendly view
print()
print(f"{'label':<40} {'MAP':>6} {'HPD68':>14} {'HPD95':>14} {'P>120':>7} {'P>150':>7}")
for _, r in df.iterrows():
    print(f"{r['label']:<40} {r['MAP']:>6.1f} "
          f"[{r['HPD68_lo']:>5.1f},{r['HPD68_hi']:>5.1f}] "
          f"[{r['HPD95_lo']:>5.1f},{r['HPD95_hi']:>5.1f}] "
          f"{r['P_gt_120']:>7.3f} {r['P_gt_150']:>7.3f}")
print()
print(f"Wrote {out}")
