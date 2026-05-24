"""
Single source of truth for every numerical claim in the MNRAS paper.

Reads the canonical sample CSVs and the sampler.log lnZ entries, then writes:
  - Results/gwtc1_phasemarg/paper_tables.csv  (one row per Table 4/5/6 entry)
  - Results/gwtc1_phasemarg/paper_tables.tex  (LaTeX fragments \input by main.tex)
  - Results/gwtc1_phasemarg/paper_diagnostics.csv  (n_eff for reweighting comparison)

Conventions:
  - MAP: histogram-mode on a 1 km/s/Mpc grid over (40, 230) km/s/Mpc.
  - HPD intervals: shortest contiguous interval containing the requested weight,
    computed directly from weighted samples (no KDE).
  - lnZ: parsed from sampler.log "Log Evidence: X +/- Y".
"""
import os, re, glob, json
import numpy as np
import pandas as pd
from anesthetic import read_chains

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TS = os.path.join(ROOT, 'results', 'test_suite')
OUT_DIR = os.path.join(ROOT, 'results', 'gwtc1_phasemarg')

H0_BINS = np.linspace(40, 230, 191)            # 1 km/s/Mpc bins
H0_CENTRES = 0.5 * (H0_BINS[:-1] + H0_BINS[1:])

LNZ_RE = re.compile(r'Log Evidence:\s*([-\d.]+)\s*\+/\-\s*([\d.]+)')


# --------------------------------------------------------------------------- #
# small helpers                                                               #
# --------------------------------------------------------------------------- #
def parse_lnZ(run_dir):
    log = os.path.join(TS, run_dir, 'sampler.log')
    if not os.path.exists(log):
        return None, None
    with open(log) as f:
        for line in f:
            m = LNZ_RE.search(line)
            if m:
                return float(m.group(1)), float(m.group(2))
    return None, None


def load_samples(run_dir, csv_name='samples.csv'):
    p = os.path.join(TS, run_dir, csv_name)
    if not os.path.exists(p):
        return None, None
    if 'reweight' in run_dir.lower() or csv_name.endswith('reweighted.csv'):
        df = pd.read_csv(p, low_memory=False)
        x = df['H_0'].to_numpy().astype(float)
        w = df['weight'].to_numpy().astype(float)
    else:
        s = read_chains(p)
        x = s['H_0'].to_numpy().astype(float)
        w = np.asarray(s.get_weights(), dtype=float)
    finite = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return x[finite], w[finite]


def map_h0(x, w):
    counts, _ = np.histogram(x, bins=H0_BINS, weights=w)
    return float(H0_CENTRES[int(np.argmax(counts))])


def weighted_median(x, w):
    """Median from weighted samples (referee m14: robust to MAP bin noise)."""
    idx = np.argsort(x)
    cum = np.cumsum(w[idx]) / w.sum()
    return float(np.interp(0.5, cum, x[idx]))


def hpd(x, w, frac):
    idx = np.argsort(x)
    xs = x[idx]; ws = w[idx]/w.sum()
    cdf = np.cumsum(ws); n = len(xs)
    best = (np.inf, xs[0], xs[-1])
    for i in range(n):
        target = cdf[i] + frac
        if target > 1: break
        j = np.searchsorted(cdf, target)
        if j >= n: break
        width = xs[j] - xs[i]
        if width < best[0]:
            best = (width, xs[i], xs[j])
    return float(best[1]), float(best[2])


def tail(x, w, thr):
    return float(w[x > thr].sum() / w.sum())


def n_eff(w):
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    if s2 == 0: return float('nan')
    return s1 * s1 / s2


# --------------------------------------------------------------------------- #
# Define the runs that go into each table                                     #
# --------------------------------------------------------------------------- #
TABLE5_PRIOR_SENSITIVITY = [
    ('Baseline ($\\pi(d_L)\\propto d_L^{2}$)',
        's14__gw170817__imrphenomxas_nrtidalv3__baseline__seed0000'),
    ('Uniform-in-$d_L$, direct',
        's14__gw170817__imrphenomxas_nrtidalv3__flatz__seed0000'),
    ('Uniform-in-$d_L$, reweighted',
        's14__gw170817__imrphenomxas_nrtidalv3__reweighted_flatz__seed0000'),
    ('$\\sigma_{v_p}=250\\,\\mathrm{km\\,s^{-1}}$',
        's14__gw170817__imrphenomxas_nrtidalv3__vp250__seed0000'),
    # v_p mean sweep (s18); vp=310 replicates baseline to <0.01 in lnZ
    ('$\\langle v_p\\rangle=215\\,\\mathrm{km\\,s^{-1}}$',
        's18__gw170817__imrphenomxas_nrtidalv3__baseline__vpmean215__seed0000'),
    ('$\\langle v_p\\rangle=405\\,\\mathrm{km\\,s^{-1}}$',
        's18__gw170817__imrphenomxas_nrtidalv3__baseline__vpmean405__seed0000'),
]

TABLE4_CROSS_WAVEFORM = [
    ('\\IMRX\\ (primary)',
        's07__gw170817__imrphenomxas_nrtidalv3__baseline_lvkbounds__seed0000'),
    ('\\IMR\\ (anchor)',
        's07__gw170817__imrphenomd_nrtidalv2__baseline_lvkbounds__seed0000'),
    ('\\TF\\ (family check)',
        's07__gw170817__taylorf2__baseline_lvkbounds__seed0000'),
]

TABLE6_BIMODALITY = [
    # seed=0 (primary)
    ('Mode A',          '$[30,75]$',
        's10__gw170817__imrphenomd_nrtidalv2__flatz__dL30-75__refGWTC1__seed0000'),
    ('Mode B',          '$[10,30]$',
        's10__gw170817__imrphenomd_nrtidalv2__flatz__dL10-30__refGWTC1__seed0000'),
    ('Unrestricted',    '$[10,75]$',
        's10__gw170817__imrphenomd_nrtidalv2__flatz__dL10-75__refModeB__seed0000'),
    # seed=1 independent verification (s18)
    ('Mode A (s=1)',    '$[30,75]$',
        's18__gw170817__imrphenomd_nrtidalv2__flatz__dL30-75__refGWTC1__seed0001'),
    ('Mode B (s=1)',    '$[10,30]$',
        's18__gw170817__imrphenomd_nrtidalv2__flatz__dL10-30__refGWTC1__seed0001'),
    ('Unrestr. (s=1)', '$[10,75]$',
        's18__gw170817__imrphenomd_nrtidalv2__flatz__dL10-75__refModeB__seed0001'),
]

# GW150914 validation: Table 1
GW150914_RUNS = [
    ('this work, \\XPHM\\ ($n_{\\rm live}=8000$)',
        's17a__gw150914__imrphenomxphm__nlive8000_mcmc160__seed0000'),
    ('this work, \\XPHM\\ ($n_{\\rm live}=5000$, cross-check)',
        's06__gw150914__imrphenomxphm__lvkbounds__seed0000'),
]


# --------------------------------------------------------------------------- #
# Build rows                                                                  #
# --------------------------------------------------------------------------- #
def build_h0_row(label, run_dir):
    x, w = load_samples(run_dir)
    if x is None:
        return dict(label=label, run=run_dir, error='no samples')
    lnz, dlnz = parse_lnZ(run_dir)
    return dict(
        label=label, run=run_dir,
        MAP=map_h0(x, w),
        median=weighted_median(x, w),
        HPD68_lo=hpd(x, w, 0.68269)[0], HPD68_hi=hpd(x, w, 0.68269)[1],
        HPD95_lo=hpd(x, w, 0.95450)[0], HPD95_hi=hpd(x, w, 0.95450)[1],
        P_gt_120=tail(x, w, 120), P_gt_150=tail(x, w, 150),
        n_eff=n_eff(w), N_samples=len(x),
        lnZ=lnz, dlnZ=dlnz,
    )


def gw150914_summary(run_dir):
    """Median + 68% HPD for the 5 GW150914 source params."""
    p = os.path.join(TS, run_dir, 'samples.csv')
    if not os.path.exists(p):
        return None
    s = read_chains(p)
    w = np.asarray(s.get_weights(), dtype=float)
    out = {'run': run_dir}
    PARAMS = [('M_c', '$\\mathcal{M}_c$ ($M_\\odot$)'),
              ('q',   '$q$'),
              ('s1_z', '$\\chi_{1z}$'),
              ('d_L', '$d_L$ (Mpc)'),
              ('iota', '$\\iota$ (rad)')]
    for col, _ in PARAMS:
        if col in s.columns:
            x = s[col].to_numpy().astype(float)
            mw = w / w.sum()
            order = np.argsort(x)
            cdf = np.cumsum(mw[order])
            med = float(x[order][np.searchsorted(cdf, 0.5)])
            lo, hi = hpd(x, w, 0.68269)
            out[col + '_med'] = med
            out[col + '_lo']  = lo
            out[col + '_hi']  = hi
    lnz, dlnz = parse_lnZ(run_dir)
    out['lnZ'] = lnz; out['dlnZ'] = dlnz
    return out


# --------------------------------------------------------------------------- #
# LaTeX writers                                                               #
# --------------------------------------------------------------------------- #
def _hpd_str(lo, hi):
    return f'$[{lo:.1f},{hi:.1f}]$'

def _lnz_str(lnz, dlnz):
    return f'${lnz:.2f}\\pm{dlnz:.2f}$' if lnz is not None else '--'

def _tail_str(p):
    if p < 1e-4 and p > 0: return '$<10^{-4}$'
    if p == 0: return '$0.000$'
    return f'${p:.3f}$'


def write_table4(rows, path):
    body = ''
    for r in rows:
        body += (f"    {r['label']:<28} & {r['MAP']:.1f} & "
                 f"{_hpd_str(r['HPD68_lo'], r['HPD68_hi'])} & "
                 f"{_hpd_str(r['HPD95_lo'], r['HPD95_hi'])} & "
                 f"{_tail_str(r['P_gt_120'])} & {_lnz_str(r['lnZ'], r['dlnZ'])} \\\\\n")
    with open(path, 'w') as f:
        f.write(body)


def write_table5(rows, path):
    body = ''
    for i, r in enumerate(rows):
        if i == 4:  # separator before vp-mean sweep group
            body += '    \\midrule\n'
        lz = _lnz_str(r['lnZ'], r['dlnZ']) if 'reweighted' not in r['run'] else '(post-hoc)'
        body += (f"    {r['label']:<46} & {r['MAP']:.1f} & {r['median']:.1f} & "
                 f"{_hpd_str(r['HPD68_lo'], r['HPD68_hi'])} & "
                 f"{_tail_str(r['P_gt_120'])} & {_tail_str(r['P_gt_150'])} & {lz} \\\\\n")
    body += '    \\bottomrule\n'
    with open(path, 'w') as f:
        f.write(body)


def write_table6(rows, path):
    body = ''
    for i, r in enumerate(rows):
        if i == 3:  # separator before seed=1 group
            body += '    \\midrule\n'
        body += (f"    {r['label']:<14} & {r['dL_range']} & {r['MAP']:.1f} & "
                 f"{_hpd_str(r['HPD68_lo'], r['HPD68_hi'])} & "
                 f"{_tail_str(r['P_gt_120'])} & {_lnz_str(r['lnZ'], r['dlnZ'])} \\\\\n")
    body += '    \\bottomrule\n'
    with open(path, 'w') as f:
        f.write(body)


def write_gw150914(rows, path):
    body = ''
    pretty = {'M_c_med': '$\\mathcal{M}_c$', 'q_med': '$q$',
              's1_z_med': '$\\chi_{1z}$', 'd_L_med': '$d_L/{\\rm Mpc}$',
              'iota_med': '$\\iota/{\\rm rad}$'}
    for r in rows:
        body += (f"    {r['label']:<48} & "
                 f"{r['M_c_med']:.2f} & {r['q_med']:.2f} & "
                 f"{r['d_L_med']:.0f} & {r['iota_med']:.2f} & "
                 f"{_lnz_str(r['lnZ'], r['dlnZ'])} \\\\\n")
    body += '    \\bottomrule\n'
    with open(path, 'w') as f:
        f.write(body)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    print('=== Table 4: cross-waveform LVK-bounds ===')
    t4 = [build_h0_row(lbl, run) for lbl, run in TABLE4_CROSS_WAVEFORM]
    for r in t4:
        print(f"  {r['label']}: MAP={r['MAP']:.1f}, HPD68={_hpd_str(r['HPD68_lo'],r['HPD68_hi'])}, "
              f"P>120={r['P_gt_120']:.3f}, lnZ={r['lnZ']:.2f}±{r['dlnZ']:.2f}, n_eff={r['n_eff']:.0f}")

    print('\n=== Table 5: prior sensitivity (default-mass full-sky) ===')
    t5 = []
    for lbl, run in TABLE5_PRIOR_SENSITIVITY:
        r = build_h0_row(lbl, run)
        t5.append(r)
        print(f"  {r['label']}: MAP={r['MAP']:.1f}, HPD68={_hpd_str(r['HPD68_lo'],r['HPD68_hi'])}, "
              f"P>120={r['P_gt_120']:.3f}, P>150={r['P_gt_150']:.3f}, "
              f"lnZ={r.get('lnZ') or 'n/a'}, n_eff={r['n_eff']:.0f}")

    print('\n=== Table 6: bimodality ===')
    t6 = []
    for lbl, dL, run in TABLE6_BIMODALITY:
        r = build_h0_row(lbl, run)
        r['dL_range'] = dL
        t6.append(r)
        print(f"  {r['label']} {dL}: MAP={r['MAP']:.1f}, HPD68={_hpd_str(r['HPD68_lo'],r['HPD68_hi'])}, "
              f"P>120={r['P_gt_120']:.3f}, lnZ={r['lnZ']:.2f}")

    print('\n=== Table 1: GW150914 validation ===')
    g15 = []
    for lbl, run in GW150914_RUNS:
        r = gw150914_summary(run)
        if r is None: continue
        r['label'] = lbl
        g15.append(r)
        print(f"  {r['label']}: M_c={r['M_c_med']:.2f}, q={r['q_med']:.2f}, "
              f"d_L={r['d_L_med']:.0f}, iota={r['iota_med']:.2f}, lnZ={r['lnZ']:.2f}")

    # Write LaTeX fragments
    os.makedirs(OUT_DIR, exist_ok=True)
    write_table4(t4, os.path.join(OUT_DIR, 'table4_cross_waveform.tex'))
    write_table5(t5, os.path.join(OUT_DIR, 'table5_prior_sensitivity.tex'))
    write_table6(t6, os.path.join(OUT_DIR, 'table6_bimodality.tex'))
    write_gw150914(g15, os.path.join(OUT_DIR, 'table1_gw150914.tex'))

    # Combined CSV
    rows = []
    for r in t4: rows.append({'table': 'T4', **r})
    for r in t5: rows.append({'table': 'T5', **r})
    for r in t6: rows.append({'table': 'T6', **r})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'paper_tables.csv'), index=False)

    # n_eff diagnostic
    diag = []
    for r in t5:
        diag.append({
            'variant': r['label'], 'run': r['run'],
            'N_samples': r['N_samples'], 'n_eff': r['n_eff'],
            'efficiency_pct': 100.0 * r['n_eff'] / r['N_samples'] if r['N_samples'] else 0.0,
        })
    pd.DataFrame(diag).to_csv(os.path.join(OUT_DIR, 'paper_diagnostics.csv'), index=False)

    print('\nWrote:')
    print(f"  {os.path.join(OUT_DIR, 'paper_tables.csv')}")
    for f in ['table1_gw150914', 'table4_cross_waveform',
              'table5_prior_sensitivity', 'table6_bimodality']:
        print(f"  {os.path.join(OUT_DIR, f + '.tex')}")
    print(f"  {os.path.join(OUT_DIR, 'paper_diagnostics.csv')}")


if __name__ == '__main__':
    main()
