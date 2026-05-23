"""Combine all heterodyned + unheterodyned scaling data into one CSV.

Output: Results/scaling_study/scaling_summary_full.csv

Sources:
  - Results/scaling_study/scaling_summary.csv  (heterodyned IMR, host-localised, 8 points)
  - Results/test_suite/s13__*  (heterodyned IMR, LVK-bounds, 6 points)
  - Results/test_suite/s11__*  (heterodyned IMR, LVK-bounds, 20k tighter tol)
  - Results/test_suite/s05__*unheterodyned__nlive00500__*  (unhet IMR n_live=500)
  - Results/test_suite/s04__*unheterodyned__nlive02500__*  (unhet IMR n_live=2500, imported)
  - paper_knowledge_base/a100_run_data.md  (unhet IMR, TF2 at n_live=1500 host-localised + full-sky)
"""
import os, json, sys
import pandas as pd

REPO = '/Users/mingyang/Desktop/Project/CambridgeProject/GPU-Accelerated-Bayesian-Inference-of-Gravitational-Waves'
TS_ROOT = os.path.join(REPO, 'Results', 'test_suite')
OUT = os.path.join(REPO, 'Results', 'scaling_study', 'scaling_summary_full.csv')

rows = []

# 1) Existing heterodyned scaling table
df = pd.read_csv(os.path.join(REPO, 'Results', 'scaling_study', 'scaling_summary.csv'))
for _, r in df.iterrows():
    rows.append({
        'source': 'scaling_study',
        'waveform': r['waveform'],
        'kind': 'heterodyned',
        'priors': 'host-localised',
        'n_live': int(r['n_live']),
        'dead_points': int(r['dead_points']),
        'log_evidence': float(r['log_evidence']),
        'sigma_log_z': float(r['sigma_log_z']),
        'sampling_s': float(r['sampling_s']),
        'total_s': float(r['total_s']),
    })

# 2) s13 LVK-bounds heterodyned IMR sweep
def _runtime(run_dir):
    fin_path = os.path.join(run_dir, 'finish.json')
    if os.path.exists(fin_path):
        cfg = json.load(open(os.path.join(run_dir, 'config.json')))
        fin = open(fin_path).read().strip().splitlines()[-1]
        fin = json.loads(fin)
        import datetime as dt
        fmt = '%Y-%m-%dT%H:%M:%SZ'
        s = dt.datetime.strptime(cfg['started'], fmt)
        e = dt.datetime.strptime(fin['finished'], fmt)
        return (e - s).total_seconds()
    log_path = os.path.join(run_dir, 'sampler.log')
    if os.path.exists(log_path):
        for line in open(log_path):
            line = line.strip()
            if line.startswith('Total:'):
                return float(line.split()[1].rstrip('s'))
    return float('nan')

def _logz_from_log(run_dir):
    """Parse log Z from sampler.log."""
    try:
        log_path = os.path.join(run_dir, 'sampler.log')
        for line in open(log_path):
            if 'Final log evidence' in line or 'log Z' in line.lower():
                pass
        # Fall back: read from summary CSV if present
    except FileNotFoundError:
        pass
    return None, None

# Use the test_suite _helpers to read log evidence
sys.path.insert(0, os.path.join(REPO, 'mnras_paper', 'test_suite', 'analysis'))
from _helpers import read_log_evidence_from_log, load_run

s13_runs = [
    ('s13__gw170817__imrphenomd_nrtidalv2__baseline__nlive00500__seed0000', 500),
    ('s13__gw170817__imrphenomd_nrtidalv2__baseline__nlive01000__seed0000', 1000),
    ('s13__gw170817__imrphenomd_nrtidalv2__baseline__nlive02500__seed0000', 2500),
    # s07 baseline_lvkbounds is the n_live=5000 entry
    ('s07__gw170817__imrphenomd_nrtidalv2__baseline_lvkbounds__seed0000', 5000),
    ('s13__gw170817__imrphenomd_nrtidalv2__baseline__nlive10000__seed0000', 10000),
    ('s13__gw170817__imrphenomd_nrtidalv2__baseline__nlive20000__seed0000', 20000),
]
for rid, n in s13_runs:
    d = os.path.join(TS_ROOT, rid)
    if not os.path.exists(d):
        print(f"  skip missing {rid}")
        continue
    lz, sig = read_log_evidence_from_log(d)
    try:
        run = load_run(rid)
        dead = int(len(run.samples))
    except Exception:
        dead = -1
    rows.append({
        'source': 's13/s07',
        'waveform': 'IMRPhenomD_NRTidalv2',
        'kind': 'heterodyned',
        'priors': 'lvk-bounds',
        'n_live': n,
        'dead_points': dead,
        'log_evidence': lz,
        'sigma_log_z': sig,
        'sampling_s': float('nan'),
        'total_s': _runtime(d),
    })

# 3) Unheterodyned points
unhet_test_suite = [
    ('s05__gw170817__imrphenomd_nrtidalv2__unheterodyned__nlive00500__seed0000', 500, 'IMRPhenomD_NRTidalv2'),
    ('s04__gw170817__imrphenomd_nrtidalv2__unheterodyned__nlive02500__seed0000', 2500, 'IMRPhenomD_NRTidalv2'),
]
for rid, n, wf in unhet_test_suite:
    d = os.path.join(TS_ROOT, rid)
    if not os.path.exists(d):
        continue
    lz, sig = read_log_evidence_from_log(d)
    cfg = json.load(open(os.path.join(d, 'config.json')))
    if 'wallclock_total_s' in cfg:
        total = cfg['wallclock_total_s']; samp = cfg.get('wallclock_sampling_s', float('nan'))
    else:
        total = _runtime(d); samp = float('nan')
    try:
        run = load_run(rid)
        dead = int(len(run.samples))
    except Exception:
        dead = -1
    rows.append({
        'source': rid.split('__')[0],
        'waveform': wf,
        'kind': 'unheterodyned',
        'priors': 'host-localised',
        'n_live': n,
        'dead_points': dead,
        'log_evidence': lz,
        'sigma_log_z': sig,
        'sampling_s': samp,
        'total_s': total,
    })

# 4) Unheterodyned points from existing gwtc1_phasemarg (n_live=1500)
# These are documented in paper_knowledge_base/a100_run_data.md
unhet_gwtc1 = [
    # waveform, priors, n_live, dead, ln Z, σ ln Z, JIT, sampling, total
    ('IMRPhenomD_NRTidalv2', 'host-localised', 1500, 45000, 491.65, 0.18, 471.4, 19543.1, 20042.0),
    ('TaylorF2',             'host-localised', 1500, 47250, 490.46, 0.18, 157.0,  8437.6,  8615.3),
    ('IMRPhenomD_NRTidalv2', 'full-sky',       1500, 51750, 485.55, 0.21, 319.0, 20128.1, 20474.4),
    ('TaylorF2',             'full-sky',       1500, 51750, 486.74, 0.19, 100.2,  5860.9,  5981.2),
]
for wf, priors, n, dead, lz, sig, jit, samp, tot in unhet_gwtc1:
    rows.append({
        'source': 'gwtc1_phasemarg',
        'waveform': wf,
        'kind': 'unheterodyned',
        'priors': priors,
        'n_live': n,
        'dead_points': dead,
        'log_evidence': lz,
        'sigma_log_z': sig,
        'sampling_s': samp,
        'total_s': tot,
    })

# 5) s11 (n_live=20000, tighter tolerance)
rid_11 = 's11__gw170817__imrphenomd_nrtidalv2__baseline__nlive20000__tol1e-4__seed0000'
d = os.path.join(TS_ROOT, rid_11)
if os.path.exists(d):
    lz, sig = read_log_evidence_from_log(d)
    rows.append({
        'source': 's11',
        'waveform': 'IMRPhenomD_NRTidalv2',
        'kind': 'heterodyned',
        'priors': 'lvk-bounds, tol1e-4',
        'n_live': 20000,
        'dead_points': 840000,
        'log_evidence': lz,
        'sigma_log_z': sig,
        'sampling_s': float('nan'),
        'total_s': _runtime(d),
    })

# 6) s15 narrow-sky heterodyned baseline runs (matched to s07 LVK-bounds full-sky).
s15_runs = [
    ('s15__gw170817__imrphenomd_nrtidalv2__baseline_lvkbounds_narrow__seed0000',  'IMRPhenomD_NRTidalv2'),
    ('s15__gw170817__imrphenomxas_nrtidalv3__baseline_lvkbounds_narrow__seed0000', 'IMRPhenomXAS_NRTidalv3'),
]
for rid, wf in s15_runs:
    d = os.path.join(TS_ROOT, rid)
    if not os.path.exists(d):
        print(f"  skip missing {rid} (s15 narrow-sky pending GPU run)")
        continue
    lz, sig = read_log_evidence_from_log(d)
    try:
        run = load_run(rid)
        dead = int(len(run.samples))
    except Exception:
        dead = -1
    rows.append({
        'source': 's15',
        'waveform': wf,
        'kind': 'heterodyned',
        'priors': 'lvk-bounds, narrow-sky',
        'n_live': 5000,
        'dead_points': dead,
        'log_evidence': lz,
        'sigma_log_z': sig,
        'sampling_s': float('nan'),
        'total_s': _runtime(d),
    })

out = pd.DataFrame(rows)
out = out.sort_values(['kind', 'waveform', 'priors', 'n_live']).reset_index(drop=True)
out.to_csv(OUT, index=False)
print(out.to_string(index=False))
print(f"\nWrote {OUT}  ({len(out)} rows)")
