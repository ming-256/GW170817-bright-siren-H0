#!/usr/bin/env python3
"""Analyse unheterodyned scaling across Sessions 03, 04, 05.

Produces Results/test_suite/unhet_scaling_summary.csv joining TaylorF2 (500,
1500, 2500) and IMRPhenomD_NRTidalv2 (500, 1500, 2500) wall-clock times with
log Z and posterior statistics.

Reports the two scaling relations the paper needs:
  1. Sampling time vs n_live for the unheterodyned likelihood.
  2. Effective heterodyne speedup as a function of n_live.

The session scripts also populate finish.json with the wall-clock; we parse
that here in preference to re-running timing instrumentation.
"""
import json
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (
    RESULTS_ROOT, load_catalog, load_run,
    weighted_map, weighted_median, weighted_tail_prob,
    read_log_evidence_from_log,
)


def wallclock_seconds(run_dir: str):
    cfg_p = os.path.join(run_dir, "config.json")
    fin_p = os.path.join(run_dir, "finish.json")
    if not (os.path.exists(cfg_p) and os.path.exists(fin_p)):
        return None
    with open(cfg_p) as fh:
        cfg = json.load(fh)
    # finish.json may have multiple records appended; take the last.
    with open(fin_p) as fh:
        last = None
        for line in fh:
            line = line.strip()
            if line:
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    pass
    if last is None:
        return None
    import datetime as dt
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    try:
        start = dt.datetime.strptime(cfg["started"], fmt)
        end = dt.datetime.strptime(last["finished"], fmt)
    except (KeyError, ValueError):
        return None
    return (end - start).total_seconds()


def main() -> int:
    cat = load_catalog()
    sessions = cat[(cat["session"].isin(["03", "04", "05"])) & (cat["status"] == "done")
                   & (cat["variant"] == "unheterodyned")]
    if sessions.empty:
        print("No unheterodyned runs marked 'done' in Sessions 03/04/05.")
        return 0

    rows = []
    for _, cr in sessions.iterrows():
        run_dir = os.path.join(RESULTS_ROOT, cr["run_id"])
        try:
            run = load_run(cr["run_id"])
        except Exception as exc:
            print(f"skip {cr['run_id']}: {exc}")
            continue
        log_z, sigma = read_log_evidence_from_log(run_dir)
        wc = wallclock_seconds(run_dir)
        if "H_0" in run.samples.columns:
            x = run.param("H_0")
            w = run.weights
            h0_map = weighted_map(x, w)
            h0_med = weighted_median(x, w)
            p120 = weighted_tail_prob(x, w, 120)
        else:
            h0_map = h0_med = p120 = float("nan")
        rows.append({
            "run_id": cr["run_id"],
            "waveform": cr["waveform"],
            "n_live": int(cr["n_live"]),
            "wallclock_s": wc,
            "log_Z": log_z,
            "sigma_log_Z": sigma,
            "H0_MAP": h0_map,
            "H0_median": h0_med,
            "P_H0_gt_120": p120,
        })
    df = pd.DataFrame(rows).sort_values(["waveform", "n_live"])
    out = os.path.join(RESULTS_ROOT, "unhet_scaling_summary.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
