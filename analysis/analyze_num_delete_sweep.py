#!/usr/bin/env python3
"""Analyse Session 08 num_delete sweep at fixed n_live=5000.

Answers: at what fraction num_delete/n_live does GPU throughput saturate,
and does the posterior summary (H0 median, P(H0>120), log Z) depend on
num_delete once the sampler is in the statistical regime?

Output: Results/test_suite/num_delete_sweep_summary.csv
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (
    RESULTS_ROOT, load_catalog, load_run,
    weighted_median, weighted_tail_prob, read_log_evidence_from_log,
)


def main() -> int:
    cat = load_catalog()
    s08 = cat[(cat["session"] == "08") & (cat["status"] == "done")].sort_values("num_delete")
    if s08.empty:
        print("No Session 08 runs marked 'done'.")
        return 0

    rows = []
    for _, cr in s08.iterrows():
        run = load_run(cr["run_id"])
        x = run.param("H_0")
        w = run.weights
        log_z, sigma = read_log_evidence_from_log(os.path.join(RESULTS_ROOT, cr["run_id"]))
        rows.append({
            "run_id": cr["run_id"],
            "n_live": int(cr["n_live"]),
            "num_delete": int(cr["num_delete"]),
            "frac": float(cr["num_delete"]) / float(cr["n_live"]),
            "log_Z": log_z, "sigma_log_Z": sigma,
            "H0_median": weighted_median(x, w),
            "P_H0_gt_120": weighted_tail_prob(x, w, 120.0),
        })
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_ROOT, "num_delete_sweep_summary.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
