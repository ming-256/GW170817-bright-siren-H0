#!/usr/bin/env python3
"""Analyse Session 09 heterodyne-bin sweep.

For n_bins ∈ {251, 501, 1001}, quantify how the posterior summary changes.
If posteriors agree to within statistical variance, 501 bins is justified
as the production setting and a single line in the paper can replace any
concern about under-resolution.

Output: Results/test_suite/het_bins_sweep_summary.csv
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (
    RESULTS_ROOT, load_catalog, load_run,
    weighted_median, weighted_tail_prob, weighted_wasserstein1,
    read_log_evidence_from_log,
)


def main() -> int:
    cat = load_catalog()
    s09 = cat[(cat["session"] == "09") & (cat["status"] == "done")].sort_values("n_bins")
    if s09.empty:
        print("No Session 09 runs marked 'done'.")
        return 0

    rows, runs = [], {}
    for _, cr in s09.iterrows():
        run = load_run(cr["run_id"])
        runs[int(cr["n_bins"])] = run
        x = run.param("H_0"); w = run.weights
        log_z, sigma = read_log_evidence_from_log(os.path.join(RESULTS_ROOT, cr["run_id"]))
        rows.append({
            "run_id": cr["run_id"],
            "n_bins": int(cr["n_bins"]),
            "log_Z": log_z, "sigma_log_Z": sigma,
            "H0_median": weighted_median(x, w),
            "P_H0_gt_120": weighted_tail_prob(x, w, 120.0),
        })
    df = pd.DataFrame(rows)

    # Pairwise W1 distances between bin counts.
    bins_sorted = sorted(runs.keys())
    w1_rows = []
    for i in range(len(bins_sorted)):
        for j in range(i + 1, len(bins_sorted)):
            a, b = bins_sorted[i], bins_sorted[j]
            ra, rb = runs[a], runs[b]
            wd = weighted_wasserstein1(ra.param("H_0"), ra.weights,
                                       rb.param("H_0"), rb.weights)
            w1_rows.append({"n_bins_a": a, "n_bins_b": b, "W1_H0": wd})
    w1 = pd.DataFrame(w1_rows)

    out = os.path.join(RESULTS_ROOT, "het_bins_sweep_summary.csv")
    df.to_csv(out, index=False)
    w1_out = os.path.join(RESULTS_ROOT, "het_bins_sweep_wasserstein.csv")
    w1.to_csv(w1_out, index=False)
    print(f"Wrote {out}\nWrote {w1_out}")
    print(df.to_string(index=False))
    print()
    print(w1.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
