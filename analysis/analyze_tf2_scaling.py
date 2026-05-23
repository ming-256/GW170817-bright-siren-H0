#!/usr/bin/env python3
"""Analyse Session 01 TaylorF2 scaling runs.

For each completed run under Results/test_suite/s01__gw170817__taylorf2__nlive*__seed0000/:
- load samples.csv and config.json
- extract log Z (and error) from the sampler log or the CSV metadata
- compute weighted H_0 MAP / median / 68% interval / P(H_0 > 120)
- write Results/test_suite/scaling_tf2_summary.csv
- print a stability check: are the summary statistics converged with n_live?
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (
    RESULTS_ROOT, load_catalog, load_run,
    weighted_map, weighted_median, weighted_quantiles, weighted_tail_prob,
    read_log_evidence_from_log,
)


def main() -> int:
    cat = load_catalog()
    s01 = cat[(cat["session"] == "01") & (cat["status"] == "done")].sort_values("n_live")
    if s01.empty:
        print("No Session 01 runs marked 'done' in the catalog.")
        return 0

    rows = []
    for _, cr in s01.iterrows():
        run = load_run(cr["run_id"])
        x = run.param("H_0")
        w = run.weights
        log_z, sigma = read_log_evidence_from_log(os.path.join(RESULTS_ROOT, cr["run_id"]))
        rows.append(
            {
                "run_id": cr["run_id"],
                "n_live": int(cr["n_live"]),
                "log_Z": log_z,
                "sigma_log_Z": sigma,
                "H0_MAP": weighted_map(x, w),
                "H0_median": weighted_median(x, w),
                "H0_q16": weighted_quantiles(x, w, [0.15865])[0],
                "H0_q84": weighted_quantiles(x, w, [0.84135])[0],
                "P_H0_gt_120": weighted_tail_prob(x, w, 120.0),
                "n_samples": len(run.samples),
            }
        )

    df = pd.DataFrame(rows).sort_values("n_live")
    out = os.path.join(RESULTS_ROOT, "scaling_tf2_summary.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(df.to_string(index=False))

    # Simple convergence check: report drift of medians between adjacent n_live rows.
    med = df["H0_median"].to_numpy()
    if len(med) >= 2:
        drift = max(abs(med[i] - med[i - 1]) for i in range(1, len(med)))
        print(f"\nMax H0-median drift between adjacent n_live rows: {drift:.3f} km/s/Mpc")
        print("If this is much larger than the within-run sampling variance, the operating point is under-sampled.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
