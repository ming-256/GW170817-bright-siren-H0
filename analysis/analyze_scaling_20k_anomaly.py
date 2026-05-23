#!/usr/bin/env python3
"""Analyse Session 11 — n_live=20000 anomaly diagnostic.

Compares dead-point count and log Z for the tight-tolerance rerun against the
existing scaling_summary.csv row. If the dead-point count increases to the
value predicted by linear extrapolation (~800k), early termination was the
cause. Output: stdout diagnosis + Results/test_suite/scaling_20k_anomaly.csv
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import RESULTS_ROOT, load_catalog, load_run, read_log_evidence_from_log, REPO_ROOT


def main() -> int:
    cat = load_catalog()
    s11 = cat[(cat["session"] == "11") & (cat["status"] == "done")]
    if s11.empty:
        print("No Session 11 rerun yet.")
        return 0

    scaling_csv = os.path.join(REPO_ROOT, "Results", "scaling_study", "scaling_summary.csv")
    scaling = pd.read_csv(scaling_csv)
    anomaly_row = scaling[scaling["n_live"] == 20000].iloc[0]

    run = load_run(s11.iloc[0]["run_id"])
    log_z, sigma = read_log_evidence_from_log(os.path.join(RESULTS_ROOT, s11.iloc[0]["run_id"]))

    n_dead = len(run.samples)
    print(f"Original n_live=20000 run: dead={int(anomaly_row['dead_points'])}, ln Z={anomaly_row['log_evidence']:.3f}")
    print(f"Tight-tol rerun:            dead={n_dead}, ln Z={log_z}")

    # Linear extrapolation prediction from neighbouring rows
    pred_dead = int((scaling[scaling["n_live"] == 10000]["dead_points"].iloc[0]
                     + scaling[scaling["n_live"] == 50000]["dead_points"].iloc[0]) / 2
                    * (20000 / 30000))
    print(f"Linear-extrapolated dead-point target (approx): {pred_dead}")
    if n_dead >= 0.9 * pred_dead:
        print("Diagnosis: early termination was the cause of the anomaly.")
    else:
        print("Diagnosis: dead-point count still below extrapolation; investigate sampler settings.")

    out = os.path.join(RESULTS_ROOT, "scaling_20k_anomaly.csv")
    pd.DataFrame([
        {"config": "original", "dead_points": int(anomaly_row["dead_points"]), "log_Z": float(anomaly_row["log_evidence"])},
        {"config": "tight_tol", "dead_points": n_dead, "log_Z": log_z},
    ]).to_csv(out, index=False)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
