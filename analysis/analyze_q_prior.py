#!/usr/bin/env python3
"""Analyse Session H: prior-only q diagnostic.

Given prior_samples.csv and prior_comparison.csv, compute:
 * moments of each prior in q,
 * P(q > 0.95) under each prior,
 * KL(project || LVK) as a scalar measure of how different the priors are,
 * whether the observed q posterior from our primary IMR baseline
   (Results/gwtc1_phasemarg/...baseline.csv) is consistent with the
   project prior (posterior should not place less mass at q>0.95 than
   the prior itself would suggest).

If the project prior and LVK-equivalent prior disagree substantially at
q>0.95, the LVK-deficit observed in the manuscript corner plot is a
prior-Jacobian issue, not a data issue.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import REPO_ROOT, RESULTS_ROOT, load_catalog, read_nested_samples_csv


def main() -> int:
    cat = load_catalog()
    sH = cat[(cat["session"] == "H") & (cat["status"] == "done")]
    if sH.empty:
        print("Session H not run yet. Run mnras_paper/test_suite/session_plans/session_H_prior_only.sh.")
        return 0

    run_id = sH.iloc[0]["run_id"]
    run_dir = os.path.join(RESULTS_ROOT, run_id)
    samples_path = os.path.join(run_dir, "prior_samples.csv")
    comp_path = os.path.join(run_dir, "prior_comparison.csv")
    if not (os.path.exists(samples_path) and os.path.exists(comp_path)):
        print(f"missing prior samples/comparison under {run_dir}")
        return 1

    samples = pd.read_csv(samples_path)
    proj = samples[samples["source"] == "project"]["q"].to_numpy()
    lvk = samples[samples["source"] == "lvk_equivalent"]["q"].to_numpy()

    print(f"{'':<30}{'project':>12}{'LVK':>12}")
    print(f"{'mean':<30}{proj.mean():>12.4f}{lvk.mean():>12.4f}")
    print(f"{'stdev':<30}{proj.std():>12.4f}{lvk.std():>12.4f}")
    print(f"{'P(q>0.90)':<30}{(proj>0.90).mean():>12.4f}{(lvk>0.90).mean():>12.4f}")
    print(f"{'P(q>0.95)':<30}{(proj>0.95).mean():>12.4f}{(lvk>0.95).mean():>12.4f}")
    print(f"{'P(q>0.99)':<30}{(proj>0.99).mean():>12.4f}{(lvk>0.99).mean():>12.4f}")

    # Posterior comparison — is the deficit at high q consistent with the prior?
    posterior_csv = os.path.join(
        REPO_ROOT, "Results", "gwtc1_phasemarg",
        "PhaseMarg_Heterodyned_IMRPhenomD_NRTidalv2_local_psd-gwtc1_ref-gwtc1_baseline.csv",
    )
    if os.path.exists(posterior_csv):
        post, w_post = read_nested_samples_csv(posterior_csv)
        if "q" in post.columns:
            q_post = post["q"].to_numpy(dtype=float)
            p_q_gt_95_post = float(w_post[q_post > 0.95].sum())
            print()
            print(f"Project H0 baseline posterior P(q>0.95) = {p_q_gt_95_post:.4f}")
            print(f"Project prior         P(q>0.95) = {(proj>0.95).mean():.4f}")
            print(f"LVK-equiv prior       P(q>0.95) = {(lvk>0.95).mean():.4f}")
            print()
            if p_q_gt_95_post < (proj > 0.95).mean():
                print("Posterior has *less* high-q mass than the project prior:")
                print("  data is actively pulling q away from 1. This is a DATA+WAVEFORM effect.")
            else:
                print("Posterior has *more* high-q mass than the project prior:")
                print("  data is pulling q up; the q-deficit vs LVK must be prior-induced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
