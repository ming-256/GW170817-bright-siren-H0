#!/usr/bin/env python3
"""Aggregate the M7 n_mcmc convergence sweep.

Looks at Results/test_suite/s21__gw170817__imrphenomxas_nrtidalv3__*__nmcmc*__seed0000/
and produces:
  - Results/test_suite/nmcmc_sweep_summary.csv
  - mnras_paper/figures/nmcmc_sweep.{pdf,png}

The sweep tests n_mcmc / NUM_DIMS in {5, 10, 20}*NUM_DIMS (=70, 140, 280 for
phase-marginalised 14-d GW170817) on both the IMRX volumetric baseline and
the direct uniform-in-d_L prior variant. Pass criterion (referee M7): the
headline tail probability P(H0>120) should be stable across the three step
counts to within run-to-run scatter.

The current code default is 8*NUM_DIMS = 112 (set in
GW170817/Scripts/GW170817_heterodyned_{1,2}.py); the paper text claims
5*NUM_DIMS = 70. If P(H0>120) is *not* stable across the sweep, the
manuscript's '5*n_dim slice steps' wording is the place to fix it.

No GPU, no sampler. Reads samples.csv + sampler.log only.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (
    REPO_ROOT, RESULTS_ROOT, load_run,
    weighted_median, weighted_tail_prob,
    read_log_evidence_from_log,
)


# ------------------------------------------------------------------ #
# Inline weighted helpers (HPD, MAP, n_eff)                          #
# ------------------------------------------------------------------ #
def hpd(x, w, frac=0.68):
    """Shortest-interval HPD from weighted samples."""
    idx = np.argsort(x)
    xs, ws = x[idx], w[idx] / w.sum()
    cdf = np.cumsum(ws)
    n = len(xs)
    best = (np.inf, float(xs[0]), float(xs[-1]))
    for i in range(n):
        target = cdf[i] + frac
        if target > 1.0:
            break
        j = int(np.searchsorted(cdf, target))
        if j >= n:
            break
        width = xs[j] - xs[i]
        if width < best[0]:
            best = (float(width), float(xs[i]), float(xs[j]))
    return best[1], best[2]


def n_eff_kish(w):
    s1, s2 = float(w.sum()), float((w * w).sum())
    return s1 * s1 / s2 if s2 > 0 else float("nan")


def map_from_hist(x, w, bin_width=1.0, lo=40.0, hi=230.0):
    edges = np.arange(lo, hi + bin_width, bin_width)
    counts, _ = np.histogram(x, bins=edges, weights=w)
    i = int(np.argmax(counts))
    return float(0.5 * (edges[i] + edges[i + 1]))


# ------------------------------------------------------------------ #
# Run-id parsing                                                     #
# ------------------------------------------------------------------ #
_RX_NMCMC = re.compile(r"__nmcmc(\d+)__")
_RX_VARIANT = re.compile(r"__(baseline|flatz)__")


def parse_nmcmc_and_variant(run_id):
    m = _RX_NMCMC.search(run_id)
    nmcmc = int(m.group(1)) if m else None
    m = _RX_VARIANT.search(run_id)
    variant = m.group(1) if m else "unknown"
    return nmcmc, variant


# ------------------------------------------------------------------ #
def main():
    pattern = os.path.join(
        RESULTS_ROOT,
        "s21__gw170817__imrphenomxas_nrtidalv3__*__nmcmc*__seed0000",
    )
    run_dirs = sorted(glob.glob(pattern))
    if not run_dirs:
        print(f"No M7 sweep runs found matching:\n  {pattern}")
        print("Launch the M7 block in Tier 2 (see referee_response_M7.md / "
              "DEFERRED_RUNS.md) before running this analyzer.")
        return 1

    rows = []
    for run_dir in run_dirs:
        run_id = os.path.basename(run_dir)
        nmcmc, variant = parse_nmcmc_and_variant(run_id)
        if nmcmc is None:
            print(f"  skipped (could not parse n_mcmc): {run_id}")
            continue
        run = load_run(run_id)
        x = run.param("H_0")
        w = run.weights
        log_z, sigma = read_log_evidence_from_log(run_dir)
        lo68, hi68 = hpd(x, w, 0.68)
        lo95, hi95 = hpd(x, w, 0.95)
        rows.append({
            "run_id": run_id,
            "variant": variant,
            "n_mcmc": nmcmc,
            "n_mcmc_per_dim": nmcmc / 14.0,  # GW170817 phase-marg is 14-d
            "H0_MAP": map_from_hist(x, w),
            "H0_median": weighted_median(x, w),
            "HPD68_lo": lo68, "HPD68_hi": hi68,
            "HPD95_lo": lo95, "HPD95_hi": hi95,
            "P_H0_gt_120": weighted_tail_prob(x, w, 120.0),
            "P_H0_gt_150": weighted_tail_prob(x, w, 150.0),
            "log_Z": log_z,
            "sigma_log_Z": sigma,
            "n_eff": n_eff_kish(w),
            "n_samples": len(w),
        })

    df = pd.DataFrame(rows).sort_values(["variant", "n_mcmc"]).reset_index(drop=True)

    out_csv = os.path.join(RESULTS_ROOT, "nmcmc_sweep_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")
    print(df.to_string(index=False))

    # ----- Figure -----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    colors = {"baseline": "#3a5fcd", "flatz": "#cd5c5c"}
    labels = {"baseline": r"Baseline ($\pi(d_L)\propto d_L^2$)",
              "flatz": r"Direct uniform-in-$d_L$"}
    for variant, g in df.groupby("variant"):
        c = colors.get(variant, "k")
        lbl = labels.get(variant, variant)
        axes[0].plot(g["n_mcmc"], g["P_H0_gt_120"], "-o", color=c, label=lbl, lw=2)
        if g["sigma_log_Z"].notna().all():
            axes[1].errorbar(g["n_mcmc"], g["log_Z"], yerr=g["sigma_log_Z"],
                             fmt="-o", color=c, label=lbl, lw=2, capsize=3)
        else:
            axes[1].plot(g["n_mcmc"], g["log_Z"], "-o", color=c, label=lbl, lw=2)

    axes[0].set_xlabel(r"$n_{\rm mcmc}$ (slice steps per update)")
    axes[0].set_ylabel(r"$P(H_0 > 120\,\mathrm{km\,s^{-1}\,Mpc^{-1}})$")
    axes[0].set_xscale("log")
    axes[0].legend(fontsize=9)
    axes[0].set_title("(a) Headline tail probability")

    axes[1].set_xlabel(r"$n_{\rm mcmc}$ (slice steps per update)")
    axes[1].set_ylabel(r"$\ln Z$")
    axes[1].set_xscale("log")
    axes[1].legend(fontsize=9)
    axes[1].set_title("(b) Log evidence")

    fig.suptitle("M7: GW170817 IMRX slice-step convergence")
    fig.tight_layout()

    fig_dir = os.path.join(REPO_ROOT, "mnras_paper", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(fig_dir, f"nmcmc_sweep.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)

    # ----- Pass-criterion print -----
    print("\nM7 pass criterion: P(H0>120) stable to within run-to-run scatter "
          "across the three step counts.")
    for variant, g in df.groupby("variant"):
        p_max, p_min = g["P_H0_gt_120"].max(), g["P_H0_gt_120"].min()
        print(f"  {variant:9s}: P(H0>120) range = [{p_min:.3f}, {p_max:.3f}], "
              f"max - min = {p_max - p_min:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
