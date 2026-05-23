#!/usr/bin/env python3
"""Aggregate the M2 seed-ensemble lnZ scatter for the IMR bimodality set.

Answers the referee's M2 complaint: the two-seed (s10 seed=0 + s18 seed=1)
bimodality 'replication' has unrestricted lnZ differing by 1.04, which is
inconsistent with the manuscript's quoted +/-0.1 per-run uncertainty. This
script ingests *all* available seeds for the three bimodality runs and
reports the empirical run-to-run lnZ scatter, then propagates that scatter
into the Mode-B / Mode-A Bayes factor with the ln(20/45) volume correction.

Runs discovered (glob, no catalog required):
  Results/test_suite/s*__gw170817__imrphenomd_nrtidalv2__flatz__dL30-75__refGWTC1__seed*  (Mode A)
  Results/test_suite/s*__gw170817__imrphenomd_nrtidalv2__flatz__dL10-30__refGWTC1__seed*  (Mode B)
  Results/test_suite/s*__gw170817__imrphenomd_nrtidalv2__flatz__dL10-75__refModeB__seed*  (unrestricted)

Outputs:
  - Results/test_suite/seed_ensemble_summary.csv          (per-run rows)
  - Results/test_suite/seed_ensemble_bayes_factor.csv     (per-seed lnB, ensemble stats)
  - mnras_paper/figures/seed_ensemble_lnZ.{pdf,png}       (lnZ scatter + lnB distribution)

The deliverable is the empirical sigma(lnZ) per mode, which replaces the
unsupported '+/-0.1 per-run' claim in section 5 of the manuscript.

No GPU, no sampler.
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
# Run-id parsing                                                     #
# ------------------------------------------------------------------ #
_RX_SEED = re.compile(r"__seed(\d+)$")
_RX_DLWIN = re.compile(r"__dL(\d+)-(\d+)__")
_RX_REF = re.compile(r"__ref(GWTC1|ModeB)__")
_RX_SESSION = re.compile(r"^s(\d+)__")


def parse_metadata(run_id):
    m = _RX_SEED.search(run_id)
    seed = int(m.group(1)) if m else None
    m = _RX_DLWIN.search(run_id)
    dl_lo = int(m.group(1)) if m else None
    dl_hi = int(m.group(2)) if m else None
    m = _RX_REF.search(run_id)
    ref = m.group(1) if m else None
    m = _RX_SESSION.match(run_id)
    session = int(m.group(1)) if m else None
    if dl_lo == 30 and dl_hi == 75:
        mode = "ModeA"
    elif dl_lo == 10 and dl_hi == 30:
        mode = "ModeB"
    elif dl_lo == 10 and dl_hi == 75:
        mode = "Unrestricted"
    else:
        mode = "unknown"
    return {"session": session, "seed": seed,
            "dl_lo": dl_lo, "dl_hi": dl_hi,
            "ref": ref, "mode": mode}


# ------------------------------------------------------------------ #
def collect_runs():
    patterns = [
        os.path.join(RESULTS_ROOT, "s*__gw170817__imrphenomd_nrtidalv2__flatz__dL30-75__refGWTC1__seed*"),
        os.path.join(RESULTS_ROOT, "s*__gw170817__imrphenomd_nrtidalv2__flatz__dL10-30__refGWTC1__seed*"),
        os.path.join(RESULTS_ROOT, "s*__gw170817__imrphenomd_nrtidalv2__flatz__dL10-75__refModeB__seed*"),
    ]
    run_dirs = []
    for pat in patterns:
        run_dirs.extend(sorted(glob.glob(pat)))
    # de-duplicate
    return sorted(set(run_dirs))


def main():
    run_dirs = collect_runs()
    if not run_dirs:
        print("No IMR bimodality runs found under Results/test_suite/.")
        return 1

    rows = []
    for run_dir in run_dirs:
        run_id = os.path.basename(run_dir)
        meta = parse_metadata(run_id)
        if meta["seed"] is None or meta["mode"] == "unknown":
            print(f"  skipped (cannot parse): {run_id}")
            continue
        run = load_run(run_id)
        x = run.param("H_0"); w = run.weights
        log_z, sigma = read_log_evidence_from_log(run_dir)
        rows.append({
            "run_id": run_id,
            **meta,
            "log_Z": log_z, "sigma_log_Z": sigma,
            "H0_median": weighted_median(x, w),
            "P_H0_gt_120": weighted_tail_prob(x, w, 120.0),
            "n_samples": len(w),
        })

    df = pd.DataFrame(rows).sort_values(["mode", "seed"]).reset_index(drop=True)
    out_csv = os.path.join(RESULTS_ROOT, "seed_ensemble_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print(df.to_string(index=False))

    # ----- Per-mode lnZ scatter -----
    print("\nEmpirical per-mode lnZ scatter across seeds:")
    mode_stats = []
    for mode, g in df.groupby("mode"):
        zs = g["log_Z"].dropna().values
        if len(zs) >= 2:
            mean = float(np.mean(zs)); std = float(np.std(zs, ddof=1))
        else:
            mean, std = float("nan"), float("nan")
        mode_stats.append({"mode": mode, "n_seeds": len(zs),
                           "mean_log_Z": mean, "std_log_Z": std})
        print(f"  {mode:13s}  n={len(zs):2d}  <lnZ>={mean:.3f}  sigma(lnZ)={std:.3f}")
    mode_stats_df = pd.DataFrame(mode_stats)

    # ----- Per-seed Bayes factor with ln(20/45) volume correction -----
    LN_VOL_RATIO = float(np.log(20.0 / 45.0))  # ln(width_B / width_A)
    print(f"\nVolume correction ln(20/45) = {LN_VOL_RATIO:.4f}")

    a = df[df["mode"] == "ModeA"].set_index("seed")["log_Z"]
    b = df[df["mode"] == "ModeB"].set_index("seed")["log_Z"]
    bf_rows = []
    for seed in sorted(set(a.index).intersection(b.index)):
        ln_b_minus_a = float(b[seed] - a[seed])
        ln_bf = ln_b_minus_a + LN_VOL_RATIO
        bf_rows.append({"seed": int(seed),
                        "logZ_A": float(a[seed]),
                        "logZ_B": float(b[seed]),
                        "lnZ_B_minus_A": ln_b_minus_a,
                        "lnBF_B_over_A": ln_bf})
    bf_df = pd.DataFrame(bf_rows)
    bf_csv = os.path.join(RESULTS_ROOT, "seed_ensemble_bayes_factor.csv")
    bf_df.to_csv(bf_csv, index=False)
    print(f"\nWrote {bf_csv}")
    print(bf_df.to_string(index=False))

    if len(bf_df) >= 2:
        bf_vals = bf_df["lnBF_B_over_A"].values
        print(f"\nlnBF(B/A) across seeds: mean={np.mean(bf_vals):.3f}  "
              f"sigma={np.std(bf_vals, ddof=1):.3f}  "
              f"range=[{bf_vals.min():.3f}, {bf_vals.max():.3f}]")
        within_unity = (np.abs(bf_vals) < 1.0).sum()
        print(f"  Seeds with |lnBF| < 1: {within_unity} / {len(bf_vals)}  "
              f"(M2 referee claim: 'not worth more than a mention' on Jeffreys scale)")

    # ----- Figure -----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    mode_colors = {"ModeA": "#cd5c5c", "ModeB": "#3a5fcd", "Unrestricted": "#2f8f3a"}

    for mode, g in df.groupby("mode"):
        axes[0].errorbar(g["seed"], g["log_Z"], yerr=g["sigma_log_Z"],
                         fmt="o", color=mode_colors.get(mode, "k"),
                         label=mode, lw=1.5, capsize=3, ms=6)
    axes[0].set_xlabel("seed")
    axes[0].set_ylabel(r"$\ln Z$ (within restricted prior)")
    axes[0].set_title("(a) Per-seed lnZ for the three bimodality runs")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    if len(bf_df) >= 1:
        axes[1].hist(bf_df["lnBF_B_over_A"], bins=max(6, len(bf_df) // 2),
                     edgecolor="black", color="#cccccc")
        axes[1].axvline(0.0, color="k", lw=1, ls="--", alpha=0.5)
        axes[1].axvline(-1.0, color="k", lw=0.5, ls=":", alpha=0.5)
        axes[1].axvline(+1.0, color="k", lw=0.5, ls=":", alpha=0.5)
        axes[1].set_xlabel(r"$\ln\mathcal{B}_{\rm B/A}$ (with $\ln(20/45)$ correction)")
        axes[1].set_ylabel("count")
        axes[1].set_title("(b) Per-seed Mode-B/Mode-A Bayes factor distribution")

    fig.suptitle("M2: GW170817 IMR/NRTidalv2 bimodality seed ensemble")
    fig.tight_layout()

    fig_dir = os.path.join(REPO_ROOT, "mnras_paper", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(fig_dir, f"seed_ensemble_lnZ.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
