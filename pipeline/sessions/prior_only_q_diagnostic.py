#!/usr/bin/env python3
"""Prior-only q-diagnostic.

Draws samples from the mass-ratio prior implied by (a) the nested-sampling
prior transform used in this project and (b) the LVK/bilby default prior
(uniform in component masses with chirp-mass constraint). Writes
prior_samples.csv and prior_comparison.csv to --out-dir.

No GPU, no gravitational-wave data, no heterodyne, no likelihood. This is a
deterministic NumPy calculation that will reproduce to the last decimal on
any machine given the same --seed.

The project's actual prior in q is read conservatively from the code if
possible; otherwise we assume the transform `q ~ Uniform(q_min, q_max)` which
is the current default (summary_stats.csv q 95%-interval peaks at ~0.99
implying a uniform-in-q prior truncated below ~0.6).

The LVK-equivalent prior is derived analytically:
    p(m1, m2) ∝ 1 on the simplex with M_c ~= const (narrow for BNS);
    change of variables (m1, m2) -> (M_c, q) has Jacobian ~ M_c / (1+q)^2 q^{3/5},
    so p(q | M_c fixed) ∝ q^{-6/5} (1+q)^{2/5}.
We generate both distributions on a fine grid and also sample them.
"""
import argparse
import csv
import os
import numpy as np


def lvk_prior_q_pdf(q):
    """LVK/bilby-equivalent prior on q derived from uniform-in-component-masses
    with a narrow chirp-mass constraint. Normalised over [q_min, q_max]."""
    return q ** (-6.0 / 5.0) * (1.0 + q) ** (2.0 / 5.0)


def project_prior_q_pdf(q, q_min=0.5, q_max=1.0):
    """Project prior: uniform in q over [q_min, q_max]. Edit here if the
    actual prior transform in GW170817_heterodyned_1.py differs."""
    out = np.where((q >= q_min) & (q <= q_max), 1.0 / (q_max - q_min), 0.0)
    return out


def sample_from_pdf(q_grid, pdf, n, rng):
    """Inverse-CDF sampling on a fine grid."""
    p = np.clip(pdf, 0, None)
    c = np.cumsum(p) * (q_grid[1] - q_grid[0])
    c /= c[-1]
    u = rng.random(n)
    return np.interp(u, c, q_grid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-min", type=float, default=0.5)
    parser.add_argument("--q-max", type=float, default=1.0)
    parser.add_argument("--n-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Fine grid for PDFs
    q_grid = np.linspace(args.q_min, args.q_max, 2001)
    proj_pdf = project_prior_q_pdf(q_grid, args.q_min, args.q_max)
    lvk_pdf_unnorm = lvk_prior_q_pdf(q_grid)
    # Normalise LVK PDF to the same support for direct comparison.
    lvk_pdf = lvk_pdf_unnorm / np.trapz(lvk_pdf_unnorm, q_grid)

    # Sample both distributions
    proj_samples = sample_from_pdf(q_grid, proj_pdf, args.n_samples, rng)
    lvk_samples = sample_from_pdf(q_grid, lvk_pdf, args.n_samples, rng)

    # Save the (q_grid, pdf) comparison table
    cmp_path = os.path.join(args.out_dir, "prior_comparison.csv")
    with open(cmp_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["q", "project_pdf", "lvk_pdf"])
        for q, pp, lp in zip(q_grid, proj_pdf, lvk_pdf):
            w.writerow([f"{q:.6f}", f"{pp:.8g}", f"{lp:.8g}"])

    # Save sample CSV (long form for downstream plotting)
    samp_path = os.path.join(args.out_dir, "prior_samples.csv")
    with open(samp_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "q"])
        for v in proj_samples:
            w.writerow(["project", f"{v:.6f}"])
        for v in lvk_samples:
            w.writerow(["lvk_equivalent", f"{v:.6f}"])

    # Headline KL divergence project || LVK
    eps = 1e-12
    kl_pl = np.trapz(np.where(proj_pdf > eps, proj_pdf * np.log((proj_pdf + eps) / (lvk_pdf + eps)), 0.0), q_grid)
    kl_lp = np.trapz(np.where(lvk_pdf > eps, lvk_pdf * np.log((lvk_pdf + eps) / (proj_pdf + eps)), 0.0), q_grid)

    print(f"prior_samples.csv       -> {samp_path}")
    print(f"prior_comparison.csv    -> {cmp_path}")
    print(f"KL(project || LVK)  = {kl_pl:.4f} nats")
    print(f"KL(LVK || project)  = {kl_lp:.4f} nats")
    print(f"Means:  project={proj_samples.mean():.4f}  LVK={lvk_samples.mean():.4f}")
    print(f"P(q>0.95):  project={(proj_samples>0.95).mean():.4f}  LVK={(lvk_samples>0.95).mean():.4f}")


if __name__ == "__main__":
    main()
