#!/usr/bin/env python3
"""Referee concern M1: the H0-dependent GW-detection selection term N_s(H0).

Abbott et al. (2017), Nature 551, 85, write the bright-siren H0 likelihood with
a normalisation N_s(H0) that accounts for conditioning on GW detection (their
Eqs 11-12). With a prior on the *observable* luminosity distance d_L this term
is H0-independent and cancels; the referee asks us to show this for our
flat-in-redshift variant rather than assume it.

This script makes the argument quantitative. It does NOT require a
nested-sampling run -- it is a one-dimensional integral over a detection model.

Definitions
-----------
N_s(H0) = integral P_det(d_L) pi(d_L | H0) d d_L,

where P_det(d_L) is the orientation- and sky-averaged probability that a BNS at
luminosity distance d_L would clear the network detection threshold, and
pi(d_L | H0) is the d_L prior of the variant under test.

P_det is built from the Finn & Chernoff (1993, 1996) projection parameter
Theta in [0, 4]: a source at d_L with network horizon distance D_h is detected
iff Theta > 4 d_L / D_h, so P_det(d_L) = P(Theta > 4 d_L / D_h). A single-
detector Theta is used, which has the widest spread of any network combination
and therefore makes P_det fall off fastest with distance -- a conservative
choice that *maximises* any H0-dependence of N_s.

Two priors are evaluated:
  (a) as-implemented variant (i): uniform in d_L over a FIXED range [dL_lo,
      dL_hi]. The density carries no H0 dependence, so N_s is H0-independent
      by construction -- shown here for completeness.
  (b) a hypothetical genuinely flat-in-redshift prior: uniform in z, which
      in the low-z limit (z = H0 d_L / c) maps to a d_L window [c z_lo / H0,
      c z_hi / H0] whose edges DO move with H0. This is NOT what the pipeline
      runs; it is evaluated only to show the size of the term one would have
      to carry had the d_L window been made H0-dependent.

Output: Results/test_suite/selection_term_Ns.csv and a human-readable report.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import RESULTS_ROOT, REPO_ROOT

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy >=2.0 rename

C_KMS = 299792.458          # speed of light, km/s
H0_FID = 70.0               # fiducial H0 used to set the flat-in-z range, km/s/Mpc
DL_LO, DL_HI = 1.0, 75.0    # d_L prior bounds of the flat-in-z run (Mpc); as-run defaults
H0_GRID = np.linspace(45.0, 250.0, 206)   # sampled-H0 prior range


def theta_samples(n=4_000_000, seed=0):
    """Monte-Carlo the Finn & Chernoff projection parameter Theta in [0, 4].

    Theta = 2 [ Fp^2 (1+cos^2 i)^2 + 4 Fx^2 cos^2 i ]^(1/2), with Fp, Fx the
    single-detector antenna-pattern functions of sky angles (theta, phi, psi)
    and i the inclination. Theta = 4 for an overhead, face-on source.
    """
    rng = np.random.default_rng(seed)
    ct = rng.uniform(-1.0, 1.0, n)          # cos(sky polar angle)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    psi = rng.uniform(0.0, 2.0 * np.pi, n)
    ci = rng.uniform(-1.0, 1.0, n)          # cos(inclination)

    fp = 0.5 * (1.0 + ct**2) * np.cos(2 * phi) * np.cos(2 * psi) \
        - ct * np.sin(2 * phi) * np.sin(2 * psi)
    fx = 0.5 * (1.0 + ct**2) * np.cos(2 * phi) * np.sin(2 * psi) \
        + ct * np.sin(2 * phi) * np.cos(2 * psi)
    theta = 2.0 * np.sqrt(fp**2 * (1.0 + ci**2)**2 + 4.0 * fx**2 * ci**2)
    return theta


def make_pdet(theta, d_horizon):
    """Return P_det(d_L) = P(Theta > 4 d_L / D_h) as a vectorised callable."""
    theta_sorted = np.sort(theta)
    n = theta_sorted.size

    def pdet(d_l):
        d_l = np.atleast_1d(np.asarray(d_l, dtype=float))
        thr = np.clip(4.0 * d_l / d_horizon, 0.0, 4.0)
        idx = np.searchsorted(theta_sorted, thr, side="left")
        return 1.0 - idx / n     # fraction of samples with Theta > thr

    return pdet


def n_s_fixed_dL(pdet, dL_lo, dL_hi, npts=20001):
    """N_s for the as-implemented prior: uniform in d_L over a fixed range.

    The integrand has no H0 dependence, so this returns a single scalar that
    holds for every H0.
    """
    grid = np.linspace(dL_lo, dL_hi, npts)
    return _trapz(pdet(grid), grid) / (dL_hi - dL_lo)


def n_s_flat_in_z(pdet, h0, dL_lo, dL_hi, h0_fid=H0_FID, npts=20001):
    """N_s(H0) for a genuinely flat-in-redshift prior.

    Uniform in z over [z_lo, z_hi] (z fixed by the fiducial cosmology from the
    [dL_lo, dL_hi] range) maps, at sampled H0, to a uniform-in-d_L window
    [c z_lo / H0, c z_hi / H0]; z = H0 d_L / c in the low-z limit.
    """
    z_lo = h0_fid * dL_lo / C_KMS
    z_hi = h0_fid * dL_hi / C_KMS
    out = np.empty_like(h0, dtype=float)
    for k, h in enumerate(np.atleast_1d(h0)):
        d_lo, d_hi = C_KMS * z_lo / h, C_KMS * z_hi / h
        grid = np.linspace(d_lo, d_hi, npts)
        out[k] = _trapz(pdet(grid), grid) / (d_hi - d_lo)
    return out


def _make_plot(theta, horizons):
    """Two-panel figure: N_s(H0) for the as-run prior and a genuine flat-in-z."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.2, 3.6))

    for d_h, col in zip(horizons, ["C0", "C1", "C2"]):
        pdet = make_pdet(theta, d_h)
        ns_fixed = n_s_fixed_dL(pdet, DL_LO, DL_HI)
        ax0.axhline(ns_fixed, ls="--", lw=1.6, color=col,
                    label=rf"$D_h={d_h:g}$ Mpc")
        ns_z = n_s_flat_in_z(pdet, H0_GRID, DL_LO, DL_HI)
        ax1.plot(H0_GRID, ns_z / ns_z.max(), lw=1.8, color=col,
                 label=rf"$D_h={d_h:g}$ Mpc")

    ax0.set_title(r"(a) as run: uniform in $d_L$ "
                  r"$\Rightarrow$ $N_s$ constant")
    ax0.set_xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax0.set_ylabel(r"$N_s$")
    ax0.set_xlim(H0_GRID[0], H0_GRID[-1])
    ax0.set_ylim(0, 1)
    ax0.legend(fontsize=8, frameon=False)

    ax1.axvspan(120, H0_GRID[-1], color="0.88", lw=0,
                label=r"headline $H_0>120$")
    ax1.set_title(r"(b) genuine flat-in-$z$: $N_s(H_0)$ (normalised)")
    ax1.set_xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax1.set_ylabel(r"$N_s(H_0)\,/\,\max N_s$")
    ax1.set_xlim(H0_GRID[0], H0_GRID[-1])
    ax1.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig_dir = os.path.join(REPO_ROOT, "mnras_paper", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(fig_dir, f"selection_term_Ns.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def main():
    theta = theta_samples()
    print("=" * 72)
    print("M1 selection term  N_s(H0)  -- GW170817 flat-in-redshift variant")
    print("=" * 72)
    print(f"Theta Monte Carlo: n = {theta.size:,}, "
          f"Theta in [{theta.min():.3f}, {theta.max():.3f}], mean {theta.mean():.3f}")
    print(f"d_L prior range (as run): [{DL_LO:g}, {DL_HI:g}] Mpc")
    print(f"H0 prior range: [{H0_GRID[0]:g}, {H0_GRID[-1]:g}] km/s/Mpc\n")

    # O2-BNS network horizon: published O2 LIGO BNS range ~100 Mpc => horizon
    # ~226 Mpc; GW170817 (SNR 32.4 at ~40 Mpc, threshold 12) implies a network
    # horizon well above 100 Mpc. We span a deliberately pessimistic-to-
    # realistic bracket.
    horizons = [100.0, 150.0, 220.0]

    rows = []
    for d_h in horizons:
        pdet = make_pdet(theta, d_h)

        ns_fixed = n_s_fixed_dL(pdet, DL_LO, DL_HI)

        ns_z = n_s_flat_in_z(pdet, H0_GRID, DL_LO, DL_HI)
        ns_z_norm = ns_z / ns_z.max()       # normalise; only variation matters

        tail = H0_GRID >= 120.0
        tail150 = H0_GRID >= 150.0
        var_full = ns_z_norm.max() - ns_z_norm.min()
        var_tail = ns_z_norm[tail].max() - ns_z_norm[tail].min()
        var_t150 = ns_z_norm[tail150].max() - ns_z_norm[tail150].min()

        print(f"--- network horizon D_h = {d_h:g} Mpc ---")
        print(f"  (a) as-implemented uniform-in-d_L prior:")
        print(f"      N_s is H0-INDEPENDENT by construction; value = {ns_fixed:.4f}")
        print(f"  (b) hypothetical genuine flat-in-z prior:")
        print(f"      N_s(H0) total variation, full range [45,250]: "
              f"{100*var_full:.2f}%")
        print(f"      N_s(H0) variation, headline tail H0>120:       "
              f"{100*var_tail:.3f}%")
        print(f"      N_s(H0) variation, deep tail   H0>150:         "
              f"{100*var_t150:.3f}%")
        print(f"      (N_s rises with H0: had this prior been used, omitting\n"
              f"       N_s would INFLATE the high-H0 tail by up to this much --\n"
              f"       hence the as-run H0-independent d_L window matters.)\n")

        for h, raw, norm in zip(H0_GRID, ns_z, ns_z_norm):
            rows.append(dict(d_horizon_mpc=d_h, H0=h,
                             Ns_flatz_raw=raw, Ns_flatz_norm=norm,
                             Ns_uniform_dL=ns_fixed))

    out_csv = os.path.join(RESULTS_ROOT, "selection_term_Ns.csv")
    with open(out_csv, "w") as f:
        f.write("d_horizon_mpc,H0,Ns_flatz_raw,Ns_flatz_norm,Ns_uniform_dL\n")
        for r in rows:
            f.write(f"{r['d_horizon_mpc']:g},{r['H0']:.4f},"
                    f"{r['Ns_flatz_raw']:.6f},{r['Ns_flatz_norm']:.6f},"
                    f"{r['Ns_uniform_dL']:.6f}\n")
    print(f"Wrote {out_csv}")

    _make_plot(theta, horizons)

    print("\nConclusion")
    print("-" * 72)
    print("(a) RESOLVES M1. The flat-in-redshift variant as run imposes a")
    print("    prior on the OBSERVABLE d_L: pi(d_L|H0) is uniform over a fixed")
    print("    Mpc range and does NOT depend on H0 -- exactly as the volumetric")
    print("    baseline imposes a fixed pi(d_L) ~ d_L^2. The detection-selection")
    print("    integral N_s(H0) = integral P_det(d_L) pi(d_L|H0) d d_L then has")
    print("    no H0 anywhere: N_s is rigorously constant and cancels from")
    print("    p(H0|d). Eq. 2 is therefore correct as written and no 1/N_s(H0)")
    print("    correction is needed for this configuration.")
    print("(b) CONTEXT, not the run. Had the d_L window been made H0-dependent")
    print("    (a genuine flat-in-z prior), N_s(H0) would vary across the")
    print("    headline H0>120 tail by ~6% (realistic O2 horizon 220 Mpc) up to")
    print("    ~23% (pessimistic 100 Mpc), in the single-detector worst case;")
    print("    a 3-detector network (concentrated Theta) shrinks this further,")
    print("    consistent with Abbott et al. (2017)'s <~5% bound. This shows")
    print("    the H0-independent d_L window of the actual implementation is")
    print("    the property that makes N_s ignorable -- it is not assumed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
