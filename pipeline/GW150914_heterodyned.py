"""
Heterodyned Nested Sampling for GW150914
==========================================

Performs Bayesian inference on the binary black hole event GW150914 using:
  - Waveform: IMRPhenomD (BBH, no tidal corrections)
  - Likelihood: Heterodyned (relative binning), optionally phase-marginalized
  - Sampler: Blackjax nested slice sampling with periodic boundary wrapping

GW150914 was the first direct detection of gravitational waves from a BBH
merger, observed by LIGO Hanford (H1) and LIGO Livingston (L1) on
September 14, 2015 during the first observing run (O1).

References:
  - Discovery: Abbott et al. (2016), arXiv:1602.03837
  - GWTC-1 catalog: Abbott et al. (2019), arXiv:1811.12907

With --phase-marginalization:
  10D parameter space, phase_c analytically marginalized via log I_0(|<d|h>|)
  (jimgw: HeterodynedPhaseMarginalizedLikelihoodFD pattern)

Without --phase-marginalization:
  11D parameter space, phase_c sampled as uniform [0, 2pi]
  (jimgw: HeterodynedTransientLikelihoodFD pattern)

Parameter space (BBH, no tidal, no standard siren):
  M_c, q, s1_z, s2_z, iota, d_L, t_c, psi, ra, dec [, phase_c]

Usage:
  python GW150914_heterodyned.py [--waveform {IMRPhenomD}]
                                  [--data-source {fetch,local}]
                                  [--psd-source {self,gwtc1}]
                                  [--ref-params {gwtc1,optimize}]
                                  [--phase-marginalization]
                                  [--output-dir DIR]
"""

# ============================================================================
# 1. IMPORTS & JAX CONFIGURATION
# ============================================================================
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import argparse
import jax
jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np
import blackjax
import h5py
import time
import tqdm
from astropy.time import Time
from scipy.interpolate import interp1d
from anesthetic import NestedSamples
from blackjax.ns.utils import finalise

from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.waveform import RippleIMRPhenomD, RippleIMRPhenomXPHM
from jimgw.core.single_event.data import Data, PowerSpectrum
from gwpy.timeseries import TimeSeries

# ============================================================================
# 0. COMMAND-LINE ARGUMENTS
# ============================================================================
parser = argparse.ArgumentParser(description='Heterodyned nested sampling for GW150914')
parser.add_argument('--waveform', choices=['IMRPhenomD', 'IMRPhenomXPHM'],
                    default='IMRPhenomD',
                    help='Waveform approximant. IMRPhenomD = aligned-spin BBH (10/11D); '
                         'IMRPhenomXPHM = precessing + higher modes (LVK GWTC-2.1+ production), '
                         '14/15D parameter set with spherical-coord sampling of in-plane spin.')
parser.add_argument('--data-source', choices=['fetch', 'local'],
                    default='fetch',
                    help='Data source: "fetch" pulls from GWOSC via gwpy (requires internet), '
                         '"local" reads HDF5 files from $GWOSC_GW150914_DIR/ (default data/GWOSC/GW150914/)')
parser.add_argument('--psd-source', choices=['self', 'gwtc2p1'],
                    default='self',
                    help='PSD source: "self" (estimated from data via gwpy Welch), '
                         '"gwtc2p1" (PSDs from GWTC-2p1 PE data release HDF5)')
parser.add_argument('--ref-params', choices=['gwtc1', 'optimize'],
                    default='gwtc1',
                    help='Reference parameters: "gwtc1" loads median from GWTC-2p1 posteriors '
                         '($GWOSC_GW150914_DIR/), "optimize" finds the maximum-likelihood '
                         'point via Adam optimization of the phase-marginalized likelihood')
parser.add_argument('--phase-marginalization', action='store_true',
                    help='Enable analytic phase marginalization (removes phase_c from sampling)')
parser.add_argument('--output-dir', default='results/gwtc1_phasemarg',
                    help='Directory to write output CSV files (default: Results)')
parser.add_argument('--n-live', type=int, default=5000,
                    help='Number of live points (default: 5000)')
parser.add_argument('--m-comp-lo', type=float, default=None,
                    help='Lower bound on component masses (M_sun). Default: script-internal value (1.0).')
parser.add_argument('--m-comp-hi', type=float, default=None,
                    help='Upper bound on component masses (M_sun). Default: script-internal value (100.0).')
parser.add_argument('--n-mcmc-steps', type=int, default=None,
                    help='Number of MCMC steps per nested sampling iteration. '
                         'Default: NUM_DIMS * 8 (112 for precessing, 88 for aligned-spin).')
args = parser.parse_args()
waveform_tag = args.waveform
data_source = args.data_source
psd_source = args.psd_source
ref_params_source = args.ref_params
phase_marg = args.phase_marginalization
output_dir = args.output_dir

# Numerically stable log I_0 for phase marginalization:
# log(I_0(x)) = log(i0e(x)) + x, where i0e(x) = exp(-|x|) * I_0(x)
from jax.scipy.special import i0e

@jax.jit
def log_i0(x):
    return jnp.log(i0e(x)) + x


# ============================================================================
# 2. PARAMETER CONFIGURATION (static arrays, no dicts in hot path)
# ============================================================================
# Aligned-spin (IMRPhenomD): 10 dims (phase-marg) / 11 dims (no marg).
# Precessing (IMRPhenomXPHM): 14 dims (phase-marg) / 15 dims (no marg).
# In the precessing branch, the in-plane spin is sampled in *spherical
# coordinates* (a, cos(tilt), phi) per body — uniform marginals on each give
# the LVK convention (uniform spin magnitude × isotropic tilt × uniform
# azimuth), without Jacobian gymnastics in the prior. Conversion to the
# Cartesian (s_x, s_y, s_z) the waveform expects is done inside the
# likelihood. The on-disk CSV columns therefore expose a_1, cost_1, phi_1,
# a_2, cost_2, phi_2 (spherical) for the precessing run.
precessing = (waveform_tag == 'IMRPhenomXPHM')

# Parameter names (used only at boundaries: init, output, jimgw API calls)
if precessing:
    PARAM_NAMES = [
        "M_c", "q",
        "a_1", "cost_1", "phi_1", "a_2", "cost_2", "phi_2",
        "iota", "d_L", "t_c",
        "psi", "ra", "dec",
    ]
    PARAM_LABELS = [
        r"$M_c$", r"$q$",
        r"$a_1$", r"$\cos\theta_1$", r"$\phi_1$",
        r"$a_2$", r"$\cos\theta_2$", r"$\phi_2$",
        r"$\iota$", r"$d_L$", r"$t_c$",
        r"$\psi$", r"$\alpha$", r"$\delta$",
    ]
    I_MC, I_Q = 0, 1
    I_A1, I_COST1, I_PHI1 = 2, 3, 4
    I_A2, I_COST2, I_PHI2 = 5, 6, 7
    I_IOTA, I_DL, I_TC = 8, 9, 10
    I_PSI, I_RA, I_DEC = 11, 12, 13
    A_MAX = 0.99   # high-spin BBH magnitude cap (LVK GWTC-2.1 convention)
    _PRIOR_LO_BASE = [
        10.0, 0.125,                             # M_c, q
        0.0, -1.0, 0.0, 0.0, -1.0, 0.0,           # a_1, cost_1, phi_1, a_2, cost_2, phi_2
        0.0, 1.0, -0.05,                          # iota, d_L, t_c
        0.0, 0.0, -jnp.pi / 2,                    # psi, ra, dec
    ]
    _PRIOR_HI_BASE = [
        80.0, 1.00,                               # M_c, q
        A_MAX, 1.0, 2 * jnp.pi, A_MAX, 1.0, 2 * jnp.pi,
        jnp.pi, 2000.0, 0.05,                     # iota, d_L, t_c
        jnp.pi, 2 * jnp.pi, jnp.pi / 2,           # psi, ra, dec
    ]
    # All spin coords are uniform (type 0); only iota (sin), d_L (power-law),
    # dec (cos) carry non-uniform priors here.
    _PRIOR_TYPE_BASE = [0, 0,  0, 0, 0, 0, 0, 0,  1, 5, 0,  0, 0, 2]
else:
    # BBH event, aligned-spin: no tidal parameters, no standard siren terms.
    PARAM_NAMES = [
        "M_c", "q", "s1_z", "s2_z", "iota", "d_L", "t_c",
        "psi", "ra", "dec",
    ]
    PARAM_LABELS = [
        r"$M_c$", r"$q$", r"$s_{1z}$", r"$s_{2z}$", r"$\iota$", r"$d_L$", r"$t_c$",
        r"$\psi$", r"$\alpha$", r"$\delta$",
    ]
    # Static parameter indices (compile-time constants for array access)
    I_MC, I_Q, I_S1Z, I_S2Z, I_IOTA, I_DL, I_TC = 0, 1, 2, 3, 4, 5, 6
    I_PSI, I_RA, I_DEC = 7, 8, 9

    # Prior bounds: BBH parameter ranges
    # M_c in [10, 80] M_sun (detector-frame), full spin range [-1, 1]
    # d_L in [1, 2000] Mpc with power-law prior p(d_L) ∝ d_L^2
    # t_c in [-0.05, 0.05] s (tighter than BNS, short BBH signal)
    _PRIOR_LO_BASE = [
        10.0, 0.125, -1.0, -1.0,                 # M_c, q, s1_z, s2_z
        0.0, 1.0, -0.05,                          # iota, d_L, t_c
        0.0, 0.0, -jnp.pi / 2,                    # psi, ra, dec
    ]
    _PRIOR_HI_BASE = [
        80.0, 1.00, 1.0, 1.0,                    # M_c, q, s1_z, s2_z
        jnp.pi, 2000.0, 0.05,                     # iota, d_L, t_c
        jnp.pi, 2 * jnp.pi, jnp.pi / 2,          # psi, ra, dec
    ]
    # Prior type encoding: 0=uniform, 1=sin(iota), 2=cos(dec), 5=power-law(d_L)
    _PRIOR_TYPE_BASE = [0, 0, 0, 0, 1, 5, 0, 0, 0, 2]

# When phase_c is NOT marginalized, append it as one extra uniform [0, 2pi] dim.
if not phase_marg:
    PARAM_NAMES.append("phase_c")
    PARAM_LABELS.append(r"$\phi_c$")
    I_PHASEC = len(PARAM_NAMES) - 1
    _PRIOR_LO_BASE.append(0.0)
    _PRIOR_HI_BASE.append(float(2 * jnp.pi))
    _PRIOR_TYPE_BASE.append(0)  # uniform

NUM_DIMS = len(PARAM_NAMES)

PRIOR_LO = jnp.array(_PRIOR_LO_BASE)
PRIOR_HI = jnp.array(_PRIOR_HI_BASE)

# Component mass bounds (applied as hard cut in M_c-q space).
# Override via --m-comp-lo/--m-comp-hi (e.g. LVK GWTC-2.1 BBH bounds [5, 100]).
M_COMP_LO = args.m_comp_lo if args.m_comp_lo is not None else 1.0     # M_sun
M_COMP_HI = args.m_comp_hi if args.m_comp_hi is not None else 100.0   # M_sun

# Prior type encoding: 0=uniform, 1=sin(iota), 2=cos(dec), 5=power-law(d_L)
PRIOR_TYPE = jnp.array(_PRIOR_TYPE_BASE)

# Pre-computed prior constants (avoid recomputation in JIT)
_PRIOR_RANGE = PRIOR_HI - PRIOR_LO
_PRIOR_LOG_RANGE = jnp.log(_PRIOR_RANGE)

# Power-law d_L prior: p(d_L) = d_L^2 / ((hi^3 - lo^3)/3)
# log p(d_L) = 2*log(d_L) - log((hi^3 - lo^3)/3)
_DL_LO = 1.0
_DL_HI = 2000.0
_DL_LOG_NORM = jnp.log((_DL_HI**3 - _DL_LO**3) / 3.0)


# ============================================================================
# 3. VECTORIZED LOG-PRIOR (no Python loops, fully JIT-traced)
# ============================================================================

@jax.jit
def logprior_fn(x):
    """Evaluate total log-prior for a flat parameter vector.

    Computes all prior types vectorially and selects via jnp.where.
    Includes hard cut on component masses: m1, m2 in [M_COMP_LO, M_COMP_HI].
    No Python loops, list comprehensions, or dict access.
    """
    in_bounds = (x >= PRIOR_LO) & (x <= PRIOR_HI)

    # Uniform: log(1/(hi-lo)) = -log(hi-lo)
    lp_uniform = jnp.where(in_bounds, -_PRIOR_LOG_RANGE, -jnp.inf)

    # Sin prior (iota): log(sin(x)/2) on [0, pi]
    lp_sin = jnp.where(in_bounds, jnp.log(jnp.abs(jnp.sin(x)) + 1e-300) - jnp.log(2.0), -jnp.inf)

    # Cos prior (dec): log(cos(x)/2) on [-pi/2, pi/2]
    lp_cos = jnp.where(in_bounds, jnp.log(jnp.abs(jnp.cos(x)) + 1e-300) - jnp.log(2.0), -jnp.inf)

    # Power-law prior (d_L): log(d_L^2 / ((hi^3 - lo^3)/3))
    lp_powerlaw = jnp.where(in_bounds, 2.0 * jnp.log(jnp.abs(x) + 1e-300) - _DL_LOG_NORM, -jnp.inf)

    # Select per-parameter prior using type index
    lp = jnp.where(PRIOR_TYPE == 0, lp_uniform,
         jnp.where(PRIOR_TYPE == 1, lp_sin,
         jnp.where(PRIOR_TYPE == 2, lp_cos,
         jnp.where(PRIOR_TYPE == 5, lp_powerlaw,
                    lp_uniform))))  # fallback (should not be reached)

    total = jnp.sum(lp)

    # Component mass constraint: m1, m2 must be in [M_COMP_LO, M_COMP_HI]
    # From M_c, q: eta = q/(1+q)^2, M_total = M_c / eta^(3/5), m1 = M_total/(1+q), m2 = q*m1
    q = x[I_Q]
    eta = q / (1 + q) ** 2
    M_total = x[I_MC] / eta ** 0.6
    m1 = M_total / (1 + q)
    m2 = q * m1
    mass_ok = (m1 >= M_COMP_LO) & (m1 <= M_COMP_HI) & (m2 >= M_COMP_LO) & (m2 <= M_COMP_HI)
    total = jnp.where(mass_ok, total, -jnp.inf)

    # Jacobian |∂(m1,m2)/∂(M_c,q)| = M_c * (1+q)^(2/5) / q^(6/5)
    # Converts uniform-in-(M_c,q) to uniform-in-(m1,m2)
    log_jacobian = jnp.log(x[I_MC]) - 1.2 * jnp.log(x[I_Q]) + 0.4 * jnp.log(1.0 + x[I_Q])
    total = total + log_jacobian

    return total


# ============================================================================
# 4. EVENT CONFIGURATION & DETECTOR DATA
# ============================================================================

gps = 1126259462.4
fmin = 20.0
fmax = 1024.0
duration = 8        # 8s coherent analysis (Abbott et al. 2016, PRL 116, 241102, Sec. II.B)
post_trigger_duration = 2
roll_off = 0.4
tukey_alpha = 2 * roll_off / duration
psd_pad = 16
psd_duration = 1024

marg_tag = 'PhaseMarg' if phase_marg else 'NoMarg'
os.makedirs(output_dir, exist_ok=True)
label = f'{output_dir}/GW15_{marg_tag}_Heterodyned_{waveform_tag}_{data_source}_psd-{psd_source}_ref-{ref_params_source}'

# Analysis segment: [gps - (duration - post_trigger), gps + post_trigger]
start = gps - (duration - post_trigger_duration)
end = gps + post_trigger_duration

# PSD segment: immediately before the analysis segment
psd_start = start - psd_pad - psd_duration
psd_end = start - psd_pad

t0 = time.time()

# Local GWOSC HDF5 file mapping: ifo name -> file path
GWOSC_LOCAL_DIR = os.environ.get('GWOSC_GW150914_DIR', 'data/GWOSC/GW150914')
# LVK GWTC-2.1 GW150914 PE data release (287 MB); see docs/reproducibility.md
# for the curl that fetches it from Zenodo 10.5281/zenodo.6513631.
GWTC2P1_GW150914_HDF5 = os.environ.get(
    'GWTC2P1_GW150914_HDF5',
    'results/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_nocosmo.h5')
GWOSC_LOCAL_FILES = {
    'H1': os.path.join(GWOSC_LOCAL_DIR, 'H-H1_LOSC_4_V1-1126256640-4096.hdf5'),
    'L1': os.path.join(GWOSC_LOCAL_DIR, 'L-L1_LOSC_4_V1-1126256640-4096.hdf5'),
}

def load_gwosc_local(ifo_name, gps_start, gps_end):
    """Load GWOSC strain from a local HDF5 file, slicing to [gps_start, gps_end]."""
    path = GWOSC_LOCAL_FILES[ifo_name]
    ts = TimeSeries.read(path, format='hdf5.gwosc')
    ts = ts.crop(gps_start, gps_end)
    return Data(ts.value, ts.dt.value, ts.epoch.value, ifo_name)

def load_gwosc_local_gwpy(ifo_name, gps_start, gps_end):
    """Load GWOSC strain as a gwpy TimeSeries (for PSD estimation)."""
    path = GWOSC_LOCAL_FILES[ifo_name]
    ts = TimeSeries.read(path, format='hdf5.gwosc')
    ts = ts.crop(gps_start, gps_end)
    return ts

# PSD estimation config (matching bilby/kazewong):
#   - 32s Tukey-windowed segments, 50% overlap, median averaging
PSD_FFT_LENGTH = 32  # seconds per FFT segment
PSD_OVERLAP_FRAC = 0.5
PSD_METHOD = 'median'

GWOSC_PSD_DIR = GWOSC_LOCAL_DIR


GWTC2P1_HDF5 = os.path.join(GWOSC_PSD_DIR,
    'IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_nocosmo.h5')

def load_external_psd(psd_src, ifo_name, target_freqs):
    """Load PSD from the GWTC-2p1 PE data release and interpolate to target frequency grid."""
    if psd_src == 'gwtc2p1':
        with h5py.File(GWTC2P1_HDF5, 'r') as f:
            psd_data = f[f'C01:IMRPhenomXPHM/psds/{ifo_name}'][:]
        freqs_psd = psd_data[:, 0]
        psd_vals = psd_data[:, 1]
    else:
        raise ValueError(f"Unknown PSD source: {psd_src}")
    _PSD_INF_SENTINEL = 1e300
    inf_mask_src = ~np.isfinite(psd_vals)
    psd_vals_safe = np.where(inf_mask_src, _PSD_INF_SENTINEL, psd_vals)
    psd_interp = interp1d(freqs_psd, psd_vals_safe, kind='linear',
                          fill_value=_PSD_INF_SENTINEL, bounds_error=False)
    psd_values = psd_interp(np.array(target_freqs))
    psd_values = np.where(psd_values >= _PSD_INF_SENTINEL * 0.5, np.inf, psd_values)
    return PowerSpectrum(
        values=jnp.array(psd_values),
        frequencies=jnp.array(target_freqs),
        name=ifo_name,
    )


detectors = [get_H1(), get_L1()]
N_DET = len(detectors)
print(f"Data source: {data_source}, PSD source: {psd_source}")

for ifo in detectors:
    t_det = time.time()

    if data_source == 'local':
        # Load from local GWOSC HDF5 files (no internet required)
        strain_data = load_gwosc_local(ifo.name, start, end)
        if psd_source == 'self':
            psd_ts = load_gwosc_local_gwpy(ifo.name, psd_start, psd_end)
    else:
        # Fetch from GWOSC via gwpy (no version needed for O1 data)
        strain_data = Data.from_gwosc(ifo.name, start, end)
        if psd_source == 'self':
            psd_ts = TimeSeries.fetch_open_data(ifo.name, psd_start, psd_end)
    t_fetch = time.time() - t_det

    strain_data.set_tukey_window(alpha=tukey_alpha)
    strain_data.fft()
    ifo.set_data(strain_data)

    # PSD loading
    t_psd0 = time.time()
    if psd_source == 'self':
        # PSD via gwpy: median-averaged, Tukey-windowed Welch (matching bilby)
        psd_alpha = 2 * roll_off / PSD_FFT_LENGTH
        psd_gwpy = psd_ts.psd(
            fftlength=PSD_FFT_LENGTH,
            overlap=PSD_FFT_LENGTH * PSD_OVERLAP_FRAC,
            window=('tukey', psd_alpha),
            method=PSD_METHOD,
        )
        # Interpolate PSD to strain frequency grid (PSD df may differ from strain df)
        psd_interp_fn = interp1d(
            psd_gwpy.frequencies.value, psd_gwpy.value,
            kind='linear', fill_value=(psd_gwpy.value[0], psd_gwpy.value[-1]),
            bounds_error=False,
        )
        strain_freqs = np.array(strain_data.frequencies)
        psd_obj = PowerSpectrum(
            values=jnp.array(psd_interp_fn(strain_freqs)),
            frequencies=jnp.array(strain_freqs),
            name=ifo.name,
        )
    else:
        strain_freqs = np.array(strain_data.frequencies)
        psd_obj = load_external_psd(psd_source, ifo.name, strain_freqs)
    ifo.set_psd(psd_obj)
    t_psd = time.time() - t_psd0

    # Set frequency bounds for the analysis band
    ifo.set_frequency_bounds(fmin, fmax)
    print(f"  {ifo.name}: data={t_fetch:.1f}s, PSD({psd_source})={t_psd:.1f}s, total={time.time()-t_det:.1f}s")

t_data = time.time() - t0
print(f"[TIMING] Data loading: {t_data:.1f}s")

H1, L1 = detectors

WAVEFORM_MAP = {
    'IMRPhenomD':    RippleIMRPhenomD,
    'IMRPhenomXPHM': RippleIMRPhenomXPHM,
}
waveform = WAVEFORM_MAP[waveform_tag](f_ref=20.0)
print(f"Waveform: {waveform_tag} ({waveform})")

frequencies = H1.sliced_frequencies
epoch = duration - post_trigger_duration
gmst = Time(gps, format="gps").sidereal_time("apparent", "greenwich").rad


# ============================================================================
# 5. REFERENCE PARAMETERS (from GWTC-1 posteriors or optimization)
# ============================================================================

def load_reference_params(hdf5_path, dataset='C01:IMRPhenomXPHM/posterior_samples'):
    """Load GWTC-2p1 posterior samples and compute median reference parameters.

    Uses the IGWN GWTC-2p1 PE data release format (pesummary convention):
      chirp_mass, mass_ratio, a_1, tilt_1, a_2, tilt_2, luminosity_distance,
      iota, ra, dec, ...
    """
    with h5py.File(hdf5_path, 'r') as f:
        data = f[dataset][:]

    Mc = np.median(data['chirp_mass'])       # detector-frame chirp mass
    q_val = np.median(data['mass_ratio'])    # q = m2/m1 <= 1
    eta_val = q_val / (1 + q_val) ** 2

    s1_z = np.median(data['spin_1z'])
    s2_z = np.median(data['spin_2z'])

    ref = {
        'M_c': float(Mc), 'q': float(q_val), 'eta': float(eta_val),
        's1_z': float(s1_z), 's2_z': float(s2_z),
        'd_L': float(np.median(data['luminosity_distance'])),
        'iota': float(np.median(data['iota'])),
        'ra': float(np.median(data['ra'])),
        'dec': float(np.median(data['dec'])),
        't_c': 0.0, 'phase_c': 0.0, 'psi': 0.0,
        'trigger_time': float(gps),
        'gmst': float(gmst),
    }
    # For precessing waveforms, also pull the in-plane spin medians from the
    # GWTC-2.1 IMRPhenomXPHM posterior. RippleIMRPhenomXPHM expects the full
    # Cartesian spin vector (s1_x, s1_y, s1_z, s2_x, s2_y, s2_z).
    if precessing:
        _spin_map = {'spin_1x': 's1_x', 'spin_1y': 's1_y', 'spin_2x': 's2_x', 'spin_2y': 's2_y'}
        for hdf_key, param_key in _spin_map.items():
            if hdf_key in data.dtype.names:
                ref[param_key] = float(np.median(data[hdf_key]))
            else:
                ref[param_key] = 0.0  # fall back to aligned spin
    if np.isclose(ref['eta'], 0.25):
        ref['eta'] = 0.249995
    return ref


def optimize_reference_params(detectors, waveform, frequencies, popsize=100, n_steps=1500,
                              learning_rate=3e-3, noise_level=0.5, freq_downsample=8):
    """Find reference parameters by maximizing the phase-marginalized likelihood.

    Uses optax Adam with cosine-decayed learning rate and multiplicative
    gradient noise, vmapped over ``popsize`` independent walkers drawn
    uniformly in the prior volume.  The optimization loop is a single
    ``jax.lax.scan`` — JIT compiles ONE step and iterates, rather than
    unrolling ``n_steps`` Python iterations (the main speedup over flowMC's
    AdamOptimization).

    Args:
        detectors: List of detector objects with loaded data and PSDs.
        waveform: Waveform model (RippleIMRPhenomD).
        frequencies: Frequency array for likelihood evaluation.
        popsize: Number of parallel walkers (default: 100).
        n_steps: Adam steps per walker (default: 1500).
        learning_rate: Peak Adam learning rate (default: 3e-3).
        noise_level: Multiplicative gradient noise scale (default: 0.5).
        freq_downsample: Frequency grid downsample factor (default: 8).

    Returns:
        dict: Best-fit reference parameters for heterodyning.
    """
    import optax

    OPT_NAMES = [
        'M_c', 'q', 's1_z', 's2_z', 'iota', 'd_L', 't_c',
        'psi', 'ra', 'dec',
    ]
    OPT_NDIM = len(OPT_NAMES)

    opt_lo = jnp.array([float(PRIOR_LO[PARAM_NAMES.index(n)]) for n in OPT_NAMES])
    opt_hi = jnp.array([float(PRIOR_HI[PARAM_NAMES.index(n)]) for n in OPT_NAMES])

    # Downsample frequency grid for faster JIT — reference only needs to be approximate
    opt_freqs = frequencies[::freq_downsample]
    opt_df = opt_freqs[1] - opt_freqs[0]

    opt_det_data = []
    opt_det_psd = []
    for det in detectors:
        opt_det_data.append(det.sliced_fd_data[::freq_downsample])
        opt_det_psd.append(det.sliced_psd[::freq_downsample])

    n_freq_full = len(frequencies)
    n_freq_opt = len(opt_freqs)
    print(f"  Optimizer frequency grid: {n_freq_opt} pts (downsampled {freq_downsample}x from {n_freq_full})")

    def neg_loglikelihood(x_unit):
        """Negative phase-marginalized log-likelihood in unit-cube coords."""
        x_phys = opt_lo + x_unit * (opt_hi - opt_lo)

        params = {
            'M_c': x_phys[0], 'q': x_phys[1], 's1_z': x_phys[2], 's2_z': x_phys[3],
            'iota': x_phys[4], 'd_L': x_phys[5], 't_c': x_phys[6],
            'psi': x_phys[7], 'ra': x_phys[8], 'dec': x_phys[9],
            'eta': x_phys[1] / (1 + x_phys[1]) ** 2,
            'phase_c': 0.0,
            'trigger_time': gps,
            'gmst': gmst,
        }

        q_val = x_phys[1]
        eta = q_val / (1 + q_val) ** 2
        M_total = x_phys[0] / eta ** 0.6
        m1 = M_total / (1 + q_val)
        m2 = q_val * m1
        mass_ok = (m1 >= M_COMP_LO) & (m1 <= M_COMP_HI) & (m2 >= M_COMP_LO) & (m2 <= M_COMP_HI)

        waveform_sky = waveform(opt_freqs, params)

        complex_d_inner_h = 0.0 + 0.0j
        optimal_SNR = 0.0
        for i, det in enumerate(detectors):
            h_dec = det.fd_response(opt_freqs, waveform_sky, params)
            complex_d_inner_h += 4.0 * jnp.sum(
                opt_det_data[i] * jnp.conj(h_dec) / opt_det_psd[i]
            ) * opt_df
            optimal_SNR += 4.0 * jnp.sum(
                jnp.abs(h_dec) ** 2 / opt_det_psd[i]
            ).real * opt_df

        ll = -optimal_SNR / 2 + log_i0(jnp.absolute(complex_d_inner_h))
        return jnp.where(mass_ok, -ll, jnp.inf)

    # Optax optimizer: Adam + gradient clipping + cosine LR decay
    schedule = optax.cosine_decay_schedule(learning_rate, n_steps)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(schedule),
    )

    def opt_step(carry, rng_key):
        """Single Adam step for one walker (vmapped over popsize)."""
        x, opt_state = carry
        loss, grad = jax.value_and_grad(neg_loglikelihood)(x)
        # Multiplicative gradient noise to escape local optima
        noise = noise_level * jax.random.normal(rng_key, grad.shape)
        grad = grad * (1.0 + noise)
        # Replace NaN/inf gradients (from invalid mass regions) with zero
        grad = jnp.where(jnp.isfinite(grad), grad, 0.0)
        updates, opt_state = optimizer.update(grad, opt_state, x)
        x = optax.apply_updates(x, updates)
        x = jnp.clip(x, 0.0, 1.0)  # enforce unit-cube bounds
        return (x, opt_state), loss

    def run_one_walker(x0, rng_key):
        """Run full optimization for a single walker via lax.scan."""
        opt_state = optimizer.init(x0)
        keys = jax.random.split(rng_key, n_steps)
        (x_final, _), losses = jax.lax.scan(opt_step, (x0, opt_state), keys)
        final_loss = neg_loglikelihood(x_final)
        return x_final, final_loss

    # vmap over walkers for parallel execution
    run_all_walkers = jax.jit(jax.vmap(run_one_walker))

    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    initial_positions = jax.random.uniform(init_key, (popsize, OPT_NDIM))

    print(f"  Optimizing reference parameters: {popsize} walkers x {n_steps} steps...")
    print("  JIT-compiling optimizer (optax + lax.scan)...")
    key, opt_key = jax.random.split(key)
    walker_keys = jax.random.split(opt_key, popsize)
    final_positions, final_neg_ll = run_all_walkers(initial_positions, walker_keys)
    jax.block_until_ready(final_positions)

    best_idx = jnp.argmin(final_neg_ll)
    best_unit = final_positions[best_idx]
    best_phys = opt_lo + best_unit * (opt_hi - opt_lo)
    best_ll = -float(final_neg_ll[best_idx])

    print(f"  Best log-likelihood: {best_ll:.2f} (walker {int(best_idx)})")

    q_best = float(best_phys[1])
    eta_best = q_best / (1 + q_best) ** 2

    ref = {
        'M_c': float(best_phys[0]), 'q': q_best, 'eta': eta_best,
        's1_z': float(best_phys[2]), 's2_z': float(best_phys[3]),
        'iota': float(best_phys[4]), 'd_L': float(best_phys[5]),
        't_c': float(best_phys[6]),
        'psi': float(best_phys[7]), 'ra': float(best_phys[8]),
        'dec': float(best_phys[9]),
        'phase_c': 0.0,
        'trigger_time': float(gps),
        'gmst': float(gmst),
    }
    if np.isclose(ref['eta'], 0.25):
        ref['eta'] = 0.249995
    return ref


if precessing and ref_params_source == 'optimize':
    raise NotImplementedError(
        "--ref-params optimize with IMRPhenomXPHM is not supported (the optimizer "
        "is hard-coded for the 10-D aligned-spin parameter set). Use --ref-params gwtc1.")

t_ref0 = time.time()
if ref_params_source == 'gwtc1':
    ref_params = load_reference_params(GWTC2P1_GW150914_HDF5)
    print(f"Reference (gwtc1): M_c={ref_params['M_c']:.4f}, q={ref_params['q']:.4f}, d_L={ref_params['d_L']:.1f}")
else:
    ref_params = optimize_reference_params(detectors, waveform, frequencies)
    print(f"Reference (optimized): M_c={ref_params['M_c']:.4f}, q={ref_params['q']:.4f}, d_L={ref_params['d_L']:.1f}")
t_ref = time.time() - t_ref0
print(f"[TIMING] Reference params: {t_ref:.1f}s")


# ============================================================================
# 6. HETERODYNING SETUP (pre-compute stacked arrays)
# ============================================================================

N_BINS = 501

def max_phase_diff(f, f_low, f_high, chi=1.0):
    """Maximum accumulated phase across PN orders. Eq.(7) of arXiv:2302.05333."""
    gamma = np.arange(-5, 6, 1) / 3.0
    f_2d = np.repeat(f[:, None], len(gamma), axis=1)
    f_star = np.repeat(f_low, len(gamma))
    f_star[gamma >= 0] = f_high
    return 2 * np.pi * chi * np.sum((f_2d / f_star) ** gamma * np.sign(gamma), axis=1)

def make_binning_scheme(freqs, n_bins, chi=1):
    phase_diff_array = max_phase_diff(freqs, freqs[0], freqs[-1], chi=chi)
    bin_f = interp1d(phase_diff_array, freqs)
    f_bins = np.array([bin_f(i) for i in np.linspace(
        phase_diff_array[0], phase_diff_array[-1], n_bins + 1)])
    f_bins_center = (f_bins[:-1] + f_bins[1:]) / 2
    return jnp.array(f_bins), jnp.array(f_bins_center)

def compute_coefficients(data, h_ref, psd, freqs, f_bins, f_bins_center):
    """Pre-compute heterodyning coefficients A0, A1, B0, B1 per bin.

    Uses np.searchsorted for O(n_bins * log n_freq) bin assignment instead
    of O(n_bins * n_freq) boolean masking. The bin boundaries are identical:
    searchsorted(side='left') gives the first index >= edge, matching the
    original (freqs >= f_bins[i]) & (freqs < f_bins[i+1]) condition.
    """
    df = freqs[1] - freqs[0]
    freqs_np = np.array(freqs)
    psd_np = np.array(psd)
    data_prod = np.array(data * h_ref.conj())
    self_prod = np.array(h_ref * h_ref.conj())

    # Binary-search bin edges: bin i spans [f_bins[i], f_bins[i+1])
    edges = np.array(f_bins)
    bin_start = np.searchsorted(freqs_np, edges[:-1], side='left')
    bin_end = np.searchsorted(freqs_np, edges[1:], side='left')
    centers_np = np.array(f_bins_center)

    n_bins = len(f_bins_center)
    A0 = np.empty(n_bins, dtype=complex)
    A1 = np.empty(n_bins, dtype=complex)
    B0 = np.empty(n_bins, dtype=complex)
    B1 = np.empty(n_bins, dtype=complex)
    for i in range(n_bins):
        s = slice(bin_start[i], bin_end[i])
        d_over_psd = data_prod[s] / psd_np[s]
        h_over_psd = self_prod[s] / psd_np[s]
        freq_diff = freqs_np[s] - centers_np[i]
        A0[i] = 4 * np.sum(d_over_psd) * df
        A1[i] = 4 * np.sum(d_over_psd * freq_diff) * df
        B0[i] = 4 * np.sum(h_over_psd) * df
        B1[i] = 4 * np.sum(h_over_psd * freq_diff) * df
    return jnp.array(A0), jnp.array(A1), jnp.array(B0), jnp.array(B1)


def setup_heterodyne(ref_params, detectors, waveform, frequencies, epoch, n_bins):
    """Build heterodyning reference state. Returns stacked arrays (n_det, n_bins).

    All detector-indexed quantities are stacked as (n_det, n_bins) JAX arrays
    instead of Python dicts, enabling vectorized computation in the likelihood.
    """
    params = {k: float(v) for k, v in ref_params.items()}
    if jnp.isclose(params.get('eta', 0.25), 0.25):
        params['eta'] = 0.249995

    print(f"Setting up heterodyning with {n_bins} bins...")
    h_sky = waveform(frequencies, params)

    freq_grid, freq_grid_center = make_binning_scheme(np.array(frequencies), n_bins)
    freq_grid_low = freq_grid[:-1]

    # Mask to valid waveform support.
    # Use a SINGLE mask on centers, then derive edges from it to maintain
    # the invariant: len(edges) = len(centers) + 1, with edge[i] <= center[i] < edge[i+1].
    h_amp = jnp.sum(jnp.array([jnp.abs(h_sky[k]) for k in h_sky]), axis=0)
    f_valid = frequencies[jnp.where(h_amp > 0)[0]]
    f_max_val, f_min_val = jnp.max(f_valid), jnp.min(f_valid)

    mask_center = jnp.where((freq_grid_center <= f_max_val) & (freq_grid_center >= f_min_val))[0]
    freq_grid_center = freq_grid_center[mask_center]
    freq_grid_low = freq_grid_low[mask_center]

    # Edges: need len(centers) + 1 edges that bracket the valid centers
    start_idx = mask_center[0]
    end_idx = mask_center[-1] + 2     # +1 inclusive, +1 for extra right edge
    freq_grid = freq_grid[start_idx:end_idx]

    h_sky_low = waveform(freq_grid_low, params)
    h_sky_center = waveform(freq_grid_center, params)

    # Build per-detector arrays, then stack
    # Note: fd_response handles time shifts internally (trigger_time - epoch + t_c)
    A0_list, A1_list, B0_list, B1_list = [], [], [], []
    ref_low_list, ref_center_list = [], []

    for det in detectors:
        waveform_ref = det.fd_response(frequencies, h_sky, params)
        ref_low_list.append(det.fd_response(freq_grid_low, h_sky_low, params))
        ref_center_list.append(det.fd_response(freq_grid_center, h_sky_center, params))

        A0, A1, B0, B1 = compute_coefficients(
            det.sliced_fd_data, waveform_ref, det.sliced_psd, frequencies, freq_grid, freq_grid_center)
        # No further masking needed: compute_coefficients already received the
        # masked grids, so its output contains only the valid bins.
        A0_list.append(A0)
        A1_list.append(A1)
        B0_list.append(B0)
        B1_list.append(B1)

    # Stack into (n_det, n_bins) arrays — no dicts in the hot path
    hetero = {
        'freq_grid_low': freq_grid_low,
        'freq_grid_center': freq_grid_center,
        'A0': jnp.stack(A0_list),            # (n_det, n_bins)
        'A1': jnp.stack(A1_list),
        'B0': jnp.stack(B0_list),
        'B1': jnp.stack(B1_list),
        'ref_low': jnp.stack(ref_low_list),   # (n_det, n_bins)
        'ref_center': jnp.stack(ref_center_list),
    }
    print(f"Heterodyning setup complete. Bin shape: {hetero['A0'].shape}")
    return hetero

t_het0 = time.time()
hetero = setup_heterodyne(ref_params, detectors, waveform, frequencies, epoch, N_BINS)
t_het = time.time() - t_het0
print(f"[TIMING] Heterodyning setup: {t_het:.1f}s")

# Extract as module-level constants for closure capture (static in JIT)
FREQ_LOW = hetero['freq_grid_low']
FREQ_CENTER = hetero['freq_grid_center']
A0 = hetero['A0']          # (n_det, n_bins)
A1 = hetero['A1']
B0 = hetero['B0']
B1 = hetero['B1']
REF_LOW = hetero['ref_low']      # (n_det, n_bins)
REF_CENTER = hetero['ref_center']


# ============================================================================
# 7. HETERODYNED LIKELIHOOD (phase-marginalized or standard)
# ============================================================================

@jax.jit
def loglikelihood_fn(x):
    """Heterodyned log-likelihood for BBH (GW-only, no standard siren terms).

    When phase_marg=True (HeterodynedPhaseMarginalizedLikelihoodFD pattern):
      ll_gw = -<h|h>/2 + log I_0(|<d|h>|)
    When phase_marg=False (HeterodynedTransientLikelihoodFD pattern):
      ll_gw = Re(<d|h> - <h|h>/2)

    The if/else on phase_marg is resolved at trace time (Python bool),
    producing two distinct compiled functions with no runtime overhead.

    BBH: no tidal parameters, no standard siren velocity terms.
    Returns ll_gw only.
    """
    # --- Build param dict for jimgw API (trace-time only) ---
    if precessing:
        # Convert sampled spherical spin coords -> Cartesian spin components
        # required by RippleIMRPhenomXPHM.
        a1, cost1, phi1 = x[I_A1], x[I_COST1], x[I_PHI1]
        a2, cost2, phi2 = x[I_A2], x[I_COST2], x[I_PHI2]
        sint1 = jnp.sqrt(jnp.maximum(1.0 - cost1 * cost1, 0.0))
        sint2 = jnp.sqrt(jnp.maximum(1.0 - cost2 * cost2, 0.0))
        params = {
            'M_c': x[I_MC], 'q': x[I_Q],
            's1_x': a1 * sint1 * jnp.cos(phi1),
            's1_y': a1 * sint1 * jnp.sin(phi1),
            's1_z': a1 * cost1,
            's2_x': a2 * sint2 * jnp.cos(phi2),
            's2_y': a2 * sint2 * jnp.sin(phi2),
            's2_z': a2 * cost2,
            'iota': x[I_IOTA], 'd_L': x[I_DL], 't_c': x[I_TC],
            'psi': x[I_PSI], 'ra': x[I_RA], 'dec': x[I_DEC],
            'eta': x[I_Q] / (1 + x[I_Q]) ** 2,
            'phase_c': 0.0 if phase_marg else x[I_PHASEC],
            'trigger_time': gps,
            'gmst': gmst,
        }
    else:
        params = {
            'M_c': x[I_MC], 'q': x[I_Q], 's1_z': x[I_S1Z], 's2_z': x[I_S2Z],
            'iota': x[I_IOTA], 'd_L': x[I_DL], 't_c': x[I_TC],
            'psi': x[I_PSI], 'ra': x[I_RA], 'dec': x[I_DEC],
            'eta': x[I_Q] / (1 + x[I_Q]) ** 2,
            'phase_c': 0.0 if phase_marg else x[I_PHASEC],
            'trigger_time': gps,
            'gmst': gmst,
        }

    # --- Waveform at bin frequencies (jimgw API, traced once) ---
    h_sky_low = waveform(FREQ_LOW, params)
    h_sky_center = waveform(FREQ_CENTER, params)

    # --- Detector responses: stack as (n_det, n_bins) ---
    det_resp_low = jnp.stack([
        det.fd_response(FREQ_LOW, h_sky_low, params)
        for det in detectors
    ])  # (n_det, n_bins)
    det_resp_center = jnp.stack([
        det.fd_response(FREQ_CENTER, h_sky_center, params)
        for det in detectors
    ])  # (n_det, n_bins)

    # --- Heterodyned computation: vectorized over detectors ---
    r0 = det_resp_center / REF_CENTER                                    # (n_det, n_bins)
    r1 = (det_resp_low / REF_LOW - r0) / (FREQ_LOW - FREQ_CENTER)       # (n_det, n_bins)

    # Complex match filter: sum over detectors and bins
    complex_match = jnp.sum(A0 * r0.conj() + A1 * r1.conj())            # scalar

    # Optimal SNR: sum over detectors and bins
    optimal_SNR = jnp.sum(
        B0 * jnp.abs(r0) ** 2 + 2 * B1 * (r0 * r1.conj()).real
    )                                                                     # scalar

    if phase_marg:
        # jimgw: HeterodynedPhaseMarginalizedLikelihoodFD
        ll_gw = -optimal_SNR.real / 2 + log_i0(jnp.absolute(complex_match))
    else:
        # jimgw: HeterodynedTransientLikelihoodFD
        ll_gw = (complex_match - optimal_SNR / 2).real

    return ll_gw


# ============================================================================
# 8. NESTED SAMPLING SETUP
# ============================================================================

num_live = args.n_live
num_delete = int(num_live * 0.3)
num_mcmc_steps = args.n_mcmc_steps if args.n_mcmc_steps is not None else int(NUM_DIMS * 8)

@jax.jit
def stepper_fn(x, d, t):
    """Linear step with periodic wrapping for psi (pi), ra (2pi), phase_c (2pi),
    and (precessing branch) phi_1, phi_2 (2pi)."""
    y = x + t * d
    y = y.at[I_PSI].set(jnp.mod(y[I_PSI], jnp.pi))
    y = y.at[I_RA].set(jnp.mod(y[I_RA], 2 * jnp.pi))
    if precessing:
        y = y.at[I_PHI1].set(jnp.mod(y[I_PHI1], 2 * jnp.pi))
        y = y.at[I_PHI2].set(jnp.mod(y[I_PHI2], 2 * jnp.pi))
    if not phase_marg:
        y = y.at[I_PHASEC].set(jnp.mod(y[I_PHASEC], 2 * jnp.pi))
    return y, True

nested_sampler = blackjax.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    num_delete=num_delete,
    num_inner_steps=num_mcmc_steps,
    stepper_fn=stepper_fn,
)


# ============================================================================
# 9. PRIOR SAMPLING & INITIALIZATION
# ============================================================================

def sample_from_prior(key, n):
    """Draw n samples for all parameters. Returns (n, NUM_DIMS) array.

    Uses rejection sampling to enforce component mass constraint [M_COMP_LO, M_COMP_HI].
    Oversamples by 4x then filters, repeating until n valid samples are obtained.
    """
    collected = []
    remaining = n
    while remaining > 0:
        key, subkey = jax.random.split(key)
        n_try = remaining * 4  # oversample
        keys = jax.random.split(subkey, NUM_DIMS)
        batch = jnp.zeros((n_try, NUM_DIMS))

        for i in range(NUM_DIMS):
            lo, hi = float(PRIOR_LO[i]), float(PRIOR_HI[i])
            ptype = int(PRIOR_TYPE[i])
            if ptype == 0:    # uniform
                col = jax.random.uniform(keys[i], (n_try,), minval=lo, maxval=hi)
            elif ptype == 1:  # sin (iota): inverse CDF = arccos(1 - 2u)
                col = jnp.arccos(1 - 2 * jax.random.uniform(keys[i], (n_try,)))
            elif ptype == 2:  # cos (dec): inverse CDF = arcsin(2u - 1)
                col = jnp.arcsin(2 * jax.random.uniform(keys[i], (n_try,)) - 1)
            elif ptype == 5:  # power-law d_L: p(d_L) ∝ d_L^2 -> CDF inversion
                # F(d_L) = (d_L^3 - lo^3) / (hi^3 - lo^3)
                # d_L = (lo^3 + u * (hi^3 - lo^3))^(1/3)
                u = jax.random.uniform(keys[i], (n_try,))
                col = (lo**3 + u * (hi**3 - lo**3)) ** (1.0 / 3.0)
            else:  # fallback: uniform
                col = jax.random.uniform(keys[i], (n_try,), minval=lo, maxval=hi)
            batch = batch.at[:, i].set(col)

        # Filter by component mass constraint
        q = batch[:, I_Q]
        eta = q / (1 + q) ** 2
        M_total = batch[:, I_MC] / eta ** 0.6
        m1 = M_total / (1 + q)
        m2 = q * m1
        valid = (m1 >= M_COMP_LO) & (m1 <= M_COMP_HI) & (m2 >= M_COMP_LO) & (m2 <= M_COMP_HI)
        good = batch[valid]
        collected.append(np.array(good[:remaining]))
        remaining -= len(collected[-1])

    return jnp.array(np.concatenate(collected)[:n])

t_init0 = time.time()
rng_key = jax.random.PRNGKey(0)
rng_key, init_key = jax.random.split(rng_key)
initial_particles = sample_from_prior(init_key, num_live)

state = nested_sampler.init(initial_particles)
t_init = time.time() - t_init0
print(f"[TIMING] Prior sampling + init: {t_init:.1f}s")


# ============================================================================
# 10. RUN NESTED SAMPLING
# ============================================================================

@jax.jit
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = nested_sampler.step(subk, state)
    return (state, k), dead_point

phase_msg = "phase_c marginalized" if phase_marg else "phase_c sampled"
print(f"Running nested sampling: {num_live} live, {NUM_DIMS}D ({phase_msg})")
print("JIT-compiling first step...")
t_jit0 = time.time()
(state, rng_key), dead_first = one_step((state, rng_key), None)
jax.block_until_ready(state)
t_jit = time.time() - t_jit0
print(f"[TIMING] JIT compilation (first step): {t_jit:.1f}s")

ns_start = time.time()
dead = [dead_first]

with tqdm.tqdm(desc="Dead points", initial=num_delete, unit=" dead points") as pbar:
    while not state.integrator.logZ_live - state.integrator.logZ < -3:
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        pbar.update(num_delete)

ns_time = time.time() - ns_start


# ============================================================================
# 11. POST-PROCESSING & OUTPUT
# ============================================================================

# Combine dead + live points via blackjax utility.
# finalise() concatenates all dead NSInfo objects with the final live particles,
# returning a single NSInfo with .particles (StateWithLogLikelihood).
result = finalise(state, dead, update_info=False)

samples = NestedSamples(
    np.array(result.particles.position),
    logL=np.array(result.particles.loglikelihood),
    logL_birth=np.array(result.particles.loglikelihood_birth),
    columns=PARAM_NAMES,
    labels=PARAM_LABELS,
)

logzs = samples.logZ(100)
print(f"Log Evidence: {logzs.mean():.2f} +/- {logzs.std():.2f}")

samples.to_csv(f'{label}.csv')
print(f"Saved to {label}.csv")

# Timing summary
t_total = time.time() - t0
print(f"\n{'='*50}")
print(f"TIMING SUMMARY")
print(f"{'='*50}")
print(f"  Data loading:     {t_data:7.1f}s")
print(f"  Reference params: {t_ref:7.1f}s")
print(f"  Heterodyne setup: {t_het:7.1f}s")
print(f"  Init + prior:     {t_init:7.1f}s")
print(f"  JIT compilation:  {t_jit:7.1f}s")
print(f"  Sampling:         {ns_time:7.1f}s")
print(f"  Total:            {t_total:7.1f}s")
print(f"{'='*50}")
