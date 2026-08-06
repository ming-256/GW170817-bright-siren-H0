"""
Unheterodyned Nested Sampling for GW170817
=============================================

Full frequency-domain likelihood over ~260k bins (no relative binning).
~2500x slower per call but no approximation error — reference for validation.

With --phase-marginalization:
  14D parameter space, phase_c analytically marginalized via log I_0(|<d|h>|)
  (jimgw: PhaseMarginalizedLikelihoodFD pattern)
Without --phase-marginalization:
  15D parameter space, phase_c sampled as uniform [0, 2pi]
  (jimgw: BaseTransientLikelihoodFD pattern)

Usage:
  python GW170817_unheterodyned_1.py [--waveform {IMRPhenomD_NRTidalv2,TaylorF2}]
                                      [--phase-marginalization]
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
import jax.scipy.stats as stats
import numpy as np
import blackjax
import pickle
import time
import tqdm
from astropy.time import Time
from anesthetic import NestedSamples
from blackjax.ns.utils import finalise

from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2, RippleTaylorF2
from jimgw.core.single_event.data import Data, PowerSpectrum
from gwpy.timeseries import TimeSeries
from scipy.interpolate import interp1d
from blackjax.ns.base import StateWithLogLikelihood
from blackjax.ns.adaptive import AdaptiveNSState
from blackjax.ns.integrator import init_integrator
from blackjax.ns.nss import update_inner_kernel_params

from jax.scipy.special import i0e

# ============================================================================
# 0. COMMAND-LINE ARGUMENTS
# ============================================================================
parser = argparse.ArgumentParser(description='Unheterodyned nested sampling for GW170817')
parser.add_argument('--waveform', choices=['IMRPhenomD_NRTidalv2', 'TaylorF2'],
                    default='IMRPhenomD_NRTidalv2', help='Waveform approximant')
parser.add_argument('--data-source', choices=['fetch', 'local'],
                    default='fetch',
                    help='Data source: "fetch" pulls from GWOSC via gwpy (requires internet), '
                         '"local" reads HDF5 files from $GWOSC_GW170817_DIR/ (default data/GWOSC/GW170817/)')
parser.add_argument('--psd-source', choices=['self', 'bilby', 'gwtc1', 'kazewong'],
                    default='gwtc1',
                    help='PSD source: "self" (estimated from data via gwpy), "bilby" (Bilby PSDs), '
                         '"gwtc1" (official BayesWave PSDs from LIGO-P1900011), '
                         '"kazewong" (kazewong pre-processed PSDs)')
parser.add_argument('--phase-marginalization', action='store_true',
                    help='Enable analytic phase marginalization (removes phase_c from sampling)')
parser.add_argument('--output-dir', default='Results',
                    help='Directory to write output CSV files (default: Results)')
parser.add_argument('--checkpoint-every', type=int, default=5,
                    help='Save checkpoint every N nested sampling steps (default: 5)')
parser.add_argument('--no-resume', action='store_true',
                    help='Start fresh even if a checkpoint file exists')
parser.add_argument('--wide-prior', action='store_true',
                    help='Use wider priors (relaxed M_c, q, spin, d_L bounds)')
parser.add_argument('--label-suffix', default='',
                    help='Suffix to append to output filename (e.g. "_narrow_prior")')
parser.add_argument('--nlive', type=int, default=1500,
                    help='Number of live points (default: 1500)')
args = parser.parse_args()
data_source = args.data_source
psd_source = args.psd_source
phase_marg = args.phase_marginalization
output_dir = args.output_dir

@jax.jit
def log_i0(x):
    return jnp.log(i0e(x)) + x


# ============================================================================
# 2. PARAMETER CONFIGURATION
# ============================================================================
PARAM_NAMES = [
    "M_c", "q", "s1_z", "s2_z", "iota", "d_L", "t_c",
    "psi", "ra", "dec", "lambda_1", "lambda_2", "H_0", "v_p",
]
PARAM_LABELS = [
    r"$M_c$", r"$q$", r"$s_{1z}$", r"$s_{2z}$", r"$\iota$", r"$d_L$", r"$t_c$",
    r"$\psi$", r"$\alpha$", r"$\delta$", r"$\Lambda_1$", r"$\Lambda_2$", r"$H_0$", r"$v_p$",
]

I_MC, I_Q, I_S1Z, I_S2Z, I_IOTA, I_DL, I_TC = 0, 1, 2, 3, 4, 5, 6
I_PSI, I_RA, I_DEC, I_L1, I_L2, I_H0, I_VP = 7, 8, 9, 10, 11, 12, 13

# NGC 4993 host galaxy sky position (GW170817 EM counterpart)
# RA = 197.4508 deg = 3.4462 rad, Dec = -23.3815 deg = -0.4081 rad
_NGC4993_RA = 3.4462   # rad
_NGC4993_DEC = -0.4081  # rad

# Base priors (shared between narrow and wide)
_PRIOR_LO_BASE = [
    1.184, 0.125, -0.05, -0.05,             # M_c, q, s1_z, s2_z
    0.0, 1.0, -0.1,                          # iota, d_L, t_c
    0.0, 0.0, -jnp.pi / 2,                   # psi, ra, dec
    0.0, 0.0, 20.0, -1000.0,                 # lambda_1, lambda_2, H_0, v_p
]
_PRIOR_HI_BASE = [
    2.168, 1.00, 0.05, 0.05,                # M_c, q, s1_z, s2_z
    jnp.pi, 75.0, 0.1,                       # iota, d_L, t_c
    jnp.pi, 2 * jnp.pi, jnp.pi / 2,         # psi, ra, dec
    5000.0, 5000.0, 250.0, 1000.0,           # lambda_1, lambda_2, H_0, v_p
]
M_COMP_LO_VAL = 0.5
M_COMP_HI_VAL = 7.7

if args.wide_prior:
    # "Wide" = RA/dec constrained to NGC 4993 host galaxy location (±0.05 rad ≈ ±3°)
    _PRIOR_LO_BASE[I_RA]  = _NGC4993_RA - 0.05
    _PRIOR_HI_BASE[I_RA]  = _NGC4993_RA + 0.05
    _PRIOR_LO_BASE[I_DEC] = _NGC4993_DEC - 0.05
    _PRIOR_HI_BASE[I_DEC] = _NGC4993_DEC + 0.05
    print(f"Using HOST-LOCALISED priors: RA=[{_PRIOR_LO_BASE[I_RA]:.4f}, {_PRIOR_HI_BASE[I_RA]:.4f}], "
          f"Dec=[{_PRIOR_LO_BASE[I_DEC]:.4f}, {_PRIOR_HI_BASE[I_DEC]:.4f}]")
else:
    # Default: full-sky RA/dec (narrow M_c/q/spin priors)
    print("Using FULL-SKY priors (narrow M_c/q/spin)")
_PRIOR_TYPE_BASE = [0, 0, 0, 0, 1, 3, 0, 0, 0, 2, 0, 0, 4, 0]

if not phase_marg:
    PARAM_NAMES.append("phase_c")
    PARAM_LABELS.append(r"$\phi_c$")
    I_PHASEC = 14
    _PRIOR_LO_BASE.append(0.0)
    _PRIOR_HI_BASE.append(float(2 * jnp.pi))
    _PRIOR_TYPE_BASE.append(0)

NUM_DIMS = len(PARAM_NAMES)

PRIOR_LO = jnp.array(_PRIOR_LO_BASE)
PRIOR_HI = jnp.array(_PRIOR_HI_BASE)

M_COMP_LO = M_COMP_LO_VAL
M_COMP_HI = M_COMP_HI_VAL

PRIOR_TYPE = jnp.array(_PRIOR_TYPE_BASE)

_PRIOR_RANGE = PRIOR_HI - PRIOR_LO
_PRIOR_LOG_RANGE = jnp.log(_PRIOR_RANGE)
_PRIOR_LOG_LOG_RATIO = jnp.log(jnp.log(PRIOR_HI / PRIOR_LO))
_BETA_LN = jax.scipy.special.betaln(3.0, 1.0)


# ============================================================================
# 3. VECTORIZED LOG-PRIOR
# ============================================================================

@jax.jit
def logprior_fn(x):
    in_bounds = (x >= PRIOR_LO) & (x <= PRIOR_HI)

    lp_uniform = jnp.where(in_bounds, -_PRIOR_LOG_RANGE, -jnp.inf)
    lp_sin = jnp.where(in_bounds, jnp.log(jnp.abs(jnp.sin(x)) + 1e-300) - jnp.log(2.0), -jnp.inf)
    lp_cos = jnp.where(in_bounds, jnp.log(jnp.abs(jnp.cos(x)) + 1e-300) - jnp.log(2.0), -jnp.inf)
    u = (x - PRIOR_LO) / _PRIOR_RANGE
    lp_beta = jnp.where(in_bounds, 2.0 * jnp.log(jnp.abs(u) + 1e-300) - _PRIOR_LOG_RANGE - _BETA_LN, -jnp.inf)
    lp_log = jnp.where(in_bounds, -_PRIOR_LOG_LOG_RATIO - jnp.log(jnp.abs(x) + 1e-300), -jnp.inf)

    lp = jnp.where(PRIOR_TYPE == 0, lp_uniform,
         jnp.where(PRIOR_TYPE == 1, lp_sin,
         jnp.where(PRIOR_TYPE == 2, lp_cos,
         jnp.where(PRIOR_TYPE == 3, lp_beta,
                    lp_log))))

    total = jnp.sum(lp)

    q = x[I_Q]
    eta = q / (1 + q) ** 2
    M_total = x[I_MC] / eta ** 0.6
    m1 = M_total / (1 + q)
    m2 = q * m1
    mass_ok = (m1 >= M_COMP_LO) & (m1 <= M_COMP_HI) & (m2 >= M_COMP_LO) & (m2 <= M_COMP_HI)
    total = jnp.where(mass_ok, total, -jnp.inf)

    # Jacobian |∂(m1,m2)/∂(M_c,q)| = M_c * (1+q)^(2/5) / q^(6/5)
    # Converts uniform-in-(M_c,q) to uniform-in-(m1,m2), as assumed in
    # Abbott et al., PhysRevX 9, 011001, Sec. II.D (z=0.0099 for NGC 4993)
    log_jacobian = jnp.log(x[I_MC]) - 1.2 * jnp.log(x[I_Q]) + 0.4 * jnp.log(1.0 + x[I_Q])
    total = total + log_jacobian

    return total


# ============================================================================
# 4. EVENT CONFIGURATION & DETECTOR DATA
# ============================================================================

gps = 1187008882.43
fmin = 23.0
fmax = 2048.0
duration = 128
post_trigger_duration = 2
roll_off = 0.4
tukey_alpha = 2 * roll_off / duration
psd_pad = 16
psd_duration = 1024

waveform_tag = args.waveform
marg_tag = 'PhaseMarg' if phase_marg else 'NoMarg'
import os; os.makedirs(output_dir, exist_ok=True)
label_suffix = args.label_suffix
label = f'{output_dir}/{marg_tag}_Unheterodyned_{waveform_tag}_{data_source}_psd-{psd_source}{label_suffix}'
checkpoint_path = f'{label}_checkpoint.pkl'
CHECKPOINT_EVERY = args.checkpoint_every

start = gps - (duration - post_trigger_duration)
end = gps + post_trigger_duration
psd_start = start - psd_pad - psd_duration
psd_end = start - psd_pad

t0 = time.time()

# Local GWOSC HDF5 file mapping: ifo name -> file path
GWOSC_LOCAL_DIR = os.environ.get('GWOSC_GW170817_DIR', 'data/GWOSC/GW170817')
# LVK GWTC-1 GW170817 reference posterior; supplies the heterodyne reference
# parameters for --ref-params gwtc1.
GWTC1_HDF5 = os.environ.get('GWTC1_HDF5', 'results/GW170817_GWTC-1.hdf5')
GWOSC_LOCAL_FILES = {
    'H1': os.path.join(GWOSC_LOCAL_DIR, 'H-H1_LOSC_CLN_4_V1-1187007040-2048.hdf5'),
    'L1': os.path.join(GWOSC_LOCAL_DIR, 'L-L1_LOSC_CLN_4_V1-1187007040-2048.hdf5'),
    'V1': os.path.join(GWOSC_LOCAL_DIR, 'V-V1_LOSC_CLN_4_V1-1187007040-2048.hdf5'),
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
KAZEWONG_PSD_DIR = os.path.join(GWOSC_LOCAL_DIR, 'kazewong')
KAZEWONG_PSD_PREFIX = 'GW170817-IMRD_data0_1187008882-43_generation_data_dump.pickle'


def load_external_psd(psd_src, ifo_name, target_freqs):
    """Load PSD from an external file and interpolate to target frequency grid."""
    if psd_src == 'bilby':
        psd_file = os.path.join(GWOSC_PSD_DIR, 'Bilby', f'{ifo_name.lower()}_psd.txt')
        psd_data = np.loadtxt(psd_file)
        freqs_psd, psd_vals = psd_data[:, 0], psd_data[:, 1]
    elif psd_src == 'gwtc1':
        psd_data = np.loadtxt(os.path.join(GWOSC_PSD_DIR, 'GWTC1_GW170817_PSDs.dat'))
        freqs_psd = psd_data[:, 0]
        col_map = {'H1': 1, 'L1': 2, 'V1': 3}
        psd_vals = psd_data[:, col_map[ifo_name]]
    elif psd_src == 'kazewong':
        psd_file = os.path.join(KAZEWONG_PSD_DIR, f'{KAZEWONG_PSD_PREFIX}_{ifo_name}_psd.txt')
        psd_data = np.loadtxt(psd_file)
        freqs_psd, psd_vals = psd_data[:, 0], psd_data[:, 1]
    else:
        raise ValueError(f"Unknown PSD source: {psd_src}")
    # Replace inf with large sentinel before interpolation to avoid
    # scipy RuntimeWarning from inf arithmetic (inf-finite=inf, inf*0=NaN).
    # After interpolation, restore inf where the result exceeds the sentinel threshold.
    _PSD_INF_SENTINEL = 1e300
    inf_mask_src = ~np.isfinite(psd_vals)
    psd_vals_safe = np.where(inf_mask_src, _PSD_INF_SENTINEL, psd_vals)
    psd_interp = interp1d(freqs_psd, psd_vals_safe, kind='linear',
                          fill_value=_PSD_INF_SENTINEL, bounds_error=False)
    psd_values = psd_interp(np.array(target_freqs))
    # Restore inf where interpolation touched sentinel-affected regions
    psd_values = np.where(psd_values >= _PSD_INF_SENTINEL * 0.5, np.inf, psd_values)
    return PowerSpectrum(
        values=jnp.array(psd_values),
        frequencies=jnp.array(target_freqs),
        name=ifo_name,
    )


detectors = [get_H1(), get_L1(), get_V1()]
N_DET = len(detectors)
print(f"Data source: {data_source}, PSD source: {psd_source}")

for ifo in detectors:
    t_det = time.time()

    if data_source == 'local':
        strain_data = load_gwosc_local(ifo.name, start, end)
        if psd_source == 'self':
            psd_ts = load_gwosc_local_gwpy(ifo.name, psd_start, psd_end)
    else:
        strain_data = Data.from_gwosc(ifo.name, start, end, version=2)
        if psd_source == 'self':
            psd_ts = TimeSeries.fetch_open_data(ifo.name, psd_start, psd_end, version=2)
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

    ifo.set_frequency_bounds(fmin, fmax)
    print(f"  {ifo.name}: data={t_fetch:.1f}s, PSD({psd_source})={t_psd:.1f}s, total={time.time()-t_det:.1f}s")

t_data = time.time() - t0
print(f"[TIMING] Data loading: {t_data:.1f}s")

H1, L1, V1 = detectors

if waveform_tag == 'TaylorF2':
    waveform = RippleTaylorF2(f_ref=20.0, use_lambda_tildes=False)
else:
    waveform = RippleIMRPhenomD_NRTidalv2(f_ref=20.0, use_lambda_tildes=False, no_taper=False)
print(f"Waveform: {waveform_tag}")

frequencies = H1.sliced_frequencies
df = float(frequencies[1] - frequencies[0])
gmst = Time(gps, format="gps").sidereal_time("apparent", "greenwich").rad


# ============================================================================
# 5. PRE-STACK DETECTOR DATA (n_det, n_freq) — avoids per-call overhead
# ============================================================================

t_stack0 = time.time()

# Stack data and PSD as (n_det, n_freq) arrays for vectorised inner products
DATA_FD = jnp.stack([det.sliced_fd_data for det in detectors])   # (n_det, n_freq)
PSD = jnp.stack([det.sliced_psd for det in detectors])           # (n_det, n_freq)

t_stack = time.time() - t_stack0
print(f"[TIMING] Data stacking: {t_stack:.1f}s")
print(f"Frequency grid: {frequencies.shape[0]} bins, df={df:.6f} Hz")


# ============================================================================
# 6. FULL LIKELIHOOD (phase-marginalized or standard)
# ============================================================================

@jax.jit
def loglikelihood_fn(x):
    """Full frequency-domain log-likelihood + standard siren terms.

    When phase_marg=True (PhaseMarginalizedLikelihoodFD pattern):
      ll_gw = -<h|h>/2 + log I_0(|<d|h>|)
    When phase_marg=False (BaseTransientLikelihoodFD pattern):
      ll_gw = Re(<d|h>) - <h|h>/2
    """
    params = {
        'M_c': x[I_MC], 'q': x[I_Q], 's1_z': x[I_S1Z], 's2_z': x[I_S2Z],
        'iota': x[I_IOTA], 'd_L': x[I_DL], 't_c': x[I_TC],
        'psi': x[I_PSI], 'ra': x[I_RA], 'dec': x[I_DEC],
        'lambda_1': x[I_L1], 'lambda_2': x[I_L2],
        'eta': x[I_Q] / (1 + x[I_Q]) ** 2,
        'phase_c': 0.0 if phase_marg else x[I_PHASEC],
        'trigger_time': gps,
        'gmst': gmst,
    }

    # Waveform at all frequency bins
    h_sky = waveform(frequencies, params)

    # Detector responses: (n_det, n_freq)
    h_det = jnp.stack([
        det.fd_response(frequencies, h_sky, params)
        for det in detectors
    ])

    # Inner products: 4 * df * sum(d * h^* / S) over frequency, summed over detectors
    complex_d_inner_h = 4 * df * jnp.sum(DATA_FD * h_det.conj() / PSD)
    optimal_snr_sq = 4 * df * jnp.sum(jnp.abs(h_det) ** 2 / PSD)

    if phase_marg:
        # jimgw: PhaseMarginalizedLikelihoodFD
        ll_gw = -optimal_snr_sq.real / 2 + log_i0(jnp.absolute(complex_d_inner_h))
    else:
        # jimgw: BaseTransientLikelihoodFD
        ll_gw = jnp.real(complex_d_inner_h) - optimal_snr_sq.real / 2

    # Standard siren terms
    ll_vr = stats.norm.logpdf(3327.0, x[I_VP] + x[I_H0] * x[I_DL], 72.0)
    ll_vp = stats.norm.logpdf(310.0, x[I_VP], 150.0)

    return ll_gw + ll_vr + ll_vp


# ============================================================================
# 7. NESTED SAMPLING SETUP
# ============================================================================

num_live = args.nlive
num_delete = int(num_live * 0.5)
num_mcmc_steps = int(NUM_DIMS * 8)

@jax.jit
def stepper_fn(x, d, t):
    y = x + t * d
    y = y.at[I_PSI].set(jnp.mod(y[I_PSI], jnp.pi))
    y = y.at[I_RA].set(jnp.mod(y[I_RA], 2 * jnp.pi))
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
# 8. PRIOR SAMPLING & INITIALIZATION (with checkpoint resume)
# ============================================================================

def sample_from_prior(key, n):
    collected = []
    remaining = n
    while remaining > 0:
        key, subkey = jax.random.split(key)
        n_try = remaining * 4
        keys = jax.random.split(subkey, NUM_DIMS)
        batch = jnp.zeros((n_try, NUM_DIMS))

        for i in range(NUM_DIMS):
            lo, hi = float(PRIOR_LO[i]), float(PRIOR_HI[i])
            ptype = int(PRIOR_TYPE[i])
            if ptype == 0:
                col = jax.random.uniform(keys[i], (n_try,), minval=lo, maxval=hi)
            elif ptype == 1:
                col = jnp.arccos(1 - 2 * jax.random.uniform(keys[i], (n_try,)))
            elif ptype == 2:
                col = jnp.arcsin(2 * jax.random.uniform(keys[i], (n_try,)) - 1)
            elif ptype == 3:
                col = jax.random.beta(keys[i], 3.0, 1.0, (n_try,)) * (hi - lo) + lo
            elif ptype == 4:
                col = lo * (hi / lo) ** jax.random.uniform(keys[i], (n_try,))
            batch = batch.at[:, i].set(col)

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


# --- Checkpoint save/load ---
def save_checkpoint(state, dead, rng_key, step_count):
    """Atomically save sampling state to disk for resume."""
    tmp = checkpoint_path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump({
            'state': jax.device_get(state),
            'dead': [jax.device_get(d) for d in dead],
            'rng_key': jax.device_get(rng_key),
            'step_count': step_count,
        }, f)
    os.replace(tmp, checkpoint_path)
    print(f"  [checkpoint] Saved at step {step_count} -> {checkpoint_path}")


def load_checkpoint():
    """Load sampling state from disk."""
    with open(checkpoint_path, 'rb') as f:
        return pickle.load(f)


# --- Try to resume from checkpoint ---
resumed = False
if os.path.exists(checkpoint_path) and not args.no_resume:
    print(f"Found checkpoint: {checkpoint_path}")
    ckpt = load_checkpoint()
    state = ckpt['state']
    dead = ckpt['dead']
    rng_key = ckpt['rng_key']
    step_count = ckpt['step_count']
    resumed = True
    t_init = 0.0
    t_jit = 0.0
    print(f"Resumed from step {step_count} with {len(dead)} dead point batches")

if not resumed:
    t_init0 = time.time()
    rng_key = jax.random.PRNGKey(0)
    rng_key, init_key = jax.random.split(rng_key)
    initial_particles = sample_from_prior(init_key, num_live)

    # Chunked init: evaluate logprior/loglikelihood in batches to avoid OOM.
    # The default nested_sampler.init() vmaps over ALL particles at once,
    # which for 2000 × 260k freq bins × 3 dets × complex128 ≈ 35 GiB.
    INIT_CHUNK = 100
    print(f"Chunked init: {num_live} particles in batches of {INIT_CHUNK}...")

    logpriors = []
    loglikes = []
    for i in range(0, num_live, INIT_CHUNK):
        chunk = initial_particles[i:i+INIT_CHUNK]
        lp = jax.vmap(logprior_fn)(chunk)
        ll = jax.vmap(loglikelihood_fn)(chunk)
        jax.block_until_ready(ll)
        logpriors.append(lp)
        loglikes.append(ll)
        print(f"  init chunk {i}-{i+len(chunk)}: done")

    logprior_all = jnp.concatenate(logpriors)
    loglike_all = jnp.concatenate(loglikes)
    loglike_birth_all = jnp.nan * jnp.ones_like(loglike_all)

    particles = StateWithLogLikelihood(
        position=initial_particles,
        logdensity=logprior_all,
        loglikelihood=loglike_all,
        loglikelihood_birth=loglike_birth_all,
    )
    integrator = init_integrator(particles)
    tmp_state = AdaptiveNSState(particles=particles, integrator=integrator, inner_kernel_params={})
    inner_kernel_params = update_inner_kernel_params(None, tmp_state, None, {})
    state = AdaptiveNSState(
        particles=particles,
        integrator=integrator,
        inner_kernel_params=inner_kernel_params,
    )

    t_init = time.time() - t_init0
    print(f"[TIMING] Prior sampling + chunked init: {t_init:.1f}s")

    step_count = 0
    dead = []


# ============================================================================
# 9. RUN NESTED SAMPLING (with checkpointing every {CHECKPOINT_EVERY} steps)
# ============================================================================

@jax.jit
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = nested_sampler.step(subk, state)
    return (state, k), dead_point

phase_msg = "phase_c marginalized" if phase_marg else "phase_c sampled"
print(f"Running nested sampling: {num_live} live, {NUM_DIMS}D ({phase_msg}, FULL likelihood)")
print(f"Checkpointing every {CHECKPOINT_EVERY} steps to {checkpoint_path}")

if not resumed:
    print("JIT-compiling first step (this will be slow — ~260k freq bins)...")
    t_jit0 = time.time()
    (state, rng_key), dead_first = one_step((state, rng_key), None)
    jax.block_until_ready(state)
    t_jit = time.time() - t_jit0
    print(f"[TIMING] JIT compilation (first step): {t_jit:.1f}s")
    dead.append(dead_first)
    step_count += 1

ns_start = time.time()

with tqdm.tqdm(desc="Dead points", initial=len(dead) * num_delete, unit=" dead points") as pbar:
    while not state.integrator.logZ_live - state.integrator.logZ < -3:
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        step_count += 1
        pbar.update(num_delete)
        if step_count % CHECKPOINT_EVERY == 0:
            save_checkpoint(state, dead, rng_key, step_count)

ns_time = time.time() - ns_start


# ============================================================================
# 10. POST-PROCESSING & OUTPUT
# ============================================================================

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

# Clean up checkpoint after successful completion
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
    print(f"Removed checkpoint: {checkpoint_path}")

t_total = time.time() - t0
print(f"\n{'='*50}")
print(f"TIMING SUMMARY (UNHETERODYNED)")
print(f"{'='*50}")
print(f"  Data loading:     {t_data:7.1f}s")
print(f"  Data stacking:    {t_stack:7.1f}s")
print(f"  Init + prior:     {t_init:7.1f}s")
print(f"  JIT compilation:  {t_jit:7.1f}s")
print(f"  Sampling:         {ns_time:7.1f}s")
print(f"  Total:            {t_total:7.1f}s")
print(f"{'='*50}")
