"""Affine-nuisance weighted KS test with the bootstrap loop in C++.

Drop-in for run_files.affine_weighted_ks_compiled.affine_ks_test (same arguments and
AffineKSResult). Preprocessing, the observed envelope, the KS statistic, ks_t and the
final overlap count are the reference's own functions, so every deterministic output is
identical. Only the B replicates run in affine_ks_bootstrap_kernel.so:

  multipliers="native": Poisson(1) multipliers from an independent seeded stream per
      replicate (xoshiro256**); p-values differ from the reference by Monte Carlo error
      only, and do not depend on the thread count.
  multipliers="numpy": the reference's NumPy stream (default_rng(seed), xi3 then xi4 per
      replicate), for checking the C++ arithmetic against the reference exactly.

Nonnegative weights only; signed 4b weights stay on affine_weighted_ks_signed_compiled.
Build: see affine_ks_bootstrap_kernel.cpp.
"""
import ctypes
import importlib.util
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
# The production module (reference with its envelope stack compiled); its ref supplies
# every deterministic step, exactly as evaluation uses it.
_name = '_private_compiled_for_fast_affine_ks'
_spec = importlib.util.spec_from_file_location(_name, ROOT / 'affine_weighted_ks_compiled.py')
compiled = importlib.util.module_from_spec(_spec); sys.modules[_name] = compiled; _spec.loader.exec_module(compiled)
ref = compiled.ref
LD = np.longdouble

_lib = ctypes.CDLL(str(ROOT / 'affine_ks_bootstrap_kernel.so'))
_lib.affine_ks_ld_mant_dig.restype = ctypes.c_int
if _lib.affine_ks_ld_mant_dig() != np.finfo(LD).nmant + 1 or ctypes.sizeof(ctypes.c_longdouble) != np.dtype(LD).itemsize:
    raise RuntimeError('C++ long double differs from numpy.longdouble')
_f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
_i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS')
_ld = np.ctypeslib.ndpointer(dtype=LD, ndim=1, flags='C_CONTIGUOUS')
_size, _u64 = ctypes.c_size_t, ctypes.c_uint64
_lib.affine_ks_bootstrap.restype = ctypes.c_int
_lib.affine_ks_bootstrap.argtypes = [
    _size, _size, _size, _f64, _f64, _f64, _i64, _i64, _f64, _f64, _f64,
    _size, _ld, _ld, _ld, _ld, ctypes.c_double, _u64, _size, _size,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, _size, _ld, _ld, _i64]
_lib.affine_ks_poisson_sample.restype = None
_lib.affine_ks_poisson_sample.argtypes = [_u64, _u64, _size, _i64]


def poisson_sample(seed, stream, n):
    """Poisson(1) draws from one native replicate stream (testing the sampler)."""
    out = np.empty(n, dtype=np.int64)
    _lib.affine_ks_poisson_sample(seed, stream, n, out)
    return out


def _c(a, dtype):
    return np.ascontiguousarray(a, dtype=dtype)


def affine_ks_test(z3, w3, z4, w4, *, L, U, B=1999, alpha=0.05, seed=None,
                   numerical_tol=1e-12, multipliers='native', threads=None, chunk=50):
    # ---- identical to the reference up to the observed statistic ----
    if not np.isfinite(L) or not np.isfinite(U) or not L < U:
        raise ValueError("Require finite L < U.")
    if not np.isfinite(U - L):
        raise ValueError("The support width U-L must be finite.")
    if isinstance(B, bool) or not isinstance(B, (int, np.integer)) or B < 1:
        raise ValueError("B must be a positive integer.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one.")
    if not np.isfinite(numerical_tol) or numerical_tol < 0:
        raise ValueError("numerical_tol must be finite and nonnegative.")
    if multipliers not in ('native', 'numpy'):
        raise ValueError("multipliers must be 'native' or 'numpy'.")
    z3s, w3s = ref._validate_sample(z3, w3, "3b", L, U)
    z4s, w4s = ref._validate_sample(z4, w4, "4b", L, U)
    x3 = (z3s - L) / (U - L)
    plus = w3s * x3
    minus = w3s * (1 - x3)
    if plus.sum() <= 0 or minus.sum() <= 0:
        raise ValueError("Both 3b endpoint-tilted weight totals must be positive.")
    up = plus / plus.sum()
    um = minus / minus.sum()
    r = w4s / w4s.sum()
    y = np.unique(np.r_[z3s, z4s])
    i3 = np.searchsorted(z3s, y, side="right")
    i4 = np.searchsorted(z4s, y, side="right")

    def cdf(q, idx):
        cumulative = np.r_[0.0, np.cumsum(q, dtype=np.float64)]
        cumulative[-1] = 1.0
        return cumulative[idx]

    fp, fm, f4 = cdf(up, i3), cdf(um, i3), cdf(r, i4)
    observed = ref._norm_envelope(fm - f4, fp - fm)
    ks, ks_t = ref._envelope_minimum(observed)

    # ---- bootstrap replicates in C++ ----
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 64))
    args = [z3s.size, z4s.size, y.size, _c(up, np.float64), _c(um, np.float64), _c(r, np.float64),
            _c(i3, np.int64), _c(i4, np.int64), _c(fp, np.float64), _c(fm, np.float64), _c(f4, np.float64),
            observed.left.size, _c(observed.left, LD), _c(observed.right, LD),
            _c(observed.slope, LD), _c(observed.intercept, LD), float(numerical_tol), int(seed) % (2 ** 64)]
    rng = np.random.default_rng(seed) if multipliers == 'numpy' else None
    nthreads = int(threads or os.environ.get('SLURM_CPUS_PER_TASK') or 1)
    all_intervals = []
    cap = 4 * (observed.left.size + 8)
    first = 0
    while first < B:
        count = min(chunk, B - first)
        xi3 = xi4 = None
        if rng is not None:  # the reference's stream: xi3 then xi4, replicate by replicate
            xi3 = np.empty((count, z3s.size), dtype=np.int64)
            xi4 = np.empty((count, z4s.size), dtype=np.int64)
            for q in range(count):
                xi3[q] = rng.poisson(1.0, size=z3s.size) - 1
                xi4[q] = rng.poisson(1.0, size=z4s.size) - 1
        while True:
            lo = np.empty(count * cap, dtype=LD)
            hi = np.empty(count * cap, dtype=LD)
            n = np.empty(count, dtype=np.int64)
            status = _lib.affine_ks_bootstrap(
                *args, first, count,
                None if xi3 is None else xi3.ctypes.data, None if xi4 is None else xi4.ctypes.data,
                nthreads, cap, lo, hi, n)
            if status == 0:
                break
            cap *= 4
        for q in range(count):
            base = q * cap
            all_intervals.extend(zip(lo[base:base + n[q]], hi[base:base + n[q]]))
        first += count

    count, p_t = ref._max_overlap(all_intervals)
    p_value = (1 + count) / (B + 1)
    q_by_name = {"3b_plus": up, "3b_minus": um, "4b": r}
    return ref.AffineKSResult(
        p_value=p_value, reject=bool(p_value <= alpha), alpha=float(alpha),
        ks_statistic=ks, ks_t=ks_t, maximizing_p_t=p_t, max_exceedances=count,
        bootstrap_replicates=int(B), n3=int(z3s.size), n4=int(z4s.size), distinct_scores=int(y.size),
        effective_sample_sizes={k: float(1 / np.dot(q, q)) for k, q in q_by_name.items()},
        max_normalized_weights={k: float(q.max()) for k, q in q_by_name.items()},
        numerical_tolerance=float(numerical_tol))
