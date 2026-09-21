"""Compiled-envelope adapter for the signed-4b affine KS extension."""
import ctypes
import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
_NAME = "_private_compiled_signed_affine_ks_reference"
_SPEC = importlib.util.spec_from_file_location(
    _NAME, ROOT / "affine_weighted_ks_signed_reference.py"
)
signed = importlib.util.module_from_spec(_SPEC)
sys.modules[_NAME] = signed
_SPEC.loader.exec_module(signed)

lib = ctypes.CDLL(str(ROOT / "affine_envelope_kernel.so"))
lib.affine_ld_mant_dig.restype = ctypes.c_int
assert ctypes.sizeof(ctypes.c_longdouble) == np.dtype(np.longdouble).itemsize
assert lib.affine_ld_mant_dig() == np.finfo(np.longdouble).nmant + 1
array64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
arrayLD = np.ctypeslib.ndpointer(
    dtype=np.longdouble, ndim=1, flags="C_CONTIGUOUS"
)
lib.affine_upper_stack.argtypes = [
    ctypes.c_size_t,
    array64,
    array64,
    arrayLD,
    arrayLD,
    arrayLD,
]
lib.affine_upper_stack.restype = ctypes.c_size_t


def upper_envelope(slopes, intercepts):
    slopes = np.asarray(slopes, dtype=np.float64)
    intercepts = np.asarray(intercepts, dtype=np.float64)
    order = np.argsort(slopes, kind="stable")
    s = slopes[order]
    a = intercepts[order]
    first = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    a = np.maximum.reduceat(a, first)
    s = s[first]
    hs = np.empty(s.size, dtype=np.longdouble)
    ha = np.empty(s.size, dtype=np.longdouble)
    hx = np.empty(s.size, dtype=np.longdouble)
    k = lib.affine_upper_stack(s.size, s, a, hs, ha, hx)
    hs, ha, hx = hs[:k], ha[:k], hx[:k]
    start = int(np.searchsorted(hx, np.longdouble(0), side="right") - 1)
    end = int(np.searchsorted(hx, np.longdouble(1), side="left"))
    ids = np.arange(start, end)
    left = np.maximum(hx[ids], np.longdouble(0))
    next_x = np.r_[hx[1:], np.longdouble(np.inf)]
    right = np.minimum(next_x[ids], np.longdouble(1))
    return signed.ref._Envelope(left, right, hs[ids], ha[ids])


signed.ref._upper_envelope = upper_envelope
affine_ks_test = signed.affine_ks_test

