"""ctypes loader and array helpers for the Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = os.path.join(ROOT, "src", "kernels.mojo")
LIB = os.environ.get("MOJO_VECTORBT_LIB") or os.path.join(
    ROOT, "dist", "libmojo-vectorbt.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mvt_get_return": ([F, F], F),
    "mvt_shift": ([I, I, I, I, I, F, I], None),
    "mvt_rolling_moments": ([I] * 8, None),
    "mvt_rolling_extreme": ([I] * 8, None),
    "mvt_returns": ([I] * 5, None),
    "mvt_cumulative": ([I, I, I, I, F, I], None),
    "mvt_single_metric": ([I, I, I, I, I, F, F, F, I], None),
    "mvt_pair_metric": ([I, I, I, I, I, I, F, F, I], None),
    "mvt_quantile_metric": ([I, I, I, I, I, I, F], None),
    "mvt_clean_signals": ([I] * 7, None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_VECTORBT_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SOURCE):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _loaded


def addr(array: np.ndarray) -> int:
    if not array.flags.c_contiguous:
        raise ValueError("only C-contiguous arrays may cross the Mojo boundary")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("cannot pass a null array pointer to Mojo")
    return address


def float64_array(a) -> np.ndarray:
    original = np.asarray(a)
    if np.issubdtype(original.dtype, np.complexfloating):
        raise TypeError("complex arrays are not supported")
    if np.issubdtype(original.dtype, np.floating) and original.dtype.itemsize > 8:
        raise TypeError("floating-point inputs wider than float64 are not supported")
    if np.issubdtype(original.dtype, np.integer) and original.size:
        limit = 2**53
        if np.any(original > limit) or np.any(original < -limit):
            raise ValueError("integer inputs outside the exact float64 range are not supported")
    return np.ascontiguousarray(original, dtype=np.float64)


def matrix(a, *, ndim: int) -> tuple[np.ndarray, int, int]:
    array = float64_array(a)
    if array.ndim != ndim:
        raise ValueError(f"expected a {ndim}-dimensional array, got {array.ndim}")
    if ndim == 1:
        return array.reshape(-1, 1), array.shape[0], 1
    return array, array.shape[0], array.shape[1]


def restore(array: np.ndarray, *, ndim: int):
    return array[:, 0] if ndim == 1 else array
