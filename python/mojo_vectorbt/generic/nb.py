"""Drop-in numeric subset of :mod:`vectorbt.generic.nb`."""

from __future__ import annotations

import numpy as np

from .._lib import addr, lib, matrix, restore


def _shift(arr, n: int, fill_value, op: int, ndim: int):
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    values, rows, cols = matrix(arr, ndim=ndim)
    result = np.empty((rows, cols), dtype=np.float64)
    lib().mvt_shift(
        addr(values), addr(result), rows, cols, n, float(fill_value), op
    )
    return restore(result, ndim=ndim)


def bshift_1d_nb(arr: np.ndarray, n: int = 1, fill_value=np.nan) -> np.ndarray:
    return _shift(arr, n, fill_value, 1, 1)


def bshift_nb(arr: np.ndarray, n: int = 1, fill_value=np.nan) -> np.ndarray:
    return _shift(arr, n, fill_value, 1, 2)


def fshift_1d_nb(arr: np.ndarray, n: int = 1, fill_value=np.nan) -> np.ndarray:
    return _shift(arr, n, fill_value, 0, 1)


def fshift_nb(arr: np.ndarray, n: int = 1, fill_value=np.nan) -> np.ndarray:
    return _shift(arr, n, fill_value, 0, 2)


def diff_1d_nb(a: np.ndarray, n: int = 1) -> np.ndarray:
    return _shift(a, n, np.nan, 2, 1)


def diff_nb(a: np.ndarray, n: int = 1) -> np.ndarray:
    return _shift(a, n, np.nan, 2, 2)


def pct_change_1d_nb(a: np.ndarray, n: int = 1) -> np.ndarray:
    return _shift(a, n, np.nan, 3, 1)


def pct_change_nb(a: np.ndarray, n: int = 1) -> np.ndarray:
    return _shift(a, n, np.nan, 3, 2)


def _rolling(a, window: int, minp: int | None, ddof: int, op: int, ndim: int):
    window = int(window)
    minp = window if minp is None else int(minp)
    if window <= 0:
        raise ValueError("window must be positive")
    if minp < 0:
        raise ValueError("minp must be non-negative")
    if minp > window:
        raise ValueError("minp must be <= window")
    values, rows, cols = matrix(a, ndim=ndim)
    result = np.full((rows, cols), np.nan, dtype=np.float64)
    if op < 2:
        lib().mvt_rolling_moments(
            addr(values), addr(result), rows, cols, window, minp, int(ddof), op
        )
    else:
        queue = np.empty(max(rows, 1), dtype=np.int64)
        lib().mvt_rolling_extreme(
            addr(values), addr(result), addr(queue), rows, cols, window, minp, op - 2
        )
    return restore(result, ndim=ndim)


def rolling_mean_1d_nb(
    a: np.ndarray, window: int, minp: int | None = None
) -> np.ndarray:
    return _rolling(a, window, minp, 0, 0, 1)


def rolling_mean_nb(
    a: np.ndarray, window: int, minp: int | None = None
) -> np.ndarray:
    return _rolling(a, window, minp, 0, 0, 2)


def rolling_std_1d_nb(
    a: np.ndarray, window: int, minp: int | None = None, ddof: int = 0
) -> np.ndarray:
    return _rolling(a, window, minp, ddof, 1, 1)


def rolling_std_nb(
    a: np.ndarray, window: int, minp: int | None = None, ddof: int = 0
) -> np.ndarray:
    return _rolling(a, window, minp, ddof, 1, 2)


def rolling_min_1d_nb(
    a: np.ndarray, window: int, minp: int | None = None
) -> np.ndarray:
    return _rolling(a, window, minp, 0, 2, 1)


def rolling_min_nb(
    a: np.ndarray, window: int, minp: int | None = None
) -> np.ndarray:
    return _rolling(a, window, minp, 0, 2, 2)


def rolling_max_1d_nb(
    a: np.ndarray, window: int, minp: int | None = None
) -> np.ndarray:
    return _rolling(a, window, minp, 0, 3, 1)


def rolling_max_nb(
    a: np.ndarray, window: int, minp: int | None = None
) -> np.ndarray:
    return _rolling(a, window, minp, 0, 3, 2)
