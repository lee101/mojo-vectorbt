"""Drop-in numeric subset of :mod:`vectorbt.signals.nb`."""

from __future__ import annotations

import numpy as np

from .._lib import addr, lib


def _clean(entries, exits, entry_first: bool, ndim: int):
    entries_array = np.ascontiguousarray(np.asarray(entries, dtype=bool))
    exits_array = np.ascontiguousarray(np.asarray(exits, dtype=bool))
    if entries_array.ndim != ndim or exits_array.ndim != ndim:
        raise ValueError(f"entries and exits must both be {ndim}-dimensional")
    if entries_array.shape != exits_array.shape:
        raise ValueError("entries and exits must have the same shape")
    if ndim == 1:
        rows, cols = entries_array.shape[0], 1
        entries_matrix = entries_array.reshape(-1, 1)
        exits_matrix = exits_array.reshape(-1, 1)
    else:
        rows, cols = entries_array.shape
        entries_matrix = entries_array
        exits_matrix = exits_array
    entries_result = np.empty((rows, cols), dtype=np.uint8)
    exits_result = np.empty((rows, cols), dtype=np.uint8)
    lib().mvt_clean_signals(
        addr(entries_matrix),
        addr(exits_matrix),
        addr(entries_result),
        addr(exits_result),
        rows,
        cols,
        int(entry_first),
    )
    if ndim == 1:
        return entries_result[:, 0].astype(bool), exits_result[:, 0].astype(bool)
    return entries_result.astype(bool), exits_result.astype(bool)


def clean_enex_1d_nb(
    entries: np.ndarray, exits: np.ndarray, entry_first: bool
) -> tuple[np.ndarray, np.ndarray]:
    return _clean(entries, exits, entry_first, 1)


def clean_enex_nb(
    entries: np.ndarray, exits: np.ndarray, entry_first: bool
) -> tuple[np.ndarray, np.ndarray]:
    return _clean(entries, exits, entry_first, 2)
