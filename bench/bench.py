"""Benchmark Mojo kernels against vectorbt on identical NumPy arrays."""

from __future__ import annotations

import math
import os
import platform
import time

import numpy as np
import vectorbt as vbt

from mojo_vectorbt.generic import nb as mojo_generic
from mojo_vectorbt.returns import nb as mojo_returns
from mojo_vectorbt.signals import nb as mojo_signals


def timeit(fn, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as file:
            for line in file:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def return_matrix(rows: int, cols: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.ascontiguousarray(rng.normal(0.0002, 0.015, (rows, cols)))
    values[::100_003, 0] = np.nan
    return values


def cases():
    rolling = return_matrix(1_000_000, 8)
    yield (
        "rolling_mean_nb (1M x 8, window 128)",
        lambda: mojo_generic.rolling_mean_nb(rolling, 128, 64),
        lambda: vbt.generic.nb.rolling_mean_nb(rolling, 128, 64),
    )
    yield (
        "rolling_std_nb (1M x 8, window 128)",
        lambda: mojo_generic.rolling_std_nb(rolling, 128, 64, 1),
        lambda: vbt.generic.nb.rolling_std_nb(rolling, 128, 64, 1),
    )

    extremes = return_matrix(500_000, 8, 1)
    yield (
        "rolling_max_nb (500k x 8, window 128)",
        lambda: mojo_generic.rolling_max_nb(extremes, 128, 64),
        lambda: vbt.generic.nb.rolling_max_nb(extremes, 128, 64),
    )

    returns = return_matrix(2_000_000, 4, 2)
    equity = np.ascontiguousarray(
        100.0 * np.cumprod(1.0 + np.nan_to_num(returns), axis=0)
    )
    initial = np.full(4, 100.0)
    yield (
        "returns_nb (2M x 4)",
        lambda: mojo_returns.returns_nb(equity, initial),
        lambda: vbt.returns.nb.returns_nb(equity, initial),
    )
    yield (
        "drawdown_nb (2M x 4)",
        lambda: mojo_returns.drawdown_nb(returns),
        lambda: vbt.returns.nb.drawdown_nb(returns),
    )

    metrics = return_matrix(4_000_000, 4, 3)
    yield (
        "sharpe_ratio_nb (4M x 4)",
        lambda: mojo_returns.sharpe_ratio_nb(metrics, 252.0),
        lambda: vbt.returns.nb.sharpe_ratio_nb(metrics, 252.0),
    )

    quantiles = return_matrix(500_000, 4, 4)
    yield (
        "value_at_risk_nb (500k x 4)",
        lambda: mojo_returns.value_at_risk_nb(quantiles),
        lambda: vbt.returns.nb.value_at_risk_nb(quantiles),
    )

    rng = np.random.default_rng(5)
    entries = np.ascontiguousarray(rng.random((2_000_000, 8)) < 0.01)
    exits = np.ascontiguousarray(rng.random((2_000_000, 8)) < 0.01)
    yield (
        "clean_enex_nb (2M x 8)",
        lambda: mojo_signals.clean_enex_nb(entries, exits, True),
        lambda: vbt.signals.nb.clean_enex_nb(entries, exits, True),
    )


def main() -> None:
    print(f"Machine: {cpu_name()} ({os.cpu_count()} logical cores), {platform.system()} {platform.machine()}")
    print(f"Python {platform.python_version()}, NumPy {np.__version__}, vectorbt {vbt.__version__}")
    print()
    print("| case | mojo-vectorbt | vectorbt | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_fn, vectorbt_fn in cases():
        mojo_fn()
        vectorbt_fn()
        mojo_time = timeit(mojo_fn)
        vectorbt_time = timeit(vectorbt_fn)
        ratio = vectorbt_time / mojo_time
        outcome = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{vectorbt_time * 1e3:.2f} ms | {outcome} |"
        )


if __name__ == "__main__":
    main()
