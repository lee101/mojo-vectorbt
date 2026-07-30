"""Drop-in numeric subset of :mod:`vectorbt.returns.nb`."""

from __future__ import annotations

import numpy as np

from .._lib import addr, float64_array, lib, matrix, restore


def get_return_nb(input_value: float, output_value: float) -> float:
    return float(lib().mvt_get_return(float(input_value), float(output_value)))


def _returns(value, init_value, ndim: int):
    values, rows, cols = matrix(value, ndim=ndim)
    initial = float64_array([init_value] if ndim == 1 else init_value)
    if initial.ndim != 1 or initial.shape[0] != cols:
        raise ValueError("init_value must have one value per column")
    result = np.empty((rows, cols), dtype=np.float64)
    lib().mvt_returns(addr(values), addr(initial), addr(result), rows, cols)
    return restore(result, ndim=ndim)


def returns_1d_nb(value: np.ndarray, init_value: float) -> np.ndarray:
    return _returns(value, init_value, 1)


def returns_nb(value: np.ndarray, init_value: np.ndarray) -> np.ndarray:
    return _returns(value, init_value, 2)


def _cumulative(returns, start_value: float, drawdown: bool, ndim: int):
    values, rows, cols = matrix(returns, ndim=ndim)
    result = np.empty((rows, cols), dtype=np.float64)
    lib().mvt_cumulative(
        addr(values), addr(result), rows, cols, float(start_value), int(drawdown)
    )
    return restore(result, ndim=ndim)


def cum_returns_1d_nb(returns: np.ndarray, start_value: float) -> np.ndarray:
    return _cumulative(returns, start_value, False, 1)


def cum_returns_nb(returns: np.ndarray, start_value: float) -> np.ndarray:
    return _cumulative(returns, start_value, False, 2)


def drawdown_1d_nb(returns: np.ndarray) -> np.ndarray:
    return _cumulative(returns, 100.0, True, 1)


def drawdown_nb(returns: np.ndarray) -> np.ndarray:
    return _cumulative(returns, 100.0, True, 2)


def _single(returns, op: int, p1=0.0, p2=0.0, p3=0.0, i1=0, ndim=1):
    values, rows, cols = matrix(returns, ndim=ndim)
    if op in (2, 3) and rows == 0 and cols > 0:
        raise ValueError(
            "zero-size array to reduction operation minimum which has no identity"
        )
    result = np.empty(cols, dtype=np.float64)
    lib().mvt_single_metric(
        addr(values),
        addr(result),
        rows,
        cols,
        op,
        float(p1),
        float(p2),
        float(p3),
        int(i1),
    )
    return float(result[0]) if ndim == 1 else result


def cum_returns_final_1d_nb(
    returns: np.ndarray, start_value: float = 0.0
) -> float:
    return _single(returns, 8, p1=start_value)


def cum_returns_final_nb(
    returns: np.ndarray, start_value: float = 0.0
) -> np.ndarray:
    return _single(returns, 8, p1=start_value, ndim=2)


def annualized_return_1d_nb(returns: np.ndarray, ann_factor: float) -> float:
    return _single(returns, 0, p1=ann_factor)


def annualized_return_nb(returns: np.ndarray, ann_factor: float) -> np.ndarray:
    return _single(returns, 0, p1=ann_factor, ndim=2)


def annualized_volatility_1d_nb(
    returns: np.ndarray,
    ann_factor: float,
    levy_alpha: float = 2.0,
    ddof: int = 1,
) -> float:
    return _single(returns, 1, p1=ann_factor, p2=levy_alpha, i1=ddof)


def annualized_volatility_nb(
    returns: np.ndarray,
    ann_factor: float,
    levy_alpha: float = 2.0,
    ddof: int = 1,
) -> np.ndarray:
    return _single(
        returns, 1, p1=ann_factor, p2=levy_alpha, i1=ddof, ndim=2
    )


def max_drawdown_1d_nb(returns: np.ndarray) -> float:
    return _single(returns, 2)


def max_drawdown_nb(returns: np.ndarray) -> np.ndarray:
    return _single(returns, 2, ndim=2)


def calmar_ratio_1d_nb(returns: np.ndarray, ann_factor: float) -> float:
    return _single(returns, 3, p1=ann_factor)


def calmar_ratio_nb(returns: np.ndarray, ann_factor: float) -> np.ndarray:
    return _single(returns, 3, p1=ann_factor, ndim=2)


def omega_ratio_1d_nb(
    returns: np.ndarray,
    ann_factor: float,
    risk_free: float = 0.0,
    required_return: float = 0.0,
) -> float:
    return _single(
        returns, 4, p1=ann_factor, p2=required_return, p3=risk_free
    )


def omega_ratio_nb(
    returns: np.ndarray,
    ann_factor: float,
    risk_free: float = 0.0,
    required_return: float = 0.0,
) -> np.ndarray:
    return _single(
        returns,
        4,
        p1=ann_factor,
        p2=required_return,
        p3=risk_free,
        ndim=2,
    )


def sharpe_ratio_1d_nb(
    returns: np.ndarray,
    ann_factor: float,
    risk_free: float = 0.0,
    ddof: int = 1,
) -> float:
    return _single(returns, 5, p1=ann_factor, p2=risk_free, i1=ddof)


def sharpe_ratio_nb(
    returns: np.ndarray,
    ann_factor: float,
    risk_free: float = 0.0,
    ddof: int = 1,
) -> np.ndarray:
    return _single(
        returns, 5, p1=ann_factor, p2=risk_free, i1=ddof, ndim=2
    )


def downside_risk_1d_nb(
    returns: np.ndarray, ann_factor: float, required_return: float = 0.0
) -> float:
    return _single(returns, 6, p1=ann_factor, p2=required_return)


def downside_risk_nb(
    returns: np.ndarray, ann_factor: float, required_return: float = 0.0
) -> np.ndarray:
    return _single(
        returns, 6, p1=ann_factor, p2=required_return, ndim=2
    )


def sortino_ratio_1d_nb(
    returns: np.ndarray, ann_factor: float, required_return: float = 0.0
) -> float:
    return _single(returns, 7, p1=ann_factor, p2=required_return)


def sortino_ratio_nb(
    returns: np.ndarray, ann_factor: float, required_return: float = 0.0
) -> np.ndarray:
    return _single(
        returns, 7, p1=ann_factor, p2=required_return, ndim=2
    )


def _pair(
    returns,
    benchmark_rets,
    op: int,
    ann_factor=0.0,
    risk_free=0.0,
    ddof=1,
    ndim=1,
):
    values, rows, cols = matrix(returns, ndim=ndim)
    benchmark, b_rows, b_cols = matrix(benchmark_rets, ndim=ndim)
    if (rows, cols) != (b_rows, b_cols):
        raise ValueError("returns and benchmark_rets must have the same shape")
    result = np.empty(cols, dtype=np.float64)
    lib().mvt_pair_metric(
        addr(values),
        addr(benchmark),
        addr(result),
        rows,
        cols,
        op,
        float(ann_factor),
        float(risk_free),
        int(ddof),
    )
    return float(result[0]) if ndim == 1 else result


def information_ratio_1d_nb(
    returns: np.ndarray, benchmark_rets: np.ndarray, ddof: int = 1
) -> float:
    return _pair(returns, benchmark_rets, 0, ddof=ddof)


def information_ratio_nb(
    returns: np.ndarray, benchmark_rets: np.ndarray, ddof: int = 1
) -> np.ndarray:
    return _pair(returns, benchmark_rets, 0, ddof=ddof, ndim=2)


def beta_1d_nb(returns: np.ndarray, benchmark_rets: np.ndarray) -> float:
    return _pair(returns, benchmark_rets, 1)


def beta_nb(returns: np.ndarray, benchmark_rets: np.ndarray) -> np.ndarray:
    return _pair(returns, benchmark_rets, 1, ndim=2)


def alpha_1d_nb(
    returns: np.ndarray,
    benchmark_rets: np.ndarray,
    ann_factor: float,
    risk_free: float = 0.0,
) -> float:
    return _pair(
        returns, benchmark_rets, 2, ann_factor=ann_factor, risk_free=risk_free
    )


def alpha_nb(
    returns: np.ndarray,
    benchmark_rets: np.ndarray,
    ann_factor: float,
    risk_free: float = 0.0,
) -> np.ndarray:
    return _pair(
        returns,
        benchmark_rets,
        2,
        ann_factor=ann_factor,
        risk_free=risk_free,
        ndim=2,
    )


def _quantile(returns, op: int, cutoff=0.05, ndim=1):
    cutoff = float(cutoff)
    if not np.isfinite(cutoff) or cutoff < 0.0 or cutoff > 1.0:
        raise ValueError("cutoff must be between 0 and 1")
    values, rows, cols = matrix(returns, ndim=ndim)
    scratch = np.empty(max(rows * cols, 1), dtype=np.float64)
    result = np.empty(cols, dtype=np.float64)
    lib().mvt_quantile_metric(
        addr(values), addr(scratch), addr(result), rows, cols, op, cutoff
    )
    return float(result[0]) if ndim == 1 else result


def tail_ratio_1d_nb(returns: np.ndarray) -> float:
    return _quantile(returns, 0)


def tail_ratio_nb(returns: np.ndarray) -> np.ndarray:
    return _quantile(returns, 0, ndim=2)


def value_at_risk_1d_nb(
    returns: np.ndarray, cutoff: float = 0.05
) -> float:
    return _quantile(returns, 1, cutoff)


def value_at_risk_nb(
    returns: np.ndarray, cutoff: float = 0.05
) -> np.ndarray:
    return _quantile(returns, 1, cutoff, ndim=2)


def cond_value_at_risk_1d_nb(
    returns: np.ndarray, cutoff: float = 0.05
) -> float:
    return _quantile(returns, 2, cutoff)


def cond_value_at_risk_nb(
    returns: np.ndarray, cutoff: float = 0.05
) -> np.ndarray:
    return _quantile(returns, 2, cutoff, ndim=2)
