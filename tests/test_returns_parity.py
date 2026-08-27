import inspect

import numpy as np
import pytest
import vectorbt as vbt

from mojo_vectorbt.returns import nb as mojo_nb


@pytest.fixture(scope="module")
def return_data():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0005, 0.018, size=(503, 4))
    benchmark = rng.normal(0.0003, 0.012, size=(503, 4))
    returns[2, 1] = np.nan
    returns[100:103, 2] = np.nan
    benchmark[2, 1] = np.nan
    return returns[:, 0].copy(), benchmark[:, 0].copy(), returns, benchmark


@pytest.mark.parametrize(
    "input_value,output_value",
    [
        (100.0, 110.0),
        (-100.0, -90.0),
        (0.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.0, np.nan),
    ],
)
def test_get_return_matches_vectorbt(input_value, output_value):
    actual = mojo_nb.get_return_nb(input_value, output_value)
    expected = vbt.returns.nb.get_return_nb(input_value, output_value)
    assert actual == pytest.approx(expected, nan_ok=True)


def test_returns_from_equity_match_vectorbt():
    values = np.array(
        [[100.0, 0.0], [105.0, 2.0], [95.0, -2.0], [0.0, -1.0], [2.0, 0.0]]
    )
    initial = np.array([100.0, 0.0])
    assert np.allclose(
        mojo_nb.returns_nb(values, initial),
        vbt.returns.nb.returns_nb(values, initial),
        equal_nan=True,
    )
    assert np.allclose(
        mojo_nb.returns_1d_nb(values[:, 0], 100.0),
        vbt.returns.nb.returns_1d_nb(values[:, 0], 100.0),
        equal_nan=True,
    )


def test_return_and_drawdown_simd_tails_match_vectorbt():
    values = np.array(
        [
            [100.0, -100.0, 0.0, 50.0, 80.0],
            [101.0, -99.0, 2.0, np.nan, 0.0],
            [0.0, 0.0, -2.0, 55.0, 82.0],
            [2.0, -1.0, 0.0, 56.0, 81.0],
        ]
    )
    initial = np.array([100.0, -100.0, 0.0, 50.0, 80.0])
    actual_returns = mojo_nb.returns_nb(values, initial)
    expected_returns = vbt.returns.nb.returns_nb(values, initial)
    assert np.allclose(actual_returns, expected_returns, equal_nan=True)

    rng = np.random.default_rng(91)
    returns = np.ascontiguousarray(rng.normal(0.0005, 0.018, size=(503, 5)))
    returns[::71, 4] = np.nan
    assert np.allclose(
        mojo_nb.drawdown_nb(returns),
        vbt.returns.nb.drawdown_nb(returns),
        equal_nan=True,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "name,args",
    [
        ("cum_returns_1d_nb", (0.0,)),
        ("cum_returns_1d_nb", (100.0,)),
        ("drawdown_1d_nb", ()),
    ],
)
def test_1d_return_transforms_match_vectorbt(return_data, name, args):
    one, _, _, _ = return_data
    assert np.allclose(
        getattr(mojo_nb, name)(one, *args),
        getattr(vbt.returns.nb, name)(one, *args),
        equal_nan=True,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "name,args",
    [
        ("cum_returns_nb", (0.0,)),
        ("cum_returns_nb", (100.0,)),
        ("drawdown_nb", ()),
    ],
)
def test_2d_return_transforms_match_vectorbt(return_data, name, args):
    _, _, returns, _ = return_data
    assert np.allclose(
        getattr(mojo_nb, name)(returns, *args),
        getattr(vbt.returns.nb, name)(returns, *args),
        equal_nan=True,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "name,args",
    [
        ("cum_returns_final_1d_nb", (0.0,)),
        ("cum_returns_final_1d_nb", (100.0,)),
        ("annualized_return_1d_nb", (252.0,)),
        ("annualized_volatility_1d_nb", (252.0, 2.0, 1)),
        ("max_drawdown_1d_nb", ()),
        ("calmar_ratio_1d_nb", (252.0,)),
        ("omega_ratio_1d_nb", (252.0, 0.0001, 0.05)),
        ("sharpe_ratio_1d_nb", (252.0, 0.0001, 1)),
        ("downside_risk_1d_nb", (252.0, 0.001)),
        ("sortino_ratio_1d_nb", (252.0, 0.001)),
        ("tail_ratio_1d_nb", ()),
        ("value_at_risk_1d_nb", (0.05,)),
        ("cond_value_at_risk_1d_nb", (0.05,)),
    ],
)
def test_1d_metrics_match_vectorbt(return_data, name, args):
    one, _, _, _ = return_data
    actual = getattr(mojo_nb, name)(one, *args)
    expected = getattr(vbt.returns.nb, name)(one, *args)
    assert actual == pytest.approx(expected, rel=2e-9, abs=1e-12, nan_ok=True)


@pytest.mark.parametrize(
    "name,args",
    [
        ("cum_returns_final_nb", (0.0,)),
        ("annualized_return_nb", (252.0,)),
        ("annualized_volatility_nb", (252.0, 2.0, 1)),
        ("max_drawdown_nb", ()),
        ("calmar_ratio_nb", (252.0,)),
        ("omega_ratio_nb", (252.0, 0.0001, 0.05)),
        ("sharpe_ratio_nb", (252.0, 0.0001, 1)),
        ("downside_risk_nb", (252.0, 0.001)),
        ("sortino_ratio_nb", (252.0, 0.001)),
        ("tail_ratio_nb", ()),
        ("value_at_risk_nb", (0.05,)),
        ("cond_value_at_risk_nb", (0.05,)),
    ],
)
def test_2d_metrics_match_vectorbt(return_data, name, args):
    _, _, returns, _ = return_data
    actual = getattr(mojo_nb, name)(returns, *args)
    expected = getattr(vbt.returns.nb, name)(returns, *args)
    assert np.allclose(
        actual, expected, equal_nan=True, rtol=2e-9, atol=1e-12
    )


@pytest.mark.parametrize(
    "name,args",
    [
        ("information_ratio_1d_nb", (1,)),
        ("beta_1d_nb", ()),
        ("alpha_1d_nb", (252.0, 0.0001)),
    ],
)
def test_1d_benchmark_metrics_match_vectorbt(return_data, name, args):
    one, benchmark_one, _, _ = return_data
    actual = getattr(mojo_nb, name)(one, benchmark_one, *args)
    expected = getattr(vbt.returns.nb, name)(one, benchmark_one, *args)
    assert actual == pytest.approx(expected, rel=2e-10, abs=1e-12, nan_ok=True)


@pytest.mark.parametrize(
    "name,args",
    [
        ("information_ratio_nb", (1,)),
        ("beta_nb", ()),
        ("alpha_nb", (252.0, 0.0001)),
    ],
)
def test_2d_benchmark_metrics_match_vectorbt(return_data, name, args):
    _, _, returns, benchmark = return_data
    actual = getattr(mojo_nb, name)(returns, benchmark, *args)
    expected = getattr(vbt.returns.nb, name)(returns, benchmark, *args)
    assert np.allclose(
        actual, expected, equal_nan=True, rtol=2e-10, atol=1e-12
    )


@pytest.mark.parametrize(
    "name",
    ["tail_ratio_1d_nb", "value_at_risk_1d_nb", "cond_value_at_risk_1d_nb"],
)
def test_all_nan_quantiles_match_vectorbt(name):
    values = np.full(11, np.nan)
    assert np.isnan(getattr(mojo_nb, name)(values))
    assert np.isnan(getattr(vbt.returns.nb, name)(values))


def test_sharpe_simd_column_tail_matches_vectorbt():
    rng = np.random.default_rng(71)
    returns = np.ascontiguousarray(rng.normal(0.0004, 0.017, size=(509, 5)))
    returns[::53, 1] = np.nan
    returns[::47, 4] = np.nan
    actual = mojo_nb.sharpe_ratio_nb(returns, 252.0, 0.0001, 1)
    expected = vbt.returns.nb.sharpe_ratio_nb(returns, 252.0, 0.0001, 1)
    assert np.allclose(
        actual, expected, equal_nan=True, rtol=2e-9, atol=1e-12
    )


@pytest.mark.parametrize("rows", [8191, 8192])
def test_value_at_risk_parallel_threshold_matches_vectorbt(rows):
    rng = np.random.default_rng(rows)
    returns = np.ascontiguousarray(rng.normal(size=(rows, 4)))
    returns[::997, 0] = np.nan
    actual = mojo_nb.value_at_risk_nb(returns, 0.137)
    expected = vbt.returns.nb.value_at_risk_nb(returns, 0.137)
    assert np.allclose(
        actual, expected, equal_nan=True, rtol=2e-9, atol=1e-12
    )


@pytest.mark.parametrize("rows", [249_999, 250_000])
def test_drawdown_parallel_threshold_matches_vectorbt(rows):
    rng = np.random.default_rng(rows)
    returns = np.ascontiguousarray(rng.normal(0.0002, 0.01, size=(rows, 4)))
    returns[::10_007, 2] = np.nan
    actual = mojo_nb.drawdown_nb(returns)
    expected = vbt.returns.nb.drawdown_nb(returns)
    assert np.allclose(
        actual, expected, equal_nan=True, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("cutoff", [0.0, 0.05, 0.137, 0.5, 0.95, 1.0])
def test_value_at_risk_selection_edges_match_vectorbt(cutoff):
    values = np.array(
        [np.nan, 3.0, -2.0, 1.0, 1.0, -2.0, 8.0, 0.0, 4.0, np.nan]
    )
    actual = mojo_nb.value_at_risk_1d_nb(values, cutoff)
    expected = vbt.returns.nb.value_at_risk_1d_nb(values, cutoff)
    assert actual == pytest.approx(expected, rel=2e-9, abs=1e-12, nan_ok=True)


@pytest.mark.parametrize("name", ["value_at_risk_1d_nb", "cond_value_at_risk_1d_nb"])
@pytest.mark.parametrize("cutoff", [-0.01, 1.01, np.nan, np.inf])
def test_quantile_cutoff_rejects_out_of_bounds_indices(name, cutoff):
    with pytest.raises(ValueError, match="between 0 and 1"):
        getattr(mojo_nb, name)(np.arange(5.0), cutoff)


def test_empty_max_drawdown_matches_vectorbt_error():
    with pytest.raises(ValueError, match="zero-size array"):
        mojo_nb.max_drawdown_1d_nb(np.empty(0))
    with pytest.raises(ValueError, match="zero-size array"):
        vbt.returns.nb.max_drawdown_1d_nb(np.empty(0))


def test_returns_signatures_match_vectorbt():
    names = [
        name
        for name in vars(mojo_nb)
        if not name.startswith("_") and name.endswith("_nb")
    ]
    for name in names:
        ours = inspect.signature(getattr(mojo_nb, name))
        theirs = inspect.signature(getattr(vbt.returns.nb, name))
        assert list(ours.parameters) == list(theirs.parameters)
        assert [p.default for p in ours.parameters.values()] == [
            p.default for p in theirs.parameters.values()
        ]
