import inspect

import numpy as np
import pytest
import vectorbt as vbt

from mojo_vectorbt.generic import nb as mojo_nb


@pytest.fixture(scope="module")
def arrays():
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(257, 5))
    matrix[3, 1] = np.nan
    matrix[10:14, 3] = np.nan
    matrix[100, :] = np.nan
    return matrix[:, 0].copy(), matrix


@pytest.mark.parametrize(
    "name,args",
    [
        ("fshift_1d_nb", (3, -7.0)),
        ("bshift_1d_nb", (3, -7.0)),
        ("diff_1d_nb", (4,)),
        ("pct_change_1d_nb", (4,)),
    ],
)
def test_1d_transforms_match_vectorbt(arrays, name, args):
    one, _ = arrays
    actual = getattr(mojo_nb, name)(one, *args)
    expected = getattr(vbt.generic.nb, name)(one, *args)
    assert np.allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize(
    "name,args",
    [
        ("fshift_nb", (3, -7.0)),
        ("bshift_nb", (3, -7.0)),
        ("diff_nb", (4,)),
        ("pct_change_nb", (4,)),
    ],
)
def test_2d_transforms_match_vectorbt(arrays, name, args):
    _, matrix = arrays
    actual = getattr(mojo_nb, name)(matrix, *args)
    expected = getattr(vbt.generic.nb, name)(matrix, *args)
    assert np.allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize(
    "name,args",
    [
        ("rolling_mean_1d_nb", (17, None)),
        ("rolling_mean_1d_nb", (17, 4)),
        ("rolling_std_1d_nb", (17, 4, 0)),
        ("rolling_std_1d_nb", (17, 4, 1)),
        ("rolling_min_1d_nb", (17, 4)),
        ("rolling_max_1d_nb", (17, 4)),
    ],
)
def test_1d_rolling_matches_vectorbt(arrays, name, args):
    one, _ = arrays
    actual = getattr(mojo_nb, name)(one, *args)
    expected = getattr(vbt.generic.nb, name)(one, *args)
    assert np.allclose(actual, expected, equal_nan=True, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "name,args",
    [
        ("rolling_mean_nb", (23, None)),
        ("rolling_mean_nb", (23, 3)),
        ("rolling_std_nb", (23, 3, 0)),
        ("rolling_std_nb", (23, 3, 2)),
        ("rolling_min_nb", (23, 3)),
        ("rolling_max_nb", (23, 3)),
    ],
)
def test_2d_rolling_matches_vectorbt(arrays, name, args):
    _, matrix = arrays
    actual = getattr(mojo_nb, name)(matrix, *args)
    expected = getattr(vbt.generic.nb, name)(matrix, *args)
    assert np.allclose(actual, expected, equal_nan=True, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", ["rolling_mean_nb", "rolling_std_nb", "rolling_min_nb", "rolling_max_nb"])
def test_rolling_validates_min_periods(arrays, name):
    _, matrix = arrays
    with pytest.raises(ValueError, match="minp must be <= window"):
        getattr(mojo_nb, name)(matrix, 3, 4)


@pytest.mark.parametrize(
    "name", ["fshift_1d_nb", "bshift_1d_nb", "diff_1d_nb", "pct_change_1d_nb"]
)
def test_shift_rejects_negative_periods(name):
    with pytest.raises(ValueError, match="non-negative"):
        getattr(mojo_nb, name)(np.arange(5.0), -1)


def test_noncontiguous_and_integer_inputs_are_safely_normalized():
    source = np.arange(20, dtype=np.int32).reshape(4, 5).T
    actual = mojo_nb.diff_nb(source, 1)
    expected = vbt.generic.nb.diff_nb(source, 1)
    assert np.allclose(actual, expected, equal_nan=True)


def test_unsafe_float64_narrowing_is_rejected():
    with pytest.raises(ValueError, match="exact float64 range"):
        mojo_nb.diff_1d_nb(np.array([2**60], dtype=np.int64))
    with pytest.raises(TypeError, match="complex"):
        mojo_nb.diff_1d_nb(np.array([1 + 2j]))


def test_generic_signatures_match_vectorbt():
    names = [
        name
        for name in vars(mojo_nb)
        if not name.startswith("_") and name.endswith("_nb")
    ]
    for name in names:
        ours = inspect.signature(getattr(mojo_nb, name))
        theirs = inspect.signature(getattr(vbt.generic.nb, name))
        assert list(ours.parameters) == list(theirs.parameters)
        assert [p.default for p in ours.parameters.values()] == [
            p.default for p in theirs.parameters.values()
        ]
