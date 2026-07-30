import inspect

import numpy as np
import pytest
import vectorbt as vbt

from mojo_vectorbt.signals import nb as mojo_nb


@pytest.mark.parametrize("entry_first", [False, True])
def test_clean_enex_1d_matches_vectorbt(entry_first):
    rng = np.random.default_rng(123)
    entries = rng.random(1000) < 0.08
    exits = rng.random(1000) < 0.08
    actual = mojo_nb.clean_enex_1d_nb(entries, exits, entry_first)
    expected = vbt.signals.nb.clean_enex_1d_nb(entries, exits, entry_first)
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])


@pytest.mark.parametrize("entry_first", [False, True])
def test_clean_enex_2d_matches_vectorbt(entry_first):
    rng = np.random.default_rng(321)
    entries = rng.random((1000, 7)) < 0.08
    exits = rng.random((1000, 7)) < 0.08
    actual = mojo_nb.clean_enex_nb(entries, exits, entry_first)
    expected = vbt.signals.nb.clean_enex_nb(entries, exits, entry_first)
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])


def test_simultaneous_signals_are_removed():
    entries = np.array([True, True, False, True, False])
    exits = np.array([True, False, True, False, True])
    clean_entries, clean_exits = mojo_nb.clean_enex_1d_nb(entries, exits, True)
    assert np.array_equal(clean_entries, [False, True, False, True, False])
    assert np.array_equal(clean_exits, [False, False, True, False, True])


def test_signal_conversion_uses_truth_values_without_uint8_narrowing():
    entries = np.array([256, 0, -256, 0])
    exits = np.array([0, 256, 0, -256])
    actual = mojo_nb.clean_enex_1d_nb(entries, exits, True)
    expected = vbt.signals.nb.clean_enex_1d_nb(
        entries.astype(bool), exits.astype(bool), True
    )
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])


def test_signal_signatures_match_vectorbt():
    for name in ("clean_enex_1d_nb", "clean_enex_nb"):
        ours = inspect.signature(getattr(mojo_nb, name))
        theirs = inspect.signature(getattr(vbt.signals.nb, name))
        assert list(ours.parameters) == list(theirs.parameters)
        assert [p.default for p in ours.parameters.values()] == [
            p.default for p in theirs.parameters.values()
        ]
