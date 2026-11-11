#!/usr/bin/env python3
"""Regression: exact NIFTY symbol filter (guards against BANKNIFTY/FINNIFTY substring leak)."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from nifty_analyst.adapters import iv_history_builder
from nifty_analyst.adapters.iv_history_builder import (
    filter_exact_nifty_udiff_rows,
    validate_nifty_futures_row,
    validate_nifty_options_row,
)


def test_fetcher_filter_source() -> None:
    """Verify `iv_history_builder` uses exact `== 'NIFTY'` equality and never substring `in`."""
    src = inspect.getsource(iv_history_builder.filter_exact_nifty_udiff_rows)
    assert 'row.get("TckrSymb") == "NIFTY"' in src
    assert '"NIFTY" in' not in src and "'NIFTY' in" not in src

    mixed_rows = [
        {"TckrSymb": "NIFTY", "FinInstrmTp": "IDF", "SttlmPric": "24210.5", "UndrlygPric": "24180.5"},
        {"TckrSymb": "BANKNIFTY", "FinInstrmTp": "IDF", "SttlmPric": "51850.0", "UndrlygPric": "51700.0"},
        {"TckrSymb": "FINNIFTY", "FinInstrmTp": "IDO", "SttlmPric": "120.0", "UndrlygPric": "23900.0"},
        {"TckrSymb": "MIDCPNIFTY", "FinInstrmTp": "IDO", "SttlmPric": "85.0", "UndrlygPric": "12400.0"},
    ]
    filtered = filter_exact_nifty_udiff_rows(mixed_rows)
    assert len(filtered) == 1
    assert filtered[0]["TckrSymb"] == "NIFTY"


def test_futures_data_ranges() -> None:
    """Verify valid NIFTY futures rows pass and out-of-range / BANKNIFTY / >6% basis rows fail."""
    px, und = validate_nifty_futures_row(
        {"TckrSymb": "NIFTY", "SttlmPric": "24235.0", "UndrlygPric": "24180.5"}
    )
    assert 15000 < px < 35000
    assert 15000 < und < 35000
    assert abs(px - und) / und < 0.06

    with pytest.raises(ValueError, match="Non-NIFTY"):
        validate_nifty_futures_row(
            {"TckrSymb": "BANKNIFTY", "SttlmPric": "51850.0", "UndrlygPric": "51700.0"}
        )

    with pytest.raises(ValueError, match="outside valid NIFTY range"):
        validate_nifty_futures_row(
            {"TckrSymb": "NIFTY", "SttlmPric": "51850.0", "UndrlygPric": "51700.0"}
        )

    with pytest.raises(ValueError, match="basis"):
        validate_nifty_futures_row(
            {"TckrSymb": "NIFTY", "SttlmPric": "26500.0", "UndrlygPric": "24000.0"}
        )


def test_options_underlying_range() -> None:
    """Verify NIFTY options rows validate underlying range [15000, 35000]."""
    u = validate_nifty_options_row({"TckrSymb": "NIFTY", "UndrlygPric": "24180.5"})
    assert 15000 < u < 35000

    with pytest.raises(ValueError, match="outside valid NIFTY range"):
        validate_nifty_options_row({"TckrSymb": "NIFTY", "UndrlygPric": "51200.0"})


if __name__ == "__main__":
    test_fetcher_filter_source()
    test_futures_data_ranges()
    test_options_underlying_range()
    print("symbol filter regression tests PASS")
