from __future__ import annotations

import pandas as pd

from scripts.materialize_strict_r3_target_free_hourly_grid_v2 import (
    _decision_open_source_validity,
)


def _series(value: object) -> pd.Series:
    return pd.Series([value], index=pd.DatetimeIndex(["2026-08-16T05:00:00Z"]))


def test_zero_volume_shared_15m_open_requires_book_agreement() -> None:
    """An observed-but-flat trade row cannot authorize a stale live entry."""
    valid, requires_book = _decision_open_source_validity(
        source=_series("shared_15m"),
        trade_volume=_series(0.0),
        hourly_volume=_series(float("nan")),
        bid=_series(0.007451),
        ask=_series(0.007481),
        book_deviation_bps=_series(144.66),
    )

    assert bool(requires_book.iloc[0])
    assert not bool(valid.iloc[0])


def test_zero_volume_shared_15m_open_can_pass_when_book_agrees() -> None:
    valid, requires_book = _decision_open_source_validity(
        source=_series("shared_15m"),
        trade_volume=_series(0.0),
        hourly_volume=_series(float("nan")),
        bid=_series(0.007351),
        ask=_series(0.007365),
        book_deviation_bps=_series(9.5),
    )

    assert bool(requires_book.iloc[0])
    assert bool(valid.iloc[0])


def test_positive_volume_direct_15m_open_does_not_need_book_validation() -> None:
    valid, requires_book = _decision_open_source_validity(
        source=_series("raw_15m"),
        trade_volume=_series(1.0),
        hourly_volume=_series(float("nan")),
        bid=_series(float("nan")),
        ask=_series(float("nan")),
        book_deviation_bps=_series(float("nan")),
    )

    assert not bool(requires_book.iloc[0])
    assert bool(valid.iloc[0])


def test_unknown_volume_direct_15m_open_rejects_large_book_gap() -> None:
    """A current-bar direct price cannot route when its book is far away."""
    valid, requires_book = _decision_open_source_validity(
        source=_series("raw_15m"),
        trade_volume=_series(float("nan")),
        hourly_volume=_series(float("nan")),
        bid=_series(0.00004015),
        ask=_series(0.00004039),
        book_deviation_bps=_series(471.87),
    )

    assert bool(requires_book.iloc[0])
    assert not bool(valid.iloc[0])
