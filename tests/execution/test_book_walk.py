from __future__ import annotations

import pytest

from src.execution.book_walk import walk_book


def test_sell_walk_uses_bids_and_reports_quote_notional_cost() -> None:
    result = walk_book(
        bids=((100.0, 2.0), (99.0, 5.0)), asks=((101.0, 4.0),), side="sell", notional_quote=250.0,
    )
    assert not result.insufficient_depth
    assert result.executable_notional == pytest.approx(250.0)
    assert result.vwap < 100.0
    assert result.cost_vs_mid_bps > 0.0


def test_buy_walk_uses_asks_and_fails_closed_when_depth_is_insufficient() -> None:
    result = walk_book(
        bids=((100.0, 10.0),), asks=((101.0, 1.0),), side="buy", notional_quote=500.0,
    )
    assert result.insufficient_depth
    assert result.executable_notional == pytest.approx(101.0)
    # The partial fill is reported for audit, but downstream callers must
    # reject it via ``insufficient_depth`` rather than extrapolating it.
    assert result.vwap == pytest.approx(101.0)


def test_walk_requires_one_unambiguous_spot_quantity_semantic() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        walk_book(bids=((100.0, 1.0),), asks=((101.0, 1.0),), side="sell", quantity_base=1.0, notional_quote=100.0)
