"""Regression coverage for the non-rich short minute-policy fallback."""

from __future__ import annotations

import pytest

from extreme_price_movements.inference.strict_r3_live_execution import (
    _directional_trailing_stop,
)


def test_legacy_trailing_stop_ratchets_down_for_short() -> None:
    # Short from 100, with 4 points MFE and a one-ATR giveback allowance:
    # its protective buy stop must move down to 97, not remain at 105.
    assert _directional_trailing_stop(
        entry_price=100.0,
        current_stop=105.0,
        maximum_favourable=4.0,
        atr=1.0,
        giveback_atr=1.0,
        side="short",
    ) == pytest.approx(97.0)


def test_legacy_trailing_stop_never_loosens_for_short() -> None:
    assert _directional_trailing_stop(
        entry_price=100.0,
        current_stop=96.0,
        maximum_favourable=3.0,
        atr=1.0,
        giveback_atr=1.0,
        side="short",
    ) == pytest.approx(96.0)


def test_legacy_trailing_stop_preserves_long_behavior() -> None:
    assert _directional_trailing_stop(
        entry_price=100.0,
        current_stop=95.0,
        maximum_favourable=4.0,
        atr=1.0,
        giveback_atr=1.0,
        side="long",
    ) == pytest.approx(103.0)
