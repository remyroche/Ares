from __future__ import annotations


def assert_units(cost) -> None:
    assert 0.0 <= float(cost.fee_side) < 0.01, "fee_side must be a fraction (e.g., 0.0005 for 5 bps)"
    assert 0.0 <= float(cost.slippage_side) < 0.01, "slippage_side must be a fraction"


def assert_pos_w(pos_w: float) -> None:
    assert -2.0 <= float(pos_w) <= 2.0, "pos_w should be fraction-of-equity (typical in [-1, 1])"
