from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    """Trading cost model in FRACTION units.

    Example: fee_side=0.0025 means 25 bps per side.
    """

    fee_side: float
    slippage_side: float = 0.0

    @property
    def round_trip(self) -> float:
        return 2.0 * (self.fee_side + self.slippage_side)


def trade_return_net(
    raw_ret_underlying: float,
    side: int,
    pos_w: float,
    cost: CostModel,
) -> float:
    """Single-trade equity return contribution (fraction of equity)."""
    if side not in (+1, -1):
        raise ValueError("side must be +1 (long) or -1 (short)")
    w = abs(float(pos_w))
    raw_trade_ret = float(side) * float(raw_ret_underlying)
    return (w * raw_trade_ret) - (w * cost.round_trip)


def trade_return_net_vec(
    raw_ret_underlying: np.ndarray,
    side: np.ndarray,
    pos_w: np.ndarray,
    cost: CostModel,
) -> np.ndarray:
    """Vectorized net-return helper using the same convention as `trade_return_net`."""
    raw = np.asarray(raw_ret_underlying, dtype=float)
    s = np.asarray(side, dtype=float)
    w = np.abs(np.asarray(pos_w, dtype=float))
    return (w * (s * raw)) - (w * float(cost.round_trip))
