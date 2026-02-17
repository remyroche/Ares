from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TradeLedgerRow:
    asset: str
    t_entry: int
    t_exit: int
    entry_price: float
    exit_price: float
    side: int
    pos_w: float
    exit_reason: str
    raw_ret_underlying: float
    net_ret_equity: float
    cost_rt: float


def make_row(asset, t0, t1, p0, p1, side, pos_w, reason, cost, net_ret):
    row = TradeLedgerRow(
        asset=str(asset),
        t_entry=int(t0),
        t_exit=int(t1),
        entry_price=float(p0),
        exit_price=float(p1),
        side=int(side),
        pos_w=float(pos_w),
        exit_reason=str(reason),
        raw_ret_underlying=float(p1 / max(p0, 1e-12) - 1.0),
        net_ret_equity=float(net_ret),
        cost_rt=float(cost.round_trip),
    )
    return asdict(row)
