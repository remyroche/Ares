from __future__ import annotations

from itertools import product

import numpy as np


def optimise_profit_exit(trade_returns: np.ndarray, tp_pct_entry: np.ndarray, fee_pct: float = 0.005) -> dict:
    """Optimise trailing controls against net PnL.

    This step intentionally consumes frozen TP/SL from previous steps.
    """
    best = None
    for act_n, be_act_n, d_min, d_max in product([0.5, 0.75, 1.0], [0.25, 0.5, 0.75], [0.1, 0.2, 0.3], [0.5, 0.8, 1.0]):
        trail_penalty = np.clip((act_n - be_act_n), 0.0, 1.0) * np.clip(tp_pct_entry, 0.0, 10.0)
        net = trade_returns - fee_pct - trail_penalty * 0.01
        score = float(np.sum(net))
        if (best is None) or (score > best[0]):
            best = (score, act_n, be_act_n, d_min, d_max)
    _, act_n, be_act_n, d_min, d_max = best
    return {
        "act_n": float(act_n),
        "be_act_n": float(be_act_n),
        "d_min": float(d_min),
        "d_max": float(d_max),
        "fee_pct": float(fee_pct),
    }
