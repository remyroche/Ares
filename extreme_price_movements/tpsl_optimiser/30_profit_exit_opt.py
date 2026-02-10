from __future__ import annotations

from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd

from extreme_price_movements.tpsl_optimiser.metrics_utils import compute_comprehensive_metrics


def optimise_profit_exit(trades: pd.DataFrame, trade_returns: np.ndarray, tp_pct_entry: np.ndarray, fee_pct: float = 0.005, test_split_idx: int = 0) -> Tuple[dict, pd.DataFrame]:
    """Optimise trailing controls against net PnL.

    This step intentionally consumes frozen TP/SL from previous steps.
    """

    # Split for optimization vs reporting
    n = len(trades)
    split = test_split_idx if test_split_idx > 0 else n

    train_mask = np.zeros(n, dtype=bool)
    train_mask[:split] = True
    test_mask = ~train_mask
    has_test = np.any(test_mask)

    best = None
    trials_data = []

    for act_n, be_act_n, d_min, d_max in product([0.5, 0.75, 1.0], [0.25, 0.5, 0.75], [0.1, 0.2, 0.3], [0.5, 0.8, 1.0]):
        # trail_penalty is % of TP? No, it's clipped to [0,1] * tp_pct_entry
        # Wait, if act_n < be_act_n, penalty is 0.
        # tp_pct_entry is the TP distance (e.g. 0.02).
        # trail_penalty is roughly the give-back amount?
        trail_penalty = np.clip((act_n - be_act_n), 0.0, 1.0) * np.clip(tp_pct_entry, 0.0, 10.0)

        # Apply penalty to raw returns (net of penalty)
        # Note: original code subtracts fee_pct here: net = trade_returns - fee_pct - penalty
        # We replicate this logic for selection score.
        net_ret = trade_returns - fee_pct - trail_penalty * 0.01

        # Train Selection
        net_train = net_ret[train_mask]
        score = float(np.sum(net_train))

        if (best is None) or (score > best[0]):
            best = (score, act_n, be_act_n, d_min, d_max)

        trial_metrics = {
            "act_n": act_n,
            "be_act_n": be_act_n,
            "d_min": d_min,
            "d_max": d_max,
            "train_score_pnl": score,
        }

        # Test Reporting
        if has_test:
            test_penalty = trail_penalty[test_mask] * 0.01
            # Apply penalty to returns (gross -> net-penalty)
            # trade_returns is Gross return from previous step? No, it's raw return.
            # In optimise.py: raw_returns = ...
            # So `net_ret` calculation above assumes `trade_returns` is Gross.
            # `compute_comprehensive_metrics` calculates Net from Exit/Entry - Fee.
            # So we need to update Exit Price to reflect `trade_returns - penalty`.
            # Note: fee is subtracted inside compute_comprehensive_metrics.
            # So here we update exit price based on `trade_returns - penalty`.

            test_raw = trade_returns[test_mask]
            test_adjusted = test_raw - test_penalty

            test_trades = trades.iloc[test_mask].copy()
            is_long = test_trades["is_long"].astype(int).to_numpy()
            new_exit = np.where(is_long == 1,
                                test_trades["entry_price"] * (1 + test_adjusted),
                                test_trades["entry_price"] * (1 - test_adjusted))
            test_trades["exit_price"] = new_exit
            test_trades["exit_reason"] = "profit_opt"

            m = compute_comprehensive_metrics(test_trades, fee_pct=fee_pct)
            trial_metrics.update(m)

        trials_data.append(trial_metrics)

    _, act_n, be_act_n, d_min, d_max = best

    best_params = {
        "act_n": float(act_n),
        "be_act_n": float(be_act_n),
        "d_min": float(d_min),
        "d_max": float(d_max),
        "fee_pct": float(fee_pct),
    }

    return best_params, pd.DataFrame(trials_data)
