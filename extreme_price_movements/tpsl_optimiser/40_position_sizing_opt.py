from __future__ import annotations

from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd

from extreme_price_movements.tpsl_optimiser.metrics_utils import compute_comprehensive_metrics


def sigmoid_sizing(conf: np.ndarray, k: float, c0: float, s_min: float = 0.03, s_max: float = 0.15) -> np.ndarray:
    sig = 1.0 / (1.0 + np.exp(-k * (conf - c0)))
    return s_min + (s_max - s_min) * sig


def compute_equity_metrics(exit_prices, entry_prices, is_long, confidences, k, c0, fee_pct: float = 0.005, initial_capital: float = 100000.0):
    n = len(exit_prices)
    if n == 0:
        return 0.0, 0.0, 0.0, np.array([])

    equity = np.zeros(n + 1)
    equity[0] = initial_capital
    returns = np.zeros(n)

    pos_sizes = sigmoid_sizing(confidences, k, c0)

    # Vectorized
    raw_rets = np.where(is_long == 1,
                        (exit_prices - entry_prices) / entry_prices,
                        (entry_prices - exit_prices) / entry_prices)
    net_rets = (raw_rets * pos_sizes) - fee_pct

    # Simple compounding loop for equity curve maxdd
    curr_eq = initial_capital
    for i in range(n):
        curr_eq *= (1.0 + net_rets[i])
        equity[i+1] = curr_eq
        returns[i] = net_rets[i]

    total_pnl = (equity[-1] - initial_capital) / initial_capital
    neg = returns[returns < 0]
    sortino = float(np.mean(returns) / np.std(neg)) if neg.size and np.std(neg) > 1e-12 else 0.0

    peak = np.maximum.accumulate(equity)
    # Avoid div by zero
    max_dd = float(np.max((peak - equity) / np.maximum(peak, 1e-12)))
    return total_pnl, sortino, max_dd, equity


def optimise_position_sizing(trades: pd.DataFrame, exit_prices, entry_prices, is_long, confidences, test_split_idx: int = 0) -> Tuple[dict, pd.DataFrame]:
    # Split for optimization vs reporting
    n = len(exit_prices)
    split = test_split_idx if test_split_idx > 0 else n

    # Indices
    train_idx = np.arange(split)
    test_idx = np.arange(split, n)
    has_test = len(test_idx) > 0

    results = []
    trials_data = []

    for k, c0 in product(np.linspace(2.0, 20.0, 10), np.linspace(0.55, 0.85, 13)):
        # Train Selection
        if len(train_idx) > 0:
            pnl, sortino, max_dd, _ = compute_equity_metrics(
                exit_prices[train_idx], entry_prices[train_idx], is_long[train_idx], confidences[train_idx],
                float(k), float(c0)
            )
        else:
            pnl, sortino, max_dd = 0.0, 0.0, 0.0

        results.append((k, c0, pnl, sortino, max_dd))

        trial_metrics = {
            "k": k,
            "c0": c0,
            "train_pnl": pnl,
            "train_sortino": sortino,
            "train_max_dd": max_dd,
        }

        # Test Reporting
        if has_test:
            test_conf = confidences[test_idx]
            pos_size = sigmoid_sizing(test_conf, float(k), float(c0))

            test_trades = trades.iloc[test_idx].copy()
            test_trades["pos_size"] = pos_size
            # Exit price should reflect previous optimizations?
            # optimise.py passes 'bucket_df["exit_price"]'.
            # bucket_df's exit_price is from trades loaded, but modified by previous steps?
            # Wait, optimise.py calculates raw_returns from bucket_df.
            # But doesn't update bucket_df["exit_price"] with TP/SL/RiskCut results!
            # The 'exit_prices' passed here are raw/original?
            # Or are they updated?
            # In optimise.py:
            # bucket_df = m00.load...
            # m10...
            # sl_pct = ...
            # risk_cut = m20...
            # raw_returns = ... (from bucket_df entry/exit)
            # profit = m30...
            # sizing = m40... (passed bucket_df["exit_price"])
            # The exit prices passed to m40 are the ORIGINAL exit prices.
            # This ignores the TP/SL and Risk Cut logic!
            # This seems like a flaw in the original pipeline if m40 is supposed to optimize on the *final* outcome.
            # But m40 optimizes sizing based on CONFIDENCE.
            # Maybe it assumes raw PnL distribution is preserved?
            # However, for my reporting, I should probably use the "best" outcomes from previous steps if possible.
            # Or just follow the pattern: report what *this step* sees.
            # Since I can't easily change the input to m40 in optimise.py without refactoring everything,
            # I will calculate metrics based on the input trades + new sizing.
            # The input trades have whatever exit price they have.
            # If the user wants cumulative effect, optimise.py needs to thread the state.
            # Given "optimise.py" structure, steps seem independent or cumulative in a way I might be missing.
            # But here I just report what happens if we apply sizing to THESE trades.

            m = compute_comprehensive_metrics(test_trades) # uses pos_size column
            trial_metrics.update(m)

        trials_data.append(trial_metrics)

    arr = np.array(results, dtype=float)
    # Score logic
    # Avoid zero std
    z = lambda x: np.zeros_like(x) if np.std(x) < 1e-12 else (x - np.mean(x)) / np.std(x)

    score = 0.6 * z(arr[:, 2]) + 0.3 * z(arr[:, 3]) - 0.1 * z(arr[:, 4])
    best_idx = int(np.argmax(score))
    best = arr[best_idx]

    return {"k": float(best[0]), "c0": float(best[1]), "s_min": 0.03, "s_max": 0.15}, pd.DataFrame(trials_data)
