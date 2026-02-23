from __future__ import annotations

from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd

from extreme_price_movements.pnl import CostModel, trade_return_net_vec
from extreme_price_movements.optimization import RiskBudgetConfig, score_backtest_risk_budgeted
from extreme_price_movements.tpsl_optimiser.metrics_utils import compute_comprehensive_metrics


def sigmoid_sizing(conf: np.ndarray, k: float, c0: float, s_min: float = 0.03, s_max: float = 0.15) -> np.ndarray:
    sig = 1.0 / (1.0 + np.exp(-k * (conf - c0)))
    return s_min + (s_max - s_min) * sig


def compute_equity_metrics(exit_prices, entry_prices, is_long, confidences, k, c0, fee_pct: float = 0.005, initial_capital: float = 100000.0, cost: CostModel | None = None):
    n = len(exit_prices)
    if n == 0:
        return {
            "Score": 0.0,
            "PnL": 0.0,
            "Sortino": 0.0,
            "MaxDD": 0.0,
            "UlcerIndex": 0.0,
            "RecoverySpeed": 0.0,
            "xbar": 0.0,
            "p_not_recovered": 0.0,
            "ui_violation": 0.0,
            "x_violation": 0.0,
            "equity": np.array([]),
            "returns": np.array([]),
            "position": np.array([]),
        }

    pos_sizes = sigmoid_sizing(confidences, k, c0)

    raw_rets = np.where(
        is_long == 1,
        (exit_prices - entry_prices) / entry_prices,
        (entry_prices - exit_prices) / entry_prices,
    )
    cost = cost or CostModel(fee_side=float(fee_pct) / 2.0)
    net_rets = trade_return_net_vec(raw_ret_underlying=raw_rets, side=np.ones(len(raw_rets)), pos_w=pos_sizes, cost=cost)

    equity = np.empty(n + 1, dtype=float)
    equity[0] = initial_capital
    curr_eq = initial_capital
    for i in range(n):
        curr_eq *= (1.0 + net_rets[i])
        equity[i + 1] = curr_eq

    neg = net_rets[net_rets < 0]
    sortino = float(np.mean(net_rets) / np.std(neg)) if neg.size and np.std(neg) > 1e-12 else 0.0
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max((peak - equity) / np.maximum(peak, 1e-12)))
    pnl_total = float((equity[-1] - initial_capital) / initial_capital)

    rb = score_backtest_risk_budgeted(
        r=net_rets,
        x=pos_sizes,
        cfg=RiskBudgetConfig(ui_max=0.05, x_min=0.03, lambda_rs=0.10, hard_fail=True),
    )

    return {
        "Score": float(rb["score"]),
        "PnL": pnl_total,
        "Sortino": sortino,
        "MaxDD": max_dd,
        "UlcerIndex": float(rb["UlcerIndex"]),
        "RecoverySpeed": float(rb["RecoverySpeed"]),
        "xbar": float(rb["xbar"]),
        "p_not_recovered": float(rb["p_not_recovered"]),
        "ui_violation": float(rb["ui_violation"]),
        "x_violation": float(rb["x_violation"]),
        "equity": equity,
        "returns": net_rets,
        "position": pos_sizes,
    }


def optimise_position_sizing(trades: pd.DataFrame, exit_prices, entry_prices, is_long, confidences, test_split_idx: int = 0, fee_pct: float = 0.005, cost: CostModel | None = None, init_params: dict | None = None) -> Tuple[dict, pd.DataFrame]:
    # Split for optimization vs reporting
    n = len(exit_prices)
    split = test_split_idx if test_split_idx > 0 else n

    # Indices
    train_idx = np.arange(split)
    test_idx = np.arange(split, n)
    has_test = len(test_idx) > 0

    results = []
    trials_data = []

    k_grid = list(np.linspace(2.0, 20.0, 10))
    c0_grid = list(np.linspace(0.55, 0.85, 13))
    if init_params:
        k_grid.append(float(init_params.get("k", 8.0)))
        c0_grid.append(float(init_params.get("c0", 0.7)))

    for k, c0 in product(sorted(set(k_grid)), sorted(set(c0_grid))):
        # Train Selection
        train_metrics = compute_equity_metrics(
            exit_prices[train_idx], entry_prices[train_idx], is_long[train_idx], confidences[train_idx],
            float(k), float(c0), fee_pct=fee_pct, cost=cost
        ) if len(train_idx) > 0 else {"Score": 0.0, "PnL": 0.0, "Sortino": 0.0, "MaxDD": 0.0, "UlcerIndex": 0.0, "RecoverySpeed": 0.0, "xbar": 0.0, "p_not_recovered": 0.0, "ui_violation": 0.0, "x_violation": 0.0}

        results.append((k, c0, train_metrics["Score"], train_metrics["PnL"], train_metrics["Sortino"], train_metrics["MaxDD"], train_metrics["UlcerIndex"], train_metrics["RecoverySpeed"], train_metrics["xbar"], train_metrics["p_not_recovered"], train_metrics["ui_violation"], train_metrics["x_violation"]))

        trial_metrics = {
            "k": k,
            "c0": c0,
            "train_score": train_metrics["Score"],
            "train_pnl": train_metrics["PnL"],
            "train_sortino": train_metrics["Sortino"],
            "train_max_dd": train_metrics["MaxDD"],
            "train_ulcer_index": train_metrics["UlcerIndex"],
            "train_recovery_speed": train_metrics["RecoverySpeed"],
            "train_xbar": train_metrics["xbar"],
            "train_p_not_recovered": train_metrics["p_not_recovered"],
            "train_ui_violation": train_metrics["ui_violation"],
            "train_x_violation": train_metrics["x_violation"],
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

            m = compute_comprehensive_metrics(test_trades, fee_pct=fee_pct, cost=cost) # uses pos_size column
            trial_metrics.update(m)

        trials_data.append(trial_metrics)

    arr = np.array(results, dtype=float)
    train_scores = arr[:, 2]
    best_idx = int(np.argmax(train_scores))
    best = arr[best_idx]

    return {"k": float(best[0]), "c0": float(best[1]), "s_min": 0.03, "s_max": 0.15}, pd.DataFrame(trials_data)
