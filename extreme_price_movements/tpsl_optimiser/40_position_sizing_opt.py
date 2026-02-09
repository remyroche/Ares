from __future__ import annotations

from itertools import product

import numpy as np


def sigmoid_sizing(conf: np.ndarray, k: float, c0: float, s_min: float = 0.03, s_max: float = 0.15) -> np.ndarray:
    sig = 1.0 / (1.0 + np.exp(-k * (conf - c0)))
    return s_min + (s_max - s_min) * sig


def compute_equity_metrics(exit_prices, entry_prices, is_long, confidences, k, c0, fee_pct: float = 0.005, initial_capital: float = 100000.0):
    n = len(exit_prices)
    equity = np.zeros(n + 1)
    equity[0] = initial_capital
    returns = np.zeros(n)
    for i in range(n):
        pos_size = sigmoid_sizing(np.array([confidences[i]]), k, c0)[0]
        raw_ret = (exit_prices[i] - entry_prices[i]) / entry_prices[i] if is_long[i] == 1 else (entry_prices[i] - exit_prices[i]) / entry_prices[i]
        net_trade_ret = (raw_ret * pos_size) - fee_pct
        equity[i + 1] = equity[i] * (1.0 + net_trade_ret)
        returns[i] = net_trade_ret
    total_pnl = (equity[-1] - initial_capital) / initial_capital
    neg = returns[returns < 0]
    sortino = float(np.mean(returns) / np.std(neg)) if neg.size and np.std(neg) > 1e-12 else 0.0
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max((peak - equity) / np.maximum(peak, 1e-12)))
    return total_pnl, sortino, max_dd, equity


def optimise_position_sizing(exit_prices, entry_prices, is_long, confidences) -> dict:
    results = []
    for k, c0 in product(np.linspace(2.0, 20.0, 10), np.linspace(0.55, 0.85, 13)):
        pnl, sortino, max_dd, _ = compute_equity_metrics(exit_prices, entry_prices, is_long, confidences, float(k), float(c0))
        results.append((k, c0, pnl, sortino, max_dd))
    arr = np.array(results, dtype=float)
    z = lambda x: np.zeros_like(x) if np.std(x) < 1e-12 else (x - np.mean(x)) / np.std(x)
    score = 0.6 * z(arr[:, 2]) + 0.3 * z(arr[:, 3]) - 0.1 * z(arr[:, 4])
    best = arr[int(np.argmax(score))]
    return {"k": float(best[0]), "c0": float(best[1]), "s_min": 0.03, "s_max": 0.15}
