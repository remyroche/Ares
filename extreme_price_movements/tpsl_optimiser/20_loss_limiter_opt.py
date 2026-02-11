from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from extreme_price_movements.tpsl_optimiser.metrics_utils import compute_comprehensive_metrics


class TriggerMode(str, Enum):
    AND = "AND"
    OR = "OR"
    TIMES = "TIMES"


@dataclass(frozen=True)
class TriggerParams:
    mode: TriggerMode = TriggerMode.TIMES
    theta_mae_min: float = 0.30
    theta0: float = 1.20
    theta_near_sl: Optional[float] = 0.85
    gamma: float = 1.0
    mae_cap: float = 2.0


def compute_times_gate(mae_n: float, theta_mae_min: float, gamma: float = 1.0, mae_cap: float = 2.0) -> float:
    if not np.isfinite(mae_n) or mae_n < theta_mae_min:
        return 0.0
    return float((min(mae_n, mae_cap) / max(theta_mae_min, 1e-12)) ** gamma)


def risk_cut_trigger(D_t: float, mae_n: float, p: TriggerParams) -> tuple[bool, float]:
    if (not np.isfinite(D_t)) or (not np.isfinite(mae_n)):
        return False, 0.0
    if p.mode == TriggerMode.AND:
        return (mae_n >= p.theta_mae_min) and (D_t >= p.theta0), float(D_t)
    if p.mode == TriggerMode.OR:
        if mae_n < p.theta_mae_min:
            return False, float(D_t)
        near_sl = (p.theta_near_sl is not None) and (mae_n >= p.theta_near_sl)
        return (D_t >= p.theta0) or near_sl, float(D_t)
    g = compute_times_gate(mae_n, p.theta_mae_min, p.gamma, p.mae_cap)
    score = float(D_t) * g
    return score >= p.theta0, score


def optimise_loss_limiter(trades: pd.DataFrame, sl_pct: np.ndarray, test_split_idx: int = 0) -> Tuple[dict, pd.DataFrame]:
    ret = np.where(trades["is_long"].astype(int).to_numpy() == 1,
                   (trades["exit_price"] - trades["entry_price"]) / trades["entry_price"],
                   (trades["entry_price"] - trades["exit_price"]) / trades["entry_price"])

    mae_n = np.clip(np.abs(trades.get("mae_pct", pd.Series(np.abs(ret))).to_numpy()) / np.maximum(sl_pct, 1e-6), 0, 3)
    speed_n = np.clip(np.abs(trades.get("ret_w", pd.Series(ret)).to_numpy()) / np.maximum(sl_pct, 1e-6), 0, 4)
    rv_n = np.clip(np.log(np.maximum(trades.get("rv", pd.Series(np.ones(len(trades)))).to_numpy(), 1.0)), 0, np.log(5.0))
    range_n = np.clip(trades.get("range_n", pd.Series(np.ones(len(trades)))).to_numpy(), 0, 1)

    # Split for optimization vs reporting
    n = len(trades)
    split = test_split_idx if test_split_idx > 0 else n

    train_mask = np.zeros(n, dtype=bool)
    train_mask[:split] = True
    test_mask = ~train_mask
    has_test = np.any(test_mask)

    trials_data = []

    # --- Stage 1 ---
    stage1 = []

    # Add a trial with "No Risk Cut" (baseline)
    # Simulate effectively unreachable threshold
    grid = list(product([15, 30, 60], [TriggerMode.AND, TriggerMode.OR, TriggerMode.TIMES], [0.8, 1.0, 1.2, 1.4, 1.6], [0.2, 0.3, 0.4]))
    # Add "disabled" config: High theta0 so it never triggers
    grid.append((30, TriggerMode.TIMES, 999.0, 0.3))

    for w, mode, theta0, theta_min in grid:
        lam_rv, lam_rng = 0.5, 0.25
        # Note: 'w' logic missing in original, keeping as is
        D = speed_n * (1 + lam_rv * rv_n) * (1 + lam_rng * np.minimum(1.0, range_n))
        p = TriggerParams(mode=mode, theta0=theta0, theta_mae_min=theta_min)

        # Vectorized trigger calc simulation? No, list comp in original
        trigs = np.array([risk_cut_trigger(float(D[i]), float(mae_n[i]), p)[0] for i in range(len(D))])

        # Apply Risk Cut logic: update PnL
        # Rule: if triggered, exit at -0.6 * SL
        pnl_new = ret.copy()
        pnl_new[trigs] = np.maximum(pnl_new[trigs], -0.6 * sl_pct[trigs])

        # --- Train Selection ---
        pnl_train = pnl_new[train_mask]
        sl_train = sl_pct[train_mask]

        p_sl = float(np.mean(pnl_train <= -sl_train))
        avg_loss = float(np.mean(np.abs(pnl_train[pnl_train < 0]) / np.maximum(sl_train[pnl_train < 0], 1e-6))) if np.any(pnl_train < 0) else 0.0

        # Train Metrics
        winners = ret[train_mask] > 0
        missed_win = 0.0
        if np.any(winners):
            diff = (ret[train_mask][winners] - pnl_train[winners]) / np.maximum(sl_train[winners], 1e-6)
            missed_win = float(np.mean(np.maximum(0.0, diff)))

        winners_to_losers = float(np.mean((ret[train_mask] > 0) & (pnl_train <= 0)))

        j3 = float(np.sum(pnl_train) - 0.2 * p_sl - 0.5 * avg_loss - 0.3 * missed_win)

        # Log Trial
        trial_metrics = {
            "stage": 1,
            "w_minutes": w,
            "mode": mode.value,
            "theta0": theta0,
            "theta_mae_min": theta_min,
            "lambda_rv": lam_rv,
            "lambda_rng": lam_rng,
            "train_j3": j3,
            "train_winners_to_losers": winners_to_losers,
        }

        if winners_to_losers <= 0.10:
             stage1.append((j3, w, mode, theta0, theta_min))
        else:
             trial_metrics["note"] = "filtered_w2l"

        # --- Test Reporting ---
        if has_test:
            test_pnl = pnl_new[test_mask]
            test_trades = trades.iloc[test_mask].copy()

            # Update exit_price
            is_long = test_trades["is_long"].astype(int).to_numpy()
            new_exit = np.where(is_long == 1,
                                test_trades["entry_price"] * (1 + test_pnl),
                                test_trades["entry_price"] * (1 - test_pnl))
            test_trades["exit_price"] = new_exit
            test_trades["exit_reason"] = "risk_cut_opt"

            # Mark triggered trades in exit reason?
            trigs_test = trigs[test_mask]
            if np.any(trigs_test):
                 test_trades.loc[test_trades.index[trigs_test], "exit_reason"] = "risk_cut_triggered"

            m = compute_comprehensive_metrics(test_trades)
            trial_metrics.update(m)

        trials_data.append(trial_metrics)

    if not stage1:
        # Fallback
        default_params = {"risk_cut_mode": "TIMES", "theta0": 1.2, "theta_mae_min": 0.3, "lambda_rv": 0.5, "lambda_rng": 0.25, "w_minutes": 30}
        return default_params, pd.DataFrame(trials_data)

    _, w_best, mode_best, theta0_best, theta_min_best = max(stage1, key=lambda x: x[0])

    # --- Stage 2 ---
    stage2 = []
    for lam_rv, lam_rng in product([0.0, 0.25, 0.5, 0.75], [0.0, 0.15, 0.25]):
        D = speed_n * (1 + lam_rv * rv_n) * (1 + lam_rng * np.minimum(1.0, range_n))
        p = TriggerParams(mode=mode_best, theta0=theta0_best, theta_mae_min=theta_min_best)
        trigs = np.array([risk_cut_trigger(float(D[i]), float(mae_n[i]), p)[0] for i in range(len(D))])

        pnl_new = ret.copy()
        pnl_new[trigs] = np.maximum(pnl_new[trigs], -0.6 * sl_pct[trigs])

        # Train Selection
        pnl_train = pnl_new[train_mask]
        j_score = float(np.sum(pnl_train)) # Simple sum for stage 2? Original code: float(np.sum(pnl))
        stage2.append((j_score, lam_rv, lam_rng))

        trial_metrics = {
            "stage": 2,
            "w_minutes": w_best,
            "mode": mode_best.value,
            "theta0": theta0_best,
            "theta_mae_min": theta_min_best,
            "lambda_rv": lam_rv,
            "lambda_rng": lam_rng,
            "train_j_score": j_score,
        }

        # Test Reporting
        if has_test:
            test_pnl = pnl_new[test_mask]
            test_trades = trades.iloc[test_mask].copy()
            is_long = test_trades["is_long"].astype(int).to_numpy()
            new_exit = np.where(is_long == 1,
                                test_trades["entry_price"] * (1 + test_pnl),
                                test_trades["entry_price"] * (1 - test_pnl))
            test_trades["exit_price"] = new_exit
            test_trades["exit_reason"] = "risk_cut_opt_s2"
            trigs_test = trigs[test_mask]
            if np.any(trigs_test):
                 test_trades.loc[test_trades.index[trigs_test], "exit_reason"] = "risk_cut_triggered"

            m = compute_comprehensive_metrics(test_trades)
            trial_metrics.update(m)

        trials_data.append(trial_metrics)

    _, lam_rv_best, lam_rng_best = max(stage2, key=lambda x: x[0])

    best_params = {
        "risk_cut_mode": mode_best.value,
        "theta0": float(theta0_best),
        "theta_mae_min": float(theta_min_best),
        "lambda_rv": float(lam_rv_best),
        "lambda_rng": float(lam_rng_best),
        "w_minutes": int(w_best),
    }

    return best_params, pd.DataFrame(trials_data)
