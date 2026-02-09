from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd


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


def optimise_loss_limiter(trades: pd.DataFrame, sl_pct: np.ndarray) -> dict:
    ret = np.where(trades["is_long"].astype(int).to_numpy() == 1,
                   (trades["exit_price"] - trades["entry_price"]) / trades["entry_price"],
                   (trades["entry_price"] - trades["exit_price"]) / trades["entry_price"])

    mae_n = np.clip(np.abs(trades.get("mae_pct", pd.Series(np.abs(ret))).to_numpy()) / np.maximum(sl_pct, 1e-6), 0, 3)
    speed_n = np.clip(np.abs(trades.get("ret_w", pd.Series(ret)).to_numpy()) / np.maximum(sl_pct, 1e-6), 0, 4)
    rv_n = np.clip(np.log(np.maximum(trades.get("rv", pd.Series(np.ones(len(trades)))).to_numpy(), 1.0)), 0, np.log(5.0))
    range_n = np.clip(trades.get("range_n", pd.Series(np.ones(len(trades)))).to_numpy(), 0, 1)

    stage1 = []
    for w, mode, theta0, theta_min in product([15, 30, 60], [TriggerMode.AND, TriggerMode.OR, TriggerMode.TIMES], [0.8, 1.0, 1.2, 1.4, 1.6], [0.2, 0.3, 0.4]):
        lam_rv, lam_rng = 0.5, 0.25
        D = speed_n * (1 + lam_rv * rv_n) * (1 + lam_rng * np.minimum(1.0, range_n))
        p = TriggerParams(mode=mode, theta0=theta0, theta_mae_min=theta_min)
        trigs = np.array([risk_cut_trigger(float(D[i]), float(mae_n[i]), p)[0] for i in range(len(D))])
        pnl = ret.copy()
        pnl[trigs] = np.maximum(pnl[trigs], -0.6 * sl_pct[trigs])
        p_sl = float(np.mean(pnl <= -sl_pct))
        avg_loss = float(np.mean(np.abs(pnl[pnl < 0]) / np.maximum(sl_pct[pnl < 0], 1e-6))) if np.any(pnl < 0) else 0.0
        winners = ret > 0
        missed_win = float(np.mean(np.maximum(0.0, (ret[winners] - pnl[winners]) / np.maximum(sl_pct[winners], 1e-6)))) if np.any(winners) else 0.0
        winners_to_losers = float(np.mean((ret > 0) & (pnl <= 0)))
        if winners_to_losers > 0.10:
            continue
        j3 = float(np.sum(pnl) - 0.2 * p_sl - 0.5 * avg_loss - 0.3 * missed_win)
        stage1.append((j3, w, mode.value, theta0, theta_min))

    if not stage1:
        return {"risk_cut_mode": "TIMES", "theta0": 1.2, "theta_mae_min": 0.3, "lambda_rv": 0.5, "lambda_rng": 0.25, "w_minutes": 30}

    _, w_best, mode_best, theta0_best, theta_min_best = max(stage1, key=lambda x: x[0])
    stage2 = []
    for lam_rv, lam_rng in product([0.0, 0.25, 0.5, 0.75], [0.0, 0.15, 0.25]):
        D = speed_n * (1 + lam_rv * rv_n) * (1 + lam_rng * np.minimum(1.0, range_n))
        p = TriggerParams(mode=TriggerMode(mode_best), theta0=theta0_best, theta_mae_min=theta_min_best)
        trigs = np.array([risk_cut_trigger(float(D[i]), float(mae_n[i]), p)[0] for i in range(len(D))])
        pnl = ret.copy()
        pnl[trigs] = np.maximum(pnl[trigs], -0.6 * sl_pct[trigs])
        stage2.append((float(np.sum(pnl)), lam_rv, lam_rng))
    _, lam_rv_best, lam_rng_best = max(stage2, key=lambda x: x[0])

    return {
        "risk_cut_mode": mode_best,
        "theta0": float(theta0_best),
        "theta_mae_min": float(theta_min_best),
        "lambda_rv": float(lam_rv_best),
        "lambda_rng": float(lam_rng_best),
        "w_minutes": int(w_best),
    }
