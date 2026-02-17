from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from extreme_price_movements.pnl import CostModel
from extreme_price_movements.tpsl_optimiser.metrics_utils import compute_comprehensive_metrics


@dataclass(frozen=True)
class TpSlCalibrationConfig:
    warmup_bars: int = 96
    lo: float = 0.6
    hi: float = 2.5
    tp_grid: tuple[float, ...] = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0)
    sl_ratio_grid: tuple[float, ...] = (1.25, 1.0, 0.75, 0.66, 0.5, 0.33, 0.25)


def _ewm_halflife(s: pd.Series, hl_bars: int) -> pd.Series:
    return s.ewm(halflife=max(1, hl_bars), adjust=False, min_periods=1).mean()


def compute_atr_scale(atr_12h_pct: pd.Series, cfg: TpSlCalibrationConfig | None = None) -> pd.Series:
    cfg = cfg or TpSlCalibrationConfig()
    a = atr_12h_pct.astype(float).clip(lower=1e-8)
    local = a / _ewm_halflife(a, hl_bars=96)
    a_slow = _ewm_halflife(a, hl_bars=5 * 24 * 4)
    atr_ref = float(np.nanmedian(a_slow.iloc[: max(cfg.warmup_bars, 1)]))
    atr_ref = atr_ref if np.isfinite(atr_ref) and atr_ref > 0 else float(a_slow.median())
    global_scale = np.sqrt((a_slow / max(atr_ref, 1e-8)).clip(lower=1e-8)).clip(0.7, 1.5)
    m = (local * global_scale).clip(cfg.lo, cfg.hi)
    return m.ffill().fillna(1.0)


def _z(v: np.ndarray) -> np.ndarray:
    m = np.nanmean(v)
    s = np.nanstd(v)
    return np.zeros_like(v) if s < 1e-12 else (v - m) / s


def _equity_metrics(rets: np.ndarray) -> tuple[float, float, float]:
    pnl = float(np.nansum(rets))
    neg = rets[rets < 0]
    sortino = float(np.nanmean(rets) / (np.nanstd(neg) + 1e-12)) if neg.size else 0.0
    eq = np.nancumsum(np.nan_to_num(rets, nan=0.0))
    peak = np.maximum.accumulate(eq) if eq.size else np.array([0.0])
    dd = peak - eq if eq.size else np.array([0.0])
    max_dd = float(np.nanmax(dd)) if dd.size else 0.0
    return pnl, sortino, max_dd


def calibrate_tp_sl(trades: pd.DataFrame, atr_scale: pd.Series, cfg: TpSlCalibrationConfig | None = None, test_split_idx: int = 0, fee_pct: float = 0.005, cost: CostModel | None = None, init_params: dict | None = None) -> Tuple[Dict[str, float], pd.DataFrame]:
    cfg = cfg or TpSlCalibrationConfig()
    df = trades.copy()
    df = df.assign(atr_scale=atr_scale.reindex(df.index).fillna(1.0).to_numpy())

    results = []
    trials_data = []

    # Calculate base returns (Gross)
    base_ret = np.where(df["is_long"].astype(int).to_numpy() == 1,
                        (df["exit_price"] - df["entry_price"]) / df["entry_price"],
                        (df["entry_price"] - df["exit_price"]) / df["entry_price"])

    # Split for optimization vs reporting
    n = len(df)
    split = test_split_idx if test_split_idx > 0 else n  # Default to full sample if split=0

    # Train set (for selection)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:split] = True

    # Test set (for reporting)
    test_mask = ~train_mask
    has_test = np.any(test_mask)


    combos = list(product(cfg.tp_grid, cfg.sl_ratio_grid))
    if init_params:
        tp0 = float(init_params.get("tp_mult", np.nan))
        sl0 = float(init_params.get("sl_mult", np.nan))
        if np.isfinite(tp0) and np.isfinite(sl0) and sl0 > 1e-9:
            combos.append((tp0, tp0 / sl0))

    for tp_mult, sl_ratio in combos:
        sl_mult = tp_mult / sl_ratio

        tp_pct = tp_mult * df["atr_scale"].to_numpy()
        sl_pct = sl_mult * df["atr_scale"].to_numpy()

        # Apply TP/SL logic (Gross returns clipped)
        # Note: logic assumes exit at TP or SL if touched.
        clipped = np.clip(base_ret, -sl_pct, tp_pct)

        # --- Train Selection ---
        train_ret = clipped[train_mask]
        if len(train_ret) == 0:
            pnl, sortino, max_dd = 0.0, 0.0, 0.0
        else:
            pnl, sortino, max_dd = _equity_metrics(train_ret)

        results.append((tp_mult, sl_mult, pnl, sortino, max_dd))

        # --- Test Reporting ---
        trial_metrics = {
            "tp_mult": tp_mult,
            "sl_mult": sl_mult,
            "sl_ratio": sl_ratio,
            "train_pnl": pnl,
            "train_sortino": sortino,
        }

        if has_test:
            test_ret = clipped[test_mask]
            test_trades = df.iloc[test_mask].copy()

            # Update exit_price to reflect TP/SL exit for accurate metrics calculation
            is_long = test_trades["is_long"].astype(int).to_numpy()
            new_exit = np.where(is_long == 1,
                                test_trades["entry_price"] * (1 + test_ret),
                                test_trades["entry_price"] * (1 - test_ret))
            test_trades["exit_price"] = new_exit
            test_trades["exit_reason"] = "tpsl_calib" # placeholder

            # Compute full suite of metrics on test set
            m = compute_comprehensive_metrics(test_trades, fee_pct=fee_pct, cost=cost)
            # Prefix keys to avoid collision if needed, but here we just dump them
            trial_metrics.update(m)

        trials_data.append(trial_metrics)

    trials_df = pd.DataFrame(trials_data)

    if not results:
        best_params = {"tp_mult": 3.0, "sl_mult": 1.0, "atr_scale_lo": cfg.lo, "atr_scale_hi": cfg.hi}
        return best_params, trials_df

    # Select best based on Train performance
    arr = np.asarray(results, dtype=float)
    # arr cols: 0=tp, 1=sl, 2=pnl, 3=sortino, 4=max_dd
    score = 0.6 * _z(arr[:, 2]) + 0.3 * _z(arr[:, 3]) - 0.1 * _z(arr[:, 4])
    best_idx = int(np.nanargmax(score))
    best = arr[best_idx]

    best_params = {
        "tp_mult": float(best[0]),
        "sl_mult": float(best[1]),
        "atr_scale_lo": cfg.lo,
        "atr_scale_hi": cfg.hi,
    }

    return best_params, trials_df
