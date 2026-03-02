from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from numba import njit

from extreme_price_movements.pnl import CostModel
from extreme_price_movements.tpsl_optimiser.metrics_utils import compute_comprehensive_metrics

@njit
def fast_resolve_hits(
    n_trades: int,
    trade_starts: np.ndarray,
    trade_ends: np.ndarray,
    is_long_arr: np.ndarray,
    sl_pct_arr: np.ndarray,
    tp_pct_arr: np.ndarray,
    base_ret: np.ndarray,
    all_highs: np.ndarray,
    all_lows: np.ndarray,
    all_closes: np.ndarray,
    entry_prices: np.ndarray
):
    resolved = base_ret.copy()
    for i in range(n_trades):
        start_idx = trade_starts[i]
        end_idx = trade_ends[i]
        if start_idx < 0 or end_idx < 0 or start_idx >= end_idx:
            resolved[i] = max(min(base_ret[i], tp_pct_arr[i]), -sl_pct_arr[i])
            continue

        entry_p = entry_prices[i]
        if entry_p <= 0:
            resolved[i] = max(min(base_ret[i], tp_pct_arr[i]), -sl_pct_arr[i])
            continue

        sl = sl_pct_arr[i]
        tp = tp_pct_arr[i]

        if is_long_arr[i] == 1:
            sl_price = entry_p * (1.0 - sl)
            tp_price = entry_p * (1.0 + tp)
        else:
            sl_price = entry_p * (1.0 + sl)
            tp_price = entry_p * (1.0 - tp)

        hit = False
        for j in range(start_idx, end_idx):
            hh = all_highs[j]
            ll = all_lows[j]
            cc = all_closes[j]

            bar_hit_tp = False
            bar_hit_sl = False

            if is_long_arr[i] == 1:
                if ll <= sl_price: bar_hit_sl = True
                if hh >= tp_price: bar_hit_tp = True
            else:
                if hh >= sl_price: bar_hit_sl = True
                if ll <= tp_price: bar_hit_tp = True

            if bar_hit_tp and not bar_hit_sl:
                resolved[i] = tp
                hit = True
                break
            elif bar_hit_sl and not bar_hit_tp:
                resolved[i] = -sl
                hit = True
                break
            elif bar_hit_sl and bar_hit_tp:
                d_tp = abs(cc - tp_price)
                d_sl = abs(cc - sl_price)
                if d_tp < d_sl:
                    resolved[i] = tp
                else:
                    resolved[i] = -sl
                hit = True
                break

        if not hit:
            resolved[i] = max(min(base_ret[i], tp), -sl)

    return resolved


@dataclass(frozen=True)
class TpSlCalibrationConfig:
    warmup_bars: int = 96
    lo: float = 0.6
    hi: float = 2.5
    tp_grid: tuple[float, ...] = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0)
    # sl_ratio == TP/SL reward-to-risk ratio. Keep >= 1.5 by default.
    sl_ratio_grid: tuple[float, ...] = (2.5, 2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.66, 0.5, 0.33, 0.25)
    min_rr_ratio: float = 1.5


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


def calibrate_tp_sl(trades: pd.DataFrame, atr_scale: pd.Series, df_15m_dict: Dict[str, pd.DataFrame] | None = None, cfg: TpSlCalibrationConfig | None = None, test_split_idx: int = 0, fee_pct: float = 0.005, cost: CostModel | None = None, init_params: dict | None = None) -> Tuple[Dict[str, float], pd.DataFrame]:
    cfg = cfg or TpSlCalibrationConfig()
    df = trades.copy()
    df = df.assign(atr_scale=atr_scale.reindex(df.index).fillna(1.0).to_numpy())

    results = []
    trials_data = []

    # Calculate base returns (Gross)
    is_long_arr = df["is_long"].astype(int).to_numpy()
    entry_prices = df["entry_price"].to_numpy()
    exit_prices = df["exit_price"].to_numpy()
    base_ret = np.where(is_long_arr == 1,
                        (exit_prices - entry_prices) / entry_prices,
                        (entry_prices - exit_prices) / entry_prices)

    # Split for optimization vs reporting
    n = len(df)
    split = test_split_idx if test_split_idx > 0 else n  # Default to full sample if split=0

    # Train set (for selection)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:split] = True

    # Test set (for reporting)
    test_mask = ~train_mask
    has_test = np.any(test_mask)

    # Pre-process 15m paths for all trades if available
    trade_starts = np.full(n, -1, dtype=np.int64)
    trade_ends = np.full(n, -1, dtype=np.int64)

    # We will build a single flattened array of 15m bars to pass to Numba
    # Since Numba does not support dictionaries easily
    all_highs_list = []
    all_lows_list = []
    all_closes_list = []

    current_idx = 0

    if df_15m_dict is not None and "asset" in df.columns and "timestamp" in df.columns:
        # Sort by asset and timestamp to align easily, but we iterate over original indices
        assets = df["asset"].values
        timestamps = pd.to_datetime(df["timestamp"], utc=True)
        max_hold = pd.Timedelta(hours=getattr(df, "max_hold_hours", 24))

        # We need a quick way to slice the 15m data per trade
        for i in range(n):
            asset = assets[i]
            if asset in df_15m_dict:
                df_15 = df_15m_dict[asset]
                ts = timestamps.iloc[i]
                ts_end = ts + max_hold

                # Slicing the dataframe
                mask = (df_15.index >= ts) & (df_15.index < ts_end)
                bars = df_15.loc[mask]

                if not bars.empty:
                    bar_len = len(bars)
                    trade_starts[i] = current_idx
                    trade_ends[i] = current_idx + bar_len

                    all_highs_list.append(bars["high"].to_numpy(dtype=np.float64))
                    all_lows_list.append(bars["low"].to_numpy(dtype=np.float64))
                    all_closes_list.append(bars["close"].to_numpy(dtype=np.float64))

                    current_idx += bar_len

    if all_highs_list:
        all_highs = np.concatenate(all_highs_list)
        all_lows = np.concatenate(all_lows_list)
        all_closes = np.concatenate(all_closes_list)
    else:
        all_highs = np.empty(0, dtype=np.float64)
        all_lows = np.empty(0, dtype=np.float64)
        all_closes = np.empty(0, dtype=np.float64)


    combos = list(product(cfg.tp_grid, cfg.sl_ratio_grid))
    if init_params:
        tp0 = float(init_params.get("tp_mult", np.nan))
        sl0 = float(init_params.get("sl_mult", np.nan))
        if np.isfinite(tp0) and np.isfinite(sl0) and sl0 > 1e-9:
            combos.append((tp0, tp0 / sl0))

    for tp_mult, sl_ratio in combos:
        # Hard floor: do not allow reward-to-risk below configured minimum.
        if float(sl_ratio) < float(cfg.min_rr_ratio):
            continue
        sl_mult = tp_mult / sl_ratio

        tp_pct_arr = tp_mult * df["atr_scale"].to_numpy()
        sl_pct_arr = sl_mult * df["atr_scale"].to_numpy()

        # Apply TP/SL logic (Gross returns clipped)
        # Note: logic assumes exit at TP or SL if touched.
        if df_15m_dict is not None and "asset" in df.columns and "timestamp" in df.columns and len(all_highs) > 0:
            clipped = fast_resolve_hits(
                n,
                trade_starts,
                trade_ends,
                is_long_arr,
                sl_pct_arr,
                tp_pct_arr,
                base_ret,
                all_highs,
                all_lows,
                all_closes,
                entry_prices
            )
        else:
            clipped = np.clip(base_ret, -sl_pct_arr, tp_pct_arr)

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
            "rr_ratio": float(sl_ratio),
            "train_pnl": pnl,
            "train_sortino": sortino,
        }

        if has_test:
            test_ret = clipped[test_mask]
            test_trades = df.iloc[test_mask].copy()

            # Update exit_price to reflect TP/SL exit for accurate metrics calculation
            is_long_test = test_trades["is_long"].astype(int).to_numpy()
            new_exit = np.where(is_long_test == 1,
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
        "rr_ratio": float(best[0] / max(best[1], 1e-12)),
        "atr_scale_lo": cfg.lo,
        "atr_scale_hi": cfg.hi,
    }

    return best_params, trials_df
