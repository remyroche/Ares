from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Tuple
import os

import numpy as np
import pandas as pd

from extreme_price_movements.pnl import CostModel
from extreme_price_movements.tpsl_optimiser.metrics_utils import compute_comprehensive_metrics
import ccxt

def get_15m_data_for_trades(trades: pd.DataFrame, hf_data_dir: str = "extreme_price_movements/15m_ohlcv") -> dict:
    """
    Loads 15m OHLCV data for all unique assets found in the trades dataframe.
    Looks in hf_data_dir directory.
    If not found, calls get_15m_ohlcv to download it for the time range.
    Returns a dict mapping asset -> pd.DataFrame (indexed by datetime).
    """
    from extreme_price_movements.hf_data_loader import get_15m_ohlcv
    data_15m = {}

    if "asset" not in trades.columns or "timestamp" not in trades.columns:
        return data_15m

    unique_assets = trades["asset"].unique()

    # We will instantiate binance just in case we need to download.
    exchange = ccxt.binance()

    for asset in unique_assets:
        path_pq = os.path.join(hf_data_dir, f"{asset}.parquet")
        path_csv = os.path.join(hf_data_dir, f"{asset}.csv")

        df_15 = None
        if os.path.exists(path_pq):
            df_15 = pd.read_parquet(path_pq)
        elif os.path.exists(path_csv):
            df_15 = pd.read_csv(path_csv)
            if 'timestamp' in df_15.columns:
                df_15['timestamp'] = pd.to_datetime(df_15['timestamp'], utc=True)
                df_15.set_index('timestamp', inplace=True)
            elif 'ts' in df_15.columns:
                df_15['ts'] = pd.to_datetime(df_15['ts'], utc=True)
                df_15.set_index('ts', inplace=True)

        if df_15 is None or df_15.empty:
            # Not found on disk, so we download it
            asset_trades = trades[trades["asset"] == asset]
            min_ts = pd.to_datetime(asset_trades["timestamp"].min(), utc=True)
            max_ts = pd.to_datetime(asset_trades["timestamp"].max(), utc=True)
            if pd.isnull(min_ts) or pd.isnull(max_ts):
                continue

            # max_hold_hours based on the gap between min and max ts
            gap_hours = max(1, int((max_ts - min_ts).total_seconds() / 3600)) + 12

            try:
                # Need to map asset format if needed, typically 'BTC/USDT' vs 'BTCUSDT'
                # Just assume standard ccxt symbol format if we don't know it, or maybe append /USDT
                # In extreme_price_movements it's typically just 'BTC', 'ETH' or 'BTCUSDT'
                symbol = asset if '/' in asset else f"{asset.replace('USDT', '')}/USDT"

                df_15 = get_15m_ohlcv(exchange, symbol, min_ts, max_hold_hours=gap_hours)
            except Exception as e:
                print(f"Failed to download 15m data for {asset}: {e}")

        if df_15 is not None and not df_15.empty:
            if not pd.api.types.is_datetime64_any_dtype(df_15.index):
                df_15.index = pd.to_datetime(df_15.index, utc=True)
            df_15 = df_15.sort_index()
            data_15m[asset] = df_15

    return data_15m

def precompute_15m_bars_for_trades(trades: pd.DataFrame, data_15m: dict) -> list:
    """
    Precomputes the relevant 15m bars for each trade so we don't index/mask
    inside the nested optimization loop.
    Returns a list of length len(trades). Each element is either None
    or a 2D numpy array: [[high, low, close], ...]
    """
    bars_list = []

    if "timestamp" not in trades.columns or "asset" not in trades.columns:
        return [None] * len(trades)

    for i in range(len(trades)):
        asset = trades["asset"].values[i]
        if asset not in data_15m:
            bars_list.append(None)
            continue

        df_15 = data_15m[asset]
        ts = pd.to_datetime(trades["timestamp"].values[i], utc=True)
        end_ts = ts + pd.Timedelta(hours=1)

        mask = (df_15.index >= ts) & (df_15.index < end_ts)
        bars = df_15.loc[mask]

        if bars.empty:
            bars_list.append(None)
        else:
            bars_arr = bars[['high', 'low', 'close']].to_numpy(dtype=float)
            bars_list.append(bars_arr)

    return bars_list

def resolve_double_hits_fast(
    trades: pd.DataFrame,
    base_ret: np.ndarray,
    tp_pct: np.ndarray,
    sl_pct: np.ndarray,
    bars_list: list
) -> np.ndarray:
    clipped = np.clip(base_ret, -sl_pct, tp_pct)

    if not bars_list:
        return clipped

    resolved = clipped.copy()

    is_long_arr = trades["is_long"].values
    entry_p_arr = trades["entry_price"].values

    for i in range(len(trades)):
        bars = bars_list[i]
        if bars is None:
            continue

        entry_p = float(entry_p_arr[i])
        if entry_p <= 0:
            continue

        tp = float(tp_pct[i])
        sl = float(sl_pct[i])
        is_long = int(is_long_arr[i])

        if is_long == 1:
            sl_price = entry_p * (1.0 - sl)
            tp_price = entry_p * (1.0 + tp)
        else:
            sl_price = entry_p * (1.0 + sl)
            tp_price = entry_p * (1.0 - tp)

        for j in range(bars.shape[0]):
            hh = bars[j, 0]
            ll = bars[j, 1]
            cc = bars[j, 2]

            bar_hit_tp = False
            bar_hit_sl = False

            if is_long == 1:
                if ll <= sl_price: bar_hit_sl = True
                if hh >= tp_price: bar_hit_tp = True
            else:
                if hh >= sl_price: bar_hit_sl = True
                if ll <= tp_price: bar_hit_tp = True

            if bar_hit_tp and not bar_hit_sl:
                resolved[i] = tp
                break
            elif bar_hit_sl and not bar_hit_tp:
                resolved[i] = -sl
                break
            elif bar_hit_sl and bar_hit_tp:
                d_tp = abs(cc - tp_price)
                d_sl = abs(cc - sl_price)
                if d_tp < d_sl:
                    resolved[i] = tp
                else:
                    resolved[i] = -sl
                break

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

    data_15m = get_15m_data_for_trades(df)
    bars_list = precompute_15m_bars_for_trades(df, data_15m)

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
        # Hard floor: do not allow reward-to-risk below configured minimum.
        if float(sl_ratio) < float(cfg.min_rr_ratio):
            continue
        sl_mult = tp_mult / sl_ratio

        tp_pct = tp_mult * df["atr_scale"].to_numpy()
        sl_pct = sl_mult * df["atr_scale"].to_numpy()

        # Apply TP/SL logic (Gross returns clipped)
        # Note: logic assumes exit at TP or SL if touched.
        clipped = resolve_double_hits_fast(df, base_ret, tp_pct, sl_pct, bars_list)

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
        "rr_ratio": float(best[0] / max(best[1], 1e-12)),
        "atr_scale_lo": cfg.lo,
        "atr_scale_hi": cfg.hi,
    }

    return best_params, trials_df
