from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TpSlCalibrationConfig:
    warmup_bars: int = 96
    lo: float = 0.6
    hi: float = 2.5
    tp_grid: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0)
    sl_grid: tuple[float, ...] = (0.75, 1.0, 1.25)


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
    return m.fillna(method="ffill").fillna(1.0)


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


def calibrate_tp_sl(trades: pd.DataFrame, atr_scale: pd.Series, cfg: TpSlCalibrationConfig | None = None) -> Dict[str, float]:
    cfg = cfg or TpSlCalibrationConfig()
    df = trades.copy()
    df = df.assign(atr_scale=atr_scale.reindex(df.index).fillna(1.0).to_numpy())

    results = []
    base_ret = np.where(df["is_long"].astype(int).to_numpy() == 1,
                        (df["exit_price"] - df["entry_price"]) / df["entry_price"],
                        (df["entry_price"] - df["exit_price"]) / df["entry_price"])

    for tp_mult, sl_mult in product(cfg.tp_grid, cfg.sl_grid):
        tp_pct = tp_mult * df["atr_scale"].to_numpy()
        sl_pct = sl_mult * df["atr_scale"].to_numpy()
        rr = tp_pct / (sl_pct + 0.5)
        mask = rr >= 1.5
        if mask.sum() < 10:
            continue
        clipped = np.clip(base_ret[mask], -sl_pct[mask], tp_pct[mask])
        pnl, sortino, max_dd = _equity_metrics(clipped)
        results.append((tp_mult, sl_mult, pnl, sortino, max_dd))

    if not results:
        return {"tp_mult": 3.0, "sl_mult": 1.0, "atr_scale_lo": cfg.lo, "atr_scale_hi": cfg.hi}

    arr = np.asarray(results, dtype=float)
    score = 0.6 * _z(arr[:, 2]) + 0.3 * _z(arr[:, 3]) - 0.1 * _z(arr[:, 4])
    best = arr[int(np.nanargmax(score))]
    return {
        "tp_mult": float(best[0]),
        "sl_mult": float(best[1]),
        "atr_scale_lo": cfg.lo,
        "atr_scale_hi": cfg.hi,
    }
