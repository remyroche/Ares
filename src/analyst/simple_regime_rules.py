from __future__ import annotations

from typing import Tuple, List

import numpy as np
import pandas as pd


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ADX from raw OHLC.

    Returns a Series aligned with df index.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    move_up = high.diff()
    move_down = low.diff()
    plus_dm = ((move_up > move_down) & (move_up > 0)) * move_up
    minus_dm = ((move_down > move_up) & (move_down > 0)) * move_down

    plus_dm = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
    minus_dm = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (plus_dm / atr)
    minus_di = 100 * (minus_dm / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx.fillna(25)


def compute_ema_adx_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA(21), EMA(55) on close and ADX(14) from OHLC.

    Expects columns: 'open','high','low','close','volume'.
    """
    out = df.copy()
    out["ema_21"] = out["close"].ewm(span=21, adjust=False).mean()
    out["ema_55"] = out["close"].ewm(span=55, adjust=False).mean()
    out["adx"] = compute_adx(out, period=14)
    return out


def classify_regime_series(df: pd.DataFrame) -> Tuple[List[str], List[float]]:
    """Classify regime for each row using EMA/ADX rules.

    Rules:
      - Bull: EMA(21) > EMA(55) AND ADX > 25
      - Bear: EMA(21) < EMA(55) AND ADX > 25
      - Sideways: if neither Bull nor Bear OR ADX < 20

    Returns:
      regimes: list[str] of 'BULL'|'BEAR'|'SIDEWAYS'
      confidences: list[float] in [0,1]
    """
    feats = compute_ema_adx_features(df)

    bull = (feats["ema_21"] > feats["ema_55"]) & (feats["adx"] > 25)
    bear = (feats["ema_21"] < feats["ema_55"]) & (feats["adx"] > 25)

    regimes = np.where(bull, "BULL", np.where(bear, "BEAR", "SIDEWAYS")).tolist()

    ema_sep = (feats["ema_21"] - feats["ema_55"]).abs()
    ema_sep_norm = (ema_sep / feats["close"].rolling(55).mean()).fillna(0.0)

    sideways = ~(bull | bear)

    # Vectorized confidence calculation
    # Sideways confidence
    conf_sideways = np.clip((20 - feats["adx"]) / 20, 0.2, 1.0)

    # Trend confidence
    adx_component = np.clip((feats["adx"] - 20) / 30, 0.0, 1.0)
    sep_component = np.clip(ema_sep_norm * 10, 0.0, 1.0)
    conf_trend = np.clip(0.5 * adx_component + 0.5 * sep_component, 0.2, 1.0)

    confidences = np.where(sideways, conf_sideways, conf_trend).tolist()
    return regimes, confidences


def classify_last(df: pd.DataFrame) -> Tuple[str, float]:
    """Classify the last row regime and confidence using EMA/ADX rules.

    Returns ('BULL'|'BEAR'|'SIDEWAYS', confidence)
    """
    if df is None or df.empty:
        return "SIDEWAYS", 0.5
    regimes, confidences = classify_regime_series(df)
    return regimes[-1], confidences[-1]