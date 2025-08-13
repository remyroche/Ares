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


def compute_ema_adx_features(
    df: pd.DataFrame,
    ema_fast: int = 21,
    ema_slow: int = 55,
    adx_period: int = 14,
) -> pd.DataFrame:
    """Compute EMA(fast), EMA(slow) on close and ADX(period) from OHLC.

    Expects columns: 'open','high','low','close','volume'.
    """
    out = df.copy()
    out[f"ema_{ema_fast}"] = out["close"].ewm(span=ema_fast, adjust=False).mean()
    out[f"ema_{ema_slow}"] = out["close"].ewm(span=ema_slow, adjust=False).mean()
    out["adx"] = compute_adx(out, period=adx_period)
    return out


def classify_regime_series(
    df: pd.DataFrame,
    *,
    ema_fast: int = 21,
    ema_slow: int = 55,
    adx_period: int = 14,
    adx_trend_threshold: float = 25.0,
    adx_sideways_threshold: float = 20.0,
    ema_sep_min_ratio: float = 0.0,
) -> Tuple[List[str], List[float]]:
    """Classify regime for each row using EMA/ADX rules.

    Rules (parameterized):
      - Trend condition: ADX >= adx_trend_threshold AND normalized EMA separation >= ema_sep_min_ratio
      - Bull: EMA(fast) > EMA(slow) AND Trend condition
      - Bear: EMA(fast) < EMA(slow) AND Trend condition
      - Sideways: otherwise OR ADX <= adx_sideways_threshold

    Returns:
      regimes: list[str] of 'BULL'|'BEAR'|'SIDEWAYS'
      confidences: list[float] in [0,1]
    """
    feats = compute_ema_adx_features(
        df,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        adx_period=adx_period,
    )

    # Normalized EMA separation relative to a smoothed price level
    ema_sep = (feats[f"ema_{ema_fast}"] - feats[f"ema_{ema_slow}"]).abs()
    ema_sep_norm = (ema_sep / feats["close"].rolling(max(ema_slow, 2)).mean()).fillna(0.0)

    # Trend condition with tunable thresholds
    meets_adx = feats["adx"] >= adx_trend_threshold
    meets_sep = ema_sep_norm >= max(ema_sep_min_ratio, 0.0)
    trend_condition = meets_adx & meets_sep

    bull = (feats[f"ema_{ema_fast}"] > feats[f"ema_{ema_slow}"]) & trend_condition
    bear = (feats[f"ema_{ema_fast}"] < feats[f"ema_{ema_slow}"]) & trend_condition

    # Explicit sideways if ADX is below the sideways threshold
    forced_sideways = feats["adx"] <= adx_sideways_threshold

    regimes = np.where(
        forced_sideways,
        "SIDEWAYS",
        np.where(bull, "BULL", np.where(bear, "BEAR", "SIDEWAYS")),
    ).tolist()

    sideways_mask = np.array(regimes, dtype=object) == "SIDEWAYS"

    # Confidence calculation (parameter-aware)
    # Sideways confidence increases as ADX drops below the sideways threshold
    denom_sw = max(adx_sideways_threshold, 1e-6)
    conf_sideways = np.clip((adx_sideways_threshold - feats["adx"]) / denom_sw, 0.2, 1.0)

    # Trend confidence increases with ADX above sideways threshold and EMA separation
    denom_tr = max(adx_trend_threshold - adx_sideways_threshold, 1e-6)
    adx_component = np.clip((feats["adx"] - adx_sideways_threshold) / denom_tr, 0.0, 1.0)
    sep_component = np.clip(ema_sep_norm * 10.0, 0.0, 1.0)
    conf_trend = np.clip(0.5 * adx_component + 0.5 * sep_component, 0.2, 1.0)

    confidences = np.where(sideways_mask, conf_sideways, conf_trend).tolist()
    return regimes, confidences


def classify_last(
    df: pd.DataFrame,
    *,
    ema_fast: int = 21,
    ema_slow: int = 55,
    adx_period: int = 14,
    adx_trend_threshold: float = 25.0,
    adx_sideways_threshold: float = 20.0,
    ema_sep_min_ratio: float = 0.0,
) -> Tuple[str, float]:
    """Classify the last row regime and confidence using EMA/ADX rules.

    Returns ('BULL'|'BEAR'|'SIDEWAYS', confidence)
    """
    if df is None or df.empty:
        return "SIDEWAYS", 0.5
    regimes, confidences = classify_regime_series(
        df,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        adx_period=adx_period,
        adx_trend_threshold=adx_trend_threshold,
        adx_sideways_threshold=adx_sideways_threshold,
        ema_sep_min_ratio=ema_sep_min_ratio,
    )
    return regimes[-1], confidences[-1]