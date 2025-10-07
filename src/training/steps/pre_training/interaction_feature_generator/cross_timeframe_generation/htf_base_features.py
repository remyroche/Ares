"""Shared HTF base feature utilities.

This module centralizes the base feature computation mapping, resampling
logic, and helper functions that are required by both HTF generation
phases. The utilities defined here are intentionally stateless so they can
be safely reused across the different generators without duplicating
implementation details.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd


def _price_ema10_pct(data: pd.DataFrame) -> pd.Series:
    """Price vs EMA10 percentage."""
    ema10 = data["close"].ewm(span=10).mean()
    return (data["close"] - ema10) / ema10


def _price_ema20_pct(data: pd.DataFrame) -> pd.Series:
    """Price vs EMA20 percentage."""
    ema20 = data["close"].ewm(span=20).mean()
    return (data["close"] - ema20) / ema20


def _bollz20(data: pd.DataFrame) -> pd.Series:
    """Bollinger z-score."""
    ma20 = data["close"].rolling(20).mean()
    sd20 = data["close"].rolling(20).std()
    return (data["close"] - ma20) / sd20


def _sigma_ew(data: pd.DataFrame) -> pd.Series:
    """Exponentially weighted standard deviation of r1."""
    r1 = np.log(data["close"] / data["close"].shift(1))
    return r1.ewm(halflife=12).std()


def _gk_w(data: pd.DataFrame) -> pd.Series:
    """Garman-Klass volatility estimator."""
    log_hl = np.log(data["high"] / data["low"])
    log_co = np.log(data["close"] / data["open"])
    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    return np.sqrt(gk.rolling(12).mean())


def _rv_bipower_12(data: pd.DataFrame) -> pd.Series:
    """Bipower variation estimate."""
    r1 = np.log(data["close"] / data["close"].shift(1))
    r1_abs = np.abs(r1)
    bipower = r1_abs * r1_abs.shift(1)
    return np.sqrt(bipower.rolling(12).mean())


def _rv_short_3(data: pd.DataFrame) -> pd.Series:
    """Short-term realized volatility."""
    r1 = np.log(data["close"] / data["close"].shift(1))
    return np.sqrt((r1 ** 2).rolling(3).sum())


def _rsi(data: pd.DataFrame, period: int) -> pd.Series:
    """Relative Strength Index calculation."""
    delta = data["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _rsi7(data: pd.DataFrame) -> pd.Series:
    return _rsi(data, 7)


def _rsi14(data: pd.DataFrame) -> pd.Series:
    return _rsi(data, 14)


def _stochk14(data: pd.DataFrame) -> pd.Series:
    """14-period Stochastic %K."""
    low14 = data["low"].rolling(14).min()
    high14 = data["high"].rolling(14).max()
    return 100 * (data["close"] - low14) / (high14 - low14)


def _autocorr_r1_w(data: pd.DataFrame) -> pd.Series:
    """Autocorrelation of r1 with a 1-period lag."""
    r1 = np.log(data["close"] / data["close"].shift(1))
    return r1.rolling(12).apply(lambda x: x.autocorr(lag=1), raw=False)


def _vwap_session_dist(data: pd.DataFrame) -> pd.Series:
    """Session VWAP distance."""
    vwap = (data["high"] + data["low"] + data["close"]) / 3
    vwap_session = vwap.rolling(12).mean()
    return (data["close"] - vwap_session) / vwap_session


def _vwap_roll12_dist(data: pd.DataFrame) -> pd.Series:
    """Rolling VWAP distance."""
    vwap = (data["high"] + data["low"] + data["close"]) / 3
    vwap_roll = vwap.rolling(12).mean()
    return (data["close"] - vwap_roll) / vwap_roll


_BASE_FEATURE_FUNCTIONS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "p/price_ema10_pct": _price_ema10_pct,
    "p/price_ema20_pct": _price_ema20_pct,
    "p/bollz20": _bollz20,
    "p/sigma_ew": _sigma_ew,
    "p/gk_w": _gk_w,
    "p/rv_bipower_12": _rv_bipower_12,
    "p/rv_short_3": _rv_short_3,
    "p/rsi7": _rsi7,
    "p/rsi14": _rsi14,
    "p/stochk14": _stochk14,
    "p/autocorr_r1_w": _autocorr_r1_w,
    "p/vwap_session_dist": _vwap_session_dist,
    "p/vwap_roll12_dist": _vwap_roll12_dist,
}


def get_base_feature_func(base_feature: str) -> Callable[[pd.DataFrame], pd.Series]:
    """Return the computation function for a base feature."""
    try:
        return _BASE_FEATURE_FUNCTIONS[base_feature]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown base feature: {base_feature}") from exc


def resample_to_htf(
    base_series: pd.Series,
    lookback_minutes: int,
    family: str,
) -> pd.Series:
    """Resample a base feature series to the requested HTF frequency."""
    rule = f"{lookback_minutes}min"

    if family in {"trend_level_vol", "anchors"}:
        return base_series.resample(rule).last()
    if family == "oscillators":
        return base_series.resample(rule).mean()
    return base_series.resample(rule).last()


__all__ = [
    "get_base_feature_func",
    "resample_to_htf",
]
