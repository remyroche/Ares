"""Risk regime feature generation.

This module defines risk-related features used by the risk regime step and
exposes them via a small helper function so that the feature definitions live
inside the feature generation feature bank.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.features_common.transforms.scaling_normalization import (
    winsorized_zscore_normalize,
)


def generate_risk_regime_features(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Generate normalized risk regime features.

    This mirrors the logic of ``MLRiskRegimeStep._generate_risk_features`` so
    that the computation of the risk features is centralized in the
    feature bank. The returned DataFrame contains the five core risk
    features already normalized with a rolling winsorized z-score.

    OPTIMIZED: Uses EWM instead of rolling for volatility features.
    EWM is O(1) per update vs O(window) for rolling windows.

    Args:
        df: Input market data with at least ``high``, ``low`` and ``close``
            columns on the risk timeframe (typically 1h).
        config: Step configuration; the same keys used in the risk regime step
            are honored (``risk_parkinson_window``, ``risk_hurst_window``,
            ``risk_kurtosis_window``, ``risk_skewness_window``,
            ``risk_vol_of_vol_window``, ``risk_normalization_window``).

    Returns:
        DataFrame with columns:
        ``parkinson_volatility``, ``hurst_exponent``, ``rolling_kurtosis``,
        ``rolling_skewness``, ``volatility_of_volatility``.
    """
    if not {"high", "low", "close"}.issubset(df.columns):
        raise ValueError("Market data must contain 'high', 'low', and 'close' columns")

    # Use float32 for memory efficiency
    use_ewm = config.get("risk_use_ewm", True)  # OPTIMIZED: Use EWM by default
    
    high = df["high"].astype(np.float32)
    low = df["low"].astype(np.float32)
    close = df["close"].astype(np.float32)

    # 1. Parkinson Volatility (window: 48 bars by default)
    # OPTIMIZED: Use EWM instead of rolling for faster computation
    parkinson_window = int(config.get("risk_parkinson_window", 48))
    low_safe = low.replace(0, np.nan)
    log_hl = np.log(high / low_safe)
    
    if use_ewm:
        # EWM std is O(1) per point vs O(window) for rolling
        parkinson_vol = log_hl.ewm(span=parkinson_window, adjust=False).std() * np.sqrt(1.0 / (4.0 * np.log(2.0)))
    else:
        parkinson_vol = log_hl.rolling(
            window=parkinson_window,
            min_periods=parkinson_window,
        ).std() * np.sqrt(1.0 / (4.0 * np.log(2.0)))

    # 2. Hurst Exponent (window: 48 bars by default)
    # Note: Hurst is inherently rolling-based, keep as is
    hurst_window = int(config.get("risk_hurst_window", 48))
    hurst_series = _calculate_hurst_exponent(close, hurst_window)

    # 3. Rolling Kurtosis (window: 36 bars by default)
    # Note: Kurtosis requires rolling for proper calculation
    kurtosis_window = int(config.get("risk_kurtosis_window", 36))
    log_returns = np.log(close / close.shift(1))
    rolling_kurtosis = log_returns.rolling(
        window=kurtosis_window,
        min_periods=kurtosis_window,
    ).kurt()

    # 4. Rolling Skewness (window: 36 bars by default)
    # Note: Skewness requires rolling for proper calculation
    skewness_window = int(config.get("risk_skewness_window", 36))
    rolling_skewness = log_returns.rolling(
        window=skewness_window,
        min_periods=skewness_window,
    ).skew()

    # 5. Volatility of Volatility (window: 30 bars by default)
    # OPTIMIZED: Use EWM for both volatility and vol-of-vol
    vol_of_vol_window = int(config.get("risk_vol_of_vol_window", 30))
    
    if use_ewm:
        # Use EWM for volatility calculation - O(1) per point
        volatility = log_returns.ewm(span=20, adjust=False).std()
        vol_of_vol = volatility.ewm(span=vol_of_vol_window, adjust=False).std()
    else:
        volatility = log_returns.rolling(window=20, min_periods=20).std()
        vol_of_vol = volatility.rolling(
            window=vol_of_vol_window,
            min_periods=vol_of_vol_window,
        ).std()

    feature_frame = pd.DataFrame(
        {
            "parkinson_volatility": parkinson_vol,
            "hurst_exponent": hurst_series,
            "rolling_kurtosis": rolling_kurtosis,
            "rolling_skewness": rolling_skewness,
            "volatility_of_volatility": vol_of_vol,
        },
        index=df.index,
    )

    window_size = int(config.get("risk_normalization_window", 500))
    scaled = winsorized_zscore_normalize(
        feature_frame,
        window=window_size,
    )

    # Ensure we always return a DataFrame with the expected columns and index.
    return scaled[[
        "parkinson_volatility",
        "hurst_exponent",
        "rolling_kurtosis",
        "rolling_skewness",
        "volatility_of_volatility",
    ]].reindex(df.index)


def _calculate_hurst_exponent(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling Hurst exponent using R/S analysis on a price series.

    This is a small, self-contained version of the helper used in the
    risk regime step so the feature bank can compute the same Hurst
    exponent series without depending on the step implementation.
    """
    values = series.astype(float).values
    hurst_values = []

    for i in range(len(values)):
        if i < window:
            hurst_values.append(np.nan)
            continue

        window_data = values[i - window : i]

        try:
            mean_val = np.mean(window_data)
            deviations = window_data - mean_val
            cumulative_deviations = np.cumsum(deviations)

            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            S = np.std(window_data)

            if S > 0 and R > 0:
                hurst = np.log(R / S) / np.log(window)
                hurst_values.append(hurst)
            else:
                hurst_values.append(0.5)
        except Exception:
            hurst_values.append(0.5)

    return pd.Series(hurst_values, index=series.index)
