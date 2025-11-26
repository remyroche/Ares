"""Liquidity regime feature generation.

This module centralizes the core liquidity features used by the
`MLLiquidityRegimeStep` so they live in the feature bank.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd


def generate_liquidity_regime_features(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Generate core liquidity regime features.

    This mirrors the core OHLCV-derived liquidity features from
    `MLLiquidityRegimeStep._generate_liquidity_features` so that their
    definitions are centralized in the feature bank.

    The function expects 15m OHLCV data with at least ``open``, ``high``,
    ``low``, ``close`` and ``volume`` columns.
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns for liquidity features: {missing}")

    df = df.copy()
    eps = 1e-9

    # Basic derived quantities
    df["range"] = (df["high"] - df["low"]).astype(float)
    df["range"] = df["range"].replace(0, np.nan)
    df["return_1h"] = np.log(df["close"] / df["close"].shift(1)).astype(float)
    df["abs_return_1h"] = df["return_1h"].abs()
    df["dollar_volume"] = (df["close"] * df["volume"]).astype(float)

    # Relative volume context
    vol_window_daily = int(config.get("liquidity_rvol_lookback_24", 96))
    vol_window_weekly = int(config.get("liquidity_rvol_lookback_168", 672))

    df["vol_sma_24"] = df["volume"].rolling(vol_window_daily, min_periods=5).mean()
    df["vol_sma_168"] = df["volume"].rolling(vol_window_weekly, min_periods=20).mean()
    df["rvol_24"] = df["volume"] / (df["vol_sma_24"] + eps)
    df["rvol_168"] = df["volume"] / (df["vol_sma_168"] + eps)

    # RVOL: Relative Volume (rolling 20-bar lookback for regime classification)
    df["vol_sma_20"] = df["volume"].rolling(80, min_periods=5).mean()
    df["rvol_20"] = df["volume"] / (df["vol_sma_20"] + eps)

    # VER: Volume-Efficiency Ratio (Volume / Range)
    df["volume_efficiency_ratio"] = df["volume"] / (df["range"] + eps)

    vol_mean_24 = df["volume"].rolling(vol_window_daily, min_periods=5).mean()
    vol_std_24 = df["volume"].rolling(vol_window_daily, min_periods=5).std()
    df["vol_z_24"] = (df["volume"] - vol_mean_24) / (vol_std_24.replace(0, np.nan) + eps)

    df["volume_stddev_stability"] = (
        df["volume"].rolling(24, min_periods=3).std()
        / (df["volume"].rolling(24, min_periods=3).mean() + eps)
    )

    # Additional stability features for regime contrast
    df["range_stddev_stability"] = (
        df["range"].rolling(24, min_periods=3).std()
        / (df["range"].rolling(24, min_periods=3).mean() + eps)
    )
    df["return_stddev_stability"] = (
        df["abs_return_1h"].rolling(24, min_periods=3).std()
        / (df["abs_return_1h"].rolling(24, min_periods=3).mean() + eps)
    )

    # Normalized range (Effort)
    range_std_lookback = int(config.get("liquidity_range_std_lookback", 192))
    range_std = df["range"].rolling(range_std_lookback, min_periods=10).std()
    df["normalized_range"] = df["range"] / (range_std.replace(0, np.nan) + eps)

    # Effort vs Result ratios
    df["normalized_volume"] = np.log1p(df["volume"])  # log volume
    df["ghost_ratio"] = df["normalized_range"] / (df["normalized_volume"] + eps)
    df["absorption_ratio"] = df["normalized_volume"] / (df["normalized_range"] + eps)

    # Amihud / Amivest
    df["amihud_validity"] = df["abs_return_1h"] / (df["dollar_volume"] + eps)
    df["amivest_efficiency"] = df["dollar_volume"] / (df["abs_return_1h"] + eps)

    # Amihud spike ratio: normalize by rolling baseline to detect illiquidity spikes
    df["amihud_baseline"] = df["amihud_validity"].rolling(96, min_periods=6).median()
    df["amihud_spike_ratio"] = df["amihud_validity"] / (df["amihud_baseline"] + eps)

    return df
