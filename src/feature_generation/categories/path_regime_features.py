"""Path geometry regime feature generation.

This module centralizes the core path-oriented features used by the
MLPathRegimeStep so they live in the feature bank.
"""

from typing import Any, Dict

import logging
import numpy as np
import pandas as pd

from .entropy import PermutationEntropyGenerator

logger = logging.getLogger(__name__)


def generate_path_regime_features(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Generate core path geometry features without normalization.

    This mirrors the computation currently performed in
    ``MLPathRegimeStep._generate_risk_features`` (which is really the
    path-geometry feature block), so that the feature definitions are
    centralized in the feature bank.
    """
    result_df = df.copy()

    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in result_df.columns]
    if missing:
        raise ValueError(f"Missing columns for path features: {missing}")

    # Base 1h returns
    returns = np.log(result_df["close"] / result_df["close"].shift(1))
    result_df["returns_1h"] = returns

    # 3h return and a simple Sharpe-like ratio
    try:
        return_3h = returns.rolling(window=12).sum()
        result_df["return_3h"] = return_3h

        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0
        downside_dev = downside_returns.rolling(window=80).std()
        sharpe_like_3h = return_3h / (downside_dev + 1e-9)
        result_df["sharpe_like_3h"] = sharpe_like_3h
    except Exception as exc:
        logger.warning("Path return / Sharpe-like feature generation failed: %s", exc)

    # KER (path efficiency) over different horizons
    # OPTIMIZED: Reduced lookback from 200+ to max 48 for faster computation
    try:
        ker_window_main = int(config.get("path_ker_window_bars", 3))
        # Cap maximum window at 48 bars for efficiency
        max_window = min(int(config.get("path_max_lookback", 48)), 48)
        ker_windows = sorted({ker_window_main, min(24, max_window)})
        for n in ker_windows:
            if n > 1:
                price_change_n = (result_df["close"] - result_df["close"].shift(n)).abs()
                path_length_n = (
                    result_df["close"].diff().abs().rolling(window=n, min_periods=2).sum()
                )
                ker_series = price_change_n / (path_length_n + 1e-9)
                result_df[f"path_ker_{n}h"] = ker_series
    except Exception as exc:
        logger.warning("Path efficiency (KER) feature generation failed: %s", exc)

    # Efficiency × return features
    try:
        ker_col = f"path_ker_{ker_window_main}h"
        if ker_col in result_df.columns and "return_3h" in result_df.columns:
            ker_series_eff = result_df[ker_col]
            ret_3h_eff = result_df["return_3h"]
            result_df["path_efficiency_return_3h"] = ker_series_eff * ret_3h_eff
            # APPROACH 2: Make direction-agnostic by using absolute value
            result_df["path_directional_eff_3h"] = np.sign(ret_3h_eff) * ker_series_eff
            result_df["path_directional_eff_3h_abs"] = np.abs(result_df["path_directional_eff_3h"])
    except Exception as exc:
        logger.warning("Path efficiency-return feature generation failed: %s", exc)

    # NEW FEATURE: Efficiency Ratio (Net Change / Sum of Absolute Changes)
    # Measures path straightness: 1.0 = straight line, near 0 = maximum zigzag
    try:
        efficiency_window = int(config.get("path_efficiency_ratio_window", 12))
        net_change = (result_df["close"] - result_df["close"].shift(efficiency_window)).abs()
        sum_moves = result_df["close"].diff().abs().rolling(
            window=efficiency_window,
            min_periods=2
        ).sum()
        efficiency_ratio = net_change / (sum_moves + 1e-9)
        result_df["efficiency_ratio"] = efficiency_ratio.clip(0, 1)
    except Exception as exc:
        logger.warning("Efficiency ratio feature generation failed: %s", exc)

    # Body / range ratio
    try:
        body = (result_df["close"] - result_df["open"]).abs()
        range_bar = (result_df["high"] - result_df["low"]).replace(0, np.nan)
        brr = body / (range_bar + 1e-9)
        result_df["body_range_ratio"] = brr
    except Exception as exc:
        logger.warning("Body-to-range ratio feature generation failed: %s", exc)

    # Traffic / overlap measures
    try:
        range_bar = (result_df["high"] - result_df["low"]).replace(0, np.nan)
        overlap_high = np.minimum(result_df["high"], result_df["high"].shift(1))
        overlap_low = np.maximum(result_df["low"], result_df["low"].shift(1))
        overlap = (overlap_high - overlap_low).clip(lower=0.0)
        traffic = overlap / (range_bar + 1e-9)
        result_df["traffic_overlap"] = traffic
        result_df["traffic_overlap_3h"] = traffic.rolling(window=12, min_periods=1).mean()
    except Exception as exc:
        logger.warning("Traffic/overlap feature generation failed: %s", exc)

    # Path trend R^2 over a configurable window
    try:
        ker_col = f"path_ker_{ker_window_main}h"
        if ker_col in result_df.columns:
            trend_window = int(config.get("path_trend_r2_window_bars", ker_window_main * 2))

            def _rolling_r2(arr: np.ndarray) -> float:
                mask = np.isfinite(arr)
                if mask.sum() < 3:
                    return np.nan
                y = arr[mask]
                x = np.arange(len(y), dtype=float)
                if y.std() == 0.0 or x.std() == 0.0:
                    return 0.0
                corr = np.corrcoef(x, y)[0, 1]
                if not np.isfinite(corr):
                    return 0.0
                return float(corr * corr)

            r2_series = result_df["close"].rolling(
                window=trend_window,
                min_periods=3,
            ).apply(_rolling_r2, raw=True)
            result_df["path_trend_r2"] = r2_series
    except Exception as exc:
        logger.warning("Path trend R2 feature generation failed: %s", exc)

    # NEW FEATURE: Impulse Quality (path_trend_r2 × abs(path_directional_eff_3h))
    # Combines trend strength with directional efficiency (direction-agnostic)
    try:
        if "path_trend_r2" in result_df.columns and "path_directional_eff_3h_abs" in result_df.columns:
            impulse_quality = result_df["path_trend_r2"] * result_df["path_directional_eff_3h_abs"]
            result_df["impulse_quality"] = impulse_quality
    except Exception as exc:
        logger.warning("Impulse quality feature generation failed: %s", exc)

    # Permutation entropy on the price path
    try:
        perm_window = int(config.get("path_permutation_entropy_window", 20))
        embedding_dim = int(config.get("path_permutation_embedding_dim", 3))
        delay = int(config.get("path_permutation_delay", 1))
        pe_gen = PermutationEntropyGenerator(
            window=perm_window,
            embedding_dim=embedding_dim,
            delay=delay,
        )
        pe_series = pe_gen._generate_feature(result_df[["close"]].copy())
        result_df["path_permutation_entropy"] = pe_series
    except Exception as exc:
        logger.warning("Permutation entropy feature generation failed: %s", exc)

    # REMOVED: Fractal dimension (poor WCoV: 0.13, weak regime separation)
    # REMOVED: Hurst exponent (poor WCoV: 0.23, weak regime separation)

    # Path alpha helper metrics (trend-up flag, efficiency high/dropping, alpha_state)
    try:
        ker_col = f"path_ker_{ker_window_main}h"
        if ker_col in result_df.columns and "return_3h" in result_df.columns:
            ker_series = result_df[ker_col]
            ker_diff = ker_series.diff()
            path_trend_up = (result_df["return_3h"] > 0).astype(int)
            eff_high_thr = float(config.get("path_efficiency_high_threshold", 0.6))
            eff_drop_thr = float(config.get("path_efficiency_drop_threshold", 0.05))
            eff_high = (ker_series >= eff_high_thr).astype(int)
            eff_dropping = (ker_diff <= -eff_drop_thr).astype(int)
            alpha_state = np.zeros(len(result_df), dtype=int)
            hold_mask = (path_trend_up == 1) & (eff_high == 1) & (eff_dropping == 0)
            tp_mask = (path_trend_up == 1) & (eff_dropping == 1)
            alpha_state[hold_mask.values] = 1
            alpha_state[tp_mask.values] = 2
            result_df["path_trend_up"] = path_trend_up
            result_df["path_efficiency_high"] = eff_high
            result_df["path_efficiency_dropping"] = eff_dropping
            result_df["path_alpha_state"] = alpha_state
    except Exception as exc:
        logger.warning("Path alpha helper feature generation failed: %s", exc)

    # Keep base OHLCV plus the key path features
    base_cols = [c for c in df.columns]
    path_keep = [
        "returns_1h",
        "path_ker_3h",
        "path_ker_6h",
        "path_efficiency_return_3h",
        "path_directional_eff_3h",
        "path_directional_eff_3h_abs",  # NEW: Direction-agnostic version
        "efficiency_ratio",              # NEW: Net change / sum of moves
        "impulse_quality",               # NEW: Trend strength × efficiency
        "body_range_ratio",
        "traffic_overlap",
        "traffic_overlap_3h",
        "path_permutation_entropy",
        # "path_fractal_dimension",      # REMOVED: Poor WCoV (0.13)
        # "hurst_exponent_path",         # REMOVED: Poor WCoV (0.23)
        "path_trend_r2",
        "path_trend_up",
        "path_efficiency_high",
        "path_efficiency_dropping",
        "path_alpha_state",
    ]
    keep_cols = base_cols + [c for c in path_keep if c in result_df.columns]
    keep_cols = list(dict.fromkeys(keep_cols))

    # Explicitly wrap in DataFrame to satisfy static type-checkers
    return pd.DataFrame(result_df[keep_cols])
