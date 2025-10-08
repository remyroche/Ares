"""Utilities for evaluating turnover and capacity constraints in backtests."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.tprint import tprint_warning, tprint_info, tprint_success, tprint_error
from src.utils.common_operations import (
    safe_divide, safe_mean, safe_std, validate_finite, optimize_dataframe_dtypes,
    calculate_data_quality_metrics, get_dataframe_info
)
from src.utils.math_validation import safe_correlation, safe_covariance
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations

SeriesLike = Union[pd.Series, np.ndarray, Dict[int, float], Tuple[float, ...], list]


def _to_series(value: Optional[SeriesLike], index: Optional[pd.Index] = None) -> pd.Series:
    """Convert arbitrary sequence-like input to a float ``Series`` aligned to *index* with enhanced utilities.

    Parameters
    ----------
    value:
        Series-like structure. ``None`` yields an empty series.
    index:
        Optional index to align the data to. If provided and the lengths do not
        match, the resulting series is reindexed to ``index`` with missing values
        filled by zero.
    """
    tprint_info(f"🔄 Converting input to Series with {len(value) if value is not None else 0} elements")

    try:
        if value is None:
            tprint_info("📊 Creating empty Series")
            return pd.Series(dtype=float)

        if isinstance(value, pd.Series):
            tprint_info("📊 Converting existing Series")
            series = value.astype(float)
            if index is not None:
                series = series.reindex(index)
            # Use safe fillna from common operations
            series = series.fillna(0.0)
            return series

        # Convert to numpy array with enhanced error handling
        array = np.asarray(value, dtype=float)

        # Validate array is finite using math validation utilities
        array = validate_finite(array, "input array")

        if index is not None and len(array) == len(index):
            tprint_info(f"📊 Creating Series with matching index ({len(array)} elements)")
            return pd.Series(array, index=index, dtype=float)

        if index is not None:
            tprint_info(f"📊 Creating Series with index padding ({len(index)} target, {len(array)} source)")
            series = pd.Series(np.zeros(len(index), dtype=float), index=index)
            # Use safe operations for array slicing
            safe_len = min(len(array), len(index))
            series.iloc[:safe_len] = array[:safe_len]
            return series

        tprint_info(f"📊 Creating Series from array ({len(array)} elements)")
        return pd.Series(array, dtype=float)

    except Exception as e:
        tprint_error(f"❌ Failed to convert input to Series: {e}")
        # Return empty series as fallback
        return pd.Series(dtype=float)


def calculate_turnover_metrics(
    positions: SeriesLike,
    returns: SeriesLike,
    position_changes: Optional[SeriesLike] = None,
    *,
    periods_per_year: float = 252.0,
) -> Dict[str, float]:
    """Calculate turnover diagnostics for a trading configuration with enhanced utilities.

    Parameters
    ----------
    positions:
        Position series (typically ``{-1, 0, 1}``).
    returns:
        Per-period strategy returns before trading frictions.
    position_changes:
        Optional pre-computed absolute position change series. When provided it
        avoids recomputation.
    periods_per_year:
        Number of evaluation periods used to annualise turnover.
    """
    tprint_info("📊 Calculating turnover metrics with enhanced utilities")

    try:
        # Get memory optimizer for performance tracking
        memory_optimizer = get_m1_memory_optimizer()

        # Initialize matrix operations for advanced correlation analysis
        matrix_ops = UnifiedMatrixOperations()

        # Convert inputs using enhanced utilities
        positions_series = _to_series(positions)
        returns_series = _to_series(returns, positions_series.index)

        # Optimize data types for better memory usage
        positions_series = optimize_dataframe_dtypes(positions_series.to_frame()).iloc[:, 0]
        returns_series = optimize_dataframe_dtypes(returns_series.to_frame()).iloc[:, 0]

        # Calculate position changes if not provided
        if position_changes is None:
            tprint_info("🔄 Computing position changes")
            position_changes_series = positions_series.diff().abs()
        else:
            position_changes_series = _to_series(position_changes, positions_series.index).abs()

        # Handle first position (no previous position to diff from)
        if not position_changes_series.empty and len(positions_series) > 0:
            position_changes_series.iloc[0] = abs(positions_series.iloc[0])

        # Calculate metrics using safe operations
        turnover_per_period = safe_mean(position_changes_series) if not position_changes_series.empty else 0.0
        turnover_annual = safe_divide(turnover_per_period * periods_per_year, 1.0)

        # Calculate average holding period
        non_zero_changes = position_changes_series[position_changes_series > 0]
        if len(non_zero_changes) > 1:
            avg_holding_period = safe_divide(len(positions_series), len(non_zero_changes))
        else:
            avg_holding_period = float(len(positions_series))

        # Calculate position stability
        stability = safe_mean((position_changes_series == 0).astype(float)) if not position_changes_series.empty else 0.0

        # Calculate additional correlation metrics using enhanced utilities and matrix operations
        correlation_returns = safe_correlation(positions_series, returns_series)
        correlation_changes = safe_correlation(positions_series, position_changes_series) if not position_changes_series.empty else 0.0

        # Use matrix operations for more sophisticated correlation analysis
        try:
            # Create correlation matrix for multiple time series
            combined_data = pd.DataFrame({
                'positions': positions_series,
                'returns': returns_series,
                'position_changes': position_changes_series
            }).dropna()

            if len(combined_data) > 10:  # Need sufficient data for correlation matrix
                correlation_matrix = matrix_ops.compute_correlation_matrix(combined_data.values)
                if correlation_matrix is not None and correlation_matrix.shape == (3, 3):
                    # Extract pairwise correlations from matrix
                    pos_returns_corr = correlation_matrix[0, 1]  # positions vs returns
                    pos_changes_corr = correlation_matrix[0, 2]  # positions vs changes
                    returns_changes_corr = correlation_matrix[1, 2]  # returns vs changes

                    # Use matrix-based correlations if they are more reliable
                    if np.isfinite(pos_returns_corr) and abs(pos_returns_corr - correlation_returns) < 0.1:
                        correlation_returns = pos_returns_corr

                    if np.isfinite(pos_changes_corr) and not position_changes_series.empty:
                        correlation_changes = pos_changes_corr

                    tprint_debug(f"📊 Matrix-based correlations computed: pos-returns={pos_returns_corr:.3f}, pos-changes={pos_changes_corr:.3f}")
                else:
                    tprint_debug("📊 Matrix correlation computation failed, using scalar correlations")
            else:
                tprint_debug("📊 Insufficient data for matrix correlation analysis")

        except Exception as e:
            tprint_warning(f"⚠️ Matrix correlation analysis failed: {e}")
            # Fall back to scalar correlations

        # Create comprehensive metrics dictionary
        metrics = {
            "turnover_per_period": turnover_per_period,
            "turnover_annual": turnover_annual,
            "avg_holding_period_bars": avg_holding_period,
            "position_stability": stability,
            "correlation_with_returns": correlation_returns,
            "correlation_with_changes": correlation_changes,
            "total_positions": len(positions_series),
            "memory_usage_mb": memory_optimizer.memory_pressure * 100 if memory_optimizer else 0.0,
        }

        # Validate all metrics are finite
        for key, value in metrics.items():
            if not np.isfinite(value):
                tprint_warning(f"⚠️ Non-finite metric detected: {key} = {value}")
                metrics[key] = 0.0

        tprint_success(f"✅ Turnover metrics calculated: annual={turnover_annual:.3f}x, stability={stability:.3f}")
        return metrics

    except Exception as e:
        tprint_error(f"❌ Failed to calculate turnover metrics: {e}")
        return {
            "turnover_per_period": 0.0,
            "turnover_annual": 0.0,
            "avg_holding_period_bars": 0.0,
            "position_stability": 0.0,
            "correlation_with_returns": 0.0,
            "correlation_with_changes": 0.0,
            "total_positions": 0,
            "memory_usage_mb": 0.0,
            "error": str(e)
        }


def apply_market_impact_model(
    returns: SeriesLike,
    position_changes: SeriesLike,
    volume: Optional[SeriesLike] = None,
    *,
    impact_coefficient: float = 0.1,
    capacity_limit_usd: float = 1_000_000.0,
    max_impact_per_trade: float = 0.01,
) -> Tuple[pd.Series, pd.Series, bool, float]:
    """Apply a square-root market impact model to returns.

    Parameters
    ----------
    returns:
        Returns prior to market impact adjustments (typically cost-adjusted).
    position_changes:
        Absolute position change per period.
    volume:
        Series representing available market volume. ``None`` defaults to a
        unit-volume proxy.
    impact_coefficient:
        Scaling factor applied to the impact term.
    capacity_limit_usd:
        Maximum allowable trade size before triggering a capacity warning.
    max_impact_per_trade:
        Hard cap on the impact deduction per trade (expressed in return units).

    Returns
    -------
    Tuple[pd.Series, pd.Series, bool, float]
        Impact-adjusted returns, per-period impact costs, flag indicating
        whether capacity was exceeded, and the maximum trade size observed.
    """

    returns_series = _to_series(returns)
    index = returns_series.index
    position_change_series = _to_series(position_changes, index).abs()

    if volume is None:
        volume_series = pd.Series(1.0, index=index, dtype=float)
    else:
        volume_series = _to_series(volume, index)
        volume_series.replace(0.0, np.nan, inplace=True)
        volume_series.fillna(1.0, inplace=True)

    relative_trade_size = position_change_series / volume_series
    relative_trade_size = relative_trade_size.clip(lower=0.0)
    market_impact = impact_coefficient * np.sqrt(relative_trade_size)
    market_impact = market_impact.clip(upper=max_impact_per_trade)

    net_returns = returns_series - market_impact

    max_trade_size = float(position_change_series.max()) if not position_change_series.empty else 0.0
    capacity_exceeded = max_trade_size > capacity_limit_usd
    if capacity_exceeded:
        tprint_warning(
            "Strategy exceeds capacity limit: "
            f"max_trade_size=${max_trade_size:,.0f} > ${capacity_limit_usd:,.0f}"
        )

    return net_returns, market_impact, capacity_exceeded, max_trade_size


def reject_high_turnover_configs(
    strategy_results: Dict[str, Any],
    *,
    max_turnover_annual: float = 50.0,
    max_sharpe_to_turnover_ratio: float = 0.1,
) -> Tuple[bool, Optional[str]]:
    """Determine whether a strategy should be rejected due to high turnover."""

    turnover = float(strategy_results.get("turnover_annual", 0.0) or 0.0)
    sharpe = float(strategy_results.get("sharpe", 0.0) or 0.0)

    if turnover > max_turnover_annual:
        message = (
            f"Config rejected: Turnover ({turnover:.1f}x) exceeds max ({max_turnover_annual:.1f}x)"
        )
        tprint_warning(message)
        return True, message

    if turnover > 0:
        sharpe_to_turnover = sharpe / turnover
        if sharpe_to_turnover < max_sharpe_to_turnover_ratio:
            message = (
                "Config rejected: Sharpe/turnover ratio "
                f"({sharpe_to_turnover:.3f}) too low (min: {max_sharpe_to_turnover_ratio:.3f})"
            )
            tprint_warning(message)
            return True, message

    return False, None


__all__ = [
    "calculate_turnover_metrics",
    "apply_market_impact_model",
    "reject_high_turnover_configs",
]
