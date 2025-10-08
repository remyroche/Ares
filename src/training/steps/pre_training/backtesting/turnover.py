"""Utilities for evaluating turnover and capacity constraints in backtests."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.tprint import tprint_warning

SeriesLike = Union[pd.Series, np.ndarray, Dict[int, float], Tuple[float, ...], list]


def _to_series(value: Optional[SeriesLike], index: Optional[pd.Index] = None) -> pd.Series:
    """Convert arbitrary sequence-like input to a float ``Series`` aligned to *index*.

    Parameters
    ----------
    value:
        Series-like structure. ``None`` yields an empty series.
    index:
        Optional index to align the data to. If provided and the lengths do not
        match, the resulting series is reindexed to ``index`` with missing values
        filled by zero.
    """

    if value is None:
        return pd.Series(dtype=float)

    if isinstance(value, pd.Series):
        series = value.astype(float)
        if index is not None:
            series = series.reindex(index)
        return series.fillna(0.0)

    array = np.asarray(value, dtype=float)
    if index is not None and len(array) == len(index):
        return pd.Series(array, index=index, dtype=float)
    if index is not None:
        series = pd.Series(np.zeros(len(index), dtype=float), index=index)
        series.iloc[: min(len(array), len(index))] = array[: len(index)]
        return series
    return pd.Series(array, dtype=float)


def calculate_turnover_metrics(
    positions: SeriesLike,
    returns: SeriesLike,
    position_changes: Optional[SeriesLike] = None,
    *,
    periods_per_year: float = 252.0,
) -> Dict[str, float]:
    """Calculate turnover diagnostics for a trading configuration.

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

    positions_series = _to_series(positions)
    returns_series = _to_series(returns, positions_series.index)

    if position_changes is None:
        position_changes_series = positions_series.diff().abs()
    else:
        position_changes_series = _to_series(position_changes, positions_series.index).abs()

    if not position_changes_series.empty:
        position_changes_series.iloc[0] = abs(positions_series.iloc[0])

    turnover_per_period = float(position_changes_series.mean()) if not position_changes_series.empty else 0.0
    turnover_annual = float(turnover_per_period * periods_per_year)

    non_zero_changes = position_changes_series[position_changes_series > 0]
    if len(non_zero_changes) > 1:
        avg_holding_period = float(len(positions_series) / len(non_zero_changes))
    else:
        avg_holding_period = float(len(positions_series))

    stability = float((position_changes_series == 0).mean()) if not position_changes_series.empty else 0.0

    return {
        "turnover_per_period": turnover_per_period,
        "turnover_annual": turnover_annual,
        "avg_holding_period_bars": avg_holding_period,
        "position_stability": stability,
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
