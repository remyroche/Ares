"""Turnover and market impact helpers for backtesting engines."""

from __future__ import annotations

import logging
from typing import Any, Dict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - numpy is expected in prod but optional here
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover - pandas is expected in prod but optional here
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


def _ensure_pandas_series(name: str, series: Any) -> "pd.Series":
    if not PANDAS_AVAILABLE or pd is None:  # pragma: no cover - defensive
        raise ImportError("pandas is required for turnover calculations")

    if isinstance(series, pd.Series):
        return series

    try:
        return pd.Series(series)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be convertible to a pandas Series") from exc


def calculate_turnover_metrics(
    positions: "pd.Series",
    returns: "pd.Series",
) -> Dict[str, float]:
    """Calculate turnover related metrics for a strategy.

    Args:
        positions: Position sizes per period.
        returns: Strategy returns per period (unused for turnover but kept for
            signature parity with review doc).

    Returns:
        Dictionary with turnover, annualized turnover, holding period and
        position stability metrics.
    """

    positions = _ensure_pandas_series("positions", positions)
    _ensure_pandas_series("returns", returns)  # validation only

    if positions.empty:
        return {
            "turnover_per_period": 0.0,
            "turnover_annual": 0.0,
            "avg_holding_period_bars": 0.0,
            "position_stability": 0.0,
        }

    position_changes = positions.diff().abs().fillna(0.0)
    turnover_per_period = float(position_changes.mean())
    turnover_annual = float(turnover_per_period * 252)

    non_zero_changes = position_changes[position_changes > 0]
    if len(non_zero_changes) > 1:
        avg_holding_period = float(len(positions) / len(non_zero_changes))
    elif len(positions) > 0:
        avg_holding_period = float(len(positions))
    else:
        avg_holding_period = 0.0

    position_stability = float((position_changes == 0).mean())

    return {
        "turnover_per_period": turnover_per_period,
        "turnover_annual": turnover_annual,
        "avg_holding_period_bars": avg_holding_period,
        "position_stability": position_stability,
    }


def apply_market_impact_model(
    returns: "pd.Series",
    positions: "pd.Series",
    volume: "pd.Series",
    impact_coefficient: float = 0.1,
    capacity_limit_usd: float = 1e6,
    max_impact: float = 0.01,
) -> "pd.Series":
    """Apply a simple square-root market impact model to returns."""

    if not (PANDAS_AVAILABLE and NUMPY_AVAILABLE) or pd is None or np is None:  # pragma: no cover
        raise ImportError("pandas and numpy are required for market impact modeling")

    returns = _ensure_pandas_series("returns", returns).astype(float)
    positions = _ensure_pandas_series("positions", positions).astype(float)
    volume = _ensure_pandas_series("volume", volume).astype(float)

    aligned_returns, aligned_positions = returns.align(positions, join="outer", fill_value=0.0)
    aligned_returns, aligned_volume = aligned_returns.align(volume, join="outer", fill_value=0.0)
    aligned_positions = aligned_positions.reindex(aligned_returns.index).fillna(0.0)
    aligned_volume = aligned_volume.reindex(aligned_returns.index).fillna(0.0)

    position_changes = aligned_positions.diff().abs().fillna(0.0)
    safe_volume = aligned_volume.clip(lower=1.0)
    relative_trade_size = position_changes / safe_volume

    market_impact = impact_coefficient * np.sqrt(relative_trade_size.clip(lower=0.0))
    market_impact = market_impact.clip(upper=max_impact)

    net_returns = aligned_returns - market_impact

    max_trade_size = float(position_changes.max()) if len(position_changes) else 0.0
    if max_trade_size > capacity_limit_usd:
        logger.warning(
            "Strategy exceeds capacity limit: max trade size %.2f > %.2f",
            max_trade_size,
            capacity_limit_usd,
        )

    return net_returns


def reject_high_turnover_configs(
    strategy_results: Dict[str, Any],
    max_turnover_annual: float = 50.0,
    max_sharpe_to_turnover_ratio: float = 0.1,
) -> bool:
    """Return True when strategy results should be rejected for high turnover."""

    turnover = float(strategy_results.get("turnover_annual", 0.0) or 0.0)
    sharpe = float(strategy_results.get("sharpe_ratio", 0.0) or 0.0)

    if turnover > max_turnover_annual:
        logger.warning(
            "Config rejected: Turnover %.2fx exceeds max %.2fx",
            turnover,
            max_turnover_annual,
        )
        return True

    if turnover > 0:
        sharpe_to_turnover = sharpe / turnover if turnover else 0.0
        if sharpe_to_turnover < max_sharpe_to_turnover_ratio:
            logger.warning(
                "Config rejected: Sharpe/turnover ratio %.3f below min %.3f",
                sharpe_to_turnover,
                max_sharpe_to_turnover_ratio,
            )
            return True

    return False
