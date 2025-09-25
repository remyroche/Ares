"""Position-aware trading adapters that reuse the unified NAS/TAS utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.utils.nas_tas.shared_utils.position_aware_trading import (
    PositionAwareConfig,
    PositionAwareResult,
    PositionAwareTradingAnalyzer,
    create_position_aware_analyzer,
    quick_position_aware_analysis,
)


class PositionAwareTrading(PositionAwareTradingAnalyzer):
    """Backward compatible wrapper for TAS workflows."""

    def __init__(self, config: Optional[PositionAwareConfig] = None, evaluation_config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.evaluation_config = evaluation_config or {}

    def evaluate(
        self,
        market_data: pd.DataFrame,
        regime_predictions: np.ndarray,
        position_directions: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        return self.calculate_position_aware_trading_viability(market_data, regime_predictions, position_directions)

    def __enter__(self) -> "PositionAwareTrading":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # No resources to release – kept for compatibility with old context manager usage
        return None


def create_position_aware_trading(
    config: Optional[PositionAwareConfig] = None,
    evaluation_config: Optional[Dict[str, Any]] = None,
) -> PositionAwareTrading:
    return PositionAwareTrading(config, evaluation_config)


def quick_position_analysis(
    market_data: pd.DataFrame,
    regime_predictions: np.ndarray,
    position_directions: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    return quick_position_aware_analysis(market_data, regime_predictions, position_directions)


__all__ = [
    "PositionAwareTrading",
    "PositionAwareTradingAnalyzer",
    "PositionAwareConfig",
    "PositionAwareResult",
    "create_position_aware_trading",
    "create_position_aware_analyzer",
    "quick_position_analysis",
    "quick_position_aware_analysis",
]
