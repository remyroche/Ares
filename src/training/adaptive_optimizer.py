# src/training/adaptive_optimizer.py

from typing import Any





def __init__(
self,
name: str,
volatility: float,
trend_strength: float,
regime_type: str,
optimal_params: dict[str, Any],
) -> None:
        self.name = name
self.volatility = volatility
self.trend_strength = trend_strength
self.regime_type = regime_type
self.optimal_params = optimal_params
self.confidence = 0.0




