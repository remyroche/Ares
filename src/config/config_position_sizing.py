# src/config/config_position_sizing.py

"""
Configuration file for optimizable position sizing parameters.
These parameters can be optimized in step12.
"""

from typing import Any
from dataclasses import dataclass


@dataclass class PlaceholderDataClass: pass  # TODO: Add implementation class PositionSizingConfig: pass  # TODO: Add implementation class PositionSizingConfig: pass  # TODO: Add implementation class PositionSizingConfig: """Optimizable position sizing parameters."""  # Base position sizing base_position_size: float = 0.05  # 5% of portfolio max_position_size: float = 0.3  # 30% maximum position size min_position_size: float = 0.01  # 1% minimum position size  # Confidence-based scaling confidence_based_scaling: bool = True confidence_thresholds: dict[str, float] = None position_size_multipliers: dict[str, float] = None  # Volatility adjustment enable_volatility_scaling: bool = True atr_multiplier: float = 1.0 volatility_thresholds: dict[str, float] = None volatility_multipliers: dict[str, float] = None  # Liquidation risk adjustment enable_liquidation_scaling: bool = True lss_thresholds: dict[str, float] = None lss_multipliers: dict[str, float] = None  # Successive position rules enable_successive_positions: bool = True min_confidence_for_successive: float = 0.85 max_successive_positions: int = 3 position_spacing_minutes: int = 15 size_reduction_factor: float = 0.8 max_total_exposure: float = 0.3  # Risk limits max_single_position: float = 0.15 max_total_exposure: float = 0.3 max_correlation_exposure: float = 0.2 max_leverage: float = 10.0  # Kelly criterion parameters kelly_multiplier: float = 0.5 kelly_max_fraction: float = 0.25  # Dynamic risk management enable_dynamic_risk: bool = True drawdown_thresholds: dict[str, float] = None position_size_reductions: dict[str, float] = None  def __post_init__(self): def __post_init__(self): def __post_init__(self): def __post_init__(self): if self.confidence_thresholds is None: self.confidence_thresholds , { "low_confidence": 0.6, "medium_confidence": 0.75, "high_confidence": 0.85, "very_high_confidence": 0.95, }  if self.position_size_multipliers is None: self.position_size_multipliers = { "low_confidence": 0.5, "medium_confidence": 1.0, "high_confidence": 1.5, "very_high_confidence": 2.0, }  if self.volatility_thresholds is None: self.volatility_thresholds = { "low_volatility": 0.02, "medium_volatility": 0.05, "high_volatility": 0.10, }  if self.volatility_multipliers is None: self.volatility_multipliers = { "low_volatility": 1.2, "medium_volatility": 1.0, "high_volatility": 0.7, }  if self.lss_thresholds is None: self.lss_thresholds = { "very_safe": 80, "safe": 60, "moderate": 40, }  if self.lss_multipliers is None: self.lss_multipliers = { "very_safe": 1.2, "safe": 1.0, "moderate": 0.8, "dangerous": 0.5, }  if self.drawdown_thresholds is None: self.drawdown_thresholds = { "light": 0.05, "moderate": 0.15, "severe": 0.25, }  if self.position_size_reductions is None: self.position_size_reductions = { "light": 0.8, "moderate": 0.5, "severe": 0.2, }   def get_position_sizing_config() -> PositionSizingConfig: """Get position sizing configuration.""" return PositionSizingConfig()


def get_position_sizing_search_space() -> dict[str, dict[str, Any]]:
    """Get search space for position sizing optimization."""
    return {
"base_position_size": {"min": 0.02, "max": 0.15, "type": "float"},
"max_position_size": {"min": 0.15, "max": 0.5, "type": "float"},
"min_position_size": {"min": 0.005, "max": 0.03, "type": "float"},
"atr_multiplier": {"min": 0.5, "max": 2.0, "type": "float"},
"min_confidence_for_successive": {"min": 0.8, "max": 0.95, "type": "float"},
"max_successive_positions": {"min": 2, "max": 5, "type": "int"},
"size_reduction_factor": {"min": 0.6, "max": 0.9, "type": "float"},
"max_total_exposure": {"min": 0.2, "max": 0.5, "type": "float"},
"max_single_position": {"min": 0.1, "max": 0.25, "type": "float"},
"max_correlation_exposure": {"min": 0.15, "max": 0.3, "type": "float"},
"max_leverage": {"min": 5.0, "max": 20.0, "type": "float"},
"kelly_multiplier": {"min": 0.3, "max": 0.8, "type": "float"},
"kelly_max_fraction": {"min": 0.15, "max": 0.4, "type": "float"},
# Confidence thresholds
"confidence_thresholds.low_confidence": {"min": 0.5, "max": 0.7, "type": "float"},
"confidence_thresholds.medium_confidence": {"min": 0.7, "max": 0.8, "type": "float"},
"confidence_thresholds.high_confidence": {"min": 0.8, "max": 0.9, "type": "float"},
"confidence_thresholds.very_high_confidence": {"min": 0.9, "max": 0.98, "type": "float"},
# Position size multipliers
"position_size_multipliers.low_confidence": {"min": 0.3, "max": 0.7, "type": "float"},
"position_size_multipliers.medium_confidence": {"min": 0.8, "max": 1.2, "type": "float"},
"position_size_multipliers.high_confidence": {"min": 1.2, "max": 2.0, "type": "float"},
"position_size_multipliers.very_high_confidence": {"min": 1.5, "max": 3.0, "type": "float"},
# Volatility thresholds
"volatility_thresholds.low_volatility": {"min": 0.01, "max": 0.03, "type": "float"},
"volatility_thresholds.medium_volatility": {"min": 0.03, "max": 0.07, "type": "float"},
"volatility_thresholds.high_volatility": {"min": 0.07, "max": 0.15, "type": "float"},
# Volatility multipliers
"volatility_multipliers.low_volatility": {"min": 1.0, "max": 1.5, "type": "float"},
"volatility_multipliers.medium_volatility": {"min": 0.8, "max": 1.2, "type": "float"},
"volatility_multipliers.high_volatility": {"min": 0.5, "max": 0.9, "type": "float"},
}