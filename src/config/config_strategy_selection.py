# src/config/config_strategy_selection.py

"""
Configuration file for optimizable strategy selection parameters.
These parameters determine which models to use in the two-tier system.
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class StrategySelectionConfig:
    """Optimizable strategy selection parameters."""
    
    # Strategy selection thresholds
    momentum_selection_threshold: float = 0.7
    mean_reversion_selection_threshold: float = 0.6
    trend_following_selection_threshold: float = 0.65
    
    # Model selection criteria
    min_model_confidence: float = 0.6
    min_model_performance: float = 0.55
    max_model_correlation: float = 0.8
    
    # Strategy-specific model preferences
    momentum_preferred_models: list[str] = None
    mean_reversion_preferred_models: list[str] = None
    trend_following_preferred_models: list[str] = None
    
    # Model diversity requirements
    min_model_diversity: float = 0.3
    max_ensemble_size: int = 5
    min_ensemble_size: int = 2
    
    # Strategy switching thresholds
    strategy_switch_threshold: float = 0.2
    strategy_confirm_periods: int = 3
    strategy_stability_threshold: float = 0.7
    
    # Performance-based selection
    performance_lookback_periods: int = 20
    performance_decay_factor: float = 0.95
    min_performance_threshold: float = 0.5
    
    # Risk-based selection
    max_risk_per_model: float = 0.15
    risk_adjustment_factor: float = 1.0
    enable_risk_based_selection: bool = True
    
    def __post_init__(self):
        if self.momentum_preferred_models is None:
            self.momentum_preferred_models = ["lstm", "transformer", "lightgbm"]
        
        if self.mean_reversion_preferred_models is None:
            self.mean_reversion_preferred_models = ["random_forest", "svm", "neural_network"]
        
        if self.trend_following_preferred_models is None:
            self.trend_following_preferred_models = ["lstm", "lightgbm", "gradient_boosting"]


def get_strategy_selection_config() -> StrategySelectionConfig:
    """Get strategy selection configuration."""
    return StrategySelectionConfig()


def get_strategy_selection_search_space() -> dict[str, dict[str, Any]]:
    """Get search space for strategy selection optimization."""
    return {
        # Strategy selection thresholds
        "momentum_selection_threshold": {"min": 0.6, "max": 0.85, "type": "float"},
        "mean_reversion_selection_threshold": {"min": 0.5, "max": 0.8, "type": "float"},
        "trend_following_selection_threshold": {"min": 0.55, "max": 0.8, "type": "float"},
        
        # Model selection criteria
        "min_model_confidence": {"min": 0.5, "max": 0.75, "type": "float"},
        "min_model_performance": {"min": 0.5, "max": 0.7, "type": "float"},
        "max_model_correlation": {"min": 0.7, "max": 0.9, "type": "float"},
        
        # Model diversity requirements
        "min_model_diversity": {"min": 0.2, "max": 0.5, "type": "float"},
        "max_ensemble_size": {"min": 3, "max": 7, "type": "int"},
        "min_ensemble_size": {"min": 2, "max": 4, "type": "int"},
        
        # Strategy switching thresholds
        "strategy_switch_threshold": {"min": 0.1, "max": 0.3, "type": "float"},
        "strategy_confirm_periods": {"min": 2, "max": 5, "type": "int"},
        "strategy_stability_threshold": {"min": 0.6, "max": 0.85, "type": "float"},
        
        # Performance-based selection
        "performance_lookback_periods": {"min": 10, "max": 30, "type": "int"},
        "performance_decay_factor": {"min": 0.9, "max": 0.99, "type": "float"},
        "min_performance_threshold": {"min": 0.4, "max": 0.6, "type": "float"},
        
        # Risk-based selection
        "max_risk_per_model": {"min": 0.1, "max": 0.2, "type": "float"},
        "risk_adjustment_factor": {"min": 0.8, "max": 1.2, "type": "float"},
    }