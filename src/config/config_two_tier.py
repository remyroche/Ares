# src/config/config_two_tier.py

"""
Configuration file for optimizable two-tier system parameters.
These parameters can be optimized in step12.
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class TwoTierConfig:
    """Optimizable two-tier system parameters."""
    
    # Two-tier enablement
    enable_two_tier: bool = True
    
    # Tier 1 (Direction/Strategy) parameters
    tier1_timeframes: list[str] = None
    direction_threshold: float = 0.7
    high_leverage_mode: bool = True
    
    # Tier 2 (Timing) parameters
    tier2_timeframes: list[str] = None
    timing_threshold: float = 0.8
    
    # Two-tier integration parameters
    tier1_weight: float = 0.6
    tier2_weight: float = 0.4
    min_combined_confidence: float = 0.75
    
    # Position sizing adjustments
    tier1_position_multiplier: float = 1.0
    tier2_position_multiplier: float = 1.2
    combined_position_multiplier: float = 1.1
    
    # Risk management adjustments
    tier1_risk_multiplier: float = 1.0
    tier2_risk_multiplier: float = 0.8
    combined_risk_multiplier: float = 0.9
    
    # Confidence thresholds for two-tier decisions
    tier1_confidence_threshold: float = 0.7
    tier2_confidence_threshold: float = 0.8
    combined_confidence_threshold: float = 0.75
    
    # Timing-specific parameters
    timing_lookback_periods: int = 10
    timing_momentum_threshold: float = 0.2
    timing_volume_threshold: float = 1.5
    
    # Strategy classification thresholds
    momentum_breakout_threshold: float = 0.6
    mean_reversion_threshold: float = 0.4
    trend_following_threshold: float = 0.7
    
    # Strategy-specific confidence thresholds
    momentum_confidence_threshold: float = 0.75
    mean_reversion_confidence_threshold: float = 0.8
    trend_following_confidence_threshold: float = 0.7
    
    # Strategy-specific position sizing
    momentum_position_multiplier: float = 1.2
    mean_reversion_position_multiplier: float = 0.8
    trend_following_position_multiplier: float = 1.0
    
    # Strategy-specific risk management
    momentum_risk_multiplier: float = 1.1
    mean_reversion_risk_multiplier: float = 0.9
    trend_following_risk_multiplier: float = 1.0
    
    def __post_init__(self):
        if self.tier1_timeframes is None:
            self.tier1_timeframes = ["1m", "5m", "15m", "1h"]
        
        if self.tier2_timeframes is None:
            self.tier2_timeframes = ["1m", "5m"]


def get_two_tier_config() -> TwoTierConfig:
    """Get two-tier configuration."""
    return TwoTierConfig()


def get_two_tier_search_space() -> dict[str, dict[str, Any]]:
    """Get search space for two-tier optimization."""
    return {
        "direction_threshold": {"min": 0.6, "max": 0.9, "type": "float"},
        "timing_threshold": {"min": 0.7, "max": 0.95, "type": "float"},
        "tier1_weight": {"min": 0.4, "max": 0.8, "type": "float"},
        "tier2_weight": {"min": 0.2, "max": 0.6, "type": "float"},
        "min_combined_confidence": {"min": 0.7, "max": 0.9, "type": "float"},
        "tier1_position_multiplier": {"min": 0.8, "max": 1.2, "type": "float"},
        "tier2_position_multiplier": {"min": 1.0, "max": 1.5, "type": "float"},
        "combined_position_multiplier": {"min": 0.9, "max": 1.3, "type": "float"},
        "tier1_risk_multiplier": {"min": 0.8, "max": 1.2, "type": "float"},
        "tier2_risk_multiplier": {"min": 0.6, "max": 1.0, "type": "float"},
        "combined_risk_multiplier": {"min": 0.7, "max": 1.1, "type": "float"},
        "tier1_confidence_threshold": {"min": 0.6, "max": 0.9, "type": "float"},
        "tier2_confidence_threshold": {"min": 0.7, "max": 0.95, "type": "float"},
        "combined_confidence_threshold": {"min": 0.7, "max": 0.9, "type": "float"},
        "timing_lookback_periods": {"min": 5, "max": 20, "type": "int"},
        "timing_momentum_threshold": {"min": 0.1, "max": 0.4, "type": "float"},
        "timing_volume_threshold": {"min": 1.2, "max": 2.0, "type": "float"},
        "momentum_breakout_threshold": {"min": 0.5, "max": 0.8, "type": "float"},
        "mean_reversion_threshold": {"min": 0.3, "max": 0.6, "type": "float"},
        "trend_following_threshold": {"min": 0.6, "max": 0.9, "type": "float"},
        # Strategy-specific confidence thresholds
        "momentum_confidence_threshold": {"min": 0.7, "max": 0.9, "type": "float"},
        "mean_reversion_confidence_threshold": {"min": 0.75, "max": 0.9, "type": "float"},
        "trend_following_confidence_threshold": {"min": 0.65, "max": 0.85, "type": "float"},
        # Strategy-specific position sizing
        "momentum_position_multiplier": {"min": 1.0, "max": 1.5, "type": "float"},
        "mean_reversion_position_multiplier": {"min": 0.6, "max": 1.0, "type": "float"},
        "trend_following_position_multiplier": {"min": 0.8, "max": 1.2, "type": "float"},
        # Strategy-specific risk management
        "momentum_risk_multiplier": {"min": 1.0, "max": 1.3, "type": "float"},
        "mean_reversion_risk_multiplier": {"min": 0.8, "max": 1.0, "type": "float"},
        "trend_following_risk_multiplier": {"min": 0.9, "max": 1.1, "type": "float"},
    }