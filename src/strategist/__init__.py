# src/strategist/__init__.py
# This file makes the 'strategist' directory a Python package.

# Import main strategist classes for easier access
from .strategist import BaseStrategist, StrategyConfig, SimpleMovingAverageStrategist
from .volatility_targeting_strategy import VolatilityTargetingStrategy, VolatilityTargetingConfig, VolatilityMethod

__all__ = [
    "BaseStrategist",
    "StrategyConfig", 
    "SimpleMovingAverageStrategist",
    "VolatilityTargetingStrategy",
    "VolatilityTargetingConfig",
    "VolatilityMethod",
]
