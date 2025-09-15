"""
Base Calculations for Feature Generation

This module provides base calculation methods that can be used by different
feature generators, including price returns and returns-based VWAP calculations.
"""

from .base_calculator import (
    BaseCalculator,
    PriceReturnsCalculator,
    ReturnsVWAPCalculator,
    get_base_calculator,
    create_base_calculator
)

__all__ = [
    "BaseCalculator",
    "PriceReturnsCalculator", 
    "ReturnsVWAPCalculator",
    "get_base_calculator",
    "create_base_calculator"
]