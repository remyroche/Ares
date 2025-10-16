"""
Centralized Leverage Configuration

This module provides centralized leverage constants and validation to ensure
consistent leverage limits across the entire trading system.

All leverage values are enforced to be between 5 and 100.
"""

from typing import Final
from dataclasses import dataclass


# Centralized leverage constants
MIN_LEVERAGE: Final[float] = 5.0
MAX_LEVERAGE: Final[float] = 100.0

# Leverage validation bounds
LEVERAGE_LOWER_BOUND: Final[float] = 5.0
LEVERAGE_UPPER_BOUND: Final[float] = 100.0


@dataclass(frozen=True)
class LeverageLimits:
    """Immutable leverage limits configuration."""
    min_leverage: float = MIN_LEVERAGE
    max_leverage: float = MAX_LEVERAGE
    
    def __post_init__(self):
        """Validate leverage limits on initialization."""
        if not (LEVERAGE_LOWER_BOUND <= self.min_leverage <= LEVERAGE_UPPER_BOUND):
            raise ValueError(f"min_leverage must be between {LEVERAGE_LOWER_BOUND} and {LEVERAGE_UPPER_BOUND}")
        if not (LEVERAGE_LOWER_BOUND <= self.max_leverage <= LEVERAGE_UPPER_BOUND):
            raise ValueError(f"max_leverage must be between {LEVERAGE_LOWER_BOUND} and {LEVERAGE_UPPER_BOUND}")
        if self.min_leverage >= self.max_leverage:
            raise ValueError("min_leverage must be less than max_leverage")


def validate_leverage(leverage: float) -> float:
    """
    Validate and clamp leverage value to be between 5 and 100.
    
    Args:
        leverage: The leverage value to validate
        
    Returns:
        float: Clamped leverage value between 5 and 100
        
    Raises:
        ValueError: If leverage is not a valid number
    """
    if not isinstance(leverage, (int, float)):
        raise ValueError("Leverage must be a number")
    
    if leverage < LEVERAGE_LOWER_BOUND:
        return LEVERAGE_LOWER_BOUND
    elif leverage > LEVERAGE_UPPER_BOUND:
        return LEVERAGE_UPPER_BOUND
    else:
        return float(leverage)


def get_leverage_limits() -> LeverageLimits:
    """
    Get the centralized leverage limits.
    
    Returns:
        LeverageLimits: The centralized leverage limits
    """
    return LeverageLimits()


def is_valid_leverage(leverage: float) -> bool:
    """
    Check if a leverage value is within the valid range.
    
    Args:
        leverage: The leverage value to check
        
    Returns:
        bool: True if leverage is between 5 and 100, False otherwise
    """
    try:
        return LEVERAGE_LOWER_BOUND <= float(leverage) <= LEVERAGE_UPPER_BOUND
    except (ValueError, TypeError):
        return False


# Default leverage limits instance
DEFAULT_LEVERAGE_LIMITS = LeverageLimits()