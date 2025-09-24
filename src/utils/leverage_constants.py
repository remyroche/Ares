"""
Leverage Constants

Centralized definition of leverage limits and validation.
All leverage-related constants should be imported from this module.
"""

# Minimum leverage allowed (5x)
MIN_LEVERAGE = 5.0

# Maximum leverage allowed (100x)
MAX_LEVERAGE = 100.0

# Default leverage for conservative trading
DEFAULT_LEVERAGE = 10.0

# Leverage risk thresholds for different leverage levels
LEVERAGE_RISK_THRESHOLDS = {
    5: 0.20,   # 5x leverage: can handle 20% adverse movement
    10: 0.10,  # 10x leverage: can handle 10% adverse movement
    15: 0.08,  # 15x leverage: can handle 8% adverse movement
    20: 0.07,  # 20x leverage: can handle 7% adverse movement
    25: 0.06,  # 25x leverage: can handle 6% adverse movement
    30: 0.05,  # 30x leverage: can handle 5% adverse movement
    40: 0.04,  # 40x leverage: can handle 4% adverse movement
    50: 0.035, # 50x leverage: can handle 3.5% adverse movement
    60: 0.03,  # 60x leverage: can handle 3% adverse movement
    75: 0.025, # 75x leverage: can handle 2.5% adverse movement
    100: 0.02, # 100x leverage: can handle 2% adverse movement
}


def validate_leverage(leverage: float) -> bool:
    """
    Validate that leverage is within allowed bounds.

    Args:
        leverage: Leverage value to validate

    Returns:
        bool: True if leverage is valid, False otherwise
    """
    return MIN_LEVERAGE <= leverage <= MAX_LEVERAGE


def validate_leverage_range(min_leverage: float, max_leverage: float) -> bool:
    """
    Validate that a leverage range is within allowed bounds.

    Args:
        min_leverage: Minimum leverage value
        max_leverage: Maximum leverage value

    Returns:
        bool: True if range is valid, False otherwise
    """
    return (MIN_LEVERAGE <= min_leverage <= max_leverage <= MAX_LEVERAGE and
            min_leverage <= max_leverage)


def ensure_valid_leverage(leverage: float, context: str = "leverage") -> float:
    """
    Ensure leverage is within valid bounds, logging warnings if not.

    Args:
        leverage: Leverage value to validate and clamp
        context: Context for logging messages

    Returns:
        float: Valid leverage value (clamped if necessary)
    """
    import logging

    if leverage < MIN_LEVERAGE:
        logging.warning(f"{context} {leverage} below minimum {MIN_LEVERAGE}, clamping to minimum")
        return MIN_LEVERAGE
    elif leverage > MAX_LEVERAGE:
        logging.warning(f"{context} {leverage} above maximum {MAX_LEVERAGE}, clamping to maximum")
        return MAX_LEVERAGE
    else:
        return leverage


def ensure_valid_leverage_range(min_leverage: float, max_leverage: float,
                               context: str = "leverage range") -> tuple[float, float]:
    """
    Ensure a leverage range is within valid bounds.

    Args:
        min_leverage: Minimum leverage value
        max_leverage: Maximum leverage value
        context: Context for logging messages

    Returns:
        tuple: (min_leverage, max_leverage) clamped to valid range
    """
    import logging

    original_min = min_leverage
    original_max = max_leverage

    # Clamp min leverage
    if min_leverage < MIN_LEVERAGE:
        logging.warning(f"{context} min {min_leverage} below minimum {MIN_LEVERAGE}, clamping to minimum")
        min_leverage = MIN_LEVERAGE

    # Clamp max leverage
    if max_leverage > MAX_LEVERAGE:
        logging.warning(f"{context} max {max_leverage} above maximum {MAX_LEVERAGE}, clamping to maximum")
        max_leverage = MAX_LEVERAGE

    # Ensure min <= max
    if min_leverage > max_leverage:
        logging.warning(f"{context} min {min_leverage} > max {max_leverage}, swapping values")
        min_leverage, max_leverage = max_leverage, min_leverage

    # Log if values were changed
    if original_min != min_leverage or original_max != max_leverage:
        logging.info(f"{context} adjusted to ({min_leverage}, {max_leverage})")

    return min_leverage, max_leverage


def clamp_leverage(leverage: float) -> float:
    """
    Clamp leverage to allowed bounds.

    Args:
        leverage: Leverage value to clamp

    Returns:
        float: Clamped leverage value
    """
    return max(MIN_LEVERAGE, min(MAX_LEVERAGE, leverage))


def get_leverage_risk_threshold(leverage: float) -> float:
    """
    Get the risk threshold for a given leverage level.

    Args:
        leverage: Leverage level

    Returns:
        float: Risk threshold for the leverage level
    """
    # Find the highest leverage level that is <= the requested leverage
    applicable_levels = [level for level in LEVERAGE_RISK_THRESHOLDS.keys() if level <= leverage]
    if not applicable_levels:
        return LEVERAGE_RISK_THRESHOLDS[MIN_LEVERAGE]

    highest_level = max(applicable_levels)
    return LEVERAGE_RISK_THRESHOLDS[highest_level]


def get_safe_leverage_multiplier(leverage: float) -> float:
    """
    Get a safe leverage multiplier based on leverage level.
    Lower leverage gets higher multipliers, higher leverage gets lower multipliers.

    Args:
        leverage: Leverage level

    Returns:
        float: Safe leverage multiplier (0.0 to 1.0)
    """
    if leverage <= MIN_LEVERAGE:
        return 1.0
    elif leverage >= MAX_LEVERAGE:
        return 0.1
    else:
        # Linear interpolation between 1.0 at min leverage and 0.1 at max leverage
        ratio = (leverage - MIN_LEVERAGE) / (MAX_LEVERAGE - MIN_LEVERAGE)
        return 1.0 - (0.9 * ratio)  # 1.0 -> 0.1 as leverage increases