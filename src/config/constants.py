"""
Constants for the Ares trading bot.

This module centralizes important constants used throughout the project
to improve maintainability and avoid inconsistencies.
"""

# Training and Data Collection Constants
FULL_TRAINING_LOOKBACK_DAYS = 1095  # 3 years for full training
DEFAULT_LOOKBACK_DAYS = 1095  # Default lookback period for all timeframes (3 years)
LEGACY_LOOKBACK_DAYS = 730  # Legacy 2-year lookback (deprecated)

# Time Constants
SECONDS_PER_DAY = 86400
MILLISECONDS_PER_DAY = SECONDS_PER_DAY * 1000

# Data Quality Constants
MIN_DATA_POINTS = "10000"
DEFAULT_EXCLUDE_RECENT_DAYS = 2

# File and Path Constants
DEFAULT_DATA_DIR = "data/training"
DEFAULT_SYMBOL = "ETHUSDT"
DEFAULT_EXCHANGE = "BINANCE"
