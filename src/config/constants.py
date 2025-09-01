"""Constants for the Ares trading system.

This module contains all the constants used throughout the system
to ensure consistency and maintainability.
"""

from typing import Final

# Time Constants
DEFAULT_SIGNAL_VALIDITY_DURATION: Final[int] = 120
DEFAULT_SIGNAL_CHECK_INTERVAL: Final[int] = 10
DEFAULT_MAX_SIGNAL_AGE: Final[float] = 300.0  # 5 minutes
DEFAULT_COMPUTATION_TIME_THRESHOLD: Final[float] = 0.1  # 100ms

# Data Configuration Constants
# Centralized lookback windows to ensure a single source of truth across the project
# Note: Keep this consistent with usages in config and training steps
DEFAULT_LOOKBACK_DAYS: Final[int] = 730  # 2 years default lookback window

# Import training mode constants from centralized configuration
# TODO: Add proper imports when needed

# Feature Engineering Constants
DEFAULT_IMPORTANCE_THRESHOLD: Final[float] = 0.15
DEFAULT_VOLATILITY_THRESHOLD: Final[float] = 0.018
DEFAULT_MIN_FREQUENCY: Final[float] = 0.01
DEFAULT_MAX_FREQUENCY: Final[float] = 0.20

# Test Data Constants
DEFAULT_TEST_DATA_SIZE: Final[int] = 1000
DEFAULT_TEST_BASE_PRICE: Final[float] = 50000.0
DEFAULT_TEST_VOLUME_RANGE: Final[tuple[int, int]] = (1000, 10000)
DEFAULT_TEST_PRICE_VOLATILITY: Final[float] = 0.1
DEFAULT_TEST_VOLUME_VOLATILITY: Final[float] = 0.15

# Exchange Constants
DEFAULT_RATE_LIMIT: Final[int] = 1200
DEFAULT_TIMEOUT: Final[int] = 30
DEFAULT_LIMIT: Final[int] = 1000

# Performance Constants
DEFAULT_PERFORMANCE_WINDOW: Final[int] = 20
DEFAULT_VOLATILITY_CALCULATION_WINDOW: Final[int] = 20
DEFAULT_VOLATILITY_TREND_WINDOW: Final[int] = 40

# Validation Constants
DEFAULT_TOLERANCE_PERCENTAGE: Final[float] = 0.15
DEFAULT_WEIGHT_TOLERANCE: Final[float] = 0.01

# Logging Constants
DEFAULT_LOG_LEVEL: Final[str] = "INFO"
DEFAULT_LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Feature Categories and Weights
DEFAULT_FEATURE_WEIGHTS: Final[dict[str, float]] = {
"volatility": 0.15,
"liquidity": 0.15,
"microstructure": 0.10,
"regime": 0.10,
"sr_features": 0.15,
"interaction": 0.15,
"volume": 0.10,
}

# Note: Regime adjustments removed - per-regime distinct LM models handle regime-specific adjustments

# Error Recovery Constants
DEFAULT_RETRY_ATTEMPTS: Final[int] = 3
DEFAULT_RETRY_DELAY: Final[float] = 1.0
DEFAULT_CIRCUIT_BREAKER_THRESHOLD: Final[int] = 5
DEFAULT_CIRCUIT_BREAKER_TIMEOUT: Final[float] = 60.0

# Memory Management Constants
DEFAULT_MEMORY_LIMIT_MB: Final[int] = 1024
DEFAULT_CACHE_SIZE: Final[int] = 1000
DEFAULT_CLEANUP_INTERVAL: Final[int] = 3600

# API Constants
DEFAULT_API_TIMEOUT: Final[float] = 30.0
DEFAULT_API_RETRY_DELAY: Final[float] = 1.0
DEFAULT_API_MAX_RETRIES: Final[int] = 3

# Monitoring Constants
DEFAULT_METRICS_INTERVAL: Final[int] = 60
DEFAULT_HEALTH_CHECK_INTERVAL: Final[int] = 30
DEFAULT_ALERT_THRESHOLD_MS: Final[float] = 3000.0

# File System Constants
DEFAULT_CACHE_DIR: Final[str] = "cache"
DEFAULT_LOG_DIR: Final[str] = "logs"
DEFAULT_DATA_DIR: Final[str] = "data"
DEFAULT_BACKUP_DIR: Final[str] = "backups"

# Network Constants
DEFAULT_CONNECTION_TIMEOUT: Final[float] = 30.0
DEFAULT_READ_TIMEOUT: Final[float] = 60.0
DEFAULT_WRITE_TIMEOUT: Final[float] = 60.0

# Security Constants
DEFAULT_ENCRYPTION_KEY_SIZE: Final[int] = 32

# Paper Trading Constants
DEFAULT_INITIAL_BALANCE: Final[float] = 10000.0
DEFAULT_MAX_POSITION_SIZE: Final[float] = 0.1
DEFAULT_COMMISSION_RATE: Final[float] = 0.001
DEFAULT_SLIPPAGE_RATE: Final[float] = 0.0005
DEFAULT_HASH_ALGORITHM: Final[str] = "sha256"
DEFAULT_TOKEN_EXPIRY_HOURS: Final[int] = 24

# Development Constants
DEFAULT_DEBUG_MODE: Final[bool] = False
DEFAULT_VERBOSE_LOGGING: Final[bool] = False
DEFAULT_ENABLE_PROFILING: Final[bool] = False
