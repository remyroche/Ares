"""
Centralized constants for the Ares trading system.
This file defines key configuration values used across all components.
"""

# Data configuration constants
DEFAULT_LOOKBACK_DAYS = 180  # Exactly 6 months for consistent data range
FULL_TRAINING_LOOKBACK_DAYS = 730  # 2 years for full training (updated from 3 years)
SHORT_BLANK_LOOKBACK_DAYS = 30  # 30 days for short blank training
BLANK_TRAINING_LOOKBACK_DAYS = 180  # 180 days for blank training
DEFAULT_EXCLUDE_RECENT_DAYS = 0  # Don't exclude recent days by default
DEFAULT_MIN_DATA_POINTS = 10000  # Minimum data points required for training

# Timeframe configuration
DEFAULT_TIMEFRAMES = ["1m", "5m", "15m", "30m"]  # Standard timeframes for analysis

# Model training constants
DEFAULT_TRAINING_SPLIT = 0.8
DEFAULT_VALIDATION_SPLIT = 0.1
DEFAULT_TEST_SPLIT = 0.1

# HMM configuration constants
HMM_DEFAULT_N_STATES = 4  # Default number of HMM states
HMM_DEFAULT_N_MIX = 2  # Default number of mixture components
HMM_DEFAULT_COVARIANCE_TYPE = "diag"  # Default covariance type
HMM_DEFAULT_N_ITER = 300  # Default number of iterations
HMM_DEFAULT_TOL = 1e-4  # Default tolerance

# Validation constants
MIN_SAMPLES_PER_SPLIT = 100  # Minimum samples required per regime split
MIN_STATE_RATIO = 0.05  # Minimum ratio for each HMM state
MAX_DOMINANT_STATE_RATIO = 0.8  # Maximum ratio for dominant state

# Cache configuration
DEFAULT_CACHE_TTL_HOURS = 24  # Default cache time-to-live in hours
DEFAULT_MAX_CACHE_SIZE_GB = 5.0  # Default maximum cache size in GB

# Performance thresholds
MEMORY_WARNING_THRESHOLD_PERCENT = 80  # Memory usage warning threshold
CPU_WARNING_THRESHOLD_PERCENT = 90  # CPU usage warning threshold

# Error handling constants
MAX_RETRY_ATTEMPTS = 3  # Maximum number of retry attempts
RETRY_DELAY_SECONDS = 5  # Delay between retry attempts

# Logging constants
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# File paths and directories
DEFAULT_DATA_DIR = "data/training"
DEFAULT_CACHE_DIR = "data_cache"
DEFAULT_MODELS_DIR = "models"
DEFAULT_REPORTS_DIR = "reports"
DEFAULT_LOGS_DIR = "logs"

# Exchange and symbol defaults
DEFAULT_EXCHANGE = "BINANCE"
DEFAULT_SYMBOL = "ETHUSDT"
DEFAULT_INTERVAL = "1m"
