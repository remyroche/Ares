"""
Constants for Analyst Models Training

Centralized constants to avoid magic numbers throughout the codebase.
"""

# Memory thresholds (in MB)
MEMORY_HIGH_THRESHOLD_MB = 8000
MEMORY_OPTIMIZATION_THRESHOLD_MB = 8000

# System health thresholds
MEMORY_CRITICAL_PERCENT = 90
MEMORY_WARNING_PERCENT = 75
DISK_CRITICAL_PERCENT = 95
DISK_WARNING_PERCENT = 85
CPU_HEALTHY_THRESHOLD = 80

# Validation thresholds
NAN_CRITICAL_PERCENT = 50
NAN_WARNING_PERCENT = 20
INF_CRITICAL_PERCENT = 10
INF_WARNING_PERCENT = 5
TARGET_NAN_CRITICAL_PERCENT = 20
TARGET_INF_CRITICAL_PERCENT = 5

# Regime thresholds
REGIME_IMBALANCE_RATIO_THRESHOLD = 10
REGIME_ENTROPY_LOW_THRESHOLD = 0.5

# Data quality thresholds
MISSING_DATA_HIGH_THRESHOLD = 20
DUPLICATE_DATA_HIGH_THRESHOLD = 5
DATA_SIZE_MEMORY_RATIO = 0.5

# HPO configuration
HPO_TRIALS_MIN = 10
HPO_TRIALS_WARNING_MAX = 1000
HPO_TIMEOUT_MIN_SECONDS = 60
HPO_TIMEOUT_WARNING_MAX_SECONDS = 7200

# Sample size thresholds
MIN_SAMPLES_LOW_THRESHOLD = 100
MIN_SAMPLES_HIGH_THRESHOLD = 10000

# Disk space requirements (in GB)
REQUIRED_DISK_SPACE_GB = 5.0
MINIMUM_DISK_SPACE_GB = 1.0

# Model complexity scores
MODEL_COMPLEXITY_SCORES = {
    'TEMPORAL_FUSION_TRANSFORMER': 0.9,
    'TABNET': 0.8,
    'HIST_GRADIENT_BOOSTING': 0.7,
    'EXTRA_TREES': 0.6,
    'TCN': 0.8,
    'CatBoostRegressor': 0.7,
    'LGBMRegressor': 0.6,
    'RandomForestRegressor': 0.5,
    'XGBRegressor': 0.7,
    'NODE': 0.9,
    'DEEPSCALER': 0.8,
    'CATBOOST': 0.7,
    'XGBOOST': 0.7,
    'MULTISCALE_NBEATS': 0.9
}

# Valid timeframes
VALID_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

# Valid model types
VALID_MODEL_TYPES = [
    "TEMPORAL_FUSION_TRANSFORMER", "TABNET", "HIST_GRADIENT_BOOSTING",
    "EXTRA_TREES", "TCN", "CatBoostRegressor", "LGBMRegressor",
    "RandomForestRegressor", "XGBRegressor", "NODE", "DEEPSCALER",
    "CATBOOST", "XGBOOST", "MULTISCALE_NBEATS"
]

# Valid evaluation metrics
VALID_METRICS = [
    "mse", "mae", "r2", "mape", "smape",
    "accuracy", "precision", "recall", "f1"
]

# Performance metric weights
METRIC_WEIGHTS = {
    'r2': 0.4,
    'mse': 0.3,
    'mae': 0.2,
    'mape': 0.1
}

# Health score thresholds
HEALTH_SCORE_GRADE_A = 90
HEALTH_SCORE_GRADE_B = 70
HEALTH_SCORE_GRADE_C = 50

# Performance thresholds
R2_EXCELLENT_THRESHOLD = 0.8
R2_GOOD_THRESHOLD = 0.6

# Ensemble diversity thresholds
ENSEMBLE_DIVERSITY_LOW_THRESHOLD = 0.1

# Error count thresholds
ERROR_COUNT_HIGH_THRESHOLD = 10
ERROR_COUNT_WARNING_THRESHOLD = 5

# Report versioning
REPORT_VERSION = "enhanced_v2.0_with_utilities"
STEP_VERSION = "enhanced_v1.0"

# Confidence interval multiplier
CONFIDENCE_INTERVAL_Z_SCORE = 1.96

# Cache TTL (seconds)
HARDWARE_STATUS_CACHE_TTL = 60
SYSTEM_METRICS_CACHE_TTL = 30
