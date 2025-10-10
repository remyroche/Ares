"""
Consolidated Constants for Feature Lookback Optimization.

This module centralizes all magic numbers and configuration constants.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
from enum import Enum


class OptimizationMethod(Enum):
    """Available optimization methods."""
    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"
    MRMR = "mrmr"
    RANDOM_SEARCH = "random_search"
    MULTI_TARGET = "multi_target"
    COARSE_TO_REFINE = "coarse_to_refine"


class MemoryPressureLevel(Enum):
    """Memory pressure levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ScoringConstants:
    """Constants for scoring calculations."""
    # Scale normalization
    VARIANCE_PENALTY_CAP: float = 0.3
    STABILITY_PENALTY_FACTOR: float = 0.1
    MAX_PENALTY_RATIO: float = 0.5
    
    # MI calculation
    MIN_MI_THRESHOLD: float = 0.0
    CORRELATION_TO_MI_FACTOR: float = 0.5
    MIN_SAMPLES_FOR_MI: int = 2
    MIN_SAMPLES_FOR_BOOTSTRAP: int = 20
    
    # Bootstrap validation
    DEFAULT_BOOTSTRAP_SAMPLES: int = 10
    BOOTSTRAP_RANDOM_SEED: int = 42
    
    # Stability calculation
    MIN_SAMPLES_FOR_STABILITY: int = 5
    STABILITY_CV_THRESHOLD: float = 1.0


@dataclass
class CacheConstants:
    """Constants for cache management."""
    # Cache sizes
    DEFAULT_FEATURE_CACHE_SIZE: int = 1000
    DEFAULT_MEMORY_CACHE_SIZE: int = 1000
    MAX_CACHE_HISTORY_SIZE: int = 1000
    
    # Memory limits
    DEFAULT_MEMORY_LIMIT_MB: float = 1024.0
    MEMORY_WARNING_THRESHOLD: float = 0.7
    MEMORY_CRITICAL_THRESHOLD: float = 0.9
    
    # Cleanup intervals
    CLEANUP_INTERVAL_SECONDS: int = 300  # 5 minutes
    MAX_CONSECUTIVE_HIGH_PRESSURE: int = 3


@dataclass
class OptimizationConstants:
    """Constants for optimization algorithms."""
    # Lookback ranges
    DEFAULT_MIN_LOOKBACK: int = 5
    DEFAULT_MAX_LOOKBACK: int = 300
    DEFAULT_LOOKBACK_STEP: int = 5
    PREFERRED_LOOKBACK: int = 50
    
    # Penalty settings
    DEFAULT_PENALTY_STRENGTH: float = 0.1
    DEFAULT_PENALTY_EXPONENT: float = 2.0
    PREFERRED_MIN_LOOKBACK: float = 40.0
    PREFERRED_MAX_LOOKBACK: float = 80.0
    
    # Optimization thresholds
    MIN_STABILITY_SCORE: float = 0.7
    CONVERGENCE_THRESHOLD: float = 0.01
    EARLY_STOPPING_PATIENCE: int = 5
    
    # Data requirements
    MIN_TRAIN_RATIO: float = 0.4
    MIN_VAL_SAMPLES: int = 20
    MIN_TRAIN_SAMPLES: int = 60
    MIN_WINDOW_SIZE: int = 25


@dataclass
class PerformanceConstants:
    """Constants for performance optimization."""
    # Vectorization
    VECTORIZATION_BATCH_SIZE: int = 1000
    GPU_ACCELERATION_THRESHOLD: int = 10
    
    # Parallel processing
    DEFAULT_MAX_WORKERS: int = 4
    PARALLEL_CHUNK_SIZE: int = 1000
    
    # Memory management
    TILE_SIZE_MB: int = 64
    L3_CACHE_SIZE_MB: int = 32
    
    # Performance monitoring
    MAX_PERFORMANCE_ENTRIES: int = 1000
    PERFORMANCE_CLEANUP_INTERVAL: int = 100


@dataclass
class ValidationConstants:
    """Constants for data validation."""
    # Data quality
    MIN_DATA_POINTS: int = 100
    MAX_MISSING_RATIO: float = 0.1
    MIN_CORRELATION_THRESHOLD: float = 0.01
    
    # Feature validation
    MIN_FEATURE_VALUES: int = 10
    MAX_FEATURE_VALUES: int = 1000000
    MIN_FEATURE_STD: float = 1e-10
    
    # Target validation
    MIN_TARGET_VALUES: int = 10
    MAX_TARGET_VALUES: int = 1000000
    MIN_TARGET_STD: float = 1e-10


@dataclass
class ErrorHandlingConstants:
    """Constants for error handling."""
    # Retry settings
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: float = 1.0
    
    # Error tracking
    MAX_ERROR_HISTORY: int = 1000
    ERROR_CLEANUP_INTERVAL: int = 100
    
    # Fallback values
    DEFAULT_MI_SCORE: float = 0.0
    DEFAULT_CORRELATION: float = 0.0
    DEFAULT_STABILITY_SCORE: float = 0.0


@dataclass
class FeatureGenerationConstants:
    """Constants for feature generation."""
    # Rolling operations
    DEFAULT_ROLLING_MIN_PERIODS: int = 1
    DEFAULT_QUANTILE_LEVELS: List[float] = None
    
    # Volume profile
    VOLUME_PROFILE_QUANTILE_HIGH: float = 0.7
    VOLUME_PROFILE_QUANTILE_LOW: float = 0.3
    PRICE_ROUNDING_DECIMALS: int = 2
    
    # Trend calculation
    TREND_CALCULATION_WINDOW: int = 5
    BOUNCE_DETECTION_THRESHOLD: float = 0.5
    
    def __post_init__(self):
        if self.DEFAULT_QUANTILE_LEVELS is None:
            self.DEFAULT_QUANTILE_LEVELS = [0.25, 0.5, 0.75]


@dataclass
class LoggingConstants:
    """Constants for logging and monitoring."""
    # Log levels
    DEFAULT_LOG_LEVEL: str = "INFO"
    DEBUG_LOG_THRESHOLD: int = 1000
    
    # Performance logging
    PERFORMANCE_LOG_INTERVAL: int = 100
    MEMORY_LOG_INTERVAL: int = 50
    
    # Error logging
    ERROR_LOG_INTERVAL: int = 10
    WARNING_LOG_INTERVAL: int = 50


class AllConstants:
    """Container for all constants."""
    
    def __init__(self):
        self.scoring = ScoringConstants()
        self.cache = CacheConstants()
        self.optimization = OptimizationConstants()
        self.performance = PerformanceConstants()
        self.validation = ValidationConstants()
        self.error_handling = ErrorHandlingConstants()
        self.feature_generation = FeatureGenerationConstants()
        self.logging = LoggingConstants()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all constants to dictionary."""
        return {
            'scoring': self.scoring.__dict__,
            'cache': self.cache.__dict__,
            'optimization': self.optimization.__dict__,
            'performance': self.performance.__dict__,
            'validation': self.validation.__dict__,
            'error_handling': self.error_handling.__dict__,
            'feature_generation': self.feature_generation.__dict__,
            'logging': self.logging.__dict__
        }
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update constants from dictionary."""
        for category, values in config_dict.items():
            if hasattr(self, category):
                category_obj = getattr(self, category)
                for key, value in values.items():
                    if hasattr(category_obj, key):
                        setattr(category_obj, key, value)


# Global constants instance
_global_constants: 'AllConstants' = None


def get_constants() -> AllConstants:
    """Get the global constants instance."""
    global _global_constants
    if _global_constants is None:
        _global_constants = AllConstants()
    return _global_constants


def update_constants(config_dict: Dict[str, Any]):
    """Update global constants from configuration dictionary."""
    constants = get_constants()
    constants.update_from_dict(config_dict)


# Convenience functions for common constants
def get_scoring_constants() -> ScoringConstants:
    """Get scoring constants."""
    return get_constants().scoring


def get_cache_constants() -> CacheConstants:
    """Get cache constants."""
    return get_constants().cache


def get_optimization_constants() -> OptimizationConstants:
    """Get optimization constants."""
    return get_constants().optimization


def get_performance_constants() -> PerformanceConstants:
    """Get performance constants."""
    return get_constants().performance


def get_validation_constants() -> ValidationConstants:
    """Get validation constants."""
    return get_constants().validation


def get_error_handling_constants() -> ErrorHandlingConstants:
    """Get error handling constants."""
    return get_constants().error_handling


def get_feature_generation_constants() -> FeatureGenerationConstants:
    """Get feature generation constants."""
    return get_constants().feature_generation


def get_logging_constants() -> LoggingConstants:
    """Get logging constants."""
    return get_constants().logging