"""
Configuration Constants for PID-Based Feature Generation

This module centralizes all configuration constants and magic numbers
to make them easily configurable and maintainable.
"""

from dataclasses import dataclass
from typing import Dict, List
from src.utils.tprint import tprint

tprint("📋 Loading PID-based feature generation constants...")


@dataclass
class ValidationConstants:
    """Constants for data validation."""
    # Data quality thresholds
    MIN_DATA_QUALITY_SCORE: float = 0.7
    MAX_MISSING_DATA_RATIO: float = 0.1
    MIN_SAMPLES_REQUIRED: int = 100
    MAX_NAN_PERCENTAGE: float = 0.5
    
    # Feature validation
    MIN_FEATURES_REQUIRED: int = 2
    MIN_VARIANCE_THRESHOLD: float = 0.01
    MAX_CORRELATION_THRESHOLD: float = 0.95
    MIN_CORRELATION_THRESHOLD: float = 0.1


@dataclass
class ComputationConstants:
    """Constants for computational operations."""
    # Rolling window settings
    DEFAULT_ROLLING_WINDOW: int = 20
    MIN_ROLLING_WINDOW: int = 3
    MAX_ROLLING_WINDOW: int = 200
    
    # Timeout settings (seconds)
    FEATURE_GENERATION_TIMEOUT: int = 300  # 5 minutes
    COMPUTATION_TIMEOUT: int = 60         # 1 minute
    
    # Memory management
    DEFAULT_BATCH_SIZE: int = 500
    MAX_MEMORY_PERCENT: float = 0.7
    CHUNK_SIZE_MB: int = 256


@dataclass
class FeatureGenerationConstants:
    """Constants for feature generation."""
    # Feature limits
    MAX_INTERACTION_FEATURES: int = 100
    MAX_POLYNOMIAL_FEATURES: int = 50
    MAX_CROSS_TIMEFRAME_FEATURES: int = 50
    
    # PID thresholds
    DEFAULT_SYNERGY_THRESHOLD: float = 0.1
    DEFAULT_REDUNDANCY_THRESHOLD: float = 0.15
    DEFAULT_UNIQUE_INFO_THRESHOLD: float = 0.05
    
    # Feature selection thresholds
    MIN_SYNERGY_SCORE: float = 0.05
    MIN_UNIQUE_INFO_SCORE: float = 0.02
    MAX_REDUNDANCY_SCORE: float = 0.8
    
    # Quality thresholds
    MIN_FEATURE_QUALITY_SCORE: float = 0.3
    MIN_STABILITY_SCORE: float = 0.5
    HIGH_CORRELATION_THRESHOLD: float = 0.8


@dataclass
class LookbackConstants:
    """Constants for lookback period optimization."""
    # Default lookback periods for different feature types
    DEFAULT_LOOKBACK_PERIODS: Dict[str, int] = None
    
    # Validation ranges
    MIN_LOOKBACK_PERIOD: int = 1
    MAX_LOOKBACK_PERIOD: int = 200
    OPTIMAL_LOOKBACK_RANGE: tuple = (5, 50)
    
    # Quality scoring
    OPTIMAL_QUALITY_SCORE: float = 1.0
    ACCEPTABLE_QUALITY_SCORE: float = 0.7
    POOR_QUALITY_SCORE: float = 0.3
    
    def __post_init__(self):
        """Initialize default lookback periods."""
        if self.DEFAULT_LOOKBACK_PERIODS is None:
            self.DEFAULT_LOOKBACK_PERIODS = {
                # Technical indicators
                'rsi': 14,
                'sma': 20,
                'ema': 12,
                'macd': 26,
                'bollinger': 20,
                'stochastic': 14,
                'williams_r': 14,
                'cci': 20,
                'atr': 14,
                'adx': 14,
                
                # Price features
                'price_momentum': 10,
                'price_acceleration': 5,
                'price_volatility': 20,
                'price_trend': 15,
                
                # Volume features
                'volume_momentum': 10,
                'volume_profile': 20,
                'volume_ratio': 14,
                'volume_trend': 15,
                
                # Volatility features
                'volatility_regime': 20,
                'volatility_momentum': 10,
                'volatility_trend': 15,
                
                # Cross-timeframe features
                'cross_timeframe_momentum': 10,
                'cross_timeframe_volatility': 20,
                'cross_timeframe_trend': 15
            }


@dataclass
class PolynomialConstants:
    """Constants for polynomial feature generation."""
    # Polynomial degree limits
    MAX_POLYNOMIAL_DEGREE: int = 3
    MIN_POLYNOMIAL_DEGREE: int = 2
    
    # Validation thresholds
    MAX_SKEWNESS_THRESHOLD: float = 5.0
    SIGNIFICANCE_THRESHOLD: float = 0.05
    
    # Safe operation bounds
    LOG_MIN_VALUE: float = 1e-10
    POWER_MAX_VALUE: float = 10
    POWER_MIN_VALUE: float = -10


@dataclass
class CrossTimeframeConstants:
    """Constants for cross-timeframe feature generation."""
    # Timeframe definitions
    STANDARD_TIMEFRAMES: List[str] = None
    
    # Lag settings
    MAX_LAG_PERIODS: int = 5
    MIN_LAG_PERIODS: int = 1
    
    # Correlation settings
    MIN_CORRELATION_WINDOW: int = 3
    DEFAULT_CORRELATION_WINDOW: int = 20
    
    def __post_init__(self):
        """Initialize standard timeframes."""
        if self.STANDARD_TIMEFRAMES is None:
            self.STANDARD_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']


@dataclass
class ErrorHandlingConstants:
    """Constants for error handling and logging."""
    # Retry settings
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_DELAY_SECONDS: float = 1.0
    
    # Logging levels
    ERROR_LOG_LEVEL: str = "ERROR"
    WARNING_LOG_LEVEL: str = "WARNING"
    INFO_LOG_LEVEL: str = "INFO"
    DEBUG_LOG_LEVEL: str = "DEBUG"
    
    # Error categorization
    CRITICAL_ERROR_KEYWORDS: List[str] = None
    
    def __post_init__(self):
        """Initialize critical error keywords."""
        if self.CRITICAL_ERROR_KEYWORDS is None:
            self.CRITICAL_ERROR_KEYWORDS = [
                'memory', 'allocation', 'segmentation', 'core', 'fatal',
                'critical', 'system', 'kernel', 'hardware'
            ]


# Global constants instances
VALIDATION = ValidationConstants()
COMPUTATION = ComputationConstants()
FEATURE_GEN = FeatureGenerationConstants()
LOOKBACK = LookbackConstants()
POLYNOMIAL = PolynomialConstants()
CROSS_TIMEFRAME = CrossTimeframeConstants()
ERROR_HANDLING = ErrorHandlingConstants()


def get_rolling_window_size(data_length: int, preferred_window: int = None) -> int:
    """
    Get appropriate rolling window size based on data length.
    
    Args:
        data_length: Length of the data
        preferred_window: Preferred window size (optional)
        
    Returns:
        Appropriate window size
    """
    if preferred_window is not None:
        return max(
            COMPUTATION.MIN_ROLLING_WINDOW,
            min(preferred_window, data_length // 4, COMPUTATION.MAX_ROLLING_WINDOW)
        )
    
    # Auto-determine based on data length
    if data_length < 20:
        return max(3, data_length // 2)
    elif data_length < 100:
        return min(20, data_length // 4)
    else:
        return COMPUTATION.DEFAULT_ROLLING_WINDOW


def validate_lookback_period(period: int, feature_type: str = None) -> bool:
    """
    Validate if a lookback period is reasonable.
    
    Args:
        period: Lookback period to validate
        feature_type: Type of feature (optional)
        
    Returns:
        True if period is valid
    """
    if not isinstance(period, (int, float)):
        return False
    
    period = int(period)
    
    if period < LOOKBACK.MIN_LOOKBACK_PERIOD or period > LOOKBACK.MAX_LOOKBACK_PERIOD:
        return False
    
    # Feature-specific validation could be added here
    return True


def get_quality_score_from_period(period: int) -> float:
    """
    Get quality score based on lookback period.
    
    Args:
        period: Lookback period
        
    Returns:
        Quality score between 0 and 1
    """
    if not validate_lookback_period(period):
        return 0.0
    
    min_range, max_range = LOOKBACK.OPTIMAL_LOOKBACK_RANGE
    
    if min_range <= period <= max_range:
        return LOOKBACK.OPTIMAL_QUALITY_SCORE
    elif period < min_range or period > max_range * 2:
        return LOOKBACK.POOR_QUALITY_SCORE
    else:
        return LOOKBACK.ACCEPTABLE_QUALITY_SCORE