"""
Configuration Constants for Feature Lookback Optimization.

This module centralizes all magic numbers and configuration values
to improve maintainability and reduce hardcoded values.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from .utils.tprint_utils import tprint

tprint("📋 Loading feature lookback optimization constants...")


@dataclass
class OptimizationConstants:
    """Constants for optimization algorithms."""
    
    # Lookback period constraints
    DEFAULT_MIN_LOOKBACK: int = 5
    DEFAULT_MAX_LOOKBACK: int = 298
    DEFAULT_LOOKBACK_STEP: int = 1
    
    # Grid search parameters
    DEFAULT_COARSE_GRID_SIZE: int = 5
    DEFAULT_FINE_GRID_SIZE: int = 5
    DEFAULT_TOP_K_COARSE: int = 6
    DEFAULT_TOP_K_FINE: int = 4
    
    # TPE parameters
    DEFAULT_TPE_TRIALS: int = 25
    DEFAULT_N_STARTUP_TRIALS: int = 10
    DEFAULT_N_WARMUP_STEPS: int = 5
    DEFAULT_INTERVAL_STEPS: int = 1
    
    # Refinement factors
    DEFAULT_COARSE_REFINEMENT_FACTOR: float = 0.3
    DEFAULT_FINE_REFINEMENT_FACTOR: float = 0.2
    
    # Correlation and scoring thresholds (removed unused constants)
    
    # Multi-objective weights
    DEFAULT_FIRST_LOOKBACK_WEIGHT: float = 0.4
    DEFAULT_SECOND_LOOKBACK_WEIGHT: float = 0.4
    DEFAULT_CORRELATION_WEIGHT: float = 0.2


@dataclass
class PerformanceConstants:
    """Constants for performance monitoring and optimization."""
    
    # Memory management
    DEFAULT_MEMORY_LIMIT_GB: float = 8.0
    DEFAULT_MAX_METRICS_MEMORY: int = 10000
    DEFAULT_CLEANUP_INTERVAL: int = 1000
    
    # Performance thresholds
    MEMORY_WARNING_THRESHOLD_MB: float = 1000.0
    CPU_WARNING_THRESHOLD_PERCENT: float = 80.0
    EXECUTION_TIME_WARNING_SECONDS: float = 300.0
    ERROR_RATE_WARNING_PERCENT: float = 10.0
    
    # Cleanup limits
    MAX_PERFORMANCE_METRICS: int = 1000
    MAX_EXECUTION_TIMES_PER_OPERATION: int = 100
    METRICS_RETENTION_FACTOR: float = 0.8


@dataclass
class ValidationConstants:
    """Constants for data validation."""
    
    # Data quality thresholds
    MIN_DATA_COMPLETENESS: float = 0.8
    MIN_DATA_QUALITY_SCORE: float = 0.7
    MAX_NULL_RATIO: float = 0.2
    
    # Data freshness
    MAX_DATA_AGE_DAYS: int = 30
    
    # Required columns
    REQUIRED_OHLCV_COLUMNS: List[str] = None
    
    def __post_init__(self):
        if self.REQUIRED_OHLCV_COLUMNS is None:
            self.REQUIRED_OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']


@dataclass
class QualityConstants:
    """Constants for quality assessment."""
    
    # Quality score thresholds
    EXCELLENT_QUALITY_THRESHOLD: float = 0.9
    GOOD_QUALITY_THRESHOLD: float = 0.7
    FAIR_QUALITY_THRESHOLD: float = 0.5
    
    # Performance rating thresholds
    EXCELLENT_PERFORMANCE_THRESHOLD: float = 0.8
    GOOD_PERFORMANCE_THRESHOLD: float = 0.6
    FAIR_PERFORMANCE_THRESHOLD: float = 0.4
    
    # Risk level thresholds
    HIGH_RISK_THRESHOLD: float = 0.8
    MEDIUM_RISK_THRESHOLD: float = 0.5
    
    # Validation score thresholds
    MIN_VALIDATION_SCORE: float = 0.7
    MIN_STABILITY_SCORE: float = 0.6
    MIN_REGIME_COVERAGE: float = 0.8


@dataclass
class FileConstants:
    """Constants for file operations."""
    
    # File extensions
    JSON_EXTENSION: str = ".json"
    PARQUET_EXTENSION: str = ".parquet"
    CSV_EXTENSION: str = ".csv"
    
    # Directory names
    REPORTS_DIR: str = "outcomes/market_analysis"
    METRICS_DIR: str = "metrics"
    VISUALIZATIONS_DIR: str = "visualizations"
    INSIGHTS_DIR: str = "insights"
    ARTIFACTS_DIR: str = "generated/market_analysis"
    
    # File naming patterns
    TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M%S"
    REPORT_PREFIX: str = "feature_lookback_optimization"
    
    # I/O timeouts
    FILE_OPERATION_TIMEOUT_SECONDS: float = 30.0


@dataclass
class AlgorithmConstants:
    """Constants for specific algorithms."""
    
    # MRMR parameters
    MRMR_N_NEIGHBORS: int = 3
    MRMR_RELEVANCE_METHOD: str = 'mutual_info'
    MRMR_REDUNDANCY_METHOD: str = 'correlation'
    
    # Elastic Net parameters
    ELASTIC_NET_N_BOOTSTRAPS: int = 20
    ELASTIC_NET_BOOTSTRAP_FRACTION: float = 0.8
    ELASTIC_NET_STABILITY_THRESHOLD: float = 0.6
    ELASTIC_NET_ALPHA_RANGE: Tuple[float, float] = (0.001, 1.0)
    ELASTIC_NET_L1_RATIO_RANGE: Tuple[float, float] = (0.1, 0.9)
    ELASTIC_NET_CV_FOLDS: int = 5
    
    # PID parameters
    PID_METHOD: str = 'bivariate'
    PID_MEASURES: List[str] = None
    PID_DISCRETIZATION_METHOD: str = 'adaptive'
    PID_N_BINS: int = 10
    
    # Correlation methods
    DEFAULT_CORRELATION_METHOD: str = 'pearson'
    ALTERNATIVE_CORRELATION_METHODS: List[str] = None
    
    def __post_init__(self):
        if self.PID_MEASURES is None:
            self.PID_MEASURES = ['i_min', 'i_ccs']
        if self.ALTERNATIVE_CORRELATION_METHODS is None:
            self.ALTERNATIVE_CORRELATION_METHODS = ['spearman', 'kendall']


# Global instances for easy access
tprint("🔧 Creating global constant instances...")
OPTIMIZATION_CONSTANTS = OptimizationConstants()
PERFORMANCE_CONSTANTS = PerformanceConstants()
VALIDATION_CONSTANTS = ValidationConstants()
QUALITY_CONSTANTS = QualityConstants()
FILE_CONSTANTS = FileConstants()
ALGORITHM_CONSTANTS = AlgorithmConstants()
tprint("✅ All constant instances created successfully")


def get_all_constants() -> Dict[str, any]:
    """Get all constants as a dictionary for easy access."""
    tprint("📋 Getting all constants dictionary...")
    constants_dict = {
        'optimization': OPTIMIZATION_CONSTANTS,
        'performance': PERFORMANCE_CONSTANTS,
        'validation': VALIDATION_CONSTANTS,
        'quality': QUALITY_CONSTANTS,
        'file': FILE_CONSTANTS,
        'algorithm': ALGORITHM_CONSTANTS
    }
    tprint(f"✅ Constants dictionary created with {len(constants_dict)} categories")
    return constants_dict