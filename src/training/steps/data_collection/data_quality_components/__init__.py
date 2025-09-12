"""Data Quality Components
Modular components for data quality checking extracted from raw_data_quality_checker.py
"""

from .quality_metrics_calculator import QualityMetricsCalculator
from .data_integrity_checker import DataIntegrityChecker
from .anomaly_detector import AnomalyDetector
from .data_preprocessor import DataPreprocessor
from .config_manager import QualityCheckConfig
from .validation_strategies import (
    ValidationStrategy,
    StructureValidationStrategy,
    CompletenessValidationStrategy,
    IntegrityValidationStrategy,
    MarketSpecificValidationStrategy,
    FeatureEngineeringValidationStrategy
)
from .data_utils import (
    determine_timeframe_from_data,
    estimate_timeframe_from_data,
    fix_datetime_index,
    calculate_interval_statistics,
    detect_data_gaps,
    calculate_data_span,
    validate_ohlc_consistency,
    calculate_volume_statistics,
    generate_data_summary
)
from .error_handler import (
    QualityCheckError,
    ValidationError,
    PreprocessingError,
    DataDownloadError,
    ConfigurationError,
    ErrorHandler
)
from .result_builder import ValidationResultBuilder
from .validation_decorators import (
    validate_data,
    log_validation_progress,
    handle_validation_errors,
    ensure_data_types,
    auto_fix_data_quality_issues
)

__all__ = [
    # Core components
    "QualityMetricsCalculator",
    "DataIntegrityChecker",
    "AnomalyDetector",
    "DataPreprocessor",
    "QualityCheckConfig",
    
    # Validation strategies
    "ValidationStrategy",
    "StructureValidationStrategy",
    "CompletenessValidationStrategy",
    "IntegrityValidationStrategy",
    "MarketSpecificValidationStrategy",
    "FeatureEngineeringValidationStrategy",
    
    # Utility functions
    "determine_timeframe_from_data",
    "estimate_timeframe_from_data",
    "fix_datetime_index",
    "calculate_interval_statistics",
    "detect_data_gaps",
    "calculate_data_span",
    "validate_ohlc_consistency",
    "calculate_volume_statistics",
    "generate_data_summary",
    
    # Error handling
    "QualityCheckError",
    "ValidationError",
    "PreprocessingError",
    "DataDownloadError",
    "ConfigurationError",
    "ErrorHandler",
    
    # Result building
    "ValidationResultBuilder",
    
    # Validation decorators
    "validate_data",
    "log_validation_progress",
    "handle_validation_errors",
    "ensure_data_types",
    "auto_fix_data_quality_issues"
]