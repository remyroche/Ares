"""
ML Common Validation Utilities

Comprehensive validation framework for the Analyst→Tactician pipeline.
Provides temporal alignment, leakage detection, window quality assessment,
and performance benchmarking utilities.
"""

from src.utils.tprint import tprint_data_format, LogLevel

from .temporal import (
    assert_aligned,
    validate_temporal_consistency,
    compute_data_hash,
    check_index_properties,
    compute_index_drift,
    TemporalAlignmentResult
)

from .leakage import (
    assert_past_only,
    validate_leakage_prevention,
    detect_negative_shifts,
    analyze_feature_shifts,
    rolling_holdout_test,
    analyze_feature_label_correlation,
    LeakageDetectionResult
)

from .windows import (
    assess_windows,
    validate_window_quality,
    validate_window_structure,
    calculate_window_statistics,
    detect_window_quality_issues,
    WindowQualityResult
)

from .benchmarks import (
    benchmark_stage,
    benchmark_function,
    benchmark_pipeline_stage,
    create_performance_report,
    validate_performance_requirements,
    PerformanceMonitor,
    PerformanceMetrics,
    BenchmarkConfig
)

from .universal_ml_validation import (
    validate_ml_model,
    get_ml_validator,
    UniversalMLValidationConfig,
    UniversalMLValidationReport
)

from .universal_temporal_validation import (
    TemporalValidationReport
)

from .enhanced_overfitting_detection import (
    get_overfitting_detector,
    OverfittingConfig,
    OverfittingReport
)

from .universal_temporal_validation import (
    get_temporal_validator,
    TemporalValidationConfig
)

from .validation_utils import (
    ConfigurationValidator
)

from .data_leakage_detector import (
    DataLeakageDetector
)

from .cv_utils import (
    CrossValidationUtilities,
    PurgedKFold,
    TemporalCrossValidator,
    TimeSeriesSplitValidator,
    OOFGenerator
)

from .unified_cv import (
    UnifiedCrossValidator,
    UnifiedCVResult,
    perform_cross_validation,
    temporal_cross_validation,
    nested_cross_validation
)

from .stability import (
    StabilityAnalyzer
)

__all__ = [
    # Temporal validation
    'assert_aligned',
    'validate_temporal_consistency',
    'compute_data_hash',
    'check_index_properties',
    'compute_index_drift',
    'TemporalAlignmentResult',
    
    # Leakage detection
    'assert_past_only',
    'validate_leakage_prevention',
    'detect_negative_shifts',
    'analyze_feature_shifts',
    'rolling_holdout_test',
    'analyze_feature_label_correlation',
    'LeakageDetectionResult',
    
    # Window quality assessment
    'assess_windows',
    'validate_window_quality',
    'validate_window_structure',
    'calculate_window_statistics',
    'detect_window_quality_issues',
    'WindowQualityResult',
    
    # Performance benchmarking
    'benchmark_stage',
    'benchmark_function',
    'benchmark_pipeline_stage',
    'create_performance_report',
    'validate_performance_requirements',
    'PerformanceMonitor',
    'PerformanceMetrics',
    'BenchmarkConfig',
    
    # Universal ML validation
    'validate_ml_model',
    'get_ml_validator',
    'UniversalMLValidationConfig',
    'UniversalMLValidationReport',
    'TemporalValidationReport',
    
    # Overfitting detection
    'get_overfitting_detector',
    'OverfittingConfig',
    'OverfittingReport',
    
    # Temporal validation
    'get_temporal_validator',
    'TemporalValidationConfig',
    
    # Configuration validation
    'ConfigurationValidator',
    
    # Data leakage detection
    'DataLeakageDetector',
    
    # Cross-validation utilities
    'CrossValidationUtilities',
    'PurgedKFold',
    'TemporalCrossValidator',
    'TimeSeriesSplitValidator',
    'OOFGenerator',
    'UnifiedCrossValidator',
    'UnifiedCVResult',
    'perform_cross_validation',
    'temporal_cross_validation',
    'nested_cross_validation',
    
    # Stability analysis
    'StabilityAnalyzer'
]