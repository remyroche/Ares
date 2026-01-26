"""
ML Common Validation Utilities

Comprehensive validation framework for the Analyst→Tactician pipeline.
Provides temporal alignment, leakage detection, window quality assessment,
and performance benchmarking utilities.
"""

from typing import TYPE_CHECKING
from src.utils.lazy_module_loader import make_lazy_getattr, make_lazy_dir

# Define lazy loading map
_EXPORT_MAP = {
    # Temporal validation
    'assert_aligned': '.temporal',
    'validate_temporal_consistency': '.temporal',
    'compute_data_hash': '.temporal',
    'check_index_properties': '.temporal',
    'compute_index_drift': '.temporal',
    'TemporalAlignmentResult': '.temporal',
    
    # Leakage detection
    'assert_past_only': '.leakage',
    'validate_leakage_prevention': '.leakage',
    'detect_negative_shifts': '.leakage',
    'analyze_feature_shifts': '.leakage',
    'rolling_holdout_test': '.leakage',
    'analyze_feature_label_correlation': '.leakage',
    'LeakageDetectionResult': '.leakage',
    
    # Window quality assessment
    'assess_windows': '.windows',
    'validate_window_quality': '.windows',
    'validate_window_structure': '.windows',
    'calculate_window_statistics': '.windows',
    'detect_window_quality_issues': '.windows',
    'WindowQualityResult': '.windows',
    
    # Performance benchmarking
    'benchmark_stage': '.benchmarks',
    'benchmark_function': '.benchmarks',
    'benchmark_pipeline_stage': '.benchmarks',
    'create_performance_report': '.benchmarks',
    'validate_performance_requirements': '.benchmarks',
    'PerformanceMonitor': '.benchmarks',
    'PerformanceMetrics': '.benchmarks',
    'BenchmarkConfig': '.benchmarks',
    
    # Universal ML validation
    'validate_ml_model': '.universal_ml_validation',
    'get_ml_validator': '.universal_ml_validation',
    'UniversalMLValidationConfig': '.universal_ml_validation',
    'UniversalMLValidationReport': '.universal_ml_validation',
    
    # Temporal Validation Report (cross-module)
    'TemporalValidationReport': '.universal_temporal_validation',
    
    # Overfitting detection
    'get_overfitting_detector': '.enhanced_overfitting_detection',
    'OverfittingConfig': '.enhanced_overfitting_detection',
    'OverfittingReport': '.enhanced_overfitting_detection',
    
    # Temporal validation (universal)
    'get_temporal_validator': '.universal_temporal_validation',
    'TemporalValidationConfig': '.universal_temporal_validation',
    
    # Configuration validation
    'ConfigurationValidator': '.validation_utils',
    
    # Data leakage detection
    'DataLeakageDetector': '.data_leakage_detector',
    
    # Cross-validation utilities
    'CrossValidationUtilities': '.cv_utils',
    'PurgedKFold': '.cv_utils',
    'TemporalCrossValidator': '.cv_utils',
    'TimeSeriesSplitValidator': '.cv_utils',
    'OOFGenerator': '.cv_utils',
    
    # Unified CV
    'UnifiedCrossValidator': '.unified_cv',
    'UnifiedCVResult': '.unified_cv',
    'perform_cross_validation': '.unified_cv',
    'temporal_cross_validation': '.unified_cv',
    'nested_cross_validation': '.unified_cv',
    
    # Stability analysis
    'StabilityAnalyzer': '.stability'
}

__all__ = list(_EXPORT_MAP.keys())

# Static typing support
if TYPE_CHECKING:
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
        TemporalValidationReport,
        get_temporal_validator,
        TemporalValidationConfig
    )
    from .enhanced_overfitting_detection import (
        get_overfitting_detector,
        OverfittingConfig,
        OverfittingReport
    )
    from .validation_utils import ConfigurationValidator
    from .data_leakage_detector import DataLeakageDetector
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
    from .stability import StabilityAnalyzer

# Use generalized lazy loading helpers
__getattr__ = make_lazy_getattr(_EXPORT_MAP, __package__)
__dir__ = make_lazy_dir(_EXPORT_MAP, globals())