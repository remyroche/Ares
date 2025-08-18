"""
Centralized Decorators Module
This module centralizes all decorators used throughout the codebase for easy import and management.
"""

# Import all decorators from their respective modules
from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
    handle_file_operations,
)

from src.utils.training_pipeline_decorators import (
    deterministic_seed,
    idempotent_step,
    artifact_write_lock,
    nan_inf_and_constant_guard,
    artifact_versioning,
    time_budget_watchdog,
    validate_step_prerequisites,
    secure_data_processing,
    prevent_data_leakage,
    resource_monitor,
    memory_efficient,
    debug_training_step,
    circuit_breaker_protection,
    validate_step_output,
    quality_gate,
)

from src.utils.data_quality_decorators import (
    validate_data_quality,
    validate_feature_engineering_with_lookahead_bias_detection,
)

from src.utils.decorators import (
    validate_call_or_runtime_types,
    pa_check_input,
    pa_check_output,
    pa_check_io,
    enforce_ndarray,
    auto_vectorize,
    guard_array_nan_inf,
    guard_dataframe_nulls,
    with_tracing_span,
)

from src.utils.enhanced_data_quality_decorators import (
    validate_constant_features,
    validate_low_variance_features,
    validate_data_completeness,
    validate_datetime_index,
    validate_multi_timeframe_alignment,
    validate_hmm_data_requirements,
    validate_data_structure,
    optimize_memory_usage,
    comprehensive_data_validation,
    validate_memory_optimized_data_quality,
    validate_feature_engineering_pipeline,
    validate_hmm_regime_discovery,
    validate_multi_timeframe_processing,
)

# Import auto_fix_data_quality_issues from raw_data_quality_checker
try:
    from src.training.steps.raw_data_quality_checker import auto_fix_data_quality_issues
except ImportError:
    from src.utils.logger import system_logger
    system_logger.warning("Could not import 'auto_fix_data_quality_issues' due to an ImportError. Using a pass-through decorator. This may be due to a circular dependency.")
    def auto_fix_data_quality_issues(func):
        return func

# Import advanced decorators
from src.utils.advanced_decorators import (
    performance_monitor,
    model_validation,
    pipeline_checkpoint,
    intelligent_caching,
    adaptive_resource_allocation,
    comprehensive_validation,
    PerformanceLevel,
    ValidationLevel,
)

# Export all decorators for easy import
__all__ = [
    # Error handling decorators
    "handle_errors",
    "handle_specific_errors", 
    "handle_file_operations",
    
    # Training pipeline decorators
    "deterministic_seed",
    "idempotent_step",
    "artifact_write_lock",
    "nan_inf_and_constant_guard",
    "artifact_versioning",
    "time_budget_watchdog",
    "validate_step_prerequisites",
    "secure_data_processing",
    "prevent_data_leakage",
    "resource_monitor",
    "memory_efficient",
    "debug_training_step",
    "circuit_breaker_protection",
    "validate_step_output",
    "quality_gate",
    
    # Data quality decorators
    "validate_data_quality",
    "validate_feature_engineering_with_lookahead_bias_detection",
    
    # General decorators
    "validate_call_or_runtime_types",
    "pa_check_input",
    "pa_check_output",
    "pa_check_io",
    "enforce_ndarray",
    "auto_vectorize",
    "guard_array_nan_inf",
    "guard_dataframe_nulls",
    "with_tracing_span",
    
    # Enhanced data quality decorators
    "validate_constant_features",
    "validate_low_variance_features",
    "validate_data_completeness",
    "validate_datetime_index",
    "validate_multi_timeframe_alignment",
    "validate_hmm_data_requirements",
    "validate_data_structure",
    "optimize_memory_usage",
    "comprehensive_data_validation",
    "validate_memory_optimized_data_quality",
    "validate_feature_engineering_pipeline",
    "validate_hmm_regime_discovery",
    "validate_multi_timeframe_processing",
    
    # Other decorators
    "auto_fix_data_quality_issues",
    
    # Advanced decorators
    "performance_monitor",
    "model_validation",
    "pipeline_checkpoint",
    "intelligent_caching",
    "adaptive_resource_allocation",
    "comprehensive_validation",
    "PerformanceLevel",
    "ValidationLevel",
]