"""Validation utilities for the pre-training step."""

# Import core utilities for easy access in validation modules
try:
    from ....utils.tprint import (
        tprint, tprint_debug, tprint_error, tprint_info, tprint_warning,
        tprint_performance, tprint_timer
    )
    from ....utils.common_operations import (
        validate_dataframe, validate_positive, validate_range, safe_divide,
        format_bytes, get_dataframe_info, calculate_data_quality_metrics,
        timed_operation, optimize_dataframe_dtypes
    )
    from ....utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from ....utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    # Import matrix operations for validation modules
    from ....utils.matrix_operations import (
        safe_correlation_matrix, matrix_correlation_analysis, optimize_dataframe,
        get_unified_matrix_operations, get_vectorized_processing_core, get_batch_matrix_processor
    )

    # Core utilities available for use in validation modules
    CORE_UTILITIES_AVAILABLE = True
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    # Fallback if core utilities are not available
    # tprint will be imported from src.utils.tprint
    def tprint_debug(*args, **kwargs): pass
    def tprint_error(*args, **kwargs): pass
    def tprint_info(*args, **kwargs): pass
    def tprint_warning(*args, **kwargs): pass
    def tprint_performance(*args, **kwargs): pass
    def tprint_timer(*args, **kwargs): pass
    def validate_dataframe(df): return True
    def validate_positive(value, name="value"): return value
    def validate_range(value, min_val=None, max_val=None, name="value"): return value
    def safe_divide(a, b, default=0.0): return a / b if b != 0 else default
    def format_bytes(bytes_value): return f"{bytes_value}B"
    def get_dataframe_info(df): return {}
    def calculate_data_quality_metrics(df): return {}
    def timed_operation(func): return func
    def optimize_dataframe_dtypes(df): return df
    def get_m1_memory_optimizer(): return None
    def get_m1_cpu_optimizer(): return None
    # Matrix operations fallbacks
    def safe_correlation_matrix(df): return df.corr() if hasattr(df, 'corr') else None
    def matrix_correlation_analysis(*args, **kwargs): return {}
    def optimize_dataframe(df): return df
    def get_unified_matrix_operations(): return None
    def get_vectorized_processing_core(): return None
    def get_batch_matrix_processor(): return None

    CORE_UTILITIES_AVAILABLE = False
    MATRIX_OPERATIONS_AVAILABLE = False

# Explicit re-exports instead of star imports for better maintainability
from .data_contracts import (
    DataContractValidationError,
    FeaturesSchema,
    LabeledDataSchema,
    SelectionResultSchema,
    validate_feature_artifact,
    validate_multi_horizon_labeling_result,
    validate_selection_artifact,
)
from .schemas import (
    ENGINEERED_FEATURE_SCHEMA,
    LABELED_DATASET_SCHEMA,
    RAW_OHLCV_SCHEMA,
    SCHEMA_REGISTRY,
    SchemaValidationException,
    enforce_feature_temporal_alignment,
    schema_metadata,
    validate_engineered_features,
    validate_labeled_dataset,
    validate_raw_ohlcv,
)
from .temporal_leakage import (
    TemporalLintError,
    TemporalLintViolation,
    lint_for_temporal_leakage,
    run_temporal_linting,
    main as temporal_lint_main,
)

__all__ = [
    # Core utility functions (available when core utilities are installed)
    "tprint",
    "tprint_debug",
    "tprint_error",
    "tprint_info",
    "tprint_warning",
    "tprint_performance",
    "tprint_timer",
    "validate_dataframe",
    "validate_positive",
    "validate_range",
    "safe_divide",
    "format_bytes",
    "get_dataframe_info",
    "calculate_data_quality_metrics",
    "timed_operation",
    "optimize_dataframe_dtypes",
    "get_m1_memory_optimizer",
    "get_m1_cpu_optimizer",
    "CORE_UTILITIES_AVAILABLE",
    "MATRIX_OPERATIONS_AVAILABLE",
    # Matrix operations utilities
    "safe_correlation_matrix",
    "matrix_correlation_analysis",
    "optimize_dataframe",
    "get_unified_matrix_operations",
    "get_vectorized_processing_core",
    "get_batch_matrix_processor",
    # From data_contracts
    "DataContractValidationError",
    "FeaturesSchema",
    "LabeledDataSchema",
    "SelectionResultSchema",
    "validate_feature_artifact",
    "validate_multi_horizon_labeling_result",
    "validate_selection_artifact",
    # From schemas
    "ENGINEERED_FEATURE_SCHEMA",
    "LABELED_DATASET_SCHEMA",
    "RAW_OHLCV_SCHEMA",
    "SCHEMA_REGISTRY",
    "SchemaValidationException",
    "enforce_feature_temporal_alignment",
    "schema_metadata",
    "validate_engineered_features",
    "validate_labeled_dataset",
    "validate_raw_ohlcv",
    # From temporal_leakage
    "TemporalLintError",
    "TemporalLintViolation",
    "lint_for_temporal_leakage",
    "run_temporal_linting",
    "temporal_lint_main",
]
