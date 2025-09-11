"""
Backwards Compatibility Module

This module provides backwards compatibility aliases for the old module structure
to ensure existing code continues to work after the consolidation.

All old imports will continue to work by redirecting to the new consolidated modules.
"""

# Import from consolidated modules
from .quality.data_quality import (
    DataQualityFramework as EnhancedDataQualityValidator,
    QualityThresholds,
    QualityResult,
    quick_validate_dataframe,
    validate_unified_dataframe,
    check_dataframe_health,
    data_quality_framework
)

from .processing.data_processing import (
    DataProcessor,
    regularize_timestamps,
    preprocess_data_for_multi_timeframe,
    validate_and_fix_data_quality,
    optimize_dataframe_dtypes,
    get_optimal_dtypes_for_features,
    apply_feature_specific_optimization,
    optimize_feature_engineering_pipeline,
    data_processor
)

from .quality.data_cleaning import (
    DataCleaner,
    GapType,
    OutlierSeverity,
    GapInfo,
    OutlierInfo,
    DataSchema,
    handle_missing_values_intelligently,
    detect_outliers,
    validate_data_schema,
    enhanced_missing_value_handler,
    enhanced_outlier_handler
)

from .processing.transformers import (
    DataStreamingManager,
    data_streaming_manager
)

from .validation.validators import (
    CrossStepValidator,
    DataLineage,
    ConsistencyIssue,
    cross_step_validator
)

# Backwards compatibility aliases for old module names
class DataFrameValidator:
    """Backwards compatibility alias for DataFrameValidator."""
    def __init__(self):
        self.processor = DataProcessor()
    
    def validate_dataframe(self, df, **kwargs):
        return self.processor.validate_and_fix_data_quality(df, **kwargs)

class DataFrameCleaner:
    """Backwards compatibility alias for DataFrameCleaner."""
    def __init__(self):
        self.processor = DataProcessor()
        self.cleaner = DataCleaner()
    
    def clean_dataframe(self, df, **kwargs):
        return self.processor.validate_and_fix_data_quality(df, **kwargs)

class DataFrameTransformer:
    """Backwards compatibility alias for DataFrameTransformer."""
    def __init__(self):
        self.processor = DataProcessor()
    
    def transform_dataframe(self, df, **kwargs):
        return self.processor.regularize_timestamps(df, **kwargs)

# Backwards compatibility functions
def validate_dataframe(df, **kwargs):
    """Backwards compatibility function for validate_dataframe."""
    processor = DataProcessor()
    return processor.validate_and_fix_data_quality(df, **kwargs)

def clean_dataframe(df, **kwargs):
    """Backwards compatibility function for clean_dataframe."""
    processor = DataProcessor()
    return processor.validate_and_fix_data_quality(df, **kwargs)

def transform_dataframe(df, **kwargs):
    """Backwards compatibility function for transform_dataframe."""
    processor = DataProcessor()
    return processor.regularize_timestamps(df, **kwargs)

# Backwards compatibility for old class names
DataFormattingFramework = DataQualityFramework
DataFormat = QualityResult
ColumnNamingConvention = DataSchema

# Backwards compatibility for old function names
data_preprocessing = preprocess_data_for_multi_timeframe
data_processing_utils = optimize_dataframe_dtypes
enhanced_data_operations = apply_feature_specific_optimization

# Backwards compatibility for old class instances
DataLoader = DataStreamingManager
OptimizedDataManager = DataProcessor

# Export all for backwards compatibility
__all__ = [
    # Old class names
    'DataFrameValidator',
    'DataFrameCleaner', 
    'DataFrameTransformer',
    'DataFormattingFramework',
    'DataFormat',
    'ColumnNamingConvention',
    'DataLoader',
    'OptimizedDataManager',
    
    # Old function names
    'validate_dataframe',
    'clean_dataframe',
    'transform_dataframe',
    'data_preprocessing',
    'data_processing_utils',
    'enhanced_data_operations',
    
    # New consolidated classes
    'EnhancedDataQualityValidator',
    'QualityThresholds',
    'QualityResult',
    'DataProcessor',
    'DataCleaner',
    'DataStreamingManager',
    'CrossStepValidator',
    
    # New consolidated functions
    'quick_validate_dataframe',
    'validate_unified_dataframe',
    'check_dataframe_health',
    'regularize_timestamps',
    'preprocess_data_for_multi_timeframe',
    'validate_and_fix_data_quality',
    'optimize_dataframe_dtypes',
    'get_optimal_dtypes_for_features',
    'apply_feature_specific_optimization',
    'optimize_feature_engineering_pipeline',
    'handle_missing_values_intelligently',
    'detect_outliers',
    'validate_data_schema',
    
    # Global instances
    'data_quality_framework',
    'data_processor',
    'enhanced_missing_value_handler',
    'enhanced_outlier_handler',
    'data_streaming_manager',
    'cross_step_validator'
]