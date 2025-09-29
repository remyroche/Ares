# Data processing utilities
# Consolidated modules for improved organization and reduced redundancy

# Import from consolidated modules
from .quality.data_quality import (
    DataQualityFramework,
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
    enhanced_outlier_handler,
    get_data_cleaner
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

from .unified_data_utils import (
    UnifiedDataUtils,
    unified_data_utils
)

# Backwards compatibility aliases - integrated directly
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

__all__ = [
    # Data Quality Framework
    'DataQualityFramework',
    'QualityThresholds', 
    'QualityResult',
    'quick_validate_dataframe',
    'validate_unified_dataframe',
    'check_dataframe_health',
    'data_quality_framework',
    
    # Data Processing
    'DataProcessor',
    'regularize_timestamps',
    'preprocess_data_for_multi_timeframe',
    'validate_and_fix_data_quality',
    'optimize_dataframe_dtypes',
    'get_optimal_dtypes_for_features',
    'apply_feature_specific_optimization',
    'optimize_feature_engineering_pipeline',
    'data_processor',
    
    # Data Cleaning
    'DataCleaner',
    'GapType',
    'OutlierSeverity',
    'GapInfo',
    'OutlierInfo',
    'DataSchema',
    'handle_missing_values_intelligently',
    'detect_outliers',
    'validate_data_schema',
    'enhanced_missing_value_handler',
    'enhanced_outlier_handler',
    'get_data_cleaner',
    
    # Data Streaming
    'DataStreamingManager',
    'data_streaming_manager',
    
    # Cross-step Validation
    'CrossStepValidator',
    'DataLineage',
    'ConsistencyIssue',
    'cross_step_validator',
    
    # Unified Interface
    'UnifiedDataUtils',
    'unified_data_utils',
    
    # Backwards Compatibility Classes
    'DataFrameValidator',
    'DataFrameCleaner',
    'DataFrameTransformer',
    'DataFormattingFramework',
    'DataFormat',
    'ColumnNamingConvention',
    'DataLoader',
    'OptimizedDataManager',
    
    # Backwards Compatibility Functions
    'validate_dataframe',
    'clean_dataframe',
    'transform_dataframe',
    'data_preprocessing',
    'data_processing_utils',
    'enhanced_data_operations'
]
