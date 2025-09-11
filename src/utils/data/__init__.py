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

from .unified_data_utils import (
    UnifiedDataUtils,
    unified_data_utils
)

# Import backwards compatibility aliases
from .backwards_compatibility import *

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
    'unified_data_utils'
]
