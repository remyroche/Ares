"""
Data quality utilities package.

This package provides comprehensive data quality validation, scoring, and cleaning utilities.
"""

from .data_quality import (
    DataQualityFramework,
    QualityThresholds,
    QualityResult,
    quick_validate_dataframe,
    validate_unified_dataframe,
    check_dataframe_health,
    data_quality_framework
)

from .data_cleaning import (
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

from .comprehensive_quality_scorer import (
    get_quality_scorer,
)

from .advanced_quality_metrics import (
    AdvancedQualityMetrics,
)

from .comprehensive_duplicate_analyzer import (
    ComprehensiveDuplicateAnalyzer,
)

from .quality_alert_system import (
    QualityAlertSystem,
)

__all__ = [
    # Data Quality
    'DataQualityFramework',
    'QualityThresholds',
    'QualityResult',
    'quick_validate_dataframe',
    'validate_unified_dataframe',
    'check_dataframe_health',
    'data_quality_framework',
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
    # Quality Scoring
    'get_quality_scorer',
    # Advanced Metrics
    'AdvancedQualityMetrics',
    # Duplicate Analysis
    'ComprehensiveDuplicateAnalyzer',
    # Alert System
    'QualityAlertSystem',
]
