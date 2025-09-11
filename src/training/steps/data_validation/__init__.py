"""
Data Validation and Quality Assurance Step

This package provides comprehensive data validation and quality assurance
for all training data, ensuring data integrity and quality before model training.

Note: This package now directly uses src.utils.ml_common.data_quality
for all data validation functionality.
"""

__version__ = "1.0.0"
__author__ = "Data Validation Framework"

# Import existing data quality utilities from ml_common
try:
    from src.utils.ml_common.data_quality import (
        DataQualityUtilities,
        detect_concept_drift,
        analyze_feature_stability,
        calculate_data_quality_score,
        enhanced_automated_data_cleaning
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

__all__ = [
    'DataQualityUtilities',
    'detect_concept_drift',
    'analyze_feature_stability', 
    'calculate_data_quality_score',
    'enhanced_automated_data_cleaning',
    'VALIDATION_AVAILABLE'
]