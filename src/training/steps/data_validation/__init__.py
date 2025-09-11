"""
Data Validation and Quality Assurance Step

This package provides comprehensive data validation and quality assurance
for all training data, ensuring data integrity and quality before model training.
"""

__version__ = "1.0.0"
__author__ = "Data Validation Framework"

# Import main validation components
try:
    from .data_quality_validator import DataQualityValidator
    from .schema_validator import SchemaValidator
    from .temporal_validator import TemporalValidator
    from .statistical_validator import StatisticalValidator
    from .data_validation_pipeline import DataValidationPipeline
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

__all__ = [
    'DataQualityValidator',
    'SchemaValidator', 
    'TemporalValidator',
    'StatisticalValidator',
    'DataValidationPipeline',
    'VALIDATION_AVAILABLE'
]