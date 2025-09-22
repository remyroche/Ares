"""
ML Common - Validation Module

This module contains all validation functionality including:
- Cross-validation utilities
- Model stability assessment
- Threshold optimization
- Validation metrics
- Enhanced overfitting detection
- Universal temporal validation
- Comprehensive ML validation
"""

from .validation_utils import ValidationFramework, ConfigurationValidator
from .cv_utils import TemporalCrossValidator, PurgedKFold, CrossValidationUtilities
from .cv import PurgedSplitConfig
from .stability import feature_selection_stability, aggregate_time_blocks, StabilityAnalyzer
from .thresholding import optimize_threshold, calibrate_probabilities

# Enhanced validation components
from .enhanced_overfitting_detection import (
    UniversalOverfittingDetector,
    OverfittingConfig,
    OverfittingReport,
    get_overfitting_detector,
    detect_overfitting_for_model
)
from .universal_temporal_validation import (
    UniversalTemporalValidator,
    UniversalTemporalCrossValidator,
    UniversalTimeSeriesSplit,
    TemporalValidationConfig,
    TemporalValidationReport,
    get_temporal_validator,
    get_temporal_cv,
    create_time_series_split
)
from .universal_ml_validation import (
    UniversalMLValidator,
    UniversalMLValidationConfig,
    UniversalMLValidationReport,
    get_ml_validator,
    validate_ml_model
)

# Re-export unified validation config helpers
try:
    from src.common.config.validation import (
        EnhancedValidationConfig as UnifiedEnhancedValidationConfig,
        UniversalMLValidationConfig as UnifiedUniversalMLValidationConfig,
        save_validation_config as save_validation_config,
        load_validation_config as load_validation_config,
    )
except Exception:
    pass

__all__ = [
    # Original Validation Utils
    'ValidationFramework', 'ConfigurationValidator',

    # Original Cross-validation
    'TemporalCrossValidator', 'PurgedKFold', 'CrossValidationUtilities', 'PurgedSplitConfig',

    # Original Stability Analysis
    'feature_selection_stability', 'aggregate_time_blocks', 'StabilityAnalyzer',

    # Original Threshold Optimization
    'optimize_threshold', 'calibrate_probabilities',
    
    # Enhanced Overfitting Detection
    'UniversalOverfittingDetector',
    'OverfittingConfig',
    'OverfittingReport',
    'get_overfitting_detector',
    'detect_overfitting_for_model',
    
    # Universal Temporal Validation
    'UniversalTemporalValidator',
    'UniversalTemporalCrossValidator',
    'UniversalTimeSeriesSplit',
    'TemporalValidationConfig',
    'TemporalValidationReport',
    'get_temporal_validator',
    'get_temporal_cv',
    'create_time_series_split',
    
    # Universal ML Validation
    'UniversalMLValidator',
    'UniversalMLValidationConfig',
    'UniversalMLValidationReport',
    'get_ml_validator',
    'validate_ml_model',
    # Unified helpers
    'save_validation_config',
    'load_validation_config'
]