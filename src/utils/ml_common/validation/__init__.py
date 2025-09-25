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

from .validation_utils import ConfigurationValidator
from .unified_cv import (
    UnifiedCrossValidator,
    UnifiedCVResult,
    perform_cross_validation,
    temporal_cross_validation,
    nested_cross_validation,
)
from .cv import PurgedSplitConfig, purged_time_series_splits

# Backward-compatibility shims for legacy imports
# Legacy code may import these from validation. Provide thin wrappers/aliases.

# Alias legacy TemporalCrossValidator to the unified implementation
TemporalCrossValidator = UnifiedCrossValidator


class CrossValidationUtilities:  # minimal shim
    """Backwards-compatible utilities wrapper.

    Currently implements walk_forward_validation using the unified CV API
    with temporal strategy.
    """

    @staticmethod
    def walk_forward_validation(model, X, y, *, n_splits: int = 5, gap: int = 0,
                                test_size: int | None = None, scoring=None, n_jobs: int = 1):
        # Delegate to unified API
        return perform_cross_validation(
            model,
            X,
            y,
            strategy="temporal",
            cv_folds=n_splits,
            temporal_gap=gap,
            temporal_test_size=test_size,
            scoring=scoring,
            n_jobs=n_jobs,
        )


# PurgedKFold legacy name: expose the available split config to avoid import errors
PurgedKFold = PurgedSplitConfig
from .stability import feature_selection_stability, aggregate_time_blocks, StabilityAnalyzer
from .thresholding import optimize_threshold, calibrate_probabilities

# Enhanced validation components (consolidated with unified CV)
from src.utils.nas_tas.advanced_validation import (
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
# Enhanced validation (now uses unified CV)
from .enhanced_validation import (
    EnhancedValidator,
    EnhancedValidationConfig,
    ValidationReport,
    get_enhanced_validator,
    validate_model_comprehensively
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
    'ConfigurationValidator',

    # Original Cross-validation
    'TemporalCrossValidator', 'PurgedKFold', 'CrossValidationUtilities', 'PurgedSplitConfig',
    'purged_time_series_splits',
    # Unified CV API
    'UnifiedCrossValidator', 'UnifiedCVResult',
    'perform_cross_validation', 'temporal_cross_validation', 'nested_cross_validation',

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

    # Enhanced Validation (consolidated with unified CV)
    'EnhancedValidator',
    'EnhancedValidationConfig',
    'ValidationReport',
    'get_enhanced_validator',
    'validate_model_comprehensively',

    # Unified helpers
    'save_validation_config',
    'load_validation_config'
]