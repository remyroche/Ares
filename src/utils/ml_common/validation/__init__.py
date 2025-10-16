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
from .cv_utils import TimeSeriesSplitValidator, OOFGenerator
from .data_leakage_detector import DataLeakageDetector

# Backward-compatibility shims for legacy imports
# Legacy code may import these from validation. Provide thin wrappers/aliases.

# Alias legacy TemporalCrossValidator to the unified implementation
TemporalCrossValidator = UnifiedCrossValidator

# Alias CrossValidator to the unified implementation for compatibility
CrossValidator = UnifiedCrossValidator

def cross_validation_utils():
    """Get cross validation utilities instance."""
    return CrossValidationUtilities()

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

# Simple validation framework for backward compatibility
class SimpleValidationFramework:
    """Simple validation framework for basic data validation needs."""

    def validate_data(self, data, data_type=None):
        """Validate data and return validation results."""
        # Simple validation - check if data is not None and has content
        if data is None:
            return False, {"error": "Data is None"}, None

        try:
            # Check if data has content
            if hasattr(data, '__len__') and len(data) == 0:
                return False, {"error": "Data is empty"}, None

            # Basic validation passed
            return True, {"status": "valid"}, data

        except Exception as e:
            return False, {"error": f"Validation failed: {str(e)}"}, None

    def validate_pipeline_state(self, pipeline_state):
        """Validate pipeline state."""
        if pipeline_state is None:
            return False, {"error": "Pipeline state is None"}

        try:
            # Basic validation - check if pipeline state has required attributes
            if not hasattr(pipeline_state, '__dict__'):
                return False, {"error": "Pipeline state is not a valid object"}

            return True, {"status": "valid"}

        except Exception as e:
            return False, {"error": f"Pipeline validation failed: {str(e)}"}

    def validate_optimization_results(self, optimization_result):
        """Validate optimization results."""
        if optimization_result is None:
            return False, {"error": "Optimization result is None"}

        try:
            # Basic validation
            return True, {"status": "valid"}

        except Exception as e:
            return False, {"error": f"Optimization validation failed: {str(e)}"}

    def generate_validation_summary(self, validation_results):
        """Generate validation summary."""
        if validation_results is None:
            return {"status": "no_results"}

        try:
            return {
                "status": "completed",
                "results": validation_results,
                "timestamp": __import__('time').time()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

def get_validation_framework():
    """Get a validation framework instance."""
    return SimpleValidationFramework()

__all__ = [
    # Original Validation Utils
    'ConfigurationValidator',

    # Original Cross-validation
    'TemporalCrossValidator', 'CrossValidator', 'PurgedKFold', 'CrossValidationUtilities', 'PurgedSplitConfig',
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
    'load_validation_config',

    # Simple validation framework
    'get_validation_framework',
    'SimpleValidationFramework'
]
