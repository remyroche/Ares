"""
ML Common Utilities

This module provides common machine learning utilities and components
for the trading system, organized into logical sub-modules.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from src.utils.logger import system_logger

# Configure logging
_LOGGER = system_logger.getChild('MLCommon')

def tprint(message: str, level: str = "INFO") -> None:
    """Print message with timestamp and level."""
    if level.upper() == "ERROR":
        _LOGGER.error(message)
    elif level.upper() == "WARNING":
        _LOGGER.warning(message)
    elif level.upper() == "DEBUG":
        _LOGGER.debug(message)
    else:
        _LOGGER.info(message)

# Lazy import functions to avoid circular imports at package initialization
def __getattr__(name: str):
    """Lazily import heavy submodules to avoid circular imports at import time."""

    # Models
    if name == 'EnhancedModelFactory':
        from .models import EnhancedModelFactory
        return EnhancedModelFactory
    elif name == 'ModelType':
        from .models import ModelType
        return ModelType
    elif name == 'ModelConfig':
        from .models import ModelConfig
        return ModelConfig
    elif name == 'create_analyst_models':
        from .models import create_analyst_models
        return create_analyst_models
    elif name == 'create_tactician_models':
        from .models import create_tactician_models
        return create_tactician_models
    elif name == 'create_model_factory':
        from .models import create_model_factory
        return create_model_factory
    elif name == 'MultiOutputConfig':
        from .models import MultiOutputConfig
        return MultiOutputConfig
    elif name == 'MultiOutputModel':
        from .models import MultiOutputModel
        return MultiOutputModel
    elif name == 'MultiOutputStackingModel':
        from .models import MultiOutputStackingModel
        return MultiOutputStackingModel
    elif name == 'MultiOutputResult':
        from .models import MultiOutputResult
        return MultiOutputResult
    elif name == 'prepare_multi_output_targets':
        from .models import prepare_multi_output_targets
        return prepare_multi_output_targets
    elif name == 'create_analyst_outputs':
        from .models import create_analyst_outputs
        return create_analyst_outputs
    elif name == 'create_tactician_outputs':
        from .models import create_tactician_outputs
        return create_tactician_outputs
    elif name == 'create_multi_output_stacking_model':
        from .models import create_multi_output_stacking_model
        return create_multi_output_stacking_model
    elif name == 'EnhancedModelTrainer':
        from .models import EnhancedModelTrainer
        return EnhancedModelTrainer
    elif name == 'train_model_with_confidence_metrics':
        from .models import train_model_with_confidence_metrics
        return train_model_with_confidence_metrics
    elif name == 'ModelEvaluator':
        from .models import ModelEvaluator
        return ModelEvaluator
    elif name == 'ModelRegistry':
        from .models import ModelRegistry
        return ModelRegistry

    # Ensembles
    elif name == 'EnsembleManager':
        from .ensembles import EnsembleManager
        return EnsembleManager
    elif name == 'EnsembleType':
        from .ensembles import EnsembleType
        return EnsembleType
    elif name == 'EnsembleConfig':
        from .ensembles import EnsembleConfig
        return EnsembleConfig
    elif name == 'StackingEnsembleManager':
        from .ensembles import StackingEnsembleManager
        return StackingEnsembleManager
    elif name == 'StackingEnsembleConfig':
        from .ensembles import StackingEnsembleConfig
        return StackingEnsembleConfig
    elif name == 'StackingEnsembleResult':
        from .ensembles import StackingEnsembleResult
        return StackingEnsembleResult
    elif name == 'create_analyst_ensemble':
        from .ensembles import create_analyst_ensemble
        return create_analyst_ensemble
    elif name == 'create_tactician_ensemble':
        from .ensembles import create_tactician_ensemble
        return create_tactician_ensemble
    elif name == 'StackingConfidenceCalibrator':
        from .ensembles import StackingConfidenceCalibrator
        return StackingConfidenceCalibrator
    elif name == 'StackingCalibrationConfig':
        from .ensembles import StackingCalibrationConfig
        return StackingCalibrationConfig
    elif name == 'StackingCalibrationResult':
        from .ensembles import StackingCalibrationResult
        return StackingCalibrationResult
    elif name == 'create_analyst_calibrator':
        from .ensembles import create_analyst_calibrator
        return create_analyst_calibrator
    elif name == 'create_tactician_calibrator':
        from .ensembles import create_tactician_calibrator
        return create_tactician_calibrator

    # Explainability
    elif name == 'ModelExplainer':
        from .explainability import ModelExplainer
        return ModelExplainer
    elif name == 'ModelInterpretabilityEngine':
        from .explainability import ModelInterpretabilityEngine
        return ModelInterpretabilityEngine
    elif name == 'ExplanationResult':
        from .explainability import ExplanationResult
        return ExplanationResult

    # Optimization
    elif name == 'ParetoOptimizer':
        from .optimization import ParetoOptimizer
        return ParetoOptimizer
    elif name == 'ParetoFront':
        from .optimization import ParetoFront
        return ParetoFront
    elif name == 'RegimeSpecificTPSLOptimizer':
        from .optimization import RegimeSpecificTPSLOptimizer
        return RegimeSpecificTPSLOptimizer

    # Data Processing
    elif name == 'EnhancedDataLabelerGetter':
        try:
            from .data_processing import get_enhanced_data_labeler
            return get_enhanced_data_labeler
        except Exception:
            return None
    elif name == 'LabelingConfigGetter':
        try:
            from .data_processing import get_labeling_config
            return get_labeling_config
        except Exception:
            return None

    # Validation
    elif name == 'ConfigurationValidator':
        from .validation import ConfigurationValidator
        return ConfigurationValidator
    elif name == 'CrossValidationUtilities':
        from .validation import CrossValidationUtilities
        return CrossValidationUtilities
    elif name == 'PurgedKFold':
        from .validation import PurgedKFold
        return PurgedKFold
    elif name == 'TemporalCrossValidator':
        from .validation import TemporalCrossValidator
        return TemporalCrossValidator
    elif name == 'StabilityAnalyzer':
        from .validation import StabilityAnalyzer
        return StabilityAnalyzer
    elif name == 'UnifiedCrossValidator':
        from .validation import UnifiedCrossValidator
        return UnifiedCrossValidator
    elif name == 'UnifiedCVResult':
        from .validation import UnifiedCVResult
        return UnifiedCVResult
    elif name == 'perform_cross_validation':
        from .validation import perform_cross_validation
        return perform_cross_validation
    elif name == 'temporal_cross_validation':
        from .validation import temporal_cross_validation
        return temporal_cross_validation
    elif name == 'nested_cross_validation':
        from .validation import nested_cross_validation
        return nested_cross_validation
    elif name == 'optimize_threshold':
        try:
            from .validation.thresholding import optimize_threshold
            return optimize_threshold
        except ImportError:
            return None
    elif name == 'calibrate_probabilities':
        try:
            from .validation.thresholding import calibrate_probabilities
            return calibrate_probabilities
        except ImportError:
            return None

    # Utils
    elif name == 'setup_logger':
        from .utils import setup_logger
        return setup_logger
    elif name == 'get_logger':
        from .utils import get_logger
        return get_logger
    elif name == 'MemoryOptimizer':
        from .utils import MemoryOptimizer
        return MemoryOptimizer
    elif name == 'MemoryIntegrator':
        from .utils import MemoryIntegrator
        return MemoryIntegrator
    elif name == 'ParallelProcessor':
        from .utils import ParallelProcessor
        return ParallelProcessor
    elif name == 'UnifiedCache':
        from .utils import UnifiedCache
        return UnifiedCache
    elif name == 'get_unified_cache':
        from .utils import get_unified_cache
        return get_unified_cache
    elif name == 'cached':
        from .utils import cached
        return cached
    elif name == 'limit_blas_threads':
        from .utils import limit_blas_threads
        return limit_blas_threads
    elif name == 'get_thread_info':
        from .utils import get_thread_info
        return get_thread_info
    elif name == 'validate_thread_environment':
        from .utils import validate_thread_environment
        return validate_thread_environment
    elif name == 'LookaheadProtection':
        from .utils import LookaheadProtection
        return LookaheadProtection
    elif name == 'MLTrainingSafeguards':
        from .utils import MLTrainingSafeguards
        return MLTrainingSafeguards
    elif name == 'RobustErrorHandler':
        from .utils import RobustErrorHandler
        return RobustErrorHandler

    # Legacy imports
    elif name == 'FeatureSelector':
        from .feature_selection_backwards_compat import FeatureSelector
        return FeatureSelector
    elif name == 'FeatureSelectionConfig':
        from .feature_selection_backwards_compat import FeatureSelectionConfig
        return FeatureSelectionConfig
    elif name == 'HMMRegimeDetector':
        try:
            from .hmm_regime_detection import HMMRegimeDetector
            return HMMRegimeDetector
        except Exception:
            return None
    elif name == 'RegimeConfig':
        try:
            from .hmm_regime_detection import RegimeConfig
            return RegimeConfig
        except Exception:
            return None
    elif name == 'calculate_confidence_metrics':
        from .confidence_metrics import calculate_confidence_metrics
        return calculate_confidence_metrics
    elif name == 'calculate_calibration_metrics':
        from .confidence_metrics import calculate_calibration_metrics
        return calculate_calibration_metrics
    elif name == 'M1EnhancedMatrixOperations':
        try:
            from ..matrix_operations import M1EnhancedMatrixOperations
            return M1EnhancedMatrixOperations
        except Exception:
            return None
    elif name == 'get_enhanced_matrix_operations':
        try:
            from ..matrix_operations import get_enhanced_matrix_operations
            return get_enhanced_matrix_operations
        except Exception:
            return None
    elif name == 'PipelineOrchestrator':
        from .pipeline_orchestrator import MLPipelineOrchestrator
        return MLPipelineOrchestrator
    elif name == 'LegacyFeatureSelector':
        from .feature_selection_backwards_compat import FeatureSelector
        return FeatureSelector
    elif name == 'FeatureImportanceAnalyzer':
        from ..feature_selection.feature_importance_analyzer import FeatureImportanceAnalyzer
        return FeatureImportanceAnalyzer
    elif name == 'FeatureImportanceConfig':
        from ..feature_selection.feature_importance_analyzer import FeatureImportanceConfig
        return FeatureImportanceConfig
    elif name == 'FeatureImportanceResult':
        from ..feature_selection.feature_importance_analyzer import FeatureImportanceResult
        return FeatureImportanceResult
    elif name == 'ImportanceMethod':
        from ..feature_selection.feature_importance_analyzer import ImportanceMethod
        return ImportanceMethod
    elif name == 'analyze_feature_importance':
        from ..feature_selection.feature_importance_analyzer import analyze_feature_importance
        return analyze_feature_importance
    elif name == 'get_important_features':
        from ..feature_selection.feature_importance_analyzer import get_important_features
        return get_important_features
    elif name == 'DataDriftDetector':
        from .data_drift_detector import DataDriftDetector
        return DataDriftDetector
    elif name == 'DriftDetectionConfig':
        from .data_drift_detector import DriftDetectionConfig
        return DriftDetectionConfig
    elif name == 'DriftReport':
        from .data_drift_detector import DriftReport
        return DriftReport
    elif name == 'DriftResult':
        from .data_drift_detector import DriftResult
        return DriftResult
    elif name == 'DriftType':
        from .data_drift_detector import DriftType
        return DriftType
    elif name == 'DriftMethod':
        from .data_drift_detector import DriftMethod
        return DriftMethod
    elif name == 'DriftSeverity':
        from .data_drift_detector import DriftSeverity
        return DriftSeverity
    elif name == 'detect_data_drift':
        from .data_drift_detector import detect_data_drift
        return detect_data_drift
    elif name == 'get_drifted_features':
        from .data_drift_detector import get_drifted_features
        return get_drifted_features

    raise AttributeError(f"module 'utils.ml_common' has no attribute {name!r}")

try:
    # Define exports
    __all__ = [
        # Models
        'EnhancedModelFactory', 'ModelType', 'ModelConfig',
        'create_analyst_models', 'create_tactician_models', 'create_model_factory',
        'MultiOutputConfig', 'MultiOutputModel', 'MultiOutputStackingModel', 'MultiOutputResult',
        'prepare_multi_output_targets', 'create_analyst_outputs', 'create_tactician_outputs',
        'create_multi_output_stacking_model',
        'EnhancedModelTrainer', 'train_model_with_confidence_metrics',
        'ModelEvaluator', 'ModelRegistry',

        # Ensembles
        'EnsembleManager', 'EnsembleType', 'EnsembleConfig',
        'StackingEnsembleManager', 'StackingEnsembleConfig', 'StackingEnsembleResult',
        'create_analyst_ensemble', 'create_tactician_ensemble',
        'StackingConfidenceCalibrator', 'StackingCalibrationConfig', 'StackingCalibrationResult',
        'create_analyst_calibrator', 'create_tactician_calibrator',

        # Explainability
        'ModelExplainer',
        'ModelInterpretabilityEngine', 'ExplanationResult',

        # Optimization
        'ParetoOptimizer', 'ParetoFront',
        'RegimeSpecificTPSLOptimizer',

        # Data Processing (lazy-loaded getters)
        'EnhancedDataLabelerGetter', 'LabelingConfigGetter',

        # Validation
        'ConfigurationValidator',
        'CrossValidationUtilities', 'PurgedKFold', 'TemporalCrossValidator',
        'StabilityAnalyzer', 'UnifiedCrossValidator', 'UnifiedCVResult',
        'perform_cross_validation', 'temporal_cross_validation', 'nested_cross_validation',
        'optimize_threshold', 'calibrate_probabilities',

        # Utils
        'setup_logger', 'get_logger',
        'MemoryOptimizer', 'MemoryIntegrator',
        'ParallelProcessor',
        'UnifiedCache', 'get_unified_cache', 'cached',
        'limit_blas_threads', 'get_thread_info', 'validate_thread_environment',
        'LookaheadProtection', 'MLTrainingSafeguards',
        'RobustErrorHandler',

        # Legacy
        'FeatureSelector', 'FeatureSelectionConfig', 'LegacyFeatureSelector',
        'HMMRegimeDetector', 'RegimeConfig',
        'calculate_confidence_metrics', 'calculate_calibration_metrics',
        'M1EnhancedMatrixOperations', 'get_enhanced_matrix_operations', 'PipelineOrchestrator',

        # Feature Importance Analysis
        'FeatureImportanceAnalyzer', 'FeatureImportanceConfig', 'FeatureImportanceResult',
        'ImportanceMethod', 'analyze_feature_importance', 'get_important_features',

        # Data Drift Detection
        'DataDriftDetector', 'DriftDetectionConfig', 'DriftReport', 'DriftResult',
        'DriftType', 'DriftMethod', 'DriftSeverity', 'detect_data_drift', 'get_drifted_features',

        # Backward compatibility
        'tprint'
    ]

    tprint("✅ ML Common utilities loaded successfully")

except ImportError as e:
    tprint(f"❌ Failed to load ML Common utilities: {e}")
    __all__ = ['tprint']