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

# Import from sub-modules
try:
    # Models - use lazy imports to avoid circular dependencies
    from .models import (
        EnhancedModelFactory, ModelType, ModelConfig,
        create_model_factory,
        MultiOutputConfig, MultiOutputModel, MultiOutputStackingModel, MultiOutputResult,
        prepare_multi_output_targets, create_analyst_outputs, create_tactician_outputs,
        create_multi_output_stacking_model,
        train_model_with_confidence_metrics,
        ModelEvaluator, ModelRegistry
    )
    
    # Lazy import for EnhancedModelTrainer to avoid circular dependency
    from .models import get_enhanced_model_trainer
    
    # Ensembles
    from .ensembles import (
        EnsembleManager, EnsembleType, EnsembleConfig,
        # VotingEnsemble, StackingEnsemble, BlendingEnsemble,
        # WeightedAverageEnsemble, DynamicWeightingEnsemble,
        StackingEnsembleManager, StackingEnsembleConfig, StackingEnsembleResult,
        create_analyst_ensemble, create_tactician_ensemble,
        StackingConfidenceCalibrator, StackingCalibrationConfig, StackingCalibrationResult,
        create_analyst_calibrator, create_tactician_calibrator
    )
    
    # Explainability
    from .explainability import (
        ModelExplainer,
        ModelInterpretabilityEngine, ExplanationResult
    )
    
    # Optimization
    from .optimization import (
        # HyperparameterOptimization,
        ParetoOptimizer, ParetoFront, ParetoFrontAnalyzer,
        RegimeSpecificTPSLOptimizer
    )
    
    # Data Processing (avoid heavy imports at module import time)
    # Expose lightweight getters instead of importing heavy classes to prevent circulars
    try:
        from .data_processing import (
            get_enhanced_data_labeler as EnhancedDataLabelerGetter,
            get_labeling_config as LabelingConfigGetter
        )
    except Exception as e:
        EnhancedDataLabelerGetter = None  # type: ignore
        LabelingConfigGetter = None  # type: ignore
        tprint(f"⚠️ Data processing getters not available at init: {e}")
    # Defer other heavy utilities to call sites
    
    # Validation
    from .validation import (
        ConfigurationValidator,
        CrossValidationUtilities, PurgedKFold, TemporalCrossValidator,
        TimeSeriesSplitValidator, OOFGenerator, DataLeakageDetector,
        StabilityAnalyzer,
        # Unified CV
        UnifiedCrossValidator, UnifiedCVResult,
        perform_cross_validation, temporal_cross_validation, nested_cross_validation
    )
    # Thresholding functions (imported separately to avoid sklearn dependency issues)
    try:
        from .validation.thresholding import optimize_threshold, calibrate_probabilities
    except ImportError as e:
        optimize_threshold = None  # type: ignore
        calibrate_probabilities = None  # type: ignore
        tprint(f"⚠️ Thresholding functions not available: {e}")
    
    # Utils
    from .utils import (
        setup_logger, get_logger,
        MemoryOptimizer, MemoryIntegrator,
        ParallelProcessor,
        UnifiedCache, get_unified_cache, cached,
        limit_blas_threads, get_thread_info, validate_thread_environment,
        LookaheadProtection, MLTrainingSafeguards,
        RobustErrorHandler
    )
    
    # Legacy imports for backward compatibility
    from .feature_selection_backwards_compat import FeatureSelector, FeatureSelectionConfig
    # HMM regime detection module has been deprecated; keep flag for compatibility probes
    HMM_REGIME_DETECTION_AVAILABLE = False
    from .confidence_metrics import calculate_confidence_metrics, calculate_calibration_metrics
    # Defer matrix operations to avoid circular import at init
    try:
        from ..matrix_operations import M1EnhancedMatrixOperations, get_enhanced_matrix_operations
    except Exception:
        M1EnhancedMatrixOperations = None  # type: ignore
        get_enhanced_matrix_operations = None  # type: ignore
    from .pipeline_orchestrator import MLPipelineOrchestrator as PipelineOrchestrator
    from .feature_selection_backwards_compat import FeatureSelector as LegacyFeatureSelector
    from ..feature_selection.feature_importance_analyzer import (
        FeatureImportanceAnalyzer, FeatureImportanceConfig, FeatureImportanceResult,
        ImportanceMethod, analyze_feature_importance, get_important_features
    )
    from .data_drift_detector import (
        DataDriftDetector, DriftDetectionConfig, DriftReport, DriftResult,
        DriftType, DriftMethod, DriftSeverity, detect_data_drift, get_drifted_features
    )
    
    # Define exports
    __all__ = [
        # Models
        'EnhancedModelFactory', 'ModelType', 'ModelConfig', 'create_model_factory',
        'MultiOutputConfig', 'MultiOutputModel', 'MultiOutputStackingModel', 'MultiOutputResult',
        'prepare_multi_output_targets', 'create_analyst_outputs', 'create_tactician_outputs',
        'create_multi_output_stacking_model',
        'get_enhanced_model_trainer', 'train_model_with_confidence_metrics',
        'ModelEvaluator', 'ModelRegistry',
        
        # Ensembles
        'EnsembleManager', 'EnsembleType', 'EnsembleConfig',
        # 'VotingEnsemble', 'StackingEnsemble', 'BlendingEnsemble',
        # 'WeightedAverageEnsemble', 'DynamicWeightingEnsemble',
        'StackingEnsembleManager', 'StackingEnsembleConfig', 'StackingEnsembleResult',
        'create_analyst_ensemble', 'create_tactician_ensemble',
        'StackingConfidenceCalibrator', 'StackingCalibrationConfig', 'StackingCalibrationResult',
        'create_analyst_calibrator', 'create_tactician_calibrator',
        
        # Explainability
        'ModelExplainer',
        'ModelInterpretabilityEngine', 'ExplanationResult',
        
        # Optimization
        # 'HyperparameterOptimization',
        'ParetoFront', 'ParetoFrontAnalyzer',
        'RegimeSpecificTPSLOptimizer',
        
        # Data Processing (expose getters rather than heavy objects)
        'EnhancedDataLabelerGetter', 'LabelingConfigGetter',
        
        # Validation
        'ConfigurationValidator',
        'TemporalCrossValidator', 'PurgedKFold', 'CrossValidationUtilities', 'PurgedSplitConfig',
        'TimeSeriesSplitValidator', 'OOFGenerator', 'DataLeakageDetector',
        # Unified CV exports
        'UnifiedCrossValidator', 'UnifiedCVResult',
        'perform_cross_validation', 'temporal_cross_validation', 'nested_cross_validation',
        'StabilityAnalyzer', 'feature_selection_stability', 'aggregate_time_blocks',
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
