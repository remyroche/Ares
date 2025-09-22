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
    # Models
    from .models import (
        EnhancedModelFactory, ModelType, ModelConfig,
        create_analyst_models, create_tactician_models, create_model_factory,
        MultiOutputConfig, MultiOutputModel, MultiOutputStackingModel, MultiOutputResult,
        prepare_multi_output_targets, create_analyst_outputs, create_tactician_outputs,
        create_multi_output_stacking_model,
        EnhancedModelTrainer, train_model_with_confidence_metrics,
        ModelEvaluator, ModelRegistry
    )
    
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
        ParetoOptimizer, ParetoFront,
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
        ValidationFramework, ConfigurationValidator,
        CrossValidationUtilities, PurgedKFold, TemporalCrossValidator,
        StabilityAnalyzer
    )

    # Universal Validation Integration (Recommended)
    try:
        from .universal_validation_integration import (
            UniversalValidationIntegrator, ValidationIntegrationConfig,
            get_validation_integrator, validate_trained_model, validate_hpo_trial
        )
    except ImportError as e:
        UniversalValidationIntegrator = None
        ValidationIntegrationConfig = None
        get_validation_integrator = None
        validate_trained_model = None
        validate_hpo_trial = None
        tprint(f"⚠️ Universal validation integration not available: {e}")

    # Legacy Comprehensive Utilities (Still Available)
    try:
        from .data_leakage_prevention import (
            DataLeakagePrevention, DataLeakagePreventionConfig,
            create_data_leakage_prevention, validate_data_integrity
        )
    except ImportError as e:
        DataLeakagePrevention = None
        DataLeakagePreventionConfig = None
        create_data_leakage_prevention = None
        validate_data_integrity = None
        tprint(f"⚠️ Data leakage prevention not available: {e}")

    try:
        from .overfitting_monitoring import (
            OverfittingMonitoring, OverfittingMonitoringConfig,
            create_overfitting_monitor, monitor_model_performance
        )
    except ImportError as e:
        OverfittingMonitoring = None
        OverfittingMonitoringConfig = None
        create_overfitting_monitor = None
        monitor_model_performance = None
        tprint(f"⚠️ Overfitting monitoring not available: {e}")

    try:
        from .enhanced_validation import (
            EnhancedValidation, EnhancedValidationConfig,
            create_enhanced_validation, validate_model_comprehensive
        )
    except ImportError as e:
        EnhancedValidation = None
        EnhancedValidationConfig = None
        create_enhanced_validation = None
        validate_model_comprehensive = None
        tprint(f"⚠️ Enhanced validation not available: {e}")

    try:
        from .hpo_overfitting_prevention import (
            HPOOverfittingPrevention, HPOOverfittingPreventionConfig,
            create_hpo_overfitting_prevention, optimize_model_hyperparameters
        )
    except ImportError as e:
        HPOOverfittingPrevention = None
        HPOOverfittingPreventionConfig = None
        create_hpo_overfitting_prevention = None
        optimize_model_hyperparameters = None
        tprint(f"⚠️ HPO with overfitting prevention not available: {e}")

    try:
        from .model_complexity_analysis import (
            ModelComplexityAnalyzer, ModelComplexityAnalysisConfig,
            create_model_complexity_analyzer, analyze_model_complexity
        )
    except ImportError as e:
        ModelComplexityAnalyzer = None
        ModelComplexityAnalysisConfig = None
        create_model_complexity_analyzer = None
        analyze_model_complexity = None
        tprint(f"⚠️ Model complexity analysis not available: {e}")

    # Training utilities with comprehensive validation
    try:
        from .training.training_utils import TrainingUtils
    except ImportError as e:
        TrainingUtils = None
        tprint(f"⚠️ Training utilities not available: {e}")
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
        SharedMLCache,
        limit_blas_threads, get_thread_info, validate_thread_environment,
        LookaheadProtection, MLTrainingSafeguards,
        RobustErrorHandler
    )
    
    # Legacy imports for backward compatibility
    from .feature_selection_backwards_compat import FeatureSelector, FeatureSelectionConfig
    # Avoid importing HMMRegimeDetector at package import to prevent circulars; callers should import directly
    try:
        from .hmm_regime_detection import HMMRegimeDetector, RegimeConfig
    except Exception:
        HMMRegimeDetector = None  # type: ignore
        RegimeConfig = None  # type: ignore
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
        'EnhancedModelFactory', 'ModelType', 'ModelConfig',
        'create_analyst_models', 'create_tactician_models', 'create_model_factory',
        'MultiOutputConfig', 'MultiOutputModel', 'MultiOutputStackingModel', 'MultiOutputResult',
        'prepare_multi_output_targets', 'create_analyst_outputs', 'create_tactician_outputs',
        'create_multi_output_stacking_model',
        'EnhancedModelTrainer', 'train_model_with_confidence_metrics',
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
        
        # Data Processing
        'DataLabeler', 'LabelingConfig',
        'DataQualityChecker', 'QualityReport',
        'RegimeDataProcessor',
        'MultiTimeframeTrainer',
        'SRFeatureIntegrator',
        
        # Validation
        'ValidationFramework', 'ConfigurationValidator',
        'TemporalCrossValidator', 'PurgedKFold', 'CrossValidationUtilities', 'PurgedSplitConfig',
        'StabilityAnalyzer', 'feature_selection_stability', 'aggregate_time_blocks',
        'optimize_threshold', 'calibrate_probabilities',
        
        # Utils
        'setup_logger', 'get_logger',
        'MemoryOptimizer', 'MemoryIntegrator',
        'ParallelProcessor',
        'SharedMLCache',
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

        # New Comprehensive Utilities
        'DataLeakagePrevention', 'DataLeakagePreventionConfig',
        'create_data_leakage_prevention', 'validate_data_integrity',
        'OverfittingMonitoring', 'OverfittingMonitoringConfig',
        'create_overfitting_monitor', 'monitor_model_performance',
        'EnhancedValidation', 'EnhancedValidationConfig',
        'create_enhanced_validation', 'validate_model_comprehensive',
        'HPOOverfittingPrevention', 'HPOOverfittingPreventionConfig',
        'create_hpo_overfitting_prevention', 'optimize_model_hyperparameters',
        'ModelComplexityAnalyzer', 'ModelComplexityAnalysisConfig',
        'create_model_complexity_analyzer', 'analyze_model_complexity',
        'TrainingUtils',

        # Universal Validation Integration
        'UniversalValidationIntegrator', 'ValidationIntegrationConfig',
        'get_validation_integrator', 'validate_trained_model', 'validate_hpo_trial',

        # Backward compatibility
        'tprint'
    ]
    
    tprint("✅ ML Common utilities loaded successfully")
    
except ImportError as e:
    tprint(f"❌ Failed to load ML Common utilities: {e}")
    __all__ = ['tprint']