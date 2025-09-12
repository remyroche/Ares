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
        VotingEnsemble, StackingEnsemble, BlendingEnsemble,
        WeightedAverageEnsemble, DynamicWeightingEnsemble,
        StackingEnsembleManager, StackingEnsembleConfig, StackingEnsembleResult,
        create_analyst_ensemble, create_tactician_ensemble,
        StackingConfidenceCalibrator, StackingCalibrationConfig, StackingCalibrationResult,
        create_analyst_calibrator, create_tactician_calibrator
    )
    
    # Explainability
    from .explainability import (
        ModelExplainer, SHAPExplainer, LIMEExplainer,
        ModelExplanations, ExplanationResult,
        ModelInterpreter, InterpretabilityResult
    )
    
    # Optimization
    from .optimization import (
        HPOOptimizer, HPOConfig, HPOResult,
        ParetoOptimizer, ParetoFront,
        RegimeSpecificTPSLOptimizer
    )
    
    # Data Processing
    from .data_processing import (
        DataLabeler, LabelingConfig,
        DataQualityChecker, QualityReport,
        RegimeDataProcessor,
        MultiTimeframeTrainer,
        SRFeatureIntegrator
    )
    
    # Validation
    from .validation import (
        ValidationUtils, ValidationConfig,
        CVUtils, CVConfig, CrossValidator,
        StabilityAnalyzer,
        ThresholdOptimizer
    )
    
    # Utils
    from .utils import (
        setup_logger, get_logger,
        MemoryOptimizer, MemoryIntegrator,
        ParallelProcessor,
        SharedCache,
        ThreadGuard,
        LookaheadProtection, BaseSafeguards,
        EnhancedErrorHandler
    )
    
    # Legacy imports for backward compatibility
    from .feature_selection import FeatureSelector, FeatureSelectionConfig
    from .hmm_regime_detection import HMMRegimeDetector, RegimeConfig
    from .confidence_metrics import calculate_confidence_metrics, calculate_calibration_metrics
    from .matrix_operations import MatrixOperations
    from .pipeline_orchestrator import PipelineOrchestrator
    from .feature_selection_backwards_compat import FeatureSelector as LegacyFeatureSelector
    
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
        'VotingEnsemble', 'StackingEnsemble', 'BlendingEnsemble',
        'WeightedAverageEnsemble', 'DynamicWeightingEnsemble',
        'StackingEnsembleManager', 'StackingEnsembleConfig', 'StackingEnsembleResult',
        'create_analyst_ensemble', 'create_tactician_ensemble',
        'StackingConfidenceCalibrator', 'StackingCalibrationConfig', 'StackingCalibrationResult',
        'create_analyst_calibrator', 'create_tactician_calibrator',
        
        # Explainability
        'ModelExplainer', 'SHAPExplainer', 'LIMEExplainer',
        'ModelExplanations', 'ExplanationResult',
        'ModelInterpreter', 'InterpretabilityResult',
        
        # Optimization
        'HPOOptimizer', 'HPOConfig', 'HPOResult',
        'ParetoOptimizer', 'ParetoFront',
        'RegimeSpecificTPSLOptimizer',
        
        # Data Processing
        'DataLabeler', 'LabelingConfig',
        'DataQualityChecker', 'QualityReport',
        'RegimeDataProcessor',
        'MultiTimeframeTrainer',
        'SRFeatureIntegrator',
        
        # Validation
        'ValidationUtils', 'ValidationConfig',
        'CVUtils', 'CVConfig', 'CrossValidator',
        'StabilityAnalyzer',
        'ThresholdOptimizer',
        
        # Utils
        'setup_logger', 'get_logger',
        'MemoryOptimizer', 'MemoryIntegrator',
        'ParallelProcessor',
        'SharedCache',
        'ThreadGuard',
        'LookaheadProtection', 'BaseSafeguards',
        'EnhancedErrorHandler',
        
        # Legacy
        'FeatureSelector', 'FeatureSelectionConfig', 'LegacyFeatureSelector',
        'HMMRegimeDetector', 'RegimeConfig',
        'calculate_confidence_metrics', 'calculate_calibration_metrics',
        'MatrixOperations', 'PipelineOrchestrator',
        
        # Backward compatibility
        'tprint'
    ]
    
    tprint("✅ ML Common utilities loaded successfully")
    
except ImportError as e:
    tprint(f"❌ Failed to load ML Common utilities: {e}")
    __all__ = ['tprint']