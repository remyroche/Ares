"""
ML Common Utilities Package

This package provides comprehensive, standardized utilities for machine learning operations
across all training steps in the Ares trading system.

Key Features:
- Cross-validation utilities with temporal integrity
- Lookahead bias prevention and detection
- Unified feature selection framework
- Comprehensive model evaluation metrics
- Advanced hyperparameter optimization
- Memory-efficient ML training
- Parallel processing coordination
- Model persistence and versioning
- Data quality and preprocessing
- ML pipeline orchestration

All utilities are designed to work seamlessly with:
- M1/M2/M3 GPU acceleration via Metal Performance Shaders
- Memory optimization for large datasets
- Error handling and recovery mechanisms
- Comprehensive logging and monitoring
"""

from .base_safeguards import *
from .cv_utils import *
from .lookahead_protection import *
from .feature_selection import *
from .feature_generation_optimization import *
from .data_labeling import *
from .hmm_regime_detection import *
from .regime_data_processing import *
from .model_evaluation import *
from .hpo_utils import *
from .memory_optimization import *
# Expose coordinator class but not its instance methods as globals
from .parallel_processing import ParallelProcessingCoordinator
from .model_registry import *
from .data_quality import *
from .pipeline_orchestrator import *
from .model_interpretability import *

__version__ = "1.0.0"
__all__ = [
    # Base safeguards
    'MLTrainingSafeguards',
    'MLTrainingError',
    'ClassImbalanceError',
    'SingleClassError',
    'DataQualityError',
    'SmartFastFailHandler',

    # Cross-validation utilities
    'CrossValidationUtilities',

    # Lookahead bias protection
    'LookaheadProtection',
    'advanced_information_barrier_checks',
    'validate_feature_timestamp_alignment',
    'automated_future_data_filtering',
    'rolling_window_bias_validation',

    # Feature selection framework
    'FeatureSelectionFramework',

    # Feature generation optimization
    'FeatureGenerationOptimizer',
    'FeatureOptimizationConfig',
    'FeatureOptimizationResult',
    'OptimizationMethod',
    'get_feature_optimizer',
    'optimize_feature_lookback',

    # Data labeling utilities
    'DataLabelingUtilities',
    'TripleBarrierConfig',
    'LabelingResult',
    'LabelingMethod',
    'get_data_labeler',
    'label_triple_barrier',
    'label_regime_aware',

    # HMM regime detection
    'HMMRegimeDetector',
    'HMMRegimeConfig',
    'RegimeDetectionResult',
    'RegimeDetectionMethod',
    'get_hmm_regime_detector',
    'detect_regimes',
    'detect_ensemble_regimes',

    # Regime data processing
    'RegimeDataProcessor',
    'RegimeProcessingConfig',
    'RegimeProcessingResult',
    'ProcessingMode',
    'get_regime_processor',
    'process_regime_data',
    'validate_regime_continuity',
    'analyze_regime_transitions',

    # Model evaluation utilities
    'ModelEvaluationUtilities',

    # Hyperparameter optimization
    'HyperparameterOptimization',

    # Memory optimization
    'MemoryEfficientTraining',

    # Parallel processing
    'ParallelProcessingCoordinator',

    # Model registry
    'ModelRegistry',

    # Data quality utilities
    'DataQualityUtilities',
    'detect_concept_drift',
    'analyze_feature_stability',
    'calculate_data_quality_score',
    'enhanced_automated_data_cleaning',

    # Pipeline orchestration
    'MLPipelineOrchestrator',
    '_to_jsonable',
    
    # Model interpretability
    'ModelInterpretabilityEngine',
    'ExplanationResult',
    'InterpretabilityReport',
    'InterpretabilityMethod',
    'ExplanationType',
]
