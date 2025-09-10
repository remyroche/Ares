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
from .model_evaluation import *
from .hpo_utils import *
from .memory_optimization import *
# Expose coordinator class but not its instance methods as globals
from .parallel_processing import ParallelProcessingCoordinator
from .model_registry import *
from .data_quality import *
from .pipeline_orchestrator import *

# Trading-specific meta-learning HPO
from .trading_meta_features import TradingMetaFeaturesExtractor
from .meta_learning_trading_hpo import MetaLearningTradingHPO, TradingOptimizationHistoryDB
from .trading_optimization_strategies import (
    TradingOptimizationOrchestrator, 
    TradingOptimizationStrategy,
    RegimeAwareOptimization,
    RiskConstrainedOptimization,
    LeverageAdaptiveOptimization
)
from .trading_hpo_examples import TradingHPOExamples, run_all_trading_examples

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

    # Trading-specific meta-learning HPO
    'TradingMetaFeaturesExtractor',
    'MetaLearningTradingHPO',
    'TradingOptimizationHistoryDB',
    'TradingOptimizationOrchestrator',
    'TradingOptimizationStrategy',
    'RegimeAwareOptimization',
    'RiskConstrainedOptimization',
    'LeverageAdaptiveOptimization',
    'TradingHPOExamples',
    'run_all_trading_examples',
]
