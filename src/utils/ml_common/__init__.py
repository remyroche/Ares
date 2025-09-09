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
from .cv import *
from .lookahead_protection import *
from .feature_selection import *
from .model_evaluation import *
from .hpo_utils import *
from .memory_optimization import *
from .parallel_processing import *
from .model_registry import *
from .data_quality import *
from .pipeline_orchestrator import *
from .pareto import *
from .thresholding import *
from .stability import *
from .ensembling import *
from .logging_utils import *

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
    'PurgedSplitConfig',
    'purged_time_series_splits',
    'analyze_splits',
    'validate_cv_integrity',

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

    # Pareto utilities
    'Solution',
    'DEFAULT_FINANCIAL_WEIGHTS',
    'filter_by_constraints',
    'compute_pareto_front',
    'select_knee_point',
    'compute_hypervolume',
    'scalarize_financial_goals',

    # Thresholding
    'optimize_threshold',
    'calibrate_probabilities',

    # Stability utils
    'feature_selection_stability',
    'aggregate_time_blocks',

    # Ensembling
    'simple_blend',
    'learn_blend_weights',
    'dynamic_regime_ensemble',

    # Logging utils
    'TrialLog',
    'log_trial',
    'summarize_trials',
    'start_trial_log',
    'end_trial_log',

    # Memory optimization
    'MemoryEfficientTraining',

    # Parallel processing
    'ParallelProcessingCoordinator',
    'gpu_accelerated_processing',
    'adaptive_load_balancing',
    'fault_tolerant_parallel_execution',
    'parallel_feature_engineering_gpu',

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
]
