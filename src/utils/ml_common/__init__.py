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

# Import modules with error handling to avoid circular dependencies
try:
    from .base_safeguards import *
except ImportError:
    pass

try:
    from .cv_utils import *
except ImportError:
    pass

try:
    from .lookahead_protection import *
except ImportError:
    pass

try:
    from .feature_selection import *
except ImportError:
    pass

# Feature generation optimization moved to src/feature_engineering/
try:
    from .data_labeling import *
except ImportError:
    pass

try:
    from .hmm_regime_detection import *
except ImportError:
    pass

try:
    from .regime_data_processing import *
except ImportError:
    pass

try:
    from .model_evaluation import *
except ImportError:
    pass

try:
    from .hpo_utils import *
except ImportError:
    pass

try:
    from .memory_optimization import *
except ImportError:
    pass

# Expose coordinator class but not its instance methods as globals
try:
    from .parallel_processing import ParallelProcessingCoordinator
except ImportError:
    ParallelProcessingCoordinator = None

try:
    from .model_registry import *
except ImportError:
    pass

try:
    from .model_explainability import *
except ImportError as e:
    tprint(f"⚠️ Model explainability import failed (likely due to circular import): {e}")
    pass

try:
    from .data_quality import *
except ImportError:
    pass

try:
    from .pipeline_orchestrator import *
except ImportError:
    pass

try:
    from .model_interpretability import *
except ImportError:
    pass

__version__ = "1.0.0"

# Build __all__ dynamically to handle conditional imports
__all__ = []

# Base safeguards
__all__.extend([
    'MLTrainingSafeguards',
    'MLTrainingError',
    'ClassImbalanceError',
    'SingleClassError',
    'DataQualityError',
    'SmartFastFailHandler',
])

# Cross-validation utilities
__all__.extend([
    'CrossValidationUtilities',
])

# Lookahead bias protection
__all__.extend([
    'LookaheadProtection',
    'advanced_information_barrier_checks',
    'validate_feature_timestamp_alignment',
    'automated_future_data_filtering',
    'rolling_window_bias_validation',
])

# Feature selection framework
__all__.extend([
    'FeatureSelectionFramework',
])

# Data labeling utilities
__all__.extend([
    'DataLabelingUtilities',
    'TripleBarrierConfig',
    'LabelingResult',
    'LabelingMethod',
    'get_data_labeler',
    'label_triple_barrier',
    'label_regime_aware',
])

# HMM regime detection
__all__.extend([
    'HMMRegimeDetector',
    'HMMRegimeConfig',
    'RegimeDetectionResult',
    'RegimeDetectionMethod',
    'get_hmm_regime_detector',
    'detect_regimes',
    'detect_ensemble_regimes',
])

# Regime data processing
__all__.extend([
    'RegimeDataProcessor',
    'RegimeProcessingConfig',
    'RegimeProcessingResult',
    'ProcessingMode',
    'get_regime_processor',
    'process_regime_data',
    'validate_regime_continuity',
    'analyze_regime_transitions',
])

# Model evaluation utilities
__all__.extend([
    'ModelEvaluationUtilities',
])

# Hyperparameter optimization
__all__.extend([
    'HyperparameterOptimization',
])

# Memory optimization
__all__.extend([
    'MemoryEfficientTraining',
])

# Parallel processing
__all__.extend([
    'ParallelProcessingCoordinator',
])

# Model registry
__all__.extend([
    'ModelRegistry',
])

# Model explainability (conditionally add if import succeeded)
try:
    from .model_explainability import ModelExplainabilityManager, ModelExplanationResult, with_explainability, explain_model_quick
    __all__.extend([
        'ModelExplainabilityManager',
        'ModelExplanationResult',
        'with_explainability',
        'explain_model_quick',
    ])
    tprint("✅ Model explainability components loaded successfully")
except (ImportError, NameError) as e:
    tprint(f"⚠️ Model explainability components not available: {e}")

# Data quality utilities
__all__.extend([
    'DataQualityUtilities',
    'detect_concept_drift',
    'analyze_feature_stability',
    'calculate_data_quality_score',
    'enhanced_automated_data_cleaning',
])

# Pipeline orchestration
__all__.extend([
    'MLPipelineOrchestrator',
    '_to_jsonable',
])

# Model interpretability
__all__.extend([
    'ModelInterpretabilityEngine',
    'ExplanationResult',
    'InterpretabilityReport',
    'InterpretabilityMethod',
    'ExplanationType',
])
