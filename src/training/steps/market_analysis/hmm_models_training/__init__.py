"""
HMM Models Training Module

Enhanced HMM models training with comprehensive validation, error handling, and reporting.
"""

from .hmm_models_training_enhanced import (
    HMMModelsTrainingEnhanced,
    create_enhanced_hmm_models_training,
    execute_enhanced_hmm_models_training,
    TrainingMetrics,
    ModelResult
)

from .validation_framework import (
    HMMTrainingValidator,
    ValidationLevel,
    ValidationResult,
    ValidationCheck,
    ValidationReport,
    validate_hmm_training_inputs,
    validate_hmm_training_results
)

from .enhanced_reporting import (
    HMMTrainingReporter,
    PerformanceMetrics,
    ModelSummary,
    TrainingSummary,
    FeatureAnalysis,
    RegimeAnalysis,
    ComputationalMetrics,
    generate_hmm_training_report
)

from .utils import (
    StandardizedLogger,
    safe_execute,
    performance_monitor,
    ConfigurationValidator
)

from .constants import (
    ValidationThresholds,
    TrainingLimits,
    CircuitBreakerSettings,
    ModelFactorySettings,
    TemporalConsistencySettings,
    ReportingSettings,
    LoggingConstants
)

from .hmm_ensemble_training import (
    HMMEnsembleTrainingComponent,
    create_hmm_ensemble_training_component,
    execute_hmm_ensemble_training
)

# New enhanced components
from .timeframe_config import (
    TimeframeConfig,
    get_timeframe_config,
    set_timeframe_config,
    validate_timeframe_consistency,
    get_primary_timeframe,
    get_cross_timeframes,
    is_cross_timeframe_enabled
)

from .early_stopping import (
    EarlyStoppingConfig,
    EarlyStoppingMonitor,
    AggressiveOverfittingDetector,
    get_early_stopping_config,
    get_overfitting_detector
)

from .temporal_validation import (
    TemporalValidationConfig,
    TemporalValidator,
    TemporalCrossValidator,
    WalkForwardValidator,
    get_temporal_config,
    get_temporal_validator,
    get_temporal_cv,
    create_walk_forward_validator
)

from .temporal_cross_validation import (
    TemporalCVConfig,
    TimeSeriesSplit,
    TemporalCrossValidator,
    TemporalValidationPipeline,
    get_temporal_cv_config,
    get_validation_pipeline,
    create_time_series_split
)

from .enhanced_training_integration import (
    EnhancedHMMTrainingPipeline,
    demonstrate_enhanced_training
)

__all__ = [
    # Main training class
    'HMMModelsTrainingEnhanced',
    'create_enhanced_hmm_models_training',
    'execute_enhanced_hmm_models_training',
    
    # Data structures
    'TrainingMetrics',
    'ModelResult',
    'PerformanceMetrics',
    'ModelSummary',
    'TrainingSummary',
    'FeatureAnalysis',
    'RegimeAnalysis',
    'ComputationalMetrics',
    
    # Validation framework
    'HMMTrainingValidator',
    'ValidationLevel',
    'ValidationResult',
    'ValidationCheck',
    'ValidationReport',
    'validate_hmm_training_inputs',
    'validate_hmm_training_results',
    
    # Reporting
    'HMMTrainingReporter',
    'generate_hmm_training_report',
    
    # HMM Ensemble Training
    'HMMEnsembleTrainingComponent',
    'create_hmm_ensemble_training_component',
    'execute_hmm_ensemble_training',
    
    # Enhanced Components - Timeframe Configuration
    'TimeframeConfig',
    'get_timeframe_config',
    'set_timeframe_config',
    'validate_timeframe_consistency',
    'get_primary_timeframe',
    'get_cross_timeframes',
    'is_cross_timeframe_enabled',
    
    # Enhanced Components - Early Stopping & Overfitting Detection
    'EarlyStoppingConfig',
    'EarlyStoppingMonitor',
    'AggressiveOverfittingDetector',
    'get_early_stopping_config',
    'get_overfitting_detector',
    
    # Enhanced Components - Temporal Validation
    'TemporalValidationConfig',
    'TemporalValidator',
    'TemporalCrossValidator',
    'WalkForwardValidator',
    'get_temporal_config',
    'get_temporal_validator',
    'get_temporal_cv',
    'create_walk_forward_validator',
    
    # Enhanced Components - Temporal Cross-Validation
    'TemporalCVConfig',
    'TimeSeriesSplit',
    'TemporalValidationPipeline',
    'get_temporal_cv_config',
    'get_validation_pipeline',
    'create_time_series_split',
    
    # Enhanced Components - Integration
    'EnhancedHMMTrainingPipeline',
    'demonstrate_enhanced_training',
    
    # Utilities
    'StandardizedLogger',
    'safe_execute',
    'performance_monitor',
    'ConfigurationValidator',
    
    # Constants
    'ValidationThresholds',
    'TrainingLimits',
    'CircuitBreakerSettings',
    'ModelFactorySettings',
    'TemporalConsistencySettings',
    'ReportingSettings',
    'LoggingConstants'
]