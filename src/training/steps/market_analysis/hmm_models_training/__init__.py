"""
HMM Models Training Module

Enhanced HMM models training with comprehensive validation, error handling, and reporting.

New streamlined approach using common_utils/ ML training pipeline.
"""

# Streamlined HMM Training (Recommended) - Complete migration to common_utils pipeline
# Now available directly in hmm_models_training_enhanced.py

# Streamlined HMM Training (Recommended) - Complete migration to common_utils pipeline
from .hmm_models_training_enhanced import (
    StreamlinedHMMTrainingStep,
    create_enhanced_hmm_models_training,
    execute_enhanced_hmm_models_training
)

# Validation framework functionality moved to ml_commons HMMValidationPipeline
# from .validation_framework import (  # DEPRECATED - use ml_commons validation instead
#     HMMTrainingValidator,
#     ValidationLevel,
#     ValidationResult,
#     ValidationCheck,
#     ValidationReport,
#     validate_hmm_training_inputs,
#     validate_hmm_training_results
# )

# Enhanced reporting functionality moved to ml_commons HMM evaluation pipeline
# from .enhanced_reporting import (  # DEPRECATED - use ml_commons evaluation instead
#     HMMTrainingReporter,
#     PerformanceMetrics,
#     ModelSummary,
#     TrainingSummary,
#     FeatureAnalysis,
#     RegimeAnalysis,
#     ComputationalMetrics,
#     generate_hmm_training_report
# )

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
# Re-export timeframe utilities from ml_common universal config
from src.utils.ml_common.config.universal_timeframe_config import (
    UniversalTimeframeConfig as TimeframeConfig,
    get_timeframe_manager as get_timeframe_config,
    set_timeframe_config,
    validate_timeframe_consistency,
    get_primary_timeframe,
    get_cross_timeframes,
    is_cross_timeframe_enabled
)

# Deprecated local early_stopping; use ml_common training utils if needed.
try:
    from src.utils.ml_common.training.enhanced_training_utils import (
        EarlyStoppingConfig,
        OverfittingMonitorConfig as AggressiveOverfittingDetector
    )
    def get_early_stopping_config():
        return EarlyStoppingConfig()
    def get_overfitting_detector():
        from src.utils.ml_common.validation.enhanced_overfitting_detection import get_overfitting_detector as _g
        return _g()
except Exception:
    # Keep names available but minimal to avoid import errors
    class EarlyStoppingConfig: ...
    class AggressiveOverfittingDetector: ...
    def get_early_stopping_config(): return EarlyStoppingConfig()
    def get_overfitting_detector(): return None

# Deprecated local temporal_validation; proxy to ml_common if available
try:
    from src.utils.ml_common.validation.temporal_validation import (
        TemporalValidationConfig,
        TemporalValidator,
        TemporalCrossValidator,
        WalkForwardValidator
    )
    def get_temporal_config(): return TemporalValidationConfig()
    def get_temporal_validator(): return TemporalValidator()
    def get_temporal_cv(): return TemporalCrossValidator()
    def create_walk_forward_validator(): return WalkForwardValidator()
except Exception:
    class TemporalValidationConfig: ...
    class TemporalValidator: ...
    class TemporalCrossValidator: ...
    class WalkForwardValidator: ...
    def get_temporal_config(): return TemporalValidationConfig()
    def get_temporal_validator(): return TemporalValidator()
    def get_temporal_cv(): return TemporalCrossValidator()
    def create_walk_forward_validator(): return WalkForwardValidator()

# Deprecated local temporal_cross_validation; proxy to ml_common if available
try:
    from src.utils.ml_common.validation.temporal_cross_validation import (
        TemporalCVConfig,
        TimeSeriesSplit,
        TemporalCrossValidator,
        TemporalValidationPipeline
    )
    def get_temporal_cv_config(): return TemporalCVConfig()
    def get_validation_pipeline(): return TemporalValidationPipeline()
    def create_time_series_split(): return TimeSeriesSplit()
except Exception:
    class TemporalCVConfig: ...
    class TimeSeriesSplit: ...
    class TemporalCrossValidator: ...
    class TemporalValidationPipeline: ...
    def get_temporal_cv_config(): return TemporalCVConfig()
    def get_validation_pipeline(): return TemporalValidationPipeline()
    def create_time_series_split(): return TimeSeriesSplit()

# Remove broken local re-exports; optional alias to ml_common if available
try:
    from src.utils.ml_common.training.training_integration import TrainingStepEnhancer as EnhancedHMMTrainingPipeline
    def demonstrate_enhanced_training():
        return "Enhanced training demonstration is available via ml_common."
except Exception:
    class EnhancedHMMTrainingPipeline: ...
    def demonstrate_enhanced_training(): return ""

# Enhanced Components - Overfitting Reporting
from .overfitting_reporting import (
    OverfittingReport,
    OverfittingTrend,
    OverfittingReporter,
    get_overfitting_reporter,
    create_overfitting_reporter
)

__all__ = [
    # Streamlined HMM Training (Recommended) - Complete migration to common_utils pipeline
    'StreamlinedHMMTrainingStep',
    'create_enhanced_hmm_models_training',
    'execute_enhanced_hmm_models_training',
    
    # Data structures - DEPRECATED: use ml_commons evaluation pipeline instead
    # 'TrainingMetrics',
    # 'ModelResult',
    # 'PerformanceMetrics',
    # 'ModelSummary',
    # 'TrainingSummary',
    # 'FeatureAnalysis',
    # 'RegimeAnalysis',
    # 'ComputationalMetrics',

    # Validation framework - DEPRECATED: use ml_commons HMMValidationPipeline instead
    # 'HMMTrainingValidator',
    # 'ValidationLevel',
    # 'ValidationResult',
    # 'ValidationCheck',
    # 'ValidationReport',
    # 'validate_hmm_training_inputs',
    # 'validate_hmm_training_results',

    # Reporting - DEPRECATED: use ml_commons HMM evaluation pipeline instead
    # 'HMMTrainingReporter',
    # 'generate_hmm_training_report',
    
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
    
    # Enhanced Components - Overfitting Reporting
    'OverfittingReport',
    'OverfittingTrend',
    'OverfittingReporter',
    'get_overfitting_reporter',
    'create_overfitting_reporter',
    
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