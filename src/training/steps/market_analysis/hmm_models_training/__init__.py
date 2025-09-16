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

from .hmm_ensemble_training import (
    HMMEnsembleTrainingComponent,
    create_hmm_ensemble_training_component,
    execute_hmm_ensemble_training
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
    'execute_hmm_ensemble_training'
]