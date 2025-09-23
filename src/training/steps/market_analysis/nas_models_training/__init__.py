"""
NAS Models Training Module

Enhanced NAS models training with comprehensive validation, error handling, and reporting.
Uses NAS-generated regime labels instead of HMM labels for superior regime detection.

Key Features:
- Uses NAS-generated regime labels for training
- Includes NAS as a base model for regime detection
- Enhanced feature engineering with NAS-specific features
- Comprehensive validation and reporting
- Full compatibility with existing ML training pipeline
"""

# NAS Models Training (Primary Implementation)
from .nas_models_training_enhanced import (
    StreamlinedNASTrainingStep,
    create_enhanced_nas_models_training,
    execute_enhanced_nas_models_training
)

# Global NAS Classifier - Single model for all NAS regimes
from .global_nas_classifier import (
    GlobalNASClassifier,
    GlobalNASTrainingStep,
    create_global_nas_training,
    execute_global_nas_training
)

# NAS-specific utilities
from .utils import (
    NASModelValidator,
    NASFeatureAnalyzer,
    NASRegimeAnalyzer,
    NASPerformanceMetrics,
    NASModelComparison,
    NASFeatureImportance,
    NASRegimeStability,
    NASModelPersistence,
    NASModelEvaluation,
    NASModelReporting
)

# NAS-specific configuration
from .config import (
    NASTrainingConfig,
    NASModelConfig,
    NASValidationConfig,
    NASReportingConfig,
    create_nas_training_config,
    create_nas_model_config,
    create_nas_validation_config,
    create_nas_reporting_config
)

# NAS-specific HPO
from .hpo import (
    NASHyperparameterOptimizer,
    NASModelHPO,
    NASEnsembleHPO,
    NASRegimeHPO,
    create_nas_hpo_config,
    execute_nas_hpo
)

# NAS-specific validation
from .validation import (
    NASValidationPipeline,
    NASModelValidation,
    NASRegimeValidation,
    NASFeatureValidation,
    NASPerformanceValidation,
    create_nas_validation_pipeline,
    execute_nas_validation
)

# NAS-specific reporting
from .reporting import (
    NASTrainingReporter,
    NASModelReporter,
    NASRegimeReporter,
    NASPerformanceReporter,
    NASFeatureReporter,
    create_nas_reporting_config,
    generate_nas_training_report
)

# Export main components
__all__ = [
    # Main training components
    'StreamlinedNASTrainingStep',
    'create_enhanced_nas_models_training',
    'execute_enhanced_nas_models_training',
    
    # Global classifier
    'GlobalNASClassifier',
    'GlobalNASTrainingStep',
    'create_global_nas_training',
    'execute_global_nas_training',
    
    # Configuration
    'NASTrainingConfig',
    'NASModelConfig',
    'NASValidationConfig',
    'NASReportingConfig',
    'create_nas_training_config',
    'create_nas_model_config',
    'create_nas_validation_config',
    'create_nas_reporting_config',
    
    # HPO
    'NASHyperparameterOptimizer',
    'NASModelHPO',
    'NASEnsembleHPO',
    'NASRegimeHPO',
    'create_nas_hpo_config',
    'execute_nas_hpo',
    
    # Validation
    'NASValidationPipeline',
    'NASModelValidation',
    'NASRegimeValidation',
    'NASFeatureValidation',
    'NASPerformanceValidation',
    'create_nas_validation_pipeline',
    'execute_nas_validation',
    
    # Reporting
    'NASTrainingReporter',
    'NASModelReporter',
    'NASRegimeReporter',
    'NASPerformanceReporter',
    'NASFeatureReporter',
    'create_nas_reporting_config',
    'generate_nas_training_report',
    
    # Utilities
    'NASModelValidator',
    'NASFeatureAnalyzer',
    'NASRegimeAnalyzer',
    'NASPerformanceMetrics',
    'NASModelComparison',
    'NASFeatureImportance',
    'NASRegimeStability',
    'NASModelPersistence',
    'NASModelEvaluation',
    'NASModelReporting'
]