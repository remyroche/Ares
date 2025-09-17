"""
Regime Data Splitting Package.

This package provides comprehensive regime data splitting functionality with enhanced
error handling, validation, and reporting capabilities.

Components:
- component: Main regime data splitting component with comprehensive error handling
- enhanced: Enhanced implementation with HMM ML model integration
- main: Main step implementation with standardized data quality management
- validator: Comprehensive validation framework for regime data splitting

Key Features:
- Silent failure prevention with explicit validation
- Comprehensive reporting with quality scores and recommendations
- Enhanced error handling with detailed context
- Multi-stage validation checkpoints
- Actionable insights for continuous improvement
"""

from .component import (
    RegimeDataSplittingComponent,
    RegimeSplittingStatus,
    RegimeSplittingMetrics,
    RegimeSplittingReport
)

from .enhanced import (
    RegimeDataSplittingEnhanced,
    HMMRegimeTagger,
    execute_enhanced_regime_data_splitting
)

from .main import (
    RegimeDataSplittingStep,
    RegimeDataResult,
    StepResult,
    StepResultStatus
)

from .validator import (
    Step4RegimeDataSplittingValidator,
    run_validator
)

from .validation_utils import (
    StandardizedValidator,
    ValidationResult,
    ValidationErrorType,
    get_validator,
    validate_training_input,
    validate_pipeline_state
)

from .config_utils import (
    RegimeDataSplittingConfig,
    PathManager,
    ConfigManager,
    get_config_manager,
    get_path_manager
)

__all__ = [
    # Component classes
    'RegimeDataSplittingComponent',
    'RegimeDataSplittingEnhanced',
    'RegimeDataSplittingStep',
    'HMMRegimeTagger',
    
    # Data classes
    'RegimeSplittingStatus',
    'RegimeSplittingMetrics',
    'RegimeSplittingReport',
    'RegimeDataResult',
    'StepResult',
    'StepResultStatus',
    
    # Validator classes
    'Step4RegimeDataSplittingValidator',
    
    # Functions
    'execute_enhanced_regime_data_splitting',
    'run_validator',
    
    # Validation utilities
    'StandardizedValidator',
    'ValidationResult',
    'ValidationErrorType',
    'get_validator',
    'validate_training_input',
    'validate_pipeline_state',
    
    # Configuration utilities
    'RegimeDataSplittingConfig',
    'PathManager',
    'ConfigManager',
    'get_config_manager',
    'get_path_manager'
]

__version__ = "1.0.0"
__author__ = "Market Analysis Team"
__description__ = "Comprehensive regime data splitting with enhanced error handling and reporting"