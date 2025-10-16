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

from src.utils.tprint import tprint
tprint('📦 Loading Regime Data Splitting Package')

tprint('📦 Importing component module')
from .regime_data_splitting_component import (
    RegimeDataSplittingComponent,
    RegimeSplittingStatus,
    RegimeSplittingMetrics,
    RegimeSplittingReport
)
tprint('✅ Component module imported')

tprint('📦 Importing NAS/TAS regime data splitting module')
from .nas_tas_regime_data_splitting import (
    NasTasRegimeDataSplitting,
    HMMRegimeTagger,
    execute_nas_tas_regime_data_splitting
)
tprint('✅ NAS/TAS regime data splitting module imported')

tprint('📦 Importing main module')
from .regime_data_splitting_main import (
    RegimeDataSplittingStep,
    RegimeDataResult,
    StepResult,
    StepResultStatus
)
tprint('✅ Main module imported')

tprint('📦 Importing validator module')
from .validator import (
    Step4RegimeDataSplittingValidator,
    run_validator
)
tprint('✅ Validator module imported')

tprint('📦 Importing validation utilities')
from .validation_utils import (
    StandardizedValidator,
    ValidationResult,
    ValidationErrorType,
    get_validator,
    validate_training_input,
    validate_pipeline_state
)
tprint('✅ Validation utilities imported')

tprint('📦 Importing configuration utilities')
from .config_utils import (
    RegimeDataSplittingConfig,
    PathManager,
    ConfigManager,
    get_config_manager,
    get_path_manager
)
tprint('✅ Configuration utilities imported')

__all__ = [
    # Component classes
    'RegimeDataSplittingComponent',
    'NasTasRegimeDataSplitting',
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
    'execute_nas_tas_regime_data_splitting',
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

tprint('✅ Regime Data Splitting Package loaded successfully')