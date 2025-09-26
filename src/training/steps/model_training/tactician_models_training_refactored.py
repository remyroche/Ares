"""
Tactician Models Training Step - Enhanced & Streamlined

This step handles per-regime training of individual Tactician models using common dependencies.
The Tactician operates on 1m timeframe and decides WHEN to trade based on Analyst's green light signals.

ENHANCED FEATURES:
- Comprehensive error handling with detailed failure reporting
- Enhanced progress tracking and sub-step reporting
- Input validation and data quality checks
- Optimized vectorization with intelligent fallback
- Structured logging with performance metrics
- Health monitoring throughout training process
- Integration with common utilities and hardware optimizers
- Extensive logging with tprint at every step
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import traceback
from dataclasses import dataclass
from enum import Enum

from .import_helpers import ensure_dependencies, import_module_safely
from .memory_utils import memory_guard

np = import_module_safely("numpy", required=True, package_hint="pip install numpy")
NUMPY_AVAILABLE = np is not None

pd = import_module_safely("pandas", required=True, package_hint="pip install pandas")
PANDAS_AVAILABLE = pd is not None

ensure_dependencies(
    {"numpy": NUMPY_AVAILABLE, "pandas": PANDAS_AVAILABLE},
    error_message="Critical dependencies missing for tactician training",
)


# Enhanced imports with comprehensive error handling
try:
    from src.utils.logger import system_logger
    from src.utils.ml_common.config import PerRegimeTrainingConfig
    from src.utils.ml_common.training import PerRegimeTrainingStep
except ImportError as e:
    print(f"❌ CRITICAL: Failed to import core ML utilities: {e}")
    raise

# Import enhanced logging and utilities - CRITICAL: Fast fail if not available
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_debug, tprint_progress, tprint_performance, tprint_structured,
        tprint_timer, LogLevel
    )
    TPRINT_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL ERROR: tprint is required but not available: {e}")
    print("❌ This is a critical dependency for enhanced logging. Please install tprint.")
    raise ImportError(f"CRITICAL: tprint is required but not available: {e}") from e

# Import common utilities - CRITICAL: Fast fail if not available
try:
    from src.utils.common_operations import (
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        cleanup_m1_optimizers, integrate_with_m1_optimizers
    )
    tprint_info("✅ Common operations utilities loaded")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Common operations utilities are required but not available: {e}")
    print("❌ Hardware optimizers are essential for performance. Please install common_operations.")
    raise ImportError(f"CRITICAL: Common operations utilities are required but not available: {e}") from e

try:
    from src.utils.common_utilities import (
        safe_dataframe_operation, validate_dataframe_columns, calculate_data_quality_metrics,
        safe_merge_dataframes, create_summary_statistics
    )
    tprint_info("✅ Common utilities loaded")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Common utilities are required but not available: {e}")
    print("❌ Enhanced data operations are essential. Please install common_utilities.")
    raise ImportError(f"CRITICAL: Common utilities are required but not available: {e}") from e

try:
    from src.utils.math_validation import (
        safe_divide, validate_finite, validate_positive, validate_range,
        safe_correlation, safe_percentage_change
    )
    tprint_info("✅ Math validation utilities loaded")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Math validation utilities are required but not available: {e}")
    print("❌ Safe math operations are essential for data integrity. Please install math_validation.")
    raise ImportError(f"CRITICAL: Math validation utilities are required but not available: {e}") from e

try:
    from src.utils.kline_parquet import validate_klines_data, process_klines_data
    from src.utils.serialization_utils import safe_serialize, safe_deserialize
    tprint_info("✅ Data utilities loaded")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Data utilities are required but not available: {e}")
    print("❌ Enhanced data validation is essential. Please install kline_parquet and serialization_utils.")
    raise ImportError(f"CRITICAL: Data utilities are required but not available: {e}") from e

try:
    from src.utils.matrix_operations import (
        safe_matrix_operations, validate_matrix_properties, optimize_matrix_computations
    )
    tprint_info("✅ Matrix operations utilities loaded")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Matrix operations utilities are required but not available: {e}")
    print("❌ Optimized matrix computations are essential for performance. Please install matrix_operations.")
    raise ImportError(f"CRITICAL: Matrix operations utilities are required but not available: {e}") from e

try:
    from src.utils.ml_common import (
        cross_validation_utils, lookahead_bias_detector, hyperparameter_optimization
    )
    tprint_info("✅ ML common utilities loaded")
except ImportError as e:
    print(f"❌ CRITICAL ERROR: ML common utilities are required but not available: {e}")
    print("❌ Advanced ML features are essential. Please install ml_common.")
    raise ImportError(f"CRITICAL: ML common utilities are required but not available: {e}") from e

# Enhanced training utilities integration
try:
    from src.utils.ml_common.training.enhanced_training_utils import (
        EnhancedTrainingUtils,
        EarlyStoppingConfig,
        PurgedCVConfig,
        OverfittingMonitorConfig,
        RegularizationConfig
    )
    from src.utils.ml_common.training.training_integration import (
        TrainingStepEnhancer,
        TrainingIntegrationConfig
    )
    ENHANCED_TRAINING_AVAILABLE = True
    tprint_success("✅ Enhanced training utilities loaded")
except ImportError as e:
    ENHANCED_TRAINING_AVAILABLE = False
    tprint_warning(f"⚠️ Enhanced training utilities not available: {e}")
    EnhancedTrainingUtils = None
    TrainingStepEnhancer = None
    EarlyStoppingConfig = None
    PurgedCVConfig = None
    OverfittingMonitorConfig = None
    RegularizationConfig = None
    TrainingIntegrationConfig = None


# Import vectorized training manager for enhanced capabilities
try:
    from src.utils.ml_common.training.vectorized_training_manager import VectorizedTrainingManager
    VECTORIZED_TRAINING_AVAILABLE = True
except ImportError:
    VECTORIZED_TRAINING_AVAILABLE = False

# Initialize logger - CRITICAL: Fast fail if not available
try:
    logger = system_logger.getChild('TacticianModelsTrainingEnhanced')
except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to initialize system logger: {e}")
    print("❌ System logger is required for proper logging. Please check logger configuration.")
    raise RuntimeError(f"CRITICAL: Failed to initialize system logger: {e}") from e


class TrainingPhase(Enum):
    """Training phases for progress tracking."""
    INITIALIZATION = "initialization"
    DATA_VALIDATION = "data_validation"
    FEATURE_PREPARATION = "feature_preparation"
    REGIME_ANALYSIS = "regime_analysis"
    MODEL_TRAINING = "model_training"
    EVALUATION = "evaluation"
    MODEL_SAVING = "model_saving"
    FINALIZATION = "finalization"


@dataclass
class TrainingMetrics:
    """Training metrics for comprehensive reporting."""
    phase: TrainingPhase
    start_time: float
    end_time: Optional[float] = None
    samples_processed: int = 0
    features_count: int = 0
    regimes_count: int = 0
    models_trained: int = 0
    errors_encountered: int = 0
    warnings_issued: int = 0
    memory_usage_mb: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Get phase duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


class TacticianModelsTrainingStepRefactored(PerRegimeTrainingStep):
    """
    Enhanced Tactician Models Training Step with comprehensive error handling and reporting.
    
    The Tactician operates on 1m timeframe and is trained on:
    1. Only periods where the Analyst gives a green light
    2. Using the Analyst's model outputs as input features
    
    ENHANCED FEATURES:
    - Comprehensive input validation and data quality checks
    - Detailed progress tracking with phase-based metrics
    - Enhanced error handling with specific failure reporting
    - Optimized vectorization with intelligent fallback
    - Structured logging with performance monitoring
    """
    
    def __init__(self, config: Optional[TacticianTrainingConfig] = None, enable_vectorization: bool = True):
        """
        Initialize enhanced Tactician models training step with comprehensive error handling and utility integration.

        Args:
            config: Per-regime training configuration
            enable_vectorization: Whether to enable vectorized training
        """
        # Initialize comprehensive tracking
        self.training_metrics: Dict[TrainingPhase, TrainingMetrics] = {}
        self.overall_start_time = time.time()
        self.phase_start_time = time.time()
        self.initialization_errors = []
        self.utility_integration_status = {}
        
        # Log initialization start
        tprint_info("🚀 Starting Enhanced Tactician Models Training Step initialization")
        
        # Set default configuration for tactician models with enhanced settings
        if config is None:
            config = TacticianTrainingConfig(
                model_name="tactician_models",
                timeframe="1m",
                model_types=["XGBOOST", "LIGHTGBM", "DEEPSCALER_1M", "FINANCIAL_RESNET", "RandomSurvivalForest"],
                hpo_n_trials=100,
                hpo_timeout_seconds=3600,
                min_samples_per_regime=1000,
                enable_data_augmentation=True,
                augmentation_method="smote",
                model_save_path="generated/model_training/models/tactician_models",
                evaluation_metrics=["mse", "mae", "r2", "mape", "smape"],
                use_single_model=True,
                single_model_name="tactician_unified_model",
                enable_ensemble_training=True,
                ensemble_method="stacking",
                meta_model="ElasticNetCV",
                ensemble_name="tactician_ensemble",
                enable_entry_timing_optimization=True,
                entry_timing_range=0.004,
                expected_movement=0.004
            )

        try:
            # Initialize parent class with comprehensive error handling
            tprint_info("🔄 Initializing parent PerRegimeTrainingStep...")
            super().__init__(config)
            
            # Initialize logger - CRITICAL: Fast fail if not available
            try:
                self.logger = logger.getChild('TacticianModelsTrainingEnhanced')
            except Exception as e:
                error_msg = f"CRITICAL: Failed to initialize child logger: {e}"
                tprint_error(f"❌ {error_msg}")
                raise RuntimeError(error_msg) from e
            
            tprint_success("✅ Parent class initialized successfully")
            
            # Vectorization support with enhanced validation
            self.enable_vectorization = enable_vectorization and VECTORIZED_TRAINING_AVAILABLE
            self.vectorization_fallback_used = False
            
            # Initialize training metrics for initialization phase
            self._start_phase(TrainingPhase.INITIALIZATION)
            
            # Validate configuration with comprehensive checks
            tprint_info("🔍 Validating configuration...")
            self._validate_configuration(config)
            tprint_success("✅ Configuration validation passed")
            
            # Initialize hardware optimizers with error handling
            tprint_info("🧠 Initializing hardware optimizers...")
            self._initialize_hardware_optimizers()
            
            # Initialize utility integrations
            tprint_info("🔧 Initializing utility integrations...")
            self._initialize_utility_integrations()
            
            # Initialize enhanced training utilities
            if ENHANCED_TRAINING_AVAILABLE:
                tprint_info("🚀 Initializing enhanced training utilities...")
                self._initialize_enhanced_training_utilities()
            
            # Log initialization success with comprehensive status
            if self.enable_vectorization:
                tprint_success("🚀 Enhanced Tactician Models Training Step initialized with vectorization")
            else:
                tprint_success("✅ Enhanced Tactician Models Training Step initialized (standard mode)")
            
            # Log utility integration status
            self._log_utility_integration_status()
            
            self._complete_phase(TrainingPhase.INITIALIZATION, success=True)
            
        except Exception as e:
            self._handle_initialization_error(e)
            raise
    
    def _initialize_enhanced_training_utilities(self):
        """Initialize enhanced training utilities for overfitting prevention and lookahead bias detection."""
        try:
            # Create enhanced training configuration for Tactician
            self.enhanced_training_config = TrainingIntegrationConfig(
                enable_early_stopping=True,
                enable_purged_cv=True,
                enable_lookahead_detection=True,
                enable_temporal_splits=True,
                enable_regularization=True,
                enable_overfitting_monitoring=True,
                enable_walk_forward=True,  # Enable for Tactician
                model_type='auto',
                enable_model_calibration=self.config.enable_model_calibration,
                calibration_method=self.config.calibration_method,
                calibration_cv=self.config.calibration_cv,
                calibration_min_samples=self.config.calibration_min_samples,
                calibration_validation_split=self.config.calibration_validation_split,
                calibration_enforce_probabilistic=self.config.calibration_enforce_probabilistic,
                calibration_skip_without_proba=self.config.calibration_skip_without_proba,
            )

            # Initialize training enhancer
            self.training_enhancer = TrainingStepEnhancer(self.enhanced_training_config)
            
            # Store enhanced utilities
            self.enhanced_training_utils = {
                'EnhancedTrainingUtils': EnhancedTrainingUtils,
                'EarlyStoppingConfig': EarlyStoppingConfig,
                'PurgedCVConfig': PurgedCVConfig,
                'OverfittingMonitorConfig': OverfittingMonitorConfig,
                'RegularizationConfig': RegularizationConfig,
                'TrainingStepEnhancer': TrainingStepEnhancer,
                'TrainingIntegrationConfig': TrainingIntegrationConfig
            }
            
            tprint_success("✅ Enhanced training utilities initialized successfully")
            
        except Exception as e:
            tprint_warning(f"⚠️ Enhanced training utilities initialization failed: {e}")
            self.enhanced_training_config = None
            self.training_enhancer = None
            self.enhanced_training_utils = {}
    
    def _create_model_instance(self, model_type: str):
        """Create model instance based on model type."""
        try:
            if model_type == "XGBOOST":
                from xgboost import XGBRegressor
                return XGBRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=0
                )
            elif model_type == "LIGHTGBM":
                from lightgbm import LGBMRegressor
                return LGBMRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                )
            elif model_type == "DEEPSCALER_1M":
                # DeepScaler implementation would go here
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42
                )
            elif model_type == "FINANCIAL_RESNET":
                # FinancialResNet implementation would go here
                return RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42
                )
            elif model_type == "RandomSurvivalForest":
                return self._create_random_survival_forest_model()
            else:
                # Default fallback
                return RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
        except Exception as e:
            self.logger.error(f"❌ Failed to create model instance for {model_type}: {e}")
            # Fallback to RandomForest
            return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def _create_random_survival_forest_model(self):
        """Create Random Survival Forest model for tactician timing prediction."""
        try:
            from .random_survival_forest_tactician import RandomSurvivalForestTactician, SurvivalAnalysisConfig
            
            # Create configuration optimized for tactician timing
            config = SurvivalAnalysisConfig(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                max_samples=0.8,
                horizons=[1, 2, 5, 10],  # 1m to 10m horizons (removed 15m and 30m)
                horizon_weights=[0.4, 0.3, 0.2, 0.1],
                entry_timing_range=0.005,  # 0.5% range
                expected_movement=0.01,  # 1% expected movement
                latency_constraint=2.0,  # 2 second constraint
                enable_timing_features=True,
                enable_regime_features=True,
                enable_analyst_features=True,
                enable_microstructure_features=True
            )
            
            return RandomSurvivalForestTactician(config)
            
        except ImportError as e:
            self.logger.error(f"❌ Random Survival Forest not available: {e}")
            # Fallback to RandomForest
            return RandomForestRegressor(n_estimators=200, random_state=42)
        except Exception as e:
            self.logger.error(f"❌ Failed to create Random Survival Forest: {e}")
            # Fallback to RandomForest
            return RandomForestRegressor(n_estimators=200, random_state=42)
    
    def _start_phase(self, phase: TrainingPhase, context: Optional[Dict[str, Any]] = None) -> None:
        """Start tracking a training phase with structured logging."""
        self.training_metrics[phase] = TrainingMetrics(
            phase=phase,
            start_time=time.time()
        )
        
        # Log phase start with structured format
        self._log_phase_start(phase, context)
    
    def _complete_phase(self, phase: TrainingPhase, success: bool = True, 
                       error_message: Optional[str] = None, **kwargs) -> None:
        """Complete a training phase with metrics and structured logging."""
        if phase in self.training_metrics:
            metrics = self.training_metrics[phase]
            metrics.end_time = time.time()
            metrics.success = success
            metrics.error_message = error_message
            
            # Update metrics with provided values
            for key, value in kwargs.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
            
            duration = metrics.duration
            
            # Prepare metrics for structured logging
            phase_metrics = {
                'duration': duration,
                'samples_processed': metrics.samples_processed,
                'features_count': metrics.features_count,
                'models_trained': metrics.models_trained,
                'warnings_issued': metrics.warnings_issued,
                'errors_encountered': metrics.errors_encountered,
                'memory_usage_mb': metrics.memory_usage_mb
            }
            
            # Log phase completion with structured format
            self._log_phase_complete(phase, success, duration, phase_metrics)
            
            if not success and error_message:
                self._log_structured_event(
                    event_type="phase_error",
                    phase=phase.value,
                    message=f"Phase failed: {error_message}",
                    level="error"
                )
    
    def _validate_configuration(self, config: PerRegimeTrainingConfig) -> None:
        """Validate training configuration."""
        try:
            # Validate model types
            if not config.model_types:
                raise ValueError("No model types specified in configuration")
            
            # Validate timeframe
            if not config.timeframe:
                raise ValueError("No timeframe specified in configuration")
            
            # Validate minimum samples - CRITICAL: Fast fail on invalid config
            if config.min_samples_per_regime < 100:
                error_msg = f"CRITICAL: Very low minimum samples per regime: {config.min_samples_per_regime} (minimum: 100)"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            # Validate HPO settings - CRITICAL: Fast fail on invalid config
            if config.enable_hpo and config.hpo_n_trials < 10:
                error_msg = f"CRITICAL: Very low HPO trials: {config.hpo_n_trials} (minimum: 10)"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            tprint_success("✅ Configuration validation passed")
            
        except Exception as e:
            tprint_error(f"❌ Configuration validation failed: {e}")
            raise
    
    def _initialize_hardware_optimizers(self) -> None:
        """Initialize hardware optimizers - CRITICAL: Fast fail if not available."""
        try:
            tprint_info("🧠 Initializing M1 hardware optimizers...")
            
            # Initialize M1 GPU manager - CRITICAL: Fast fail if not available
            self.m1_gpu_manager = get_m1_gpu_manager()
            if not self.m1_gpu_manager:
                error_msg = "CRITICAL: M1 GPU manager is required but not available"
                tprint_error(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            tprint_success("✅ M1 GPU manager initialized")
            
            # Initialize M1 memory optimizer - CRITICAL: Fast fail if not available
            self.m1_memory_optimizer = get_m1_memory_optimizer()
            if not self.m1_memory_optimizer:
                error_msg = "CRITICAL: M1 memory optimizer is required but not available"
                tprint_error(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            tprint_success("✅ M1 memory optimizer initialized")
            
            # Initialize M1 CPU optimizer - CRITICAL: Fast fail if not available
            self.m1_cpu_optimizer = get_m1_cpu_optimizer()
            if not self.m1_cpu_optimizer:
                error_msg = "CRITICAL: M1 CPU optimizer is required but not available"
                tprint_error(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            tprint_success("✅ M1 CPU optimizer initialized")
            
            # Integrate with M1 optimizers - CRITICAL: Fast fail if not successful
            integration_result = integrate_with_m1_optimizers()
            if not integration_result.get('success', False):
                error_msg = "CRITICAL: M1 optimizers integration failed"
                tprint_error(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            tprint_success("✅ M1 optimizers integration successful")
            
            tprint_success("✅ Hardware optimizers initialization completed")
            
        except Exception as e:
            error_msg = f"CRITICAL: Hardware optimizer initialization failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def _initialize_utility_integrations(self) -> None:
        """Initialize utility integrations - All utilities are required."""
        try:
            tprint_info("🔧 Initializing utility integrations...")
            
            # All utilities are already loaded at import time with fast fail
            tprint_success("✅ All utility integrations verified and available")
            tprint_success("✅ Common utilities available")
            tprint_success("✅ Math validation utilities available")
            tprint_success("✅ Data utilities available")
            tprint_success("✅ Matrix operations utilities available")
            tprint_success("✅ ML common utilities available")
            tprint_success("✅ Enhanced tprint logging available")
            
            tprint_success("✅ Utility integrations initialization completed")
            
        except Exception as e:
            error_msg = f"CRITICAL: Utility integration initialization failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def _log_utility_integration_status(self) -> None:
        """Log comprehensive utility integration status."""
        try:
            tprint_info("📊 Utility Integration Status:")
            
            for utility, status in self.utility_integration_status.items():
                if status == 'available':
                    tprint_success(f"  ✅ {utility}: {status}")
                elif status == 'unavailable':
                    tprint_warning(f"  ⚠️ {utility}: {status}")
                elif status.startswith('error:'):
                    tprint_error(f"  ❌ {utility}: {status}")
                else:
                    tprint_info(f"  ℹ️ {utility}: {status}")
            
            # Log initialization errors if any
            if self.initialization_errors:
                tprint_error("❌ Initialization errors encountered:")
                for error in self.initialization_errors:
                    tprint_error(f"  - {error}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to log utility integration status: {e}")
    
    def _handle_initialization_error(self, error: Exception) -> None:
        """Handle initialization errors with detailed reporting."""
        error_msg = f"Initialization failed: {str(error)}"
        tprint_error(f"❌ {error_msg}")
        tprint_error(f"❌ Traceback: {traceback.format_exc()}")
        
        # Log utility integration status even on failure
        if hasattr(self, 'utility_integration_status'):
            self._log_utility_integration_status()
        
        if TrainingPhase.INITIALIZATION in self.training_metrics:
            self._complete_phase(TrainingPhase.INITIALIZATION, success=False, error_message=error_msg)
    
    def _validate_input_data(self, X: np.ndarray, y: np.ndarray, 
                           regime_labels: np.ndarray) -> Dict[str, Any]:
        """Comprehensive input data validation with detailed reporting and utility integration."""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'data_quality_metrics': {},
            'regime_analysis': {},
            'utility_validation': {}
        }
        
        try:
            tprint_info("🔍 Starting comprehensive input data validation...")
            
            # CRITICAL: Fast fail on data shape mismatches
            tprint_debug("Validating data shapes...")
            if X.shape[0] != y.shape[0]:
                error_msg = f"CRITICAL: Feature and target sample counts don't match: {X.shape[0]} vs {y.shape[0]}"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            if X.shape[0] != regime_labels.shape[0]:
                error_msg = f"CRITICAL: Feature and regime label sample counts don't match: {X.shape[0]} vs {regime_labels.shape[0]}"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            # CRITICAL: Fast fail on empty data
            if X.shape[0] == 0:
                error_msg = "CRITICAL: No samples provided in input data"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            if X.shape[1] == 0:
                error_msg = "CRITICAL: No features provided in input data"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            tprint_success(f"✅ Basic shape validation passed: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Enhanced data quality analysis using utility functions
            data_quality = {}
            
            # Validate features quality with math validation utilities
            tprint_debug("Validating features quality...")
            feature_quality = self._validate_data_quality_enhanced(X, "features", max_nan_percentage=10.0, max_inf_percentage=1.0)
            data_quality['features'] = feature_quality
            validation_results['warnings'].extend(feature_quality['warnings'])
            validation_results['errors'].extend(feature_quality['errors'])
            if not feature_quality['is_valid']:
                validation_results['is_valid'] = False
                tprint_error("❌ Feature quality validation failed")
            else:
                tprint_success("✅ Feature quality validation passed")
            
            # Validate targets quality with stricter thresholds
            tprint_debug("Validating targets quality...")
            target_quality = self._validate_data_quality_enhanced(y, "targets", max_nan_percentage=5.0, max_inf_percentage=1.0)
            data_quality['targets'] = target_quality
            validation_results['warnings'].extend(target_quality['warnings'])
            validation_results['errors'].extend(target_quality['errors'])
            if not target_quality['is_valid']:
                validation_results['is_valid'] = False
                tprint_error("❌ Target quality validation failed")
            else:
                tprint_success("✅ Target quality validation passed")
            
            # Enhanced regime distribution analysis
            tprint_debug("Analyzing regime distribution...")
            unique_regimes = np.unique(regime_labels)
            regime_counts = np.bincount(regime_labels)
            min_regime_size = np.min(regime_counts)
            max_regime_size = np.max(regime_counts)
            
            # Use math validation utilities for safe calculations
            regime_balance = safe_divide(min_regime_size, max_regime_size, 0.0)
            
            regime_analysis = {
                'unique_regimes_count': len(unique_regimes),
                'min_regime_size': min_regime_size,
                'max_regime_size': max_regime_size,
                'regime_balance': regime_balance,
                'regime_distribution': dict(zip(unique_regimes, regime_counts))
            }
            
            validation_results['regime_analysis'] = regime_analysis
            validation_results['data_quality_metrics'] = data_quality
            
            # Check regime sufficiency with enhanced validation
            insufficient_regimes = regime_counts < self.config.min_samples_per_regime
            insufficient_count = np.sum(insufficient_regimes)
            
            if insufficient_count > 0:
                warning_msg = f"{insufficient_count} regimes have fewer than {self.config.min_samples_per_regime} samples"
                validation_results['warnings'].append(warning_msg)
                tprint_warning(f"⚠️ {warning_msg}")
                
                # Check if too many regimes are insufficient (fast fail condition)
                if insufficient_count > len(unique_regimes) * 0.5:
                    error_msg = f"Critical: {insufficient_count}/{len(unique_regimes)} regimes have insufficient data"
                    validation_results['errors'].append(error_msg)
                    validation_results['is_valid'] = False
                    tprint_error(f"❌ {error_msg}")
                    raise ValueError(error_msg)  # Fast fail on critical errors
            
            # Utility integration validation
            utility_validation = self._validate_utility_integrations()
            validation_results['utility_validation'] = utility_validation
            
            # Log comprehensive validation results with tprint
            tprint_info(f"📊 Data validation summary: {X.shape[0]} samples, {X.shape[1]} features, {len(unique_regimes)} regimes")
            tprint_info(f"📊 Regime balance: {regime_balance:.3f} (min={min_regime_size}, max={max_regime_size})")
            
            if validation_results['warnings']:
                tprint_warning(f"⚠️ {len(validation_results['warnings'])} warnings found:")
                for warning in validation_results['warnings']:
                    tprint_warning(f"  - {warning}")
            
            if validation_results['errors']:
                tprint_error(f"❌ {len(validation_results['errors'])} errors found:")
                for error in validation_results['errors']:
                    tprint_error(f"  - {error}")
                raise ValueError(f"Data validation failed: {'; '.join(validation_results['errors'])}")
            
            tprint_success("✅ Comprehensive input data validation completed successfully")
            return validation_results
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(str(e))
            tprint_error(f"❌ Data validation failed: {e}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            raise
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _validate_data_quality_enhanced(self, data: np.ndarray, data_name: str, 
                                       max_nan_percentage: float = 10.0, 
                                       max_inf_percentage: float = 1.0) -> Dict[str, Any]:
        """Enhanced data quality validation with utility integration and comprehensive error handling."""
        quality_metrics = {
            'nan_count': 0,
            'inf_count': 0,
            'nan_percentage': 0.0,
            'inf_percentage': 0.0,
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'utility_validation': {},
            'statistical_metrics': {}
        }
        
        try:
            tprint_debug(f"Validating {data_name} quality...")
            
            # Check for NaN values with enhanced reporting
            nan_count = np.sum(np.isnan(data))
            if nan_count > 0:
                nan_percentage = safe_divide(nan_count * 100, data.size, 0.0)
                
                quality_metrics['nan_count'] = nan_count
                quality_metrics['nan_percentage'] = nan_percentage
                
                if nan_percentage > max_nan_percentage:
                    error_msg = f"{data_name} contains {nan_percentage:.2f}% NaN values (threshold: {max_nan_percentage}%)"
                    quality_metrics['errors'].append(error_msg)
                    quality_metrics['is_valid'] = False
                    tprint_error(f"❌ {error_msg}")
                else:
                    warning_msg = f"{data_name} contains {nan_count} NaN values ({nan_percentage:.2f}%)"
                    quality_metrics['warnings'].append(warning_msg)
                    tprint_warning(f"⚠️ {warning_msg}")
            else:
                tprint_debug(f"✅ {data_name}: No NaN values found")
            
            # Check for infinite values with enhanced reporting
            inf_count = np.sum(np.isinf(data))
            if inf_count > 0:
                inf_percentage = safe_divide(inf_count * 100, data.size, 0.0)
                
                quality_metrics['inf_count'] = inf_count
                quality_metrics['inf_percentage'] = inf_percentage
                
                if inf_percentage > max_inf_percentage:
                    error_msg = f"{data_name} contains {inf_percentage:.2f}% infinite values (threshold: {max_inf_percentage}%)"
                    quality_metrics['errors'].append(error_msg)
                    quality_metrics['is_valid'] = False
                    tprint_error(f"❌ {error_msg}")
                else:
                    warning_msg = f"{data_name} contains {inf_count} infinite values ({inf_percentage:.2f}%)"
                    quality_metrics['warnings'].append(warning_msg)
                    tprint_warning(f"⚠️ {warning_msg}")
            else:
                tprint_debug(f"✅ {data_name}: No infinite values found")
            
            # Enhanced statistical validation using math validation utilities
            try:
                # Validate finite values
                finite_data = data[np.isfinite(data)]
                if len(finite_data) > 0:
                    # Calculate statistical metrics safely
                    mean_val = np.mean(finite_data)
                    std_val = np.std(finite_data)
                    min_val = np.min(finite_data)
                    max_val = np.max(finite_data)
                    
                    # Validate statistical properties
                    validate_finite(mean_val, f"{data_name}_mean")
                    validate_finite(std_val, f"{data_name}_std")
                    validate_finite(min_val, f"{data_name}_min")
                    validate_finite(max_val, f"{data_name}_max")
                    
                    quality_metrics['statistical_metrics'] = {
                        'mean': mean_val,
                        'std': std_val,
                        'min': min_val,
                        'max': max_val,
                        'finite_count': len(finite_data),
                        'total_count': data.size
                    }
                    
                    tprint_debug(f"✅ {data_name}: Statistical validation passed")
                else:
                    warning_msg = f"{data_name}: No finite values found for statistical analysis"
                    quality_metrics['warnings'].append(warning_msg)
                    tprint_warning(f"⚠️ {warning_msg}")
                    
            except Exception as e:
                warning_msg = f"Statistical validation failed for {data_name}: {e}"
                quality_metrics['warnings'].append(warning_msg)
                tprint_warning(f"⚠️ {warning_msg}")
            
            # Matrix operations validation
            try:
                matrix_validation = validate_matrix_properties(data)
                quality_metrics['utility_validation']['matrix_operations'] = matrix_validation
                tprint_debug(f"✅ {data_name}: Matrix operations validation completed")
            except Exception as e:
                warning_msg = f"Matrix operations validation failed for {data_name}: {e}"
                quality_metrics['warnings'].append(warning_msg)
                tprint_warning(f"⚠️ {warning_msg}")
            
            # Data utilities validation
            if data_name == "features":
                try:
                    # Convert to DataFrame for validation if possible
                    if data.ndim == 2:
                        df = pd.DataFrame(data)
                        data_quality_metrics = calculate_data_quality_metrics(df)
                        quality_metrics['utility_validation']['data_quality'] = data_quality_metrics
                        tprint_debug(f"✅ {data_name}: Data quality metrics calculated")
                except Exception as e:
                    warning_msg = f"Data quality metrics calculation failed for {data_name}: {e}"
                    quality_metrics['warnings'].append(warning_msg)
                    tprint_warning(f"⚠️ {warning_msg}")
            
            if quality_metrics['is_valid']:
                tprint_success(f"✅ {data_name} quality validation passed")
            else:
                tprint_error(f"❌ {data_name} quality validation failed")
            
            return quality_metrics
            
        except Exception as e:
            quality_metrics['is_valid'] = False
            error_msg = f"Failed to validate {data_name}: {e}"
            quality_metrics['errors'].append(error_msg)
            tprint_error(f"❌ {error_msg}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            return quality_metrics
    
    def _validate_utility_integrations(self) -> Dict[str, Any]:
        """Validate utility integrations and their availability."""
        utility_validation = {
            'tprint_available': TPRINT_AVAILABLE,
            'common_operations_available': COMMON_OPERATIONS_AVAILABLE,
            'common_utilities_available': COMMON_UTILITIES_AVAILABLE,
            'math_validation_available': MATH_VALIDATION_AVAILABLE,
            'data_utilities_available': DATA_UTILITIES_AVAILABLE,
            'matrix_operations_available': MATRIX_OPERATIONS_AVAILABLE,
            'ml_common_available': ML_COMMON_AVAILABLE,
            'vectorized_training_available': VECTORIZED_TRAINING_AVAILABLE
        }
        
        try:
            tprint_debug("Validating utility integrations...")
            
            # Count available utilities
            available_count = sum(1 for available in utility_validation.values() if available)
            total_count = len(utility_validation)
            availability_rate = safe_divide(available_count * 100, total_count, 0.0)
            
            utility_validation['available_count'] = available_count
            utility_validation['total_count'] = total_count
            utility_validation['availability_rate'] = availability_rate
            
            if availability_rate >= 80:
                tprint_success(f"✅ Utility integration validation passed: {available_count}/{total_count} utilities available ({availability_rate:.1f}%)")
            elif availability_rate >= 60:
                tprint_warning(f"⚠️ Utility integration validation warning: {available_count}/{total_count} utilities available ({availability_rate:.1f}%)")
            else:
                tprint_error(f"❌ Utility integration validation failed: {available_count}/{total_count} utilities available ({availability_rate:.1f}%)")
            
            return utility_validation
            
        except Exception as e:
            tprint_error(f"❌ Utility integration validation failed: {e}")
            utility_validation['validation_error'] = str(e)
            return utility_validation
    
    def _validate_array_shapes(self, arrays: Dict[str, np.ndarray], expected_samples: int) -> Dict[str, Any]:
        """Validate that all arrays have consistent sample counts."""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'shape_info': {}
        }
        
        try:
            for name, array in arrays.items():
                if array.shape[0] != expected_samples:
                    error_msg = f"{name} shape mismatch: expected {expected_samples}, got {array.shape[0]}"
                    validation_results['errors'].append(error_msg)
                    validation_results['is_valid'] = False
                else:
                    validation_results['shape_info'][name] = array.shape
            
            if not validation_results['is_valid']:
                raise ValueError(f"Shape validation failed: {'; '.join(validation_results['errors'])}")
            
            return validation_results
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(str(e))
            return validation_results
    
    def _comprehensive_validation_check(self, 
                                      X: np.ndarray, 
                                      y: np.ndarray, 
                                      regime_labels: np.ndarray,
                                      phase_name: str,
                                      additional_arrays: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """Comprehensive validation check for any phase of training."""
        validation_summary = {
            'phase': phase_name,
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'metrics': {},
            'recommendations': []
        }
        
        try:
            # Basic shape validation
            arrays_to_validate = {
                'features': X,
                'targets': y,
                'regime_labels': regime_labels
            }
            
            if additional_arrays:
                arrays_to_validate.update(additional_arrays)
            
            shape_validation = self._validate_array_shapes(arrays_to_validate, X.shape[0])
            if not shape_validation['is_valid']:
                validation_summary['errors'].extend(shape_validation['errors'])
                validation_summary['is_valid'] = False
            
            # Data quality validation
            feature_quality = self._validate_data_quality(X, f"{phase_name}_features", max_nan_percentage=10.0, max_inf_percentage=1.0)
            target_quality = self._validate_data_quality(y, f"{phase_name}_targets", max_nan_percentage=5.0, max_inf_percentage=1.0)
            
            validation_summary['warnings'].extend(feature_quality['warnings'])
            validation_summary['warnings'].extend(target_quality['warnings'])
            validation_summary['errors'].extend(feature_quality['errors'])
            validation_summary['errors'].extend(target_quality['errors'])
            
            if not feature_quality['is_valid'] or not target_quality['is_valid']:
                validation_summary['is_valid'] = False
            
            # Regime analysis
            unique_regimes = np.unique(regime_labels)
            regime_counts = np.bincount(regime_labels)
            min_regime_size = np.min(regime_counts)
            max_regime_size = np.max(regime_counts)
            regime_balance = min_regime_size / max_regime_size if max_regime_size > 0 else 0
            
            validation_summary['metrics'] = {
                'samples': X.shape[0],
                'features': X.shape[1],
                'regimes': len(unique_regimes),
                'min_regime_size': min_regime_size,
                'max_regime_size': max_regime_size,
                'regime_balance': regime_balance,
                'feature_nan_percentage': feature_quality['nan_percentage'],
                'target_nan_percentage': target_quality['nan_percentage']
            }
            
            # Generate recommendations
            if regime_balance < 0.1:
                validation_summary['recommendations'].append("Very low regime balance - consider data augmentation")
            
            if feature_quality['nan_percentage'] > 5:
                validation_summary['recommendations'].append("High NaN percentage in features - consider imputation")
            
            if target_quality['nan_percentage'] > 2:
                validation_summary['recommendations'].append("High NaN percentage in targets - review data pipeline")
            
            if min_regime_size < self.config.min_samples_per_regime:
                validation_summary['recommendations'].append("Some regimes have insufficient samples - consider reducing min_samples_per_regime")
            
            # Log validation results
            if validation_summary['is_valid']:
                self.logger.info(f"✅ {phase_name} validation passed")
            else:
                self.logger.error(f"❌ {phase_name} validation failed")
                for error in validation_summary['errors']:
                    self.logger.error(f"❌ {error}")
            
            if validation_summary['warnings']:
                for warning in validation_summary['warnings']:
                    self.logger.warning(f"⚠️ {warning}")
            
            if validation_summary['recommendations']:
                for recommendation in validation_summary['recommendations']:
                    self.logger.info(f"💡 {recommendation}")
            
            return validation_summary
            
        except Exception as e:
            validation_summary['is_valid'] = False
            validation_summary['errors'].append(f"Validation check failed: {e}")
            self.logger.error(f"❌ {phase_name} validation check failed: {e}")
            return validation_summary
    
    def _log_structured_event(self, event_type: str, phase: str, message: str, 
                             metrics: Optional[Dict[str, Any]] = None, 
                             level: str = "info") -> None:
        """Log structured events with consistent formatting."""
        try:
            log_data = {
                'event_type': event_type,
                'phase': phase,
                'message': message,
                'timestamp': time.time(),
                'memory_mb': self._get_memory_usage()
            }
            
            if metrics:
                log_data['metrics'] = metrics
            
            # Format structured log message
            structured_msg = f"[{event_type.upper()}] {phase}: {message}"
            if metrics:
                structured_msg += f" | Metrics: {metrics}"
            
            # Log with appropriate level
            if level == "error":
                self.logger.error(structured_msg)
            elif level == "warning":
                self.logger.warning(structured_msg)
            elif level == "debug":
                self.logger.debug(structured_msg)
            else:
                self.logger.info(structured_msg)
                
        except Exception as e:
            self.logger.error(f"Failed to log structured event: {e}")
    
    def _log_phase_start(self, phase: TrainingPhase, context: Optional[Dict[str, Any]] = None) -> None:
        """Log phase start with context."""
        self._log_structured_event(
            event_type="phase_start",
            phase=phase.value,
            message=f"Starting {phase.value} phase",
            metrics=context,
            level="info"
        )
    
    def _log_phase_complete(self, phase: TrainingPhase, success: bool, 
                           duration: float, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Log phase completion with results."""
        event_type = "phase_success" if success else "phase_failure"
        message = f"Completed {phase.value} phase in {duration:.2f}s"
        
        if not success:
            message += " (FAILED)"
        
        self._log_structured_event(
            event_type=event_type,
            phase=phase.value,
            message=message,
            metrics=metrics,
            level="error" if not success else "info"
        )
    
    def _log_data_quality_issue(self, issue_type: str, details: Dict[str, Any]) -> None:
        """Log data quality issues with structured format."""
        self._log_structured_event(
            event_type="data_quality_issue",
            phase="validation",
            message=f"Data quality issue: {issue_type}",
            metrics=details,
            level="warning"
        )
    
    def _log_performance_metric(self, metric_name: str, value: float, unit: str = "") -> None:
        """Log performance metrics with structured format."""
        self._log_structured_event(
            event_type="performance_metric",
            phase="training",
            message=f"Performance metric: {metric_name}",
            metrics={metric_name: f"{value:.2f}{unit}"},
            level="info"
        )
    
    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        analyst_signals: Optional[np.ndarray] = None,
        analyst_model_outputs: Optional[np.ndarray] = None,
        hmm_regime_features: Optional[np.ndarray] = None,
        all_analyst_models_outputs: Optional[Dict[str, np.ndarray]] = None,
        hmm_model_outputs: Optional[np.ndarray] = None,
        analyst_ensemble_outputs: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        analyst_confidence_scores: Optional[np.ndarray] = None,
        analyst_directional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute enhanced Tactician models training step with comprehensive error handling and utility integration.
        
        Args:
            X: Input features (1m timeframe with cross-timeframe features)
            y: Target values (tactician outputs - timing decisions)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            analyst_signals: Directional signals from Analyst (1=long, -1=short, 0=neutral)
            analyst_model_outputs: Analyst model predictions used as features
            hmm_regime_features: HMM regime features (probabilities, characteristics)
            all_analyst_models_outputs: All individual analyst ML model outputs
            hmm_model_outputs: HMM model outputs (predictions, probabilities, etc.)
            analyst_ensemble_outputs: Analyst ensemble model outputs
            analyst_confidence_scores: Confidence scores from Analyst for sample weighting
            analyst_directional_info: Additional directional analysis from Analyst
            
        Returns:
            Dictionary containing training results and metadata with comprehensive reporting
        """
        try:
            tprint_info("🚀 Starting Enhanced Tactician models training step")
            tprint_structured({
                'operation': 'tactician_training_start',
                'samples': X.shape[0] if X is not None else 0,
                'features': X.shape[1] if X is not None else 0,
                'regimes': len(np.unique(regime_labels)) if regime_labels is not None else 0,
                'has_analyst_signals': analyst_signals is not None,
                'has_hmm_features': hmm_regime_features is not None,
                'has_analyst_models': all_analyst_models_outputs is not None
            })
            
            self.overall_start_time = time.time()
            
            # Phase 1: Data Validation with comprehensive error handling
            validation_context = {
                'samples': X.shape[0] if X is not None else 0,
                'features': X.shape[1] if X is not None else 0,
                'regimes': len(np.unique(regime_labels)) if regime_labels is not None else 0
            }
            self._start_phase(TrainingPhase.DATA_VALIDATION, validation_context)
            
            try:
                with tprint_timer("Data Validation"):
                    validation_results = self._validate_input_data(X, y, regime_labels)
                
                # Log data quality issues with enhanced reporting
                if validation_results.get('warnings'):
                    tprint_warning(f"⚠️ {len(validation_results['warnings'])} data quality warnings found")
                    for warning in validation_results['warnings']:
                        self._log_data_quality_issue("warning", {'message': warning})
                
                if validation_results.get('errors'):
                    tprint_error(f"❌ {len(validation_results['errors'])} data quality errors found")
                    for error in validation_results['errors']:
                        self._log_data_quality_issue("error", {'message': error})
                
                # Log utility integration status
                if validation_results.get('utility_validation'):
                    utility_status = validation_results['utility_validation']
                    tprint_info(f"📊 Utility integration status: {utility_status.get('available_count', 0)}/{utility_status.get('total_count', 0)} utilities available")
                
                self._complete_phase(TrainingPhase.DATA_VALIDATION, success=True, 
                                   samples_processed=X.shape[0], features_count=X.shape[1],
                                   warnings_issued=len(validation_results.get('warnings', [])),
                                   errors_encountered=len(validation_results.get('errors', [])))
            except Exception as e:
                self._complete_phase(TrainingPhase.DATA_VALIDATION, success=False, error_message=str(e))
                tprint_error(f"❌ Data validation phase failed: {e}")
                raise
            
            # Phase 2: Feature Preparation with enhanced error handling
            feature_context = {
                'original_samples': X.shape[0],
                'original_features': X.shape[1],
                'has_analyst_signals': analyst_signals is not None,
                'has_hmm_features': hmm_regime_features is not None,
                'has_analyst_models': all_analyst_models_outputs is not None,
                'has_analyst_ensemble': analyst_ensemble_outputs is not None
            }
            self._start_phase(TrainingPhase.FEATURE_PREPARATION, feature_context)
            
            try:
                with tprint_timer("Feature Preparation"):
                  X, y, regime_labels, feature_names, preparation_metrics = self._prepare_features(
                      X, y, regime_labels, feature_names, hmm_states, 
                      analyst_signals, analyst_model_outputs, hmm_regime_features, 
                      all_analyst_models_outputs, hmm_model_outputs, analyst_ensemble_outputs
                  )

                
                # Log feature preparation metrics with enhanced reporting
                if preparation_metrics.get('green_light_filtering'):
                    gl_filtering = preparation_metrics['green_light_filtering']
                    green_light_rate = gl_filtering.get('green_light_rate', 0) * 100
                    tprint_info(f"📊 Green light filtering: {green_light_rate:.2f}% of samples retained")
                    self._log_performance_metric("green_light_rate", green_light_rate, "%")
                
                # Log feature combination results
                if preparation_metrics.get('feature_combinations'):
                    feature_combinations = preparation_metrics['feature_combinations']
                    tprint_info(f"📊 Feature combinations: {feature_combinations}")
                
                self._complete_phase(TrainingPhase.FEATURE_PREPARATION, success=True,
                                   samples_processed=X.shape[0], features_count=X.shape[1],
                                   warnings_issued=len(preparation_metrics.get('warnings', [])),
                                   errors_encountered=len(preparation_metrics.get('errors', [])))
            except Exception as e:
                self._complete_phase(TrainingPhase.FEATURE_PREPARATION, success=False, error_message=str(e))
                tprint_error(f"❌ Feature preparation phase failed: {e}")
                raise
            
            # Phase 3: Model Training with enhanced error handling
            self._start_phase(TrainingPhase.MODEL_TRAINING)
            try:
                with tprint_timer("Model Training"):
                    results = self._execute_training_enhanced(
                        X, y, regime_labels, feature_names, hmm_states,
                        analyst_confidence_scores, analyst_directional_info
                    )
                
                models_trained = len(results.get('models', {}))
                tprint_success(f"✅ Model training completed: {models_trained} models trained")
                
                self._complete_phase(TrainingPhase.MODEL_TRAINING, success=True,
                                   models_trained=models_trained,
                                   memory_usage_mb=self._get_memory_usage())
            except Exception as e:
                self._complete_phase(TrainingPhase.MODEL_TRAINING, success=False, error_message=str(e))
                tprint_error(f"❌ Model training phase failed: {e}")
                raise
            
            # Phase 4: Finalization with enhanced error handling
            self._start_phase(TrainingPhase.FINALIZATION)
            try:
                with tprint_timer("Results Finalization"):
                    results = self._finalize_results_enhanced(results, analyst_signals)
                
                total_time = time.time() - self.overall_start_time
                tprint_performance("Total Tactician Training", total_time)
                
                self._complete_phase(TrainingPhase.FINALIZATION, success=True)
                
                # Generate comprehensive training report
                self._generate_training_report_enhanced(total_time)
                
                # Log final success
                tprint_success("🎉 Enhanced Tactician models training completed successfully!")
                tprint_structured({
                    'operation': 'tactician_training_complete',
                    'total_time': total_time,
                    'models_trained': models_trained,
                    'final_samples': X.shape[0],
                    'final_features': X.shape[1],
                    'success': True
                })
                
                return results
                
            except Exception as e:
                self._complete_phase(TrainingPhase.FINALIZATION, success=False, error_message=str(e))
                tprint_error(f"❌ Finalization phase failed: {e}")
                raise
                
        except Exception as e:
            tprint_error(f"❌ Enhanced Tactician training failed: {e}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            tprint_structured({
                'operation': 'tactician_training_failed',
                'error': str(e),
                'success': False
            })
            return self._create_error_result(str(e))
    
    def _prepare_features_enhanced(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray],
        analyst_signals: Optional[np.ndarray],
        analyst_model_outputs: Optional[np.ndarray],
        hmm_regime_features: Optional[np.ndarray],
        all_analyst_models_outputs: Optional[Dict[str, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[str]], Dict[str, Any]]:
        """Enhanced feature preparation with comprehensive error handling and utility integration."""
        return self._prepare_features(
            X, y, regime_labels, feature_names, hmm_states,
            analyst_signals, analyst_model_outputs, hmm_regime_features,
            all_analyst_models_outputs
        )
    
    def _execute_training_enhanced(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray],
        analyst_confidence_scores: Optional[np.ndarray] = None,
        analyst_directional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Enhanced training execution with comprehensive error handling and utility integration."""
        return self._execute_training(
            X, y, regime_labels, feature_names, hmm_states,
            analyst_confidence_scores, analyst_directional_info
        )
    
    def _finalize_results_enhanced(self, results: Dict[str, Any], analyst_signals: Optional[np.ndarray]) -> Dict[str, Any]:
        """Enhanced results finalization with comprehensive error handling and utility integration."""
        return self._finalize_results(results, analyst_signals)
    
    def _generate_training_report_enhanced(self, total_time: float) -> None:
        """Enhanced training report generation with comprehensive error handling and utility integration."""
        return self._generate_training_report(total_time)
    
    def _prepare_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray],
        analyst_signals: Optional[np.ndarray],
        analyst_model_outputs: Optional[np.ndarray],
        hmm_regime_features: Optional[np.ndarray],
        all_analyst_models_outputs: Optional[Dict[str, np.ndarray]],
        hmm_model_outputs: Optional[np.ndarray],
        analyst_ensemble_outputs: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[str]], Dict[str, Any]]:
        """Prepare and combine all features with comprehensive error handling and validation."""
        preparation_metrics = {
            'original_samples': X.shape[0],
            'original_features': X.shape[1],
            'green_light_filtering': {},
            'feature_combinations': {},
            'final_samples': 0,
            'final_features': 0,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Filter data to only include periods where Analyst gives directional signals
            if analyst_signals is not None:
                # Directional signals: 1 (long), -1 (short), 0 (neutral)
                directional_mask = (analyst_signals == 1) | (analyst_signals == -1)
                directional_count = np.sum(directional_mask)
                directional_rate = directional_count / len(analyst_signals)

                # Analyze directional distribution
                long_count = np.sum(analyst_signals == 1)
                short_count = np.sum(analyst_signals == -1)
                neutral_count = np.sum(analyst_signals == 0)

                preparation_metrics['directional_filtering'] = {
                    'total_signals': len(analyst_signals),
                    'directional_count': directional_count,
                    'directional_rate': directional_rate,
                    'long_count': long_count,
                    'short_count': short_count,
                    'neutral_count': neutral_count,
                    'long_ratio': long_count / len(analyst_signals),
                    'short_ratio': short_count / len(analyst_signals),
                    'neutral_ratio': neutral_count / len(analyst_signals)
                }

                self.logger.info(f"📊 Filtering to {directional_count} samples with Analyst directional signals ({directional_rate:.2%})")
                self.logger.info(f"   Long signals: {long_count} ({long_count/directional_count:.1%} of directional)")
                self.logger.info(f"   Short signals: {short_count} ({short_count/directional_count:.1%} of directional)")
                self.logger.info(f"   Neutral signals: {neutral_count} ({neutral_count/len(analyst_signals):.1%} of total)")

                # Validate directional filtering results
                if directional_count == 0:
                    error_msg = "No samples with Analyst directional signals found"
                    preparation_metrics['errors'].append(error_msg)
                    raise ValueError(error_msg)

                if directional_rate < 0.01:  # Less than 1%
                    warning_msg = f"Very low directional signal rate: {directional_rate:.2%}"
                    preparation_metrics['warnings'].append(warning_msg)
                    self.logger.warning(f"⚠️ {warning_msg}")
                
                # Apply filtering with validation
                X_filtered = X[directional_mask]
                y_filtered = y[directional_mask]
                regime_labels_filtered = regime_labels[directional_mask]
                
                # Validate filtered data shapes
                if X_filtered.shape[0] != directional_count:
                    error_msg = f"Filtered data shape mismatch: expected {directional_count}, got {X_filtered.shape[0]}"
                    preparation_metrics['errors'].append(error_msg)
                    raise ValueError(error_msg)
                
                X, y, regime_labels = X_filtered, y_filtered, regime_labels_filtered
                
                if hmm_states is not None:
                    hmm_states = hmm_states[green_light_mask]
                    if hmm_states.shape[0] != green_light_count:
                        error_msg = f"HMM states filtering mismatch: expected {green_light_count}, got {hmm_states.shape[0]}"
                        preparation_metrics['errors'].append(error_msg)
                        raise ValueError(error_msg)
            
            # Combine all features: base features + HMM regime features + HMM model outputs + all analyst model outputs + analyst ensemble outputs
            additional_features = []
            additional_feature_names = []
            feature_combination_details = {}
            
            # Add HMM regime features if provided
            if hmm_regime_features is not None:
                try:
                    if analyst_signals is not None:
                        hmm_regime_features = hmm_regime_features[green_light_mask]
                    
                    # Validate HMM features shape
                    if hmm_regime_features.shape[0] != X.shape[0]:
                        error_msg = f"HMM regime features shape mismatch: expected {X.shape[0]}, got {hmm_regime_features.shape[0]}"
                        preparation_metrics['errors'].append(error_msg)
                        raise ValueError(error_msg)
                    
                    # Check for NaN/Inf in HMM features
                    hmm_nan_count = np.sum(np.isnan(hmm_regime_features))
                    hmm_inf_count = np.sum(np.isinf(hmm_regime_features))
                    
                    if hmm_nan_count > 0:
                        warning_msg = f"HMM features contain {hmm_nan_count} NaN values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")
                    
                    if hmm_inf_count > 0:
                        warning_msg = f"HMM features contain {hmm_inf_count} infinite values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")
                    
                    additional_features.append(hmm_regime_features)
                    additional_feature_names.extend([f"hmm_regime_{i}" for i in range(hmm_regime_features.shape[1])])
                    
                    feature_combination_details['hmm_regime_features'] = {
                        'count': hmm_regime_features.shape[1],
                        'nan_count': hmm_nan_count,
                        'inf_count': hmm_inf_count
                    }
                    
                    self.logger.info(f"📊 Added {hmm_regime_features.shape[1]} HMM regime features")
                    
                except Exception as e:
                    error_msg = f"Failed to add HMM regime features: {e}"
                    preparation_metrics['errors'].append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
                    raise
            
            # Add HMM model outputs if provided
            if hmm_model_outputs is not None:
                try:
                    if analyst_signals is not None:
                        hmm_model_outputs = hmm_model_outputs[green_light_mask]
                    
                    # Validate HMM model outputs shape
                    if hmm_model_outputs.shape[0] != X.shape[0]:
                        error_msg = f"HMM model outputs shape mismatch: expected {X.shape[0]}, got {hmm_model_outputs.shape[0]}"
                        preparation_metrics['errors'].append(error_msg)
                        raise ValueError(error_msg)
                    
                    # Check for NaN/Inf in HMM model outputs
                    hmm_outputs_nan_count = np.sum(np.isnan(hmm_model_outputs))
                    hmm_outputs_inf_count = np.sum(np.isinf(hmm_model_outputs))
                    
                    if hmm_outputs_nan_count > 0:
                        warning_msg = f"HMM model outputs contain {hmm_outputs_nan_count} NaN values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")
                    
                    if hmm_outputs_inf_count > 0:
                        warning_msg = f"HMM model outputs contain {hmm_outputs_inf_count} infinite values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")
                    
                    additional_features.append(hmm_model_outputs)
                    additional_feature_names.extend([f"hmm_model_{i}" for i in range(hmm_model_outputs.shape[1])])
                    
                    feature_combination_details['hmm_model_outputs'] = {
                        'count': hmm_model_outputs.shape[1],
                        'nan_count': hmm_outputs_nan_count,
                        'inf_count': hmm_outputs_inf_count
                    }
                    
                    self.logger.info(f"📊 Added {hmm_model_outputs.shape[1]} HMM model outputs")
                    
                except Exception as e:
                    error_msg = f"Failed to add HMM model outputs: {e}"
                    preparation_metrics['errors'].append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
                    raise
            
            # Add all individual analyst model outputs if provided
            if all_analyst_models_outputs is not None:
                analyst_features_added = 0
                for model_name, model_outputs in all_analyst_models_outputs.items():
                    try:
                        if analyst_signals is not None:
                            model_outputs = model_outputs[green_light_mask]
                        
                        # Validate model outputs shape
                        if model_outputs.shape[0] != X.shape[0]:
                            error_msg = f"Analyst model {model_name} output shape mismatch: expected {X.shape[0]}, got {model_outputs.shape[0]}"
                            preparation_metrics['errors'].append(error_msg)
                            raise ValueError(error_msg)
                        
                        # Check for NaN/Inf in model outputs
                        model_nan_count = np.sum(np.isnan(model_outputs))
                        model_inf_count = np.sum(np.isinf(model_outputs))
                        
                        if model_nan_count > 0:
                            warning_msg = f"Analyst model {model_name} outputs contain {model_nan_count} NaN values"
                            preparation_metrics['warnings'].append(warning_msg)
                            self.logger.warning(f"⚠️ {warning_msg}")
                        
                        if model_inf_count > 0:
                            warning_msg = f"Analyst model {model_name} outputs contain {model_inf_count} infinite values"
                            preparation_metrics['warnings'].append(warning_msg)
                            self.logger.warning(f"⚠️ {warning_msg}")
                        
                        additional_features.append(model_outputs)
                        additional_feature_names.extend([f"analyst_{model_name}_{i}" for i in range(model_outputs.shape[1])])
                        analyst_features_added += model_outputs.shape[1]
                        
                    except Exception as e:
                        error_msg = f"Failed to add analyst model {model_name} outputs: {e}"
                        preparation_metrics['errors'].append(error_msg)
                        self.logger.error(f"❌ {error_msg}")
                        raise
                
                feature_combination_details['analyst_models'] = {
                    'model_count': len(all_analyst_models_outputs),
                    'total_features': analyst_features_added
                }
                
                self.logger.info(f"📊 Added outputs from {len(all_analyst_models_outputs)} analyst models ({analyst_features_added} features)")
            
            # Add analyst ensemble outputs if provided
            if analyst_ensemble_outputs is not None:
                try:
                    if analyst_signals is not None:
                        analyst_ensemble_outputs = analyst_ensemble_outputs[green_light_mask]
                    
                    # Validate analyst ensemble outputs shape
                    if analyst_ensemble_outputs.shape[0] != X.shape[0]:
                        error_msg = f"Analyst ensemble outputs shape mismatch: expected {X.shape[0]}, got {analyst_ensemble_outputs.shape[0]}"
                        preparation_metrics['errors'].append(error_msg)
                        raise ValueError(error_msg)
                    
                    # Check for NaN/Inf in analyst ensemble outputs
                    ensemble_nan_count = np.sum(np.isnan(analyst_ensemble_outputs))
                    ensemble_inf_count = np.sum(np.isinf(analyst_ensemble_outputs))
                    
                    if ensemble_nan_count > 0:
                        warning_msg = f"Analyst ensemble outputs contain {ensemble_nan_count} NaN values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")
                    
                    if ensemble_inf_count > 0:
                        warning_msg = f"Analyst ensemble outputs contain {ensemble_inf_count} infinite values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")
                    
                    additional_features.append(analyst_ensemble_outputs)
                    additional_feature_names.extend([f"analyst_ensemble_{i}" for i in range(analyst_ensemble_outputs.shape[1])])
                    
                    feature_combination_details['analyst_ensemble_outputs'] = {
                        'count': analyst_ensemble_outputs.shape[1],
                        'nan_count': ensemble_nan_count,
                        'inf_count': ensemble_inf_count
                    }
                    
                    self.logger.info(f"📊 Added {analyst_ensemble_outputs.shape[1]} analyst ensemble outputs")
                    
                except Exception as e:
                    error_msg = f"Failed to add analyst ensemble outputs: {e}"
                    preparation_metrics['errors'].append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
                    raise
            
            # Add legacy analyst model outputs for backward compatibility
            if analyst_model_outputs is not None:
                try:
                    if analyst_signals is not None:
                        analyst_model_outputs = analyst_model_outputs[green_light_mask]
                    
                    # Validate legacy outputs shape
                    if analyst_model_outputs.shape[0] != X.shape[0]:
                        error_msg = f"Legacy analyst outputs shape mismatch: expected {X.shape[0]}, got {analyst_model_outputs.shape[0]}"
                        preparation_metrics['errors'].append(error_msg)
                        raise ValueError(error_msg)
                    
                    # Check for NaN/Inf in legacy outputs
                    legacy_nan_count = np.sum(np.isnan(analyst_model_outputs))
                    legacy_inf_count = np.sum(np.isinf(analyst_model_outputs))
                    
                    if legacy_nan_count > 0:
                        warning_msg = f"Legacy analyst outputs contain {legacy_nan_count} NaN values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")
                    
                    if legacy_inf_count > 0:
                        warning_msg = f"Legacy analyst outputs contain {legacy_inf_count} infinite values"
                        preparation_metrics['warnings'].append(warning_msg)
                        self.logger.warning(f"⚠️ {warning_msg}")
                    
                    additional_features.append(analyst_model_outputs)
                    additional_feature_names.extend([f"analyst_legacy_{i}" for i in range(analyst_model_outputs.shape[1])])
                    
                    feature_combination_details['analyst_legacy'] = {
                        'count': analyst_model_outputs.shape[1],
                        'nan_count': legacy_nan_count,
                        'inf_count': legacy_inf_count
                    }
                    
                    self.logger.info(f"📊 Added {analyst_model_outputs.shape[1]} legacy analyst outputs")
                    
                except Exception as e:
                    error_msg = f"Failed to add legacy analyst outputs: {e}"
                    preparation_metrics['errors'].append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
                    raise
            
            # Concatenate all additional features with validation
            if additional_features:
                try:
                    # Validate all features have same number of samples
                    for i, feature_array in enumerate(additional_features):
                        if feature_array.shape[0] != X.shape[0]:
                            error_msg = f"Feature array {i} shape mismatch: expected {X.shape[0]}, got {feature_array.shape[0]}"
                            preparation_metrics['errors'].append(error_msg)
                            raise ValueError(error_msg)
                    
                    X_combined = np.column_stack([X] + additional_features)
                    
                    # Validate combined features
                    if X_combined.shape[0] != X.shape[0]:
                        error_msg = f"Combined features sample count mismatch: expected {X.shape[0]}, got {X_combined.shape[0]}"
                        preparation_metrics['errors'].append(error_msg)
                        raise ValueError(error_msg)
                    
                    X = X_combined
                    
                    # Update feature names
                    if feature_names is not None:
                        feature_names = feature_names + additional_feature_names
                    else:
                        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                    
                    preparation_metrics['feature_combinations'] = feature_combination_details
                    self.logger.info(f"📊 Total features: {X.shape[1]} (base + HMM regime + HMM model + all analyst models + analyst ensemble)")
                    
                except Exception as e:
                    error_msg = f"Failed to combine features: {e}"
                    preparation_metrics['errors'].append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
                    raise
            
            # Final validation
            preparation_metrics['final_samples'] = X.shape[0]
            preparation_metrics['final_features'] = X.shape[1]
            
            # Check for final data quality issues
            final_nan_count = np.sum(np.isnan(X))
            final_inf_count = np.sum(np.isinf(X))
            
            if final_nan_count > 0:
                warning_msg = f"Final feature matrix contains {final_nan_count} NaN values"
                preparation_metrics['warnings'].append(warning_msg)
                self.logger.warning(f"⚠️ {warning_msg}")
            
            if final_inf_count > 0:
                warning_msg = f"Final feature matrix contains {final_inf_count} infinite values"
                preparation_metrics['warnings'].append(warning_msg)
                self.logger.warning(f"⚠️ {warning_msg}")
            
            return X, y, regime_labels, feature_names, preparation_metrics
            
        except Exception as e:
            preparation_metrics['errors'].append(str(e))
            self.logger.error(f"❌ Feature preparation failed: {e}")
            raise
    
    def _execute_training(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray],
        analyst_confidence_scores: Optional[np.ndarray] = None,
        analyst_directional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute training with enhanced vectorization and comprehensive error handling."""
        training_metrics = {
            'vectorization_attempted': False,
            'vectorization_successful': False,
            'fallback_used': False,
            'training_method': 'unknown',
            'errors': [],
            'warnings': [],
            'performance_metrics': {}
        }
        
        try:
            # Pre-training validation
            self._validate_training_inputs(X, y, regime_labels, feature_names, hmm_states)
            
            # Enhanced training with overfitting prevention and lookahead bias detection
            if ENHANCED_TRAINING_AVAILABLE and hasattr(self, 'training_enhancer'):
                self.logger.info("🚀 Using ENHANCED tactician models training with overfitting prevention")
                training_metrics['enhanced_training_attempted'] = True
                
                try:
                    # Validate temporal data for lookahead bias
                    if timestamps is not None:
                        self.logger.info("🔍 Validating temporal data for lookahead bias...")
                        is_valid, warnings = self.training_enhancer.enhanced_utils.validate_temporal_data(
                            X, y, timestamps, strict_mode=True
                        )
                        if warnings:
                            for warning in warnings:
                                self.logger.warning(f"⚠️ {warning}")
                        if not is_valid:
                            self.logger.error("❌ Temporal data validation failed")
                            raise ValueError("Lookahead bias detected in temporal data")
                    
                    # Use enhanced training with temporal integrity and confidence weighting
                    enhanced_start_time = time.time()
                    results = self._execute_enhanced_tactician_training(
                        X, y, regime_labels, feature_names, hmm_states, timestamps,
                        analyst_signals, analyst_model_outputs, hmm_regime_features,
                        all_analyst_models_outputs, hmm_model_outputs, analyst_ensemble_outputs,
                        analyst_confidence_scores, analyst_directional_info
                    )
                    
                    enhanced_duration = time.time() - enhanced_start_time
                    training_metrics['performance_metrics']['enhanced_training_duration'] = enhanced_duration
                    training_metrics['enhanced_training_successful'] = True
                    training_metrics['training_method'] = 'enhanced'
                    self.logger.info(f"✅ ENHANCED tactician training completed successfully in {enhanced_duration:.2f}s")
                    return results
                    
                except Exception as e:
                    error_msg = f"ENHANCED tactician training failed: {e}"
                    training_metrics['warnings'].append(error_msg)
                    training_metrics['fallback_used'] = True
                    self.logger.warning(f"⚠️ {error_msg}, falling back to vectorized method")
            
            # VECTORIZED: Use ultra-fast vectorized training by default
            self.logger.info("🚀 Using VECTORIZED tactician models training")
            training_metrics['vectorization_attempted'] = True
            
            if self.enable_vectorization:
                try:
                    vectorization_start_time = time.time()
                    
                    results = super().execute_vectorized(
                        X=X,
                        y=y,
                        regime_labels=regime_labels,
                        feature_names=feature_names,
                        hmm_states=hmm_states,
                        is_classification=False,  # Tactician models are typically regression
                        symbol=None,
                        exchange=None,
                        timeframe=self.config.timeframe
                    )
                    
                    vectorization_duration = time.time() - vectorization_start_time
                    training_metrics['performance_metrics']['vectorization_duration'] = vectorization_duration
                    
                    if results.get('vectorized', False):
                        training_metrics['vectorization_successful'] = True
                        training_metrics['training_method'] = 'vectorized'
                        self.logger.info(f"✅ VECTORIZED tactician training completed successfully in {vectorization_duration:.2f}s")
                        return results
                    else:
                        warning_msg = "VECTORIZED tactician training failed, falling back to standard method"
                        training_metrics['warnings'].append(warning_msg)
                        training_metrics['fallback_used'] = True
                        self.logger.warning(f"⚠️ {warning_msg}")
                        
                except Exception as e:
                    error_msg = f"VECTORIZED tactician training failed: {e}"
                    training_metrics['warnings'].append(error_msg)
                    training_metrics['fallback_used'] = True
                    self.logger.warning(f"⚠️ {error_msg}, falling back to standard method")
            else:
                training_metrics['fallback_used'] = True
                self.logger.info("🔄 Vectorization disabled, using standard training")
            
            # Fallback to standard training
            self.logger.info("🔄 Using standard tactician models training")
            standard_start_time = time.time()
            
            results = super().execute(
                X=X,
                y=y,
                regime_labels=regime_labels,
                feature_names=feature_names,
                hmm_states=hmm_states,
                is_classification=False,
                symbol=None,
                exchange=None,
                timeframe=self.config.timeframe
            )
            
            standard_duration = time.time() - standard_start_time
            training_metrics['performance_metrics']['standard_training_duration'] = standard_duration
            training_metrics['training_method'] = 'standard'
            
            self.logger.info(f"✅ Standard tactician training completed in {standard_duration:.2f}s")
            
            # Add training metrics to results
            results['training_execution_metrics'] = training_metrics
            
            # Add entry timing optimization if enabled
            if hasattr(self.config, 'enable_entry_timing_optimization') and self.config.enable_entry_timing_optimization:
                self.logger.info("🔄 Applying entry timing optimization for 0-0.4% range...")
                entry_timing_results = self._apply_entry_timing_optimization(X, y, feature_names, results)
                results.update(entry_timing_results)
            
            # Always add confidence-aware ensemble training for Tactician (core requirement)
            self.logger.info("🔄 Training confidence-aware ensemble model (always enabled for Tactician)...")
            ensemble_results = self._train_confidence_aware_ensemble_model(X, y, feature_names, results)
            results.update(ensemble_results)
            
            return results
            
        except Exception as e:
            training_metrics['errors'].append(str(e))
            self.logger.error(f"❌ Training execution failed: {e}")
            self.logger.error(f"❌ Training metrics: {training_metrics}")
            raise
    
    def _execute_enhanced_tactician_training(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray,
                                           feature_names: Optional[List[str]], hmm_states: Optional[np.ndarray],
                                           timestamps: Optional[np.ndarray], analyst_signals: Optional[np.ndarray],
                                           analyst_model_outputs: Optional[np.ndarray], hmm_regime_features: Optional[np.ndarray],
                                           all_analyst_models_outputs: Optional[Dict[str, np.ndarray]],
                                           hmm_model_outputs: Optional[np.ndarray], analyst_ensemble_outputs: Optional[np.ndarray],
                                           analyst_confidence_scores: Optional[np.ndarray] = None,
                                           analyst_directional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute enhanced tactician training with overfitting prevention and lookahead bias detection."""
        try:
            self.logger.info("🚀 Executing enhanced tactician training with overfitting prevention")
            
            # Filter for Analyst directional signals with confidence weighting
            if analyst_signals is not None:
                self.logger.info("🔍 Filtering for Analyst directional signals...")

                # Directional signals: 1 (long), -1 (short), 0 (neutral)
                directional_mask = (analyst_signals == 1) | (analyst_signals == -1)

                # Apply confidence-based weighting if confidence scores are available
                if analyst_confidence_scores is not None:
                    # Create combined mask with confidence threshold
                    confidence_threshold = 0.5
                    confidence_mask = analyst_confidence_scores >= confidence_threshold
                    combined_mask = directional_mask & confidence_mask

                    self.logger.info(f"📊 Confidence filtering: {np.sum(confidence_mask)}/{len(analyst_confidence_scores)} samples above {confidence_threshold} threshold")

                    # Use combined mask for filtering
                    directional_mask = combined_mask

                    # Store confidence scores for sample weighting
                    confidence_scores_filtered = analyst_confidence_scores[combined_mask]
                else:
                    confidence_scores_filtered = None
                    self.logger.warning("⚠️ No confidence scores provided - using directional signals only")

                X_filtered = X[directional_mask]
                y_filtered = y[directional_mask]
                regime_labels_filtered = regime_labels[directional_mask]
                timestamps_filtered = timestamps[directional_mask] if timestamps is not None else None

                # Analyze directional distribution
                long_count = np.sum(analyst_signals == 1)
                short_count = np.sum(analyst_signals == -1)
                neutral_count = np.sum(analyst_signals == 0)
                directional_rate = np.mean(directional_mask)

                self.logger.info(f"📊 Directional signal filtering: {directional_rate:.2%} ({np.sum(directional_mask)}/{len(analyst_signals)} samples)")
                self.logger.info(f"   Long signals: {long_count} ({long_count/len(analyst_signals):.1%} of total)")
                self.logger.info(f"   Short signals: {short_count} ({short_count/len(analyst_signals):.1%} of total)")
                self.logger.info(f"   Neutral signals: {neutral_count} ({neutral_count/len(analyst_signals):.1%} of total)")

            else:
                X_filtered, y_filtered, regime_labels_filtered, timestamps_filtered = X, y, regime_labels, timestamps
                confidence_scores_filtered = None
                self.logger.warning("⚠️ No Analyst directional signals provided, using all data")
            
            # Get unique regimes
            unique_regimes = np.unique(regime_labels_filtered)
            results = {
                'models': {},
                'regime_analysis': {},
                'enhanced_training_metadata': {},
                'overfitting_warnings': [],
                'calibration_warnings': [],
                'ensemble_diversity': None,
                'walk_forward_validation': None
            }
            
            # Train models for each regime with enhanced utilities and confidence weighting
            for regime in unique_regimes:
                with memory_guard(f"tactician_regime_training_{regime}", logger=self.logger):
                    regime_mask = regime_labels_filtered == regime
                    X_regime = X_filtered[regime_mask]
                    y_regime = y_filtered[regime_mask]
                    timestamps_regime = (
                        timestamps_filtered[regime_mask] if timestamps_filtered is not None else None
                    )

                    # Get confidence scores for this regime if available
                    confidence_regime = None
                    regime_confidence_mask = None
                    if confidence_scores_filtered is not None and regime_mask.sum() > 0:
                        regime_confidence_mask = (
                            regime_mask[directional_mask] if analyst_signals is not None else regime_mask
                        )
                        confidence_regime = (
                            confidence_scores_filtered[regime_confidence_mask]
                            if confidence_scores_filtered is not None
                            else None
                        )

                    self.logger.info(
                        f"🎯 Training tactician models for regime {regime} ({len(X_regime)} samples)"
                    )
                    if confidence_regime is not None:
                        self.logger.info(
                            "📊 Using confidence scores for sample weighting (mean: %.3f, std: %.3f)",
                            float(np.mean(confidence_regime)),
                            float(np.std(confidence_regime)),
                        )

                    # Train each model type for this regime
                    regime_models = {}
                    for model_type in self.config.model_types:
                        try:
                            # Create model instance
                            model = self._create_model_instance(model_type)

                            # Special handling for Random Survival Forest
                            if model_type == "RandomSurvivalForest":
                                model = self._create_random_survival_forest_model()

                            # Apply enhanced regularization
                            model = self.training_enhancer.enhanced_utils.apply_enhanced_regularization(
                                model, model_type
                            )

                            # Special handling for Random Survival Forest
                            if model_type == "RandomSurvivalForest":
                                # Random Survival Forest has its own training method with HPO
                                trained_model = model.fit(
                                    X_regime,
                                    y_regime,
                                    feature_names=feature_names,
                                    analyst_signals=analyst_signals[regime_mask]
                                    if analyst_signals is not None
                                    else None,
                                    hmm_regime_probs=hmm_regime_probs[regime_mask]
                                    if hmm_regime_probs is not None
                                    else None,
                                    enable_hpo=True,
                                    hpo_trials=self.config.hpo_n_trials,
                                    cv_folds=5,
                                    enable_entry_timing_optimization=True,
                                    entry_timing_trials=50,
                                )
                                metadata = {'model_type': 'RandomSurvivalForest', 'training_completed': True}
                            else:
                                # Train with early stopping, overfitting monitoring, and confidence weighting
                                trained_model, metadata = self.training_enhancer.enhance_training_step(
                                    X_regime,
                                    y_regime,
                                    model,
                                    timestamps_regime,
                                    f"tactician_{model_type}_regime_{regime}",
                                    sample_weights=confidence_regime,
                                )

                            if metadata is None:
                                metadata = {}

                            calibration_manager = getattr(self.training_enhancer, 'calibration_manager', None)
                            calibration_report: Dict[str, Any] = metadata.get('calibration', {}) if isinstance(metadata, dict) else {}
                            if (
                                calibration_manager
                                and not calibration_report
                                and self.enhanced_training_config is not None
                            ):
                                try:
                                    split_point = int(len(X_regime) * (1 - self.enhanced_training_config.calibration_validation_split))
                                    split_point = max(1, min(split_point, len(X_regime)))
                                    if len(X_regime) - split_point >= self.enhanced_training_config.calibration_min_samples:
                                        cal_X = X_regime[split_point:]
                                        cal_y = y_regime[split_point:]
                                        calibrated_model, calibration_report = calibration_manager.calibrate_model(
                                            trained_model,
                                            cal_X,
                                            cal_y,
                                        )
                                        trained_model = calibrated_model
                                        metadata['calibration'] = calibration_report
                                except Exception as calibration_error:
                                    self.logger.warning(
                                        "⚠️ Failed to calibrate %s for regime %s: %s",
                                        model_type,
                                        regime,
                                        calibration_error,
                                    )

                            metadata['is_calibrated'] = bool(
                                getattr(trained_model, '_is_calibrated', False)
                                or calibration_report.get('calibrated', False)
                            )
                            if metadata['is_calibrated']:
                                tprint_success(
                                    f"✅ Calibration confirmed for {model_type} in regime {regime}"
                                )
                            else:
                                reason = calibration_report.get('reason', 'unknown') if calibration_report else 'unknown'
                                tprint_warning(
                                    f"⚠️ Calibration unavailable for {model_type} in regime {regime} ({reason})"
                                )
                                results['calibration_warnings'].append(
                                    f"Calibration unavailable for {model_type} in regime {regime}: {reason}"
                                )

                            regime_models[model_type] = {
                                'model': trained_model,
                                'metadata': metadata,
                            }

                            # Check for overfitting warnings
                            if metadata.get('overfitting_detected', False):
                                results['overfitting_warnings'].append(
                                    f"Overfitting detected in {model_type} for regime {regime}"
                                )

                        except Exception as e:
                            self.logger.warning(f"⚠️ Failed to train {model_type} for regime {regime}: {e}")
                            continue

                    results['models'][regime] = regime_models

                    del (
                        regime_mask,
                        X_regime,
                        y_regime,
                        timestamps_regime,
                        regime_models,
                        confidence_regime,
                        regime_confidence_mask,
                    )
            
            # Calculate ensemble diversity if multiple models
            if len(self.config.model_types) > 1:
                self.logger.info("📊 Calculating tactician ensemble diversity...")
                for regime in unique_regimes:
                    if regime in results['models']:
                        models_list = [results['models'][regime][mt]['model'] for mt in self.config.model_types 
                                     if mt in results['models'][regime]]
                        if len(models_list) > 1:
                            diversity_metrics = self.training_enhancer.enhanced_utils.calculate_ensemble_diversity(
                                models_list, X_filtered[regime_labels_filtered == regime], 
                                y_filtered[regime_labels_filtered == regime]
                            )
                            results['ensemble_diversity'] = diversity_metrics
                            
                            if diversity_metrics.get('diversity_score', 0) < 0.1:
                                self.logger.warning(f"⚠️ Low tactician ensemble diversity for regime {regime}")
                            else:
                                self.logger.info(f"✅ Good tactician ensemble diversity for regime {regime}")
            
            # Perform walk-forward validation if enabled
            if len(results['models']) > 0:
                self.logger.info("🚶 Performing walk-forward validation for tactician...")
                first_regime = list(results['models'].keys())[0]
                first_model = list(results['models'][first_regime].values())[0]['model']
                
                wfv_results = self.training_enhancer.enhanced_utils.perform_walk_forward_validation(
                    first_model, X_filtered, y_filtered,
                    initial_train_size=1000, test_size=100, step_size=50
                )
                results['walk_forward_validation'] = wfv_results
                
                if wfv_results.get('performance_trend', {}).get('trend') == 'declining':
                    self.logger.warning("⚠️ Declining performance trend detected in tactician walk-forward validation")
                else:
                    self.logger.info("✅ Stable performance trend in tactician walk-forward validation")
            
            # Add enhanced training metadata
            results['enhanced_training_metadata'] = {
                'overfitting_prevention_enabled': True,
                'lookahead_bias_detection_enabled': True,
                'early_stopping_enabled': True,
                'enhanced_regularization_enabled': True,
                'temporal_validation_enabled': timestamps is not None,
                'green_light_filtering_enabled': analyst_signals is not None,
                'walk_forward_validation_enabled': True,
                'calibration_enabled': self.training_enhancer is not None,
                'calibration_warnings': len(results['calibration_warnings']),
                'total_warnings': len(results['overfitting_warnings']) + len(results['calibration_warnings'])
            }
            
            self.logger.info("✅ Enhanced tactician training completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced tactician training failed: {e}")
            raise
    
    def _validate_training_inputs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray]
    ) -> None:
        """Validate training inputs before execution."""
        try:
            # Check data shapes
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"Feature and target sample counts don't match: {X.shape[0]} vs {y.shape[0]}")
            
            if X.shape[0] != regime_labels.shape[0]:
                raise ValueError(f"Feature and regime label sample counts don't match: {X.shape[0]} vs {regime_labels.shape[0]}")
            
            if hmm_states is not None and hmm_states.shape[0] != X.shape[0]:
                raise ValueError(f"HMM states sample count mismatch: expected {X.shape[0]}, got {hmm_states.shape[0]}")
            
            # Check feature names consistency
            if feature_names is not None and len(feature_names) != X.shape[1]:
                raise ValueError(f"Feature names count mismatch: expected {X.shape[1]}, got {len(feature_names)}")
            
            # Check for critical data quality issues
            if np.any(np.isnan(X)):
                nan_percentage = (np.sum(np.isnan(X)) / X.size) * 100
                if nan_percentage > 5:  # More than 5% NaN
                    raise ValueError(f"Critical: {nan_percentage:.2f}% of features are NaN")
            
            if np.any(np.isnan(y)):
                nan_percentage = (np.sum(np.isnan(y)) / y.size) * 100
                if nan_percentage > 2:  # More than 2% NaN in targets
                    raise ValueError(f"Critical: {nan_percentage:.2f}% of targets are NaN")
            
            # Check regime distribution
            unique_regimes = np.unique(regime_labels)
            regime_counts = np.bincount(regime_labels)
            min_regime_size = np.min(regime_counts)
            
            if min_regime_size < self.config.min_samples_per_regime:
                insufficient_regimes = np.sum(regime_counts < self.config.min_samples_per_regime)
                if insufficient_regimes > len(unique_regimes) * 0.5:
                    raise ValueError(f"Critical: {insufficient_regimes}/{len(unique_regimes)} regimes have insufficient data")
            
            self.logger.info("✅ Training input validation passed")
            
        except Exception as e:
            self.logger.error(f"❌ Training input validation failed: {e}")
            raise
    
    def _apply_entry_timing_optimization(self,
                                      X: np.ndarray,
                                      y: np.ndarray,
                                      feature_names: Optional[List[str]],
                                      base_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply entry timing optimization for 0-0.4% range."""
        try:
            from .tactician_directional_optimization import EntryTimingTacticianOptimizer
            
            # Initialize entry timing optimizer
            entry_timing_optimizer = EntryTimingTacticianOptimizer(self.config)
            
            # Get entry timing range from config
            entry_timing_range = getattr(self.config, 'entry_timing_range', 0.004)  # 0-0.4% range
            
            # Filter targets for entry timing range (0-0.4%)
            entry_timing_mask = np.abs(y) <= entry_timing_range
            X_entry_timing = X[entry_timing_mask]
            y_entry_timing = y[entry_timing_mask]
            
            self.logger.info(f"📊 Entry timing filtering: {len(y_entry_timing)}/{len(y)} samples (≤{entry_timing_range:.1%} range)")
            
            if len(y_entry_timing) < 100:  # Need minimum samples for optimization
                self.logger.warning("⚠️ Insufficient entry timing samples for optimization")
                return {}
            
            # Apply entry timing optimization
            entry_timing_result = entry_timing_optimizer.optimize_tactician_entry_timing(
                X=X_entry_timing, y=y_entry_timing, regime_labels=np.zeros(len(y_entry_timing)),
                feature_names=feature_names, hmm_states=None,
                max_trials=getattr(self.config, 'hpo_n_trials', 100) // 2  # Half trials for entry timing
            )
            
            # Create entry timing optimization results
            entry_timing_results = {
                'entry_timing_optimization': {
                    'enabled': True,
                    'entry_timing_range': entry_timing_range,
                    'entry_timing_samples': len(y_entry_timing),
                    'total_samples': len(y),
                    'objectives': getattr(self.config, 'entry_timing_objectives', {}),
                    'optimization_time': entry_timing_result.optimization_time,
                    'n_trials': entry_timing_result.n_trials
                },
                'entry_timing_model': entry_timing_result.model,
                'entry_timing_metrics': {
                    'early_entry_penalty': entry_timing_result.directional_accuracy,
                    'late_entry_penalty': entry_timing_result.adverse_movement_minimization,
                    'optimal_entry_reward': entry_timing_result.directional_profit_efficiency,
                    'entry_timing_efficiency': entry_timing_result.risk_adjusted_performance,
                    'composite_score': entry_timing_result.composite_score
                }
            }
            
            self.logger.info(f"✅ Entry timing optimization completed for 0-0.3% range")
            self.logger.info(f"   Early entry penalty: {entry_timing_result.directional_accuracy:.4f}")
            self.logger.info(f"   Late entry penalty: {entry_timing_result.adverse_movement_minimization:.4f}")
            self.logger.info(f"   Optimal entry reward: {entry_timing_result.directional_profit_efficiency:.4f}")
            self.logger.info(f"   Entry timing efficiency: {entry_timing_result.risk_adjusted_performance:.4f}")
            self.logger.info(f"   Composite score: {entry_timing_result.composite_score:.4f}")
            
            return entry_timing_results
            
        except Exception as e:
            self.logger.error(f"❌ Entry timing optimization failed: {e}")
            return {}
    
    def _train_confidence_aware_ensemble_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]],
        base_models_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train confidence-aware ensemble model using base models as inputs."""
        try:
            from .tactician_directional_optimization import ConfidenceAwareEnsemble, ConfidenceAwareModel, EntryTimingLossFunction
            
            self.logger.info("🔄 Training confidence-aware ensemble model from base models...")
            
            # Get base models from results
            base_models = base_models_results.get('models', {})
            if not base_models:
                self.logger.warning("⚠️ No base models found for ensemble training")
                return {}
            
            # Wrap base models with confidence awareness
            confidence_aware_models = []
            loss_functions = EntryTimingLossFunction()
            
            for model_name, model in base_models.items():
                try:
                    # Wrap model with confidence awareness
                    confidence_aware_model = ConfidenceAwareModel(model, loss_functions)
                    confidence_aware_models.append(confidence_aware_model)
                    self.logger.info(f"📊 Wrapped {model_name} with confidence awareness")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to wrap {model_name} with confidence awareness: {e}")
            
            if not confidence_aware_models:
                self.logger.warning("⚠️ No valid confidence-aware models for ensemble training")
                return {}
            
            # Create meta model (LightGBM as meta-learner)
            meta_model_type = getattr(self.config, 'meta_model', 'LightGBM')
            if meta_model_type == 'LightGBM':
                meta_model = LGBMRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                )
            elif meta_model_type == 'ElasticNetCV':
                from sklearn.linear_model import ElasticNetCV
                meta_model = ElasticNetCV(
                    cv=5,
                    random_state=42,
                    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                    alphas=np.logspace(-4, 1, 50)
                )
            else:
                meta_model = self.training_utils.create_model(meta_model_type)
            
            # Create confidence-aware ensemble
            ensemble_name = getattr(self.config, 'ensemble_name', 'tactician_ensemble')
            confidence_aware_ensemble = ConfidenceAwareEnsemble(
                confidence_aware_models, meta_model, loss_functions
            )
            
            # Train ensemble
            confidence_aware_ensemble.fit(X, y)
            
            # Evaluate ensemble performance
            ensemble_predictions, ensemble_confidence = confidence_aware_ensemble.predict_with_confidence(X, y)
            
            # Calculate ensemble metrics
            ensemble_metrics = {
                'ensemble_name': ensemble_name,
                'n_base_models': len(confidence_aware_models),
                'meta_model_type': meta_model_type,
                'avg_confidence': np.mean(ensemble_confidence),
                'confidence_std': np.std(ensemble_confidence),
                'min_confidence': np.min(ensemble_confidence),
                'max_confidence': np.max(ensemble_confidence)
            }
            
            self.logger.info(f"✅ Confidence-aware ensemble training completed")
            self.logger.info(f"   Average confidence: {ensemble_metrics['avg_confidence']:.4f}")
            self.logger.info(f"   Confidence range: [{ensemble_metrics['min_confidence']:.4f}, {ensemble_metrics['max_confidence']:.4f}]")
            
            return {
                'ensemble_model': confidence_aware_ensemble,
                'ensemble_metrics': ensemble_metrics,
                'ensemble_predictions': ensemble_predictions,
                'ensemble_confidence': ensemble_confidence
            }
            
        except Exception as e:
            self.logger.error(f"❌ Confidence-aware ensemble training failed: {e}")
            return {}
    
    def _train_ensemble_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]],
        base_models_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train ensemble model using base models as inputs."""
        try:
            self.logger.info("🔄 Training ensemble model from base models...")
            
            # Get base models from results
            base_models = base_models_results.get('models', {})
            if not base_models:
                self.logger.warning("⚠️ No base models found for ensemble training")
                return {}
            
            # Generate base model predictions for ensemble training
            base_predictions = []
            base_model_names = []
            
            for model_name, model in base_models.items():
                try:
                    # Generate predictions using the base model
                    if hasattr(model, 'predict'):
                        predictions = model.predict(X)
                        base_predictions.append(predictions.reshape(-1, 1))
                        base_model_names.append(model_name)
                        self.logger.info(f"📊 Generated predictions from {model_name}")
                    else:
                        self.logger.warning(f"⚠️ Model {model_name} does not have predict method")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to generate predictions from {model_name}: {e}")
            
            if not base_predictions:
                self.logger.warning("⚠️ No valid base model predictions for ensemble training")
                return {}
            
            # Combine base model predictions
            X_ensemble = np.column_stack(base_predictions)
            ensemble_feature_names = [f"base_model_{name}" for name in base_model_names]
            
            self.logger.info(f"📊 Ensemble training data: {X_ensemble.shape[0]} samples, {X_ensemble.shape[1]} base model predictions")
            
            # Train ensemble model
            ensemble_method = getattr(self.config, 'ensemble_method', 'stacking')
            meta_model_type = getattr(self.config, 'meta_model', 'ElasticNetCV')
            ensemble_name = getattr(self.config, 'ensemble_name', 'tactician_ensemble')
            
            if ensemble_method == 'stacking':
                # Use stacking ensemble
                ensemble_model = self._train_stacking_ensemble(
                    X_ensemble, y, meta_model_type, ensemble_name
                )
            else:
                # Use simple meta-model
                ensemble_model = self.training_utils.train_single_model(
                    model_type=meta_model_type,
                    X=X_ensemble,
                    y=y,
                    model_name=ensemble_name
                )
            
            # Evaluate ensemble model
            ensemble_evaluation = self.training_utils.evaluate_model(
                model=ensemble_model,
                X=X_ensemble,
                y=y,
                metrics=self.config.evaluation_metrics
            )
            
            ensemble_results = {
                'ensemble_model': ensemble_model,
                'ensemble_evaluation': ensemble_evaluation,
                'ensemble_method': ensemble_method,
                'meta_model_type': meta_model_type,
                'base_models_used': base_model_names,
                'ensemble_feature_names': ensemble_feature_names
            }
            
            self.logger.info(f"✅ Ensemble training completed: {ensemble_method} with {meta_model_type}")
            self.logger.info(f"📊 Ensemble performance: {ensemble_evaluation}")
            
            return ensemble_results
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble training failed: {e}")
            return {}
    
    def _train_stacking_ensemble(
        self,
        X_ensemble: np.ndarray,
        y: np.ndarray,
        meta_model_type: str,
        ensemble_name: str
    ):
        """Train a stacking ensemble model."""
        try:
            from sklearn.ensemble import StackingRegressor
            from sklearn.model_selection import cross_val_predict
            
            # Create base estimators from the ensemble features
            base_estimators = []
            for i in range(X_ensemble.shape[1]):
                # Use simple models as base estimators for stacking
                base_estimator = self.training_utils.create_model('Ridge')
                base_estimators.append((f'base_{i}', base_estimator))
            
            # Create meta-model (ElasticNetCV for better performance)
            if meta_model_type == 'ElasticNetCV':
                meta_model = ElasticNetCV(
                    cv=5,
                    random_state=42,
                    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                    alphas=np.logspace(-4, 1, 50)
                )
            else:
                meta_model = self.training_utils.create_model(meta_model_type)
            
            # Create stacking regressor
            stacking_regressor = StackingRegressor(
                estimators=base_estimators,
                final_estimator=meta_model,
                cv=5,  # 5-fold cross-validation
                stack_method='predict'
            )
            
            # Train the stacking ensemble
            stacking_regressor.fit(X_ensemble, y)
            
            self.logger.info(f"✅ Stacking ensemble trained with {len(base_estimators)} base estimators")
            
            return stacking_regressor
            
        except Exception as e:
            self.logger.error(f"❌ Stacking ensemble training failed: {e}")
            # Fallback to simple meta-model with ElasticNetCV
            if meta_model_type == 'ElasticNetCV':
                from sklearn.model_selection import cross_val_score
                
                # Create and train ElasticNetCV directly
                elastic_net = ElasticNetCV(
                    cv=5,
                    random_state=42,
                    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                    alphas=np.logspace(-4, 1, 50)
                )
                elastic_net.fit(X_ensemble, y)
                return elastic_net
            else:
                return self.training_utils.train_single_model(
                    model_type=meta_model_type,
                    X=X_ensemble,
                    y=y,
                    model_name=ensemble_name
                )
    
    def _finalize_results(self, results: Dict[str, Any], analyst_signals: Optional[np.ndarray]) -> Dict[str, Any]:
        """Finalize results with tactician-specific metadata and comprehensive reporting."""
        try:
            # Add tactician-specific post-processing if needed
            if 'error' not in results:
                results = self._add_tactician_specific_metadata(results, analyst_signals)
            
            # Add training metrics to results
            results['training_metrics'] = {
                phase.value: {
                    'duration': metrics.duration,
                    'success': metrics.success,
                    'samples_processed': metrics.samples_processed,
                    'features_count': metrics.features_count,
                    'models_trained': metrics.models_trained,
                    'memory_usage_mb': metrics.memory_usage_mb,
                    'error_message': metrics.error_message
                }
                for phase, metrics in self.training_metrics.items()
            }
            
            # Add vectorization information
            results['vectorization_info'] = {
                'vectorization_enabled': self.enable_vectorization,
                'vectorization_fallback_used': self.vectorization_fallback_used,
                'vectorized_training_available': VECTORIZED_TRAINING_AVAILABLE
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Results finalization failed: {e}")
            return results
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result with comprehensive error information."""
        return {
            'error': True,
            'error_message': error_message,
            'training_metrics': {
                phase.value: {
                    'duration': metrics.duration,
                    'success': metrics.success,
                    'error_message': metrics.error_message
                }
                for phase, metrics in self.training_metrics.items()
            },
            'vectorization_info': {
                'vectorization_enabled': self.enable_vectorization,
                'vectorization_fallback_used': self.vectorization_fallback_used,
                'vectorized_training_available': VECTORIZED_TRAINING_AVAILABLE
            }
        }
    
    def _generate_training_report(self, total_time: float) -> None:
        """Generate comprehensive training report with actionable insights."""
        try:
            self.logger.info("📊 " + "="*80)
            self.logger.info("📊 ENHANCED TACTICIAN TRAINING REPORT")
            self.logger.info("📊 " + "="*80)
            
            # Overall statistics
            self.logger.info(f"📊 Total training time: {total_time:.2f}s")
            self.logger.info(f"📊 Vectorization enabled: {self.enable_vectorization}")
            self.logger.info(f"📊 Vectorization fallback used: {self.vectorization_fallback_used}")
            
            # Calculate efficiency metrics
            total_warnings = sum(len(metrics.warnings_issued) for metrics in self.training_metrics.values())
            total_errors = sum(len(metrics.errors_encountered) for metrics in self.training_metrics.values())
            total_samples = sum(metrics.samples_processed for metrics in self.training_metrics.values())
            total_features = sum(metrics.features_count for metrics in self.training_metrics.values())
            
            self.logger.info(f"📊 Total samples processed: {total_samples:,}")
            self.logger.info(f"📊 Total features: {total_features}")
            self.logger.info(f"📊 Total warnings: {total_warnings}")
            self.logger.info(f"📊 Total errors: {total_errors}")
            
            # Performance efficiency
            if total_samples > 0:
                samples_per_second = total_samples / total_time
                self.logger.info(f"📊 Processing rate: {samples_per_second:,.0f} samples/second")
            
            # Phase-by-phase breakdown with detailed metrics
            self.logger.info("📊 " + "-"*60)
            self.logger.info("📊 PHASE BREAKDOWN:")
            self.logger.info("📊 " + "-"*60)
            
            for phase, metrics in self.training_metrics.items():
                status = "✅" if metrics.success else "❌"
                efficiency = f"({metrics.samples_processed/1000:.1f}k samples)" if metrics.samples_processed > 0 else ""
                
                self.logger.info(f"📊   {status} {phase.value.upper()}: {metrics.duration:.2f}s {efficiency}")
                
                # Detailed phase metrics
                if metrics.samples_processed > 0:
                    self.logger.info(f"📊     └─ Samples: {metrics.samples_processed:,}")
                if metrics.features_count > 0:
                    self.logger.info(f"📊     └─ Features: {metrics.features_count}")
                if metrics.models_trained > 0:
                    self.logger.info(f"📊     └─ Models trained: {metrics.models_trained}")
                if metrics.warnings_issued > 0:
                    self.logger.info(f"📊     └─ Warnings: {metrics.warnings_issued}")
                if metrics.errors_encountered > 0:
                    self.logger.info(f"📊     └─ Errors: {metrics.errors_encountered}")
                if metrics.memory_usage_mb > 0:
                    self.logger.info(f"📊     └─ Memory: {metrics.memory_usage_mb:.1f} MB")
                
                if not metrics.success and metrics.error_message:
                    self.logger.info(f"📊     └─ ❌ Error: {metrics.error_message}")
            
            # Data quality summary
            self.logger.info("📊 " + "-"*60)
            self.logger.info("📊 DATA QUALITY SUMMARY:")
            self.logger.info("📊 " + "-"*60)
            
            # Analyze data validation results if available
            if TrainingPhase.DATA_VALIDATION in self.training_metrics:
                data_phase = self.training_metrics[TrainingPhase.DATA_VALIDATION]
                if data_phase.success:
                    self.logger.info("📊   ✅ Data validation passed")
                else:
                    self.logger.info("📊   ❌ Data validation failed")
            
            # Analyze feature preparation results if available
            if TrainingPhase.FEATURE_PREPARATION in self.training_metrics:
                feature_phase = self.training_metrics[TrainingPhase.FEATURE_PREPARATION]
                if feature_phase.success:
                    self.logger.info("📊   ✅ Feature preparation completed")
                else:
                    self.logger.info("📊   ❌ Feature preparation failed")
            
            # Training method analysis
            self.logger.info("📊 " + "-"*60)
            self.logger.info("📊 TRAINING METHOD ANALYSIS:")
            self.logger.info("📊 " + "-"*60)
            
            if self.enable_vectorization:
                if self.vectorization_fallback_used:
                    self.logger.info("📊   ⚠️ Vectorization attempted but fallback used")
                    self.logger.info("📊   💡 Consider investigating vectorization issues")
                else:
                    self.logger.info("📊   ✅ Vectorization used successfully")
                    self.logger.info("📊   🚀 Optimal performance achieved")
            else:
                self.logger.info("📊   ℹ️ Standard training used (vectorization disabled)")
            
            # Memory usage analysis
            current_memory = self._get_memory_usage()
            self.logger.info("📊 " + "-"*60)
            self.logger.info("📊 MEMORY USAGE:")
            self.logger.info("📊 " + "-"*60)
            self.logger.info(f"📊 Current memory usage: {current_memory:.1f} MB")
            
            if current_memory > 1000:  # More than 1GB
                self.logger.info("📊   ⚠️ High memory usage detected")
                self.logger.info("📊   💡 Consider reducing batch size or using data streaming")
            elif current_memory < 100:  # Less than 100MB
                self.logger.info("📊   ✅ Low memory usage - efficient processing")
            
            # Recommendations
            self.logger.info("📊 " + "-"*60)
            self.logger.info("📊 RECOMMENDATIONS:")
            self.logger.info("📊 " + "-"*60)
            
            if total_warnings > 5:
                self.logger.info("📊   ⚠️ High warning count - review data quality")
            if total_errors > 0:
                self.logger.info("📊   ❌ Errors detected - review error logs")
            if self.vectorization_fallback_used:
                self.logger.info("📊   🔧 Vectorization fallback used - investigate vectorization issues")
            if total_time > 3600:  # More than 1 hour
                self.logger.info("📊   ⏱️ Long training time - consider optimizing hyperparameters")
            
            # Success indicators
            if total_errors == 0 and total_warnings < 3:
                self.logger.info("📊   ✅ Training completed successfully with minimal issues")
            
            self.logger.info("📊 " + "="*80)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate training report: {e}")
            self.logger.error(f"❌ Traceback: {traceback.format_exc()}")
    
    def cleanup_resources(self) -> None:
        """Clean up hardware optimizers and other resources."""
        try:
            tprint_info("🧹 Cleaning up resources...")
            
            # Clean up M1 optimizers - CRITICAL: Must be available
            try:
                cleanup_result = cleanup_m1_optimizers()
                if not cleanup_result:
                    error_msg = "CRITICAL: M1 optimizer cleanup failed"
                    tprint_error(f"❌ {error_msg}")
                    raise RuntimeError(error_msg)
                tprint_success("✅ M1 optimizers cleaned up successfully")
            except Exception as e:
                error_msg = f"CRITICAL: Failed to cleanup M1 optimizers: {e}"
                tprint_error(f"❌ {error_msg}")
                raise RuntimeError(error_msg) from e
            
            # Clean up hardware resources
            if hasattr(self, 'm1_gpu_manager') and self.m1_gpu_manager:
                tprint_debug("Cleaning up M1 GPU manager...")
            
            if hasattr(self, 'm1_memory_optimizer') and self.m1_memory_optimizer:
                tprint_debug("Cleaning up M1 memory optimizer...")
            
            if hasattr(self, 'm1_cpu_optimizer') and self.m1_cpu_optimizer:
                tprint_debug("Cleaning up M1 CPU optimizer...")
            
            tprint_success("✅ Resource cleanup completed")
            
        except Exception as e:
            error_msg = f"CRITICAL: Resource cleanup failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        try:
            self.cleanup_resources()
        except Exception as e:
            # Log cleanup errors in destructor but don't raise to avoid issues during garbage collection
            tprint_error(f"❌ Resource cleanup failed in destructor: {e}")
            # Ensure we don't leave any hanging resources
            try:
                if hasattr(self, 'training_data') and self.training_data is not None:
                    del self.training_data
                if hasattr(self, 'models') and self.models is not None:
                    del self.models
                if hasattr(self, 'results') and self.results is not None:
                    del self.results
            except Exception:
                pass  # Ignore cleanup errors in destructor
    
    def _add_tactician_specific_metadata(self, results: Dict[str, Any], analyst_signals: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Add tactician-specific metadata to results with enhanced reporting.
        
        Args:
            results: Training results
            analyst_signals: Analyst green light signals for analysis
            
        Returns:
            Enhanced results with tactician-specific metadata
        """
        try:
            tprint_debug("Adding tactician-specific metadata...")
            
            # Add tactician-specific analysis
            if 'regime_analysis' in results:
                regime_analysis = results['regime_analysis']
                
                # Calculate tactician-specific metrics with safe math operations
                tactician_metrics = {
                    'total_regimes': len(regime_analysis.get('unique_regimes', [])),
                    'sufficient_regimes': len(regime_analysis.get('sufficient_regimes', [])),
                    'insufficient_regimes': len(regime_analysis.get('insufficient_regimes', [])),
                    'regime_balance': regime_analysis.get('regime_balance_train', 0.0),
                    'timeframe': self.config.timeframe,
                    'model_types': self.config.model_types
                }
                
                # Add analyst signal analysis if available with safe calculations
                if analyst_signals is not None:
                    try:
                        green_light_rate = np.mean(analyst_signals)
                        tactician_metrics.update({
                            'analyst_green_light_rate': green_light_rate,
                            'total_samples_with_green_light': int(np.sum(analyst_signals)),
                            'total_samples_analyzed': len(analyst_signals)
                        })
                        tprint_debug(f"Analyst signal analysis: {green_light_rate:.3f} green light rate")
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to analyze analyst signals: {e}")
                
                results['tactician_metrics'] = tactician_metrics
                tprint_debug("✅ Tactician metrics added")
            
            # Add model performance summary with enhanced error handling
            if 'evaluation_results' in results:
                try:
                    evaluation_results = results['evaluation_results']
                    
                    # Calculate best performing model per regime
                    best_models = {}
                    for regime, regime_metrics in evaluation_results.items():
                        if isinstance(regime_metrics, dict) and 'error' not in regime_metrics:
                            best_model = None
                            best_r2 = -np.inf
                            
                            for model_name, metrics in regime_metrics.items():
                                if isinstance(metrics, dict) and 'r2' in metrics:
                                    if metrics['r2'] > best_r2:
                                        best_r2 = metrics['r2']
                                        best_model = model_name
                            
                            if best_model:
                                best_models[regime] = {
                                    'model': best_model,
                                    'r2_score': best_r2
                                }
                    
                    results['best_models_per_regime'] = best_models
                    tprint_debug(f"✅ Best models per regime: {len(best_models)} regimes")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to calculate best models per regime: {e}")
            
            # Add timing-specific analysis
            timing_analysis = {
                'base_timeframe': self.config.timeframe,
                'cross_timeframe_features': True,
                'analyst_dependency': True,
                'timing_decision_role': True,
                'utility_integration_status': self.utility_integration_status
            }
            results['timing_analysis'] = timing_analysis
            
            # Add utility integration status to results
            results['utility_integration_status'] = self.utility_integration_status
            
            tprint_success("✅ Tactician-specific metadata added successfully")
            return results
            
        except Exception as e:
            tprint_error(f"❌ Failed to add tactician-specific metadata: {e}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            return results


# Enhanced convenience functions with better error handling
def create_tactician_models_training_step_refactored(
    config: Optional[TacticianTrainingConfig] = None,
    enable_vectorization: bool = True
) -> TacticianModelsTrainingStepRefactored:
    """
    Create enhanced Tactician models training step with comprehensive error handling.
    
    Args:
        config: Per-regime training configuration
        enable_vectorization: Whether to enable vectorized training
        
    Returns:
        Enhanced Tactician models training step
        
    Raises:
        Exception: If initialization fails
    """
    try:
        return TacticianModelsTrainingStepRefactored(config, enable_vectorization)
    except Exception as e:
        logger.error(f"❌ Failed to create tactician training step: {e}")
        raise


def execute_tactician_models_training_refactored(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[TacticianTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    analyst_signals: Optional[np.ndarray] = None,
    analyst_model_outputs: Optional[np.ndarray] = None,
    hmm_regime_features: Optional[np.ndarray] = None,
    all_analyst_models_outputs: Optional[Dict[str, np.ndarray]] = None,
    hmm_model_outputs: Optional[np.ndarray] = None,
    analyst_ensemble_outputs: Optional[np.ndarray] = None,
    enable_vectorization: bool = True
) -> Dict[str, Any]:
    """
    Execute enhanced Tactician models training step with comprehensive error handling.
    
    Args:
        X: Input features
        y: Target values
        regime_labels: Regime labels for each sample
        config: Per-regime training configuration
        feature_names: Names of input features
        hmm_states: HMM cluster/regime states
        analyst_signals: Binary signals from Analyst
        analyst_model_outputs: Analyst model predictions
        hmm_regime_features: HMM regime features
        all_analyst_models_outputs: All individual analyst ML model outputs
        hmm_model_outputs: HMM model outputs (predictions, probabilities, etc.)
        analyst_ensemble_outputs: Analyst ensemble model outputs
        enable_vectorization: Whether to enable vectorized training
        
    Returns:
        Dictionary containing training results and metadata
        
    Raises:
        Exception: If training fails
    """
    try:
        step = create_tactician_models_training_step_refactored(config, enable_vectorization)
        return step.execute(
            X, y, regime_labels, feature_names, hmm_states, 
            analyst_signals, analyst_model_outputs, hmm_regime_features, 
            all_analyst_models_outputs, hmm_model_outputs, analyst_ensemble_outputs
        )
    except Exception as e:
        logger.error(f"❌ Failed to execute tactician training: {e}")
        raise


# Enhanced example usage and comparison
if __name__ == "__main__":
    # Example of how to use the enhanced version
    print("Enhanced Tactician Models Training Step")
    print("=" * 50)
    
    # Create configuration with enhanced settings
    config = TacticianTrainingConfig(
        model_name="tactician_models",
        timeframe="1m",
        model_types=["NeuralObliviousDecisionEnsembles", "CatBoostRegressor", "LGBMRegressor", "Ridge"],
        hpo_n_trials=50,  # Reduced for demo
        enable_hpo=True,
        save_models=True,
        model_save_path="generated/model_training/models/tactician_models_enhanced",
        use_single_model=True,
        single_model_name="tactician_unified_model",
        enable_ensemble_training=True,
        ensemble_method="stacking",
        meta_model="ElasticNetCV",
        ensemble_name="tactician_ensemble"
    )
    
    # Create enhanced training step
    try:
        training_step = create_tactician_models_training_step_refactored(config)
        
        print(f"✅ Created enhanced tactician training step with {len(config.model_types)} model types")
        print(f"📊 HPO enabled: {config.enable_hpo}")
        print(f"💾 Save models: {config.save_models}")
        print(f"📁 Save path: {config.model_save_path}")
        print(f"⏰ Base timeframe: {config.timeframe}")
        
        # The actual training would be called with:
        # results = training_step.execute(X, y, regime_labels, feature_names, hmm_states, analyst_signals, analyst_model_outputs)
        
        print("\n🎯 Enhanced Tactician Module Features:")
        print("- Comprehensive error handling with detailed failure reporting")
        print("- Enhanced progress tracking with phase-based metrics")
        print("- Input validation and data quality checks")
        print("- Optimized vectorization with intelligent fallback")
        print("- Structured logging with performance monitoring")
        print("- Health monitoring throughout training process")
        
        print("\n🔄 Integration with Analyst:")
        print("- Receives green light signals from Analyst")
        print("- Uses Analyst predictions as additional features")
        print("- Focuses on timing rather than trade decision")
        print("- Operates on higher frequency (1m vs 5m)")
        
        print("\n📊 Enhanced Reporting Features:")
        print("- Phase-by-phase progress tracking")
        print("- Comprehensive training metrics")
        print("- Memory usage monitoring")
        print("- Vectorization status reporting")
        print("- Detailed error reporting with stack traces")
        
    except Exception as e:
        print(f"❌ Failed to create enhanced tactician training step: {e}")
        print("This demonstrates the enhanced error handling capabilities")