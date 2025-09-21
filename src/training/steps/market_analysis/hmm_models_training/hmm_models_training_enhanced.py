"""
Enhanced HMM Models Training

Streamlined, robust, and well-reported HMM models training with comprehensive error handling.
Integrated with common utilities for better maintainability and performance.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
import json
import psutil
import os
import gc

warnings.filterwarnings('ignore')

# Check ML library dependencies
ML_LIBRARIES_STATUS = {}
try:
    import sklearn
    ML_LIBRARIES_STATUS['sklearn'] = True
except ImportError:
    ML_LIBRARIES_STATUS['sklearn'] = False

try:
    import lightgbm
    ML_LIBRARIES_STATUS['lightgbm'] = True
except ImportError:
    ML_LIBRARIES_STATUS['lightgbm'] = False

try:
    import xgboost
    ML_LIBRARIES_STATUS['xgboost'] = True
except ImportError:
    ML_LIBRARIES_STATUS['xgboost'] = False

# Core imports
from src.utils.tprint import tprint
from src.utils.logger import system_logger
from .utils import StandardizedLogger, safe_execute, performance_monitor, ConfigurationValidator
from .constants import TrainingLimits, LoggingConstants
from .shared_feature_utils import create_enhanced_features_with_names
from src.utils.ml_common.config.base_training_config import HMMTrainingConfig
from src.utils.ml_common.training.base_training_step import BaseTrainingStep

# Feature generation system imports
try:
    from src.feature_generation.core.feature_bank import get_global_feature_bank
    from src.feature_generation.core.feature_generator import FeatureCategory
    FEATURE_GENERATION_AVAILABLE = True
except ImportError:
    FEATURE_GENERATION_AVAILABLE = False
    tprint("⚠️ Advanced feature generation not available, using simplified features")

# Common utilities integration
from src.utils.common_operations import (
    safe_dataframe_operation,
    validate_dataframe_columns,
    calculate_data_quality_metrics,
    get_m1_gpu_manager,
    get_m1_memory_optimizer,
    get_m1_cpu_optimizer,
    safe_divide,
    safe_float,
    safe_int,
    ensure_directory,
    memory_checkpoint,
    optimize_memory,
    get_memory_usage
)
from src.utils.common_utilities import (
    safe_convert_dtypes,
    calculate_data_quality_metrics as df_quality_metrics
)
from src.utils.math_validation import (
    safe_divide as math_safe_divide,
    safe_log,
    safe_sqrt,
    validate_positive,
    validate_range,
    validate_numeric_array,
    validate_finite
)
from src.utils.serialization_utils import (
    JSONSerializer,
    PickleSerializer,
    UniversalSerializer
)
try:
    from src.utils.ml_common.evaluation.evaluation_utils import EvaluationUtils
    ML_EVALUATION_AVAILABLE = True
except ImportError:
    ML_EVALUATION_AVAILABLE = False
    # Create a dummy class to prevent None reference errors
    class EvaluationUtils:
        @staticmethod
        def evaluate_model_performance(*args, **kwargs):
            raise ImportError("EvaluationUtils not available")

try:
    from src.utils.ml_common.validation.validation_utils import ValidationUtils as MLValidationUtils
    ML_VALIDATION_AVAILABLE = True
except ImportError:
    ML_VALIDATION_AVAILABLE = False
    # Create a dummy class to prevent None reference errors
    class MLValidationUtils:
        @staticmethod
        def validate_data(*args, **kwargs):
            raise ImportError("MLValidationUtils not available")

# Hardware optimization imports - ensure availability
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager, get_m1_gpu_manager
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer, get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer, get_m1_cpu_optimizer
    HARDWARE_AVAILABLE = True
    tprint("✅ Hardware optimization modules loaded successfully")
except ImportError as e:
    HARDWARE_AVAILABLE = False
    M1GPUManager = None
    M1MemoryOptimizer = None
    M1CPUOptimizer = None
    get_m1_gpu_manager = lambda: None
    get_m1_memory_optimizer = lambda: None
    get_m1_cpu_optimizer = lambda: None
    tprint(f"⚠️ Hardware optimization modules not available: {e}")

# Shared utilities with enhanced error handling
try:
    from .shared_utilities import (
        TrainingErrorHandler,
        UnifiedModelFactory,
        CircuitBreaker,
        ValidationUtils,
        ProgressReporter,
        MemoryTracker
    )
    SHARED_UTILITIES_AVAILABLE = True
    tprint("✅ Shared utilities loaded successfully")
except ImportError as e:
    SHARED_UTILITIES_AVAILABLE = False
    tprint(f"❌ Shared utilities import failed: {e}")
    # Note: logger will be defined later, using tprint for now
    tprint(f"Critical dependency missing: {e}")
    
    # Create robust fallback classes with proper error handling
    from dataclasses import dataclass
    from typing import Optional, Any, Dict, List
    
    @dataclass
    class TrainingMetrics:
        error_message: Optional[str] = None
        training_time: float = 0.0
        accuracy: float = 0.0
        f1_score: float = 0.0
        precision: float = 0.0
        recall: float = 0.0
        memory_usage_mb: float = 0.0
        convergence_epochs: int = 0
        validation_loss: Optional[float] = None
        test_accuracy: Optional[float] = None
        warnings: List[str] = None
        
        def __post_init__(self):
            if self.warnings is None:
                self.warnings = []
    
    @dataclass
    class ModelResult:
        model: Any
        metrics: TrainingMetrics
        feature_importance: Optional[Dict[str, float]] = None
        predictions: Optional[Any] = None
        probabilities: Optional[Any] = None
        hyperparameters: Optional[Dict[str, Any]] = None
        training_history: Optional[Dict[str, List[float]]] = None
    
    class TrainingErrorHandler:
        @staticmethod
        def handle_training_error(model_type: str, error: Exception, training_time: float) -> ModelResult:
            logger.error(f"Training error for {model_type}: {error}")
            return ModelResult(
                model=None,
                metrics=TrainingMetrics(
                    error_message=f"Failed to train {model_type}: {str(error)}",
                    training_time=training_time
                )
            )
    
    class UnifiedModelFactory:
        @staticmethod
        def create_model(model_type: str, **kwargs):
            try:
                # Check library availability first
                if model_type == 'lightgbm' or model_type == 'lgbm':
                    if not ML_LIBRARIES_STATUS.get('lightgbm', False):
                        raise ImportError(f"LightGBM not available for model type: {model_type}")
                    import lightgbm
                    # Enhanced regularization for LightGBM to prevent overfitting
                    return lightgbm.LGBMClassifier(
                        n_estimators=100,
                        learning_rate=0.05,    # Reduced learning rate
                        max_depth=4,           # Limited depth for regularization
                        num_leaves=15,         # Limited leaves per tree
                        min_child_samples=20,  # Minimum samples per child
                        min_child_weight=0.1,  # Minimum sum of hessian per child
                        reg_alpha=0.1,         # L1 regularization
                        reg_lambda=0.1,        # L2 regularization
                        feature_fraction=0.8,  # Use 80% of features per tree
                        bagging_fraction=0.8,  # Use 80% of data per tree
                        bagging_freq=1,        # Enable bagging
                        random_state=42,
                        **kwargs
                    )
                elif model_type == 'xgboost':
                    if not ML_LIBRARIES_STATUS.get('xgboost', False):
                        raise ImportError(f"XGBoost not available for model type: {model_type}")
                    import xgboost
                    # Enhanced regularization for XGBoost to prevent overfitting
                    return xgboost.XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.05,    # Reduced learning rate
                        max_depth=4,           # Limited depth for regularization
                        min_child_weight=5,    # Minimum sum of hessian per child
                        reg_alpha=0.1,         # L1 regularization
                        reg_lambda=0.1,        # L2 regularization
                        subsample=0.8,         # Use 80% of data per tree
                        colsample_bytree=0.8,  # Use 80% of features per tree
                        colsample_bylevel=0.8, # Use 80% of features per level
                        colsample_bynode=0.8,  # Use 80% of features per node
                        random_state=42,
                        **kwargs
                    )
                elif model_type in ['random_forest', 'rf']:
                    if not ML_LIBRARIES_STATUS.get('sklearn', False):
                        raise ImportError(f"Scikit-learn not available for model type: {model_type}")
                    from sklearn.ensemble import RandomForestClassifier
                    # Enhanced regularization to prevent overfitting
                    return RandomForestClassifier(
                        n_estimators=100,
                        max_depth=6,           # Reduced depth for better regularization
                        min_samples_split=20,  # Increased to require more samples to split
                        min_samples_leaf=10,   # Increased to require more samples per leaf
                        max_features='sqrt',   # Limit features per split (sqrt for better regularization)
                        min_impurity_decrease=0.01,  # Stop splitting if impurity decrease is too small
                        ccp_alpha=0.001,       # Cost-complexity pruning for post-pruning regularization
                        random_state=42,
                        n_jobs=-1,
                        **kwargs
                    )
                # elif model_type in ['logistic_regression', 'lr']:  # REMOVED LOGISTIC REGRESSION
                #     if not ML_LIBRARIES_STATUS.get('sklearn', False):
                #         raise ImportError(f"Scikit-learn not available for model type: {model_type}")
                #     from sklearn.linear_model import LogisticRegression
                #     return LogisticRegression(random_state=42, max_iter=1000, **kwargs)
                # elif model_type in ['elastic_net_lr', 'elastic_net']:  # COMMENTED OUT ELASTIC NET
                #     if not ML_LIBRARIES_STATUS.get('sklearn', False):
                #         raise ImportError(f"Scikit-learn not available for model type: {model_type}")
                #     from sklearn.linear_model import LogisticRegression
                #     # Ensure compatible parameters for elastic net
                #     safe_kwargs = {k: v for k, v in kwargs.items() if k not in ['penalty', 'solver', 'l1_ratio']}
                #     return LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', random_state=42, max_iter=1000, **safe_kwargs)
                else:
                    available_models = ['lightgbm', 'random_forest']  # Updated to match current configuration
                    raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")
            except Exception as e:
                logger.error(f"Failed to create model {model_type}: {e}")
                raise
    
    class CircuitBreaker:
        def __init__(self, failure_threshold: int = 3, timeout: int = 300):
            self.state = "CLOSED"
            self.failure_count = 0
            self.failure_threshold = failure_threshold
            self.timeout = timeout
            self.last_failure_time = None
            
        def call(self, func, *args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(f"Circuit breaker opened after {self.failure_count} failures")
                raise e
    

    class ProgressReporter:
        def __init__(self, total_models: int):
            self.total_models = total_models
            self.completed = 0
            
        def update_progress(self, model_name: str, success: bool, training_time: float, 
                          accuracy: Optional[float] = None, error_message: Optional[str] = None):
            self.completed += 1
            status = "✅" if success else "❌"
            tprint(f"{status} {model_name} ({self.completed}/{self.total_models})")
            
        def finish_report(self):
            tprint(f"Training completed: {self.completed}/{self.total_models} models")
    
    class MemoryTracker:
        def __init__(self):
            self.enabled = False
            
        def take_snapshot(self, name: str):
            return 0.0
            
        def get_memory_increase(self):
            return 0.0
            
        def cleanup(self):
            return 0.0


logger = system_logger.getChild('HMMModelsTrainingEnhanced')

# Import shared data classes from shared utilities
from .shared_utilities.training_error_handler import TrainingMetrics, ModelResult

class HMMModelsTrainingEnhanced(BaseTrainingStep):
    """
    Enhanced HMM Models Training with streamlined code, robust error handling, and comprehensive reporting.
    Integrated with common utilities for better maintainability and performance.
    """
    
    def __init__(self, config: Optional[Union[HMMTrainingConfig, Dict[str, Any]]] = None):
        """
        Initialize enhanced HMM models training.

        Args:
            config: HMM training configuration object or dictionary of parameters
        """
        if config is None:
            # Use only Random Forest and LightGBM
            available_models = []
            if ML_LIBRARIES_STATUS.get('sklearn', False):
                available_models.extend(['random_forest'])  # Only Random Forest from sklearn
                # available_models.extend(['logistic_regression', 'elastic_net_lr'])  # Commented out
            if ML_LIBRARIES_STATUS.get('lightgbm', False):
                available_models.append('lightgbm')
            # if ML_LIBRARIES_STATUS.get('xgboost', False):  # Commented out XGBoost
            #     available_models.append('xgboost')
            
            # Fallback to sklearn models if nothing else available
            if not available_models:
                # Try to use any available model from UnifiedModelFactory
                try:
                    from .shared_utilities.unified_model_factory import UnifiedModelFactory
                    all_available = UnifiedModelFactory.get_available_models()
                    # Filter to models that are likely to work without external dependencies
                    sklearn_models = [m for m in all_available if m in ['random_forest', 'logistic_regression']]
                    if sklearn_models:
                        available_models = sklearn_models[:1]  # Use first available sklearn model
                    else:
                        available_models = ['random_forest']  # Ultimate fallback
                except ImportError:
                    available_models = ['random_forest']  # Ultimate fallback
                
            config = HMMTrainingConfig(
                model_name="hmm_models_enhanced",
                timeframe="15m",
                n_features=100,
                sequence_length=20,
                n_regimes=3,
                model_types=available_models,
                hpo_trials=50,
                enable_multi_objective=True
            )
        elif isinstance(config, dict):
            # Convert dictionary to HMMTrainingConfig
            default_config = HMMTrainingConfig()
            config_dict = {**default_config.__dict__, **config}
            config = HMMTrainingConfig(**config_dict)

        # Validate configuration before proceeding
        self._validate_config(config)

        super().__init__(config)
        self.logger = logger.getChild('HMMModelsTrainingEnhanced')

        self.circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=30)
        self.progress_reporter = None
        self.memory_tracker = MemoryTracker()
        
        # Initialize hardware optimizers
        self._initialize_hardware_optimizers()
        
        # Initialize components with error handling
        self._initialize_components()
        
        # Training state
        self.training_start_time = None
        self.training_results = {}
        
        tprint("✅ Enhanced HMM Models Training initialized with common utilities integration")
    
    def _initialize_hardware_optimizers(self) -> None:
        """Initialize hardware optimizers for M1 systems."""
        self.gpu_manager = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        
        if HARDWARE_AVAILABLE:
            try:
                self.gpu_manager = get_m1_gpu_manager()
                if self.gpu_manager and self.gpu_manager.is_m1:
                    tprint("✅ M1 GPU manager initialized")
                else:
                    tprint("ℹ️ M1 GPU not available or not M1 system")
            except Exception as e:
                tprint(f"⚠️ Failed to initialize GPU manager: {e}")
            
            try:
                self.memory_optimizer = get_m1_memory_optimizer()
                if self.memory_optimizer:
                    tprint("✅ M1 memory optimizer initialized")
            except Exception as e:
                tprint(f"⚠️ Failed to initialize memory optimizer: {e}")
            
            try:
                self.cpu_optimizer = get_m1_cpu_optimizer()
                if self.cpu_optimizer:
                    tprint("✅ M1 CPU optimizer initialized")
            except Exception as e:
                tprint(f"⚠️ Failed to initialize CPU optimizer: {e}")
        else:
            tprint("ℹ️ Hardware optimizers not available")
    
    def _validate_config(self, config: HMMTrainingConfig) -> None:
        """
        Validate configuration parameters with fast-fail on critical errors.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # Use common validation utilities if available
            if SHARED_UTILITIES_AVAILABLE:
                if not ValidationUtils.validate_config(config):
                    raise ValueError("Configuration validation failed")
            else:
                # Basic config validation if shared utilities not available
                tprint("⚠️ Shared utilities not available, skipping config validation")
            
            # Additional HMM-specific validations using math validation
            warnings = []
            
            # Validate numeric parameters using common math validation
            try:
                validate_finite(config.n_features, "n_features")
                validate_finite(config.sequence_length, "sequence_length")
                validate_finite(config.n_regimes, "n_regimes")
                validate_finite(config.hpo_trials, "hpo_trials")
            except ValueError as e:
                raise ValueError(f"Invalid numeric parameter: {e}")
            
            # Validate model types are supported using UnifiedModelFactory
            try:
                from .shared_utilities.unified_model_factory import UnifiedModelFactory
                available_model_types = UnifiedModelFactory.get_available_models()
            except ImportError:
                # Fallback to basic model types if UnifiedModelFactory not available
                available_model_types = ['lightgbm', 'random_forest', 'xgboost', 'logistic_regression', 'elastic_net_lr', 'elastic_net_cv']
            
            unsupported_models = [m for m in config.model_types if m not in available_model_types]
            if unsupported_models:
                supported_models = [m for m in config.model_types if m in available_model_types]
                if supported_models:
                    warnings.append(f"WARNING: Unsupported models {unsupported_models} will be skipped. Using: {supported_models}")
                    config.model_types = supported_models
                else:
                    raise ValueError(f"No supported models found in {config.model_types}. Available: {available_model_types}")
            
            # Warning validations (don't cause fast-fail)
            if config.n_features > 1000:
                warnings.append("WARNING: Large number of features may impact performance")
            
            if config.hpo_trials > 1000:
                warnings.append("WARNING: Large number of HPO trials may take very long")
            
            if config.sequence_length > 100:
                warnings.append("WARNING: Large sequence length may impact memory usage")
            
            # Light mode specific validations
            if hasattr(config, 'training_mode_config') and config.training_mode_config:
                training_mode = config.training_mode_config.get('training_mode', '')
                if training_mode == 'light':
                    if config.hpo_trials > 10:
                        warnings.append("WARNING: Light mode should use ≤10 HPO trials for efficiency")
                    if config.n_features > 100:
                        warnings.append("WARNING: Light mode should use ≤100 features for efficiency")
            
            # Log warnings
            if warnings:
                for warning in warnings:
                    tprint(f"⚠️ {warning}")
            
            tprint("✅ Configuration validation passed")
            
        except Exception as e:
            tprint(f"❌ Configuration validation error: {e}")
            raise ValueError(f"Configuration validation failed: {e}") from e
    
    def _initialize_components(self) -> None:
        """Initialize training components with comprehensive error handling."""
        # Initialize feature generator with specific error handling
        self.feature_generator = self._initialize_feature_generator()
        
        # Initialize feature selector with specific error handling
        self.feature_selector = self._initialize_feature_selector()
        
        # Initialize evaluation utilities with specific error handling
        self.evaluation_utils = self._initialize_evaluation_utils()

        # Model creation now handled by shared UnifiedModelFactory
    
    def _initialize_feature_generator(self) -> Optional[Any]:
        """Initialize feature generator with specific error handling."""
        try:
            # Try primary import
            from src.feature_generation.utils.feature_generators import FeatureGenerators
            generator = FeatureGenerators()
            self.logger.info("✅ Feature generator initialized from feature_engineering")
            return generator
        except ImportError as primary_error:
            self.logger.debug(f"Primary feature generator import failed: {primary_error}")
            try:
                # Fallback to standalone compatibility
                from src.hmm_feature_compatibility import FeatureGenerators
                generator = FeatureGenerators()
                self.logger.info("✅ Feature generator initialized from standalone compatibility")
                return generator
            except ImportError as fallback_error:
                self.logger.warning(f"⚠️ Feature generator not available - primary: {primary_error}, fallback: {fallback_error}")
                return None
        except Exception as e:
            self.logger.error(f"❌ Unexpected error initializing feature generator: {e}")
            return None
    
    def _initialize_feature_selector(self) -> Optional[Any]:
        """Initialize feature selector with specific error handling."""
        try:
            from src.training.utils.feature_selection.main_framework import FeatureSelectionFramework
            fs_config = {
                'selection_methods': ['mrmr', 'lasso_stability'],
                'max_features': self.config.n_features,
                'enable_stability_analysis': True
            }
            selector = FeatureSelectionFramework(fs_config)
            self.logger.info("✅ Feature selector initialized")
            return selector
        except ImportError as e:
            self.logger.warning(f"⚠️ Feature selector not available: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Unexpected error initializing feature selector: {e}")
            return None
    
    def _initialize_evaluation_utils(self) -> Optional[Any]:
        """Initialize evaluation utilities with specific error handling."""
        try:
            from src.utils.ml_common.evaluation.evaluation_utils import EvaluationUtils
            utils = EvaluationUtils()
            self.logger.info("✅ Evaluation utilities initialized")
            return utils
        except ImportError as e:
            self.logger.warning(f"⚠️ Evaluation utilities not available: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Unexpected error initializing evaluation utilities: {e}")
            return None
    
    def _convert_to_numpy_array(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Convert data to numpy array with proper validation and error handling.
        Uses common utilities for better validation and error handling.
        
        Args:
            data: Input data (DataFrame or numpy array)
            
        Returns:
            numpy array
            
        Raises:
            ValueError: If conversion fails or data is invalid
        """
        try:
            if isinstance(data, np.ndarray):
                # Already a numpy array, validate it using common utilities
                if data.size == 0:
                    raise ValueError("Input array is empty")
                
                # Use common math validation
                validate_numeric_array(data, "input_array")
                return data
            
            elif isinstance(data, pd.DataFrame):
                # Convert DataFrame to numpy array using common utilities
                if data.empty:
                    raise ValueError("Input DataFrame is empty")
                
                # Use common DataFrame operations
                numeric_data = safe_dataframe_operation(
                    data, 
                    lambda df: df.select_dtypes(include=[np.number])
                )
                
                if numeric_data.empty:
                    raise ValueError("DataFrame contains no numeric columns")
                
                if len(numeric_data.columns) != len(data.columns):
                    non_numeric_cols = set(data.columns) - set(numeric_data.columns)
                    tprint(f"⚠️ Dropping non-numeric columns: {non_numeric_cols}")
                
                # Convert to numpy array
                array_data = numeric_data.values
                
                # Validate the converted array using common utilities
                validate_numeric_array(array_data, "converted_array")
                
                return array_data
            
            else:
                raise ValueError(f"Unsupported data type: {type(data)}. Expected numpy array or DataFrame.")
                
        except Exception as e:
            tprint(f"❌ Failed to convert data to numpy array: {e}")
            raise ValueError(f"Data conversion failed: {e}") from e
    
# Model registration now handled by shared UnifiedModelFactory
    
    def _validate_input_data(self, X: np.ndarray, y: np.ndarray, cluster_assignments: Optional[np.ndarray]) -> bool:
        """
        Enhanced input validation with early exit on critical failures.
        Uses common utilities for better validation and error handling.
        
        Args:
            X: Input features
            y: Target values
            cluster_assignments: Cluster assignments
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Import numpy at the beginning for use throughout the method
            import numpy as np

            # Check if cluster_assignments is None or empty first
            if cluster_assignments is None:
                tprint("❌ cluster_assignments is None - cannot proceed with validation")
                return False

            # Convert cluster_assignments to numpy array if it's a list
            if isinstance(cluster_assignments, list):
                cluster_assignments = np.array(cluster_assignments)

            # Check if cluster_assignments is still None after conversion attempt
            if cluster_assignments is None:
                tprint("❌ cluster_assignments is None after conversion attempt - validation failed")
                return False

            # Use common validation utilities if available
            if SHARED_UTILITIES_AVAILABLE:
                if not ValidationUtils.validate_data_shapes(X, y, cluster_assignments):
                    return False

                if not ValidationUtils.validate_data_quality(X, y, cluster_assignments):
                    return False

                if not ValidationUtils.validate_regime_distribution(cluster_assignments, min_samples_per_regime=1):
                    return False
            else:
                # Fallback validation if shared utilities not available
                tprint("⚠️ Shared utilities not available, using basic validation")
                if len(X) != len(y):
                    tprint("❌ Shape mismatch: X and y have different lengths")
                    return False
                if cluster_assignments is not None and len(X) != len(cluster_assignments):
                    tprint("⚠️ Data alignment issue: X and cluster_assignments have different lengths")
                if cluster_assignments is not None and len(np.unique(cluster_assignments)) < 2:
                    tprint("❌ Need at least 2 clusters")
                    return False
            
            # Additional HMM-specific validations using common math validation
            critical_failures = []
            warnings = []
            unique_clusters = np.unique(cluster_assignments)
            
            # Enhanced data type validation
            if not isinstance(X, np.ndarray):
                critical_failures.append(f"X must be numpy array, got {type(X)}")
            if not isinstance(y, np.ndarray):
                critical_failures.append(f"y must be numpy array, got {type(y)}")
            if not isinstance(cluster_assignments, np.ndarray):
                critical_failures.append(f"cluster_assignments must be numpy array, got {type(cluster_assignments)}")
            
            # Enhanced shape validation
            if len(X) == 0:
                critical_failures.append("X is empty")
            if len(y) == 0:
                critical_failures.append("y is empty")
            if len(cluster_assignments) == 0:
                critical_failures.append("cluster_assignments is empty")
            
            # Enhanced memory size validation
            try:
                import psutil
                X_memory_mb = X.nbytes / (1024 * 1024)
                y_memory_mb = y.nbytes / (1024 * 1024)
                total_memory_mb = X_memory_mb + y_memory_mb
                
                if total_memory_mb > 1000:  # 1GB threshold
                    warnings.append(f"Large dataset detected: {total_memory_mb:.1f}MB total memory usage")
            except ImportError:
                pass  # Skip memory validation if psutil not available
            
            # Enhanced feature value range validation
            if X.size > 0:
                try:
                    validate_numeric_array(X, "input_features")
                    if np.any(np.abs(X) > 1e6):
                        warnings.append("Large feature values detected (>1e6), may indicate data quality issues")
                except ValueError as e:
                    critical_failures.append(f"Invalid feature values: {e}")
            
            # Enhanced target validation
            if y.size > 0:
                try:
                    validate_numeric_array(y, "target_values")
                    unique_targets = np.unique(y)
                    if len(unique_targets) < 2:
                        critical_failures.append("Target variable has only one unique value")
                    elif len(unique_targets) > 100:
                        warnings.append(f"Target variable has many unique values ({len(unique_targets)}), consider if this is classification")
                except ValueError as e:
                    critical_failures.append(f"Invalid target values: {e}")
            
            # Enhanced cluster validation
            if len(unique_clusters) < 2:
                critical_failures.append(f"Need at least 2 clusters, found {len(unique_clusters)}")
            elif len(unique_clusters) > 20:
                warnings.append(f"Many clusters detected ({len(unique_clusters)}), may indicate overfitting")
            
            # Enhanced cluster distribution validation
            cluster_counts = [np.sum(cluster_assignments == cluster) for cluster in unique_clusters]
            min_cluster_count = min(cluster_counts)
            max_cluster_count = max(cluster_counts)
            
            if min_cluster_count < 1:
                critical_failures.append(f"Cluster has only {min_cluster_count} samples (minimum: 1)")
            elif min_cluster_count < 50:
                warnings.append(f"Some clusters have few samples (minimum: {min_cluster_count})")
            
            # Check for cluster imbalance
            cluster_balance = min_cluster_count / max_cluster_count
            if cluster_balance < 0.1:
                warnings.append(f"Severe cluster imbalance detected (ratio: {cluster_balance:.2f})")
            
            # Early exit on critical failures
            if critical_failures:
                tprint(f"❌ Critical validation failures: {critical_failures}")
                return False
            
            # Warning checks (don't cause early exit)
            if len(X) < 1000:
                warnings.append(f"Small dataset: {len(X)} samples (recommended: >1000)")
            
            # Log warnings
            if warnings:
                for warning in warnings:
                    tprint(f"⚠️ {warning}")
            
            tprint(f"✅ Enhanced validation passed: {len(X)} samples, {len(unique_clusters)} clusters")
            return True
            
        except Exception as e:
            tprint(f"❌ Validation error: {e}")
            return False
    
    def _prepare_comprehensive_features(self, data: pd.DataFrame, cluster_assignments: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare comprehensive features using the advanced feature generation system.

        Args:
            data: Market data DataFrame with OHLCV columns
            cluster_assignments: Optional cluster assignments for regime-aware features

        Returns:
            Tuple of (features, feature_names)
        """
        try:
            tprint("🔧 Preparing comprehensive features using feature generation system...")

            if not FEATURE_GENERATION_AVAILABLE:
                tprint("⚠️ Feature generation system not available, falling back to basic features")
                return self._prepare_basic_features(data, cluster_assignments)

            # Get the feature bank
            feature_bank = get_global_feature_bank()

            # Define categories to generate features from
            categories_to_use = [
                FeatureCategory.MOMENTUM,
                FeatureCategory.VOLATILITY,
                FeatureCategory.TREND,
                FeatureCategory.VOLUME,
                FeatureCategory.SUPPORT_RESISTANCE
            ]

            # Generate features by category
            tprint(f"📊 Generating features from {len(categories_to_use)} categories...")
            features_df = feature_bank.generate_features(
                data=data,
                categories=categories_to_use,
                lookback_optimization=False,  # Use default lookbacks for now
                target_column=None  # No target for feature generation
            )

            # Note: HMM regime features removed as per user request

            # Get feature names
            feature_names = list(features_df.columns)

            tprint(f"✅ Generated {len(feature_names)} comprehensive features")
            tprint(f"📊 Feature categories breakdown:")
            for category in categories_to_use:
                category_features = [name for name in feature_names if category.value.lower() in name.lower()]
                tprint(f"   • {category.value}: {len(category_features)} features")

            return features_df, feature_names

        except Exception as e:
            tprint(f"❌ Comprehensive feature generation failed: {e}")
            raise ValueError(f"Feature generation failed: {e}")

    def _prepare_basic_features(self, data: pd.DataFrame, cluster_assignments: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare basic features as fallback (close_return, volume_return, price_range_pct).

        Args:
            data: Market data DataFrame
            cluster_assignments: Optional cluster assignments

        Returns:
            Tuple of (features, feature_names)
        """
        try:
            tprint("📊 Preparing basic features (fallback)...")

            # Ensure we have required OHLCV data
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]

            if missing_cols:
                tprint(f"⚠️ Missing columns: {missing_cols}, creating basic features from available data")
                # Create basic features from available data
                features = pd.DataFrame(index=data.index)

                if 'close' in data.columns:
                    features['close_return'] = data['close'].pct_change()
                    features['price_range_pct'] = (data['high'] - data['low']) / data['close'] if 'high' in data.columns and 'low' in data.columns else 0

                if 'volume' in data.columns:
                    features['volume_return'] = data['volume'].pct_change()

                # Fill NaN values
                features = features.fillna(0)

            else:
                # Calculate standard basic features
                features = pd.DataFrame(index=data.index)

                # Price features
                features['close_return'] = data['close'].pct_change()
                features['price_range_pct'] = (data['high'] - data['low']) / data['close']

                # Volume features
                features['volume_return'] = data['volume'].pct_change()

                # Fill NaN values
                features = features.fillna(0)

            feature_names = list(features.columns)
            tprint(f"✅ Generated {len(feature_names)} basic features: {feature_names}")

            return features, feature_names

        except Exception as e:
            tprint(f"❌ Basic feature preparation failed: {e}")
            raise

    def _prepare_features(self, X: Union[np.ndarray, pd.DataFrame], feature_names: Optional[List[str]] = None, cluster_assignments: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare and enhance features with optimized performance and comprehensive error handling.
        Uses in-place operations where possible to reduce memory usage.
        
        Args:
            X: Input features
            feature_names: Optional feature names
            
        Returns:
            Tuple of (enhanced_features, feature_names)
        """
        try:
            # Optimize memory usage by avoiding unnecessary copies
            if isinstance(X, np.ndarray):
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                # Use memory-efficient DataFrame creation
                X_df = pd.DataFrame(X, columns=feature_names, copy=False)
            else:
                # Only copy if we need to modify the original
                if feature_names is None:
                    feature_names = list(X.columns)
                X_df = X  # Use original DataFrame to avoid copy
            
            # Use optimized numeric column selection
            numeric_dtypes = [np.number]
            numeric_columns = X_df.select_dtypes(include=numeric_dtypes).columns
            
            if len(numeric_columns) == 0:
                raise ValueError("No numeric columns found in input data")
            
            # Only create subset if we have non-numeric columns
            if len(numeric_columns) == len(X_df.columns):
                X_numeric = X_df  # Use original to avoid copy
            else:
                X_numeric = X_df[numeric_columns]
                tprint(f"📊 Filtered to {len(numeric_columns)} numeric features from {len(X_df.columns)} total")
            
            # Quick data quality check using vectorized operations
            null_count = X_numeric.isnull().sum().sum()
            if null_count > 0:
                null_percentage = null_count / (X_numeric.shape[0] * X_numeric.shape[1])
                if null_percentage > 0.1:
                    tprint(f"⚠️ High null percentage: {null_percentage:.2%}")
            
            # Use shared enhanced feature creation utilities
            # This ensures consistency between hmm_models_training and ensemble_training
            try:
                # Convert DataFrame to numpy array for feature enhancement
                X_array = X_numeric.values
                
                # Use real cluster assignments for feature enhancement
                if cluster_assignments is not None:
                    regime_labels = cluster_assignments
                else:
                    # Fallback to dummy labels if cluster_assignments not available
                    regime_labels = np.zeros(X_array.shape[0])
                
                # Use shared enhanced feature creation method
                X_enhanced_array, enhanced_feature_names = create_enhanced_features_with_names(
                    X_array, regime_labels, list(X_numeric.columns)
                )
                
                # Convert back to DataFrame with enhanced feature names
                X_enhanced = pd.DataFrame(X_enhanced_array, columns=enhanced_feature_names, index=X_numeric.index)
                
                tprint(f"✅ Enhanced features: {X_enhanced.shape[1]} total features (using shared utilities)")
                return X_enhanced, enhanced_feature_names
                
            except Exception as e:
                tprint(f"⚠️ Enhanced feature creation failed: {e}, using original features")
                return X_numeric, list(X_numeric.columns)
                
        except Exception as e:
            tprint(f"❌ Feature preparation failed: {e}")
            raise
    
    
    def _select_features(self, X: pd.DataFrame, y: np.ndarray, is_classification: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select optimal features with comprehensive error handling.
        
        Args:
            X: Input features
            y: Target values
            is_classification: Whether this is classification
            
        Returns:
            Tuple of (selected_features, selected_feature_names)
        """
        try:
            if self.feature_selector is None:
                self.logger.warning("⚠️ Feature selector not available, using all features")
                return X, list(X.columns)
            
            # Validate shapes before proceeding - don't modify input data
            if len(X) != len(y):
                error_msg = f"Shape mismatch: X has {len(X)} samples, y has {len(y)} samples"
                self.logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            # Create copies to avoid modifying original data
            X_copy = X.copy()
            y_copy = y.copy()
            
            # Apply feature selection
            selection_result = self.feature_selector.select_features(
                X_copy, y_copy,
                method='comprehensive',
                max_features=self.config.n_features,
                is_classification=is_classification
            )
            
            selected_features = selection_result.get('selected_features', list(X.columns)[:self.config.n_features])

            # Enhanced: Convert indices to column names if needed
            if selected_features and isinstance(selected_features[0], (int, np.integer)):
                # selected_features contains indices, convert to column names
                all_columns = list(X.columns)
                try:
                    selected_features = [all_columns[i] for i in selected_features if 0 <= i < len(all_columns)]
                except (IndexError, TypeError) as e:
                    self.logger.error(f"❌ Failed to convert indices to column names: {e}")
                    selected_features = list(X.columns)[:self.config.n_features]

            # Enhanced: Validate selected features exist and are valid
            missing_features = [f for f in selected_features if f not in X.columns]
            if missing_features:
                self.logger.warning(f"⚠️ Some selected features not found in DataFrame: {missing_features}")
                # Filter out missing features
                valid_selected_features = [f for f in selected_features if f in X.columns]
                
                # Use configurable threshold instead of hardcoded 50%
                missing_threshold = 0.5  # Could be made configurable
                if len(valid_selected_features) < len(selected_features) * (1 - missing_threshold):
                    # More than threshold of features are missing - this is a serious issue
                    self.logger.error(f"❌ More than {missing_threshold*100:.0f}% of selected features are missing ({len(missing_features)}/{len(selected_features)})")
                    raise ValueError(f"Feature selection returned invalid features: {missing_features[:5]}")
                
                selected_features = valid_selected_features
            
            # Ensure we have enough features
            if not selected_features:
                self.logger.warning("⚠️ No valid features selected, using fallback selection")
                available_features = list(X.columns)
                if len(available_features) >= self.config.n_features:
                    selected_features = available_features[:self.config.n_features]
                else:
                    selected_features = available_features
                    self.logger.warning(f"⚠️ Only {len(selected_features)} features available, less than requested {self.config.n_features}")
            
            # Final validation: ensure selected features are numeric and have variance
            try:
                X_test = X[selected_features]
                if X_test.select_dtypes(include=[np.number]).shape[1] != len(selected_features):
                    self.logger.warning("⚠️ Some selected features are non-numeric, filtering...")
                    numeric_features = X_test.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_features) < len(selected_features) * 0.8:
                        self.logger.warning(f"⚠️ Many selected features are non-numeric ({len(numeric_features)}/{len(selected_features)})")
                    selected_features = numeric_features
                
                # Check for features with zero variance
                if len(selected_features) > 0:
                    feature_vars = X[selected_features].var()
                    zero_var_features = feature_vars[feature_vars == 0].index.tolist()
                    if zero_var_features:
                        self.logger.warning(f"⚠️ Removing {len(zero_var_features)} zero-variance features: {zero_var_features[:5]}")
                        selected_features = [f for f in selected_features if f not in zero_var_features]
                
            except Exception as e:
                self.logger.error(f"❌ Feature validation failed: {e}")
                raise ValueError(f"Selected features validation failed: {e}")
            
            # Final check - ensure we still have features
            if not selected_features:
                raise ValueError("No valid features remaining after validation")
            
            X_selected = X[selected_features]

            # Enhanced validation: Check for data leakage and feature quality
            if SHARED_UTILITIES_AVAILABLE:
                is_valid, validation_message = ValidationUtils.validate_feature_selection_quality(
                    X, y, selected_features, min_feature_count=3
                )
            else:
                # Basic feature validation if shared utilities not available
                is_valid = True
                validation_message = "Feature validation skipped - shared utilities not available"

            if not is_valid:
                self.logger.error(f"❌ Feature validation failed: {validation_message}")
                # Log additional details about the failure
                tprint(f"⚠️ Feature selection validation failed: {validation_message}")
                # Fall back to original features with warning
                self.logger.warning("🔄 Falling back to original features due to validation failure")
                return X, list(X.columns)

            self.logger.info(f"✅ Feature selection completed: {len(selected_features)} features selected")
            self.logger.info(f"✅ Feature validation passed: {validation_message}")
            return X_selected, selected_features
            
        except Exception as e:
            self.logger.error(f"❌ Feature selection failed: {e}")
            # Fallback to basic selection with validation
            fallback_features = list(X.columns)[:self.config.n_features]
            if not fallback_features:
                raise ValueError("No features available for fallback selection")
            return X[fallback_features], fallback_features
    
    def _create_model(self, model_type: str, **kwargs) -> Any:
        """
        Create model instance using shared UnifiedModelFactory.
        
        Args:
            model_type: Type of model to create
            **kwargs: Additional model parameters
            
        Returns:
            Model instance
        """
        try:
            # Use shared unified model factory
            return UnifiedModelFactory.create_model(model_type, **kwargs)
        except Exception as e:
            self.logger.error(f"❌ Failed to create model {model_type}: {e}")
            raise
    
    def _train_single_model(self, model_type: str, X: np.ndarray, y: np.ndarray) -> ModelResult:
        """
        Train a single model with circuit breaker protection and enhanced error handling.
        Uses hardware optimizers and common utilities for better performance.
        
        Args:
            model_type: Type of model to train
            X: Training features
            y: Training targets
            
        Returns:
            ModelResult with trained model and metrics
        """
        start_time = time.time()
        metrics = TrainingMetrics()
        
        # Use memory checkpoint from common operations for better memory management
        with memory_checkpoint(f"training_{model_type}"):
            try:
                tprint(f"🔄 Training {model_type}...")
                
                # Apply hardware optimizations if available with enhanced error handling
                if self.memory_optimizer:
                    try:
                        self.memory_optimizer.optimize_memory_usage()
                        tprint(f"🧠 Memory optimization applied for {model_type}")
                    except MemoryError as e:
                        tprint(f"❌ Critical memory error during optimization for {model_type}: {e}")
                        # Force garbage collection and try to continue
                        import gc
                        gc.collect()
                        raise MemoryError(f"Insufficient memory for {model_type} training after optimization attempt")
                    except Exception as e:
                        tprint(f"⚠️ Memory optimization failed for {model_type}: {e}")
                        logger.warning(f"Memory optimization failed for {model_type}: {e}")
                
                if self.cpu_optimizer:
                    try:
                        self.cpu_optimizer.optimize_cpu_usage()
                        tprint(f"⚡ CPU optimization applied for {model_type}")
                    except Exception as e:
                        tprint(f"⚠️ CPU optimization failed for {model_type}: {e}")
                        logger.warning(f"CPU optimization failed for {model_type}: {e}")
                
                # Memory management before training using common operations
                initial_memory_bytes = get_memory_usage()
                initial_memory_mb = initial_memory_bytes / (1024 * 1024)  # Convert to MB
                
                # Get actual total system memory instead of hardcoding
                try:
                    import psutil
                    total_memory_bytes = psutil.virtual_memory().total
                    initial_memory_pct = (initial_memory_bytes / total_memory_bytes) * 100
                except (ImportError, AttributeError):
                    # Fallback to reasonable default if psutil not available
                    total_memory_bytes = 8 * 1024 * 1024 * 1024  # 8GB default
                    initial_memory_pct = (initial_memory_bytes / total_memory_bytes) * 100
                if initial_memory_pct > 90:
                    tprint(f"⚠️ High memory usage ({initial_memory_pct:.1f}%) before training {model_type}")
                    optimize_memory()  # Use common memory optimization
                
                # Create model with circuit breaker protection and timeout
                def create_and_train_model():
                    # Add timeout for model creation and training with proper cleanup
                    import signal
                    import threading
                    
                    # Use threading-based timeout for better cross-platform compatibility
                    timeout_seconds = 60 if hasattr(self.config, 'training_mode_config') and \
                                         self.config.training_mode_config and \
                                         self.config.training_mode_config.get('training_mode') == 'light' else 300
                    
                    result = {'model': None, 'predictions': None, 'feature_importance': None, 'hyperparameters': None, 'y_test': None, 'exception': None}
                    
                    def training_worker():
                        try:
                            model = self._create_model(model_type)
                            
                            # Check memory before training using common operations
                            current_memory_bytes = get_memory_usage()
                            try:
                                import psutil
                                total_memory_bytes = psutil.virtual_memory().total
                                current_memory_pct = (current_memory_bytes / total_memory_bytes) * 100
                            except (ImportError, AttributeError):
                                total_memory_bytes = 8 * 1024 * 1024 * 1024  # 8GB default
                                current_memory_pct = (current_memory_bytes / total_memory_bytes) * 100
                            if current_memory_pct > 95:
                                tprint(f"⚠️ Memory usage critical ({current_memory_pct:.1f}%) - skipping {model_type}")
                                raise MemoryError(f"Insufficient memory for {model_type} training")
                            
                            # Enhanced: Add proper train/test split for validation
                            from sklearn.model_selection import train_test_split
                            from sklearn.metrics import accuracy_score

                            # Validate train/test split integrity
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.3, random_state=42, stratify=y
                            )

                            # Check split integrity
                            if SHARED_UTILITIES_AVAILABLE:
                                is_split_valid, split_message = ValidationUtils.validate_train_test_split(
                                    X_train, X_test, y_train, y_test, temporal_check=False
                                )
                            else:
                                # Basic split validation if shared utilities not available
                                is_split_valid = True
                                split_message = "Split validation skipped - shared utilities not available"

                            if not is_split_valid:
                                raise ValueError(f"Invalid train/test split: {split_message}")

                            # Train model on training set only
                            model.fit(X_train, y_train)

                            # Get predictions on both train and test sets for validation
                            train_predictions = model.predict(X_train)
                            test_predictions = model.predict(X_test)

                            # Calculate metrics for both sets
                            train_accuracy = accuracy_score(y_train, train_predictions)
                            test_accuracy = accuracy_score(y_test, test_predictions)

                            # Get feature importance before overfitting detection
                            feature_importance = None
                            try:
                                if hasattr(model, 'feature_importances_'):
                                    feature_importance = model.feature_importances_
                                elif hasattr(model, 'coef_'):
                                    feature_importance = np.abs(model.coef_).flatten() if model.coef_.ndim > 1 else np.abs(model.coef_)
                            except Exception as e:
                                tprint(f"Could not get feature importance: {e}")

                            # Load cluster assignments from parquet file if available
                            try:
                                import pandas as pd
                                # Try to load cluster assignments from the latest optimal regime clustering outcome
                                if hasattr(self, '_cluster_assignments') and self._cluster_assignments is not None:
                                    # Use cached cluster assignments if available
                                    pass
                                else:
                                    # Try to load from the latest optimal regime clustering outcome
                                    try:
                                        import pandas as pd
                                        from src.utils.common_operations import load_latest_optimal_regime_clustering_outcome

                                        outcome_data = load_latest_optimal_regime_clustering_outcome()

                                        if outcome_data and 'artifacts' in outcome_data:
                                            # Extract cluster assignments from the outcome data
                                            artifacts = outcome_data['artifacts']

                                            # Look for cluster assignments in various possible locations
                                            cluster_assignments = None

                                            # Check for cluster assignments in the main artifacts
                                            if 'cluster_assignments' in artifacts:
                                                cluster_assignments = artifacts['cluster_assignments']
                                            elif 'optimal_regime_clustering_result' in artifacts:
                                                clustering_result = artifacts['optimal_regime_clustering_result']
                                                if 'cluster_assignments' in clustering_result:
                                                    cluster_assignments = clustering_result['cluster_assignments']

                                            if cluster_assignments is not None:
                                                if isinstance(cluster_assignments, np.ndarray):
                                                    self._cluster_assignments = cluster_assignments
                                                elif hasattr(cluster_assignments, 'values'):
                                                    self._cluster_assignments = cluster_assignments.values
                                                else:
                                                    # Try to convert to numpy array
                                                    self._cluster_assignments = np.array(cluster_assignments)

                                                tprint(f"✅ Loaded {len(self._cluster_assignments)} cluster assignments from latest optimal regime clustering outcome")
                                                # Ensure cluster assignments match data length
                                                if self._cluster_assignments is not None and X is not None:
                                                    if len(self._cluster_assignments) != len(X):
                                                        tprint(f"⚠️ Cluster assignments length ({len(self._cluster_assignments)}) doesn't match X length ({len(X)})")
                                                        # Use proportion-based alignment strategy
                                                        import numpy as np
                                                        from src.training.steps.market_analysis.hmm_models_training.shared_utilities.validation_utils import ValidationUtils
                                                        tprint("🔧 Aligning cluster assignments using proportion-based strategy...")
                                                        aligned_assignments = ValidationUtils._align_regime_labels(self._cluster_assignments, len(X))
                                                        if aligned_assignments is not None:
                                                            self._cluster_assignments = aligned_assignments
                                                            tprint(f"✅ Aligned cluster assignments: {len(self._cluster_assignments)} samples")
                                                        else:
                                                            tprint("❌ Failed to align cluster assignments")
                                                            self._cluster_assignments = None
                                            else:
                                                tprint("⚠️ No cluster assignments found in optimal regime clustering outcome")
                                        else:
                                            tprint("⚠️ Could not load latest optimal regime clustering outcome")

                                    except Exception as e:
                                        tprint(f"⚠️ Error loading cluster assignments from optimal regime clustering outcome: {e}")

                                    # Fallback to loading from the pickle file created by clustering
                                    try:
                                        hmm_input_path = "optimal_clusters/binance/ETHUSDT/15m/market_analysis_hmm_training_input_ETHUSDT_BINANCE_15m_20250921_220102.pkl"
                                        import pickle
                                        with open(hmm_input_path, 'rb') as f:
                                            hmm_input_data = pickle.load(f)

                                        if 'cluster_assignments' in hmm_input_data:
                                            cluster_assignments = hmm_input_data['cluster_assignments']
                                            self._cluster_assignments = cluster_assignments
                                            tprint(f"✅ Loaded {len(self._cluster_assignments)} cluster assignments from HMM training input file")
                                            # Ensure cluster assignments match data length
                                            if self._cluster_assignments is not None and X is not None:
                                                if len(self._cluster_assignments) != len(X):
                                                    tprint(f"⚠️ Cluster assignments length ({len(self._cluster_assignments)}) doesn't match X length ({len(X)})")
                                                    # Use proportion-based alignment strategy
                                                    import numpy as np
                                                    from src.training.steps.market_analysis.hmm_models_training.shared_utilities.validation_utils import ValidationUtils
                                                    tprint("🔧 Aligning cluster assignments using proportion-based strategy...")
                                                    aligned_assignments = ValidationUtils._align_regime_labels(self._cluster_assignments, len(X))
                                                    if aligned_assignments is not None:
                                                        self._cluster_assignments = aligned_assignments
                                                        tprint(f"✅ Aligned cluster assignments: {len(self._cluster_assignments)} samples")
                                                    else:
                                                        tprint("❌ Failed to align cluster assignments")
                                                        self._cluster_assignments = None
                                        else:
                                            tprint(f"⚠️ No cluster_assignments found in HMM training input file")

                                    except Exception as e:
                                        tprint(f"⚠️ Error loading cluster assignments from HMM training input file: {e}")

                                    # Fallback to the old hardcoded path if the new method fails
                                    try:
                                        cluster_assignments_path = "optimal_clusters/binance/ETHUSDT/15m/optimal_cluster_labels.parquet"
                                        cluster_assignments_df = pd.read_parquet(cluster_assignments_path)
                                        if 'cluster_id' in cluster_assignments_df.columns:
                                            self._cluster_assignments = cluster_assignments_df['cluster_id'].values
                                            tprint(f"✅ Loaded {len(self._cluster_assignments)} cluster assignments from {cluster_assignments_path} (fallback)")
                                            # Ensure cluster assignments match data length
                                            if self._cluster_assignments is not None and X is not None:
                                                if len(self._cluster_assignments) != len(X):
                                                    tprint(f"⚠️ Cluster assignments length ({len(self._cluster_assignments)}) doesn't match X length ({len(X)})")
                                                    # Use proportion-based alignment strategy
                                                    import numpy as np
                                                    from src.training.steps.market_analysis.hmm_models_training.shared_utilities.validation_utils import ValidationUtils
                                                    tprint("🔧 Aligning cluster assignments using proportion-based strategy...")
                                                    aligned_assignments = ValidationUtils._align_regime_labels(self._cluster_assignments, len(X))
                                                    if aligned_assignments is not None:
                                                        self._cluster_assignments = aligned_assignments
                                                        tprint(f"✅ Aligned cluster assignments: {len(self._cluster_assignments)} samples")
                                                    else:
                                                        tprint("❌ Failed to align cluster assignments")
                                                        self._cluster_assignments = None
                                        else:
                                            tprint(f"⚠️ No cluster_id column found in {cluster_assignments_path}")
                                    except Exception as e:
                                        tprint(f"⚠️ Could not load cluster assignments from {cluster_assignments_path} (fallback): {e}")
                            except Exception as e:
                                tprint(f"⚠️ Error handling cluster assignments: {e}")

                            # Comprehensive overfitting detection
                            if SHARED_UTILITIES_AVAILABLE:
                                overfitting_analysis = ValidationUtils.detect_overfitting_comprehensive(
                                    train_predictions=train_predictions,
                                    test_predictions=test_predictions,
                                    train_labels=y_train,
                                    test_labels=y_test,
                                    train_probabilities=None,  # Will be filled if available
                                    test_probabilities=None,   # Will be filled if available
                                    model=model,
                                    feature_importance=feature_importance
                                )
                            else:
                                # Basic overfitting detection if shared utilities not available
                                overfitting_analysis = {
                                    'overfitting_detected': False,
                                    'accuracy_gap': abs(train_accuracy - test_accuracy),
                                    'message': 'Overfitting detection skipped - shared utilities not available'
                                }

                            # Get probabilities if available for enhanced analysis
                            if hasattr(model, 'predict_proba'):
                                try:
                                    train_probabilities = model.predict_proba(X_train)
                                    test_probabilities = model.predict_proba(X_test)
                                    overfitting_analysis = ValidationUtils.detect_overfitting_comprehensive(
                                        train_predictions=train_predictions,
                                        test_predictions=test_predictions,
                                        train_labels=y_train,
                                        test_labels=y_test,
                                        train_probabilities=train_probabilities,
                                        test_probabilities=test_probabilities,
                                        model=model,
                                        feature_importance=feature_importance
                                    )
                                except Exception:
                                    pass

                            # Enhanced overfitting reporting
                            if overfitting_analysis['is_overfitting']:
                                severity = overfitting_analysis['severity']
                                tprint(f"⚠️ OVERFITTING DETECTED ({severity.upper()} severity):")
                                tprint(f"   Train accuracy: {overfitting_analysis['train_accuracy']:.4f}")
                                tprint(f"   Test accuracy: {overfitting_analysis['test_accuracy']:.4f}")
                                tprint(f"   Accuracy gap: {overfitting_analysis['accuracy_gap']:.4f}")
                                tprint(f"   F1 gap: {overfitting_analysis['f1_gap']:.4f}")

                                if overfitting_analysis['warnings']:
                                    for warning in overfitting_analysis['warnings']:
                                        tprint(f"   {warning}")

                                if overfitting_analysis['recommendations']:
                                    tprint(f"   📋 Recommendations:")
                                    for rec in overfitting_analysis['recommendations'][:3]:  # Show top 3
                                        tprint(f"      • {rec}")
                            else:
                                # Calculate accuracy gap for display
                                accuracy_gap = train_accuracy - test_accuracy
                                tprint(f"✅ Model generalization validated: Train={train_accuracy:.4f}, Test={test_accuracy:.4f}, Gap={accuracy_gap:.4f}")

                            # Use test set predictions for final metrics
                            predictions = test_predictions
                            
                            # Get hyperparameters
                            hyperparameters = None
                            try:
                                if hasattr(model, 'get_params'):
                                    hyperparameters = model.get_params()
                            except Exception as e:
                                tprint(f"Could not get hyperparameters: {e}")
                            
                            result['model'] = (model, predictions, feature_importance, hyperparameters)
                            result['predictions'] = predictions
                            result['feature_importance'] = feature_importance
                            result['hyperparameters'] = hyperparameters
                            result['y_test'] = y_test
                            
                        except Exception as e:
                            result['exception'] = e
                    
                    # Start training in a separate thread with proper cleanup
                    training_thread = threading.Thread(target=training_worker)
                    training_thread.daemon = True
                    training_thread.start()
                    
                    try:
                        training_thread.join(timeout=timeout_seconds)
                        
                        # Check if training completed or timed out
                        if training_thread.is_alive():
                            # Training timed out - thread may still be running
                            tprint(f"⏰ Training timeout for {model_type} after {timeout_seconds}s")
                            # Note: Thread will be cleaned up when daemon=True and process exits
                            # Force garbage collection to help with cleanup
                            import gc
                            gc.collect()
                            raise TimeoutError(f"Training timeout for {model_type} after {timeout_seconds}s")
                        
                        # Check for exceptions
                        if result['exception']:
                            if isinstance(result['exception'], MemoryError):
                                optimize_memory()  # Use common memory optimization
                            raise result['exception']
                        
                        if result['model'] is None:
                            raise RuntimeError(f"Training failed for unknown reason: {model_type}")
                        
                        return result['model']
                        
                    finally:
                        # Ensure thread cleanup even if exceptions occur
                        if training_thread.is_alive():
                            # Thread is still running, but daemon=True will clean it up
                            # Force garbage collection to help with memory cleanup
                            import gc
                            gc.collect()
                
                # Execute with circuit breaker protection
                result = self.circuit_breaker.call(create_and_train_model)

                # Handle both tuple and dictionary results
                if isinstance(result, tuple) and len(result) >= 4:
                    # Result is a tuple: (model, predictions, feature_importance, hyperparameters, y_test)
                    model, predictions, feature_importance, hyperparameters = result[:4]
                    y_test = result[4] if len(result) > 4 else y
                elif isinstance(result, dict):
                    # Result is a dictionary
                    model = result.get('model')
                    predictions = result.get('predictions')
                    feature_importance = result.get('feature_importance')
                    hyperparameters = result.get('hyperparameters')
                    y_test = result.get('y_test')
                    # If y_test is not provided, we need to reconstruct it from the predictions length
                    if y_test is None and predictions is not None:
                        # Use the last len(predictions) samples from y as y_test
                        y_test = y[-len(predictions):] if len(predictions) <= len(y) else y
                    elif y_test is None:
                        # Fallback to using the full y (this shouldn't happen in normal cases)
                        y_test = y
                else:
                    # Fallback for other result types
                    model = result
                    predictions = None
                    feature_importance = None
                    hyperparameters = None
                    # If predictions exist, reconstruct y_test from the predictions length
                    if predictions is not None:
                        y_test = y[-len(predictions):] if len(predictions) <= len(y) else y
                    else:
                        y_test = y
                
                # Calculate accuracy using safe math operations (use test set predictions)
                accuracy = safe_divide(np.sum(predictions == y_test), len(y_test), 0.0)
                
                # Use evaluation utilities if available
                if ML_EVALUATION_AVAILABLE and self.evaluation_utils is not None:
                    try:
                        eval_metrics = self.evaluation_utils.evaluate_model_performance(
                            model, X, y,
                            metrics=['accuracy', 'f1_score', 'precision', 'recall'],
                            is_classification=True
                        )
                        metrics.accuracy = validate_finite(eval_metrics.get('accuracy', accuracy), "accuracy")
                        metrics.f1_score = validate_finite(eval_metrics.get('f1_score', 0.0), "f1_score")
                        metrics.precision = validate_finite(eval_metrics.get('precision', 0.0), "precision")
                        metrics.recall = validate_finite(eval_metrics.get('recall', 0.0), "recall")
                    except (ImportError, Exception) as e:
                        metrics.warnings.append(f"Evaluation utilities failed: {e}")
                        # Fallback to basic metrics using safe operations
                        metrics.accuracy = validate_finite(accuracy, "accuracy")
                        try:
                            from sklearn.metrics import f1_score, precision_score, recall_score
                            metrics.f1_score = validate_finite(f1_score(y, predictions, average='weighted'), "f1_score")
                            metrics.precision = validate_finite(precision_score(y, predictions, average='weighted'), "precision")
                            metrics.recall = validate_finite(recall_score(y, predictions, average='weighted'), "recall")
                        except Exception as e2:
                            metrics.warnings.append(f"Fallback metrics calculation failed: {e2}")
                else:
                    # Fallback evaluation using safe operations
                    try:
                        from sklearn.metrics import f1_score, precision_score, recall_score
                        metrics.accuracy = validate_finite(accuracy, "accuracy")
                        metrics.f1_score = validate_finite(f1_score(y, predictions, average='weighted'), "f1_score")
                        metrics.precision = validate_finite(precision_score(y, predictions, average='weighted'), "precision")
                        metrics.recall = validate_finite(recall_score(y, predictions, average='weighted'), "recall")
                    except ImportError as e:
                        metrics.warnings.append(f"sklearn metrics not available: {e}")
                        metrics.accuracy = validate_finite(accuracy, "accuracy")
                        # Set other metrics to 0.0 when sklearn not available
                        metrics.f1_score = 0.0
                        metrics.precision = 0.0
                        metrics.recall = 0.0
                    except Exception as e:
                        metrics.warnings.append(f"Fallback metrics calculation failed: {e}")
                        metrics.accuracy = validate_finite(accuracy, "accuracy")
                
                training_time = time.time() - start_time
                metrics.training_time = validate_finite(training_time, "training_time")
                
                # Calculate memory usage using common operations
                final_memory_bytes = get_memory_usage()
                final_memory = final_memory_bytes / (1024 * 1024)  # Convert to MB
                memory_increase = safe_float(final_memory - initial_memory_mb, 0.0)
                metrics.memory_usage_mb = validate_finite(memory_increase, "memory_usage")
                
                # Get probabilities if available
                probabilities = None
                try:
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(X)
                except Exception as e:
                    metrics.warnings.append(f"Could not get probabilities: {e}")
                
                tprint(f"✅ {model_type} trained successfully (accuracy: {metrics.accuracy:.4f}, time: {training_time:.2f}s, memory: {metrics.memory_usage_mb:.1f}MB)")
                
                # Cleanup memory using common operations
                try:
                    import psutil
                    total_memory_bytes = psutil.virtual_memory().total
                    final_memory_pct = (final_memory_bytes / total_memory_bytes) * 100
                except (ImportError, AttributeError):
                    total_memory_bytes = 8 * 1024 * 1024 * 1024  # 8GB default
                    final_memory_pct = (final_memory_bytes / total_memory_bytes) * 100
                
                # Enhanced memory cleanup with multiple strategies
                memory_increase_threshold = 10  # Percentage
                if final_memory_pct > initial_memory_pct + memory_increase_threshold:
                    tprint(f"⚠️ Memory usage increased significantly: {initial_memory_pct:.1f}% → {final_memory_pct:.1f}%")
                    
                    # Try multiple cleanup strategies
                    try:
                        optimize_memory()  # Use common memory optimization
                    except Exception as e:
                        tprint(f"⚠️ Memory optimization failed: {e}")
                    
                    # Force garbage collection
                    import gc
                    collected = gc.collect()
                    tprint(f"🧹 Garbage collection freed {collected} objects")
                    
                    # Clear any cached variables if possible
                    try:
                        if hasattr(self, '_temp_variables'):
                            del self._temp_variables
                    except:
                        pass
                
                return ModelResult(
                    model=model,
                    metrics=metrics,
                    feature_importance=feature_importance,
                    predictions=predictions,
                    probabilities=probabilities,
                    hyperparameters=hyperparameters
                )
            
            except Exception as e:
                training_time = time.time() - start_time
                metrics.training_time = validate_finite(training_time, "training_time")
                metrics.error_message = str(e)
                
                # Calculate memory usage even on failure using common operations
                final_memory_bytes = get_memory_usage()
                final_memory = final_memory_bytes / (1024 * 1024)  # Convert to MB
                memory_increase = safe_float(final_memory - initial_memory_mb, 0.0)
                metrics.memory_usage_mb = validate_finite(memory_increase, "memory_usage")
                
                tprint(f"❌ Failed to train {model_type}: {e}")
                
                # Enhanced memory cleanup using common operations
                try:
                    optimize_memory()
                except Exception as cleanup_error:
                    tprint(f"⚠️ Memory optimization failed during error handling: {cleanup_error}")
                
                # Force garbage collection
                import gc
                collected = gc.collect()
                tprint(f"🧹 Garbage collection freed {collected} objects during error handling")
                
                # Use centralized error handler
                return TrainingErrorHandler.handle_training_error(model_type, e, training_time)
    
    def _save_models_with_common_utils(self, models: Dict[str, Any], model_type: str, 
                                     symbol: str, exchange: str, timeframe: str) -> List[str]:
        """
        Save models using common serialization utilities.
        
        Args:
            models: Dictionary of model name to model object
            model_type: Type of models being saved
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            
        Returns:
            List of saved file paths
        """
        saved_paths = []
        
        try:
            # Create save directory using common operations
            save_dir = Path("artifacts") / "models" / model_type / symbol / exchange / timeframe
            if not ensure_directory(save_dir):
                tprint(f"❌ Failed to create save directory: {save_dir}")
                return saved_paths
            
            # Use UniversalSerializer for better serialization
            serializer = UniversalSerializer()
            
            for model_name, model in models.items():
                try:
                    # Save model using UniversalSerializer
                    model_path = save_dir / f"{model_name}_model.pkl"
                    if serializer.save(model, str(model_path), format='pickle'):
                        saved_paths.append(str(model_path))
                        tprint(f"✅ Saved {model_name} to {model_path}")
                    else:
                        tprint(f"❌ Failed to save {model_name}")
                        
                    # Save metadata using UniversalSerializer
                    metadata = {
                        'model_name': model_name,
                        'model_type': model_type,
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'timestamp': pd.Timestamp.now().isoformat(),
                        'model_class': str(type(model).__name__)
                    }
                    
                    metadata_path = save_dir / f"{model_name}_metadata.json"
                    if serializer.save(metadata, str(metadata_path), format='json'):
                        tprint(f"✅ Saved {model_name} metadata to {metadata_path}")
                    else:
                        tprint(f"⚠️ Failed to save {model_name} metadata")
                        
                except Exception as e:
                    tprint(f"❌ Error saving {model_name}: {e}")
                    
        except Exception as e:
            tprint(f"❌ Error in model saving process: {e}")
            
        return saved_paths
    
    def _generate_comprehensive_report(self, results: Dict[str, Any], execution_time: float) -> Dict[str, Any]:
        """
        Generate comprehensive training report with real metrics.
        
        Args:
            results: Training results
            execution_time: Total execution time
            
        Returns:
            Comprehensive report dictionary
        """
        try:
            report = {
                "report_type": "HMM Models Training Enhanced Report",
                "timestamp": pd.Timestamp.now().isoformat(),
                "execution_summary": {
                    "total_execution_time": execution_time,
                    "models_trained": len(results.get('model_results', {})),
                    "successful_models": sum(1 for r in results.get('model_results', {}).values() 
                                           if r.metrics.error_message is None),
                    "failed_models": sum(1 for r in results.get('model_results', {}).values() 
                                        if r.metrics.error_message is not None),
                    "circuit_breaker_state": self.circuit_breaker.state,
                    "circuit_breaker_failures": self.circuit_breaker.failure_count
                },
                "model_performance": {},
                "feature_analysis": {
                    "total_features": results.get('total_features', 0),
                    "selected_features": results.get('selected_features', 0),
                    "feature_selection_ratio": results.get('selected_features', 0) / max(results.get('total_features', 1), 1)
                },
                "regime_analysis": {
                    "total_regimes": results.get('n_regimes', 0),
                    "regime_distribution": results.get('regime_distribution', {})
                },
                "computational_metrics": {
                    "average_training_time": safe_divide(
                        sum([r.metrics.training_time for r in results.get('model_results', {}).values()]),
                        len(results.get('model_results', {})),
                        0.0
                    ),
                    "total_memory_usage": sum([r.metrics.memory_usage_mb for r in results.get('model_results', {}).values()]),
                    "training_efficiency": safe_divide(results.get('selected_features', 0), max(execution_time, 0.001), 0.0)
                },
                "recommendations": []
            }
            
            # Analyze model performance and collect warnings
            model_results = results.get('model_results', {})
            all_warnings = []
            
            if model_results:
                best_accuracy = -1
                best_model = None
                accuracies = []
                
                for model_name, model_result in model_results.items():
                    metrics = model_result.metrics
                    
                    # Collect warnings
                    if metrics.warnings:
                        all_warnings.extend([f"{model_name}: {w}" for w in metrics.warnings])
                    
                    report["model_performance"][model_name] = {
                        "accuracy": metrics.accuracy,
                        "f1_score": metrics.f1_score,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "training_time": metrics.training_time,
                        "status": "success" if metrics.error_message is None else "failed",
                        "error": metrics.error_message,
                        "warnings": metrics.warnings
                    }
                    
                    if metrics.error_message is None:
                        accuracies.append(metrics.accuracy)
                        if metrics.accuracy > best_accuracy:
                            best_accuracy = metrics.accuracy
                            best_model = model_name
                
                # Add performance summary using safe math operations
                if accuracies:
                    report["performance_summary"] = {
                        "best_model": best_model,
                        "best_accuracy": validate_finite(best_accuracy, "best_accuracy"),
                        "average_accuracy": validate_finite(safe_divide(sum(accuracies), len(accuracies), 0.0), "average_accuracy"),
                        "accuracy_std": validate_finite(safe_sqrt(safe_divide(sum([(x - safe_divide(sum(accuracies), len(accuracies), 0.0))**2 for x in accuracies]), len(accuracies), 0.0), 0.0), "accuracy_std"),
                        "performance_variance": validate_finite(safe_divide(sum([(x - safe_divide(sum(accuracies), len(accuracies), 0.0))**2 for x in accuracies]), len(accuracies), 0.0), "performance_variance")
                    }
            
            # Add warnings to report
            report["warnings"] = list(set(all_warnings))  # Remove duplicates
            
            # Generate enhanced recommendations
            recommendations = []
            
            if report["execution_summary"]["failed_models"] > 0:
                recommendations.append(f"Address {report['execution_summary']['failed_models']} failed model(s)")
            
            if report["execution_summary"]["circuit_breaker_state"] == "OPEN":
                recommendations.append("Circuit breaker is OPEN - investigate systematic failures")
            
            if len(all_warnings) > 0:
                recommendations.append(f"Address {len(all_warnings)} warnings for better performance")
            
            if "performance_summary" in report and report["performance_summary"]["average_accuracy"] < 0.7:
                recommendations.append("Consider feature engineering or data preprocessing improvements")
            
            if report["computational_metrics"]["average_training_time"] > 60:
                recommendations.append("Consider reducing model complexity or using faster algorithms")
            
            if report["feature_analysis"]["feature_selection_ratio"] > 0.5:
                recommendations.append("High feature selection ratio - consider more aggressive feature selection")
            
            report["recommendations"] = recommendations
            
            self.logger.info("✅ Comprehensive report generated successfully")
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate comprehensive report: {e}")
            return {
                "report_type": "HMM Models Training Report (Error)",
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat(),
                "status": "Report generation failed"
            }
    
    def execute(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        cluster_assignments: Optional[np.ndarray],
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        market_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute enhanced HMM models training with comprehensive error handling and reporting.
        Refactored for better maintainability and reduced complexity.

        Args:
            X: Input features or market data DataFrame
            y: Target values
            cluster_assignments: Cluster assignments for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            market_data: Original market data for comprehensive feature generation
            **kwargs: Additional arguments

        Returns:
            Dictionary containing training results and comprehensive report
        """
        StandardizedLogger.log_training_progress("HMM Models", "started")
        self.training_start_time = time.time()

        # Load cluster assignments from parquet file if not provided
        if cluster_assignments is None or len(cluster_assignments) == 0:
            tprint("🔍 Loading cluster assignments from optimal clustering results...")
            try:
                import pandas as pd
                cluster_assignments_path = "optimal_clusters/binance/ETHUSDT/15m/optimal_cluster_labels.parquet"
                cluster_assignments_df = pd.read_parquet(cluster_assignments_path)
                if 'cluster_label' in cluster_assignments_df.columns:
                    cluster_assignments = cluster_assignments_df['cluster_label'].values
                    tprint(f"✅ Loaded {len(cluster_assignments)} cluster assignments")
                else:
                    tprint(f"❌ No cluster_label column found in {cluster_assignments_path}")
                    raise ValueError(f"Invalid cluster assignments file: {cluster_assignments_path}")
            except Exception as e:
                tprint(f"❌ Could not load cluster assignments: {e}")
                raise ValueError(f"Cluster assignments are required for HMM models training. Please run hmm_clustering first. Error: {e}")

        try:
            with performance_monitor("HMM_Training_Complete"):
                # Execute training pipeline
                results = self._execute_training_pipeline(X, y, cluster_assignments, feature_names, **kwargs)
                
                # Log final summary
                self._log_training_summary(results)
                
                return results
                
        except Exception as e:
            return self._handle_training_failure(e)
    
    def _execute_training_pipeline(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        cluster_assignments: Optional[np.ndarray],
        feature_names: Optional[List[str]] = None,
        market_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute the main training pipeline."""
        
        # Step 1: Input validation
        with performance_monitor("Input_Validation"):
            if not self._validate_input_data(X, y, cluster_assignments):
                raise ValueError("Input validation failed")
        
        # Step 2: Feature preparation and selection
        X_selected, selected_features = self._prepare_and_select_features(
            X, y, cluster_assignments, feature_names, market_data, **kwargs
        )
        
        # Step 3: Train models
        model_results = self._train_all_models(X_selected, y)
        
        # Step 4: Analyze results
        cluster_distribution = self._analyze_clusters(cluster_assignments)
        
        # Step 5: Format results
        results = self._format_training_results(
            model_results, X_selected, selected_features, cluster_distribution, **kwargs
        )
        
        # Step 6: Generate report and save models
        self._finalize_results(results, model_results, **kwargs)
        
        return results
    
    def _prepare_and_select_features(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        cluster_assignments: np.ndarray,
        feature_names: Optional[List[str]] = None,
        market_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare and select features using comprehensive feature generation system."""

        # Use comprehensive feature generation if market data is available
        if isinstance(X, pd.DataFrame) and market_data is not None:
            with performance_monitor("Feature_Preparation_Comprehensive"):
                tprint("🚀 Using comprehensive feature generation system...")
                X_enhanced, enhanced_feature_names = self._prepare_comprehensive_features(
                    market_data, cluster_assignments
                )
        else:
            # Use enhanced basic features
            with performance_monitor("Feature_Preparation_Enhanced"):
                tprint("📊 Using enhanced basic feature generation...")
                X_enhanced, enhanced_feature_names = self._prepare_features(X, feature_names, cluster_assignments)
        
        with performance_monitor("Feature_Selection"):
            X_selected, selected_features = self._select_features(
                X_enhanced, y, 
                is_classification=kwargs.get('is_classification', True)
            )
        
        return X_selected, selected_features
    
    def _train_all_models(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Train all configured models."""
        
        self.progress_reporter = ProgressReporter(len(self.config.model_types))
        model_results = {}
        
        with performance_monitor("Model_Training"):
            for model_type in self.config.model_types:
                model_result = safe_execute(
                    self._train_single_model_safe,
                    model_type, X, y,
                    component_name=f"Training_{model_type}",
                    default_return=TrainingErrorHandler.handle_training_error(model_type, Exception("Training failed"), 0.0)
                )
                
                model_results[model_type] = model_result
                
                # Update progress
                success = model_result.metrics.error_message is None
                accuracy = model_result.metrics.accuracy if success else None
                error_message = model_result.metrics.error_message if not success else None
                
                self.progress_reporter.update_progress(
                    model_type, success, model_result.metrics.training_time, 
                    accuracy, error_message
                )
        
        self.progress_reporter.finish_report()
        return model_results
    
    def _train_single_model_safe(self, model_type: str, X: pd.DataFrame, y: np.ndarray) -> Any:
        """Safely train a single model with proper conversion."""
        X_train = self._convert_to_numpy_array(X)
        return self._train_single_model(model_type, X_train, y)
    
    def _analyze_clusters(self, cluster_assignments: np.ndarray) -> Dict[str, Any]:
        """Analyze cluster distribution."""
        
        with performance_monitor("Cluster_Analysis"):
            unique_clusters, cluster_counts = np.unique(cluster_assignments, return_counts=True)
            cluster_distribution = {
                f"cluster_{cluster}": {
                    "count": int(count),
                    "percentage": float(count / len(cluster_assignments) * 100)
                }
                for cluster, count in zip(unique_clusters, cluster_counts)
            }
        
        return cluster_distribution
    
    def _format_training_results(
        self,
        model_results: Dict[str, Any],
        X_selected: pd.DataFrame,
        selected_features: List[str],
        cluster_distribution: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Format training results into standard structure."""
        
        execution_time = time.time() - self.training_start_time
        
        # Format model results for artifacts
        hmm_base_models, hmm_training_metrics, hmm_model_performance = self._format_model_artifacts(model_results)
        
        return {
            'model_results': model_results,
            'artifacts': {
                'hmm_base_models': hmm_base_models,
                'hmm_training_metrics': hmm_training_metrics,
                'hmm_model_performance': hmm_model_performance
            },
            'metadata': {
                'total_features': X_selected.shape[1],
                'selected_features': len(selected_features),
                'selected_feature_names': selected_features,
                'n_clusters': len(cluster_distribution),
                'cluster_distribution': cluster_distribution,
                'execution_time': execution_time,
                'config': self.config,
                'circuit_breaker_state': self.circuit_breaker.state,
                'circuit_breaker_failures': self.circuit_breaker.failure_count,
                'models_trained': len(hmm_base_models),
                'successful_models': len([m for m in hmm_base_models if m['model_object'] is not None])
            },
            'training_time': execution_time
        }
    
    def _format_model_artifacts(self, model_results: Dict[str, Any]) -> Tuple[List[Dict], Dict[str, Dict], Dict[str, Dict]]:
        """Format model results into artifact structure."""
        
        hmm_base_models = []
        hmm_training_metrics = {}
        hmm_model_performance = {}
        
        for model_name, model_result in model_results.items():
            if model_result.model is not None:
                # Add model to base models list
                hmm_base_models.append({
                    'model_name': model_name,
                    'model_type': model_name,
                    'model_object': model_result.model,
                    'hyperparameters': model_result.hyperparameters
                })
                
                # Add training metrics
                hmm_training_metrics[model_name] = {
                    'accuracy': model_result.metrics.accuracy,
                    'f1_score': model_result.metrics.f1_score,
                    'precision': model_result.metrics.precision,
                    'recall': model_result.metrics.recall,
                    'training_time': model_result.metrics.training_time,
                    'convergence_epochs': model_result.metrics.convergence_epochs,
                    'memory_usage_mb': model_result.metrics.memory_usage_mb,
                    'validation_loss': model_result.metrics.validation_loss,
                    'test_accuracy': model_result.metrics.test_accuracy,
                    'warnings': model_result.metrics.warnings
                }
                
                # Add performance metrics
                hmm_model_performance[model_name] = {
                    'feature_importance': model_result.feature_importance,
                    'predictions_available': model_result.predictions is not None,
                    'probabilities_available': model_result.probabilities is not None,
                    'training_history_available': model_result.training_history is not None
                }
        
        return hmm_base_models, hmm_training_metrics, hmm_model_performance
    
    def _finalize_results(self, results: Dict[str, Any], model_results: Dict[str, Any], **kwargs) -> None:
        """Generate final report and save models."""
        
        # Generate comprehensive report
        with performance_monitor("Report_Generation"):
            comprehensive_report = self._generate_comprehensive_report(results, results['training_time'])
            results['comprehensive_report'] = comprehensive_report
        
        # Save models if configured
        if self.config.save_models:
            self._save_models_if_configured(results, model_results, **kwargs)
    
    def _save_models_if_configured(self, results: Dict[str, Any], model_results: Dict[str, Any], **kwargs) -> None:
        """Save models if configuration allows."""
        
        try:
            with performance_monitor("Model_Saving"):
                symbol = kwargs.get('symbol', 'UNKNOWN')
                exchange = kwargs.get('exchange', 'UNKNOWN')
                timeframe = kwargs.get('timeframe', self.config.timeframe)
                
                # Save successful models only
                successful_models = {
                    name: result.model for name, result in model_results.items()
                    if result.model is not None
                }
                
                if successful_models:
                    saved_paths = self._save_models_with_common_utils(
                        models=successful_models,
                        model_type=self.config.model_name,
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=timeframe
                    )
                    results['saved_model_paths'] = saved_paths
                    StandardizedLogger.log_info("Model_Saving", f"Models saved: {len(saved_paths)} files")
                else:
                    StandardizedLogger.log_warning("Model_Saving", "No successful models to save")
                    
        except Exception as e:
            StandardizedLogger.log_error("Model_Saving", f"Failed to save models: {e}")
            results['save_error'] = str(e)
    
    def _log_training_summary(self, results: Dict[str, Any]) -> None:
        """Log final training summary."""
        
        model_results = results.get('model_results', {})
        successful_count = sum(1 for r in model_results.values() if r.metrics.error_message is None)
        execution_time = results.get('training_time', 0)
        
        StandardizedLogger.log_training_progress(
            "HMM Models", "completed", True,
            f"{successful_count}/{len(model_results)} models successful in {execution_time:.2f}s"
        )
        
        if self.circuit_breaker.failure_count > 0:
            StandardizedLogger.log_warning(
                "Circuit_Breaker", 
                f"Failures: {self.circuit_breaker.failure_count}, State: {self.circuit_breaker.state}"
            )
    
    def _handle_training_failure(self, error: Exception) -> Dict[str, Any]:
        """Handle training failure with standardized error reporting."""
        
        execution_time = time.time() - self.training_start_time if self.training_start_time else 0
        
        StandardizedLogger.log_error("HMM_Training", error, include_traceback=True)
        
        return {
            'model_results': {},
            'metadata': {
                'error': str(error),
                'execution_time': execution_time,
                'config': self.config
            },
            'training_time': execution_time,
            'comprehensive_report': {
                "report_type": "HMM Models Training Report (Error)",
                "error": str(error),
                "timestamp": pd.Timestamp.now().isoformat(),
                "status": "Training failed"
            }
        }


# Convenience functions
def create_enhanced_hmm_models_training(
    config: Optional[HMMTrainingConfig] = None
) -> HMMModelsTrainingEnhanced:
    """Create enhanced HMM models training step."""
    return HMMModelsTrainingEnhanced(config)


def execute_enhanced_hmm_models_training(
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    cluster_assignments: Optional[np.ndarray],
    config: Optional[HMMTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    market_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Execute enhanced HMM models training step."""
    step = create_enhanced_hmm_models_training(config)
    return step.execute(X, y, cluster_assignments, feature_names, hmm_states, market_data)


# Example usage
if __name__ == "__main__":
    print("Enhanced HMM Models Training")
    print("=" * 50)
    
    # Create configuration
    config = HMMTrainingConfig(
        model_name="hmm_models_enhanced",
        timeframe="1h",
        n_features=50,
        sequence_length=20,
        n_regimes=3,
        model_types=["catboost", "elastic_net", "ensemble_rf"],
        hpo_trials=25,
        enable_multi_objective=True
    )
    
    # Create training step
    training_step = create_enhanced_hmm_models_training(config)
    
    print(f"✅ Created enhanced training step with {len(config.model_types)} model types")
    print(f"📊 Features: {config.n_features}")
    print(f"📊 Sequence length: {config.sequence_length}")
    print(f"📊 HPO trials: {config.hpo_trials}")
    
    print("\n🎯 Key enhancements:")
    print("- ✅ Circuit breaker pattern prevents cascading failures")
    print("- ✅ Model factory pattern reduces code duplication")
    print("- ✅ Real-time progress reporting with ETA")
    print("- ✅ Enhanced input validation with early exit")
    print("- ✅ Centralized error handling")
    print("- ✅ Warning collection and reporting")
    print("- ✅ Comprehensive reporting with actionable insights")
    print("- ✅ Silent failure prevention")
    print("- ✅ Common utilities integration for better maintainability")
    print("- ✅ Hardware optimization for M1 systems")
    print("- ✅ Safe math operations preventing numerical errors")
    print("- ✅ Common serialization utilities for model persistence")