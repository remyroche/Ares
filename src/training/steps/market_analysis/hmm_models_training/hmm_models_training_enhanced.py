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
from src.utils.ml_common.config.base_training_config import HMMTrainingConfig
from src.utils.ml_common.training.base_training_step import BaseTrainingStep

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
    EvaluationUtils = None

try:
    from src.utils.ml_common.validation.validation_utils import ValidationUtils as MLValidationUtils
    ML_VALIDATION_AVAILABLE = True
except ImportError:
    ML_VALIDATION_AVAILABLE = False
    MLValidationUtils = None

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

# Shared utilities with error handling
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
    # Create fallback classes
    class TrainingErrorHandler:
        @staticmethod
        def handle_training_error(model_type, error, training_time):
            from dataclasses import dataclass
            @dataclass
            class TrainingMetrics:
                error_message: str = ""
                training_time: float = 0.0
            @dataclass 
            class ModelResult:
                model: None = None
                metrics: TrainingMetrics = None
            return ModelResult(None, TrainingMetrics(f"Failed to train {model_type}: {error}", training_time))
    
    class UnifiedModelFactory:
        @staticmethod
        def create_model(model_type, **kwargs):
            # Check library availability first
            if model_type == 'lightgbm' or model_type == 'lgbm':
                if not ML_LIBRARIES_STATUS.get('lightgbm', False):
                    raise ImportError(f"LightGBM not available for model type: {model_type}")
                import lightgbm
                return lightgbm.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, **kwargs)
            elif model_type == 'xgboost':
                if not ML_LIBRARIES_STATUS.get('xgboost', False):
                    raise ImportError(f"XGBoost not available for model type: {model_type}")
                import xgboost
                return xgboost.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, **kwargs)
            elif model_type in ['random_forest', 'rf']:
                if not ML_LIBRARIES_STATUS.get('sklearn', False):
                    raise ImportError(f"Scikit-learn not available for model type: {model_type}")
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=100, random_state=42, **kwargs)
            elif model_type in ['logistic_regression', 'lr']:
                if not ML_LIBRARIES_STATUS.get('sklearn', False):
                    raise ImportError(f"Scikit-learn not available for model type: {model_type}")
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(random_state=42, max_iter=1000, **kwargs)
            elif model_type in ['elastic_net_lr', 'elastic_net']:
                if not ML_LIBRARIES_STATUS.get('sklearn', False):
                    raise ImportError(f"Scikit-learn not available for model type: {model_type}")
                from sklearn.linear_model import LogisticRegression
                # Ensure compatible parameters for elastic net
                safe_kwargs = {k: v for k, v in kwargs.items() if k not in ['penalty', 'solver', 'l1_ratio']}
                return LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', random_state=42, max_iter=1000, **safe_kwargs)
            else:
                available_models = ['lightgbm', 'xgboost', 'random_forest', 'logistic_regression', 'elastic_net_lr']
                raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")
    
    class CircuitBreaker:
        def __init__(self, *args, **kwargs):
            self.state = "CLOSED"
            self.failure_count = 0
        def call(self, func, *args, **kwargs):
            return func(*args, **kwargs)
    
    class ValidationUtils:
        @staticmethod
        def validate_config(config):
            return True
        @staticmethod
        def validate_data_shapes(*args):
            return True
        @staticmethod
        def validate_data_quality(*args):
            return True
        @staticmethod
        def validate_regime_distribution(*args, **kwargs):
            return True
    
    class ProgressReporter:
        def __init__(self, *args, **kwargs):
            pass
        def update_progress(self, *args, **kwargs):
            pass
        def finish_report(self):
            pass
    
    class MemoryTracker:
        def __init__(self):
            pass
        def take_snapshot(self, name):
            pass
        def get_memory_increase(self):
            return 0.0
        def cleanup(self):
            pass


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
            # Use only available model types
            available_models = []
            if ML_LIBRARIES_STATUS.get('sklearn', False):
                available_models.extend(['logistic_regression', 'random_forest', 'elastic_net_lr'])
            if ML_LIBRARIES_STATUS.get('lightgbm', False):
                available_models.append('lightgbm')
            if ML_LIBRARIES_STATUS.get('xgboost', False):
                available_models.append('xgboost')
            
            # Fallback to sklearn models if nothing else available
            if not available_models:
                available_models = ['logistic_regression', 'random_forest']
                
            config = HMMTrainingConfig(
                model_name="hmm_models_enhanced",
                timeframe="1h",
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

        self.circuit_breaker = CircuitBreaker(failure_threshold=2, timeout=60)
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
            if SHARED_UTILITIES_AVAILABLE and not ValidationUtils.validate_config(config):
                raise ValueError("Configuration validation failed")
            
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
            
            # Validate model types are supported
            available_model_types = ['lightgbm', 'xgboost', 'random_forest', 'logistic_regression', 'elastic_net_lr']
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
    
    def _validate_input_data(self, X: np.ndarray, y: np.ndarray, cluster_assignments: np.ndarray) -> bool:
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
            # Use common validation utilities if available
            if SHARED_UTILITIES_AVAILABLE:
                if not ValidationUtils.validate_data_shapes(X, y, cluster_assignments):
                    return False
                
                if not ValidationUtils.validate_data_quality(X, y, cluster_assignments):
                    return False
                
                if not ValidationUtils.validate_regime_distribution(cluster_assignments, min_samples_per_regime=10):
                    return False
            
            # Additional HMM-specific validations using common math validation
            critical_failures = []
            warnings = []
            unique_clusters = np.unique(cluster_assignments)
            
            if len(unique_clusters) < 2:
                critical_failures.append(f"Need at least 2 clusters, found {len(unique_clusters)}")
            
            # Early exit on critical failures
            if critical_failures:
                tprint(f"❌ Critical validation failures: {critical_failures}")
                return False
            
            # Warning checks (don't cause early exit)
            if len(X) < 1000:
                warnings.append(f"Small dataset: {len(X)} samples (recommended: >1000)")
            
            # Check minimum samples per cluster
            for cluster in unique_clusters:
                cluster_count = np.sum(cluster_assignments == cluster)
                if cluster_count < 10:  # Minimum samples per cluster
                    warnings.append(f"Cluster {cluster} has only {cluster_count} samples (minimum: 10)")
            
            # Log warnings
            if warnings:
                for warning in warnings:
                    tprint(f"⚠️ {warning}")
            
            tprint(f"✅ Enhanced validation passed: {len(X)} samples, {len(unique_clusters)} clusters")
            return True
            
        except Exception as e:
            tprint(f"❌ Validation error: {e}")
            return False
    
    def _prepare_features(self, X: Union[np.ndarray, pd.DataFrame], feature_names: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare and enhance features with comprehensive error handling.
        Uses common utilities for better data processing and validation.
        
        Args:
            X: Input features
            feature_names: Optional feature names
            
        Returns:
            Tuple of (enhanced_features, feature_names)
        """
        try:
            # Convert to DataFrame if needed using common utilities
            if isinstance(X, np.ndarray):
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                X_df = pd.DataFrame(X, columns=feature_names)
            else:
                X_df = X.copy()
                if feature_names is None:
                    feature_names = list(X_df.columns)
            
            # Use common DataFrame operations for numeric column selection
            numeric_columns = safe_dataframe_operation(
                X_df, 
                lambda df: df.select_dtypes(include=[np.number]).columns
            )
            
            if len(numeric_columns) == 0:
                raise ValueError("No numeric columns found in input data")
            
            X_numeric = X_df[numeric_columns]
            tprint(f"📊 Using {len(numeric_columns)} numeric features")
            
            # Calculate data quality metrics using common utilities
            quality_metrics = calculate_data_quality_metrics(X_numeric)
            if quality_metrics.get('null_percentage', 0) > 0.1:
                tprint(f"⚠️ High null percentage: {quality_metrics.get('null_percentage', 0):.2%}")
            
            # Enhance features if generator is available
            if self.feature_generator is not None:
                try:
                    X_enhanced = self.feature_generator.generate_features_for_hmm(X_numeric)
                    tprint(f"✅ Enhanced features: {X_enhanced.shape[1]} total features")
                    return X_enhanced, list(X_enhanced.columns)
                except Exception as e:
                    tprint(f"⚠️ Feature enhancement failed: {e}, using original features")
                    return X_numeric, list(X_numeric.columns)
            else:
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
            
            # Validate selected features exist in the DataFrame
            missing_features = [f for f in selected_features if f not in X.columns]
            if missing_features:
                self.logger.warning(f"⚠️ Some selected features not found in DataFrame: {missing_features}")
                # Filter out missing features
                selected_features = [f for f in selected_features if f in X.columns]
            
            if not selected_features:
                self.logger.warning("⚠️ No valid features selected, using first n_features")
                selected_features = list(X.columns)[:self.config.n_features]
            
            X_selected = X[selected_features]
            
            self.logger.info(f"✅ Feature selection completed: {len(selected_features)} features selected")
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
                
                # Apply hardware optimizations if available
                if self.memory_optimizer:
                    try:
                        self.memory_optimizer.optimize_memory_usage()
                        tprint(f"🧠 Memory optimization applied for {model_type}")
                    except Exception as e:
                        tprint(f"⚠️ Memory optimization failed for {model_type}: {e}")
                
                if self.cpu_optimizer:
                    try:
                        self.cpu_optimizer.optimize_cpu_usage()
                        tprint(f"⚡ CPU optimization applied for {model_type}")
                    except Exception as e:
                        tprint(f"⚠️ CPU optimization failed for {model_type}: {e}")
                
                # Memory management before training using common operations
                initial_memory = get_memory_usage()
                if initial_memory > 80:
                    tprint(f"⚠️ High memory usage ({initial_memory}%) before training {model_type}")
                    optimize_memory()  # Use common memory optimization
            
            # Create model with circuit breaker protection and timeout
            def create_and_train_model():
                # Add timeout for model creation and training
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Training timeout for {model_type}")
                
                # Set timeout based on training mode (light = 60s, others = 300s)
                timeout_seconds = 60 if hasattr(self.config, 'training_mode_config') and \
                                     self.config.training_mode_config and \
                                     self.config.training_mode_config.get('training_mode') == 'light' else 300
                
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout_seconds)
                    
                    model = self._create_model(model_type)
                    
                    # Check memory before training using common operations
                    current_memory = get_memory_usage()
                    if current_memory > 85:
                        tprint(f"⚠️ Memory usage critical ({current_memory}%) - skipping {model_type}")
                        raise MemoryError(f"Insufficient memory for {model_type} training")
                    
                    # Train model
                    model.fit(X, y)
                    
                    # Cancel timeout
                    signal.alarm(0)
                    
                except TimeoutError as e:
                    signal.alarm(0)
                    raise e
                except MemoryError as e:
                    signal.alarm(0)
                    optimize_memory()  # Use common memory optimization
                    raise e
                except Exception as e:
                    signal.alarm(0)
                    raise e
                
                # Evaluate model using safe math operations
                predictions = model.predict(X)
                accuracy = safe_divide(np.sum(predictions == y), len(y), 0.0)
                
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
                    except Exception as e:
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
                    except Exception as e:
                        metrics.warnings.append(f"Fallback metrics calculation failed: {e}")
                        metrics.accuracy = validate_finite(accuracy, "accuracy")
                
                # Get feature importance using safe operations
                feature_importance = None
                try:
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        feature_importance = dict(zip(range(len(importances)), [validate_finite(x, f"importance_{i}") for i, x in enumerate(importances)]))
                    elif hasattr(model, 'coef_'):
                        coefs = np.abs(model.coef_[0])
                        feature_importance = dict(zip(range(len(coefs)), [validate_finite(x, f"coef_{i}") for i, x in enumerate(coefs)]))
                except Exception as e:
                    tprint(f"Feature importance not available for {model_type}: {e}")
                
                # Get hyperparameters
                hyperparameters = None
                try:
                    if hasattr(model, 'get_params'):
                        hyperparameters = model.get_params()
                except Exception as e:
                    tprint(f"Could not get hyperparameters: {e}")
                
                return model, predictions, feature_importance, hyperparameters
            
                # Execute with circuit breaker protection
                model, predictions, feature_importance, hyperparameters = self.circuit_breaker.call(create_and_train_model)
                
                training_time = time.time() - start_time
                metrics.training_time = validate_finite(training_time, "training_time")
                
                # Calculate memory usage using common operations
                final_memory = get_memory_usage()
                memory_increase = safe_float(final_memory - initial_memory, 0.0)
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
                if final_memory > initial_memory + 10:
                    tprint(f"⚠️ Memory usage increased significantly: {initial_memory}% → {final_memory}%")
                    optimize_memory()  # Use common memory optimization
                
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
                final_memory = get_memory_usage()
                memory_increase = safe_float(final_memory - initial_memory, 0.0)
                metrics.memory_usage_mb = validate_finite(memory_increase, "memory_usage")
                
                tprint(f"❌ Failed to train {model_type}: {e}")
                
                # Cleanup memory using common operations
                optimize_memory()
                
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
            
            if report["performance_summary"]["average_accuracy"] < 0.7:
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
        X: np.ndarray,
        y: np.ndarray,
        cluster_assignments: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute enhanced HMM models training with comprehensive error handling and reporting.
        
        Args:
            X: Input features
            y: Target values
            cluster_assignments: Cluster assignments for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing training results and comprehensive report
        """
        self.logger.info("🚀 Starting Enhanced HMM Models Training")
        self.training_start_time = time.time()
        
        try:
            # Step 1: Input validation
            self.logger.info("🔄 Step 1: Validating inputs...")
            if not self._validate_input_data(X, y, cluster_assignments):
                raise ValueError("Input validation failed")
            
            # Step 2: Feature preparation
            self.logger.info("🔄 Step 2: Preparing features...")
            X_enhanced, enhanced_feature_names = self._prepare_features(X, feature_names)
            
            # Step 3: Feature selection
            self.logger.info("🔄 Step 3: Selecting features...")
            X_selected, selected_features = self._select_features(
                X_enhanced, y, 
                is_classification=kwargs.get('is_classification', True)
            )
            
            # Step 4: Initialize progress reporter and train models
            self.logger.info("🔄 Step 4: Training models with real-time progress...")
            self.progress_reporter = ProgressReporter(len(self.config.model_types))
            model_results = {}
            
            for model_type in self.config.model_types:
                try:
                    # Convert to numpy array for training with proper validation
                    X_train = self._convert_to_numpy_array(X_selected)
                    
                    # Train model
                    model_result = self._train_single_model(model_type, X_train, y)
                    model_results[model_type] = model_result
                    
                    # Update progress
                    success = model_result.metrics.error_message is None
                    accuracy = model_result.metrics.accuracy if success else None
                    error_message = model_result.metrics.error_message if not success else None
                    self.progress_reporter.update_progress(
                        model_type, success, model_result.metrics.training_time, 
                        accuracy, error_message
                    )
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to train {model_type}: {e}")
                    # Create failed result using centralized error handler
                    model_results[model_type] = TrainingErrorHandler.handle_training_error(model_type, e, 0.0)
                    self.progress_reporter.update_progress(model_type, False, 0.0, None, str(e))
            
            # Step 5: Finish progress reporting
            self.progress_reporter.finish_report()
            
            # Step 6: Analyze clusters
            unique_clusters, cluster_counts = np.unique(cluster_assignments, return_counts=True)
            cluster_distribution = {
                f"cluster_{cluster}": {
                    "count": int(count),
                    "percentage": float(count / len(cluster_assignments) * 100)
                }
                for cluster, count in zip(unique_clusters, cluster_counts)
            }
            
            # Step 7: Create final results with proper artifact formatting
            execution_time = time.time() - self.training_start_time
            
            # Format model results for artifacts
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
            
            results = {
                'model_results': model_results,
                'artifacts': {
                    'hmm_base_models': hmm_base_models,
                    'hmm_training_metrics': hmm_training_metrics,
                    'hmm_model_performance': hmm_model_performance
                },
                'metadata': {
                    'total_features': X_enhanced.shape[1],
                    'selected_features': len(selected_features),
                    'selected_feature_names': selected_features,
                    'n_clusters': len(unique_clusters),
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
            
            # Step 8: Generate comprehensive report
            self.logger.info("🔄 Step 8: Generating comprehensive report...")
            comprehensive_report = self._generate_comprehensive_report(results, execution_time)
            results['comprehensive_report'] = comprehensive_report
            
            # Step 9: Save results if configured using common serialization utilities
            if self.config.save_models:
                tprint("🔄 Step 9: Saving models...")
                try:
                    symbol = kwargs.get('symbol', 'UNKNOWN')
                    exchange = kwargs.get('exchange', 'UNKNOWN')
                    timeframe = kwargs.get('timeframe', self.config.timeframe)
                    
                    # Save successful models only
                    successful_models = {
                        name: result.model for name, result in model_results.items()
                        if result.model is not None
                    }
                    
                    if successful_models:
                        # Use common serialization utilities
                        saved_paths = self._save_models_with_common_utils(
                            models=successful_models,
                            model_type=self.config.model_name,
                            symbol=symbol,
                            exchange=exchange,
                            timeframe=timeframe
                        )
                        results['saved_model_paths'] = saved_paths
                        tprint(f"✅ Models saved: {len(saved_paths)} files")
                    else:
                        tprint("⚠️ No successful models to save")
                        
                except Exception as e:
                    tprint(f"❌ Failed to save models: {e}")
                    results['save_error'] = str(e)
            
            # Log enhanced final summary
            successful_count = sum(1 for r in model_results.values() if r.metrics.error_message is None)
            tprint(f"✅ Enhanced HMM Models Training completed: {successful_count}/{len(model_results)} models successful")
            tprint(f"📊 Total execution time: {execution_time:.2f}s")
            tprint(f"🔧 Circuit breaker state: {self.circuit_breaker.state}")
            if self.circuit_breaker.failure_count > 0:
                tprint(f"⚠️ Circuit breaker failures: {self.circuit_breaker.failure_count}")
            
            return results
            
        except Exception as e:
            execution_time = time.time() - self.training_start_time if self.training_start_time else 0
            tprint(f"❌ Enhanced HMM Models Training failed: {e}")
            
            return {
                'model_results': {},
                'metadata': {
                    'error': str(e),
                    'execution_time': execution_time,
                    'config': self.config
                },
                'training_time': execution_time,
                'comprehensive_report': {
                    "report_type": "HMM Models Training Report (Error)",
                    "error": str(e),
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
    X: np.ndarray,
    y: np.ndarray,
    cluster_assignments: np.ndarray,
    config: Optional[HMMTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Execute enhanced HMM models training step."""
    step = create_enhanced_hmm_models_training(config)
    return step.execute(X, y, cluster_assignments, feature_names, hmm_states)


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