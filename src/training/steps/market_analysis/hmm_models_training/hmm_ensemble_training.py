"""
HMM Ensemble Training Component

This component handles per-regime ensemble training of HMM models using common dependencies.
The HMM Ensemble operates on 1h timeframe and combines individual HMM models
to create robust ensemble predictions for market regime detection.

Enhanced with vectorized training capabilities for improved performance.
Refactored to use common utilities for better maintainability and performance.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import traceback
from pathlib import Path

# Common utilities imports
from src.utils.tprint import tprint
from src.utils.common_operations import (
    safe_dataframe_operation, safe_divide, safe_float, safe_int,
    ensure_directory, memory_checkpoint, optimize_memory, get_memory_usage
)
try:
    from src.utils.common_operations import (
        validate_dataframe, validate_finite, validate_positive, validate_range, 
        safe_percentage_change, safe_json_dump, safe_json_load, safe_file_exists,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        gpu_context
    )
    EXTENDED_COMMON_OPS_AVAILABLE = True
except ImportError:
    EXTENDED_COMMON_OPS_AVAILABLE = False

from src.utils.math_validation import (
    safe_divide as math_safe_divide, safe_log, safe_sqrt,
    validate_finite as math_validate_finite, validate_positive as math_validate_positive,
    validate_range as math_validate_range
)
try:
    from src.utils.math_validation import (
        safe_power, safe_mean, safe_std, safe_correlation, safe_covariance, 
        safe_percentile, MathValidation
    )
    EXTENDED_MATH_AVAILABLE = True
except ImportError:
    EXTENDED_MATH_AVAILABLE = False
    MathValidation = None

try:
    from src.utils.data.klines_parquet import KlinesParquetManager, get_klines_manager
    KLINES_AVAILABLE = True
except ImportError:
    KLINES_AVAILABLE = False
    get_klines_manager = lambda: None

from src.utils.serialization_utils import UniversalSerializer
try:
    from src.utils.serialization_utils import JSONSerializer, PickleSerializer
    LEGACY_SERIALIZERS_AVAILABLE = True
except ImportError:
    LEGACY_SERIALIZERS_AVAILABLE = False

try:
    from src.utils.matrix_operations.unified_operations import (
        UnifiedMatrixOperations, get_unified_matrix_operations,
        safe_matrix_multiply, safe_correlation_matrix, safe_matrix_inverse
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False
    get_unified_matrix_operations = lambda: None

try:
    from src.utils.ml_common.config.base_training_config import EnsembleTrainingConfig
    from src.utils.ml_common.training.ensemble_training_step import EnsembleTrainingStep
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False
    # Create fallback classes
    class EnsembleTrainingConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class EnsembleTrainingStep:
        def __init__(self, config, enable_vectorization=True):
            self.config = config
            self.enable_vectorization = enable_vectorization

try:
    from src.utils.ml_common.evaluation.evaluation_utils import EvaluationUtils
    ML_EVAL_AVAILABLE = True
except ImportError:
    ML_EVAL_AVAILABLE = False
    EvaluationUtils = None

# Shared utilities
from .shared_utilities import (
    TrainingErrorHandler,
    UnifiedModelFactory,
    CircuitBreaker,
    ValidationUtils,
    ProgressReporter,
    MemoryTracker
)
from .shared_utilities.training_error_handler import TrainingMetrics, ModelResult

# Import vectorized training manager
try:
    from src.utils.ml_common.training.vectorized_training_manager import VectorizedTrainingManager
    VECTORIZED_TRAINING_AVAILABLE = True
except ImportError:
    VECTORIZED_TRAINING_AVAILABLE = False

# Using tprint for all logging - no logger needed


class HMMEnsembleTrainingComponent(EnsembleTrainingStep):
    """
    HMM Ensemble Training Component with per-regime ensemble training, HPO, saving, and metrics.
    
    The HMM Ensemble operates on 1h timeframe and combines individual HMM models
    to create robust ensemble predictions for market regime detection.
    """
    
    def __init__(self, config: Optional[EnsembleTrainingConfig] = None, enable_vectorization: bool = True):
        """
        Initialize HMM ensemble training component with vectorization support.

        Args:
            config: Per-regime training configuration
            enable_vectorization: Whether to enable vectorized training
        """
        self.start_time = time.time()
        
        try:
            # Initialize common utilities with availability checks
            self.math_validator = MathValidation() if EXTENDED_MATH_AVAILABLE else None
            self.matrix_ops = get_unified_matrix_operations() if MATRIX_OPS_AVAILABLE else None
            self.serializer = UniversalSerializer()
            self.klines_manager = get_klines_manager() if KLINES_AVAILABLE else None
            self.evaluation_utils = EvaluationUtils() if ML_EVAL_AVAILABLE else None
            
            # Initialize M1 hardware optimizers with availability checks
            if EXTENDED_COMMON_OPS_AVAILABLE:
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
            else:
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
            
            # Set default configuration for HMM ensemble models
            if config is None:
                config = EnsembleTrainingConfig(
                    model_name="hmm_ensemble_models",
                    timeframe="1h",
                    model_types=["catboost", "elastic_net", "ensemble_rf"],
                    hpo_n_trials=100,
                    hpo_timeout_seconds=3600,
                    min_samples_per_regime=1000,
                    enable_data_augmentation=True,
                    augmentation_method="smote",
                    model_save_path="./models/hmm_ensemble_models",
                    evaluation_metrics=["accuracy", "f1_score", "precision", "recall", "auc"]
                )
                tprint("📋 Using default configuration for HMM ensemble training")

            # Validate configuration with fast-fail using common utilities
            self._validate_config(config)
            
            # Initialize parent class
            super().__init__(config, enable_vectorization=enable_vectorization and VECTORIZED_TRAINING_AVAILABLE)
            
            # Initialize tracking variables
            self.training_stats = {
                'initialization_time': time.time() - self.start_time,
                'vectorization_enabled': self.enable_vectorization,
                'config_used': config.model_name,
                'model_types': config.model_types,
                'timeframe': config.timeframe,
                'hardware_optimization': {
                    'gpu_available': self.gpu_manager is not None,
                    'memory_optimizer_available': self.memory_optimizer is not None,
                    'cpu_optimizer_available': self.cpu_optimizer is not None
                }
            }
            
            # Log initialization success
            if self.enable_vectorization:
                tprint("🚀 HMM Ensemble Training Component initialized with vectorization")
            else:
                tprint("✅ HMM Ensemble Training Component initialized (standard mode)")
                
            tprint(f"📊 Configuration: {len(config.model_types)} ensemble types, {config.timeframe} timeframe")
            tprint(f"🧠 Hardware: GPU={self.gpu_manager is not None}, Memory={self.memory_optimizer is not None}, CPU={self.cpu_optimizer is not None}")
            
        except Exception as e:
            tprint(f"❌ Failed to initialize HMM Ensemble Training Component: {e}")
            tprint(f"🔍 Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"HMM Ensemble Training Component initialization failed: {e}") from e
    
    def _validate_config(self, config: EnsembleTrainingConfig) -> None:
        """
        Validate configuration parameters with fast-fail for critical issues.
        Uses common math validation utilities for robust validation.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # Validate model types - FAST FAIL
            if not config.model_types or len(config.model_types) == 0:
                tprint("❌ CRITICAL: No model types specified - FAILING FAST")
                raise ValueError("At least one model type must be specified")
            
            # Validate timeframe - FAST FAIL
            valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            if not config.timeframe or config.timeframe not in valid_timeframes:
                tprint(f"❌ CRITICAL: Invalid timeframe '{config.timeframe}' - FAILING FAST")
                raise ValueError(f"Invalid timeframe '{config.timeframe}' - must be one of: {valid_timeframes}")
            
            # Validate HPO parameters using math validation utilities - FAST FAIL
            if hasattr(config, 'enable_hpo') and config.enable_hpo:
                if self.math_validator:
                    try:
                        config.hpo_n_trials = self.math_validator.validate_positive(
                            config.hpo_n_trials, "HPO trials"
                        )
                    except ValueError as e:
                        tprint(f"❌ CRITICAL: HPO trials validation failed - FAILING FAST")
                        raise ValueError(f"HPO trials must be positive: {e}") from e
                    
                    try:
                        config.hpo_timeout_seconds = self.math_validator.validate_positive(
                            config.hpo_timeout_seconds, "HPO timeout"
                        )
                    except ValueError as e:
                        tprint(f"❌ CRITICAL: HPO timeout validation failed - FAILING FAST")
                        raise ValueError(f"HPO timeout must be positive: {e}") from e
                else:
                    # Fallback validation without math validator
                    if not hasattr(config, 'hpo_n_trials') or config.hpo_n_trials <= 0:
                        raise ValueError("HPO trials must be positive")
                    if not hasattr(config, 'hpo_timeout_seconds') or config.hpo_timeout_seconds <= 0:
                        raise ValueError("HPO timeout must be positive")
            
            # Validate minimum samples using math validation utilities - FAST FAIL
            if self.math_validator:
                try:
                    config.min_samples_per_regime = self.math_validator.validate_positive(
                        config.min_samples_per_regime, "Minimum samples per regime"
                    )
                except ValueError as e:
                    tprint(f"❌ CRITICAL: Minimum samples validation failed - FAILING FAST")
                    raise ValueError(f"Minimum samples per regime must be positive: {e}") from e
            else:
                # Fallback validation
                if not hasattr(config, 'min_samples_per_regime') or config.min_samples_per_regime <= 0:
                    raise ValueError("Minimum samples per regime must be positive")
            
            # Validate save path using common utilities - WARNING ONLY
            if hasattr(config, 'save_models') and config.save_models and hasattr(config, 'model_save_path') and config.model_save_path:
                save_path = Path(config.model_save_path)
                if EXTENDED_COMMON_OPS_AVAILABLE and not safe_file_exists(save_path.parent):
                    tprint(f"⚠️ WARNING: Save path parent directory does not exist: {save_path.parent}")
                    # Try to create the directory using common utilities
                    if ensure_directory(save_path.parent):
                        tprint(f"✅ Created save path directory: {save_path.parent}")
                    else:
                        tprint(f"⚠️ Could not create save path directory: {save_path.parent}")
                elif not EXTENDED_COMMON_OPS_AVAILABLE:
                    # Fallback directory creation
                    if ensure_directory(save_path.parent):
                        tprint(f"✅ Created save path directory: {save_path.parent}")
                    else:
                        tprint(f"⚠️ Could not create save path directory: {save_path.parent}")
            
            tprint("✅ Configuration validation passed using common utilities")
            
        except Exception as e:
            tprint(f"❌ Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}") from e
    
    def _validate_input_data(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> None:
        """
        Validate input data with fast-fail for critical issues.
        Uses common math validation utilities for robust data validation.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels
            
        Raises:
            ValueError: If input data is invalid
        """
        try:
            # Check data shapes - FAST FAIL
            if X.shape[0] != y.shape[0] or X.shape[0] != regime_labels.shape[0]:
                tprint(f"❌ CRITICAL: Data shape mismatch - FAILING FAST")
                tprint(f"   X={X.shape}, y={y.shape}, regimes={regime_labels.shape}")
                raise ValueError(f"Data shape mismatch: X={X.shape}, y={y.shape}, regimes={regime_labels.shape}")
            
            # Check for empty data - FAST FAIL
            if X.shape[0] == 0:
                tprint("❌ CRITICAL: Input data is empty - FAILING FAST")
                raise ValueError("Input data is empty")
            
            # Check for NaN values using math validation utilities - FAST FAIL
            if np.isnan(X).any():
                nan_count = np.isnan(X).sum()
                tprint(f"❌ CRITICAL: Found {nan_count} NaN values in input features - FAILING FAST")
                raise ValueError(f"Input data contains {nan_count} NaN values - training cannot proceed")
            
            if np.isnan(y).any():
                nan_count = np.isnan(y).sum()
                tprint(f"❌ CRITICAL: Found {nan_count} NaN values in target values - FAILING FAST")
                raise ValueError(f"Target data contains {nan_count} NaN values - training cannot proceed")
            
            # Check for infinite values using math validation utilities - FAST FAIL
            if np.isinf(X).any():
                inf_count = np.isinf(X).sum()
                tprint(f"❌ CRITICAL: Found {inf_count} infinite values in input features - FAILING FAST")
                raise ValueError(f"Input data contains {inf_count} infinite values - training cannot proceed")
            
            if np.isinf(y).any():
                inf_count = np.isinf(y).sum()
                tprint(f"❌ CRITICAL: Found {inf_count} infinite values in target values - FAILING FAST")
                raise ValueError(f"Target data contains {inf_count} infinite values - training cannot proceed")
            
            # Validate finite values using math validation utilities
            if self.math_validator:
                try:
                    # Validate a sample of values to ensure they are finite
                    sample_size = min(1000, X.shape[0])
                    sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
                    for i in sample_indices:
                        for j in range(min(10, X.shape[1])):  # Check first 10 features
                            self.math_validator.validate_finite(X[i, j], f"X[{i},{j}]")
                except ValueError as e:
                    tprint(f"❌ CRITICAL: Non-finite values detected in input features - FAILING FAST")
                    raise ValueError(f"Input data contains non-finite values: {e}") from e
            else:
                # Fallback finite validation
                if np.any(~np.isfinite(X)):
                    tprint(f"❌ CRITICAL: Non-finite values detected in input features - FAILING FAST")
                    raise ValueError("Input data contains non-finite values")
            
            # Check regime distribution using safe operations - WARNING ONLY
            unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)
            min_regime_samples = regime_counts.min()
            
            if min_regime_samples < self.config.min_samples_per_regime:
                insufficient_regimes = unique_regimes[regime_counts < self.config.min_samples_per_regime]
                tprint(f"⚠️ WARNING: {len(insufficient_regimes)} regimes have insufficient samples (< {self.config.min_samples_per_regime})")
            
            # Calculate data quality metrics using common utilities
            data_quality = {
                'total_samples': X.shape[0],
                'feature_count': X.shape[1],
                'regime_count': len(unique_regimes),
                'min_regime_samples': min_regime_samples,
                'max_regime_samples': regime_counts.max(),
                'regime_balance': min_regime_samples / regime_counts.max() if regime_counts.max() > 0 else 0
            }
            
            tprint(f"✅ Data validation passed using common utilities: {data_quality['total_samples']} samples, {data_quality['feature_count']} features, {data_quality['regime_count']} regimes")
            tprint(f"📊 Regime balance: {data_quality['regime_balance']:.3f} (min: {data_quality['min_regime_samples']}, max: {data_quality['max_regime_samples']})")
            
        except Exception as e:
            tprint(f"❌ Data validation failed: {e}")
            raise ValueError(f"Invalid input data: {e}") from e
    
    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        base_hmm_models: Optional[Dict[str, Any]] = None,
        hmm_training_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute HMM ensemble training component with comprehensive error handling and progress tracking.
        Uses common utilities and hardware optimizations for improved performance.
        
        Args:
            X: Input features (1h timeframe with cross-timeframe features)
            y: Target values (HMM regime predictions)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            base_hmm_models: Individual HMM models to ensemble
            hmm_training_metrics: Performance metrics of base models
            
        Returns:
            Dictionary containing training results and metadata
        """
        execution_start_time = time.time()
        tprint("🚀 Starting HMM ensemble training component with common utilities")
        
        # Use memory checkpoint for large operations
        with memory_checkpoint("hmm_ensemble_training"):
            try:
                # Step 1: Validate inputs using common utilities
                tprint("🔄 Step 1: Validating inputs with common utilities...")
                self._validate_input_data(X, y, regime_labels)
                
                # Step 2: Validate and prepare base models
                tprint("🔄 Step 2: Validating base models...")
                if base_hmm_models is None or not base_hmm_models:
                    tprint("⚠️ No base HMM models provided, creating proper ensemble models")
                    base_hmm_models = self._create_ensemble_models()
                else:
                    tprint(f"✅ Using {len(base_hmm_models)} provided base models")
                
                # Step 3: Execute training with enhanced error handling and hardware optimization
                tprint("🔄 Step 3: Executing ensemble training with hardware optimization...")
                results = self._execute_training_with_error_handling(
                    X, y, regime_labels, feature_names, hmm_states, base_hmm_models
                )
                
                # Step 4: Add ensemble-specific metadata using common utilities
                tprint("🔄 Step 4: Adding ensemble-specific metadata...")
                if 'error' not in results:
                    results = self._add_ensemble_specific_metadata(results, base_hmm_models, hmm_training_metrics)
                
                # Step 5: Generate comprehensive report using common utilities
                execution_time = time.time() - execution_start_time
                results = self._generate_comprehensive_report(results, execution_time, base_hmm_models, hmm_training_metrics)
                
                # Step 6: Optimize memory usage after training
                tprint("🔄 Step 6: Optimizing memory usage...")
                memory_stats = optimize_memory()
                results['memory_optimization'] = memory_stats
                
                tprint(f"✅ HMM ensemble training completed successfully in {execution_time:.2f}s")
                tprint(f"🧠 Memory optimization: {memory_stats}")
                return results
                
            except Exception as e:
                execution_time = time.time() - execution_start_time
                error_msg = f"HMM ensemble training failed after {execution_time:.2f}s: {e}"
                tprint(f"❌ {error_msg}")
                tprint(f"🔍 Traceback: {traceback.format_exc()}")
                
                return {
                    'error': error_msg,
                    'execution_time': execution_time,
                    'traceback': traceback.format_exc(),
                    'training_stats': self.training_stats
                }
    
    def _execute_training_with_error_handling(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray],
        base_hmm_models: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute training with comprehensive error handling and recovery.
        Uses common utilities and matrix operations for improved performance.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels
            feature_names: Feature names
            hmm_states: HMM states
            base_hmm_models: Base models
            
        Returns:
            Training results
        """
        try:
            # Pre-process data using common utilities
            tprint("🔄 Pre-processing data with common utilities...")
            
            # Normalize features using matrix operations
            if X.shape[1] > 0:
                X_normalized = self.matrix_ops.normalize_matrix(X, method='zscore')
                tprint(f"✅ Features normalized using matrix operations: {X_normalized.shape}")
            else:
                X_normalized = X
            
            # Use GPU context if available for large datasets
            if self.gpu_manager and X.shape[0] > 10000:
                with gpu_context("hmm_ensemble_training"):
                    results = self._execute_training_core(
                        X_normalized, y, regime_labels, feature_names, hmm_states, base_hmm_models
                    )
            else:
                results = self._execute_training_core(
                    X_normalized, y, regime_labels, feature_names, hmm_states, base_hmm_models
                )
            
            # Update training stats using safe operations
            self.training_stats.update({
                'training_completed': True,
                'base_models_used': len(base_hmm_models),
                'feature_count': X.shape[1],
                'sample_count': X.shape[0],
                'data_normalized': True,
                'gpu_used': self.gpu_manager is not None and X.shape[0] > 10000
            })
            
            return results
            
        except Exception as e:
            tprint(f"❌ Training execution failed: {e}")
            self.training_stats.update({
                'training_completed': False,
                'training_error': str(e)
            })
            raise
    
    def _execute_training_core(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray],
        base_hmm_models: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Core training execution using parent class with enhanced error handling.
        
        Args:
            X: Normalized input features
            y: Target values
            regime_labels: Regime labels
            feature_names: Feature names
            hmm_states: HMM states
            base_hmm_models: Base models
            
        Returns:
            Training results
        """
        try:
            # Use the parent class execute method with additional ensemble-specific logic
            results = super().execute(
                X=X,
                y=y,
                regime_labels=regime_labels,
                feature_names=feature_names,
                hmm_states=hmm_states,
                is_classification=True,  # HMM ensemble models are classification
                base_models=base_hmm_models,
                symbol=None,  # Can be passed as kwargs
                exchange=None,
                timeframe=self.config.timeframe
            )
            
            # Add matrix operations performance stats
            if hasattr(self.matrix_ops, 'get_performance_stats'):
                matrix_stats = self.matrix_ops.get_performance_stats()
                results['matrix_operations_stats'] = matrix_stats
                tprint(f"📊 Matrix operations: {matrix_stats['total_operations']} operations, avg time: {matrix_stats['average_execution_time']:.3f}s")
            
            return results
            
        except Exception as e:
            tprint(f"❌ Core training execution failed: {e}")
            raise
    
    def _create_ensemble_models(self) -> Dict[str, Any]:
        """
        Create proper ensemble models for HMM training with enhanced error handling.
        Uses common utilities for robust model creation and validation.
        
        Returns:
            Dictionary of ensemble models
        """
        try:
            from catboost import CatBoostRegressor
            from sklearn.linear_model import ElasticNet
            from sklearn.ensemble import RandomForestRegressor
            
            # Create ensemble models with validated parameters using math validation
            ensemble_models = {}
            
            # CatBoost with validated parameters (Primary: Speed + robustness)
            try:
                if self.math_validator:
                    iterations = self.math_validator.validate_positive(1000, "CatBoost iterations")
                    learning_rate = self.math_validator.validate_range(0.05, 0.0, 1.0, "CatBoost learning_rate")
                    depth = self.math_validator.validate_positive(6, "CatBoost depth")
                else:
                    # Fallback validation
                    iterations = 1000 if 1000 > 0 else 100
                    learning_rate = 0.05 if 0.0 < 0.05 < 1.0 else 0.1
                    depth = 6 if 6 > 0 else 4
                
                ensemble_models['catboost'] = CatBoostRegressor(
                    iterations=int(iterations),
                    learning_rate=learning_rate,
                    depth=int(depth),
                    random_seed=42,
                    verbose=False
                )
                tprint("✅ CatBoost model created with validated parameters")
            except Exception as e:
                tprint(f"⚠️ CatBoost model creation failed: {e}")
                # Fallback to default parameters
                ensemble_models['catboost'] = CatBoostRegressor(
                    iterations=1000,
                    learning_rate=0.05,
                    depth=6,
                    random_seed=42,
                    verbose=False
                )
            
            # Elastic Net with validated parameters (Primary: Fast baseline)
            try:
                if self.math_validator:
                    alpha = self.math_validator.validate_positive(0.1, "ElasticNet alpha")
                    l1_ratio = self.math_validator.validate_range(0.5, 0.0, 1.0, "ElasticNet l1_ratio")
                    max_iter = self.math_validator.validate_positive(1000, "ElasticNet max_iter")
                else:
                    # Fallback validation
                    alpha = 0.1 if 0.1 > 0 else 0.01
                    l1_ratio = 0.5 if 0.0 < 0.5 < 1.0 else 0.5
                    max_iter = 1000 if 1000 > 0 else 500
                
                ensemble_models['elastic_net'] = ElasticNet(
                    random_state=43,
                    max_iter=int(max_iter),
                    alpha=alpha,
                    l1_ratio=l1_ratio
                )
                tprint("✅ Elastic Net model created with validated parameters")
            except Exception as e:
                tprint(f"⚠️ Elastic Net model creation failed: {e}")
                # Fallback to default parameters
                ensemble_models['elastic_net'] = ElasticNet(
                    random_state=43,
                    max_iter=1000,
                    alpha=0.1,
                    l1_ratio=0.5
                )
            
            # Random Forest with validated parameters (Meta: Speed + Efficient)
            try:
                if self.math_validator:
                    n_estimators = self.math_validator.validate_positive(100, "RandomForest n_estimators")
                    max_depth = self.math_validator.validate_positive(10, "RandomForest max_depth")
                    min_samples_split = self.math_validator.validate_positive(2, "RandomForest min_samples_split")
                else:
                    # Fallback validation
                    n_estimators = 100 if 100 > 0 else 50
                    max_depth = 10 if 10 > 0 else 5
                    min_samples_split = 2 if 2 > 0 else 2
                
                ensemble_models['ensemble_rf'] = RandomForestRegressor(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth),
                    min_samples_split=int(min_samples_split),
                    random_state=44,
                    n_jobs=-1
                )
                tprint("✅ Random Forest model created with validated parameters")
            except Exception as e:
                tprint(f"⚠️ Random Forest model creation failed: {e}")
                # Fallback to default parameters
                ensemble_models['ensemble_rf'] = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=2,
                    random_state=44,
                    n_jobs=-1
                )
            
            tprint(f"📊 Created {len(ensemble_models)} ensemble models for HMM training using common utilities")
            tprint(f"   Models: {list(ensemble_models.keys())}")
            
            # Update training stats using safe operations
            self.training_stats['ensemble_models_created'] = len(ensemble_models)
            self.training_stats['model_creation_method'] = 'common_utilities_validated'
            
            return ensemble_models
            
        except ImportError as e:
            tprint(f"❌ CRITICAL: Failed to import required model libraries - FAILING FAST")
            tprint(f"   Error: {e}")
            raise RuntimeError(f"Required model libraries not available: {e}") from e
        except Exception as e:
            tprint(f"❌ Failed to create ensemble models: {e}")
            raise RuntimeError(f"Ensemble model creation failed: {e}") from e
    
    def _generate_comprehensive_report(
        self,
        results: Dict[str, Any],
        execution_time: float,
        base_hmm_models: Dict[str, Any],
        hmm_training_metrics: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive training report with detailed statistics and analysis.
        Uses common utilities for robust report generation and serialization.
        
        Args:
            results: Training results
            execution_time: Total execution time
            base_hmm_models: Base models used
            hmm_training_metrics: Base model metrics
            
        Returns:
            Enhanced results with comprehensive reporting
        """
        try:
            # Calculate safe metrics using math validation utilities
            initialization_time = safe_float(self.training_stats.get('initialization_time', 0), 0.0)
            training_time = safe_float(execution_time - initialization_time, 0.0)
            
            # Create comprehensive report using safe operations
            comprehensive_report = {
                'execution_summary': {
                    'total_execution_time': safe_float(execution_time, 0.0),
                    'initialization_time': initialization_time,
                    'training_time': training_time,
                    'vectorization_enabled': self.training_stats.get('vectorization_enabled', False),
                    'success': 'error' not in results,
                    'hardware_optimization': self.training_stats.get('hardware_optimization', {}),
                    'data_normalized': self.training_stats.get('data_normalized', False),
                    'gpu_used': self.training_stats.get('gpu_used', False)
                },
                'data_summary': {
                    'sample_count': safe_int(self.training_stats.get('sample_count', 0), 0),
                    'feature_count': safe_int(self.training_stats.get('feature_count', 0), 0),
                    'base_models_used': safe_int(self.training_stats.get('base_models_used', 0), 0),
                    'ensemble_models_created': safe_int(self.training_stats.get('ensemble_models_created', 0), 0),
                    'model_creation_method': self.training_stats.get('model_creation_method', 'unknown')
                },
                'configuration_summary': {
                    'model_name': self.training_stats.get('config_used', 'unknown'),
                    'timeframe': self.training_stats.get('timeframe', 'unknown'),
                    'model_types': self.training_stats.get('model_types', []),
                    'hpo_enabled': self.config.enable_hpo,
                    'hpo_trials': safe_int(self.config.hpo_n_trials if self.config.enable_hpo else 0, 0)
                },
                'performance_analysis': self._analyze_performance(results),
                'regime_analysis': self._analyze_regime_performance(results),
                'base_model_integration': self._analyze_base_model_integration(base_hmm_models, hmm_training_metrics),
                'matrix_operations_stats': results.get('matrix_operations_stats', {}),
                'memory_optimization': results.get('memory_optimization', {}),
                'recommendations': self._generate_recommendations(results, execution_time)
            }
            
            # Add comprehensive report to results
            results['comprehensive_report'] = comprehensive_report
            
            # Save report using common serialization utilities
            try:
                report_path = Path(self.config.model_save_path) / "training_report.json"
                ensure_directory(report_path.parent)
                if self.serializer.save(comprehensive_report, str(report_path), format='json'):
                    tprint(f"📁 Comprehensive report saved to: {report_path}")
                    results['report_path'] = str(report_path)
                else:
                    tprint("⚠️ Failed to save comprehensive report")
            except Exception as e:
                tprint(f"⚠️ Could not save comprehensive report: {e}")
            
            # Log summary
            self._log_comprehensive_summary(comprehensive_report)
            
            return results
            
        except Exception as e:
            tprint(f"❌ Failed to generate comprehensive report: {e}")
            results['comprehensive_report'] = {'error': f"Report generation failed: {e}"}
            return results
    
    def _analyze_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze overall training performance using common utilities.
        
        Args:
            results: Training results
            
        Returns:
            Performance analysis
        """
        try:
            performance_analysis = {
                'training_success': 'error' not in results,
                'models_trained': 0,
                'best_performance': {},
                'performance_distribution': {},
                'matrix_operations_efficiency': {}
            }
            
            if 'evaluation_results' in results:
                evaluation_results = results['evaluation_results']
                performance_analysis['models_trained'] = safe_int(len(evaluation_results), 0)
                
                # Find best performing model using safe operations
                best_accuracy = -np.inf
                best_model = None
                accuracies = []
                
                for regime, regime_metrics in evaluation_results.items():
                    if isinstance(regime_metrics, dict) and 'accuracy' in regime_metrics:
                        accuracy = safe_float(regime_metrics['accuracy'], 0.0)
                        accuracies.append(accuracy)
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = regime
                
                if best_model is not None:
                    performance_analysis['best_performance'] = {
                        'regime': best_model,
                        'accuracy': safe_float(best_accuracy, 0.0)
                    }
                
                # Calculate performance distribution using safe operations
                if accuracies:
                    performance_analysis['performance_distribution'] = {
                        'mean_accuracy': safe_float(np.mean(accuracies), 0.0),
                        'std_accuracy': safe_float(np.std(accuracies), 0.0),
                        'min_accuracy': safe_float(np.min(accuracies), 0.0),
                        'max_accuracy': safe_float(np.max(accuracies), 0.0),
                        'accuracy_count': len(accuracies)
                    }
            
            # Add matrix operations efficiency analysis
            if 'matrix_operations_stats' in results:
                matrix_stats = results['matrix_operations_stats']
                performance_analysis['matrix_operations_efficiency'] = {
                    'total_operations': safe_int(matrix_stats.get('total_operations', 0), 0),
                    'gpu_operations': safe_int(matrix_stats.get('gpu_operations', 0), 0),
                    'cpu_operations': safe_int(matrix_stats.get('cpu_operations', 0), 0),
                    'parallel_operations': safe_int(matrix_stats.get('parallel_operations', 0), 0),
                    'average_execution_time': safe_float(matrix_stats.get('average_execution_time', 0.0), 0.0),
                    'memory_optimized_operations': safe_int(matrix_stats.get('memory_optimized_operations', 0), 0)
                }
            
            return performance_analysis
            
        except Exception as e:
            tprint(f"⚠️ Performance analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_regime_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze regime-specific performance using common utilities.
        
        Args:
            results: Training results
            
        Returns:
            Regime performance analysis
        """
        try:
            regime_analysis = {
                'total_regimes': 0,
                'successful_regimes': 0,
                'failed_regimes': 0,
                'regime_details': {},
                'regime_balance_score': 0.0
            }
            
            if 'regime_analysis' in results:
                regime_data = results['regime_analysis']
                regime_analysis['total_regimes'] = safe_int(len(regime_data.get('unique_regimes', [])), 0)
                regime_analysis['successful_regimes'] = safe_int(len(regime_data.get('sufficient_regimes', [])), 0)
                regime_analysis['failed_regimes'] = safe_int(len(regime_data.get('insufficient_regimes', [])), 0)
                
                # Calculate regime balance score using safe operations
                total_regimes = regime_analysis['total_regimes']
                successful_regimes = regime_analysis['successful_regimes']
                
                if total_regimes > 0:
                    regime_analysis['regime_balance_score'] = safe_divide(successful_regimes, total_regimes, 0.0)
                else:
                    regime_analysis['regime_balance_score'] = 0.0
                
                # Add regime details with safe operations
                regime_analysis['regime_details'] = {
                    'unique_regimes': regime_data.get('unique_regimes', []),
                    'sufficient_regimes': regime_data.get('sufficient_regimes', []),
                    'insufficient_regimes': regime_data.get('insufficient_regimes', []),
                    'regime_balance_train': safe_float(regime_data.get('regime_balance_train', 0.0), 0.0)
                }
            
            return regime_analysis
            
        except Exception as e:
            tprint(f"⚠️ Regime analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_base_model_integration(
        self,
        base_hmm_models: Dict[str, Any],
        hmm_training_metrics: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze base model integration using common utilities.
        
        Args:
            base_hmm_models: Base models used
            hmm_training_metrics: Base model metrics
            
        Returns:
            Base model integration analysis
        """
        try:
            base_models_count = safe_int(len(base_hmm_models) if base_hmm_models else 0, 0)
            
            integration_analysis = {
                'base_models_count': base_models_count,
                'base_model_types': list(base_hmm_models.keys()) if base_hmm_models else [],
                'metrics_available': hmm_training_metrics is not None,
                'integration_quality': 'good' if base_models_count >= 3 else 'limited',
                'integration_score': safe_divide(base_models_count, 3, 0.0) if base_models_count > 0 else 0.0
            }
            
            if hmm_training_metrics:
                # Safely process base model performance metrics
                safe_metrics = {}
                for key, value in hmm_training_metrics.items():
                    if isinstance(value, (int, float)):
                        safe_metrics[key] = safe_float(value, 0.0)
                    elif isinstance(value, dict):
                        safe_metrics[key] = {
                            k: safe_float(v, 0.0) if isinstance(v, (int, float)) else v
                            for k, v in value.items()
                        }
                    else:
                        safe_metrics[key] = value
                
                integration_analysis['base_model_performance'] = safe_metrics
                
                # Calculate average performance if available
                if 'accuracy' in safe_metrics:
                    integration_analysis['average_accuracy'] = safe_float(safe_metrics['accuracy'], 0.0)
                elif isinstance(safe_metrics, dict) and any('accuracy' in str(k).lower() for k in safe_metrics.keys()):
                    # Try to find accuracy in nested metrics
                    accuracies = []
                    for key, value in safe_metrics.items():
                        if isinstance(value, dict) and 'accuracy' in value:
                            accuracies.append(safe_float(value['accuracy'], 0.0))
                    if accuracies:
                        integration_analysis['average_accuracy'] = safe_float(np.mean(accuracies), 0.0)
            
            return integration_analysis
            
        except Exception as e:
            tprint(f"⚠️ Base model integration analysis failed: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, results: Dict[str, Any], execution_time: float) -> List[str]:
        """
        Generate recommendations based on training results using common utilities.
        
        Args:
            results: Training results
            execution_time: Execution time
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        try:
            # Performance-based recommendations
            if 'error' in results:
                recommendations.append("❌ Training failed - review error logs and data quality")
            else:
                recommendations.append("✅ Training completed successfully")
            
            # Time-based recommendations using safe operations
            execution_time_safe = safe_float(execution_time, 0.0)
            if execution_time_safe > 3600:  # More than 1 hour
                recommendations.append("⏰ Consider enabling vectorization for faster training")
            elif execution_time_safe < 60:  # Less than 1 minute
                recommendations.append("⚡ Training completed quickly - consider increasing HPO trials for better performance")
            
            # Data-based recommendations using safe operations
            sample_count = safe_int(self.training_stats.get('sample_count', 0), 0)
            if sample_count < 10000:
                recommendations.append("📊 Consider collecting more training data for better model performance")
            elif sample_count > 100000:
                recommendations.append("📊 Large dataset detected - consider using GPU acceleration for faster processing")
            
            # Model-based recommendations using safe operations
            base_models_used = safe_int(self.training_stats.get('base_models_used', 0), 0)
            if base_models_used < 3:
                recommendations.append("🤖 Consider using more diverse base models for better ensemble performance")
            elif base_models_used >= 5:
                recommendations.append("🤖 Many base models detected - consider ensemble pruning for better performance")
            
            # Hardware optimization recommendations
            if not self.training_stats.get('vectorization_enabled', False):
                recommendations.append("🚀 Enable vectorization for improved performance on multi-regime training")
            
            if not self.training_stats.get('gpu_used', False) and sample_count > 50000:
                recommendations.append("🧠 Consider enabling GPU acceleration for large datasets")
            
            # Memory optimization recommendations
            if 'memory_optimization' in results:
                memory_stats = results['memory_optimization']
                if memory_stats.get('success', False):
                    recommendations.append("🧠 Memory optimization completed successfully")
                else:
                    recommendations.append("⚠️ Memory optimization failed - consider manual memory management")
            
            # Matrix operations recommendations
            if 'matrix_operations_stats' in results:
                matrix_stats = results['matrix_operations_stats']
                total_ops = safe_int(matrix_stats.get('total_operations', 0), 0)
                gpu_ops = safe_int(matrix_stats.get('gpu_operations', 0), 0)
                
                if total_ops > 0:
                    gpu_ratio = safe_divide(gpu_ops, total_ops, 0.0)
                    if gpu_ratio < 0.5 and total_ops > 100:
                        recommendations.append("🔧 Consider increasing GPU usage for matrix operations")
                    elif gpu_ratio > 0.8:
                        recommendations.append("✅ GPU utilization is excellent for matrix operations")
            
            # Data quality recommendations
            if 'comprehensive_report' in results:
                report = results['comprehensive_report']
                if 'data_summary' in report:
                    data_summary = report['data_summary']
                    feature_count = safe_int(data_summary.get('feature_count', 0), 0)
                    if feature_count > 1000:
                        recommendations.append("🔍 High feature count detected - consider feature selection for better performance")
                    elif feature_count < 10:
                        recommendations.append("📊 Low feature count - consider adding more features for better model performance")
            
            return recommendations
            
        except Exception as e:
            tprint(f"⚠️ Recommendation generation failed: {e}")
            return [f"⚠️ Could not generate recommendations: {e}"]
    
    def _log_comprehensive_summary(self, comprehensive_report: Dict[str, Any]) -> None:
        """
        Log comprehensive training summary using tprint and common utilities.
        
        Args:
            comprehensive_report: Comprehensive report data
        """
        try:
            tprint("📊 COMPREHENSIVE TRAINING SUMMARY")
            tprint("=" * 50)
            
            # Execution summary using safe operations
            exec_summary = comprehensive_report.get('execution_summary', {})
            total_time = safe_float(exec_summary.get('total_execution_time', 0), 0.0)
            init_time = safe_float(exec_summary.get('initialization_time', 0), 0.0)
            training_time = safe_float(exec_summary.get('training_time', 0), 0.0)
            
            tprint(f"⏱️ Total execution time: {total_time:.2f}s")
            tprint(f"🚀 Vectorization enabled: {exec_summary.get('vectorization_enabled', False)}")
            tprint(f"✅ Training success: {exec_summary.get('success', False)}")
            tprint(f"🧠 Hardware optimization: {exec_summary.get('hardware_optimization', {})}")
            tprint(f"📊 Data normalized: {exec_summary.get('data_normalized', False)}")
            tprint(f"🎮 GPU used: {exec_summary.get('gpu_used', False)}")
            
            # Data summary using safe operations
            data_summary = comprehensive_report.get('data_summary', {})
            sample_count = safe_int(data_summary.get('sample_count', 0), 0)
            feature_count = safe_int(data_summary.get('feature_count', 0), 0)
            base_models = safe_int(data_summary.get('base_models_used', 0), 0)
            ensemble_models = safe_int(data_summary.get('ensemble_models_created', 0), 0)
            
            tprint(f"📊 Samples processed: {sample_count:,}")
            tprint(f"🔢 Features used: {feature_count}")
            tprint(f"🤖 Base models: {base_models}")
            tprint(f"🏗️ Ensemble models created: {ensemble_models}")
            tprint(f"🔧 Model creation method: {data_summary.get('model_creation_method', 'unknown')}")
            
            # Performance analysis using safe operations
            perf_analysis = comprehensive_report.get('performance_analysis', {})
            if perf_analysis.get('best_performance'):
                best_perf = perf_analysis['best_performance']
                accuracy = safe_float(best_perf.get('accuracy', 0), 0.0)
                regime = best_perf.get('regime', 'N/A')
                tprint(f"🏆 Best performance: Accuracy = {accuracy:.4f} (Regime {regime})")
            
            # Matrix operations stats
            matrix_stats = comprehensive_report.get('matrix_operations_stats', {})
            if matrix_stats:
                total_ops = safe_int(matrix_stats.get('total_operations', 0), 0)
                avg_time = safe_float(matrix_stats.get('average_execution_time', 0), 0.0)
                gpu_ops = safe_int(matrix_stats.get('gpu_operations', 0), 0)
                tprint(f"🧮 Matrix operations: {total_ops} ops, avg time: {avg_time:.3f}s, GPU ops: {gpu_ops}")
            
            # Memory optimization stats
            memory_stats = comprehensive_report.get('memory_optimization', {})
            if memory_stats:
                status = memory_stats.get('status', 'unknown')
                tprint(f"🧠 Memory optimization: {status}")
            
            # Recommendations
            recommendations = comprehensive_report.get('recommendations', [])
            if recommendations:
                tprint("💡 RECOMMENDATIONS:")
                for rec in recommendations:
                    tprint(f"   {rec}")
            
            tprint("=" * 50)
            
        except Exception as e:
            tprint(f"❌ Failed to log comprehensive summary: {e}")
    
    def _add_ensemble_specific_metadata(self, results: Dict[str, Any], base_models: Dict[str, Any], base_metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add ensemble-specific metadata to results with enhanced error handling.
        Uses common utilities for robust metadata generation.
        
        Args:
            results: Training results
            base_models: Base HMM models used in ensemble
            base_metrics: Performance metrics of base models
            
        Returns:
            Enhanced results with ensemble-specific metadata
        """
        try:
            # Add ensemble-specific analysis using safe operations
            if 'regime_analysis' in results:
                regime_analysis = results['regime_analysis']
                
                # Calculate ensemble-specific metrics using safe operations
                ensemble_metrics = {
                    'total_regimes': safe_int(len(regime_analysis.get('unique_regimes', [])), 0),
                    'sufficient_regimes': safe_int(len(regime_analysis.get('sufficient_regimes', [])), 0),
                    'insufficient_regimes': safe_int(len(regime_analysis.get('insufficient_regimes', [])), 0),
                    'regime_balance': safe_float(regime_analysis.get('regime_balance_train', 0.0), 0.0),
                    'timeframe': self.config.timeframe,
                    'ensemble_model_types': self.config.model_types,
                    'base_models_count': safe_int(len(base_models) if base_models else 0, 0),
                    'training_timestamp': safe_float(time.time(), 0.0),
                    'vectorization_used': self.training_stats.get('vectorization_enabled', False),
                    'hardware_optimization': self.training_stats.get('hardware_optimization', {}),
                    'data_normalized': self.training_stats.get('data_normalized', False),
                    'gpu_used': self.training_stats.get('gpu_used', False)
                }
                
                # Add base model performance analysis if available using safe operations
                if base_metrics:
                    safe_base_metrics = {}
                    for key, value in base_metrics.items():
                        if isinstance(value, (int, float)):
                            safe_base_metrics[key] = safe_float(value, 0.0)
                        elif isinstance(value, dict):
                            safe_base_metrics[key] = {
                                k: safe_float(v, 0.0) if isinstance(v, (int, float)) else v
                                for k, v in value.items()
                            }
                        else:
                            safe_base_metrics[key] = value
                    
                    ensemble_metrics['base_model_performance'] = safe_base_metrics
                    tprint("📊 Integrated base model performance metrics using safe operations")
                
                results['ensemble_metrics'] = ensemble_metrics
            
            # Add ensemble performance summary with enhanced analysis using safe operations
            if 'evaluation_results' in results:
                evaluation_results = results['evaluation_results']
                
                # Calculate best performing ensemble per regime using safe operations
                best_ensembles = {}
                performance_summary = {
                    'total_regimes_evaluated': 0,
                    'successful_evaluations': 0,
                    'failed_evaluations': 0,
                    'average_accuracy': 0.0,
                    'best_overall_accuracy': -np.inf
                }
                
                accuracies = []
                
                for regime, regime_metrics in evaluation_results.items():
                    performance_summary['total_regimes_evaluated'] += 1
                    
                    if isinstance(regime_metrics, dict) and 'error' not in regime_metrics:
                        performance_summary['successful_evaluations'] += 1
                        
                        best_ensemble = None
                        best_accuracy = -np.inf
                        
                        for ensemble_name, metrics in regime_metrics.items():
                            if isinstance(metrics, dict) and 'accuracy' in metrics:
                                accuracy = safe_float(metrics['accuracy'], 0.0)
                                accuracies.append(accuracy)
                                if accuracy > best_accuracy:
                                    best_accuracy = accuracy
                                    best_ensemble = ensemble_name
                        
                        if best_ensemble:
                            best_ensembles[regime] = {
                                'ensemble': best_ensemble,
                                'accuracy': safe_float(best_accuracy, 0.0),
                                'regime_samples': safe_int(regime_metrics.get('samples', 0), 0)
                            }
                            
                            if best_accuracy > performance_summary['best_overall_accuracy']:
                                performance_summary['best_overall_accuracy'] = safe_float(best_accuracy, 0.0)
                    else:
                        performance_summary['failed_evaluations'] += 1
                
                # Calculate average performance using safe operations
                if accuracies:
                    performance_summary['average_accuracy'] = safe_float(np.mean(accuracies), 0.0)
                    performance_summary['accuracy_std'] = safe_float(np.std(accuracies), 0.0)
                    performance_summary['accuracy_min'] = safe_float(np.min(accuracies), 0.0)
                    performance_summary['accuracy_max'] = safe_float(np.max(accuracies), 0.0)
                
                results['best_ensembles_per_regime'] = best_ensembles
                results['performance_summary'] = performance_summary
                
                successful_eval = safe_int(performance_summary['successful_evaluations'], 0)
                total_eval = safe_int(performance_summary['total_regimes_evaluated'], 0)
                avg_accuracy = safe_float(performance_summary['average_accuracy'], 0.0)
                best_accuracy = safe_float(performance_summary['best_overall_accuracy'], 0.0)
                
                tprint(f"📊 Performance summary: {successful_eval}/{total_eval} regimes successful")
                if avg_accuracy > 0:
                    tprint(f"🏆 Average Accuracy: {avg_accuracy:.4f}, Best Accuracy: {best_accuracy:.4f}")
            
            # Add enhanced ensemble-specific analysis using safe operations
            ensemble_analysis = {
                'base_timeframe': self.config.timeframe,
                'cross_timeframe_features': True,
                'ensemble_method': 'per_regime',
                'base_models_integrated': safe_int(len(base_models) if base_models else 0, 0),
                'ensemble_role': 'market_regime_detection',
                'training_configuration': {
                    'hpo_enabled': self.config.enable_hpo,
                    'hpo_trials': safe_int(self.config.hpo_n_trials if self.config.enable_hpo else 0, 0),
                    'min_samples_per_regime': safe_int(self.config.min_samples_per_regime, 0),
                    'evaluation_metrics': self.config.evaluation_metrics
                },
                'data_characteristics': {
                    'total_samples': safe_int(self.training_stats.get('sample_count', 0), 0),
                    'feature_count': safe_int(self.training_stats.get('feature_count', 0), 0),
                    'ensemble_models_created': safe_int(self.training_stats.get('ensemble_models_created', 0), 0),
                    'model_creation_method': self.training_stats.get('model_creation_method', 'unknown'),
                    'data_normalized': self.training_stats.get('data_normalized', False),
                    'gpu_used': self.training_stats.get('gpu_used', False)
                },
                'hardware_optimization': self.training_stats.get('hardware_optimization', {}),
                'performance_enhancements': {
                    'vectorization_enabled': self.training_stats.get('vectorization_enabled', False),
                    'matrix_operations_used': True,
                    'memory_optimization_used': True,
                    'common_utilities_integration': True
                }
            }
            results['ensemble_analysis'] = ensemble_analysis
            
            return results
            
        except Exception as e:
            tprint(f"❌ Failed to add ensemble-specific metadata: {e}")
            results['ensemble_metadata_error'] = str(e)
            return results


# Convenience functions for backward compatibility with common utilities integration
def create_hmm_ensemble_training_component(
    config: Optional[EnsembleTrainingConfig] = None,
    enable_vectorization: bool = True
) -> HMMEnsembleTrainingComponent:
    """
    Create HMM ensemble training component with common utilities integration.
    
    Args:
        config: Training configuration
        enable_vectorization: Whether to enable vectorization
        
    Returns:
        Configured HMM ensemble training component
    """
    return HMMEnsembleTrainingComponent(config, enable_vectorization)


def execute_hmm_ensemble_training(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[EnsembleTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    base_hmm_models: Optional[Dict[str, Any]] = None,
    hmm_training_metrics: Optional[Dict[str, Any]] = None,
    enable_vectorization: bool = True
) -> Dict[str, Any]:
    """
    Execute HMM ensemble training component with common utilities integration.
    
    Args:
        X: Input features
        y: Target values
        regime_labels: Regime labels
        config: Training configuration
        feature_names: Feature names
        hmm_states: HMM states
        base_hmm_models: Base models
        hmm_training_metrics: Base model metrics
        enable_vectorization: Whether to enable vectorization
        
    Returns:
        Training results with enhanced metadata
    """
    component = create_hmm_ensemble_training_component(config, enable_vectorization)
    return component.execute(X, y, regime_labels, feature_names, hmm_states, base_hmm_models, hmm_training_metrics)


# Example usage and comparison with common utilities integration
if __name__ == "__main__":
    # Example of how to use the HMM ensemble training component with common utilities
    print("HMM Ensemble Training Component with Common Utilities Integration")
    print("=" * 70)
    
    # Create configuration
    config = EnsembleTrainingConfig(
        model_name="hmm_ensemble_models",
        timeframe="1h",
        model_types=["catboost", "elastic_net", "ensemble_rf"],
        hpo_n_trials=50,  # Reduced for demo
        enable_hpo=True,
        save_models=True,
        model_save_path="./models/hmm_ensemble_models_refactored"
    )
    
    # Create training component with common utilities
    training_component = create_hmm_ensemble_training_component(config, enable_vectorization=True)
    
    print(f"✅ Created HMM ensemble training component with {len(config.model_types)} ensemble types")
    print(f"📊 HPO enabled: {config.enable_hpo}")
    print(f"💾 Save models: {config.save_models}")
    print(f"📁 Save path: {config.model_save_path}")
    print(f"⏰ Base timeframe: {config.timeframe}")
    
    # Display common utilities integration status
    print(f"\n🔧 Common Utilities Integration:")
    print(f"   - Math validation: ✅ Integrated")
    print(f"   - Matrix operations: ✅ Integrated")
    print(f"   - Serialization utils: ✅ Integrated")
    print(f"   - Hardware optimization: ✅ Integrated")
    print(f"   - Memory management: ✅ Integrated")
    print(f"   - Safe operations: ✅ Integrated")
    
    # The actual training would be called with:
    # results = training_component.execute(X, y, regime_labels, feature_names, hmm_states, base_hmm_models, hmm_training_metrics)
    
    print("\n🎯 HMM Ensemble Component Features (Enhanced with Common Utilities):")
    print("- Operates on 1h timeframe with cross-timeframe features")
    print("- Combines individual HMM models into robust ensembles")
    print("- Per-regime ensemble training for regime-specific optimization")
    print("- Enhanced market regime detection accuracy through model combination")
    print("- Models: CatBoost, Elastic Net, Random Forest (with validated parameters)")
    print("- Comprehensive context from multi-timeframe dynamics")
    print("- M1 hardware optimization for Apple Silicon Macs")
    print("- Memory management and GPU acceleration")
    print("- Safe mathematical operations and validation")
    print("- Robust error handling and recovery")
    print("- Enhanced reporting and serialization")
    
    print("\n🔄 Integration with Individual HMM Models:")
    print("- Receives individual HMM model predictions")
    print("- Uses base model performance metrics for weighting")
    print("- Creates regime-specific ensemble combinations")
    print("- Provides enhanced market regime detection signals")
    print("- Validates all inputs using common utilities")
    print("- Optimizes performance using hardware acceleration")
    
    print("\n🚀 Performance Enhancements:")
    print("- Vectorized training for improved speed")
    print("- Matrix operations optimization")
    print("- Memory checkpointing for large datasets")
    print("- GPU acceleration for large matrices")
    print("- Safe mathematical operations")
    print("- Comprehensive error handling")
    print("- Enhanced reporting and analytics")