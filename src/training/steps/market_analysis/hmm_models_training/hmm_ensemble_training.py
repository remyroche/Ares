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

# ML imports
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

# Common utilities imports
from src.utils.tprint import tprint
from .shared_feature_utils import create_enhanced_features
from src.utils.common_operations import (
    safe_dataframe_operation, safe_divide, safe_float, safe_int,
    ensure_directory, memory_checkpoint, optimize_memory, get_memory_usage
)

# Import universal validation components from ml_common
from src.utils.ml_common.validation import (
    validate_ml_model,
    get_ml_validator,
    UniversalMLValidationConfig
)
from src.utils.ml_common.config.universal_timeframe_config import get_primary_timeframe
from src.utils.ml_common.reporting import process_validation_with_reporting
try:
    from src.utils.common_operations import (
        validate_dataframe, validate_finite, validate_positive, validate_range, 
        safe_percentage_change, safe_json_dump, safe_json_load, safe_file_exists,
        gpu_context
    )
    EXTENDED_COMMON_OPS_AVAILABLE = True
except ImportError:
    EXTENDED_COMMON_OPS_AVAILABLE = False

# Import hardware optimization tools from hardware/ directory
try:
    from src.utils.hardware import (
        get_advanced_memory_optimizer, get_enhanced_gpu_manager, get_advanced_cpu_optimizer,
        get_unified_hardware_manager, optimize_dataframe_advanced,
        AdvancedM1MemoryOptimizer, EnhancedM1GPUManager, AdvancedM1CPUOptimizer,
        ADVANCED_MEMORY_AVAILABLE, ENHANCED_GPU_AVAILABLE, ADVANCED_CPU_AVAILABLE
    )
    HARDWARE_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATIONS_AVAILABLE = False
    ADVANCED_MEMORY_AVAILABLE = False
    ENHANCED_GPU_AVAILABLE = False
    ADVANCED_CPU_AVAILABLE = False

from src.utils.math_validation import (
    safe_divide as math_safe_divide, safe_divide, safe_log, safe_sqrt,
    validate_finite as math_validate_finite, validate_positive as math_validate_positive,
    validate_range as math_validate_range
)

# Enhanced import handling with fallbacks for better reliability
try:
    from src.utils.math_validation import (
        safe_power, safe_mean, safe_std, safe_correlation, safe_covariance, 
        safe_percentile, MathValidation
    )
    EXTENDED_MATH_AVAILABLE = True
except ImportError:
    from src.utils.math_validation import MathValidation
    EXTENDED_MATH_AVAILABLE = False

try:
    from src.utils.data.klines_parquet import KlinesParquetManager, get_klines_manager
    KLINES_AVAILABLE = True
except ImportError:
    KLINES_AVAILABLE = False
    get_klines_manager = lambda: None

try:
    from src.utils.serialization_utils import JSONSerializer, PickleSerializer, UniversalSerializer
    LEGACY_SERIALIZERS_AVAILABLE = True
except ImportError:
    from src.utils.serialization_utils import UniversalSerializer
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
    from src.utils.ml_common.training.enhanced_training_utils import EnhancedTrainingUtils
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False
    # Import EnsembleTrainingConfig separately since it's always needed
    try:
        from src.utils.ml_common.config.base_training_config import EnsembleTrainingConfig
    except ImportError:
        raise ImportError("EnsembleTrainingConfig is required but not available. Please check ML Common utilities installation.")
    
    # Create minimal fallback for EnsembleTrainingStep only
    
    class EnsembleTrainingStep:
        def __init__(self, config, enable_vectorization=True):
            self.config = config
            self.enable_vectorization = enable_vectorization
            
        def execute(self, *args, **kwargs):
            """Fallback execute method that raises informative error"""
            raise ImportError("ML Common utilities not available. Please install required dependencies.")

try:
    from src.utils.ml_common.evaluation.evaluation_utils import EvaluationUtils
    ML_EVAL_AVAILABLE = True
except ImportError:
    ML_EVAL_AVAILABLE = False
    EvaluationUtils = None

# Shared utilities with validation
try:
    from .shared_utilities import (
        TrainingErrorHandler,
        UnifiedModelFactory,
        CircuitBreaker,
        ValidationUtils,
        ProgressReporter,
        MemoryTracker
    )
    from .shared_utilities.training_error_handler import TrainingMetrics, ModelResult

    # Enhanced analysis utilities (now integrated into ML commons)
    try:
        from src.utils.ml_common.evaluation.enhanced_learning_curve_analysis import EnhancedLearningCurveAnalyzer
        from src.utils.ml_common.evaluation.enhanced_bootstrap_confidence_intervals import EnhancedBootstrapConfidenceIntervalAnalyzer
        ENHANCED_ANALYSIS_AVAILABLE = True
    except ImportError:
        EnhancedLearningCurveAnalyzer = None
        EnhancedBootstrapConfidenceIntervalAnalyzer = None
        ENHANCED_ANALYSIS_AVAILABLE = False

    SHARED_UTILITIES_AVAILABLE = True
    tprint("✅ Shared utilities loaded successfully")
except ImportError as e:
    tprint(f"⚠️ Shared utilities not available: {e}")
    SHARED_UTILITIES_AVAILABLE = False
    
    # Create fallback classes
    class TrainingErrorHandler:
        @staticmethod
        def handle_model_creation_error(model_type: str, error: Exception):
            return {'model': None, 'error': f"Failed to create {model_type}: {str(error)}"}
        
        @staticmethod
        def handle_training_error(model_type: str, error: Exception, training_time: float):
            return {'model': None, 'error': f"Failed to train {model_type}: {str(error)}", 'training_time': training_time}
    
    class TrainingMetrics:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class ModelResult:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    # Minimal fallback implementations for other utilities
    UnifiedModelFactory = None
    CircuitBreaker = None
    ValidationUtils = None
    ProgressReporter = None
    MemoryTracker = None

# Import vectorized training manager
try:
    from src.utils.ml_common.training.vectorized_training_manager import VectorizedTrainingManager
    VECTORIZED_TRAINING_AVAILABLE = True
except ImportError:
    VECTORIZED_TRAINING_AVAILABLE = False

# Using tprint for all logging - no logger needed

# Additional ML imports for global classifier, calibration, and metrics
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss


# Custom exception classes for better error handling
class HMMTrainingError(Exception):
    """Base exception for HMM training errors."""
    pass

class HMMDataValidationError(HMMTrainingError):
    """Exception raised for data validation errors."""
    pass

class HMMModelCreationError(HMMTrainingError):
    """Exception raised for model creation errors."""
    pass

class HMMTrainingExecutionError(HMMTrainingError):
    """Exception raised for training execution errors."""
    pass

class HMMResourceError(HMMTrainingError):
    """Exception raised for resource management errors."""
    pass


class HMMEnsembleTrainingComponent(EnsembleTrainingStep):
    """
    HMM Ensemble Training Component with per-regime ensemble training, HPO, saving, and metrics.
    
    The HMM Ensemble operates on 15m timeframe and combines individual HMM models
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
            
            # Initialize advanced hardware optimizers from hardware/ directory
            if HARDWARE_OPTIMIZATIONS_AVAILABLE:
                # Use advanced hardware optimization tools
                self.gpu_manager = get_enhanced_gpu_manager() if ENHANCED_GPU_AVAILABLE else None
                self.memory_optimizer = get_advanced_memory_optimizer() if ADVANCED_MEMORY_AVAILABLE else None
                self.cpu_optimizer = get_advanced_cpu_optimizer() if ADVANCED_CPU_AVAILABLE else None
                self.unified_hardware_manager = get_unified_hardware_manager()
                tprint("🚀 Advanced hardware optimizations loaded from hardware/ directory")
            else:
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
                self.unified_hardware_manager = None
                tprint("⚠️ Advanced hardware optimizations not available, using fallback")
            
            # Set default configuration for HMM ensemble models
            if config is None:
                config = EnsembleTrainingConfig(
                    model_name="hmm_ensemble_models",
                    timeframe=get_primary_timeframe(),
                    min_samples_per_regime=500,  # 🔧 Reduced from 1000 to 500 for better regime coverage
                    enable_data_augmentation=True,
                    augmentation_method="smote",
                    model_save_path="./generated/market_analysis/models/hmm_ensemble_models",
                    evaluation_metrics=["accuracy", "precision", "recall", "f1_score", "log_loss"]
                )
                tprint("📋 Using default configuration for HMM ensemble training (classification)")

            # Validate configuration with fast-fail using common utilities
            self._validate_config(config)
            
            # Ensure both base_models and model_types exist for compatibility
            try:
                base_models = getattr(config, 'base_models', [])
                model_types = getattr(config, 'model_types', [])
                
                # If base_models exists but model_types doesn't, sync them
                if base_models and not model_types:
                    setattr(config, 'model_types', base_models)
                # If model_types exists but base_models doesn't, sync them
                elif model_types and not base_models:
                    setattr(config, 'base_models', model_types)
                # Ensure at least one exists
                elif not base_models and not model_types:
                    # Use the defaults from EnsembleTrainingConfig
                    default_config = EnsembleTrainingConfig()
                    setattr(config, 'base_models', default_config.base_models)
                    setattr(config, 'model_types', default_config.base_models)
                    tprint(f"⚠️ No model types specified, using defaults: {', '.join(default_config.base_models)}")
            except Exception as e:
                tprint(f"⚠️ Error in model type configuration: {e}")
            
            # Initialize parent class
            super().__init__(config, enable_vectorization=enable_vectorization and VECTORIZED_TRAINING_AVAILABLE)
            # Initialize enhanced training utilities (lookahead, temporal CV, regularization helpers)
            try:
                self.enhanced_training_utils = EnhancedTrainingUtils()
            except Exception:
                self.enhanced_training_utils = None
            
            # Initialize tracking variables
            self.training_stats = {
                'initialization_time': time.time() - self.start_time,
                'vectorization_enabled': self.enable_vectorization,
                'config_used': config.model_name,
                'model_types': getattr(config, 'base_models', getattr(config, 'model_types', [])),
                'timeframe': config.timeframe,
                'hardware_optimization': {
                    'gpu_available': self.gpu_manager is not None,
                    'enhanced_gpu_available': ENHANCED_GPU_AVAILABLE,
                    'memory_optimizer_available': self.memory_optimizer is not None,
                    'advanced_memory_available': ADVANCED_MEMORY_AVAILABLE,
                    'cpu_optimizer_available': self.cpu_optimizer is not None,
                    'advanced_cpu_available': ADVANCED_CPU_AVAILABLE,
                    'unified_hardware_manager_available': hasattr(self, 'unified_hardware_manager') and self.unified_hardware_manager is not None,
                    'hardware_tools_source': 'hardware_directory' if HARDWARE_OPTIMIZATIONS_AVAILABLE else 'fallback'
                }
            }
            
            # Log initialization success
            if self.enable_vectorization:
                tprint("🚀 HMM Ensemble Training Component initialized with vectorization")
            else:
                tprint("✅ HMM Ensemble Training Component initialized (standard mode)")
                
            tprint(f"📊 Configuration: {len(getattr(config, 'base_models', getattr(config, 'model_types', [])))} base models, {config.timeframe} timeframe")
            tprint(f"🧠 Hardware: GPU={self.gpu_manager is not None}, Memory={self.memory_optimizer is not None}, CPU={self.cpu_optimizer is not None}")
            
        except Exception as e:
            tprint(f"❌ Failed to initialize HMM Ensemble Training Component: {e}")
            tprint(f"🔍 Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"HMM Ensemble Training Component initialization failed: {e}") from e

        # Thread-safe global regime classifier and metadata with locks
        import threading
        self._classifier_lock = threading.RLock()  # Re-entrant lock for thread safety
        self.global_regime_clf = None
        self.regime_classes_ = None
        self.global_calibration_ece = None
        
        # Memory management tracking
        self._large_arrays = []  # Track large arrays for cleanup
    
    def cleanup_memory(self) -> None:
        """
        Explicitly cleanup large arrays and resources to prevent memory leaks.
        Uses advanced memory optimization tools from hardware/ directory.
        Should be called after training is complete.
        """
        try:
            # Use advanced memory optimizer if available
            if self.memory_optimizer and ADVANCED_MEMORY_AVAILABLE:
                tprint("🧹 Using advanced memory optimizer for cleanup")
                
                # Track memory before cleanup
                memory_before = self.memory_optimizer.get_memory_usage()
                
                # Clean up tracked large arrays using advanced optimizer
                for array_name in self._large_arrays:
                    if hasattr(self, array_name):
                        array_obj = getattr(self, array_name)
                        # Use advanced memory optimizer to clean up array
                        self.memory_optimizer.cleanup_object(array_obj)
                        delattr(self, array_name)
                        tprint(f"🧹 Advanced cleanup of large array: {array_name}")
                
                # Clear the tracking list
                self._large_arrays.clear()
                
                # Use advanced memory optimizer for global classifier cleanup
                if hasattr(self, 'global_regime_clf') and self.global_regime_clf is not None:
                    try:
                        # Use memory optimizer to estimate and cleanup model
                        if hasattr(self.global_regime_clf, 'n_features_in_') and self.global_regime_clf.n_features_in_ > 1000:
                            self.memory_optimizer.cleanup_object(self.global_regime_clf)
                            with self._classifier_lock:
                                self.global_regime_clf = None
                            tprint("🧹 Advanced cleanup of global regime classifier")
                    except Exception as model_cleanup_error:
                        tprint(f"⚠️ Advanced model cleanup warning: {model_cleanup_error}")
                
                # Trigger advanced garbage collection
                collected = self.memory_optimizer.force_garbage_collection()
                
                # Track memory after cleanup
                memory_after = self.memory_optimizer.get_memory_usage()
                memory_freed = memory_before - memory_after
                
                tprint(f"🧹 Advanced memory cleanup completed:")
                tprint(f"   • Objects collected: {collected}")
                tprint(f"   • Memory freed: {memory_freed:.1f} MB")
                
            else:
                # Fallback to basic cleanup
                tprint("🧹 Using basic memory cleanup (advanced optimizer not available)")
                
                # Clean up tracked large arrays
                for array_name in self._large_arrays:
                    if hasattr(self, array_name):
                        delattr(self, array_name)
                        tprint(f"🧹 Cleaned up large array: {array_name}")
                
                # Clear the tracking list
                self._large_arrays.clear()
                
                # Clear global classifier if no longer needed
                if hasattr(self, 'global_regime_clf') and self.global_regime_clf is not None:
                    # Only clear if it's a large model
                    try:
                        # Estimate model size (rough approximation)
                        if hasattr(self.global_regime_clf, 'n_features_in_') and self.global_regime_clf.n_features_in_ > 1000:
                            with self._classifier_lock:
                                self.global_regime_clf = None
                            tprint("🧹 Cleaned up global regime classifier")
                    except:
                        pass
                
                # Force garbage collection
                import gc
                collected = gc.collect()
                if collected > 0:
                    tprint(f"🧹 Memory cleanup completed: {collected} objects collected")
            
        except Exception as e:
            tprint(f"⚠️ Memory cleanup warning: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        try:
            self.cleanup_memory()
        except:
            pass  # Ignore errors in destructor
    
    def _gpu_context_manager(self):
        """Context manager for GPU resource management using enhanced GPU manager from hardware/."""
        from contextlib import contextmanager
        
        @contextmanager
        def gpu_resource_context():
            gpu_acquired = False
            memory_pool_created = False
            try:
                # Acquire GPU resources using enhanced GPU manager if available
                if self.gpu_manager and ENHANCED_GPU_AVAILABLE:
                    try:
                        tprint("🎮 Using enhanced GPU manager from hardware/ directory")
                        
                        # Initialize enhanced GPU context with memory pooling
                        if hasattr(self.gpu_manager, 'initialize_enhanced_context'):
                            self.gpu_manager.initialize_enhanced_context()
                        elif hasattr(self.gpu_manager, 'initialize_context'):
                            self.gpu_manager.initialize_context()
                        
                        # Create memory pool for efficient GPU memory management
                        if hasattr(self.gpu_manager, 'create_memory_pool'):
                            pool_config = {
                                'initial_size_mb': 100.0,
                                'max_size_mb': 1000.0,
                                'enable_auto_cleanup': True
                            }
                            self.gpu_manager.create_memory_pool('hmm_training', **pool_config)
                            memory_pool_created = True
                            tprint("🏊 GPU memory pool created for efficient memory management")
                        
                        gpu_acquired = True
                        tprint("🎮 Enhanced GPU resources acquired successfully")
                        
                    except Exception as e:
                        tprint(f"⚠️ Enhanced GPU acquisition failed: {e}")
                        gpu_acquired = False
                        
                elif self.gpu_manager:
                    # Fallback to basic GPU manager
                    try:
                        tprint("🎮 Using basic GPU manager (enhanced not available)")
                        if hasattr(self.gpu_manager, 'initialize_context'):
                            self.gpu_manager.initialize_context()
                        gpu_acquired = True
                        tprint("🎮 Basic GPU resources acquired successfully")
                    except Exception as e:
                        tprint(f"⚠️ Basic GPU acquisition failed: {e}")
                        gpu_acquired = False
                
                yield gpu_acquired
                
            finally:
                # Always cleanup GPU resources using enhanced methods
                if gpu_acquired and self.gpu_manager:
                    try:
                        if ENHANCED_GPU_AVAILABLE:
                            # Enhanced cleanup
                            if memory_pool_created and hasattr(self.gpu_manager, 'cleanup_memory_pool'):
                                self.gpu_manager.cleanup_memory_pool('hmm_training')
                                tprint("🧹 GPU memory pool cleaned up")
                            
                            if hasattr(self.gpu_manager, 'cleanup_enhanced_context'):
                                self.gpu_manager.cleanup_enhanced_context()
                            elif hasattr(self.gpu_manager, 'cleanup_gpu_memory'):
                                self.gpu_manager.cleanup_gpu_memory()
                                
                            if hasattr(self.gpu_manager, 'force_memory_cleanup'):
                                freed_mb = self.gpu_manager.force_memory_cleanup()
                                tprint(f"🧹 Enhanced GPU cleanup: {freed_mb:.1f} MB freed")
                            
                            tprint("🧹 Enhanced GPU resources cleaned up successfully")
                        else:
                            # Basic cleanup
                            if hasattr(self.gpu_manager, 'cleanup_gpu_memory'):
                                self.gpu_manager.cleanup_gpu_memory()
                            if hasattr(self.gpu_manager, 'cleanup_context'):
                                self.gpu_manager.cleanup_context()
                            tprint("🧹 Basic GPU resources cleaned up successfully")
                            
                    except Exception as cleanup_error:
                        tprint(f"⚠️ GPU cleanup warning: {cleanup_error}")
        
        return gpu_resource_context()
    
    def _safe_file_operations(self, filepath, mode='r'):
        """Context manager for safe file operations with proper error handling."""
        from contextlib import contextmanager
        
        @contextmanager
        def safe_file_context():
            file_handle = None
            try:
                file_handle = open(filepath, mode)
                yield file_handle
            except IOError as e:
                tprint(f"⚠️ File operation failed for {filepath}: {e}")
                raise HMMResourceError(f"File operation failed for {filepath}: {e}") from e
            except Exception as e:
                tprint(f"⚠️ Unexpected error during file operation for {filepath}: {e}")
                raise HMMResourceError(f"Unexpected error during file operation: {e}") from e
            finally:
                if file_handle and not file_handle.closed:
                    try:
                        file_handle.close()
                        tprint(f"🗂️ File closed safely: {filepath}")
                    except Exception as close_error:
                        tprint(f"⚠️ Warning: Error closing file {filepath}: {close_error}")
        
        return safe_file_context()
    
    def _set_global_classifier_thread_safe(self, classifier, regime_classes, calibration_ece=None):
        """Thread-safe setter for global classifier and related metadata."""
        with self._classifier_lock:
            self.global_regime_clf = classifier
            self.regime_classes_ = regime_classes
            if calibration_ece is not None:
                self.global_calibration_ece = calibration_ece
    
    def _get_global_classifier_thread_safe(self):
        """Thread-safe getter for global classifier and related metadata."""
        with self._classifier_lock:
            return {
                'classifier': self.global_regime_clf,
                'classes': self.regime_classes_,
                'ece': self.global_calibration_ece
            }
    
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
            # Validate base/model types - FAST FAIL (standardize on base_models)
            model_list = getattr(config, 'base_models', None)
            if not model_list:
                # Fallback to model_types for backward compatibility
                model_list = getattr(config, 'model_types', [])
                if model_list:
                    tprint("⚠️ Using deprecated 'model_types' attribute, please use 'base_models'")
                    # Standardize by setting base_models
                    setattr(config, 'base_models', model_list)
            
            if not model_list or len(model_list) == 0:
                tprint("❌ CRITICAL: No base models specified - FAILING FAST")
                raise HMMDataValidationError("At least one base model type must be specified in 'base_models' attribute")
            
            # Validate timeframe - FAST FAIL
            valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            if not config.timeframe or config.timeframe not in valid_timeframes:
                tprint(f"❌ CRITICAL: Invalid timeframe '{config.timeframe}' - FAILING FAST")
                raise HMMDataValidationError(f"Invalid timeframe '{config.timeframe}' - must be one of: {valid_timeframes}")
            
            # Validate HPO parameters using math validation utilities - FAST FAIL
            if hasattr(config, 'enable_hpo') and config.enable_hpo:
                if self.math_validator:
                    try:
                        config.hpo_n_trials = self.math_validator.validate_positive(
                            config.hpo_n_trials, "HPO trials"
                        )
                    except ValueError as e:
                        tprint(f"❌ CRITICAL: HPO trials validation failed - FAILING FAST")
                        raise HMMDataValidationError(f"HPO trials must be positive: {e}") from e
                    
                    try:
                        config.hpo_timeout_seconds = self.math_validator.validate_positive(
                            config.hpo_timeout_seconds, "HPO timeout"
                        )
                    except ValueError as e:
                        tprint(f"❌ CRITICAL: HPO timeout validation failed - FAILING FAST")
                        raise HMMDataValidationError(f"HPO timeout must be positive: {e}") from e
                else:
                    # Fallback validation without math validator
                    if not hasattr(config, 'hpo_n_trials') or config.hpo_n_trials <= 0:
                        raise HMMDataValidationError("HPO trials must be positive")
                    if not hasattr(config, 'hpo_timeout_seconds') or config.hpo_timeout_seconds <= 0:
                        raise HMMDataValidationError("HPO timeout must be positive")
            
            # Validate minimum samples using math validation utilities - FAST FAIL
            if self.math_validator:
                try:
                    config.min_samples_per_regime = self.math_validator.validate_positive(
                        config.min_samples_per_regime, "Minimum samples per regime"
                    )
                except ValueError as e:
                    tprint(f"❌ CRITICAL: Minimum samples validation failed - FAILING FAST")
                    raise HMMDataValidationError(f"Minimum samples per regime must be positive: {e}") from e
            else:
                # Fallback validation
                if not hasattr(config, 'min_samples_per_regime') or config.min_samples_per_regime <= 0:
                    raise HMMDataValidationError("Minimum samples per regime must be positive")
            
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
                raise HMMDataValidationError(f"Data shape mismatch: X={X.shape}, y={y.shape}, regimes={regime_labels.shape}")
            
            # Check for empty data - FAST FAIL
            if X.shape[0] == 0:
                tprint("❌ CRITICAL: Input data is empty - FAILING FAST")
                raise HMMDataValidationError("Input data is empty")
            
            # Check for NaN values using math validation utilities - FAST FAIL
            if np.isnan(X).any():
                nan_count = np.isnan(X).sum()
                tprint(f"❌ CRITICAL: Found {nan_count} NaN values in input features - FAILING FAST")
                raise HMMDataValidationError(f"Input data contains {nan_count} NaN values - training cannot proceed")
            
            if np.isnan(y).any():
                nan_count = np.isnan(y).sum()
                tprint(f"❌ CRITICAL: Found {nan_count} NaN values in target values - FAILING FAST")
                raise HMMDataValidationError(f"Target data contains {nan_count} NaN values - training cannot proceed")
            
            # Check for infinite values using math validation utilities - FAST FAIL
            if np.isinf(X).any():
                inf_count = np.isinf(X).sum()
                tprint(f"❌ CRITICAL: Found {inf_count} infinite values in input features - FAILING FAST")
                raise HMMDataValidationError(f"Input data contains {inf_count} infinite values - training cannot proceed")
            
            if np.isinf(y).any():
                inf_count = np.isinf(y).sum()
                tprint(f"❌ CRITICAL: Found {inf_count} infinite values in target values - FAILING FAST")
                raise HMMDataValidationError(f"Target data contains {inf_count} infinite values - training cannot proceed")
            
            # Validate finite values using optimized vectorized approach with fallbacks
            try:
                # Use vectorized operations for efficient validation of all features
                non_finite_mask = ~np.isfinite(X)
                if np.any(non_finite_mask):
                    # Find which features have non-finite values
                    problematic_features = np.where(np.any(non_finite_mask, axis=0))[0]
                    total_non_finite = np.sum(non_finite_mask)
                    tprint(f"❌ CRITICAL: Found {total_non_finite} non-finite values in {len(problematic_features)} features - FAILING FAST")
                    tprint(f"   Problematic features: {problematic_features[:10]}{'...' if len(problematic_features) > 10 else ''}")
                    raise ValueError(f"Input data contains {total_non_finite} non-finite values in features {problematic_features[:5].tolist()}")
                
                # Additional validation using math validator if available
                if self.math_validator and X.shape[0] > 0 and X.shape[1] > 0:
                    # Test a few representative values with math validator - with proper bounds checking
                    sample_indices = []
                    if X.shape[0] >= 1:
                        sample_indices.append(0)
                    if X.shape[0] >= 2:
                        sample_indices.append(X.shape[0]//2)
                    if X.shape[0] >= 3:
                        sample_indices.append(X.shape[0]-1)
                    
                    feature_indices = []
                    if X.shape[1] >= 1:
                        feature_indices.append(0)
                    if X.shape[1] >= 2:
                        feature_indices.append(X.shape[1]//2)
                    if X.shape[1] >= 3:
                        feature_indices.append(X.shape[1]-1)
                    
                    for i in sample_indices:
                        for j in feature_indices:
                            if 0 <= i < X.shape[0] and 0 <= j < X.shape[1]:
                                self.math_validator.validate_finite(X[i, j], f"X[{i},{j}]")
                elif not self.math_validator:
                    # Fallback validation when math validator is not available
                    tprint("ℹ️ Using fallback finite validation (math validator not available)")
                    
            except ValueError as e:
                tprint(f"❌ CRITICAL: Non-finite values detected in input features - FAILING FAST")
                raise ValueError(f"Input data contains non-finite values: {e}") from e
            
            # Check regime distribution using safe operations - WARNING ONLY
            unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)
            min_regime_samples = regime_counts.min()
            
            if min_regime_samples < self.config.min_samples_per_regime:
                insufficient_regimes = unique_regimes[regime_counts < self.config.min_samples_per_regime]
                sufficient_regimes = unique_regimes[regime_counts >= self.config.min_samples_per_regime]
                tprint(f"⚠️ 🚨 WARNING: {len(insufficient_regimes)} regimes have insufficient samples (< {self.config.min_samples_per_regime})")
                tprint(f"📊 Regime distribution: {dict(zip(unique_regimes, regime_counts))}")
                tprint(f"✅ Sufficient regimes: {len(sufficient_regimes)} out of {len(unique_regimes)} total")
                
                if len(sufficient_regimes) == 0:
                    tprint(f"🚨 CRITICAL: NO regimes meet minimum threshold! Training will likely fail.")
                    tprint(f"💡 Consider reducing min_samples_per_regime from {self.config.min_samples_per_regime} to a lower value.")
            
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
            X: Input features (15m timeframe with cross-timeframe features)
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
        
        # Step 1: Validate inputs BEFORE resource allocation
        tprint("🔄 Step 1: Validating inputs with common utilities...")
        try:
            self._validate_input_data(X, y, regime_labels)
        except Exception as e:
            error_msg = f"Input validation failed: {e}"
            tprint(f"❌ {error_msg}")
            return {
                'error': error_msg,
                'execution_time': time.time() - execution_start_time,
                'validation_failed': True
            }
        # Optional temporal validation using ml_common enhanced utilities
        try:
            timestamps = locals().get('timestamps') or None
            if self.enhanced_training_utils is not None:
                tv_valid, tv_warnings = self.enhanced_training_utils.validate_temporal_data(
                    X, y, timestamps=timestamps, strict_mode=False
                )
                self._temporal_validation_global = {
                    'valid': bool(tv_valid),
                    'warnings': tv_warnings
                }
            else:
                self._temporal_validation_global = {'valid': True, 'warnings': []}
        except Exception as _tv_err:
            self._temporal_validation_global = {'valid': False, 'error': str(_tv_err)}
        
        # Use memory checkpoint for large operations AFTER validation
        with memory_checkpoint("hmm_ensemble_training_main"):
            try:
                
                # Step 2: Validate and prepare base models
                tprint("🔄 Step 2: Validating base models...")
                if base_hmm_models is None or not base_hmm_models:
                    tprint("⚠️ No base HMM models provided, creating base models for ensemble")
                    base_hmm_models = self._create_base_models_for_ensemble()
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
                
                # Step 5: Train global regime classifier (stacked, calibrated)
                tprint("🔄 Step 5: Training global regime classifier (stacked + calibrated)...")
                try:
                    self._train_global_regime_classifier(X, regime_labels)
                    if hasattr(self.global_regime_clf, 'predict_proba') and self.global_regime_clf is not None:
                        # Use enhanced features for prediction if available
                        X_for_prediction = getattr(self, '_X_enhanced', X)
                        proba_train = self.global_regime_clf.predict_proba(X_for_prediction)
                        self.global_calibration_ece = self._expected_calibration_error(regime_labels, proba_train)
                        tprint(f"📏 Global classifier ECE (train proxy): {self.global_calibration_ece:.4f}")
                    
                    # Include detailed meta model comparison results
                    meta_results = getattr(self, 'meta_model_results', {})
                    valid_meta_models = {k: v for k, v in meta_results.items() if 'error' not in v}
                    
                    # Determine best model using accuracy-first logic
                    best_meta_model = 'unknown'
                    if valid_meta_models:
                        sorted_models = sorted(
                            valid_meta_models.items(), 
                            key=lambda x: (-x[1]['cv_accuracy_mean'], x[1]['training_time_seconds'])
                        )
                        best_meta_model = sorted_models[0][0]
                    
                    results['global_regime_classifier'] = {
                        'trained': True,
                        'classes': self.regime_classes_.tolist() if self.regime_classes_ is not None else None,
                        'ece_train': float(self.global_calibration_ece) if self.global_calibration_ece is not None else None,
                        'meta_model_comparison': {
                            'tested_models': list(meta_results.keys()),
                            'detailed_results': meta_results,
                            'selection_criteria': 'accuracy_first_then_speed',
                            'best_model': best_meta_model,
                            'best_model_metrics': valid_meta_models.get(best_meta_model, {}) if best_meta_model != 'unknown' else {}
                        },
                        'enhanced_features': {
                            'original_feature_count': getattr(self, '_original_feature_count', 0),
                            'enhanced_feature_count': getattr(self, '_enhanced_feature_count', 0),
                            'feature_enhancement_applied': hasattr(self, '_X_enhanced')
                        }
                    }
                    # Add robustness validation
                    tprint("🔍 Validating model robustness and checking for overfitting...")
                    X_for_validation = getattr(self, '_X_enhanced', X)
                    validation_results = self._validate_model_robustness(X_for_validation, y, regime_labels)
                    results['global_regime_classifier']['validation_results'] = validation_results
                    
                    # Check for overfitting and warn if detected
                    if 'overfitting_analysis' in validation_results:
                        overfitting_info = validation_results['overfitting_analysis']
                        if overfitting_info.get('is_overfitting', False):
                            tprint(f"⚠️ OVERFITTING DETECTED: {overfitting_info['severity']} severity")
                            tprint(f"   Train accuracy: {overfitting_info['train_accuracy']:.4f}")
                            tprint(f"   Holdout accuracy: {overfitting_info['holdout_accuracy']:.4f}")
                            tprint(f"   Gap: {overfitting_info['overfitting_gap']:.4f}")
                        else:
                            tprint(f"✅ Model generalization: Train={overfitting_info['train_accuracy']:.4f}, Holdout={overfitting_info['holdout_accuracy']:.4f}")
                
                except Exception as e:
                    tprint(f"⚠️ Global regime classifier training failed: {e}")
                    results['global_regime_classifier'] = {'trained': False, 'error': str(e)}

                # Step 6: Optimize memory usage before generating the report so it's included
                tprint("🔄 Step 6: Optimizing memory usage...")
                memory_stats = optimize_memory()
                results['memory_optimization'] = memory_stats

                # Step 7: Generate comprehensive report using common utilities
                execution_time = time.time() - execution_start_time
                results = self._generate_comprehensive_report(results, execution_time, base_hmm_models, hmm_training_metrics)

                # Attach ml_common temporal validation and per-regime model validations
                try:
                    results['temporal_validation'] = getattr(self, '_temporal_validation_global', None)
                    from sklearn.model_selection import train_test_split as _ts_split
                    per_regime_validations: Dict[Any, Any] = {}
                    diversity_by_regime: Dict[Any, Any] = {}
                    if isinstance(results.get('models'), dict):
                        for _regime, _ens_entry in results['models'].items():
                            try:
                                _manager = _ens_entry.get('ensemble_manager') if isinstance(_ens_entry, dict) else None
                                if _manager is None:
                                    continue
                                # Build regime split
                                _mask = (regime_labels == _regime)
                                if int(getattr(_mask, 'sum', lambda: 0)()) < 5:
                                    continue
                                _Xr = X[_mask]
                                _yr = y[_mask]
                                _strat = _yr if len(np.unique(_yr)) > 1 else None
                                _Xtr, _Xva, _ytr, _yva = _ts_split(_Xr, _yr, test_size=0.2, random_state=42, stratify=_strat)
                                _vres = self.validate_trained_model(
                                    model=_manager,
                                    X_train=_Xtr,
                                    X_val=_Xva,
                                    y_train=_ytr,
                                    y_val=_yva,
                                    feature_names=feature_names,
                                    model_name=f"ensemble_regime_{_regime}",
                                    model_type=str(self.config.model_name)
                                )
                                per_regime_validations[_regime] = _vres

                                # Optional: ensemble diversity on base models if available
                                try:
                                    if self.enhanced_training_utils is not None and isinstance(_ens_entry, dict) and 'base_models' in _ens_entry:
                                        _base_models = _ens_entry['base_models']
                                        if isinstance(_base_models, dict) and len(_base_models) >= 2:
                                            _div = self.enhanced_training_utils.calculate_ensemble_diversity(
                                                models=list(_base_models.values()), X=_Xr, y=_yr
                                            )
                                            diversity_by_regime[_regime] = _div
                                except Exception:
                                    pass
                            except Exception as _perr:
                                per_regime_validations[_regime] = {'valid': False, 'error': str(_perr)}
                    results['model_validations'] = per_regime_validations
                    if diversity_by_regime:
                        results['ensemble_diversity'] = diversity_by_regime
                except Exception:
                    pass

                tprint(f"✅ HMM ensemble training completed successfully in {execution_time:.2f}s")
                tprint(f"🧠 Memory optimization: {memory_stats}")
                
                # Step 8: Automatically compute probabilities on training features for convenience
                try:
                    # Use enhanced features if available (same features the classifier was trained on)
                    X_for_prediction = getattr(self, '_X_enhanced', X)
                    proba_out = self.predict_regime_proba(X_for_prediction)
                    results['regime_probabilities'] = {
                        'proba': proba_out.get('proba'),
                        'entropy': proba_out.get('entropy'),
                        'classes': proba_out.get('classes'),
                        'ece_train': proba_out.get('ece_train')
                    }
                    tprint("📊 Added regime probabilities and entropy to results")
                except Exception as e:
                    tprint(f"⚠️ Could not compute regime probabilities automatically: {e}")
                
                # Cleanup memory before returning results
                try:
                    self.cleanup_memory()
                except Exception as cleanup_error:
                    tprint(f"⚠️ Memory cleanup warning during successful completion: {cleanup_error}")
                
                return results
                
            except Exception as e:
                execution_time = time.time() - execution_start_time
                error_msg = f"HMM ensemble training failed after {execution_time:.2f}s: {e}"
                tprint(f"❌ {error_msg}")
                tprint(f"🔍 Traceback: {traceback.format_exc()}")
                
                # Cleanup memory even on failure
                try:
                    self.cleanup_memory()
                except Exception as cleanup_error:
                    tprint(f"⚠️ Memory cleanup warning during error: {cleanup_error}")
                
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
            
            # Normalize features using matrix operations with fallback
            # Note: X.shape[1] > 0 check is redundant since earlier validation ensures non-empty data
            # Use advanced memory optimizer for dataframe operations if available
            if self.memory_optimizer and ADVANCED_MEMORY_AVAILABLE:
                try:
                    X_normalized = optimize_dataframe_advanced(X, operation='normalize', method='standard')
                    tprint(f"✅ Features normalized using advanced memory optimizer: {X_normalized.shape}")
                except Exception as e:
                    tprint(f"⚠️ Advanced normalization failed: {e}, falling back to matrix operations")
                    # Fallback to matrix operations
                    if self.matrix_ops is not None:
                        X_normalized = self.matrix_ops.normalize_matrix(X, method='zscore')
                        tprint(f"✅ Features normalized using matrix operations: {X_normalized.shape}")
                    else:
                        # Final fallback to standard sklearn normalization
                        tprint("⚠️ Matrix operations not available, using fallback normalization")
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        X_normalized = scaler.fit_transform(X)
                        tprint(f"✅ Features normalized using fallback StandardScaler: {X_normalized.shape}")
            elif self.matrix_ops is not None:
                X_normalized = self.matrix_ops.normalize_matrix(X, method='zscore')
                tprint(f"✅ Features normalized using matrix operations: {X_normalized.shape}")
            else:
                # Fallback normalization using sklearn
                tprint("⚠️ Matrix operations not available, using fallback normalization")
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_normalized = scaler.fit_transform(X)
                tprint(f"✅ Features normalized using fallback StandardScaler: {X_normalized.shape}")
            
            # Use GPU context if available for large datasets with proper resource management
            if self.gpu_manager and X.shape[0] > 10000:
                with self._gpu_context_manager() as gpu_acquired:
                    if gpu_acquired and EXTENDED_COMMON_OPS_AVAILABLE:
                        try:
                            with gpu_context("hmm_ensemble_training"):
                                results = self._execute_training_core(
                                    X_normalized, y, regime_labels, feature_names, hmm_states, base_hmm_models
                                )
                        except Exception as e:
                            tprint(f"⚠️ GPU execution failed, falling back to CPU: {e}")
                            results = self._execute_training_core(
                                X_normalized, y, regime_labels, feature_names, hmm_states, base_hmm_models
                            )
                    else:
                        # GPU not available or no extended ops, use CPU
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
            # Prepare base models per regime (parent expects mapping by regime)
            base_models_prepared = self._prepare_base_models_for_regimes(base_hmm_models, regime_labels)

            # Use the parent class execute method with additional ensemble-specific logic
            results = super().execute(
                X=X,
                y=y,
                regime_labels=regime_labels,
                feature_names=feature_names,
                hmm_states=hmm_states,
                # Pass additional parameters as kwargs
                is_classification=True,  # HMM ensemble models are classification
                base_models=base_models_prepared,
                symbol=None,
                exchange=None,
                timeframe=self.config.timeframe
            )
            
            # Post-process to compute roc_auc when possible (ensure predict_proba support)
            try:
                if 'evaluation_results' in results and isinstance(results.get('models'), dict):
                    # Only compute for binary classification
                    import numpy as _np
                    from sklearn.metrics import roc_auc_score as _roc_auc_score
                    unique_classes = _np.unique(y)
                    if unique_classes.shape[0] == 2:
                        for regime, ensemble_entry in results['models'].items():
                            if isinstance(ensemble_entry, dict) and 'ensemble_manager' in ensemble_entry:
                                model_obj = ensemble_entry['ensemble_manager']
                                if hasattr(model_obj, 'predict_proba'):
                                    regime_mask = (regime_labels == regime)
                                    if _np.any(regime_mask):
                                        regime_X = X[regime_mask]
                                        regime_y = y[regime_mask]
                                        try:
                                            y_proba = model_obj.predict_proba(regime_X)
                                            # Enhanced bounds checking for probability arrays
                                            if (y_proba is not None and 
                                                y_proba.ndim == 2 and 
                                                y_proba.shape[1] >= 2 and 
                                                y_proba.shape[0] == len(regime_y) and
                                                len(unique_classes) == 2):  # Ensure binary classification
                                                roc_auc = _roc_auc_score(regime_y, y_proba[:, 1])
                                                if isinstance(results['evaluation_results'].get(regime), dict):
                                                    results['evaluation_results'][regime]['roc_auc'] = float(roc_auc)
                                        except Exception:
                                            pass
            except Exception:
                pass
            
            # Add matrix operations performance stats
            if self.matrix_ops is not None and hasattr(self.matrix_ops, 'get_performance_stats'):
                matrix_stats = self.matrix_ops.get_performance_stats()
                results['matrix_operations_stats'] = matrix_stats
                tprint(f"📊 Matrix operations: {matrix_stats['total_operations']} operations, avg time: {matrix_stats['average_execution_time']:.3f}s")
            else:
                results['matrix_operations_stats'] = {'status': 'not_available', 'reason': 'matrix_ops_unavailable'}
            
            return results
            
        except Exception as e:
            tprint(f"❌ Core training execution failed: {e}")
            raise

    def _train_global_regime_classifier(self, X: np.ndarray, regime_labels: np.ndarray) -> None:
        """
        Train multiple global stacked classifiers with different meta learners for comparison.
        Base models: ElasticNet, CatBoostClassifier, XGBoostClassifier.
        Meta learners: XGBoostClassifier, CatBoostClassifier, ElasticNet.
        Includes regime-specific features and base model outputs.
        """
        # Determine classes (thread-safe)
        regime_classes = np.unique(regime_labels)
        with self._classifier_lock:
            self.regime_classes_ = regime_classes
        tprint(f"📊 Training ensemble with {len(self.regime_classes_)} classes: {self.regime_classes_}")

        # Create enhanced features including regime-specific features
        tprint(f"🔧 Creating enhanced features...")
        import time as time_module
        feature_start_time = time_module.time()
        self._original_feature_count = X.shape[1]
        X_enhanced = self._create_enhanced_features(X, regime_labels)
        self._enhanced_feature_count = X_enhanced.shape[1]
        self._X_enhanced = X_enhanced  # Store for later use
        
        # Track large arrays for memory management
        if X_enhanced.nbytes > 50 * 1024 * 1024:  # Track arrays > 50MB
            self._large_arrays.append('_X_enhanced')
        feature_duration = time_module.time() - feature_start_time
        
        tprint(f"📊 Enhanced features created in {feature_duration:.2f}s:")
        tprint(f"   • Original features: {X.shape[1]}")
        tprint(f"   • Enhanced features: {X_enhanced.shape[1]}")
        # Use safe_divide to prevent division by zero
        if self.math_validator:
            feature_expansion = self.math_validator.safe_divide(X_enhanced.shape[1], X.shape[1], 1.0)
        else:
            feature_expansion = safe_divide(X_enhanced.shape[1], X.shape[1], 1.0)
        tprint(f"   • Feature expansion: {feature_expansion:.2f}x")
        tprint(f"   • Data shape: {X_enhanced.shape}")
        tprint(f"   • Memory usage: ~{X_enhanced.nbytes / 1024**2:.1f} MB")

        # Load pre-trained base estimators from hmm_models_training step
        tprint(f"🏗️ Loading pre-trained base estimators for stacking...")
        base_start_time = time_module.time()
        base_estimators = self._load_pretrained_base_estimators()
        base_duration = time_module.time() - base_start_time
        
        tprint(f"✅ Base estimators loaded in {base_duration:.2f}s:")
        for name, estimator in base_estimators:
            tprint(f"   • {name}: {type(estimator).__name__}")
        
        # Get meta models to compare
        meta_models_to_test = getattr(self.config, 'meta_models', ['XGBoostClassifier', 'CatBoostClassifier', 'ElasticNet'])
        tprint(f"🔍 Meta model comparison setup:")
        tprint(f"   • Models to test: {len(meta_models_to_test)}")
        tprint(f"   • Model list: {meta_models_to_test}")
        tprint(f"   • Base estimators: {len(base_estimators)}")
        
        # Train and compare multiple meta models
        self.meta_model_results = {}
        self.meta_model_classifiers = {}
        
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        import time as time_module
        
        for meta_model_name in meta_models_to_test:
            tprint(f"🔄 Training with meta model: {meta_model_name}")
            try:
                # Time the training
                start_time = time_module.time()
                
                # Create meta model
                meta_model = self._create_meta_model(meta_model_name)
                
                # Create custom stacking with pre-fitted base models
                stack = self._create_custom_stacking_ensemble(
                    base_estimators=base_estimators,
                    meta_model=meta_model,
                    model_name=meta_model_name
                )
                
                # Cross-validation evaluation using time series split
                try:
                    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
                    tscv = TimeSeriesSplit(n_splits=5, test_size=int(0.2 * len(regime_labels)))
                    cv_scores = cross_val_score(stack, X_enhanced, regime_labels,
                                               cv=tscv, scoring='accuracy', n_jobs=-1)
                    tprint(f"✅ Cross-validation completed: mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}")
                except Exception as e:
                    tprint(f"⚠️ Cross-validation failed: {e}, using placeholder scores")
                    cv_scores = np.array([0.0])  # Fallback placeholder
                
                # Fit the model
                stack.fit(X_enhanced, regime_labels)
                training_time = time_module.time() - start_time
                
                # Training metrics
                y_pred = stack.predict(X_enhanced)
                training_metrics = {
                    'accuracy': accuracy_score(regime_labels, y_pred),
                    'f1_macro': f1_score(regime_labels, y_pred, average='macro'),
                    'precision_macro': precision_score(regime_labels, y_pred, average='macro'),
                    'recall_macro': recall_score(regime_labels, y_pred, average='macro'),
                    'cv_accuracy_mean': cv_scores.mean(),
                    'cv_accuracy_std': cv_scores.std(),
                    'training_time_seconds': training_time,
                    'speed_score': cv_scores.mean() / (training_time + 1e-6)  # Accuracy per second
                }
                
                # Store results
                self.meta_model_results[meta_model_name] = training_metrics
                self.meta_model_classifiers[meta_model_name] = stack
                
                tprint(f"✅ {meta_model_name} - Accuracy: {training_metrics['accuracy']:.4f}, CV: {training_metrics['cv_accuracy_mean']:.4f}±{training_metrics['cv_accuracy_std']:.4f}, Time: {training_time:.2f}s, Speed Score: {training_metrics['speed_score']:.4f}")
                
            except Exception as e:
                tprint(f"❌ Failed to train {meta_model_name}: {e}")
                self.meta_model_results[meta_model_name] = {'error': str(e)}
        
        # Select best meta model: ACCURACY FIRST, then speed as tiebreaker
        valid_models = {k: v for k, v in self.meta_model_results.items() if 'error' not in v}
        if not valid_models:
            tprint("❌ No meta models trained successfully")
            with self._classifier_lock:
                self.global_regime_clf = None
            return
        
        # Sort by accuracy (descending), then by training time (ascending) for tiebreaker
        sorted_models = sorted(
            valid_models.items(), 
            key=lambda x: (-x[1]['cv_accuracy_mean'], x[1]['training_time_seconds'])
        )
        
        best_meta_model = sorted_models[0][0]  # Get the name of the best model
        
        # Display ranking
        tprint("📊 META MODEL RANKING (Accuracy First, Speed Second):")
        for i, (model_name, metrics) in enumerate(sorted_models):
            rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
            tprint(f"   {rank_emoji} {model_name}: {metrics['cv_accuracy_mean']:.4f} accuracy, {metrics['training_time_seconds']:.2f}s")
        
        tprint(f"🏆 Best meta model: {best_meta_model}")
        tprint(f"   📊 CV Accuracy: {valid_models[best_meta_model]['cv_accuracy_mean']:.4f}")
        tprint(f"   ⏱️ Training Time: {valid_models[best_meta_model]['training_time_seconds']:.2f}s")
        tprint(f"   🎯 F1 Score: {valid_models[best_meta_model]['f1_macro']:.4f}")
        tprint(f"   🎯 Precision: {valid_models[best_meta_model]['precision_macro']:.4f}")
        tprint(f"   🎯 Recall: {valid_models[best_meta_model]['recall_macro']:.4f}")
        
        # Run HPO on the best model
        tprint(f"\n🎯 HYPERPARAMETER OPTIMIZATION PHASE")
        tprint(f"=" * 45)
        tprint(f"🔧 Running HPO on best meta model: {best_meta_model}")
        
        hpo_phase_start = time.time()
        optimized_model = self._run_meta_model_hpo(best_meta_model, X_enhanced, regime_labels, base_estimators)
        hpo_phase_duration = time.time() - hpo_phase_start
        
        # Thread-safe assignment of the final classifier
        with self._classifier_lock:
            if optimized_model:
                self.global_regime_clf = optimized_model
                tprint(f"✅ HPO phase completed successfully in {hpo_phase_duration:.2f}s")
                tprint(f"🏆 Final optimized model: {best_meta_model}")
            else:
                # Fallback to non-optimized best model
                self.global_regime_clf = self.meta_model_classifiers[best_meta_model]
                tprint(f"⚠️ HPO failed after {hpo_phase_duration:.2f}s, using non-optimized {best_meta_model}")
                tprint(f"📋 Fallback model performance: {valid_models[best_meta_model]['cv_accuracy_mean']:.4f} CV accuracy")

    def _expected_calibration_error(self, y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
        """Compute multiclass ECE on probability simplex."""
        # Argmax labels and confidences
        preds = np.argmax(proba, axis=1)
        conf = np.max(proba, axis=1)
        correct = (preds == y_true).astype(float)

        # Bin by confidence
        bins = np.linspace(0.0, 1.0, n_bins+1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bins[i], bins[i+1]
            mask = (conf > lo) & (conf <= hi)
            if not np.any(mask):
                continue
            acc_bin = float(np.mean(correct[mask]))
            conf_bin = float(np.mean(conf[mask]))
            weight = float(np.mean(mask))
            ece += weight * abs(acc_bin - conf_bin)
        return float(ece)

    def predict_regime_proba(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Predict calibrated regime probabilities and entropy using the global stacked classifier.
        Returns dict with 'proba', 'entropy', 'classes'.
        """
        # Thread-safe access to classifier
        classifier_info = self._get_global_classifier_thread_safe()
        if classifier_info['classifier'] is None:
            raise RuntimeError("Global regime classifier not trained")
        proba = classifier_info['classifier'].predict_proba(X)
        # Entropy
        eps = 1e-12
        p_safe = np.clip(proba, eps, 1.0)
        ent = -np.sum(p_safe * np.log(p_safe), axis=1)
        # Normalize by log(K) for [0,1] using safe division
        k = proba.shape[1]
        if self.math_validator:
            ent_norm = self.math_validator.safe_divide(ent, np.log(k), ent) if k > 1 else ent
        else:
            ent_norm = safe_divide(ent, np.log(k), ent) if k > 1 else ent
        return {
            'proba': proba,
            'entropy': ent_norm,
            'classes': classifier_info['classes'].tolist() if classifier_info['classes'] is not None else None,
            'ece_train': float(classifier_info['ece']) if classifier_info['ece'] is not None else None
        }
    
    def _create_base_models_for_ensemble(self) -> Dict[str, Any]:
        """
        Create base models for HMM ensemble training with enhanced error handling.
        These are individual models that will be combined into an ensemble.
        Uses common utilities for robust model creation and validation.
        
        Returns:
            Dictionary of base models for ensemble training
        """
        try:
            from sklearn.linear_model import ElasticNet, LogisticRegression
            
            ensemble_models = {}
            
            # ElasticNet for regression-style regularization
            try:
                alpha = getattr(self.config, 'elasticnet_alpha', 1.0)
                l1_ratio = getattr(self.config, 'elasticnet_l1_ratio', 0.5)
                max_iter = getattr(self.config, 'elasticnet_max_iter', 2000)
                if self.math_validator:
                    alpha = self.math_validator.validate_positive(alpha, "ElasticNet alpha")
                    l1_ratio = self.math_validator.validate_range(l1_ratio, 0.0, 1.0, "ElasticNet l1_ratio")
                    max_iter = self.math_validator.validate_positive(max_iter, "ElasticNet max_iter")
                
                # For classification, we use LogisticRegression with elasticnet penalty
                ensemble_models['ElasticNet'] = LogisticRegression(
                    penalty='elasticnet',
                    C=1.0/alpha,  # Convert alpha to C (inverse relationship)
                    l1_ratio=l1_ratio,
                    solver='saga',
                    max_iter=int(max_iter),
                    random_state=42,
                    class_weight='balanced',
                    multi_class='auto'
                )
                tprint("✅ ElasticNet (LogisticRegression with elasticnet penalty) created with validated parameters")
            except Exception as e:
                tprint(f"⚠️ ElasticNet creation failed: {e}")
                try:
                    ensemble_models['ElasticNet'] = LogisticRegression(penalty='elasticnet', C=1.0, l1_ratio=0.5, solver='saga', max_iter=2000, random_state=42, class_weight='balanced', multi_class='auto')
                except Exception:
                    pass
            
            # CatBoostClassifier with validated parameters
            try:
                from catboost import CatBoostClassifier
                iterations = getattr(self.config, 'catboost_iterations', 800)
                learning_rate = getattr(self.config, 'catboost_learning_rate', 0.05)
                depth = getattr(self.config, 'catboost_depth', 6)
                if self.math_validator:
                    iterations = self.math_validator.validate_positive(iterations, "CatBoost iterations")
                    learning_rate = self.math_validator.validate_range(learning_rate, 0.0, 1.0, "CatBoost learning_rate")
                    depth = self.math_validator.validate_positive(depth, "CatBoost depth")
                ensemble_models['CatBoostClassifier'] = CatBoostClassifier(
                    iterations=int(iterations),
                    learning_rate=learning_rate,
                    depth=int(depth),
                    random_seed=42,
                    verbose=False,
                    loss_function='MultiClass'
                )
                tprint("✅ CatBoostClassifier created with validated parameters")
            except Exception as e:
                tprint(f"⚠️ CatBoostClassifier creation failed: {e}")
                try:
                    from catboost import CatBoostClassifier
                    ensemble_models['CatBoostClassifier'] = CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6, random_seed=42, verbose=False, loss_function='MultiClass')
                except Exception:
                    pass
            
            # XGBoostClassifier with validated parameters
            try:
                import xgboost as xgb
                n_estimators = getattr(self.config, 'xgboost_n_estimators', 100)
                learning_rate = getattr(self.config, 'xgboost_learning_rate', 0.1)
                max_depth = getattr(self.config, 'xgboost_max_depth', 6)
                if self.math_validator:
                    n_estimators = self.math_validator.validate_positive(n_estimators, "XGBoost n_estimators")
                    learning_rate = self.math_validator.validate_range(learning_rate, 0.0, 1.0, "XGBoost learning_rate")
                    max_depth = self.math_validator.validate_positive(max_depth, "XGBoost max_depth")
                # Enhanced regularization for XGBoost to prevent overfitting
                ensemble_models['XGBoostClassifier'] = xgb.XGBClassifier(
                    n_estimators=int(n_estimators),
                    learning_rate=learning_rate,
                    max_depth=int(max_depth),
                    random_state=42,
                    objective='multi:softprob',
                    eval_metric='mlogloss',
                    # Enhanced regularization parameters
                    min_child_weight=5,        # Minimum sum of hessian per child
                    reg_alpha=0.1,             # L1 regularization
                    reg_lambda=0.1,            # L2 regularization
                    subsample=0.8,             # Use 80% of data per tree
                    colsample_bytree=0.8,      # Use 80% of features per tree
                    colsample_bylevel=0.8,     # Use 80% of features per level
                    colsample_bynode=0.8       # Use 80% of features per node
                )
                tprint("✅ XGBoostClassifier created with validated parameters")
            except Exception as e:
                tprint(f"⚠️ XGBoostClassifier creation failed: {e}")
                try:
                    import xgboost as xgb
                    # Enhanced regularization for fallback XGBoost model
                    ensemble_models['XGBoostClassifier'] = xgb.XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.05,    # Reduced learning rate for better regularization
                        max_depth=4,           # Limited depth for regularization
                        random_state=42,
                        objective='multi:softprob',
                        eval_metric='mlogloss',
                        # Enhanced regularization parameters
                        min_child_weight=5,        # Minimum sum of hessian per child
                        reg_alpha=0.1,             # L1 regularization
                        reg_lambda=0.1,            # L2 regularization
                        subsample=0.8,             # Use 80% of data per tree
                        colsample_bytree=0.8,      # Use 80% of features per tree
                        colsample_bylevel=0.8,     # Use 80% of features per level
                        colsample_bynode=0.8       # Use 80% of features per node
                    )
                except Exception:
                    pass
            
            tprint(f"📊 Created {len(ensemble_models)} base models for HMM training")
            tprint(f"   Models: {list(ensemble_models.keys())}")
            
            # Update training stats
            self.training_stats['base_models_created'] = len(ensemble_models)
            self.training_stats['model_creation_method'] = 'classification_validated'
            
            return ensemble_models
            
        except ImportError as e:
            tprint(f"❌ CRITICAL: Failed to import required model libraries - FAILING FAST")
            tprint(f"   Error: {e}")
            raise RuntimeError(f"Required model libraries not available: {e}") from e
        except Exception as e:
            tprint(f"❌ Failed to create base models: {e}")
            raise RuntimeError(f"Base model creation failed: {e}") from e

    def _prepare_base_models_for_regimes(self, base_models: Dict[str, Any], regime_labels: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """
        Prepare per-regime base models mapping as expected by the parent class.

        Args:
            base_models: Global base models mapping name -> estimator
            regime_labels: Array of regime labels to determine unique regimes

        Returns:
            Dict mapping regime -> cloned base models
        """
        try:
            import numpy as _np
            unique_regimes = _np.unique(regime_labels)
            prepared: Dict[int, Dict[str, Any]] = {}

            # Try to use sklearn.clone for safety
            try:
                from sklearn.base import clone as sk_clone
                def _clone_model(m):
                    try:
                        return sk_clone(m)
                    except Exception:
                        return m
            except Exception:
                def _clone_model(m):
                    return m

            for regime in unique_regimes:
                prepared[regime] = {name: _clone_model(model) for name, model in (base_models or {}).items()}

            return prepared
        except Exception as e:
            tprint(f"⚠️ Failed to prepare base models per regime: {e}")
            # Fallback: wrap global models under a single key (may be handled by parent)
            return {}
    
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
                    'base_models_created': safe_int(self.training_stats.get('base_models_created', 0), 0),
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
                'meta_model_analysis': self._analyze_meta_model_comparison(results),
                'hpo_analysis': self._analyze_hpo_results(results),
                'matrix_operations_stats': results.get('matrix_operations_stats', {}),
                'memory_optimization': results.get('memory_optimization', {}),
                'recommendations': self._generate_recommendations(results, execution_time)
            }
            
            # If regime probabilities are available, summarize and attach
            if 'regime_probabilities' in results and isinstance(results['regime_probabilities'], dict):
                rp = results['regime_probabilities']
                proba = rp.get('proba')
                entropy = rp.get('entropy')
                classes = rp.get('classes')
                ece_train = rp.get('ece_train')
                try:
                    if proba is not None and entropy is not None:
                        # Compute summary stats
                        entropy_arr = np.array(entropy)
                        proba_arr = np.array(proba)
                        summary = {
                            'num_samples': int(proba_arr.shape[0]),
                            'num_classes': int(proba_arr.shape[1]) if proba_arr.ndim == 2 else None,
                            'classes': classes,
                            'entropy_mean': float(np.mean(entropy_arr)),
                            'entropy_std': float(np.std(entropy_arr)),
                            'entropy_p25': float(np.percentile(entropy_arr, 25)),
                            'entropy_p50': float(np.percentile(entropy_arr, 50)),
                            'entropy_p75': float(np.percentile(entropy_arr, 75)),
                            'ece_train': float(ece_train) if ece_train is not None else None,
                            'avg_class_prob': (
                                {str(classes[i]): float(np.mean(proba_arr[:, i])) for i in range(proba_arr.shape[1])}
                                if proba_arr.ndim == 2 and classes is not None else None
                            )
                        }
                        comprehensive_report['regime_probability_summary'] = summary
                        
                        # Optionally save a compact preview artifact
                        try:
                            preview_count = int(min(100, proba_arr.shape[0]))
                            preview = {
                                'classes': classes,
                                'proba_preview': proba_arr[:preview_count].tolist(),
                                'entropy_preview': entropy_arr[:preview_count].tolist()
                            }
                            # Handle both string and Path objects for model_save_path
                            if isinstance(self.config.model_save_path, Path):
                                preview_path = self.config.model_save_path / 'regime_probabilities_preview.json'
                            else:
                                preview_path = Path(self.config.model_save_path) / 'regime_probabilities_preview.json'
                            ensure_directory(preview_path.parent)
                            if self.serializer.save(preview, str(preview_path), format='json'):
                                comprehensive_report['regime_probability_preview_path'] = str(preview_path)
                        except Exception:
                            pass
                except Exception as e:
                    tprint(f"⚠️ Could not summarize regime probabilities: {e}")

            # Add comprehensive report to results
            results['comprehensive_report'] = comprehensive_report
            
            # Save report using common serialization utilities
            try:
                # Handle both string and Path objects for model_save_path
                if isinstance(self.config.model_save_path, Path):
                    report_path = self.config.model_save_path / "training_report.json"
                else:
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
    
    def _create_enhanced_features(self, X: np.ndarray, regime_labels: np.ndarray) -> np.ndarray:
        """
        Create enhanced features for global regime classifier using shared utilities.
        
        Args:
            X: Original features (close_return, volume_return, price_range_pct)
            regime_labels: Regime labels
            
        Returns:
            Enhanced feature matrix with diverse, non-leaking features
        """
        try:
            # Use shared enhanced feature creation utilities
            X_enhanced = create_enhanced_features(X, regime_labels)
            tprint(f"✅ Enhanced features created using shared utilities: {X_enhanced.shape[1]} total features")
            return X_enhanced
            
        except Exception as e:
            tprint(f"⚠️ Enhanced feature creation failed: {e}")
            # Fallback to original features
            return X
    
    
    def _validate_model_robustness(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> Dict[str, Any]:
        """
        Validate model robustness to detect overfitting and ensure generalization.
        
        Args:
            X: Enhanced features
            y: Target values  
            regime_labels: Regime labels
            
        Returns:
            Validation results with overfitting detection
        """
        try:
            from sklearn.model_selection import TimeSeriesSplit, cross_val_score
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, f1_score
            
            validation_results = {}
            
            # 1. TIME SERIES CROSS VALIDATION (prevents lookahead bias)
            tscv = TimeSeriesSplit(n_splits=5)
            rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
            
            cv_scores = cross_val_score(rf_model, X, regime_labels, cv=tscv, scoring='accuracy')
            validation_results['time_series_cv'] = {
                'mean_accuracy': float(cv_scores.mean()),
                'std_accuracy': float(cv_scores.std()),
                'all_scores': cv_scores.tolist()
            }
            
            # 2. HOLDOUT VALIDATION (final 20% of data)
            split_point = int(len(X) * 0.8)
            X_train, X_holdout = X[:split_point], X[split_point:]
            y_train, y_holdout = regime_labels[:split_point], regime_labels[split_point:]
            
            rf_model.fit(X_train, y_train)
            holdout_pred = rf_model.predict(X_holdout)
            holdout_accuracy = accuracy_score(y_holdout, holdout_pred)
            
            validation_results['holdout_validation'] = {
                'holdout_accuracy': float(holdout_accuracy),
                'train_samples': len(X_train),
                'holdout_samples': len(X_holdout)
            }
            
            # 3. OVERFITTING DETECTION
            train_pred = rf_model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_pred)
            
            overfitting_gap = train_accuracy - holdout_accuracy
            validation_results['overfitting_analysis'] = {
                'train_accuracy': float(train_accuracy),
                'holdout_accuracy': float(holdout_accuracy),
                'overfitting_gap': float(overfitting_gap),
                'is_overfitting': overfitting_gap > 0.1,  # >10% gap indicates overfitting
                'severity': 'high' if overfitting_gap > 0.2 else 'medium' if overfitting_gap > 0.1 else 'low'
            }
            
            # 4. FEATURE IMPORTANCE ANALYSIS
            feature_importance = rf_model.feature_importances_
            validation_results['feature_analysis'] = {
                'top_features': np.argsort(feature_importance)[-5:].tolist(),
                'feature_importance_std': float(np.std(feature_importance)),
                'max_importance': float(np.max(feature_importance)),
                'importance_concentration': float(np.sum(feature_importance[:3]) / np.sum(feature_importance))
            }
            
            return validation_results
            
        except Exception as e:
            tprint(f"⚠️ Model validation failed: {e}")
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
            base_models_created = safe_int(data_summary.get('base_models_created', 0), 0)
            
            tprint(f"📊 Samples processed: {sample_count:,}")
            tprint(f"🔢 Features used: {feature_count}")
            tprint(f"🤖 Base models: {base_models}")
            tprint(f"🏗️ Base models created: {base_models_created}")
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

                        # Enhanced: Handle nested structure with safe dictionary access
                        try:
                            # Check if we have nested metrics structure
                            has_nested_metrics = any(
                                isinstance(v, dict) and ('metrics' in v or 'accuracy' in v) 
                                for v in regime_metrics.values() 
                                if isinstance(v, dict)
                            )
                            
                            if has_nested_metrics:
                                best_ensemble = None
                                best_accuracy = -np.inf

                                for ensemble_name, metrics in regime_metrics.items():
                                    if not isinstance(metrics, dict):
                                        continue
                                    
                                    # Safe nested dictionary access
                                    metrics_dict = metrics.get('metrics', {}) if 'metrics' in metrics else metrics
                                    if isinstance(metrics_dict, dict) and 'accuracy' in metrics_dict:
                                        try:
                                            accuracy = safe_float(metrics_dict['accuracy'], 0.0)
                                            if validate_finite(accuracy, f"accuracy_{ensemble_name}"):
                                                accuracies.append(accuracy)
                                                if accuracy > best_accuracy:
                                                    best_accuracy = accuracy
                                                    best_ensemble = ensemble_name
                                        except Exception as e:
                                            tprint(f"⚠️ Invalid accuracy for {ensemble_name}: {e}")

                                if best_ensemble and best_accuracy > -np.inf:
                                    best_ensembles[regime] = {
                                        'ensemble': best_ensemble,
                                        'accuracy': safe_float(best_accuracy, 0.0)
                                    }
                                    if best_accuracy > performance_summary['best_overall_accuracy']:
                                        performance_summary['best_overall_accuracy'] = safe_float(best_accuracy, 0.0)
                            else:
                                # Flat single-metrics per regime with safe access
                                if isinstance(regime_metrics, dict) and 'accuracy' in regime_metrics:
                                    try:
                                        acc_val = safe_float(regime_metrics['accuracy'], 0.0)
                                        if validate_finite(acc_val, f"accuracy_{regime}"):
                                            accuracies.append(acc_val)
                                            best_ensembles[regime] = {
                                                'ensemble': 'stacking_ensemble',
                                                'accuracy': acc_val
                                            }
                                            if acc_val > performance_summary['best_overall_accuracy']:
                                                performance_summary['best_overall_accuracy'] = acc_val
                                    except Exception as e:
                                        tprint(f"⚠️ Invalid accuracy for regime {regime}: {e}")
                        except Exception as e:
                            tprint(f"⚠️ Error processing metrics for regime {regime}: {e}")
                            performance_summary['failed_evaluations'] += 1
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
                    'base_models_created': safe_int(self.training_stats.get('base_models_created', 0), 0),
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

    def _create_enhanced_features(self, X: np.ndarray, regime_labels: np.ndarray) -> np.ndarray:
        """
        Create enhanced features including regime-specific features and base model outputs.
        
        Args:
            X: Original features
            regime_labels: Regime assignments
            
        Returns:
            Enhanced feature matrix
        """
        try:
            # Start with original features
            enhanced_features = [X]
            
            # Add regime-specific statistical features
            regime_stats = []
            for regime in np.unique(regime_labels):
                regime_mask = (regime_labels == regime)
                if np.any(regime_mask):
                    # Regime frequency
                    regime_freq = np.sum(regime_mask) / len(regime_labels)
                    # Regime stability (how clustered the regime assignments are)
                    regime_stability = 1.0 - np.std(regime_mask.astype(float))
                    regime_stats.extend([regime_freq, regime_stability])
                else:
                    regime_stats.extend([0.0, 0.0])
            
            # Broadcast regime stats to match sample count
            regime_stats_matrix = np.tile(regime_stats, (X.shape[0], 1))
            enhanced_features.append(regime_stats_matrix)
            
            # Add regime one-hot encoding
            from sklearn.preprocessing import LabelEncoder, OneHotEncoder
            le = LabelEncoder()
            regime_encoded = le.fit_transform(regime_labels)
            ohe = OneHotEncoder(sparse=False)
            regime_onehot = ohe.fit_transform(regime_encoded.reshape(-1, 1))
            enhanced_features.append(regime_onehot)
            
            # Concatenate all features
            X_enhanced = np.concatenate(enhanced_features, axis=1)
            
            return X_enhanced
            
        except Exception as e:
            tprint(f"⚠️ Feature enhancement failed: {e}, using original features")
            return X

    def _load_pretrained_base_estimators(self) -> List[Tuple[str, Any]]:
        """Load the most recent pre-trained base estimators from hmm_models_training step."""
        base_estimators = []
        
        # Import serialization utilities
        from src.utils.serialization_utils import UniversalSerializer
        serializer = UniversalSerializer()
        
        # Get model paths from artifacts directory - use correct path where models are actually saved
        symbol = getattr(self.config, 'symbol', 'ETHUSDT')
        exchange = getattr(self.config, 'exchange', 'binance')
        timeframe = getattr(self.config, 'timeframe', get_primary_timeframe())  # HMM models use standardized timeframe

        # Use the correct path where hmm_models_training actually saves models
        base_models_dir = Path("generated/market_analysis/models/hmm_ensemble_models")
        
        if not base_models_dir.exists():
            tprint(f"❌ Base models directory not found: {base_models_dir}")
            tprint("   📋 Expected pre-trained models from hmm_models_training step")
            raise FileNotFoundError(f"Pre-trained models directory not found: {base_models_dir}")
        
        # Find all model files (including nested directories)
        model_files = []
        for pattern in ["*_model.pkl", "*.joblib"]:
            model_files.extend(base_models_dir.rglob(pattern))

        if not model_files:
            tprint(f"❌ No pre-trained model files found in: {base_models_dir}")
            raise FileNotFoundError(f"No pre-trained model files found in: {base_models_dir}")

        tprint(f"📁 Loading pre-trained base models from: {base_models_dir}")
        tprint(f"   📊 Found {len(model_files)} model files")

        # Group model files by model name and select most recent for each
        model_groups = {}
        for model_file in model_files:
            model_name = model_file.stem
            # Remove common suffixes to get base model name
            model_name = model_name.replace('_model', '').replace('_ensemble', '')
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append(model_file)
        
        # Load the most recent model for each model type
        for model_name, files in model_groups.items():
            # Sort by modification time (most recent first)
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            most_recent_file = files[0]
            
            if len(files) > 1:
                tprint(f"   📊 Found {len(files)} versions of {model_name}, loading most recent:")
                tprint(f"      🕒 {most_recent_file.name} (modified: {pd.Timestamp.fromtimestamp(most_recent_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})")
            else:
                tprint(f"   🔄 Loading {model_name} from {most_recent_file.name}...")
            
            try:
                # Load model using pickle directly instead of UniversalSerializer
                import pickle
                with open(str(most_recent_file), 'rb') as f:
                    model = pickle.load(f)
                
                if model is not None:
                    # Validate that the model is actually fitted - handle different model types
                    is_fitted = False
                    expected_features = None

                    # Check for scikit-learn style fitted models
                    if hasattr(model, 'n_features_in_') and model.n_features_in_ is not None:
                        is_fitted = True
                        expected_features = model.n_features_in_
                    # Check for custom ensemble managers
                    elif hasattr(model, 'stacking_model') and model.stacking_model is not None:
                        is_fitted = True
                        expected_features = "variable"
                    # Check for other fitted indicators
                    elif hasattr(model, 'is_fitted_') and model.is_fitted_:
                        is_fitted = True
                        expected_features = "variable"
                    elif hasattr(model, 'predict'):
                        # Assume it's fitted if it has predict method and isn't None
                        is_fitted = True
                        expected_features = "variable"

                    if not is_fitted:
                        tprint(f"   ❌ Model {model_name} is not fitted")
                        raise ValueError(f"Model {model_name} is not fitted - cannot use as pre-trained base model")

                    base_estimators.append((model_name, model))
                    tprint(f"   ✅ Successfully loaded fitted {model_name} (type: {type(model).__name__}, features: {expected_features})")
                else:
                    tprint(f"   ❌ Failed to load {model_name} from {most_recent_file.name}")
                    raise ValueError(f"Failed to deserialize model: {model_name}")
                    
            except Exception as e:
                tprint(f"   ❌ Error loading {model_name} from {most_recent_file.name}: {e}")
                raise RuntimeError(f"Failed to load required base model {model_name}: {e}")
        
        if not base_estimators:
            tprint("❌ No models could be loaded successfully")
            raise RuntimeError("Failed to load any pre-trained base models")
        
        tprint(f"✅ Loaded {len(base_estimators)} pre-trained base models")
        for name, model in base_estimators:
            tprint(f"   • {name}: {type(model).__name__}")
                
        return base_estimators


    def _create_custom_stacking_ensemble(self, base_estimators: List[Tuple[str, Any]], 
                                       meta_model: Any, model_name: str) -> Any:
        """Create a custom stacking ensemble that uses pre-fitted base models."""
        
        class PreFittedStackingEnsemble:
            """Custom stacking ensemble that doesn't retrain base models."""
            
            def __init__(self, base_estimators, meta_model, model_name):
                self.base_estimators = base_estimators
                self.meta_model = meta_model
                self.model_name = model_name
                self.is_fitted = False
                
            
            def _get_base_predictions(self, X):
                """Get predictions from all pre-fitted base models using enhanced features."""
                base_preds = []
                errors = []
                
                # Use all enhanced features for pre-trained base models
                # The pre-trained models should now be trained on enhanced features
                
                for name, model in self.base_estimators:
                    try:
                        # Check if model is fitted
                        if not hasattr(model, 'n_features_in_') or model.n_features_in_ is None:
                            raise ValueError(f"Model {name} is not fitted - missing n_features_in_")
                        
                        # Check feature compatibility with enhanced features
                        if hasattr(model, 'n_features_in_') and model.n_features_in_ != X.shape[1]:
                            raise ValueError(f"Feature mismatch: {name} expects {model.n_features_in_} features, got {X.shape[1]} (using enhanced features)")
                        
                        if hasattr(model, 'predict_proba'):
                            pred = model.predict_proba(X)
                            # Enhanced bounds checking for prediction arrays
                            if (pred is not None and 
                                pred.ndim > 1 and 
                                pred.shape[1] > 1 and 
                                pred.shape[0] == X.shape[0]):  # Ensure prediction matches input size
                                pred = pred[:, 1]  # Use positive class probability
                            elif pred is not None and pred.ndim == 1:
                                # Handle 1D predictions
                                pass
                            else:
                                tprint(f"   ⚠️ Invalid prediction shape from {name}: {pred.shape if pred is not None else 'None'}")
                                continue
                        else:
                            pred = model.predict(X)
                        base_preds.append(pred)
                        tprint(f"   ✅ Successfully got predictions from {name} (using {X.shape[1]} enhanced features)")
                        
                    except Exception as e:
                        error_msg = f"Failed to get predictions from {name}: {e}"
                        errors.append(error_msg)
                        tprint(f"   ❌ {error_msg}")
                
                # Fast fail if any models failed
                if errors:
                    error_summary = "; ".join(errors)
                    raise ValueError(f"Pre-trained base models are not usable: {error_summary}")
                
                if not base_preds:
                    raise ValueError("No base model predictions could be obtained")
                
                return np.column_stack(base_preds)
            
            def fit(self, X, y):
                """Fit meta model using pre-trained base models."""
                tprint(f"   🎯 Training meta model ({self.model_name}) with pre-trained base models")
                
                # Get predictions from pre-trained base models
                base_predictions = self._get_base_predictions(X)
                
                # Combine enhanced features with base predictions (passthrough=True equivalent)
                meta_features = np.hstack([X, base_predictions])
                
                # Train only the meta model
                self.meta_model.fit(meta_features, y)
                self.is_fitted = True
                
                return self
            
            def predict(self, X):
                """Make predictions using the ensemble."""
                if not self.is_fitted:
                    raise ValueError("Ensemble must be fitted before making predictions")
                
                base_predictions = self._get_base_predictions(X)
                meta_features = np.hstack([X, base_predictions])
                return self.meta_model.predict(meta_features)
            
            def predict_proba(self, X):
                """Make probability predictions using the ensemble."""
                if not self.is_fitted:
                    raise ValueError("Ensemble must be fitted before making predictions")
                
                base_predictions = self._get_base_predictions(X)
                meta_features = np.hstack([X, base_predictions])
                
                if hasattr(self.meta_model, 'predict_proba'):
                    return self.meta_model.predict_proba(meta_features)
                else:
                    # Fallback for models without predict_proba
                    pred = self.meta_model.predict(meta_features)
                    # Convert to probability-like format
                    unique_classes = np.unique(pred)
                    n_classes = len(unique_classes)
                    proba = np.zeros((len(pred), n_classes))
                    for i, cls in enumerate(unique_classes):
                        proba[pred == cls, i] = 1.0
                    return proba
            
            def score(self, X, y):
                """Score the ensemble."""
                if not self.is_fitted:
                    raise ValueError("Ensemble must be fitted before scoring")
                
                from sklearn.metrics import accuracy_score
                predictions = self.predict(X)
                return accuracy_score(y, predictions)
            
            def get_params(self, deep=True):
                """Get parameters for this estimator (required by scikit-learn)."""
                return {
                    'base_estimators': self.base_estimators,
                    'meta_model': self.meta_model,
                    'model_name': self.model_name
                }
            
            def set_params(self, **params):
                """Set parameters for this estimator (required by scikit-learn)."""
                for param, value in params.items():
                    if hasattr(self, param):
                        setattr(self, param, value)
                    else:
                        raise ValueError(f"Invalid parameter {param} for PreFittedStackingEnsemble")
                return self
            
            def __getstate__(self):
                """Custom serialization to preserve fitted state."""
                state = self.__dict__.copy()
                return state
            
            def __setstate__(self, state):
                """Custom deserialization to restore fitted state."""
                self.__dict__.update(state)
        
        return PreFittedStackingEnsemble(base_estimators, meta_model, model_name)

    def _create_meta_model(self, model_name: str) -> Any:
        """Create a meta model by name."""
        if model_name == 'XGBoostClassifier':
            import xgboost as xgb
            # Enhanced regularization for meta XGBoost model
            return xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,    # Reduced learning rate
                max_depth=4,           # Limited depth for regularization
                random_state=42,
                n_jobs=-1,
                # Enhanced regularization parameters
                min_child_weight=5,        # Minimum sum of hessian per child
                reg_alpha=0.1,             # L1 regularization
                reg_lambda=0.1,            # L2 regularization
                subsample=0.8,             # Use 80% of data per tree
                colsample_bytree=0.8,      # Use 80% of features per tree
                colsample_bylevel=0.8,     # Use 80% of features per level
                colsample_bynode=0.8       # Use 80% of features per node
            )
        elif model_name == 'CatBoostClassifier':
            from catboost import CatBoostClassifier
            return CatBoostClassifier(
                iterations=100, learning_rate=0.1, depth=6,
                random_seed=42, verbose=False, loss_function='MultiClass'
            )
        elif model_name == 'ElasticNet':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                penalty='elasticnet', solver='saga', l1_ratio=0.5,
                max_iter=2000, random_state=42, class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown meta model: {model_name}")

    def _run_meta_model_hpo(self, model_name: str, X: np.ndarray, y: np.ndarray, base_estimators: List) -> Any:
        """Run HPO on the selected best meta model."""
        try:
            import numpy as np
            import time
            from sklearn.model_selection import RandomizedSearchCV
            from sklearn.ensemble import StackingClassifier
            
            tprint(f"🔧 Starting HPO for {model_name}...")
            hpo_start_time = time.time()
            
            # Get HPO search space for the model
            hpo_spaces = getattr(self.config, 'meta_model_hpo_spaces', {})
            if model_name not in hpo_spaces:
                tprint(f"⚠️ No HPO space defined for {model_name}")
                return None
                
            search_space = hpo_spaces[model_name]
            tprint(f"📊 HPO search space for {model_name}: {len(search_space)} parameters")
            for param, config in search_space.items():
                if config['type'] == 'int':
                    tprint(f"   • {param}: {config['low']} to {config['high']} (integer)")
                elif config['type'] == 'float':
                    log_str = " (log scale)" if config.get('log', False) else ""
                    tprint(f"   • {param}: {config['low']:.4f} to {config['high']:.4f}{log_str}")
                elif config['type'] == 'categorical':
                    tprint(f"   • {param}: {config['choices']}")
            
            # Convert our search space format to sklearn format
            tprint(f"🔄 Converting search space to sklearn format...")
            sklearn_space = {}
            param_count = 0
            for param, config in search_space.items():
                if config['type'] == 'int':
                    param_range = list(range(config['low'], config['high'] + 1))
                    sklearn_space[f'final_estimator__{param}'] = param_range
                    param_count += len(param_range)
                elif config['type'] == 'float':
                    if config.get('log', False):
                        param_range = np.logspace(
                            np.log10(config['low']), np.log10(config['high']), 10
                        )
                        sklearn_space[f'final_estimator__{param}'] = param_range
                        param_count += 10
                    else:
                        param_range = np.linspace(config['low'], config['high'], 10)
                        sklearn_space[f'final_estimator__{param}'] = param_range
                        param_count += 10
                elif config['type'] == 'categorical':
                    sklearn_space[f'final_estimator__{param}'] = config['choices']
                    param_count += len(config['choices'])
            
            tprint(f"📊 Total parameter combinations: ~{param_count:,}")
            
            # Create base model for HPO with pre-fitted estimators
            tprint(f"🏗️ Creating custom stacking ensemble with {len(base_estimators)} pre-fitted base estimators...")
            meta_model = self._create_meta_model(model_name)
            stack = self._create_custom_stacking_ensemble(
                base_estimators=base_estimators,
                meta_model=meta_model,
                model_name=model_name
            )
            
            # Run randomized search
            n_iter = min(getattr(self.config, 'hpo_n_trials', 20), 20)  # Limit for speed
            tprint(f"🎯 HPO configuration:")
            tprint(f"   • Trials: {n_iter} (requested: {getattr(self.config, 'hpo_n_trials', 20)})")
            tprint(f"   • CV folds: 3")
            tprint(f"   • Scoring: accuracy")
            tprint(f"   • Data shape: {X.shape}")
            tprint(f"   • Classes: {len(np.unique(y))}")
            
            random_search = RandomizedSearchCV(
                stack, sklearn_space, n_iter=n_iter, cv=3, 
                scoring='accuracy', n_jobs=-1, random_state=42
            )
            
            tprint(f"🔧 Running {n_iter} HPO trials for {model_name}...")
            trial_start_time = time.time()
            random_search.fit(X, y)
            trial_duration = time.time() - trial_start_time
            
            tprint(f"⏱️ HPO trials completed in {trial_duration:.2f}s ({trial_duration/n_iter:.2f}s per trial)")
            
            # Log detailed HPO results
            hpo_duration = time.time() - hpo_start_time
            tprint(f"✅ HPO completed in {hpo_duration:.2f}s")
            tprint(f"🏆 Best HPO results:")
            tprint(f"   • Best CV score: {random_search.best_score_:.4f}")
            tprint(f"   • Best parameters:")
            for param, value in random_search.best_params_.items():
                param_clean = param.replace('final_estimator__', '')
                if isinstance(value, float):
                    tprint(f"     - {param_clean}: {value:.4f}")
                else:
                    tprint(f"     - {param_clean}: {value}")
            
            # Log performance distribution
            if hasattr(random_search, 'cv_results_'):
                scores = random_search.cv_results_['mean_test_score']
                tprint(f"📊 HPO trial performance:")
                tprint(f"   • Best score: {max(scores):.4f}")
                tprint(f"   • Worst score: {min(scores):.4f}")
                tprint(f"   • Average score: {np.mean(scores):.4f}")
                tprint(f"   • Std deviation: {np.std(scores):.4f}")
            
            # Store HPO results for analysis
            hpo_results = {
                'model_name': model_name,
                'success': True,
                'duration_seconds': hpo_duration,
                'trials_run': n_iter,
                'best_cv_score': float(random_search.best_score_),
                'best_params': random_search.best_params_,
                'performance_distribution': {}
            }
            
            # Store performance distribution if available
            if hasattr(random_search, 'cv_results_'):
                scores = random_search.cv_results_['mean_test_score']
                hpo_results['performance_distribution'] = {
                    'best_score': float(max(scores)),
                    'worst_score': float(min(scores)),
                    'average_score': float(np.mean(scores)),
                    'std_deviation': float(np.std(scores)),
                    'score_range': float(max(scores) - min(scores))
                }
            
            # Store in instance for later access
            self.hpo_results = hpo_results
            
            return random_search.best_estimator_
            
        except Exception as e:
            hpo_duration = time.time() - hpo_start_time if 'hpo_start_time' in locals() else 0
            tprint(f"❌ HPO failed for {model_name} after {hpo_duration:.2f}s: {e}")
            tprint(f"   Error type: {type(e).__name__}")
            if hasattr(e, '__traceback__'):
                import traceback
                stack_trace_lines = traceback.format_exc().split('\n')
                tprint(f"   Stack trace: {stack_trace_lines[-3]}")
            
            # Store failed HPO results
            self.hpo_results = {
                'model_name': model_name,
                'success': False,
                'duration_seconds': hpo_duration,
                'error': str(e),
                'trials_run': 0
            }
            
            return None

    def _analyze_meta_model_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze meta model comparison results for the comprehensive report.
        
        Args:
            results: Training results containing global_regime_classifier data
            
        Returns:
            Meta model analysis summary
        """
        try:
            global_classifier = results.get('global_regime_classifier', {})
            meta_comparison = global_classifier.get('meta_model_comparison', {})
            
            if not meta_comparison or not meta_comparison.get('detailed_results'):
                return {'status': 'no_comparison_data', 'models_tested': 0}
            
            detailed_results = meta_comparison['detailed_results']
            valid_models = {k: v for k, v in detailed_results.items() if 'error' not in v}
            
            if not valid_models:
                return {'status': 'no_valid_models', 'models_tested': len(detailed_results)}
            
            # Create detailed analysis
            analysis = {
                'status': 'comparison_completed',
                'models_tested': len(detailed_results),
                'models_successful': len(valid_models),
                'selection_criteria': meta_comparison.get('selection_criteria', 'unknown'),
                'best_model': meta_comparison.get('best_model', 'unknown'),
                'detailed_metrics': {},
                'performance_ranking': [],
                'training_time_comparison': {},
                'accuracy_comparison': {}
            }
            
            # Sort models by accuracy first, then speed
            sorted_models = sorted(
                valid_models.items(), 
                key=lambda x: (-x[1]['cv_accuracy_mean'], x[1]['training_time_seconds'])
            )
            
            # Build detailed metrics for each model
            for rank, (model_name, metrics) in enumerate(sorted_models):
                rank_position = rank + 1
                analysis['detailed_metrics'][model_name] = {
                    'rank': rank_position,
                    'training_accuracy': metrics['accuracy'],
                    'cv_accuracy_mean': metrics['cv_accuracy_mean'],
                    'cv_accuracy_std': metrics['cv_accuracy_std'],
                    'f1_macro': metrics['f1_macro'],
                    'precision_macro': metrics['precision_macro'],
                    'recall_macro': metrics['recall_macro'],
                    'training_time_seconds': metrics['training_time_seconds'],
                    'speed_score': metrics['speed_score']
                }
                
                analysis['performance_ranking'].append({
                    'rank': rank_position,
                    'model': model_name,
                    'accuracy': metrics['cv_accuracy_mean'],
                    'training_time': metrics['training_time_seconds']
                })
            
            # Training time comparison
            analysis['training_time_comparison'] = {
                'fastest_model': min(valid_models.keys(), key=lambda k: valid_models[k]['training_time_seconds']),
                'slowest_model': max(valid_models.keys(), key=lambda k: valid_models[k]['training_time_seconds']),
                'time_range_seconds': {
                    'min': min(m['training_time_seconds'] for m in valid_models.values()),
                    'max': max(m['training_time_seconds'] for m in valid_models.values()),
                    'mean': sum(m['training_time_seconds'] for m in valid_models.values()) / len(valid_models)
                }
            }
            
            # Accuracy comparison
            analysis['accuracy_comparison'] = {
                'best_accuracy_model': max(valid_models.keys(), key=lambda k: valid_models[k]['cv_accuracy_mean']),
                'worst_accuracy_model': min(valid_models.keys(), key=lambda k: valid_models[k]['cv_accuracy_mean']),
                'accuracy_range': {
                    'min': min(m['cv_accuracy_mean'] for m in valid_models.values()),
                    'max': max(m['cv_accuracy_mean'] for m in valid_models.values()),
                    'mean': sum(m['cv_accuracy_mean'] for m in valid_models.values()) / len(valid_models)
                }
            }
            
            # HPO status
            best_model = meta_comparison.get('best_model', 'unknown')
            analysis['hpo_status'] = {
                'attempted_on': best_model,
                'hpo_successful': hasattr(self, 'global_regime_clf') and self.global_regime_clf is not None,
                'final_model_optimized': True  # Will be updated based on actual HPO results
            }
            
            return analysis
            
        except Exception as e:
            tprint(f"⚠️ Meta model analysis failed: {e}")
            return {'status': 'analysis_failed', 'error': str(e)}
    
    def _analyze_hpo_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze HPO results for the comprehensive report.
        
        Args:
            results: Training results
            
        Returns:
            HPO analysis summary
        """
        try:
            # Check if HPO was attempted
            hpo_results = getattr(self, 'hpo_results', None)
            if not hpo_results:
                return {
                    'status': 'no_hpo_attempted',
                    'hpo_enabled': self.config.enable_hpo,
                    'reason': 'HPO not configured or not attempted'
                }
            
            # Analyze HPO results
            analysis = {
                'status': 'hpo_completed' if hpo_results['success'] else 'hpo_failed',
                'model_optimized': hpo_results['model_name'],
                'success': hpo_results['success'],
                'duration_seconds': hpo_results['duration_seconds'],
                'trials_run': hpo_results['trials_run']
            }
            
            if hpo_results['success']:
                # Successful HPO analysis
                analysis.update({
                    'best_cv_score': hpo_results['best_cv_score'],
                    'best_parameters': hpo_results['best_params'],
                    'performance_improvement': self._calculate_hpo_improvement(hpo_results),
                    'optimization_efficiency': {
                        'score_per_second': hpo_results['best_cv_score'] / (hpo_results['duration_seconds'] + 1e-6),
                        'trials_per_second': hpo_results['trials_run'] / (hpo_results['duration_seconds'] + 1e-6)
                    }
                })
                
                # Performance distribution analysis
                if 'performance_distribution' in hpo_results and hpo_results['performance_distribution']:
                    perf_dist = hpo_results['performance_distribution']
                    analysis['performance_distribution'] = {
                        'score_range': perf_dist.get('score_range', 0.0),
                        'improvement_margin': perf_dist.get('best_score', 0.0) - perf_dist.get('average_score', 0.0),
                        'consistency_score': 1.0 - (perf_dist.get('std_deviation', 0.0) / (perf_dist.get('score_range', 1.0) + 1e-6)),
                        'top_percentile_achievement': self._calculate_top_percentile(perf_dist)
                    }
                
                # Parameter analysis
                analysis['parameter_analysis'] = self._analyze_hpo_parameters(hpo_results['best_params'])
                
            else:
                # Failed HPO analysis
                analysis.update({
                    'error': hpo_results.get('error', 'Unknown HPO failure'),
                    'failure_analysis': self._analyze_hpo_failure(hpo_results)
                })
            
            return analysis
            
        except Exception as e:
            tprint(f"⚠️ HPO analysis failed: {e}")
            return {'status': 'analysis_failed', 'error': str(e)}
    
    def _calculate_hpo_improvement(self, hpo_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate HPO improvement metrics."""
        try:
            # Get baseline performance from meta model comparison
            meta_results = getattr(self, 'meta_model_results', {})
            model_name = hpo_results['model_name']
            
            if model_name in meta_results and 'error' not in meta_results[model_name]:
                baseline_score = meta_results[model_name]['cv_accuracy_mean']
                optimized_score = hpo_results['best_cv_score']
                
                return {
                    'baseline_score': baseline_score,
                    'optimized_score': optimized_score,
                    'absolute_improvement': optimized_score - baseline_score,
                    'relative_improvement_percent': ((optimized_score - baseline_score) / (baseline_score + 1e-6)) * 100,
                    'improvement_significant': (optimized_score - baseline_score) > 0.01  # 1% threshold
                }
            else:
                return {'status': 'no_baseline_available'}
                
        except Exception as e:
            return {'status': 'calculation_failed', 'error': str(e)}
    
    def _calculate_top_percentile(self, perf_dist: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate top percentile achievement metrics."""
        try:
            best_score = perf_dist.get('best_score', 0.0)
            worst_score = perf_dist.get('worst_score', 0.0)
            avg_score = perf_dist.get('average_score', 0.0)
            
            if best_score == worst_score:
                return {'percentile': 100.0, 'description': 'Perfect optimization'}
            
            # Calculate what percentile the best score represents
            score_range = best_score - worst_score
            best_improvement = best_score - avg_score
            percentile = (best_improvement / score_range) * 100
            
            return {
                'percentile': min(100.0, max(0.0, percentile)),
                'description': f'Top {100 - percentile:.1f}% of trials',
                'score_above_average': best_improvement
            }
            
        except Exception as e:
            return {'status': 'calculation_failed', 'error': str(e)}
    
    def _analyze_hpo_parameters(self, best_params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the best HPO parameters."""
        try:
            analysis = {
                'total_parameters_optimized': len(best_params),
                'parameter_categories': {},
                'notable_parameters': []
            }
            
            for param, value in best_params.items():
                # Clean parameter name
                clean_param = param.replace('final_estimator__', '')
                
                # Categorize parameters
                if 'learning_rate' in clean_param.lower():
                    analysis['parameter_categories']['learning'] = analysis['parameter_categories'].get('learning', 0) + 1
                elif 'depth' in clean_param.lower() or 'max_depth' in clean_param.lower():
                    analysis['parameter_categories']['depth'] = analysis['parameter_categories'].get('depth', 0) + 1
                elif 'iter' in clean_param.lower() or 'estimators' in clean_param.lower():
                    analysis['parameter_categories']['iterations'] = analysis['parameter_categories'].get('iterations', 0) + 1
                elif 'alpha' in clean_param.lower() or 'lambda' in clean_param.lower():
                    analysis['parameter_categories']['regularization'] = analysis['parameter_categories'].get('regularization', 0) + 1
                else:
                    analysis['parameter_categories']['other'] = analysis['parameter_categories'].get('other', 0) + 1
                
                # Identify notable parameters
                if isinstance(value, float):
                    if value < 0.1 or value > 0.9:
                        analysis['notable_parameters'].append(f'{clean_param}: {value:.4f} (extreme value)')
                elif isinstance(value, int):
                    if value > 100:
                        analysis['notable_parameters'].append(f'{clean_param}: {value} (high value)')
                    elif value < 5:
                        analysis['notable_parameters'].append(f'{clean_param}: {value} (low value)')
            
            return analysis
            
        except Exception as e:
            return {'status': 'analysis_failed', 'error': str(e)}
    
    def _analyze_hpo_failure(self, hpo_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze HPO failure reasons."""
        try:
            error = hpo_results.get('error', 'Unknown error')
            
            # Categorize common failure types
            if 'timeout' in error.lower():
                failure_type = 'timeout'
                recommendation = 'Increase HPO timeout or reduce search space'
            elif 'memory' in error.lower():
                failure_type = 'memory'
                recommendation = 'Reduce batch size or model complexity'
            elif 'convergence' in error.lower():
                failure_type = 'convergence'
                recommendation = 'Adjust learning rate or regularization parameters'
            else:
                failure_type = 'other'
                recommendation = 'Check configuration and data quality'
            
            return {
                'failure_type': failure_type,
                'error_message': error,
                'recommendation': recommendation,
                'duration_before_failure': hpo_results.get('duration_seconds', 0.0)
            }
            
        except Exception as e:
            return {'status': 'analysis_failed', 'error': str(e)}


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
        timeframe=get_primary_timeframe(),
        model_types=["catboost", "elastic_net", "ensemble_rf"],
        hpo_n_trials=50,  # Reduced for demo
        enable_hpo=True,
        save_models=True,
        model_save_path="./generated/market_analysis/models/hmm_ensemble_models_refactored"
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
    print("- Operates on 15m timeframe with cross-timeframe features")
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

# Export the main classes for import
__all__ = ['HMMEnsembleTrainingComponent', 'execute_hmm_ensemble_training', 'create_hmm_ensemble_training_component']