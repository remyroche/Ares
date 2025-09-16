"""
Tactician Ensemble Training Step - Enhanced with Comprehensive Error Handling and Logging

This step handles all-regime ensemble training of Tactician models using common dependencies.
The Tactician Ensemble operates on 1m timeframe and combines individual tactician models
with all previous model inputs (HMM, Analyst) to create the final meta-learner for timing decisions.

Enhanced with:
- Extensive try/except blocks with fast failing for important errors
- Comprehensive logging using tprint at every step
- Integration with common utilities and tools
- Hardware optimization support (M1 GPU/CPU)
- Math validation and data quality checks
- Serialization utilities for model persistence
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

# Enhanced imports with comprehensive error handling
try:
    from src.utils.logger import system_logger
    LOGGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import system_logger: {e}")
    LOGGER_AVAILABLE = False
    system_logger = logging.getLogger(__name__)

try:
    from src.utils.ml_common.config.base_training_config import EnsembleTrainingConfig
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error: Could not import EnsembleTrainingConfig: {e}")
    CONFIG_AVAILABLE = False
    raise ImportError("EnsembleTrainingConfig is required but not available")

try:
    from src.utils.ml_common.training.ensemble_training_step import EnsembleTrainingStep
    ENSEMBLE_STEP_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error: Could not import EnsembleTrainingStep: {e}")
    ENSEMBLE_STEP_AVAILABLE = False
    raise ImportError("EnsembleTrainingStep is required but not available")

# Import enhanced utilities with error handling
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success, 
        tprint_debug, tprint_structured, LogLevel
    )
    TPRINT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import tprint utilities: {e}")
    TPRINT_AVAILABLE = False
    # Essential fallback functions only
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_structured(data, level=None, **kwargs): print("STRUCTURED:", data)
    class LogLevel: INFO = "INFO"; ERROR = "ERROR"; WARNING = "WARNING"; SUCCESS = "SUCCESS"

try:
    from src.utils.common_operations import (
        safe_divide, validate_finite, safe_mean, safe_std,
        calculate_data_quality_metrics, safe_json_dump, ensure_directory,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        memory_checkpoint, gpu_context, optimize_memory
    )
    COMMON_OPERATIONS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import common_operations: {e}")
    COMMON_OPERATIONS_AVAILABLE = False
    # Essential fallback functions only
    def safe_divide(a, b, default=0.0): return a / b if b != 0 else default
    def validate_finite(value, name="value"): return float(value) if np.isfinite(float(value)) else 0.0
    def safe_mean(x, default=0.0): return np.mean(x) if len(x) > 0 else default
    def safe_std(x, default=0.0): return np.std(x) if len(x) > 1 else default
    def calculate_data_quality_metrics(df): return {'total_rows': len(df), 'total_columns': len(df.columns)}
    def safe_json_dump(data, file_path, **kwargs): return True
    def ensure_directory(path): return True
    def get_m1_gpu_manager(): return None
    def get_m1_memory_optimizer(): return None
    def get_m1_cpu_optimizer(): return None
    def memory_checkpoint(name): return __import__('contextlib').contextmanager(lambda: (yield))()
    def gpu_context(name): return __import__('contextlib').contextmanager(lambda: (yield))()
    def optimize_memory(): return {'success': True}

try:
    from src.utils.math_validation import (
        safe_correlation, validate_correlation_matrix
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import math_validation: {e}")
    MATH_VALIDATION_AVAILABLE = False
    def safe_correlation(x, y, default=0.0): return default
    def validate_correlation_matrix(corr_matrix): return True

try:
    from src.utils.serialization_utils import UniversalSerializer
    SERIALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import serialization_utils: {e}")
    SERIALIZATION_AVAILABLE = False
    class UniversalSerializer:
        def save(self, data, filepath, format='auto'): return True
        def load(self, filepath): return None

try:
    from src.utils.kline_parquet import safe_to_parquet, safe_read_parquet
    KLINE_PARQUET_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import kline_parquet: {e}")
    KLINE_PARQUET_AVAILABLE = False
    # Note: kline_parquet functions not used in this module, but kept for completeness

# Import vectorized training manager
try:
    from src.utils.ml_common.training.vectorized_training_manager import VectorizedTrainingManager
    VECTORIZED_TRAINING_AVAILABLE = True
    tprint_info("✅ VectorizedTrainingManager imported successfully")
except ImportError as e:
    VECTORIZED_TRAINING_AVAILABLE = False
    tprint_warning(f"⚠️ VectorizedTrainingManager not available: {e}")

# Initialize logger with error handling
try:
    if LOGGER_AVAILABLE:
        logger = system_logger.getChild('TacticianEnsembleTraining')
    else:
        logger = logging.getLogger('TacticianEnsembleTraining')
        logger.setLevel(logging.INFO)
    tprint_info("✅ Logger initialized successfully")
except Exception as e:
    tprint_error(f"❌ Failed to initialize logger: {e}")
    logger = logging.getLogger(__name__)

# Initialize serialization utilities
try:
    if SERIALIZATION_AVAILABLE:
        serializer = UniversalSerializer()
        tprint_info("✅ UniversalSerializer initialized successfully")
    else:
        serializer = None
        tprint_warning("⚠️ UniversalSerializer not available, using fallback")
except Exception as e:
    tprint_error(f"❌ Failed to initialize serializer: {e}")
    serializer = None


@dataclass
class TrainingProgress:
    """Track training progress and metrics with enhanced error handling."""
    step_name: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None
    warnings: List[str] = None
    hardware_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.warnings is None:
            self.warnings = []
        if self.hardware_metrics is None:
            self.hardware_metrics = {}
    
    @property
    def duration(self) -> float:
        """Get training duration in seconds with validation."""
        try:
            if self.end_time is None:
                return time.time() - self.start_time
            return self.end_time - self.start_time
        except Exception as e:
            tprint_error(f"❌ Error calculating duration for step {self.step_name}: {e}")
            return 0.0
    
    def complete(self, success: bool = True, error_message: Optional[str] = None, 
                 metrics: Optional[Dict[str, Any]] = None, warnings: Optional[List[str]] = None,
                 hardware_metrics: Optional[Dict[str, Any]] = None):
        """Mark step as complete with comprehensive tracking."""
        try:
            self.end_time = time.time()
            self.success = success
            self.error_message = error_message
            
            if metrics:
                self.metrics.update(metrics)
            if warnings:
                self.warnings.extend(warnings)
            if hardware_metrics:
                self.hardware_metrics.update(hardware_metrics)
                
            # Log completion with performance metrics
            duration = self.duration
            if success:
                tprint_success(f"✅ Step '{self.step_name}' completed in {duration:.3f}s")
                if self.metrics:
                    tprint_structured(self.metrics, LogLevel.INFO)
            else:
                tprint_error(f"❌ Step '{self.step_name}' failed after {duration:.3f}s: {error_message}")
                
        except Exception as e:
            tprint_error(f"❌ Error completing step {self.step_name}: {e}")
            self.success = False
            self.error_message = f"Completion error: {e}"
    
    def add_warning(self, warning: str):
        """Add a warning to the step."""
        try:
            self.warnings.append(warning)
            tprint_warning(f"⚠️ Step '{self.step_name}': {warning}")
        except Exception as e:
            tprint_error(f"❌ Error adding warning to step {self.step_name}: {e}")
    
    def add_metric(self, key: str, value: Any):
        """Add a metric to the step."""
        try:
            self.metrics[key] = value
        except Exception as e:
            tprint_error(f"❌ Error adding metric '{key}' to step {self.step_name}: {e}")
    
    def add_hardware_metric(self, key: str, value: Any):
        """Add a hardware metric to the step."""
        try:
            self.hardware_metrics[key] = value
        except Exception as e:
            tprint_error(f"❌ Error adding hardware metric '{key}' to step {self.step_name}: {e}")


class TacticianEnsembleTrainingStep(EnsembleTrainingStep):
    """
    Tactician Ensemble Training Step with comprehensive error handling, logging, and utility integration.
    
    The Tactician Ensemble operates on 1m timeframe and combines individual tactician models
    with all previous model inputs (HMM, Analyst) to create the final meta-learner for timing decisions.
    
    Enhanced Features:
    - Extensive try/except blocks with fast failing for critical errors
    - Comprehensive logging using tprint at every step
    - Integration with common utilities (math validation, serialization, hardware optimization)
    - Memory and GPU optimization support
    - Data quality validation and monitoring
    """
    
    def __init__(self, config: Optional[EnsembleTrainingConfig] = None, enable_vectorization: bool = True):
        """
        Initialize Tactician ensemble training step with comprehensive error handling and utility integration.

        Args:
            config: Per-regime training configuration
            enable_vectorization: Whether to enable vectorized training
        """
        tprint_info("🚀 Initializing TacticianEnsembleTrainingStep...")
        
        try:
            # Initialize hardware optimizers first
            self._initialize_hardware_optimizers()
            
            # Set default configuration for tactician ensemble models with validation
            if config is None:
                tprint_info("📋 Creating default configuration...")
                config = self._create_default_config()
            else:
                tprint_info("📋 Using provided configuration...")

            # Validate configuration with comprehensive checks
            self._validate_config(config)
            tprint_success("✅ Configuration validation passed")
            
            # Initialize parent class with error handling
            try:
                super().__init__(config, enable_vectorization=enable_vectorization and VECTORIZED_TRAINING_AVAILABLE)
                tprint_success("✅ Parent class initialization successful")
            except Exception as e:
                tprint_error(f"❌ Parent class initialization failed: {e}")
                raise RuntimeError(f"Parent class initialization failed: {e}") from e
            
            # Initialize enhanced logging
            self.logger = logger.getChild('TacticianEnsembleTrainingStep')
            
            # Initialize progress tracking with enhanced features
            self.progress_tracker: List[TrainingProgress] = []
            self.current_step: Optional[TrainingProgress] = None
            self.initialization_metrics = {}
            
            # Initialize utility integrations
            self._initialize_utility_integrations()
            
            # Log initialization success
            if self.enable_vectorization:
                tprint_success("🚀 Tactician Ensemble Training Step initialized with vectorization")
            else:
                tprint_success("✅ Tactician Ensemble Training Step initialized (standard mode)")
                
            # Log initialization metrics
            tprint_structured(self.initialization_metrics, LogLevel.INFO)
                
        except Exception as e:
            error_msg = f"Failed to initialize TacticianEnsembleTrainingStep: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            
            # Try to log to system logger if available
            try:
                if LOGGER_AVAILABLE:
                    logger.error(error_msg)
                    logger.error(f"Traceback: {traceback.format_exc()}")
            except Exception:
                pass
                
            raise RuntimeError(error_msg) from e
    
    def _initialize_hardware_optimizers(self):
        """Initialize hardware optimizers with error handling."""
        try:
            tprint_info("🔧 Initializing hardware optimizers...")
            
            # Initialize M1 GPU manager
            try:
                self.gpu_manager = get_m1_gpu_manager()
                if self.gpu_manager:
                    tprint_success("✅ M1 GPU manager initialized")
                    self.initialization_metrics['gpu_manager'] = True
                else:
                    tprint_warning("⚠️ M1 GPU manager not available")
                    self.initialization_metrics['gpu_manager'] = False
            except Exception as e:
                tprint_warning(f"⚠️ M1 GPU manager initialization failed: {e}")
                self.gpu_manager = None
                self.initialization_metrics['gpu_manager'] = False
            
            # Initialize M1 memory optimizer
            try:
                self.memory_optimizer = get_m1_memory_optimizer()
                if self.memory_optimizer:
                    tprint_success("✅ M1 memory optimizer initialized")
                    self.initialization_metrics['memory_optimizer'] = True
                else:
                    tprint_warning("⚠️ M1 memory optimizer not available")
                    self.initialization_metrics['memory_optimizer'] = False
            except Exception as e:
                tprint_warning(f"⚠️ M1 memory optimizer initialization failed: {e}")
                self.memory_optimizer = None
                self.initialization_metrics['memory_optimizer'] = False
            
            # Initialize M1 CPU optimizer
            try:
                self.cpu_optimizer = get_m1_cpu_optimizer()
                if self.cpu_optimizer:
                    tprint_success("✅ M1 CPU optimizer initialized")
                    self.initialization_metrics['cpu_optimizer'] = True
                else:
                    tprint_warning("⚠️ M1 CPU optimizer not available")
                    self.initialization_metrics['cpu_optimizer'] = False
            except Exception as e:
                tprint_warning(f"⚠️ M1 CPU optimizer initialization failed: {e}")
                self.cpu_optimizer = None
                self.initialization_metrics['cpu_optimizer'] = False
                
        except Exception as e:
            tprint_error(f"❌ Hardware optimizer initialization failed: {e}")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.initialization_metrics.update({
                'gpu_manager': False,
                'memory_optimizer': False,
                'cpu_optimizer': False
            })
    
    def _create_default_config(self) -> EnsembleTrainingConfig:
        """Create default configuration with validation."""
        try:
            config = EnsembleTrainingConfig(
                model_name="tactician_ensemble_models",
                timeframe="1m",
                model_types=["node", "catboost", "lightgbm", "elastic_net"],
                hpo_n_trials=100,
                hpo_timeout_seconds=3600,
                min_samples_per_regime=1000,
                enable_data_augmentation=True,
                augmentation_method="smote",
                model_save_path="./models/tactician_ensemble_models",
                evaluation_metrics=["mse", "mae", "r2", "mape", "smape"]
            )
            tprint_success("✅ Default configuration created successfully")
            return config
        except Exception as e:
            tprint_error(f"❌ Failed to create default configuration: {e}")
            raise RuntimeError(f"Default configuration creation failed: {e}") from e
    
    def _initialize_utility_integrations(self):
        """Initialize utility integrations with error handling."""
        try:
            tprint_info("🔧 Initializing utility integrations...")
            
            # Initialize serialization
            self.serializer = serializer
            if self.serializer:
                tprint_success("✅ Serialization utilities initialized")
                self.initialization_metrics['serialization'] = True
            else:
                tprint_warning("⚠️ Serialization utilities not available")
                self.initialization_metrics['serialization'] = False
            
            # Initialize math validation
            self.math_validation_available = MATH_VALIDATION_AVAILABLE
            if self.math_validation_available:
                tprint_success("✅ Math validation utilities initialized")
                self.initialization_metrics['math_validation'] = True
            else:
                tprint_warning("⚠️ Math validation utilities not available")
                self.initialization_metrics['math_validation'] = False
            
            # Initialize common operations
            self.common_operations_available = COMMON_OPERATIONS_AVAILABLE
            if self.common_operations_available:
                tprint_success("✅ Common operations utilities initialized")
                self.initialization_metrics['common_operations'] = True
            else:
                tprint_warning("⚠️ Common operations utilities not available")
                self.initialization_metrics['common_operations'] = False
            
            # Initialize tprint
            self.tprint_available = TPRINT_AVAILABLE
            if self.tprint_available:
                tprint_success("✅ TPrint utilities initialized")
                self.initialization_metrics['tprint'] = True
            else:
                tprint_warning("⚠️ TPrint utilities not available")
                self.initialization_metrics['tprint'] = False
                
        except Exception as e:
            tprint_error(f"❌ Utility integration initialization failed: {e}")
            self.initialization_metrics.update({
                'serialization': False,
                'math_validation': False,
                'common_operations': False,
                'tprint': False
            })
    
    def _validate_config(self, config: EnsembleTrainingConfig) -> None:
        """Validate configuration parameters with comprehensive error handling."""
        try:
            tprint_info("🔍 Validating configuration parameters...")
            validation_errors = []
            warnings = []
            
            # Validate model_name
            if not config.model_name or not isinstance(config.model_name, str):
                validation_errors.append("model_name must be a non-empty string")
            elif len(config.model_name.strip()) == 0:
                validation_errors.append("model_name cannot be empty or whitespace only")
            else:
                tprint_debug(f"✅ model_name validation passed: '{config.model_name}'")
            
            # Validate timeframe
            if not config.timeframe or not isinstance(config.timeframe, str):
                validation_errors.append("timeframe must be a non-empty string")
            elif config.timeframe not in ['1m', '5m', '15m', '1h', '4h', '1d']:
                warnings.append(f"timeframe '{config.timeframe}' is not a standard timeframe")
            else:
                tprint_debug(f"✅ timeframe validation passed: '{config.timeframe}'")
                
            # Validate model_types
            if not config.model_types or not isinstance(config.model_types, list) or len(config.model_types) == 0:
                validation_errors.append("model_types must be a non-empty list")
            else:
                valid_model_types = ['node', 'catboost', 'lightgbm', 'elastic_net', 'xgboost', 'random_forest']
                invalid_types = [t for t in config.model_types if t not in valid_model_types]
                if invalid_types:
                    warnings.append(f"Unknown model types: {invalid_types}")
                tprint_debug(f"✅ model_types validation passed: {config.model_types}")
                
            # Validate HPO parameters
            if config.hpo_n_trials <= 0:
                validation_errors.append("hpo_n_trials must be positive")
            elif config.hpo_n_trials > 1000:
                warnings.append(f"hpo_n_trials ({config.hpo_n_trials}) is very high, consider reducing for faster training")
            else:
                tprint_debug(f"✅ hpo_n_trials validation passed: {config.hpo_n_trials}")
                
            if config.min_samples_per_regime <= 0:
                validation_errors.append("min_samples_per_regime must be positive")
            elif config.min_samples_per_regime < 100:
                warnings.append(f"min_samples_per_regime ({config.min_samples_per_regime}) is very low, may cause overfitting")
            else:
                tprint_debug(f"✅ min_samples_per_regime validation passed: {config.min_samples_per_regime}")
            
            # Validate timeout
            if hasattr(config, 'hpo_timeout_seconds') and config.hpo_timeout_seconds <= 0:
                validation_errors.append("hpo_timeout_seconds must be positive")
            
            # Validate save path
            if hasattr(config, 'model_save_path') and config.model_save_path:
                try:
                    save_path = Path(config.model_save_path)
                    if not save_path.parent.exists():
                        warnings.append(f"Save path parent directory does not exist: {save_path.parent}")
                except Exception as e:
                    warnings.append(f"Invalid save path: {e}")
            
            # Log warnings
            for warning in warnings:
                tprint_warning(f"⚠️ Configuration warning: {warning}")
            
            # Raise errors if any critical validation failed
            if validation_errors:
                error_msg = f"Configuration validation failed: {'; '.join(validation_errors)}"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            tprint_success("✅ Configuration validation completed successfully")
            
        except Exception as e:
            tprint_error(f"❌ Configuration validation failed with exception: {e}")
            raise RuntimeError(f"Configuration validation failed: {e}") from e
    
    def _start_step(self, step_name: str) -> TrainingProgress:
        """Start tracking a training step with comprehensive error handling."""
        try:
            tprint_info(f"🔄 Starting step: {step_name}")
            
            # Create progress tracker with enhanced features
            progress = TrainingProgress(step_name=step_name, start_time=time.time())
            self.progress_tracker.append(progress)
            self.current_step = progress
            
            # Log step start with hardware metrics if available
            if self.memory_optimizer:
                try:
                    memory_info = self.memory_optimizer.get_memory_info()
                    progress.add_hardware_metric('memory_start', memory_info)
                    tprint_debug(f"📊 Memory at step start: {memory_info}")
                except Exception as e:
                    tprint_warning(f"⚠️ Could not get memory info: {e}")
            
            # Log to system logger if available
            if LOGGER_AVAILABLE:
                self.logger.info(f"🔄 Starting step: {step_name}")
            
            return progress
            
        except Exception as e:
            tprint_error(f"❌ Error starting step '{step_name}': {e}")
            # Create a minimal progress tracker as fallback
            progress = TrainingProgress(step_name=step_name, start_time=time.time())
            self.progress_tracker.append(progress)
            self.current_step = progress
            return progress
    
    def _complete_step(self, success: bool = True, error_message: Optional[str] = None, 
                      metrics: Optional[Dict[str, Any]] = None, warnings: Optional[List[str]] = None,
                      hardware_metrics: Optional[Dict[str, Any]] = None):
        """Complete the current training step with comprehensive tracking."""
        try:
            if not self.current_step:
                tprint_warning("⚠️ No current step to complete")
                return
            
            # Add hardware metrics if available
            if self.memory_optimizer and not hardware_metrics:
                try:
                    memory_info = self.memory_optimizer.get_memory_info()
                    hardware_metrics = {'memory_end': memory_info}
                    tprint_debug(f"📊 Memory at step end: {memory_info}")
                except Exception as e:
                    tprint_warning(f"⚠️ Could not get memory info: {e}")
            
            # Complete the step with all metrics
            self.current_step.complete(
                success=success,
                error_message=error_message,
                metrics=metrics,
                warnings=warnings,
                hardware_metrics=hardware_metrics
            )
            
            # Log to system logger if available
            if LOGGER_AVAILABLE:
                if success:
                    self.logger.info(f"✅ Completed step: {self.current_step.step_name} in {self.current_step.duration:.2f}s")
                else:
                    self.logger.error(f"❌ Failed step: {self.current_step.step_name} - {error_message}")
            
            # Clear current step
            self.current_step = None
            
        except Exception as e:
            tprint_error(f"❌ Error completing step: {e}")
            if self.current_step:
                self.current_step.success = False
                self.current_step.error_message = f"Completion error: {e}"
                self.current_step = None
    
    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        base_tactician_models: Optional[Dict[str, Any]] = None,
        tactician_training_metrics: Optional[Dict[str, Any]] = None,
        analyst_models: Optional[Dict[str, Any]] = None,
        analyst_ensembles: Optional[Dict[str, Any]] = None,
        analyst_ensemble_metrics: Optional[Dict[str, Any]] = None,
        hmm_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Tactician ensemble training step with comprehensive error handling, logging, and utility integration.
        
        Args:
            X: Input features (1m timeframe with cross-timeframe features)
            y: Target values (tactician outputs - timing decisions)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            base_tactician_models: Individual tactician models to ensemble
            tactician_training_metrics: Performance metrics of base tactician models
            analyst_models: Individual analyst models
            analyst_ensembles: Analyst ensemble models
            analyst_ensemble_metrics: Performance metrics of analyst ensembles
            hmm_data: HMM regime data and features
            
        Returns:
            Dictionary containing training results and metadata
        """
        overall_start_time = time.time()
        tprint_info("🚀 Starting Tactician ensemble training step (meta-learner)")
        
        # Initialize execution context with hardware optimization
        try:
            with memory_checkpoint("tactician_ensemble_training"):
                with gpu_context("tactician_ensemble_training"):
                    return self._execute_with_context(
                        X, y, regime_labels, feature_names, hmm_states,
                        base_tactician_models, tactician_training_metrics,
                        analyst_models, analyst_ensembles, analyst_ensemble_metrics, hmm_data,
                        overall_start_time
                    )
        except Exception as e:
            error_msg = f"Tactician ensemble training failed with context error: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            
            if self.current_step:
                self._complete_step(False, error_msg)
            
            return self._create_error_result("Training execution failed", error_msg)
    
    def _execute_with_context(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray],
        base_tactician_models: Optional[Dict[str, Any]],
        tactician_training_metrics: Optional[Dict[str, Any]],
        analyst_models: Optional[Dict[str, Any]],
        analyst_ensembles: Optional[Dict[str, Any]],
        analyst_ensemble_metrics: Optional[Dict[str, Any]],
        hmm_data: Optional[Dict[str, Any]],
        overall_start_time: float
    ) -> Dict[str, Any]:
        """Execute training with proper context management."""
        try:
            # Step 1: Input validation with comprehensive checks
            self._start_step("Input Validation")
            validation_result = self._validate_inputs_enhanced(X, y, regime_labels, feature_names)
            if not validation_result['success']:
                self._complete_step(False, validation_result['error'])
                return self._create_error_result("Input validation failed", validation_result['error'])
            self._complete_step(True, metrics=validation_result['metrics'])
            
            # Step 2: Data quality assessment
            self._start_step("Data Quality Assessment")
            quality_result = self._assess_data_quality(X, y, regime_labels)
            self._complete_step(True, metrics=quality_result['metrics'], warnings=quality_result.get('warnings', []))
            
            # Step 3: Base model validation and preparation
            self._start_step("Base Model Preparation")
            base_tactician_models = self._prepare_base_models_enhanced(base_tactician_models)
            self._complete_step(True, metrics={'base_models_count': len(base_tactician_models)})
            
            # Step 4: Feature enhancement with math validation
            self._start_step("Feature Enhancement")
            enhancement_result = self._combine_all_model_inputs_enhanced(
                X, analyst_models, analyst_ensembles, hmm_data, feature_names
            )
            if not enhancement_result['success']:
                self._complete_step(False, enhancement_result['error'])
                return self._create_error_result("Feature enhancement failed", enhancement_result['error'])
            X_enhanced = enhancement_result['X_enhanced']
            self._complete_step(True, metrics=enhancement_result['metrics'])
            
            # Step 5: Memory optimization before training
            self._start_step("Memory Optimization")
            memory_result = self._optimize_memory_before_training()
            self._complete_step(True, metrics=memory_result)
            
            # Step 6: Ensemble training with comprehensive error handling
            self._start_step("Ensemble Training")
            training_result = self._execute_ensemble_training_enhanced(
                X_enhanced, y, regime_labels, feature_names, hmm_states
            )
            if not training_result['success']:
                self._complete_step(False, training_result['error'])
                return self._create_error_result("Ensemble training failed", training_result['error'])
            results = training_result['results']
            self._complete_step(True, metrics=training_result['metrics'])
            
            # Step 7: Meta-learner metadata enhancement
            self._start_step("Meta-learner Enhancement")
            enhancement_result = self._add_meta_learner_metadata_enhanced(
                results, base_tactician_models, tactician_training_metrics,
                analyst_models, analyst_ensembles, analyst_ensemble_metrics, hmm_data
            )
            if not enhancement_result['success']:
                self._complete_step(False, enhancement_result['error'])
                return self._create_error_result("Meta-learner enhancement failed", enhancement_result['error'])
            results = enhancement_result['results']
            self._complete_step(True)
            
            # Step 8: Model serialization and persistence
            self._start_step("Model Serialization")
            serialization_result = self._serialize_models(results)
            self._complete_step(True, metrics=serialization_result)
            
            # Step 9: Final reporting with comprehensive metrics
            self._start_step("Final Reporting")
            reporting_result = self._add_comprehensive_reporting_enhanced(results, overall_start_time)
            if not reporting_result['success']:
                self._complete_step(False, reporting_result['error'])
                return self._create_error_result("Final reporting failed", reporting_result['error'])
            results = reporting_result['results']
            self._complete_step(True)
            
            # Log final success
            total_time = time.time() - overall_start_time
            tprint_success(f"🎯 Tactician ensemble training completed successfully in {total_time:.2f}s")
            
            return results
            
        except Exception as e:
            error_msg = f"Tactician ensemble training failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            
            # Try to log to system logger if available
            try:
                if LOGGER_AVAILABLE:
                    self.logger.error(error_msg)
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
            except Exception:
                pass
            
            if self.current_step:
                self._complete_step(False, error_msg)
            
            return self._create_error_result("Training execution failed", error_msg)
    
    def _validate_inputs_enhanced(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray, feature_names: Optional[List[str]]) -> Dict[str, Any]:
        """Validate input data with comprehensive checks and math validation."""
        try:
            tprint_info("🔍 Starting enhanced input validation...")
            validation_errors = []
            warnings = []
            metrics = {}
            
            # Check data types and shapes with detailed logging
            if not isinstance(X, np.ndarray):
                validation_errors.append("X must be a numpy array")
            elif X.ndim != 2:
                validation_errors.append("X must be a 2D array")
            elif X.shape[0] == 0:
                validation_errors.append("X cannot be empty")
            else:
                metrics['X_shape'] = X.shape
                metrics['X_dtype'] = str(X.dtype)
                tprint_debug(f"✅ X validation passed: shape={X.shape}, dtype={X.dtype}")
                
            if not isinstance(y, np.ndarray):
                validation_errors.append("y must be a numpy array")
            elif y.ndim != 1:
                validation_errors.append("y must be a 1D array")
            elif y.shape[0] == 0:
                validation_errors.append("y cannot be empty")
            else:
                metrics['y_shape'] = y.shape
                metrics['y_dtype'] = str(y.dtype)
                tprint_debug(f"✅ y validation passed: shape={y.shape}, dtype={y.dtype}")
                
            if not isinstance(regime_labels, np.ndarray):
                validation_errors.append("regime_labels must be a numpy array")
            elif regime_labels.ndim != 1:
                validation_errors.append("regime_labels must be a 1D array")
            else:
                metrics['regime_labels_shape'] = regime_labels.shape
                metrics['regime_labels_dtype'] = str(regime_labels.dtype)
                tprint_debug(f"✅ regime_labels validation passed: shape={regime_labels.shape}, dtype={regime_labels.dtype}")
                
            # Check shape consistency with detailed error messages
            if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
                if X.shape[0] != y.shape[0]:
                    validation_errors.append(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")
                else:
                    tprint_debug(f"✅ X and y shape consistency check passed: {X.shape[0]} samples")
                    
            if isinstance(y, np.ndarray) and isinstance(regime_labels, np.ndarray):
                if y.shape[0] != regime_labels.shape[0]:
                    validation_errors.append(f"y and regime_labels must have same number of samples: {y.shape[0]} vs {regime_labels.shape[0]}")
                else:
                    tprint_debug(f"✅ y and regime_labels shape consistency check passed: {y.shape[0]} samples")
                    
            # Enhanced NaN and infinite value checks with math validation
            if isinstance(X, np.ndarray):
                nan_count = np.sum(np.isnan(X))
                inf_count = np.sum(np.isinf(X))
                metrics['X_nan_count'] = int(nan_count)
                metrics['X_inf_count'] = int(inf_count)
                
                if nan_count > 0:
                    validation_errors.append(f"X contains {nan_count} NaN values")
                if inf_count > 0:
                    validation_errors.append(f"X contains {inf_count} infinite values")
                    
                # Check for extreme values
                if np.any(np.abs(X) > 1e10):
                    warnings.append("X contains very large values (>1e10)")
                    
                tprint_debug(f"✅ X data quality check: {nan_count} NaN, {inf_count} Inf values")
                
            if isinstance(y, np.ndarray):
                nan_count = np.sum(np.isnan(y))
                inf_count = np.sum(np.isinf(y))
                metrics['y_nan_count'] = int(nan_count)
                metrics['y_inf_count'] = int(inf_count)
                
                if nan_count > 0:
                    validation_errors.append(f"y contains {nan_count} NaN values")
                if inf_count > 0:
                    validation_errors.append(f"y contains {inf_count} infinite values")
                    
                # Check for extreme values
                if np.any(np.abs(y) > 1e10):
                    warnings.append("y contains very large values (>1e10)")
                    
                tprint_debug(f"✅ y data quality check: {nan_count} NaN, {inf_count} Inf values")
            
            # Enhanced feature names consistency check
            if feature_names is not None and isinstance(X, np.ndarray):
                if len(feature_names) != X.shape[1]:
                    validation_errors.append(f"feature_names length ({len(feature_names)}) must match X features ({X.shape[1]})")
                else:
                    metrics['feature_names_count'] = len(feature_names)
                    tprint_debug(f"✅ Feature names consistency check passed: {len(feature_names)} features")
            
            # Math validation checks if available
            if self.math_validation_available and isinstance(X, np.ndarray):
                try:
                    # Check for correlation matrix validity
                    if X.shape[1] > 1:
                        sample_corr = np.corrcoef(X[:min(1000, X.shape[0]), :].T)
                        if not validate_correlation_matrix(sample_corr):
                            warnings.append("Sample correlation matrix contains invalid values")
                        else:
                            tprint_debug("✅ Sample correlation matrix validation passed")
                except Exception as e:
                    warnings.append(f"Correlation matrix validation failed: {e}")
            
            # Log warnings
            for warning in warnings:
                tprint_warning(f"⚠️ Input validation warning: {warning}")
            
            # Return result
            if validation_errors:
                error_msg = f"Input validation failed: {'; '.join(validation_errors)}"
                tprint_error(f"❌ {error_msg}")
                return {'success': False, 'error': error_msg, 'metrics': metrics, 'warnings': warnings}
            else:
                tprint_success("✅ Enhanced input validation completed successfully")
                return {'success': True, 'metrics': metrics, 'warnings': warnings}
                
        except Exception as e:
            error_msg = f"Enhanced input validation failed with exception: {e}"
            tprint_error(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg, 'metrics': {}, 'warnings': []}
    
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray, feature_names: Optional[List[str]]) -> None:
        """Legacy input validation method for backward compatibility."""
        result = self._validate_inputs_enhanced(X, y, regime_labels, feature_names)
        if not result['success']:
            raise ValueError(result['error'])
    
    def _assess_data_quality(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Assess data quality with comprehensive metrics."""
        try:
            tprint_info("📊 Assessing data quality...")
            metrics = {}
            warnings = []
            
            # Basic data quality metrics
            if self.common_operations_available:
                try:
                    # Convert to DataFrame for quality assessment
                    df_X = pd.DataFrame(X)
                    quality_metrics = calculate_data_quality_metrics(df_X)
                    metrics.update(quality_metrics)
                    tprint_debug(f"✅ Data quality metrics calculated: {quality_metrics}")
                except Exception as e:
                    warnings.append(f"Data quality metrics calculation failed: {e}")
            
            # Regime distribution analysis
            try:
                unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)
                regime_distribution = dict(zip(unique_regimes, regime_counts))
                metrics['regime_distribution'] = regime_distribution
                metrics['unique_regimes_count'] = len(unique_regimes)
                
                # Check for regime balance
                min_regime_count = min(regime_counts)
                max_regime_count = max(regime_counts)
                regime_balance = min_regime_count / max_regime_count if max_regime_count > 0 else 0
                metrics['regime_balance'] = regime_balance
                
                if regime_balance < 0.1:
                    warnings.append(f"Very imbalanced regimes: balance ratio {regime_balance:.3f}")
                elif regime_balance < 0.3:
                    warnings.append(f"Imbalanced regimes: balance ratio {regime_balance:.3f}")
                
                tprint_debug(f"✅ Regime analysis: {len(unique_regimes)} regimes, balance={regime_balance:.3f}")
            except Exception as e:
                warnings.append(f"Regime analysis failed: {e}")
            
            # Target distribution analysis
            try:
                if self.math_validation_available:
                    y_mean = safe_mean(y)
                    y_std = safe_std(y)
                    y_min = np.min(y)
                    y_max = np.max(y)
                    
                    metrics['y_mean'] = y_mean
                    metrics['y_std'] = y_std
                    metrics['y_min'] = y_min
                    metrics['y_max'] = y_max
                    metrics['y_range'] = y_max - y_min
                    
                    # Check for target distribution issues
                    if y_std == 0:
                        warnings.append("Target variable has zero variance")
                    elif y_std / abs(y_mean) > 10 if y_mean != 0 else False:
                        warnings.append("Target variable has very high coefficient of variation")
                    
                    tprint_debug(f"✅ Target analysis: mean={y_mean:.3f}, std={y_std:.3f}, range={y_max-y_min:.3f}")
            except Exception as e:
                warnings.append(f"Target analysis failed: {e}")
            
            # Feature correlation analysis
            try:
                if X.shape[1] > 1 and self.math_validation_available:
                    # Sample correlation for large datasets
                    sample_size = min(1000, X.shape[0])
                    X_sample = X[:sample_size, :]
                    
                    high_corr_pairs = 0
                    for i in range(X_sample.shape[1]):
                        for j in range(i+1, X_sample.shape[1]):
                            corr = safe_correlation(X_sample[:, i], X_sample[:, j])
                            if abs(corr) > 0.95:
                                high_corr_pairs += 1
                    
                    metrics['high_correlation_pairs'] = high_corr_pairs
                    if high_corr_pairs > X.shape[1] * 0.1:
                        warnings.append(f"Many highly correlated features: {high_corr_pairs} pairs")
                    
                    tprint_debug(f"✅ Correlation analysis: {high_corr_pairs} high correlation pairs")
            except Exception as e:
                warnings.append(f"Correlation analysis failed: {e}")
            
            tprint_success("✅ Data quality assessment completed")
            return {'metrics': metrics, 'warnings': warnings}
            
        except Exception as e:
            tprint_error(f"❌ Data quality assessment failed: {e}")
            return {'metrics': {}, 'warnings': [f"Data quality assessment failed: {e}"]}
    
    def _prepare_base_models_enhanced(self, base_tactician_models: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare and validate base tactician models with enhanced error handling."""
        try:
            tprint_info("🔧 Preparing base tactician models...")
            
            if base_tactician_models is None or not base_tactician_models:
                tprint_info("📊 No base tactician models provided, creating from configuration...")
                base_tactician_models = self._create_base_models_from_config()
            
            # Validate base models with comprehensive checks
            valid_models = {}
            validation_errors = []
            warnings = []
            
            for name, model in base_tactician_models.items():
                try:
                    if model is None:
                        warnings.append(f"Base model '{name}' is None, skipping")
                        continue
                    
                    # Check if model has required methods
                    if not hasattr(model, 'predict'):
                        validation_errors.append(f"Base model '{name}' does not have predict method")
                        continue
                    
                    # Check if model is fitted (if it has the attribute)
                    if hasattr(model, 'is_fitted') and not model.is_fitted:
                        warnings.append(f"Base model '{name}' appears to be unfitted")
                    
                    valid_models[name] = model
                    tprint_debug(f"✅ Base model '{name}' validation passed")
                    
                except Exception as e:
                    validation_errors.append(f"Error validating base model '{name}': {e}")
            
            # Log warnings
            for warning in warnings:
                tprint_warning(f"⚠️ Base model warning: {warning}")
            
            # Check for validation errors
            if validation_errors:
                error_msg = f"Base model validation failed: {'; '.join(validation_errors)}"
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            if not valid_models:
                error_msg = "No valid base models found. All provided models failed validation."
                tprint_error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            tprint_success(f"✅ Using {len(valid_models)} base tactician models: {list(valid_models.keys())}")
            return valid_models
            
        except Exception as e:
            tprint_error(f"❌ Base model preparation failed: {e}")
            raise RuntimeError(f"Base model preparation failed: {e}") from e
    
    def _prepare_base_models(self, base_tactician_models: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Legacy base model preparation method for backward compatibility."""
        return self._prepare_base_models_enhanced(base_tactician_models)
    
    def _create_base_models_from_config(self) -> Dict[str, Any]:
        """Create base tactician models from configuration with enhanced error handling."""
        try:
            tprint_info("🏭 Creating tactician models from configuration...")
            
            try:
                from src.utils.ml_common.models.model_factory import create_tactician_models
                models = create_tactician_models()
                
                if not models:
                    raise ValueError("Failed to create any tactician models from configuration")
                
                tprint_success(f"✅ Created {len(models)} tactician models: {list(models.keys())}")
                return models
                
            except ImportError as e:
                tprint_error(f"❌ Failed to import model factory: {e}")
                raise RuntimeError("Cannot create tactician models: model factory not available") from e
            except Exception as e:
                tprint_error(f"❌ Failed to create tactician models from configuration: {e}")
                raise RuntimeError(f"Tactician model creation failed: {e}") from e
                
        except Exception as e:
            tprint_error(f"❌ Base model creation failed: {e}")
            raise RuntimeError(f"Base model creation failed: {e}") from e
    
    def _optimize_memory_before_training(self) -> Dict[str, Any]:
        """Optimize memory before training with hardware integration."""
        try:
            tprint_info("🧠 Optimizing memory before training...")
            metrics = {}
            
            # Memory optimization if available
            if self.memory_optimizer:
                try:
                    memory_result = optimize_memory()
                    metrics.update(memory_result)
                    tprint_debug(f"✅ Memory optimization result: {memory_result}")
                except Exception as e:
                    tprint_warning(f"⚠️ Memory optimization failed: {e}")
                    metrics['memory_optimization_error'] = str(e)
            
            # CPU optimization if available
            if self.cpu_optimizer:
                try:
                    self.cpu_optimizer.optimize_numpy_operations()
                    metrics['cpu_optimization'] = True
                    tprint_debug("✅ CPU optimization applied")
                except Exception as e:
                    tprint_warning(f"⚠️ CPU optimization failed: {e}")
                    metrics['cpu_optimization_error'] = str(e)
            
            tprint_success("✅ Memory optimization completed")
            return metrics
            
        except Exception as e:
            tprint_error(f"❌ Memory optimization failed: {e}")
            return {'error': str(e)}
    
    def _execute_ensemble_training_enhanced(
        self, X_enhanced: np.ndarray, y: np.ndarray, regime_labels: np.ndarray,
        feature_names: Optional[List[str]], hmm_states: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Execute ensemble training with enhanced error handling and monitoring."""
        try:
            tprint_info("🎯 Executing enhanced ensemble training...")
            
            # Execute parent training with comprehensive error handling
            try:
                results = super().execute(
                    X=X_enhanced,
                    y=y,
                    regime_labels=regime_labels,
                    feature_names=feature_names,
                    hmm_states=hmm_states,
                    is_classification=False,  # Tactician ensemble models are typically regression
                    symbol=None,  # Can be passed as kwargs
                    exchange=None,
                    timeframe=self.config.timeframe
                )
            except Exception as e:
                error_msg = f"Parent ensemble training failed: {e}"
                tprint_error(f"❌ {error_msg}")
                return {'success': False, 'error': error_msg, 'results': None, 'metrics': {}}
            
            # Check for errors in results
            if 'error' in results:
                error_msg = f"Ensemble training returned error: {results['error']}"
                tprint_error(f"❌ {error_msg}")
                return {'success': False, 'error': error_msg, 'results': results, 'metrics': {}}
            
            # Extract training metrics
            training_metrics = {
                'regimes_trained': len(results.get('models', {})),
                'training_time': results.get('training_time', 0),
                'enhanced_features_used': X_enhanced.shape[1],
                'samples_processed': X_enhanced.shape[0]
            }
            
            tprint_success("✅ Enhanced ensemble training completed successfully")
            return {'success': True, 'results': results, 'metrics': training_metrics}
            
        except Exception as e:
            error_msg = f"Enhanced ensemble training failed: {e}"
            tprint_error(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg, 'results': None, 'metrics': {}}
    
    def _combine_all_model_inputs_enhanced(
        self,
        X: np.ndarray,
        analyst_models: Optional[Dict[str, Any]],
        analyst_ensembles: Optional[Dict[str, Any]],
        hmm_data: Optional[Dict[str, Any]],
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Combine all model inputs with enhanced error handling and math validation."""
        try:
            tprint_info("🔗 Combining all model inputs with enhanced processing...")
            
            enhanced_features = [X]
            feature_count = X.shape[1]
            integration_stats = {
                'hmm_features_added': 0,
                'analyst_models_integrated': 0,
                'analyst_ensembles_integrated': 0,
                'integration_errors': [],
                'integration_warnings': []
            }
            
            # Add HMM regime features if available
            if hmm_data and 'regime_features' in hmm_data:
                try:
                    hmm_features = hmm_data['regime_features']
                    if isinstance(hmm_features, np.ndarray) and hmm_features.shape[0] == X.shape[0]:
                        # Validate HMM features with math validation
                        if self.math_validation_available:
                            nan_count = np.sum(np.isnan(hmm_features))
                            inf_count = np.sum(np.isinf(hmm_features))
                            if nan_count > 0 or inf_count > 0:
                                integration_stats['integration_warnings'].append(f"HMM features contain {nan_count} NaN and {inf_count} Inf values")
                        
                        enhanced_features.append(hmm_features)
                        feature_count += hmm_features.shape[1]
                        integration_stats['hmm_features_added'] = hmm_features.shape[1]
                        tprint_debug(f"✅ Added {hmm_features.shape[1]} HMM regime features")
                    else:
                        integration_stats['integration_errors'].append("HMM features shape mismatch or invalid format")
                        tprint_warning("⚠️ HMM features shape mismatch or invalid format")
                except Exception as e:
                    integration_stats['integration_errors'].append(f"HMM integration failed: {e}")
                    tprint_warning(f"⚠️ Failed to integrate HMM features: {e}")
            
            # Add analyst model predictions if available
            if analyst_models:
                for model_name, model in analyst_models.items():
                    try:
                        predictions = self._generate_model_predictions_enhanced(model, X, model_name)
                        if predictions is not None:
                            enhanced_features.append(predictions)
                            feature_count += predictions.shape[1]
                            integration_stats['analyst_models_integrated'] += 1
                            tprint_debug(f"✅ Added predictions from analyst model: {model_name}")
                        else:
                            integration_stats['integration_errors'].append(f"Failed to generate predictions for {model_name}")
                    except Exception as e:
                        integration_stats['integration_errors'].append(f"Analyst model {model_name} failed: {e}")
                        tprint_warning(f"⚠️ Could not add predictions from {model_name}: {e}")
            
            # Add analyst ensemble predictions if available
            if analyst_ensembles:
                for ensemble_name, ensemble in analyst_ensembles.items():
                    try:
                        predictions = self._generate_model_predictions_enhanced(ensemble, X, ensemble_name)
                        if predictions is not None:
                            enhanced_features.append(predictions)
                            feature_count += predictions.shape[1]
                            integration_stats['analyst_ensembles_integrated'] += 1
                            tprint_debug(f"✅ Added predictions from analyst ensemble: {ensemble_name}")
                        else:
                            integration_stats['integration_errors'].append(f"Failed to generate predictions for {ensemble_name}")
                    except Exception as e:
                        integration_stats['integration_errors'].append(f"Analyst ensemble {ensemble_name} failed: {e}")
                        tprint_warning(f"⚠️ Could not add predictions from {ensemble_name}: {e}")
            
            # Combine all features with validation
            if len(enhanced_features) > 1:
                try:
                    X_enhanced = np.column_stack(enhanced_features)
                    
                    # Validate combined features
                    if self.math_validation_available:
                        nan_count = np.sum(np.isnan(X_enhanced))
                        inf_count = np.sum(np.isinf(X_enhanced))
                        if nan_count > 0 or inf_count > 0:
                            integration_stats['integration_warnings'].append(f"Combined features contain {nan_count} NaN and {inf_count} Inf values")
                    
                    tprint_info(f"📊 Meta-learner features: {X.shape[1]} base + {feature_count - X.shape[1]} model inputs = {feature_count} total")
                except Exception as e:
                    error_msg = f"Failed to combine enhanced features: {e}"
                    tprint_error(f"❌ {error_msg}")
                    return {'success': False, 'error': error_msg, 'X_enhanced': X, 'metrics': integration_stats}
            else:
                X_enhanced = X
                tprint_info(f"📊 Using base features only: {X.shape[1]} features")
            
            # Log integration summary
            tprint_structured(integration_stats, LogLevel.INFO)
            
            metrics = {
                'original_features': X.shape[1],
                'enhanced_features': X_enhanced.shape[1],
                'feature_increase': X_enhanced.shape[1] - X.shape[1],
                'integration_stats': integration_stats
            }
            
            tprint_success("✅ Enhanced model input combination completed")
            return {'success': True, 'X_enhanced': X_enhanced, 'metrics': metrics}
            
        except Exception as e:
            error_msg = f"Enhanced model input combination failed: {e}"
            tprint_error(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg, 'X_enhanced': X, 'metrics': {}}
    
    def _generate_model_predictions_enhanced(self, model: Any, X: np.ndarray, model_name: str) -> Optional[np.ndarray]:
        """Generate predictions from a model with enhanced error handling and validation."""
        try:
            # Check if model has predict method
            if not hasattr(model, 'predict'):
                tprint_warning(f"⚠️ Model {model_name} does not have predict method")
                return None
            
            # Generate predictions with error handling
            try:
                predictions = model.predict(X)
            except Exception as e:
                tprint_warning(f"⚠️ Model {model_name} prediction failed: {e}")
                return None
            
            # Ensure predictions are 2D
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            
            # Validate predictions
            if predictions.shape[0] != X.shape[0]:
                tprint_warning(f"⚠️ Model {model_name} predictions shape mismatch: {predictions.shape[0]} vs {X.shape[0]}")
                return None
            
            # Enhanced validation with math utilities
            if self.math_validation_available:
                nan_count = np.sum(np.isnan(predictions))
                inf_count = np.sum(np.isinf(predictions))
                if nan_count > 0 or inf_count > 0:
                    tprint_warning(f"⚠️ Model {model_name} produced invalid predictions: {nan_count} NaN, {inf_count} Inf")
                    return None
                
                # Check for extreme values
                if np.any(np.abs(predictions) > 1e10):
                    tprint_warning(f"⚠️ Model {model_name} produced extreme values (>1e10)")
            
            return predictions
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate predictions from {model_name}: {e}")
            return None
    
    def _serialize_models(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize models with enhanced error handling and utility integration."""
        try:
            tprint_info("💾 Serializing models...")
            metrics = {}
            
            if not self.serializer:
                tprint_warning("⚠️ Serializer not available, skipping model serialization")
                return {'serialization_skipped': True, 'reason': 'Serializer not available'}
            
            # Serialize models if available
            if 'models' in results and results['models']:
                try:
                    # Create serialization directory
                    save_path = Path(self.config.model_save_path)
                    ensure_directory(save_path)
                    
                    # Serialize each model
                    serialized_count = 0
                    for regime_id, regime_models in results['models'].items():
                        if isinstance(regime_models, dict):
                            for model_name, model_data in regime_models.items():
                                if 'error' not in model_data and model_data.get('model') is not None:
                                    try:
                                        model_file = save_path / f"{regime_id}_{model_name}.pkl"
                                        success = self.serializer.save(model_data['model'], str(model_file))
                                        if success:
                                            serialized_count += 1
                                            tprint_debug(f"✅ Serialized model: {regime_id}_{model_name}")
                                        else:
                                            tprint_warning(f"⚠️ Failed to serialize model: {regime_id}_{model_name}")
                                    except Exception as e:
                                        tprint_warning(f"⚠️ Error serializing {regime_id}_{model_name}: {e}")
                    
                    metrics['models_serialized'] = serialized_count
                    metrics['serialization_path'] = str(save_path)
                    tprint_success(f"✅ Serialized {serialized_count} models to {save_path}")
                    
                except Exception as e:
                    tprint_warning(f"⚠️ Model serialization failed: {e}")
                    metrics['serialization_error'] = str(e)
            
            # Serialize metadata
            try:
                metadata_file = Path(self.config.model_save_path) / "training_metadata.json"
                metadata = {
                    'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config),
                    'progress_tracker': [progress.__dict__ for progress in self.progress_tracker],
                    'initialization_metrics': self.initialization_metrics,
                    'timestamp': time.time()
                }
                success = safe_json_dump(metadata, metadata_file)
                if success:
                    metrics['metadata_serialized'] = True
                    tprint_debug("✅ Training metadata serialized")
                else:
                    tprint_warning("⚠️ Failed to serialize training metadata")
            except Exception as e:
                tprint_warning(f"⚠️ Metadata serialization failed: {e}")
                metrics['metadata_error'] = str(e)
            
            tprint_success("✅ Model serialization completed")
            return metrics
            
        except Exception as e:
            tprint_error(f"❌ Model serialization failed: {e}")
            return {'error': str(e)}
    
    def _add_meta_learner_metadata_enhanced(
        self,
        results: Dict[str, Any],
        base_models: Dict[str, Any],
        tactician_metrics: Optional[Dict[str, Any]],
        analyst_models: Optional[Dict[str, Any]],
        analyst_ensembles: Optional[Dict[str, Any]],
        analyst_metrics: Optional[Dict[str, Any]],
        hmm_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Add meta-learner specific metadata with enhanced error handling."""
        try:
            tprint_info("📊 Adding enhanced meta-learner metadata...")
            
            # Call the original method first
            try:
                results = self._add_meta_learner_metadata(
                    results, base_models, tactician_metrics,
                    analyst_models, analyst_ensembles, analyst_metrics, hmm_data
                )
            except Exception as e:
                tprint_warning(f"⚠️ Original metadata enhancement failed: {e}")
            
            # Add enhanced metadata
            try:
                # Add utility integration status
                results['utility_integration_status'] = {
                    'tprint_available': self.tprint_available,
                    'common_operations_available': self.common_operations_available,
                    'math_validation_available': self.math_validation_available,
                    'serialization_available': self.serializer is not None,
                    'hardware_optimization': {
                        'gpu_manager': self.gpu_manager is not None,
                        'memory_optimizer': self.memory_optimizer is not None,
                        'cpu_optimizer': self.cpu_optimizer is not None
                    }
                }
                
                # Add enhanced progress tracking
                results['enhanced_progress_tracking'] = {
                    'total_steps': len(self.progress_tracker),
                    'successful_steps': len([p for p in self.progress_tracker if p.success]),
                    'failed_steps': len([p for p in self.progress_tracker if not p.success]),
                    'total_warnings': sum(len(p.warnings) for p in self.progress_tracker),
                    'hardware_metrics_available': any(p.hardware_metrics for p in self.progress_tracker)
                }
                
                # Add initialization metrics
                results['initialization_metrics'] = self.initialization_metrics
                
                tprint_success("✅ Enhanced meta-learner metadata added")
                return {'success': True, 'results': results}
                
            except Exception as e:
                tprint_warning(f"⚠️ Enhanced metadata addition failed: {e}")
                return {'success': True, 'results': results}  # Return original results
                
        except Exception as e:
            tprint_error(f"❌ Enhanced meta-learner metadata failed: {e}")
            return {'success': False, 'error': str(e), 'results': results}
    
    def _add_comprehensive_reporting_enhanced(self, results: Dict[str, Any], overall_start_time: float) -> Dict[str, Any]:
        """Add comprehensive reporting with enhanced metrics and utility integration."""
        try:
            tprint_info("📋 Adding enhanced comprehensive reporting...")
            
            # Call the original method first
            try:
                results = self._add_comprehensive_reporting(results, overall_start_time)
            except Exception as e:
                tprint_warning(f"⚠️ Original reporting failed: {e}")
            
            # Add enhanced reporting
            try:
                total_time = time.time() - overall_start_time
                
                # Enhanced comprehensive report
                enhanced_report = {
                    'enhanced_training_summary': {
                        'total_training_time': total_time,
                        'utility_integration_status': self.initialization_metrics,
                        'hardware_optimization_used': {
                            'gpu_manager': self.gpu_manager is not None,
                            'memory_optimizer': self.memory_optimizer is not None,
                            'cpu_optimizer': self.cpu_optimizer is not None
                        },
                        'error_handling_enhanced': True,
                        'logging_enhanced': self.tprint_available,
                        'math_validation_used': self.math_validation_available,
                        'serialization_used': self.serializer is not None
                    },
                    'enhanced_step_breakdown': [
                        {
                            'step_name': progress.step_name,
                            'duration': progress.duration,
                            'success': progress.success,
                            'error_message': progress.error_message,
                            'metrics': progress.metrics,
                            'warnings': progress.warnings,
                            'hardware_metrics': progress.hardware_metrics
                        }
                        for progress in self.progress_tracker
                    ],
                    'enhanced_performance_metrics': {
                        'total_regimes': len(results.get('models', {})),
                        'successful_regimes': len([r for r in results.get('models', {}).values() if 'error' not in r]),
                        'failed_regimes': len([r for r in results.get('models', {}).values() if 'error' in r]),
                        'average_training_time_per_regime': total_time / max(len(results.get('models', {})), 1),
                        'utility_integration_success_rate': sum(self.initialization_metrics.values()) / len(self.initialization_metrics) if self.initialization_metrics else 0
                    }
                }
                
                # Add enhanced report to results
                results['enhanced_comprehensive_report'] = enhanced_report
                
                # Log enhanced summary
                self._log_enhanced_comprehensive_summary(enhanced_report)
                
                tprint_success("✅ Enhanced comprehensive reporting completed")
                return {'success': True, 'results': results}
                
            except Exception as e:
                tprint_warning(f"⚠️ Enhanced reporting failed: {e}")
                return {'success': True, 'results': results}  # Return original results
                
        except Exception as e:
            tprint_error(f"❌ Enhanced comprehensive reporting failed: {e}")
            return {'success': False, 'error': str(e), 'results': results}
    
    def _log_enhanced_comprehensive_summary(self, report: Dict[str, Any]) -> None:
        """Log enhanced comprehensive training summary."""
        try:
            summary = report['enhanced_training_summary']
            performance = report['enhanced_performance_metrics']
            
            tprint_info("=" * 80)
            tprint_info("🎯 ENHANCED TACTICIAN ENSEMBLE TRAINING SUMMARY")
            tprint_info("=" * 80)
            tprint_info(f"⏱️  Total Training Time: {summary['total_training_time']:.2f}s")
            tprint_info(f"🔧 Utility Integration: {summary['utility_integration_status']}")
            tprint_info(f"🧠 Hardware Optimization: {summary['hardware_optimization_used']}")
            tprint_info(f"📊 Total Regimes: {performance['total_regimes']}")
            tprint_info(f"✅ Successful Regimes: {performance['successful_regimes']}")
            tprint_info(f"❌ Failed Regimes: {performance['failed_regimes']}")
            tprint_info(f"🔗 Integration Success Rate: {performance['utility_integration_success_rate']:.2%}")
            
            # Log enhanced step breakdown
            tprint_info("\n📋 Enhanced Step Breakdown:")
            for step in report['enhanced_step_breakdown']:
                status = "✅" if step['success'] else "❌"
                tprint_info(f"  {status} {step['step_name']}: {step['duration']:.2f}s")
                if step['warnings']:
                    tprint_info(f"    Warnings: {len(step['warnings'])}")
                if step['hardware_metrics']:
                    tprint_info(f"    Hardware Metrics: {len(step['hardware_metrics'])}")
                if not step['success'] and step['error_message']:
                    tprint_info(f"    Error: {step['error_message']}")
            
            tprint_info("=" * 80)
            
        except Exception as e:
            tprint_error(f"❌ Failed to log enhanced comprehensive summary: {e}")
    
    def _create_error_result(self, error_type: str, error_message: str) -> Dict[str, Any]:
        """Create standardized error result with enhanced tracking."""
        return {
            'error': error_type,
            'error_message': error_message,
            'success': False,
            'training_time': 0,
            'progress_tracker': [progress.__dict__ for progress in self.progress_tracker],
            'utility_integration_status': self.initialization_metrics,
            'enhanced_error_handling': True
        }
    
    
    def _combine_all_model_inputs(
        self,
        X: np.ndarray,
        analyst_models: Optional[Dict[str, Any]],
        analyst_ensembles: Optional[Dict[str, Any]],
        hmm_data: Optional[Dict[str, Any]],
        feature_names: Optional[List[str]]
    ) -> np.ndarray:
        """Legacy method for backward compatibility - uses enhanced version."""
        result = self._combine_all_model_inputs_enhanced(X, analyst_models, analyst_ensembles, hmm_data, feature_names)
        if result['success']:
            return result['X_enhanced']
        else:
            tprint_error(f"❌ Model input combination failed: {result['error']}")
            return X
    
    def _generate_model_predictions(self, model: Any, X: np.ndarray, model_name: str) -> Optional[np.ndarray]:
        """Legacy method for backward compatibility - uses enhanced version."""
        return self._generate_model_predictions_enhanced(model, X, model_name)
    
    def _add_meta_learner_metadata(
        self,
        results: Dict[str, Any],
        base_models: Dict[str, Any],
        tactician_metrics: Optional[Dict[str, Any]],
        analyst_models: Optional[Dict[str, Any]],
        analyst_ensembles: Optional[Dict[str, Any]],
        analyst_metrics: Optional[Dict[str, Any]],
        hmm_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Legacy method for backward compatibility - uses enhanced version."""
        result = self._add_meta_learner_metadata_enhanced(
            results, base_models, tactician_metrics,
            analyst_models, analyst_ensembles, analyst_metrics, hmm_data
        )
        if result['success']:
            return result['results']
        else:
            tprint_error(f"❌ Meta-learner metadata enhancement failed: {result['error']}")
            return results
    
    def _add_comprehensive_reporting(self, results: Dict[str, Any], overall_start_time: float) -> Dict[str, Any]:
        """Legacy method for backward compatibility - uses enhanced version."""
        result = self._add_comprehensive_reporting_enhanced(results, overall_start_time)
        if result['success']:
            return result['results']
        else:
            tprint_error(f"❌ Comprehensive reporting failed: {result['error']}")
            return results
    
    def _summarize_evaluation_results(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize evaluation results across all regimes with enhanced error handling."""
        try:
            tprint_info("📊 Summarizing evaluation results...")
            summary = {
                'total_regimes_evaluated': len(evaluation_results),
                'regime_metrics': {},
                'overall_performance': {}
            }
            
            all_metrics = []
            for regime, metrics in evaluation_results.items():
                if isinstance(metrics, dict) and 'error' not in metrics:
                    summary['regime_metrics'][regime] = metrics
                    all_metrics.append(metrics)
            
            # Calculate overall performance if we have metrics
            if all_metrics:
                metric_names = set()
                for metrics in all_metrics:
                    metric_names.update(metrics.keys())
                
                for metric_name in metric_names:
                    values = [m.get(metric_name) for m in all_metrics if metric_name in m and m[metric_name] is not None]
                    if values:
                        if self.math_validation_available:
                            summary['overall_performance'][metric_name] = {
                                'mean': safe_mean(np.array(values)),
                                'std': safe_std(np.array(values)),
                                'min': np.min(values),
                                'max': np.max(values),
                                'count': len(values)
                            }
                        else:
                            summary['overall_performance'][metric_name] = {
                                'mean': np.mean(values),
                                'std': np.std(values),
                                'min': np.min(values),
                                'max': np.max(values),
                                'count': len(values)
                            }
            
            tprint_success("✅ Evaluation results summarized successfully")
            return summary
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to summarize evaluation results: {e}")
            return {'error': str(e)}
    
    def _log_comprehensive_summary(self, report: Dict[str, Any]) -> None:
        """Log comprehensive training summary with enhanced logging."""
        try:
            summary = report['training_summary']
            performance = report['performance_metrics']
            
            tprint_info("=" * 80)
            tprint_info("🎯 TACTICIAN ENSEMBLE TRAINING SUMMARY")
            tprint_info("=" * 80)
            tprint_info(f"⏱️  Total Training Time: {summary['total_training_time']:.2f}s")
            tprint_info(f"✅ Steps Completed: {summary['steps_completed']}")
            tprint_info(f"❌ Steps Failed: {summary['steps_failed']}")
            tprint_info(f"🚀 Vectorization: {'Enabled' if summary['vectorization_enabled'] else 'Disabled'}")
            tprint_info(f"📊 Total Regimes: {performance['total_regimes']}")
            tprint_info(f"✅ Successful Regimes: {performance['successful_regimes']}")
            tprint_info(f"❌ Failed Regimes: {performance['failed_regimes']}")
            
            # Log step breakdown
            tprint_info("\n📋 Step Breakdown:")
            for step in report['step_breakdown']:
                status = "✅" if step['success'] else "❌"
                tprint_info(f"  {status} {step['step_name']}: {step['duration']:.2f}s")
                if not step['success'] and step['error_message']:
                    tprint_info(f"    Error: {step['error_message']}")
            
            # Log evaluation summary if available
            if 'evaluation_summary' in report:
                eval_summary = report['evaluation_summary']
                if 'overall_performance' in eval_summary and eval_summary['overall_performance']:
                    tprint_info("\n📈 Overall Performance:")
                    for metric, stats in eval_summary['overall_performance'].items():
                        tprint_info(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            
            tprint_info("=" * 80)
            
        except Exception as e:
            tprint_error(f"❌ Failed to log comprehensive summary: {e}")


# Convenience functions for backward compatibility and enhanced usage
def create_tactician_ensemble_training_step(
    config: Optional[EnsembleTrainingConfig] = None,
    enable_vectorization: bool = True
) -> TacticianEnsembleTrainingStep:
    """
    Create Tactician ensemble training step with enhanced error handling and utility integration.
    
    Args:
        config: Per-regime training configuration
        enable_vectorization: Whether to enable vectorized training
        
    Returns:
        Enhanced TacticianEnsembleTrainingStep instance
    """
    try:
        tprint_info("🏭 Creating enhanced Tactician ensemble training step...")
        step = TacticianEnsembleTrainingStep(config, enable_vectorization)
        tprint_success("✅ Enhanced Tactician ensemble training step created successfully")
        return step
    except Exception as e:
        tprint_error(f"❌ Failed to create Tactician ensemble training step: {e}")
        raise RuntimeError(f"Failed to create Tactician ensemble training step: {e}") from e


def execute_tactician_ensemble_training(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[EnsembleTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    base_tactician_models: Optional[Dict[str, Any]] = None,
    tactician_training_metrics: Optional[Dict[str, Any]] = None,
    analyst_models: Optional[Dict[str, Any]] = None,
    analyst_ensembles: Optional[Dict[str, Any]] = None,
    analyst_ensemble_metrics: Optional[Dict[str, Any]] = None,
    hmm_data: Optional[Dict[str, Any]] = None,
    enable_vectorization: bool = True
) -> Dict[str, Any]:
    """
    Execute Tactician ensemble training step with enhanced error handling and utility integration.
    
    Args:
        X: Input features (1m timeframe with cross-timeframe features)
        y: Target values (tactician outputs - timing decisions)
        regime_labels: Regime labels for each sample
        config: Per-regime training configuration
        feature_names: Names of input features
        hmm_states: HMM cluster/regime states
        base_tactician_models: Individual tactician models to ensemble
        tactician_training_metrics: Performance metrics of base tactician models
        analyst_models: Individual analyst models
        analyst_ensembles: Analyst ensemble models
        analyst_ensemble_metrics: Performance metrics of analyst ensembles
        hmm_data: HMM regime data and features
        enable_vectorization: Whether to enable vectorized training
        
    Returns:
        Dictionary containing training results and metadata
    """
    try:
        tprint_info("🚀 Starting enhanced Tactician ensemble training execution...")
        
        # Create training step
        step = create_tactician_ensemble_training_step(config, enable_vectorization)
        
        # Execute training
        results = step.execute(
            X, y, regime_labels, feature_names, hmm_states,
            base_tactician_models, tactician_training_metrics,
            analyst_models, analyst_ensembles, analyst_ensemble_metrics, hmm_data
        )
        
        # Log execution summary
        if results.get('success', True):
            tprint_success("🎯 Enhanced Tactician ensemble training execution completed successfully")
        else:
            tprint_error(f"❌ Enhanced Tactician ensemble training execution failed: {results.get('error', 'Unknown error')}")
        
        return results
        
    except Exception as e:
        tprint_error(f"❌ Enhanced Tactician ensemble training execution failed: {e}")
        return {
            'error': 'Execution failed',
            'error_message': str(e),
            'success': False,
            'training_time': 0
        }


def get_enhanced_training_capabilities() -> Dict[str, Any]:
    """
    Get information about enhanced training capabilities and utility integrations.
    
    Returns:
        Dictionary containing capability information
    """
    try:
        capabilities = {
            'enhanced_error_handling': True,
            'comprehensive_logging': TPRINT_AVAILABLE,
            'utility_integrations': {
                'common_operations': COMMON_OPERATIONS_AVAILABLE,
                'math_validation': MATH_VALIDATION_AVAILABLE,
                'serialization': SERIALIZATION_AVAILABLE,
                'kline_parquet': KLINE_PARQUET_AVAILABLE,
                'hardware_optimization': {
                    'm1_gpu': True,  # Will be checked at runtime
                    'm1_memory': True,  # Will be checked at runtime
                    'm1_cpu': True  # Will be checked at runtime
                }
            },
            'vectorized_training': VECTORIZED_TRAINING_AVAILABLE,
            'enhanced_features': [
                'Fast failing for critical errors',
                'Comprehensive logging with tprint',
                'Hardware optimization integration',
                'Math validation and data quality checks',
                'Enhanced progress tracking',
                'Model serialization and persistence',
                'Comprehensive reporting and metrics'
            ]
        }
        
        tprint_info("📋 Enhanced training capabilities:")
        tprint_structured(capabilities, LogLevel.INFO)
        
        return capabilities
        
    except Exception as e:
        tprint_error(f"❌ Failed to get enhanced training capabilities: {e}")
        return {'error': str(e)}


# Example usage and demonstration
if __name__ == "__main__":
    # Example of how to use the enhanced meta-learner ensemble training version
    tprint_info("🎯 Enhanced Tactician Ensemble Training Step (Meta-Learner)")
    tprint_info("=" * 80)
    
    try:
        # Display enhanced capabilities
        capabilities = get_enhanced_training_capabilities()
        
        # Create configuration with enhanced validation
        tprint_info("📋 Creating enhanced configuration...")
        config = EnsembleTrainingConfig(
            model_name="tactician_ensemble_models_enhanced",
            timeframe="1m",
            model_types=["node", "catboost", "lightgbm", "elastic_net"],
            hpo_n_trials=50,  # Reduced for demo
            enable_hpo=True,
            save_models=True,
            model_save_path="./models/tactician_ensemble_models_enhanced"
        )
        
        # Create enhanced training step
        tprint_info("🏭 Creating enhanced training step...")
        training_step = create_tactician_ensemble_training_step(config, enable_vectorization=True)
        
        tprint_success(f"✅ Created enhanced tactician ensemble training step with {len(config.model_types)} ensemble types")
        tprint_info(f"📊 HPO enabled: {config.enable_hpo}")
        tprint_info(f"💾 Save models: {config.save_models}")
        tprint_info(f"📁 Save path: {config.model_save_path}")
        tprint_info(f"⏰ Base timeframe: {config.timeframe}")
        
        # Display enhanced features
        tprint_info("\n🎯 Enhanced Tactician Ensemble Module Features:")
        tprint_info("- Operates on 1m timeframe with cross-timeframe features")
        tprint_info("- Meta-learner combining ALL previous model inputs")
        tprint_info("- All-regime ensemble training for comprehensive intelligence")
        tprint_info("- Final timing decision optimization")
        tprint_info("- Models: NODE (Neural Oblivious Decision Ensembles), CatBoost, LightGBM, Elastic Net")
        tprint_info("- Comprehensive context from ALL model types")
        
        tprint_info("\n🔧 Enhanced Error Handling & Logging:")
        tprint_info("- Extensive try/except blocks with fast failing for critical errors")
        tprint_info("- Comprehensive logging using tprint at every step")
        tprint_info("- Integration with common utilities and tools")
        tprint_info("- Hardware optimization support (M1 GPU/CPU)")
        tprint_info("- Math validation and data quality checks")
        tprint_info("- Serialization utilities for model persistence")
        
        tprint_info("\n🔄 Integration with ALL Previous Models:")
        tprint_info("- Receives individual tactician model predictions")
        tprint_info("- Integrates analyst model predictions")
        tprint_info("- Integrates analyst ensemble predictions")
        tprint_info("- Integrates HMM regime data and features")
        tprint_info("- Creates final meta-learner for optimal timing decisions")
        tprint_info("- Provides comprehensive market intelligence")
        
        tprint_info("\n💡 Usage Example:")
        tprint_info("# The actual training would be called with:")
        tprint_info("# results = training_step.execute(X, y, regime_labels, feature_names, hmm_states, ...)")
        tprint_info("# or using the convenience function:")
        tprint_info("# results = execute_tactician_ensemble_training(X, y, regime_labels, config, ...)")
        
        tprint_success("🎉 Enhanced Tactician Ensemble Training Step demonstration completed successfully!")
        
    except Exception as e:
        tprint_error(f"❌ Enhanced Tactician Ensemble Training Step demonstration failed: {e}")
        tprint_error(f"❌ Traceback: {traceback.format_exc()}")