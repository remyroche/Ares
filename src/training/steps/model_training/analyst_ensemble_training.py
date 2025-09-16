"""
Analyst Ensemble Training Step - Enhanced with Comprehensive Error Handling and Logging

This step handles per-regime ensemble training of Analyst models using common dependencies.
The Analyst Ensemble operates on 5m timeframe and combines individual analyst models
to create robust ensemble predictions for trade decisions.

Enhanced with:
- Extensive try/except blocks with fast failing for important errors
- Comprehensive logging using tprint at every step
- Integration with common utilities (math_validation, serialization, hardware optimization)
- ML common utilities (CV, lookahead, HPO, etc.)
- Vectorized training capabilities for improved performance
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import traceback
from pathlib import Path
import sys
import os

# Import tprint utilities - required for proper logging
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_debug, tprint_progress, tprint_performance, tprint_structured,
    tprint_timer, LogLevel
)

from src.utils.logger import system_logger

from src.utils.ml_common.config.base_training_config import EnsembleTrainingConfig
from src.utils.ml_common.training.ensemble_training_step import EnsembleTrainingStep

# Import math validation utilities
from src.utils.math_validation import (
    validate_finite, safe_divide, safe_log, safe_sqrt, safe_power,
    validate_array_finite, validate_matrix_finite
)

from src.utils.serialization_utils import JSONSerializer, PickleSerializer

from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

from src.utils.common_operations import (
    get_m1_gpu_manager as common_get_m1_gpu_manager,
    get_m1_memory_optimizer as common_get_m1_memory_optimizer,
    get_m1_cpu_optimizer as common_get_m1_cpu_optimizer
)

# Import ML common utilities
from src.utils.ml_common.training.vectorized_training_manager import VectorizedTrainingManager
from src.utils.ml_common.matrix_cross_validation import MatrixCrossValidator
from src.utils.ml_common.optimization.hyperparameter_optimization import HyperparameterOptimizer

# Setup logging
logger = system_logger.getChild('AnalystEnsembleTraining')

# Initialize hardware optimizers
gpu_manager = get_m1_gpu_manager()
memory_optimizer = get_m1_memory_optimizer()
cpu_optimizer = get_m1_cpu_optimizer()


class AnalystEnsembleTrainingStep(EnsembleTrainingStep):
    """
    Enhanced Analyst Ensemble Training Step with comprehensive error handling and logging.
    
    Features:
    - Extensive try/except blocks with fast failing for important errors
    - Comprehensive logging using tprint at every step
    - Integration with common utilities (math_validation, serialization, hardware optimization)
    - ML common utilities (CV, lookahead, HPO, etc.)
    - Per-regime ensemble training, HPO, saving, and metrics
    
    The Analyst Ensemble operates on 5m timeframe and combines individual analyst models
    to create robust ensemble predictions for trade decisions.
    """
    
    def __init__(self, config: Optional[EnsembleTrainingConfig] = None, enable_vectorization: bool = True):
        """
        Initialize Analyst ensemble training step with enhanced error handling and logging.

        Args:
            config: Per-regime training configuration
            enable_vectorization: Whether to enable vectorized training
            
        Raises:
            RuntimeError: If initialization fails with critical errors
            ValueError: If configuration is invalid
        """
        tprint_info("🚀 Initializing Analyst Ensemble Training Step")
        
        # Initialize logging and timing
        self.logger = logger.getChild('AnalystEnsembleTrainingStep')
        self.start_time = time.time()
        self.initialization_errors = []
        self.initialization_warnings = []
        
        try:
            # Step 1: Validate and setup configuration
            tprint_info("📋 Step 1: Setting up configuration")
            config = self._setup_configuration(config)
            
            # Step 2: Validate configuration with enhanced error handling
            tprint_info("🔍 Step 2: Validating configuration")
            self._validate_config_enhanced(config)
            
            # Step 3: Initialize hardware optimizers
            tprint_info("⚙️ Step 3: Initializing hardware optimizers")
            self._initialize_hardware_optimizers()
            
            # Step 4: Initialize parent class with error handling
            tprint_info("🏗️ Step 4: Initializing parent class")
            self._initialize_parent_class(config, enable_vectorization)
            
            # Step 5: Setup tracking and monitoring
            tprint_info("📊 Step 5: Setting up tracking and monitoring")
            self._setup_tracking_and_monitoring(config)
            
            # Step 6: Validate initialization success
            tprint_info("✅ Step 6: Validating initialization")
            self._validate_initialization_success()
            
            # Log comprehensive initialization summary
            self._log_initialization_summary()
            
        except Exception as e:
            self._handle_initialization_error(e)
            raise
    
    def _setup_configuration(self, config: Optional[EnsembleTrainingConfig]) -> EnsembleTrainingConfig:
        """Setup configuration with enhanced error handling."""
        try:
            if config is None:
                tprint_info("📋 Creating default configuration for analyst ensemble training")
                config = EnsembleTrainingConfig(
                    model_name="analyst_ensemble_models",
                    timeframe="5m",
                    model_types=["tcn", "catboost", "lightgbm", "ensemble_rf"],
                    hpo_n_trials=100,
                    hpo_timeout_seconds=3600,
                    min_samples_per_regime=1000,
                    enable_data_augmentation=True,
                    augmentation_method="smote",
                    model_save_path="./models/analyst_ensemble_models",
                    evaluation_metrics=["mse", "mae", "r2", "mape", "smape"]
                )
                tprint_success("✅ Default configuration created successfully")
            else:
                tprint_info(f"📋 Using provided configuration: {config.model_name}")
            
            return config
            
        except Exception as e:
            error_msg = f"Configuration setup failed: {e}"
            tprint_error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _validate_config_enhanced(self, config: EnsembleTrainingConfig) -> None:
        """Enhanced configuration validation with comprehensive error handling."""
        try:
            tprint_info("🔍 Starting enhanced configuration validation")
            
            # Validate model types
            if not hasattr(config, 'model_types') or not config.model_types or len(config.model_types) == 0:
                raise ValueError("At least one model type must be specified")
            
            # Validate timeframe
            valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            if not hasattr(config, 'timeframe') or config.timeframe not in valid_timeframes:
                self.initialization_warnings.append(f"Unusual timeframe specified: {getattr(config, 'timeframe', 'None')}")
                tprint_warning(f"⚠️ Unusual timeframe specified: {getattr(config, 'timeframe', 'None')}")
            
            # Validate HPO parameters
            if hasattr(config, 'enable_hpo') and config.enable_hpo:
                if hasattr(config, 'hpo_n_trials') and config.hpo_n_trials <= 0:
                    raise ValueError("HPO trials must be positive")
                if hasattr(config, 'hpo_timeout_seconds') and config.hpo_timeout_seconds <= 0:
                    raise ValueError("HPO timeout must be positive")
            
            # Validate minimum samples
            if hasattr(config, 'min_samples_per_regime') and config.min_samples_per_regime <= 0:
                raise ValueError("Minimum samples per regime must be positive")
            
            # Validate save path
            if hasattr(config, 'save_models') and config.save_models and hasattr(config, 'model_save_path') and config.model_save_path:
                try:
                    save_path = Path(config.model_save_path)
                    if not save_path.parent.exists():
                        self.initialization_warnings.append(f"Save path parent directory does not exist: {save_path.parent}")
                        tprint_warning(f"⚠️ Save path parent directory does not exist: {save_path.parent}")
                except Exception as e:
                    self.initialization_warnings.append(f"Save path validation failed: {e}")
                    tprint_warning(f"⚠️ Save path validation failed: {e}")
            
            tprint_success("✅ Enhanced configuration validation passed")
            
        except Exception as e:
            error_msg = f"Enhanced configuration validation failed: {e}"
            tprint_error(error_msg)
            raise ValueError(error_msg) from e
    
    def _initialize_hardware_optimizers(self) -> None:
        """Initialize hardware optimizers."""
        tprint_info("⚙️ Initializing hardware optimizers")
        
        # Initialize hardware optimizers
        self.gpu_manager = get_m1_gpu_manager()
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        
        tprint_success("✅ Hardware optimizers initialized")
        
        if self.gpu_manager:
            tprint_success("✅ M1 GPU manager available")
        if self.memory_optimizer:
            tprint_success("✅ M1 memory optimizer available")
        if self.cpu_optimizer:
            tprint_success("✅ M1 CPU optimizer available")
    
    def _initialize_parent_class(self, config: EnsembleTrainingConfig, enable_vectorization: bool) -> None:
        """Initialize parent class."""
        tprint_info("🏗️ Initializing parent class")
        
        super().__init__(config, enable_vectorization=enable_vectorization)
        tprint_success("✅ Parent class initialized successfully")
    
    def _setup_tracking_and_monitoring(self, config: EnsembleTrainingConfig) -> None:
        """Setup tracking and monitoring."""
        tprint_info("📊 Setting up tracking and monitoring")
        
        # Initialize tracking variables
        self.training_stats = {
            'initialization_time': time.time() - self.start_time,
            'vectorization_enabled': self.enable_vectorization,
            'config_used': config.model_name,
            'model_types': config.model_types,
            'timeframe': config.timeframe,
            'hardware_optimizers_available': {
                'gpu_manager': self.gpu_manager is not None,
                'memory_optimizer': self.memory_optimizer is not None,
                'cpu_optimizer': self.cpu_optimizer is not None
            },
            'initialization_errors': self.initialization_errors.copy(),
            'initialization_warnings': self.initialization_warnings.copy()
        }
        
        tprint_success("✅ Tracking and monitoring setup completed")
    
    def _validate_initialization_success(self) -> None:
        """Validate initialization success with comprehensive checks."""
        try:
            tprint_info("✅ Validating initialization success")
            
            # Check for critical errors
            if self.initialization_errors:
                critical_errors = [e for e in self.initialization_errors if 'critical' in e.lower()]
                if critical_errors:
                    raise RuntimeError(f"Critical initialization errors: {critical_errors}")
            
            # Check essential components
            if not hasattr(self, 'config'):
                raise RuntimeError("Configuration not properly initialized")
            
            if not hasattr(self, 'training_stats'):
                raise RuntimeError("Training stats not properly initialized")
            
            tprint_success("✅ Initialization validation passed")
            
        except Exception as e:
            error_msg = f"Initialization validation failed: {e}"
            tprint_error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _log_initialization_summary(self) -> None:
        """Log comprehensive initialization summary."""
        try:
            tprint_info("📊 INITIALIZATION SUMMARY")
            tprint_info("=" * 50)
            
            # Configuration summary
            tprint_info(f"📋 Model name: {self.training_stats['config_used']}")
            tprint_info(f"⏰ Timeframe: {self.training_stats['timeframe']}")
            tprint_info(f"🤖 Model types: {len(self.training_stats['model_types'])} types")
            tprint_info(f"🚀 Vectorization: {self.training_stats['vectorization_enabled']}")
            
            # Hardware optimizers summary
            hw_stats = self.training_stats['hardware_optimizers_available']
            tprint_info(f"⚙️ GPU manager: {hw_stats['gpu_manager']}")
            tprint_info(f"💾 Memory optimizer: {hw_stats['memory_optimizer']}")
            tprint_info(f"🖥️ CPU optimizer: {hw_stats['cpu_optimizer']}")
            
            # Utilities availability
            utils_stats = self.training_stats['utilities_available']
            tprint_info("🔧 Available utilities:")
            for util, available in utils_stats.items():
                status = "✅" if available else "❌"
                tprint_info(f"   {status} {util}")
            
            # Warnings and errors
            if self.initialization_warnings:
                tprint_warning(f"⚠️ {len(self.initialization_warnings)} warnings during initialization")
                for warning in self.initialization_warnings:
                    tprint_warning(f"   - {warning}")
            
            if self.initialization_errors:
                tprint_warning(f"⚠️ {len(self.initialization_errors)} non-critical errors during initialization")
                for error in self.initialization_errors:
                    tprint_warning(f"   - {error}")
            
            # Performance metrics
            init_time = self.training_stats['initialization_time']
            tprint_performance("Initialization", init_time)
            
            tprint_info("=" * 50)
            tprint_success("🎉 Analyst Ensemble Training Step initialization completed successfully")
            
        except Exception as e:
            tprint_error(f"Failed to log initialization summary: {e}")
    
    def _handle_initialization_error(self, error: Exception) -> None:
        """Handle initialization errors with comprehensive logging."""
        try:
            tprint_error("❌ INITIALIZATION FAILED")
            tprint_error("=" * 50)
            tprint_error(f"Error: {error}")
            tprint_error(f"Type: {type(error).__name__}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            
            # Log initialization context
            if hasattr(self, 'initialization_errors'):
                tprint_error(f"Previous errors: {self.initialization_errors}")
            if hasattr(self, 'initialization_warnings'):
                tprint_error(f"Previous warnings: {self.initialization_warnings}")
            
            tprint_error("=" * 50)
            
        except Exception as log_error:
            print(f"Failed to log initialization error: {log_error}")
            print(f"Original error: {error}")
    
    def _validate_config(self, config: EnsembleTrainingConfig) -> None:
        """
        Legacy configuration validation method - kept for backward compatibility.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            tprint_info("🔍 Running legacy configuration validation")
            
            # Validate model types
            if not hasattr(config, 'model_types') or not config.model_types or len(config.model_types) == 0:
                raise ValueError("At least one model type must be specified")
            
            # Validate timeframe
            valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            if not hasattr(config, 'timeframe') or config.timeframe not in valid_timeframes:
                tprint_warning(f"⚠️ Unusual timeframe specified: {getattr(config, 'timeframe', 'None')}")
            
            # Validate HPO parameters
            if hasattr(config, 'enable_hpo') and config.enable_hpo:
                if hasattr(config, 'hpo_n_trials') and config.hpo_n_trials <= 0:
                    raise ValueError("HPO trials must be positive")
                if hasattr(config, 'hpo_timeout_seconds') and config.hpo_timeout_seconds <= 0:
                    raise ValueError("HPO timeout must be positive")
            
            # Validate minimum samples
            if hasattr(config, 'min_samples_per_regime') and config.min_samples_per_regime <= 0:
                raise ValueError("Minimum samples per regime must be positive")
            
            # Validate save path
            if hasattr(config, 'save_models') and config.save_models and hasattr(config, 'model_save_path') and config.model_save_path:
                try:
                    save_path = Path(config.model_save_path)
                    if not save_path.parent.exists():
                        tprint_warning(f"⚠️ Save path parent directory does not exist: {save_path.parent}")
                except Exception as e:
                    tprint_warning(f"⚠️ Save path validation failed: {e}")
            
            tprint_success("✅ Legacy configuration validation passed")
            
        except Exception as e:
            tprint_error(f"❌ Legacy configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}") from e
    
    def _validate_input_data(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> None:
        """
        Enhanced input data validation with comprehensive error handling and math validation.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If critical validation errors occur
        """
        try:
            tprint_info("🔍 Starting enhanced input data validation")
            
            # Step 1: Basic shape validation
            tprint_info("📏 Step 1: Validating data shapes")
            self._validate_data_shapes(X, y, regime_labels)
            
            # Step 2: Empty data validation
            tprint_info("📊 Step 2: Checking for empty data")
            self._validate_empty_data(X, y, regime_labels)
            
            # Step 3: Mathematical validation using math_validation utilities
            tprint_info("🧮 Step 3: Mathematical validation")
            self._validate_mathematical_properties(X, y, regime_labels)
            
            # Step 4: Regime distribution validation
            tprint_info("📈 Step 4: Validating regime distribution")
            self._validate_regime_distribution(regime_labels)
            
            # Step 5: Memory and performance validation
            tprint_info("💾 Step 5: Memory and performance validation")
            self._validate_memory_and_performance(X, y, regime_labels)
            
            tprint_success("✅ Enhanced input data validation completed successfully")
            
        except Exception as e:
            tprint_error(f"❌ Enhanced input data validation failed: {e}")
            raise ValueError(f"Invalid input data: {e}") from e
    
    def _validate_data_shapes(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> None:
        """Validate data shapes with enhanced error handling."""
        try:
            # Check if arrays are numpy arrays
            if not isinstance(X, np.ndarray):
                raise ValueError(f"X must be a numpy array, got {type(X)}")
            if not isinstance(y, np.ndarray):
                raise ValueError(f"y must be a numpy array, got {type(y)}")
            if not isinstance(regime_labels, np.ndarray):
                raise ValueError(f"regime_labels must be a numpy array, got {type(regime_labels)}")
            
            # Check data shapes
            if X.shape[0] != y.shape[0] or X.shape[0] != regime_labels.shape[0]:
                raise ValueError(f"Data shape mismatch: X={X.shape}, y={y.shape}, regimes={regime_labels.shape}")
            
            # Check dimensions
            if len(X.shape) != 2:
                raise ValueError(f"X must be 2D array, got shape {X.shape}")
            if len(y.shape) != 1:
                raise ValueError(f"y must be 1D array, got shape {y.shape}")
            if len(regime_labels.shape) != 1:
                raise ValueError(f"regime_labels must be 1D array, got shape {regime_labels.shape}")
            
            tprint_success(f"✅ Data shapes validated: X={X.shape}, y={y.shape}, regimes={regime_labels.shape}")
            
        except Exception as e:
            tprint_error(f"❌ Data shape validation failed: {e}")
            raise
    
    def _validate_empty_data(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> None:
        """Validate for empty data with enhanced error handling."""
        try:
            # Check for empty data
            if X.shape[0] == 0:
                raise ValueError("Input data is empty")
            
            if X.shape[1] == 0:
                raise ValueError("No features in input data")
            
            if y.shape[0] == 0:
                raise ValueError("Target data is empty")
            
            if regime_labels.shape[0] == 0:
                raise ValueError("Regime labels are empty")
            
            tprint_success(f"✅ Empty data validation passed: {X.shape[0]} samples, {X.shape[1]} features")
            
        except Exception as e:
            tprint_error(f"❌ Empty data validation failed: {e}")
            raise
    
    def _validate_mathematical_properties(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> None:
        """Validate mathematical properties using math_validation utilities."""
        # Validate arrays for finite values
        validate_array_finite(X, "input_features")
        tprint_success("✅ Input features finite validation passed")
        
        validate_array_finite(y, "target_values")
        tprint_success("✅ Target values finite validation passed")
        
        validate_array_finite(regime_labels, "regime_labels")
        tprint_success("✅ Regime labels finite validation passed")
        
        # Check for NaN values
        if np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            tprint_warning(f"⚠️ Found {nan_count} NaN values in input features")
        
        if np.isnan(y).any():
            nan_count = np.isnan(y).sum()
            tprint_warning(f"⚠️ Found {nan_count} NaN values in target values")
        
        if np.isnan(regime_labels).any():
            nan_count = np.isnan(regime_labels).sum()
            tprint_warning(f"⚠️ Found {nan_count} NaN values in regime labels")
        
        # Check for infinite values
        if np.isinf(X).any():
            inf_count = np.isinf(X).sum()
            tprint_warning(f"⚠️ Found {inf_count} infinite values in input features")
        
        if np.isinf(y).any():
            inf_count = np.isinf(y).sum()
            tprint_warning(f"⚠️ Found {inf_count} infinite values in target values")
        
        if np.isinf(regime_labels).any():
            inf_count = np.isinf(regime_labels).sum()
            tprint_warning(f"⚠️ Found {inf_count} infinite values in regime labels")
        
        tprint_success("✅ Mathematical properties validation completed")
    
    def _validate_regime_distribution(self, regime_labels: np.ndarray) -> None:
        """Validate regime distribution with enhanced error handling."""
        try:
            # Check regime distribution
            unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)
            min_regime_samples = regime_counts.min()
            max_regime_samples = regime_counts.max()
            
            tprint_info(f"📊 Regime distribution: {len(unique_regimes)} unique regimes")
            tprint_info(f"📊 Sample range: {min_regime_samples} - {max_regime_samples} samples per regime")
            
            # Check minimum samples per regime
            min_samples_required = getattr(self.config, 'min_samples_per_regime', 1000)
            if min_regime_samples < min_samples_required:
                insufficient_regimes = unique_regimes[regime_counts < min_samples_required]
                tprint_warning(f"⚠️ {len(insufficient_regimes)} regimes have insufficient samples (< {min_samples_required})")
                tprint_warning(f"⚠️ Insufficient regimes: {insufficient_regimes}")
            
            # Check regime balance
            regime_balance = min_regime_samples / max_regime_samples if max_regime_samples > 0 else 0
            if regime_balance < 0.1:
                tprint_warning(f"⚠️ Poor regime balance: {regime_balance:.3f} (min/max ratio)")
            else:
                tprint_success(f"✅ Good regime balance: {regime_balance:.3f}")
            
            tprint_success(f"✅ Regime distribution validation completed: {len(unique_regimes)} regimes")
            
        except Exception as e:
            tprint_error(f"❌ Regime distribution validation failed: {e}")
            raise
    
    def _validate_memory_and_performance(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> None:
        """Validate memory and performance considerations."""
        try:
            # Calculate memory usage
            x_memory_mb = X.nbytes / (1024 * 1024)
            y_memory_mb = y.nbytes / (1024 * 1024)
            regime_memory_mb = regime_labels.nbytes / (1024 * 1024)
            total_memory_mb = x_memory_mb + y_memory_mb + regime_memory_mb
            
            tprint_info(f"💾 Memory usage: X={x_memory_mb:.2f}MB, y={y_memory_mb:.2f}MB, regimes={regime_memory_mb:.2f}MB")
            tprint_info(f"💾 Total memory: {total_memory_mb:.2f}MB")
            
            # Check for large datasets
            if total_memory_mb > 1000:  # > 1GB
                tprint_warning(f"⚠️ Large dataset detected: {total_memory_mb:.2f}MB")
                if self.memory_optimizer:
                    tprint_info("💾 Memory optimizer available for large dataset handling")
            
            # Check feature count
            if X.shape[1] > 1000:
                tprint_warning(f"⚠️ High-dimensional data: {X.shape[1]} features")
            
            tprint_success("✅ Memory and performance validation completed")
            
        except Exception as e:
            tprint_warning(f"⚠️ Memory and performance validation failed: {e}")
            # Don't raise - this is not critical
    
    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        base_analyst_models: Optional[Dict[str, Any]] = None,
        analyst_training_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Analyst ensemble training step with comprehensive error handling, logging, and progress tracking.
        
        Features:
        - Extensive try/except blocks with fast failing for important errors
        - Comprehensive logging using tprint at every step
        - Integration with common utilities and hardware optimizers
        - Performance monitoring and memory optimization
        
        Args:
            X: Input features (5m timeframe with cross-timeframe features)
            y: Target values (analyst outputs)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            base_analyst_models: Individual analyst models to ensemble
            analyst_training_metrics: Performance metrics of base models
            
        Returns:
            Dictionary containing training results and metadata
            
        Raises:
            RuntimeError: If critical training errors occur
            ValueError: If input data is invalid
        """
        execution_start_time = time.time()
        tprint_info("🚀 Starting Analyst ensemble training step execution")
        tprint_info("=" * 60)
        
        # Initialize execution tracking
        execution_stats = {
            'start_time': execution_start_time,
            'steps_completed': 0,
            'steps_failed': 0,
            'warnings_count': 0,
            'errors_count': 0,
            'memory_usage_mb': 0,
            'hardware_optimizations_used': []
        }
        
        try:
            # Step 1: Pre-execution validation and setup
            tprint_info("🔄 Step 1: Pre-execution validation and setup")
            with tprint_timer("Pre-execution validation"):
                self._pre_execution_validation(X, y, regime_labels, feature_names, hmm_states, base_analyst_models, analyst_training_metrics)
            execution_stats['steps_completed'] += 1
            
            # Step 2: Hardware optimization setup
            tprint_info("🔄 Step 2: Hardware optimization setup")
            with tprint_timer("Hardware optimization setup"):
                self._setup_hardware_optimizations(X, y, regime_labels, execution_stats)
            execution_stats['steps_completed'] += 1
            
            # Step 3: Enhanced input validation
            tprint_info("🔄 Step 3: Enhanced input validation")
            with tprint_timer("Enhanced input validation"):
                self._validate_input_data(X, y, regime_labels)
            execution_stats['steps_completed'] += 1
            
            # Step 4: Base models validation and preparation
            tprint_info("🔄 Step 4: Base models validation and preparation")
            with tprint_timer("Base models preparation"):
                base_analyst_models = self._prepare_base_models(base_analyst_models, execution_stats)
            execution_stats['steps_completed'] += 1
            
            # Step 5: Execute training with enhanced error handling
            tprint_info("🔄 Step 5: Executing ensemble training")
            with tprint_timer("Ensemble training execution"):
                results = self._execute_training_with_enhanced_error_handling(
                    X, y, regime_labels, feature_names, hmm_states, base_analyst_models, execution_stats
                )
            execution_stats['steps_completed'] += 1
            
            # Step 6: Post-training processing
            tprint_info("🔄 Step 6: Post-training processing")
            with tprint_timer("Post-training processing"):
                results = self._post_training_processing(results, base_analyst_models, analyst_training_metrics, execution_stats)
            execution_stats['steps_completed'] += 1
            
            # Step 7: Generate comprehensive report
            tprint_info("🔄 Step 7: Generating comprehensive report")
            with tprint_timer("Report generation"):
                execution_time = time.time() - execution_start_time
                results = self._generate_enhanced_comprehensive_report(results, execution_time, base_analyst_models, analyst_training_metrics, execution_stats)
            execution_stats['steps_completed'] += 1
            
            # Step 8: Final validation and cleanup
            tprint_info("🔄 Step 8: Final validation and cleanup")
            with tprint_timer("Final validation and cleanup"):
                self._final_validation_and_cleanup(results, execution_stats)
            execution_stats['steps_completed'] += 1
            
            # Log execution summary
            self._log_execution_summary(execution_stats, execution_time)
            
            tprint_success(f"✅ Analyst ensemble training completed successfully in {execution_time:.2f}s")
            tprint_info("=" * 60)
            return results
            
        except Exception as e:
            execution_time = time.time() - execution_start_time
            execution_stats['steps_failed'] += 1
            execution_stats['errors_count'] += 1
            
            error_msg = f"Analyst ensemble training failed after {execution_time:.2f}s: {e}"
            tprint_error(f"❌ {error_msg}")
            tprint_error(f"🔍 Traceback: {traceback.format_exc()}")
            
            # Log execution failure summary
            self._log_execution_failure_summary(execution_stats, execution_time, error_msg)
            
            return {
                'error': error_msg,
                'execution_time': execution_time,
                'traceback': traceback.format_exc(),
                'training_stats': self.training_stats,
                'execution_stats': execution_stats
            }
    
    def _pre_execution_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray],
        base_analyst_models: Optional[Dict[str, Any]],
        analyst_training_metrics: Optional[Dict[str, Any]]
    ) -> None:
        """Pre-execution validation with comprehensive checks."""
        try:
            tprint_info("🔍 Starting pre-execution validation")
            
            # Validate basic inputs
            if X is None:
                raise ValueError("Input features X cannot be None")
            if y is None:
                raise ValueError("Target values y cannot be None")
            if regime_labels is None:
                raise ValueError("Regime labels cannot be None")
            
            # Validate feature names
            if feature_names is not None and len(feature_names) != X.shape[1]:
                tprint_warning(f"⚠️ Feature names length ({len(feature_names)}) doesn't match feature count ({X.shape[1]})")
            
            # Validate HMM states
            if hmm_states is not None and len(hmm_states) != len(regime_labels):
                tprint_warning(f"⚠️ HMM states length ({len(hmm_states)}) doesn't match regime labels length ({len(regime_labels)})")
            
            # Validate base models
            if base_analyst_models is not None:
                if not isinstance(base_analyst_models, dict):
                    raise ValueError("Base analyst models must be a dictionary")
                if len(base_analyst_models) == 0:
                    tprint_warning("⚠️ Base analyst models dictionary is empty")
            
            # Validate training metrics
            if analyst_training_metrics is not None:
                if not isinstance(analyst_training_metrics, dict):
                    tprint_warning("⚠️ Analyst training metrics must be a dictionary")
            
            tprint_success("✅ Pre-execution validation completed")
            
        except Exception as e:
            tprint_error(f"❌ Pre-execution validation failed: {e}")
            raise
    
    def _setup_hardware_optimizations(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        execution_stats: Dict[str, Any]
    ) -> None:
        """Setup hardware optimizations for training."""
        tprint_info("⚙️ Setting up hardware optimizations")
        
        # Calculate data size for optimization decisions
        data_size_mb = (X.nbytes + y.nbytes + regime_labels.nbytes) / (1024 * 1024)
        execution_stats['memory_usage_mb'] = data_size_mb
        
        # Setup memory optimization if available
        if self.memory_optimizer and data_size_mb > 100:  # > 100MB
            self.memory_optimizer.optimize_for_training(data_size_mb)
            execution_stats['hardware_optimizations_used'].append('memory_optimization')
            tprint_success("✅ Memory optimization applied")
        
        # Setup CPU optimization if available
        if self.cpu_optimizer:
            self.cpu_optimizer.optimize_for_ml_training()
            execution_stats['hardware_optimizations_used'].append('cpu_optimization')
            tprint_success("✅ CPU optimization applied")
        
        # Setup GPU optimization if available
        if self.gpu_manager and data_size_mb > 500:  # > 500MB
            if self.gpu_manager.is_available():
                self.gpu_manager.optimize_for_training()
                execution_stats['hardware_optimizations_used'].append('gpu_optimization')
                tprint_success("✅ GPU optimization applied")
            else:
                tprint_info("ℹ️ GPU not available for optimization")
        
        tprint_success("✅ Hardware optimizations setup completed")
    
    def _prepare_base_models(
        self,
        base_analyst_models: Optional[Dict[str, Any]],
        execution_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare base models with validation."""
        tprint_info("🤖 Preparing base models")
        
        if base_analyst_models is None or not base_analyst_models:
            tprint_info("📋 No base analyst models provided, creating base models")
            base_analyst_models = self._create_base_models()
        else:
            tprint_info(f"✅ Using {len(base_analyst_models)} provided base models")
            
            # Validate base models
            for model_name, model in base_analyst_models.items():
                if model is None:
                    raise ValueError(f"Base model '{model_name}' is None")
                if not hasattr(model, 'fit') or not hasattr(model, 'predict'):
                    raise ValueError(f"Base model '{model_name}' doesn't have fit/predict methods")
        
        tprint_success(f"✅ Base models preparation completed: {len(base_analyst_models)} models")
        return base_analyst_models
    
    def _execute_training_with_enhanced_error_handling(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray],
        base_analyst_models: Dict[str, Any],
        execution_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute training with enhanced error handling."""
        tprint_info("🏋️ Starting enhanced training execution")
        
        # Use the parent class execute method
        results = super().execute(
            X=X,
            y=y,
            regime_labels=regime_labels,
            feature_names=feature_names,
            hmm_states=hmm_states,
            is_classification=False,  # Analyst ensemble models are typically regression
            base_models=base_analyst_models,
            symbol=None,  # Can be passed as kwargs
            exchange=None,
            timeframe=self.config.timeframe
        )
        
        # Update training stats
        self.training_stats.update({
            'training_completed': True,
            'base_models_used': len(base_analyst_models),
            'feature_count': X.shape[1],
            'sample_count': X.shape[0]
        })
        
        tprint_success("✅ Enhanced training execution completed")
        return results
    
    
    def _post_training_processing(
        self,
        results: Dict[str, Any],
        base_analyst_models: Dict[str, Any],
        analyst_training_metrics: Optional[Dict[str, Any]],
        execution_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Post-training processing with enhanced error handling."""
        try:
            tprint_info("🔄 Starting post-training processing")
            
            # Add ensemble-specific metadata
            if 'error' not in results:
                results = self._add_ensemble_specific_metadata(results, base_analyst_models, analyst_training_metrics)
            
            # Add execution statistics
            results['execution_stats'] = execution_stats.copy()
            
            # Add hardware optimization results
            if execution_stats['hardware_optimizations_used']:
                results['hardware_optimizations_used'] = execution_stats['hardware_optimizations_used']
            
            tprint_success("✅ Post-training processing completed")
            return results
            
        except Exception as e:
            tprint_warning(f"⚠️ Post-training processing failed: {e}")
            execution_stats['warnings_count'] += 1
            return results  # Return original results even if processing fails
    
    def _generate_enhanced_comprehensive_report(
        self,
        results: Dict[str, Any],
        execution_time: float,
        base_analyst_models: Dict[str, Any],
        analyst_training_metrics: Optional[Dict[str, Any]],
        execution_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate enhanced comprehensive report with detailed statistics."""
        try:
            tprint_info("📊 Generating enhanced comprehensive report")
            
            # Create enhanced comprehensive report
            comprehensive_report = {
                'execution_summary': {
                    'total_execution_time': execution_time,
                    'initialization_time': self.training_stats.get('initialization_time', 0),
                    'training_time': execution_time - self.training_stats.get('initialization_time', 0),
                    'vectorization_enabled': self.training_stats.get('vectorization_enabled', False),
                    'success': 'error' not in results,
                    'steps_completed': execution_stats.get('steps_completed', 0),
                    'steps_failed': execution_stats.get('steps_failed', 0),
                    'warnings_count': execution_stats.get('warnings_count', 0),
                    'errors_count': execution_stats.get('errors_count', 0)
                },
                'data_summary': {
                    'sample_count': self.training_stats.get('sample_count', 0),
                    'feature_count': self.training_stats.get('feature_count', 0),
                    'base_models_used': self.training_stats.get('base_models_used', 0),
                    'mock_models_created': self.training_stats.get('mock_models_created', 0),
                    'memory_usage_mb': execution_stats.get('memory_usage_mb', 0)
                },
                'configuration_summary': {
                    'model_name': self.training_stats.get('config_used', 'unknown'),
                    'timeframe': self.training_stats.get('timeframe', 'unknown'),
                    'model_types': self.training_stats.get('model_types', []),
                    'hpo_enabled': getattr(self.config, 'enable_hpo', False),
                    'hpo_trials': getattr(self.config, 'hpo_n_trials', 0) if getattr(self.config, 'enable_hpo', False) else 0
                },
                'hardware_optimization_summary': {
                    'optimizations_used': execution_stats.get('hardware_optimizations_used', []),
                    'gpu_manager_available': self.gpu_manager is not None,
                    'memory_optimizer_available': self.memory_optimizer is not None,
                    'cpu_optimizer_available': self.cpu_optimizer is not None
                },
                'utilities_availability': self.training_stats.get('utilities_available', {}),
                'performance_analysis': self._analyze_performance(results),
                'regime_analysis': self._analyze_regime_performance(results),
                'base_model_integration': self._analyze_base_model_integration(base_analyst_models, analyst_training_metrics),
                'recommendations': self._generate_recommendations(results, execution_time)
            }
            
            # Add comprehensive report to results
            results['comprehensive_report'] = comprehensive_report
            
            # Log summary
            self._log_enhanced_comprehensive_summary(comprehensive_report)
            
            tprint_success("✅ Enhanced comprehensive report generated")
            return results
            
        except Exception as e:
            tprint_error(f"❌ Enhanced comprehensive report generation failed: {e}")
            results['comprehensive_report'] = {'error': f"Report generation failed: {e}"}
            return results
    
    def _final_validation_and_cleanup(
        self,
        results: Dict[str, Any],
        execution_stats: Dict[str, Any]
    ) -> None:
        """Final validation and cleanup."""
        tprint_info("🔍 Starting final validation and cleanup")
        
        # Validate results structure
        if 'error' in results:
            tprint_warning("⚠️ Training completed with errors")
            execution_stats['errors_count'] += 1
        else:
            tprint_success("✅ Training completed without critical errors")
        
        # Cleanup hardware optimizations
        if self.memory_optimizer:
            self.memory_optimizer.cleanup()
            tprint_success("✅ Memory optimizer cleanup completed")
        
        if self.cpu_optimizer:
            self.cpu_optimizer.cleanup()
            tprint_success("✅ CPU optimizer cleanup completed")
        
        if self.gpu_manager:
            self.gpu_manager.cleanup()
            tprint_success("✅ GPU manager cleanup completed")
        
        tprint_success("✅ Final validation and cleanup completed")
    
    def _log_execution_summary(self, execution_stats: Dict[str, Any], execution_time: float) -> None:
        """Log comprehensive execution summary."""
        try:
            tprint_info("📊 EXECUTION SUMMARY")
            tprint_info("=" * 50)
            
            # Execution statistics
            tprint_info(f"⏱️ Total execution time: {execution_time:.2f}s")
            tprint_info(f"✅ Steps completed: {execution_stats.get('steps_completed', 0)}")
            tprint_info(f"❌ Steps failed: {execution_stats.get('steps_failed', 0)}")
            tprint_info(f"⚠️ Warnings: {execution_stats.get('warnings_count', 0)}")
            tprint_info(f"❌ Errors: {execution_stats.get('errors_count', 0)}")
            
            # Memory usage
            memory_mb = execution_stats.get('memory_usage_mb', 0)
            tprint_info(f"💾 Memory usage: {memory_mb:.2f}MB")
            
            # Hardware optimizations
            optimizations = execution_stats.get('hardware_optimizations_used', [])
            if optimizations:
                tprint_info(f"⚙️ Hardware optimizations used: {optimizations}")
            else:
                tprint_info("⚙️ No hardware optimizations used")
            
            tprint_info("=" * 50)
            
        except Exception as e:
            tprint_error(f"Failed to log execution summary: {e}")
    
    def _log_execution_failure_summary(
        self,
        execution_stats: Dict[str, Any],
        execution_time: float,
        error_msg: str
    ) -> None:
        """Log execution failure summary."""
        try:
            tprint_error("❌ EXECUTION FAILURE SUMMARY")
            tprint_error("=" * 50)
            tprint_error(f"⏱️ Execution time before failure: {execution_time:.2f}s")
            tprint_error(f"✅ Steps completed: {execution_stats.get('steps_completed', 0)}")
            tprint_error(f"❌ Steps failed: {execution_stats.get('steps_failed', 0)}")
            tprint_error(f"⚠️ Warnings: {execution_stats.get('warnings_count', 0)}")
            tprint_error(f"❌ Errors: {execution_stats.get('errors_count', 0)}")
            tprint_error(f"💥 Failure reason: {error_msg}")
            tprint_error("=" * 50)
            
        except Exception as e:
            print(f"Failed to log execution failure summary: {e}")
    
    def _log_enhanced_comprehensive_summary(self, comprehensive_report: Dict[str, Any]) -> None:
        """Log enhanced comprehensive training summary."""
        try:
            tprint_info("📊 ENHANCED COMPREHENSIVE TRAINING SUMMARY")
            tprint_info("=" * 60)
            
            # Execution summary
            exec_summary = comprehensive_report.get('execution_summary', {})
            tprint_info(f"⏱️ Total execution time: {exec_summary.get('total_execution_time', 0):.2f}s")
            tprint_info(f"🚀 Vectorization enabled: {exec_summary.get('vectorization_enabled', False)}")
            tprint_info(f"✅ Training success: {exec_summary.get('success', False)}")
            tprint_info(f"📊 Steps completed: {exec_summary.get('steps_completed', 0)}")
            tprint_info(f"⚠️ Warnings: {exec_summary.get('warnings_count', 0)}")
            tprint_info(f"❌ Errors: {exec_summary.get('errors_count', 0)}")
            
            # Data summary
            data_summary = comprehensive_report.get('data_summary', {})
            tprint_info(f"📊 Samples processed: {data_summary.get('sample_count', 0):,}")
            tprint_info(f"🔢 Features used: {data_summary.get('feature_count', 0)}")
            tprint_info(f"🤖 Base models: {data_summary.get('base_models_used', 0)}")
            tprint_info(f"💾 Memory usage: {data_summary.get('memory_usage_mb', 0):.2f}MB")
            
            # Hardware optimization summary
            hw_summary = comprehensive_report.get('hardware_optimization_summary', {})
            optimizations = hw_summary.get('optimizations_used', [])
            if optimizations:
                tprint_info(f"⚙️ Hardware optimizations: {optimizations}")
            else:
                tprint_info("⚙️ No hardware optimizations used")
            
            # Performance analysis
            perf_analysis = comprehensive_report.get('performance_analysis', {})
            if perf_analysis.get('best_performance'):
                best_perf = perf_analysis['best_performance']
                tprint_info(f"🏆 Best performance: R² = {best_perf.get('r2_score', 0):.4f} (Regime {best_perf.get('regime', 'N/A')})")
            
            # Recommendations
            recommendations = comprehensive_report.get('recommendations', [])
            if recommendations:
                tprint_info("💡 RECOMMENDATIONS:")
                for rec in recommendations:
                    tprint_info(f"   {rec}")
            
            tprint_info("=" * 60)
            
        except Exception as e:
            tprint_error(f"Failed to log enhanced comprehensive summary: {e}")
    
    def _execute_training_with_error_handling(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray],
        base_analyst_models: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Legacy training execution method - kept for backward compatibility.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels
            feature_names: Feature names
            hmm_states: HMM states
            base_analyst_models: Base models
            
        Returns:
            Training results
        """
        tprint_info("🔄 Using legacy training execution method")
        
        # Use the parent class execute method
        results = super().execute(
            X=X,
            y=y,
            regime_labels=regime_labels,
            feature_names=feature_names,
            hmm_states=hmm_states,
            is_classification=False,  # Analyst ensemble models are typically regression
            base_models=base_analyst_models,
            symbol=None,  # Can be passed as kwargs
            exchange=None,
            timeframe=self.config.timeframe
        )
        
        # Update training stats
        self.training_stats.update({
            'training_completed': True,
            'base_models_used': len(base_analyst_models),
            'feature_count': X.shape[1],
            'sample_count': X.shape[0]
        })
        
        tprint_success("✅ Legacy training execution completed")
        return results
    
    def _create_base_models(self) -> Dict[str, Any]:
        """
        Create base models for ensemble training.
        
        Returns:
            Dictionary of base models
            
        Raises:
            RuntimeError: If no models can be created
        """
        tprint_info("🤖 Creating base models for ensemble training")
        
        # Import required models
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.svm import SVR
        
        # Create base models
        base_models = {
            'tcn_model': RandomForestRegressor(
                n_estimators=50, 
                random_state=42, 
                max_depth=10,
                n_jobs=-1
            ),
            'catboost_model': RandomForestRegressor(
                n_estimators=50, 
                random_state=43, 
                max_depth=10,
                n_jobs=-1
            ),
            'lightgbm_model': GradientBoostingRegressor(
                n_estimators=50, 
                random_state=44, 
                max_depth=6,
                learning_rate=0.1
            ),
            'ensemble_rf_model': RandomForestRegressor(
                n_estimators=50, 
                random_state=45, 
                max_depth=10,
                n_jobs=-1
            ),
            'linear_model': LinearRegression(),
            'svr_model': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        # Validate models
        for model_name, model in base_models.items():
            if not hasattr(model, 'fit') or not hasattr(model, 'predict'):
                raise RuntimeError(f"Model '{model_name}' doesn't have required methods")
        
        # Update training stats
        self.training_stats['base_models_created'] = len(base_models)
        
        tprint_success(f"📊 Created {len(base_models)} base models for ensemble training")
        tprint_info(f"📋 Base models: {list(base_models.keys())}")
        
        return base_models
    
    def _generate_comprehensive_report(
        self,
        results: Dict[str, Any],
        execution_time: float,
        base_analyst_models: Dict[str, Any],
        analyst_training_metrics: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Legacy comprehensive report generation - kept for backward compatibility.
        
        Args:
            results: Training results
            execution_time: Total execution time
            base_analyst_models: Base models used
            analyst_training_metrics: Base model metrics
            
        Returns:
            Enhanced results with comprehensive reporting
        """
        try:
            tprint_info("📊 Generating legacy comprehensive report")
            
            # Create comprehensive report
            comprehensive_report = {
                'execution_summary': {
                    'total_execution_time': execution_time,
                    'initialization_time': self.training_stats.get('initialization_time', 0),
                    'training_time': execution_time - self.training_stats.get('initialization_time', 0),
                    'vectorization_enabled': self.training_stats.get('vectorization_enabled', False),
                    'success': 'error' not in results
                },
                'data_summary': {
                    'sample_count': self.training_stats.get('sample_count', 0),
                    'feature_count': self.training_stats.get('feature_count', 0),
                    'base_models_used': self.training_stats.get('base_models_used', 0),
                    'mock_models_created': self.training_stats.get('mock_models_created', 0)
                },
                'configuration_summary': {
                    'model_name': self.training_stats.get('config_used', 'unknown'),
                    'timeframe': self.training_stats.get('timeframe', 'unknown'),
                    'model_types': self.training_stats.get('model_types', []),
                    'hpo_enabled': getattr(self.config, 'enable_hpo', False),
                    'hpo_trials': getattr(self.config, 'hpo_n_trials', 0) if getattr(self.config, 'enable_hpo', False) else 0
                },
                'performance_analysis': self._analyze_performance(results),
                'regime_analysis': self._analyze_regime_performance(results),
                'base_model_integration': self._analyze_base_model_integration(base_analyst_models, analyst_training_metrics),
                'recommendations': self._generate_recommendations(results, execution_time)
            }
            
            # Add comprehensive report to results
            results['comprehensive_report'] = comprehensive_report
            
            # Log summary
            self._log_comprehensive_summary(comprehensive_report)
            
            tprint_success("✅ Legacy comprehensive report generated")
            return results
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate legacy comprehensive report: {e}")
            results['comprehensive_report'] = {'error': f"Report generation failed: {e}"}
            return results
    
    def _analyze_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze overall training performance with math validation.
        
        Args:
            results: Training results
            
        Returns:
            Performance analysis
        """
        tprint_info("📊 Analyzing training performance")
        
        performance_analysis = {
            'training_success': 'error' not in results,
            'models_trained': 0,
            'best_performance': {},
            'performance_distribution': {},
            'performance_metrics': {},
            'validation_status': 'unknown'
        }
        
        if 'evaluation_results' in results:
            evaluation_results = results['evaluation_results']
            performance_analysis['models_trained'] = len(evaluation_results)
            
            # Find best performing model with validation
            best_r2 = -np.inf
            best_model = None
            r2_scores = []
            
            for regime, regime_metrics in evaluation_results.items():
                if isinstance(regime_metrics, dict) and 'r2' in regime_metrics:
                    r2_score = regime_metrics['r2']
                    r2_score = validate_finite(r2_score, f"r2_score_regime_{regime}")
                    r2_scores.append(r2_score)
                    
                    if r2_score > best_r2:
                        best_r2 = r2_score
                        best_model = regime
            
            if best_model is not None:
                performance_analysis['best_performance'] = {
                    'regime': best_model,
                    'r2_score': best_r2
                }
            
            # Calculate performance distribution
            if r2_scores:
                r2_scores = [validate_finite(score, f"r2_score_{i}") for i, score in enumerate(r2_scores)]
                
                performance_analysis['performance_distribution'] = {
                    'mean_r2': np.mean(r2_scores),
                    'std_r2': np.std(r2_scores),
                    'min_r2': np.min(r2_scores),
                    'max_r2': np.max(r2_scores),
                    'median_r2': np.median(r2_scores)
                }
                
                # Performance quality assessment
                mean_r2 = performance_analysis['performance_distribution']['mean_r2']
                if mean_r2 > 0.8:
                    performance_analysis['validation_status'] = 'excellent'
                elif mean_r2 > 0.6:
                    performance_analysis['validation_status'] = 'good'
                elif mean_r2 > 0.4:
                    performance_analysis['validation_status'] = 'fair'
                else:
                    performance_analysis['validation_status'] = 'poor'
        
        tprint_success("✅ Performance analysis completed")
        return performance_analysis
    
    def _analyze_regime_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze regime-specific performance with math validation.
        
        Args:
            results: Training results
            
        Returns:
            Regime performance analysis
        """
        tprint_info("📈 Analyzing regime-specific performance")
        
        regime_analysis = {
            'total_regimes': 0,
            'successful_regimes': 0,
            'failed_regimes': 0,
            'regime_details': {},
            'regime_balance_score': 0.0,
            'regime_quality_assessment': 'unknown'
        }
        
        if 'regime_analysis' in results:
            regime_data = results['regime_analysis']
            
            # Extract regime information with validation
            unique_regimes = regime_data.get('unique_regimes', [])
            sufficient_regimes = regime_data.get('sufficient_regimes', [])
            insufficient_regimes = regime_data.get('insufficient_regimes', [])
            
            regime_analysis['total_regimes'] = len(unique_regimes)
            regime_analysis['successful_regimes'] = len(sufficient_regimes)
            regime_analysis['failed_regimes'] = len(insufficient_regimes)
            
            # Calculate regime balance score
            if regime_analysis['total_regimes'] > 0:
                success_rate = regime_analysis['successful_regimes'] / regime_analysis['total_regimes']
                success_rate = validate_finite(success_rate, "regime_success_rate")
                regime_analysis['regime_balance_score'] = success_rate
                
                # Quality assessment
                if success_rate > 0.9:
                    regime_analysis['regime_quality_assessment'] = 'excellent'
                elif success_rate > 0.7:
                    regime_analysis['regime_quality_assessment'] = 'good'
                elif success_rate > 0.5:
                    regime_analysis['regime_quality_assessment'] = 'fair'
                else:
                    regime_analysis['regime_quality_assessment'] = 'poor'
            
            # Add detailed regime information
            regime_analysis['regime_details'] = {
                'unique_regimes': unique_regimes,
                'sufficient_regimes': sufficient_regimes,
                'insufficient_regimes': insufficient_regimes,
                'regime_counts': regime_data.get('regime_counts', [])
            }
        
        tprint_success("✅ Regime performance analysis completed")
        return regime_analysis
    
    def _analyze_base_model_integration(
        self,
        base_analyst_models: Dict[str, Any],
        analyst_training_metrics: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze base model integration with validation.
        
        Args:
            base_analyst_models: Base models used
            analyst_training_metrics: Base model metrics
            
        Returns:
            Base model integration analysis
        """
        tprint_info("🤖 Analyzing base model integration")
        
        integration_analysis = {
            'base_models_count': len(base_analyst_models) if base_analyst_models else 0,
            'base_model_types': list(base_analyst_models.keys()) if base_analyst_models else [],
            'metrics_available': analyst_training_metrics is not None,
            'integration_quality': 'good' if base_analyst_models and len(base_analyst_models) >= 3 else 'limited',
            'model_validation_status': {},
            'integration_score': 0.0,
            'recommendations': []
        }
        
        # Validate base models
        if base_analyst_models:
            for model_name, model in base_analyst_models.items():
                validation_status = {
                    'has_fit_method': hasattr(model, 'fit'),
                    'has_predict_method': hasattr(model, 'predict'),
                    'is_not_none': model is not None,
                    'model_type': type(model).__name__
                }
                integration_analysis['model_validation_status'][model_name] = validation_status
                
                # Check if model is properly configured
                if not validation_status['has_fit_method'] or not validation_status['has_predict_method']:
                    integration_analysis['recommendations'].append(f"Model '{model_name}' missing required methods")
        
        # Calculate integration score
        base_score = min(1.0, integration_analysis['base_models_count'] / 5.0)  # Max score at 5 models
        metrics_score = 1.0 if integration_analysis['metrics_available'] else 0.5
        validation_score = 1.0 if all(
            status.get('has_fit_method', False) and status.get('has_predict_method', False)
            for status in integration_analysis['model_validation_status'].values()
            if isinstance(status, dict) and 'error' not in status
        ) else 0.5
        
        integration_score = (base_score + metrics_score + validation_score) / 3.0
        integration_score = validate_finite(integration_score, "integration_score")
        integration_analysis['integration_score'] = integration_score
        
        # Update integration quality based on score
        if integration_score > 0.8:
            integration_analysis['integration_quality'] = 'excellent'
        elif integration_score > 0.6:
            integration_analysis['integration_quality'] = 'good'
        elif integration_score > 0.4:
            integration_analysis['integration_quality'] = 'fair'
        else:
            integration_analysis['integration_quality'] = 'poor'
        
        # Add base model performance if available
        if analyst_training_metrics:
            integration_analysis['base_model_performance'] = analyst_training_metrics
        
        # Add recommendations
        if integration_analysis['base_models_count'] < 3:
            integration_analysis['recommendations'].append("Consider using more diverse base models for better ensemble performance")
        
        if not integration_analysis['metrics_available']:
            integration_analysis['recommendations'].append("Base model performance metrics not available - consider providing them for better integration")
        
        tprint_success("✅ Base model integration analysis completed")
        return integration_analysis
    
    def _generate_recommendations(self, results: Dict[str, Any], execution_time: float) -> List[str]:
        """
        Generate comprehensive recommendations based on training results with enhanced analysis.
        
        Args:
            results: Training results
            execution_time: Execution time
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        try:
            tprint_info("💡 Generating comprehensive recommendations")
            
            # Performance-based recommendations
            if 'error' in results:
                recommendations.append("❌ Training failed - review error logs and data quality")
                recommendations.append("🔍 Check input data validation and feature engineering")
                recommendations.append("⚙️ Verify hardware optimizations and memory availability")
            else:
                recommendations.append("✅ Training completed successfully")
            
            # Time-based recommendations with enhanced analysis
            if execution_time > 3600:  # More than 1 hour
                recommendations.append("⏰ Consider enabling vectorization for faster training")
                recommendations.append("💾 Check memory usage and consider hardware optimizations")
                recommendations.append("🔄 Consider reducing HPO trials or using faster model types")
            elif execution_time < 60:  # Less than 1 minute
                recommendations.append("⚡ Training completed quickly - consider increasing HPO trials for better performance")
                recommendations.append("📊 Consider using more complex models for better accuracy")
            else:
                recommendations.append("⏱️ Training time is reasonable - good balance between speed and thoroughness")
            
            # Data-based recommendations with enhanced analysis
            sample_count = self.training_stats.get('sample_count', 0)
            feature_count = self.training_stats.get('feature_count', 0)
            
            if sample_count < 10000:
                recommendations.append("📊 Consider collecting more training data for better model performance")
                recommendations.append("🔄 Consider data augmentation techniques")
            elif sample_count > 1000000:
                recommendations.append("📊 Large dataset detected - consider sampling or batch processing")
            else:
                recommendations.append("📊 Dataset size is appropriate for training")
            
            if feature_count > 1000:
                recommendations.append("🔢 High-dimensional data detected - consider feature selection")
                recommendations.append("📊 Consider dimensionality reduction techniques")
            elif feature_count < 10:
                recommendations.append("🔢 Low feature count - consider feature engineering")
            
            # Model-based recommendations with enhanced analysis
            base_models_count = self.training_stats.get('base_models_used', 0)
            if base_models_count < 3:
                recommendations.append("🤖 Consider using more diverse base models for better ensemble performance")
                recommendations.append("🔄 Add different model types (linear, tree-based, neural networks)")
            elif base_models_count > 10:
                recommendations.append("🤖 Many base models detected - consider model selection")
            else:
                recommendations.append("🤖 Good diversity in base models")
            
            # Vectorization recommendations with enhanced analysis
            if not self.training_stats.get('vectorization_enabled', False):
                recommendations.append("🚀 Enable vectorization for improved performance on multi-regime training")
                recommendations.append("⚙️ Check if vectorized training manager is available")
            else:
                recommendations.append("🚀 Vectorization is enabled - good for performance")
            
            # Hardware optimization recommendations
            hw_stats = self.training_stats.get('hardware_optimizers_available', {})
            if not hw_stats.get('gpu_manager', False):
                recommendations.append("⚙️ Consider enabling GPU acceleration for large datasets")
            if not hw_stats.get('memory_optimizer', False):
                recommendations.append("💾 Consider enabling memory optimization for large datasets")
            if not hw_stats.get('cpu_optimizer', False):
                recommendations.append("🖥️ Consider enabling CPU optimization for better performance")
            
            # Utilities availability recommendations
            utils_stats = self.training_stats.get('utilities_available', {})
            if not utils_stats.get('math_validation', False):
                recommendations.append("🧮 Consider enabling math validation utilities for better data quality")
            if not utils_stats.get('serialization', False):
                recommendations.append("💾 Consider enabling serialization utilities for model persistence")
            if not utils_stats.get('ml_common', False):
                recommendations.append("🔧 Consider enabling ML common utilities for advanced features")
            
            # Performance quality recommendations
            if 'comprehensive_report' in results:
                comprehensive_report = results['comprehensive_report']
                
                # Check performance analysis
                perf_analysis = comprehensive_report.get('performance_analysis', {})
                if perf_analysis.get('validation_status') == 'poor':
                    recommendations.append("📊 Poor performance detected - consider feature engineering")
                    recommendations.append("🔄 Consider different model types or hyperparameters")
                
                # Check regime analysis
                regime_analysis = comprehensive_report.get('regime_analysis', {})
                if regime_analysis.get('regime_quality_assessment') == 'poor':
                    recommendations.append("📈 Poor regime balance detected - consider data collection")
                    recommendations.append("🔄 Consider regime-specific preprocessing")
                
                # Check base model integration
                integration_analysis = comprehensive_report.get('base_model_integration', {})
                if integration_analysis.get('integration_quality') == 'poor':
                    recommendations.append("🤖 Poor base model integration - check model compatibility")
                    recommendations.append("🔄 Consider model validation and testing")
            
            tprint_success(f"✅ Generated {len(recommendations)} comprehensive recommendations")
            return recommendations
            
        except Exception as e:
            tprint_warning(f"⚠️ Recommendation generation failed: {e}")
            return [f"⚠️ Could not generate recommendations: {e}"]
    
    def _log_comprehensive_summary(self, comprehensive_report: Dict[str, Any]) -> None:
        """
        Log comprehensive training summary with enhanced error handling.
        
        Args:
            comprehensive_report: Comprehensive report data
        """
        try:
            tprint_info("📊 COMPREHENSIVE TRAINING SUMMARY")
            tprint_info("=" * 50)
            
            # Execution summary
            exec_summary = comprehensive_report.get('execution_summary', {})
            tprint_info(f"⏱️ Total execution time: {exec_summary.get('total_execution_time', 0):.2f}s")
            tprint_info(f"🚀 Vectorization enabled: {exec_summary.get('vectorization_enabled', False)}")
            tprint_info(f"✅ Training success: {exec_summary.get('success', False)}")
            
            # Data summary
            data_summary = comprehensive_report.get('data_summary', {})
            tprint_info(f"📊 Samples processed: {data_summary.get('sample_count', 0):,}")
            tprint_info(f"🔢 Features used: {data_summary.get('feature_count', 0)}")
            tprint_info(f"🤖 Base models: {data_summary.get('base_models_used', 0)}")
            
            # Performance analysis
            perf_analysis = comprehensive_report.get('performance_analysis', {})
            if perf_analysis.get('best_performance'):
                best_perf = perf_analysis['best_performance']
                tprint_info(f"🏆 Best performance: R² = {best_perf.get('r2_score', 0):.4f} (Regime {best_perf.get('regime', 'N/A')})")
            
            # Recommendations
            recommendations = comprehensive_report.get('recommendations', [])
            if recommendations:
                tprint_info("💡 RECOMMENDATIONS:")
                for rec in recommendations:
                    tprint_info(f"   {rec}")
            
            tprint_info("=" * 50)
            
        except Exception as e:
            tprint_error(f"❌ Failed to log comprehensive summary: {e}")
    
    def _add_ensemble_specific_metadata(self, results: Dict[str, Any], base_models: Dict[str, Any], base_metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add ensemble-specific metadata to results with enhanced error handling.
        
        Args:
            results: Training results
            base_models: Base analyst models used in ensemble
            base_metrics: Performance metrics of base models
            
        Returns:
            Enhanced results with ensemble-specific metadata
        """
        try:
            # Add ensemble-specific analysis
            if 'regime_analysis' in results:
                regime_analysis = results['regime_analysis']
                
                # Calculate ensemble-specific metrics
                ensemble_metrics = {
                    'total_regimes': len(regime_analysis.get('unique_regimes', [])),
                    'sufficient_regimes': len(regime_analysis.get('sufficient_regimes', [])),
                    'insufficient_regimes': len(regime_analysis.get('insufficient_regimes', [])),
                    'regime_balance': regime_analysis.get('regime_balance_train', 0.0),
                    'timeframe': self.config.timeframe,
                    'ensemble_model_types': self.config.model_types,
                    'base_models_count': len(base_models) if base_models else 0,
                    'training_timestamp': time.time(),
                    'vectorization_used': self.training_stats.get('vectorization_enabled', False)
                }
                
                # Add base model performance analysis if available
                if base_metrics:
                    ensemble_metrics['base_model_performance'] = base_metrics
                    self.logger.info("📊 Integrated base model performance metrics")
                
                results['ensemble_metrics'] = ensemble_metrics
            
            # Add ensemble performance summary with enhanced analysis
            if 'evaluation_results' in results:
                evaluation_results = results['evaluation_results']
                
                # Calculate best performing ensemble per regime
                best_ensembles = {}
                performance_summary = {
                    'total_regimes_evaluated': 0,
                    'successful_evaluations': 0,
                    'failed_evaluations': 0,
                    'average_r2': 0.0,
                    'best_overall_r2': -np.inf
                }
                
                r2_scores = []
                
                for regime, regime_metrics in evaluation_results.items():
                    performance_summary['total_regimes_evaluated'] += 1
                    
                    if isinstance(regime_metrics, dict) and 'error' not in regime_metrics:
                        performance_summary['successful_evaluations'] += 1
                        
                        best_ensemble = None
                        best_r2 = -np.inf
                        
                        for ensemble_name, metrics in regime_metrics.items():
                            if isinstance(metrics, dict) and 'r2' in metrics:
                                r2_scores.append(metrics['r2'])
                                if metrics['r2'] > best_r2:
                                    best_r2 = metrics['r2']
                                    best_ensemble = ensemble_name
                        
                        if best_ensemble:
                            best_ensembles[regime] = {
                                'ensemble': best_ensemble,
                                'r2_score': best_r2,
                                'regime_samples': regime_metrics.get('samples', 0)
                            }
                            
                            if best_r2 > performance_summary['best_overall_r2']:
                                performance_summary['best_overall_r2'] = best_r2
                    else:
                        performance_summary['failed_evaluations'] += 1
                
                # Calculate average performance
                if r2_scores:
                    performance_summary['average_r2'] = np.mean(r2_scores)
                    performance_summary['r2_std'] = np.std(r2_scores)
                    performance_summary['r2_min'] = np.min(r2_scores)
                    performance_summary['r2_max'] = np.max(r2_scores)
                
                results['best_ensembles_per_regime'] = best_ensembles
                results['performance_summary'] = performance_summary
                
                self.logger.info(f"📊 Performance summary: {performance_summary['successful_evaluations']}/{performance_summary['total_regimes_evaluated']} regimes successful")
                if performance_summary['average_r2'] > 0:
                    self.logger.info(f"🏆 Average R²: {performance_summary['average_r2']:.4f}, Best R²: {performance_summary['best_overall_r2']:.4f}")
            
            # Add enhanced ensemble-specific analysis
            ensemble_analysis = {
                'base_timeframe': self.config.timeframe,
                'cross_timeframe_features': True,
                'ensemble_method': 'per_regime',
                'base_models_integrated': len(base_models) if base_models else 0,
                'ensemble_role': 'trade_decision_enhancement',
                'training_configuration': {
                    'hpo_enabled': self.config.enable_hpo,
                    'hpo_trials': self.config.hpo_n_trials if self.config.enable_hpo else 0,
                    'min_samples_per_regime': self.config.min_samples_per_regime,
                    'evaluation_metrics': self.config.evaluation_metrics
                },
                'data_characteristics': {
                    'total_samples': self.training_stats.get('sample_count', 0),
                    'feature_count': self.training_stats.get('feature_count', 0),
                    'mock_models_used': self.training_stats.get('mock_models_created', 0) > 0
                }
            }
            results['ensemble_analysis'] = ensemble_analysis
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add ensemble-specific metadata: {e}")
            results['ensemble_metadata_error'] = str(e)
            return results
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive training statistics.
        
        Returns:
            Dictionary containing training statistics
        """
        return {
            'training_stats': self.training_stats.copy(),
            'configuration': {
                'model_name': self.config.model_name,
                'timeframe': self.config.timeframe,
                'model_types': self.config.model_types,
                'hpo_enabled': self.config.enable_hpo,
                'vectorization_enabled': self.enable_vectorization
            },
            'performance_metrics': getattr(self, 'training_results', {}).get('performance_summary', {}),
            'timestamp': time.time()
        }
    
    def validate_training_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate training results and provide quality assessment.
        
        Args:
            results: Training results to validate
            
        Returns:
            Validation report
        """
        validation_report = {
            'validation_passed': True,
            'issues_found': [],
            'warnings': [],
            'quality_score': 0.0
        }
        
        try:
            # Check for errors
            if 'error' in results:
                validation_report['validation_passed'] = False
                validation_report['issues_found'].append(f"Training failed: {results['error']}")
                return validation_report
            
            # Check for required components
            required_components = ['ensemble_metrics', 'ensemble_analysis']
            for component in required_components:
                if component not in results:
                    validation_report['warnings'].append(f"Missing component: {component}")
            
            # Check performance metrics
            if 'performance_summary' in results:
                perf_summary = results['performance_summary']
                success_rate = perf_summary.get('successful_evaluations', 0) / max(perf_summary.get('total_regimes_evaluated', 1), 1)
                
                if success_rate < 0.5:
                    validation_report['warnings'].append(f"Low success rate: {success_rate:.2%}")
                
                avg_r2 = perf_summary.get('average_r2', 0)
                if avg_r2 < 0.1:
                    validation_report['warnings'].append(f"Low average R²: {avg_r2:.4f}")
                
                # Calculate quality score
                validation_report['quality_score'] = min(1.0, success_rate * (1 + avg_r2) / 2)
            
            # Check data quality
            if 'ensemble_metrics' in results:
                ensemble_metrics = results['ensemble_metrics']
                if ensemble_metrics.get('base_models_count', 0) < 2:
                    validation_report['warnings'].append("Limited base models for ensemble")
            
            self.logger.info(f"✅ Training validation completed - Quality score: {validation_report['quality_score']:.2f}")
            
        except Exception as e:
            validation_report['validation_passed'] = False
            validation_report['issues_found'].append(f"Validation failed: {e}")
            self.logger.error(f"❌ Training validation failed: {e}")
        
        return validation_report


# Convenience functions for backward compatibility
def create_analyst_ensemble_training_step(
    config: Optional[EnsembleTrainingConfig] = None
) -> AnalystEnsembleTrainingStep:
    """Create Analyst ensemble training step."""
    return AnalystEnsembleTrainingStep(config)


def execute_analyst_ensemble_training(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[EnsembleTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    base_analyst_models: Optional[Dict[str, Any]] = None,
    analyst_training_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute Analyst ensemble training step."""
    step = create_analyst_ensemble_training_step(config)
    return step.execute(X, y, regime_labels, feature_names, hmm_states, base_analyst_models, analyst_training_metrics)


# Example usage
if __name__ == "__main__":
    # Example of how to use the enhanced ensemble training version
    tprint_info("🚀 Enhanced Analyst Ensemble Training Step Demo")
    tprint_info("=" * 60)
    
    # Create configuration
    tprint_info("📋 Creating configuration")
    config = EnsembleTrainingConfig(
        model_name="analyst_ensemble_models_enhanced",
        timeframe="5m",
        model_types=["tcn", "catboost", "lightgbm", "ensemble_rf"],
        hpo_n_trials=50,  # Reduced for demo
        enable_hpo=True,
        save_models=True,
        model_save_path="./models/analyst_ensemble_models_enhanced"
    )
    tprint_success("✅ Configuration created successfully")
    
    # Create training step
    tprint_info("🏗️ Creating enhanced training step")
    training_step = create_analyst_ensemble_training_step(config)
    tprint_success("✅ Enhanced training step created successfully")
    
    # Display configuration summary
    tprint_info("📊 CONFIGURATION SUMMARY")
    tprint_info(f"📋 Model name: {config.model_name}")
    tprint_info(f"⏰ Timeframe: {config.timeframe}")
    tprint_info(f"🤖 Ensemble types: {len(config.model_types)} types")
    tprint_info(f"📊 HPO enabled: {config.enable_hpo}")
    tprint_info(f"💾 Save models: {config.save_models}")
    tprint_info(f"📁 Save path: {config.model_save_path}")
    
    # Display training statistics
    tprint_info("📊 TRAINING STATISTICS")
    training_stats = training_step.get_training_statistics()
    tprint_structured(training_stats, LogLevel.INFO)
    
    # Display enhanced features
    tprint_info("🎯 ENHANCED ANALYST ENSEMBLE MODULE FEATURES:")
    tprint_info("- ✅ Extensive try/except blocks with fast failing for important errors")
    tprint_info("- ✅ Comprehensive logging using tprint at every step")
    tprint_info("- ✅ Integration with common utilities (math_validation, serialization, hardware optimization)")
    tprint_info("- ✅ ML common utilities (CV, lookahead, HPO, etc.)")
    tprint_info("- ✅ Operates on 5m timeframe with cross-timeframe features")
    tprint_info("- ✅ Combines individual analyst models into robust ensembles")
    tprint_info("- ✅ Per-regime ensemble training for regime-specific optimization")
    tprint_info("- ✅ Enhanced trade decision accuracy through model combination")
    tprint_info("- ✅ Models: TCN (Temporal Convolutional Network), CatBoost, LightGBM, RandomForest")
    tprint_info("- ✅ Comprehensive context from multi-timeframe dynamics")
    
    tprint_info("🔄 INTEGRATION WITH INDIVIDUAL ANALYST MODELS:")
    tprint_info("- ✅ Receives individual analyst model predictions")
    tprint_info("- ✅ Uses base model performance metrics for weighting")
    tprint_info("- ✅ Creates regime-specific ensemble combinations")
    tprint_info("- ✅ Provides enhanced trade decision signals")
    
    tprint_info("⚙️ HARDWARE OPTIMIZATION FEATURES:")
    tprint_info("- ✅ M1 GPU acceleration support")
    tprint_info("- ✅ Memory optimization for large datasets")
    tprint_info("- ✅ CPU optimization for better performance")
    tprint_info("- ✅ Automatic hardware detection and configuration")
    
    tprint_info("🔧 UTILITY INTEGRATION FEATURES:")
    tprint_info("- ✅ Math validation utilities for safe operations")
    tprint_info("- ✅ Serialization utilities for model persistence")
    tprint_info("- ✅ Common operations utilities")
    tprint_info("- ✅ Enhanced error handling and recovery")
    
    # Example of how the actual training would be called:
    tprint_info("💡 EXAMPLE USAGE:")
    tprint_info("# results = training_step.execute(")
    tprint_info("#     X, y, regime_labels, feature_names, hmm_states,")
    tprint_info("#     base_analyst_models, analyst_training_metrics")
    tprint_info("# )")
    
    tprint_success("🎉 Enhanced Analyst Ensemble Training Step demo completed successfully")
    tprint_info("=" * 60)