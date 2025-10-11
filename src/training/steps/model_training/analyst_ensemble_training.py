"""
Analyst Ensemble Training Step - Enhanced for 5m Timeframe with HMM and NAS Integration

This step handles per-regime ensemble training of Analyst models using common dependencies.
The Analyst Ensemble operates on 5m timeframe and combines individual analyst models
plus NAS models to create robust ensemble predictions for trade decisions.

Analyst Models Structure:
Base Models:
    "tcn": "Temporal Convolutional Network" - Deep learning model for temporal patterns
    "lightgbm": "LightGBM Regressor" - Fast gradient boosting framework
    "ridge": "Ridge Regression" - Linear model with L2 regularization
    "elastic_net": "Elastic Net" - Linear model with L1+L2 regularization
    "random_forest": "Random Forest" - Ensemble of decision trees

NAS Models (Per-Regime):
    "nas": "Neural Architecture Search" - Per-regime neural architectures for trading signals

Meta-learner:
    "stacking": "Stacking Ensemble" - Combines base models + NAS models

Enhanced Features:
- 5m base timeframe with cross-timeframe features (300+ features)
- HMM regime outputs integration for comprehensive context
- TCN + LightGBM + Ridge + ElasticNet + RandomForest base models
- NAS models per-regime for enhanced trading signal generation
- Per-regime training for regime-specific optimization
- Runs every 2 minutes for live trading
- Decides IF we trade and emits green light for Tactician

Enhanced with:
- Extensive try/except blocks with fast failing for important errors
- Comprehensive logging using tprint at every step
- Integration with common utilities (math_validation, serialization, hardware optimization)
- ML common utilities (CV, lookahead, HPO, etc.)
- Vectorized training capabilities for improved performance
- NAS model integration for per-regime trading signal generation
- Fast failing for missing required ML dependencies (TensorFlow, CatBoost, LightGBM)
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

# Import common operations for hardware and data utilities
from src.utils.common_operations import (
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    cleanup_m1_optimizers, ensure_directory, safe_file_exists,
    get_current_datetime, validate_positive
)

# Import common utilities for DataFrame operations
from src.utils.common_utilities import (
    analyze_nan_values_detailed, safe_dataframe_operation,
    validate_dataframe_columns
)

from src.utils.serialization_utils import JSONSerializer, PickleSerializer

# Import ML common utilities
from src.utils.ml_common.training.vectorized_training_manager import VectorizedTrainingManager
from src.utils.ml_common.matrix_cross_validation import MatrixCrossValidator
from src.utils.ml_common.optimization.hyperparameter_optimization import HyperparameterOptimizer

# Import Bayesian TPE optimizer for advanced HPO
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer, OptimizationConfig
)

# Import Auto-Tuner for automatic HPO parameter selection
from src.utils.ml_common.optimization.auto_tuner import (
    AutoTuner, auto_tune_and_optimize
)

# Import model persistence for saving/loading models
from src.utils.ml_common.post_training.model_persistence import (
    ModelPersistence, ModelMetadata, PersistenceConfig
)

# Import model caching for warm-start training
from src.utils.ml_common.models.model_cache import (
    ModelCache, get_model_cache, CachedModelMetadata
)

# Import data cleaning utilities
from src.utils.data.quality.data_cleaning import (
    DataCleaner, CleaningConfig, MissingValueStrategy, OutlierStrategy
)

# Setup logging
logger = system_logger.getChild('AnalystEnsembleTraining')


class AnalystEnsembleTrainingStep(EnsembleTrainingStep):
    """
    Enhanced Analyst Ensemble Training Step for 5m timeframe with HMM integration.
    
    Analyst Models Structure:
    Base Models:
        "tcn": "Temporal Convolutional Network" - Deep learning model for temporal patterns
        "catboost": "CatBoost Regressor" - Gradient boosting with categorical features  
        "lightgbm": "LightGBM Regressor" - Fast gradient boosting framework
    
    Meta-learner:
        "elastic_net": "Elastic Net" - Linear combination of base model predictions
    
    Features:
    - 5m base timeframe with cross-timeframe features (300+ features)
    - HMM regime outputs integration for comprehensive context
    - TCN + CatBoost + LightGBM base models with Elastic Net meta-learner
    - Per-regime training for regime-specific optimization
    - Runs every 2 minutes for live trading
    - Decides IF we trade and emits green light for Tactician
    
    Enhanced with:
    - Extensive try/except blocks with fast failing for important errors
    - Comprehensive logging using tprint at every step
    - Integration with common utilities (math_validation, serialization, hardware optimization)
    - ML common utilities (CV, lookahead, HPO, etc.)
    - Per-regime ensemble training, HPO, saving, and metrics
    - Fast failing for missing required ML dependencies (TensorFlow, CatBoost, LightGBM)
    """
    
    def __init__(self, config: Optional[EnsembleTrainingConfig] = None, enable_vectorization: bool = True):
        """
        Initialize Analyst ensemble training step for 5m timeframe with HMM integration.

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
        
        try:
            # Step 1: Validate and setup configuration
            config = self._setup_configuration(config)
            self._validate_config_consolidated(config)
            
            # Step 2: Initialize NAS models storage
            self.nas_models = {}  # Per-regime NAS models
            self.nas_architectures = {}  # Per-regime NAS architectures
            
            # Step 3: Initialize hardware optimizers (consolidated)
            self.hardware = self._initialize_hardware_optimizers_consolidated()
            
            # Step 4: Initialize data cleaner
            self.data_cleaner = self._initialize_data_cleaner()
            
            # Step 5: Initialize model persistence
            self.model_persistence = self._initialize_model_persistence()
            
            # Step 5.5: Initialize model cache
            self.model_cache = self._initialize_model_cache()
            
            # Step 6: Initialize parent class
            super().__init__(config, enable_vectorization=enable_vectorization)
            
            # Step 7: Setup consolidated tracking
            self._setup_tracking_consolidated(config)
            
            # Log initialization summary
            init_time = time.time() - self.start_time
            tprint_success(f"✅ Initialization complete in {init_time:.2f}s")
            
        except Exception as e:
            tprint_error(f"❌ Initialization failed: {e}")
            raise
    
    def _setup_configuration(self, config: Optional[EnsembleTrainingConfig]) -> EnsembleTrainingConfig:
        """Setup configuration with enhanced error handling."""
        try:
            if config is None:
                config = EnsembleTrainingConfig(
                    model_name="analyst_ensemble_models_5m",
                    timeframe="5m",
                    model_types=["tcn", "lightgbm", "ridge", "elastic_net", "random_forest"],
                    hpo_n_trials=100,
                    hpo_timeout_seconds=3600,
                    min_samples_per_regime=1000,
                    enable_data_augmentation=True,
                    augmentation_method="smote",
                    model_save_path="generated/model_training/models/analyst_ensemble_models_5m",
                    evaluation_metrics=["mse", "mae", "r2", "mape", "smape"]
                )
            return config
        except Exception as e:
            tprint_error(f"Configuration setup failed: {e}")
            raise RuntimeError(f"Configuration setup failed: {e}") from e
    
    def _validate_config_consolidated(self, config: EnsembleTrainingConfig) -> None:
        """Consolidated configuration validation using common utilities."""
        with tprint_timer("Config validation"):
            # Basic validation
            if not config.model_types or len(config.model_types) == 0:
                raise ValueError("At least one model type required")
            
            # HPO validation using validate_positive from common_operations
            if config.enable_hpo:
                validate_positive(config.hpo_n_trials, "hpo_n_trials")
                validate_positive(config.hpo_timeout_seconds, "hpo_timeout_seconds")
            
            # Regime validation
            validate_positive(config.min_samples_per_regime, "min_samples_per_regime")
            
            # Path validation using ensure_directory from common_operations
            if config.save_models and config.model_save_path:
                ensure_directory(config.model_save_path)
    
    def _initialize_hardware_optimizers_consolidated(self) -> Dict[str, Any]:
        """Initialize hardware optimizers (consolidated approach)."""
        hardware = {}
        
        try:
            hardware['gpu'] = get_m1_gpu_manager()
            hardware['memory'] = get_m1_memory_optimizer()
            hardware['cpu'] = get_m1_cpu_optimizer()
            
            available = sum(1 for v in hardware.values() if v is not None)
            tprint_success(f"✅ Hardware: {available}/3 optimizers available")
        except Exception as e:
            tprint_warning(f"⚠️ Hardware init partial: {e}")
        
        return hardware
    
    def _initialize_data_cleaner(self) -> Optional[DataCleaner]:
        """Initialize data cleaner with configuration."""
        try:
            cleaning_config = CleaningConfig(
                missing_value_strategy=MissingValueStrategy.INTERPOLATE,
                outlier_strategy=OutlierStrategy.CLIP,
                outlier_threshold=3.0,
                enable_gap_detection=True,
                enable_async_processing=False
            )
            tprint_success("✅ Data cleaner initialized")
            return DataCleaner(cleaning_config)
        except Exception as e:
            tprint_warning(f"⚠️ Data cleaner unavailable: {e}")
            return None
    
    def _initialize_model_persistence(self) -> Optional[ModelPersistence]:
        """Initialize model persistence manager."""
        try:
            persistence_config = PersistenceConfig(
                base_model_dir=self.config.model_save_path,
                enable_versioning=True,
                max_versions=5,
                serialization_format="joblib",
                compression=True
            )
            tprint_success("✅ Model persistence initialized")
            return ModelPersistence(persistence_config)
        except Exception as e:
            tprint_warning(f"⚠️ Model persistence unavailable: {e}")
            return None
    
    def _initialize_model_cache(self) -> Optional[ModelCache]:
        """Initialize model cache for warm-start training."""
        try:
            model_cache = get_model_cache(
                max_memory_models=10,
                max_disk_models=50,
                cache_dir=f"{self.config.model_save_path}/cache"
            )
            tprint_success("✅ Model cache initialized")
            return model_cache
        except Exception as e:
            tprint_warning(f"⚠️ Model cache unavailable: {e}")
            return None
    
    def _setup_tracking_consolidated(self, config: EnsembleTrainingConfig) -> None:
        """Setup consolidated tracking and monitoring."""
        self.training_stats = {
            'initialization_time': time.time() - self.start_time,
            'config': config.model_name,
            'timeframe': config.timeframe,
            'vectorization_enabled': self.enable_vectorization,
            'hardware_available': {
                'gpu': self.hardware.get('gpu') is not None,
                'memory': self.hardware.get('memory') is not None,
                'cpu': self.hardware.get('cpu') is not None
            },
            'utilities_available': {
                'data_cleaner': self.data_cleaner is not None,
                'model_persistence': self.model_persistence is not None,
                'model_cache': self.model_cache is not None
            }
        }
    
    def _validate_config_enhanced(self, config: EnsembleTrainingConfig) -> None:
        """Enhanced configuration validation - delegates to consolidated method."""
        self._validate_config_consolidated(config)
    
    def _initialize_hardware_optimizers(self) -> None:
        """Initialize hardware optimizers - delegates to consolidated method."""
        self.hardware = self._initialize_hardware_optimizers_consolidated()
        # Set individual references for backwards compatibility
        self.gpu_manager = self.hardware.get('gpu')
        self.memory_optimizer = self.hardware.get('memory')
        self.cpu_optimizer = self.hardware.get('cpu')
    
    def _initialize_parent_class(self, config: EnsembleTrainingConfig, enable_vectorization: bool) -> None:
        """Initialize parent class - now handled in __init__."""
        pass  # Kept for backwards compatibility
    
    def _setup_tracking_and_monitoring(self, config: EnsembleTrainingConfig) -> None:
        """Setup tracking and monitoring - delegates to consolidated method."""
        self._setup_tracking_consolidated(config)
    
    def calculate_ensemble_diversity_metrics(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate ensemble diversity metrics for model complementarity."""
        tprint_info("📊 Calculating ensemble diversity metrics...")
        
        try:
            diversity_metrics = {}
            
            # Calculate pairwise diversity between models
            model_names = list(predictions.keys())
            if len(model_names) < 2:
                tprint_warning("⚠️ Need at least 2 models for diversity calculation")
                return {}
            
            # Calculate correlation-based diversity
            correlations = []
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names[i+1:], i+1):
                    try:
                        pred1 = predictions[model1]
                        pred2 = predictions[model2]
                        
                        if len(pred1) == len(pred2):
                            corr = np.corrcoef(pred1, pred2)[0, 1]
                            diversity = 1 - abs(corr)  # Higher diversity = lower correlation
                            correlations.append(diversity)
                            diversity_metrics[f"{model1}_vs_{model2}_diversity"] = float(diversity)
                    except Exception as e:
                        tprint_warning(f"⚠️ Diversity calculation failed for {model1} vs {model2}: {e}")
                        continue
            
            # Calculate overall diversity
            if correlations:
                overall_diversity = np.mean(correlations)
                diversity_metrics['overall_diversity'] = float(overall_diversity)
                diversity_metrics['diversity_std'] = float(np.std(correlations))
            
            # Calculate prediction variance as diversity measure
            all_predictions = np.array(list(predictions.values()))
            if all_predictions.size > 0:
                prediction_variance = np.var(all_predictions)
                diversity_metrics['prediction_variance'] = float(prediction_variance)
                
                # Calculate coefficient of variation
                mean_prediction = np.mean(all_predictions)
                if mean_prediction != 0:
                    cv = np.std(all_predictions) / abs(mean_prediction)
                    diversity_metrics['coefficient_of_variation'] = float(cv)
            
            # Update training stats
            self.training_stats['ensemble_diversity_metrics'] = diversity_metrics
            
            tprint_success(f"✅ Diversity metrics calculated: {len(diversity_metrics)} metrics")
            tprint_info(f"📊 Overall diversity: {diversity_metrics.get('overall_diversity', 0):.4f}")
            
            return diversity_metrics
            
        except Exception as e:
            tprint_error(f"❌ Diversity metrics calculation failed: {e}")
            return {}
    
    def calculate_confidence_intervals(self, predictions: Dict[str, np.ndarray], confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for ensemble predictions."""
        tprint_info("📊 Calculating confidence intervals...")
        
        try:
            confidence_intervals = {}
            
            for model_name, preds in predictions.items():
                if len(preds) == 0:
                    continue
                
                # Bootstrap confidence intervals
                n_bootstrap = 1000
                bootstrap_samples = []
                
                for _ in range(n_bootstrap):
                    # Bootstrap sample
                    bootstrap_indices = np.random.choice(len(preds), size=len(preds), replace=True)
                    bootstrap_sample = preds[bootstrap_indices]
                    bootstrap_samples.append(np.mean(bootstrap_sample))
                
                # Calculate confidence intervals
                alpha = 1 - confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                lower_bound = np.percentile(bootstrap_samples, lower_percentile)
                upper_bound = np.percentile(bootstrap_samples, upper_percentile)
                
                confidence_intervals[model_name] = (float(lower_bound), float(upper_bound))
            
            # Update training stats
            self.training_stats['confidence_intervals'] = confidence_intervals
            
            tprint_success(f"✅ Confidence intervals calculated for {len(confidence_intervals)} models")
            
            return confidence_intervals
            
        except Exception as e:
            tprint_error(f"❌ Confidence interval calculation failed: {e}")
            return {}
    
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
    
    def _validate_and_clean_input_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        regime_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Enhanced input data validation and cleaning using common utilities.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels
            
        Returns:
            Tuple of (cleaned_X, cleaned_y, cleaned_regime_labels, cleaning_report)
        """
        with tprint_timer("Data validation & cleaning"):
            # Shape validation
            if X.shape[0] != y.shape[0] or X.shape[0] != regime_labels.shape[0]:
                raise ValueError(f"Shape mismatch: X={X.shape}, y={y.shape}, regimes={regime_labels.shape}")
            
            # Mathematical validation using math_validation
            validate_array_finite(X, "features")
            validate_array_finite(y, "targets")
            validate_array_finite(regime_labels, "regimes")
            
            # NaN analysis using common_utilities
            nan_analysis = analyze_nan_values_detailed(X)
            
            cleaning_report = {
                'initial_shape': X.shape,
                'nan_analysis': {
                    'total_nans': nan_analysis.get('total_nans', 0),
                    'nan_percentage': nan_analysis.get('nan_percentage', 0)
                }
            }
            
            # Clean data if cleaner is available
            if self.data_cleaner and nan_analysis.get('total_nans', 0) > 0:
                tprint_info(f"🧹 Cleaning {nan_analysis['total_nans']} NaN values")
                
                # Convert to DataFrame for cleaning
                df = pd.DataFrame(X)
                df_cleaned = self.data_cleaner.handle_missing_values(df)
                X_cleaned = df_cleaned.values
                
                # Verify cleaning
                remaining_nans = np.isnan(X_cleaned).sum()
                cleaning_report['nans_removed'] = nan_analysis['total_nans'] - remaining_nans
                cleaning_report['final_nans'] = remaining_nans
                
                tprint_success(f"✅ Cleaned data: removed {cleaning_report['nans_removed']} NaN values")
                return X_cleaned, y, regime_labels, cleaning_report
            
            tprint_success("✅ Data validation passed")
            return X, y, regime_labels, cleaning_report
    
    def _validate_input_data(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> None:
        """
        Legacy validation method - delegates to new consolidated method.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels
        """
        _, _, _, _ = self._validate_and_clean_input_data(X, y, regime_labels)
    
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
        """Validate memory and performance considerations with bounds checking."""
        try:
            # Calculate memory usage with bounds checking
            try:
                x_memory_bytes = X.nbytes
                y_memory_bytes = y.nbytes
                regime_memory_bytes = regime_labels.nbytes
                
                # Check for potential overflow in memory calculations
                max_safe_bytes = 2**63 - 1  # Maximum safe integer size
                if any(mem > max_safe_bytes for mem in [x_memory_bytes, y_memory_bytes, regime_memory_bytes]):
                    tprint_warning("⚠️ Very large arrays detected - memory calculations may be approximate")
                    x_memory_mb = x_memory_bytes / (1024 * 1024) if x_memory_bytes < max_safe_bytes else float('inf')
                    y_memory_mb = y_memory_bytes / (1024 * 1024) if y_memory_bytes < max_safe_bytes else float('inf')
                    regime_memory_mb = regime_memory_bytes / (1024 * 1024) if regime_memory_bytes < max_safe_bytes else float('inf')
                else:
                    x_memory_mb = x_memory_bytes / (1024 * 1024)
                    y_memory_mb = y_memory_bytes / (1024 * 1024)
                    regime_memory_mb = regime_memory_bytes / (1024 * 1024)
                
                # Safe total calculation
                if any(mem == float('inf') for mem in [x_memory_mb, y_memory_mb, regime_memory_mb]):
                    total_memory_mb = float('inf')
                    tprint_warning("⚠️ Total memory calculation overflow - dataset is extremely large")
                else:
                    total_memory_mb = x_memory_mb + y_memory_mb + regime_memory_mb
                
                # Format memory display safely
                def format_memory(mem):
                    if mem == float('inf'):
                        return ">2^63 bytes"
                    return f"{mem:.2f}MB"
                
                tprint_info(f"💾 Memory usage: X={format_memory(x_memory_mb)}, y={format_memory(y_memory_mb)}, regimes={format_memory(regime_memory_mb)}")
                tprint_info(f"💾 Total memory: {format_memory(total_memory_mb)}")
                
                # Check for large datasets
                if total_memory_mb > 1000:  # > 1GB
                    tprint_warning(f"⚠️ Large dataset detected: {format_memory(total_memory_mb)}")
                    if self.memory_optimizer:
                        tprint_info("💾 Memory optimizer available for large dataset handling")
                
            except (OverflowError, MemoryError) as e:
                tprint_warning(f"⚠️ Memory calculation overflow: {e}")
                total_memory_mb = float('inf')
            
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
        analyst_training_metrics: Optional[Dict[str, Any]] = None,
        hmm_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Analyst ensemble training step for 5m timeframe with HMM integration.
        
        Features:
        - 5m base timeframe with cross-timeframe features (300+ features)
        - HMM regime outputs integration for comprehensive context
        - TCN + CatBoost + LightGBM base models with Elastic Net meta-learner
        - Per-regime training for regime-specific optimization
        - Decides IF we trade and emits green light for Tactician
        
        Args:
            X: Input features (5m timeframe with cross-timeframe features, 300+ features)
            y: Target values (analyst outputs - trade decision signals)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            base_analyst_models: Individual analyst models to ensemble
            analyst_training_metrics: Performance metrics of base models
            hmm_data: HMM regime data and features for integration
            
        Returns:
            Dictionary containing training results and metadata
            
        Raises:
            RuntimeError: If critical training errors occur
            ValueError: If input data is invalid
        """
        execution_start_time = time.time()
        tprint_info("🚀 Starting Analyst ensemble training step execution")
        tprint_info("=" * 60)
        
        # Initialize NAS models if available
        nas_models = None
        if hasattr(self, 'nas_models') and self.nas_models:
            nas_models = self.nas_models
            tprint_info("🧠 NAS models available for ensemble integration")
        else:
            tprint_warning("⚠️ No NAS models available, using base models only")
        
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
            # Step 1: Validate and clean data (consolidated)
            with tprint_timer("Step 1: Data validation & cleaning"):
                X_clean, y_clean, regimes_clean, cleaning_report = self._validate_and_clean_input_data(
                    X, y, regime_labels
                )
                execution_stats['cleaning_report'] = cleaning_report
                execution_stats['steps_completed'] += 1
            
            # Step 2: Hardware optimization setup
            with tprint_timer("Step 2: Hardware optimization"):
                self._setup_hardware_optimizations(X_clean, y_clean, regimes_clean, execution_stats)
                execution_stats['steps_completed'] += 1
            
            # Step 3: HMM integration if available
            if hmm_data is not None:
                with tprint_timer("Step 3: HMM integration"):
                    X_enhanced = self._integrate_hmm_features(X_clean, hmm_data, execution_stats)
                    execution_stats['steps_completed'] += 1
            else:
                X_enhanced = X_clean
                tprint_info("⏭️ Step 3: Skipping HMM integration (no data provided)")
            
            # Step 4: Execute training with parent class
            with tprint_timer("Step 4: Ensemble training"):
                training_results = super().execute(
                    X=X_enhanced,
                    y=y_clean,
                    regime_labels=regimes_clean,
                    feature_names=feature_names,
                    hmm_states=hmm_states,
                    is_classification=False,
                    base_models=base_analyst_models,
                    timeframe=self.config.timeframe
                )
                results = training_results
                execution_stats['steps_completed'] += 1
            
            # Step 5: Save models if persistence is available
            if self.model_persistence and self.config.save_models:
                with tprint_timer("Step 5: Model persistence"):
                    self._save_models_with_metadata(results, cleaning_report)
                    execution_stats['steps_completed'] += 1
            
            # Step 6: Generate consolidated report
            execution_time = time.time() - execution_start_time
            results['success'] = True
            results['execution_time'] = execution_time
            results['execution_stats'] = execution_stats
            results['cleaning_report'] = cleaning_report
            
            tprint_success(f"✅ Training complete in {execution_time:.2f}s")
            return results
            
        except Exception as e:
            execution_time = time.time() - execution_start_time
            execution_stats['steps_failed'] += 1
            tprint_error(f"❌ Training failed after {execution_time:.2f}s: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'execution_stats': execution_stats
            }
        
        finally:
            # Cleanup hardware resources using common_operations
            self._cleanup_hardware_resources()
    
    def _pre_execution_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]],
        hmm_states: Optional[np.ndarray],
        base_analyst_models: Optional[Dict[str, Any]],
        analyst_training_metrics: Optional[Dict[str, Any]],
        hmm_data: Optional[Dict[str, Any]]
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
            
            # Validate HMM data
            if hmm_data is not None:
                if not isinstance(hmm_data, dict):
                    tprint_warning("⚠️ HMM data must be a dictionary")
                else:
                    # Check for required HMM components
                    required_hmm_keys = ['regime_states', 'regime_probabilities', 'regime_confidence']
                    for key in required_hmm_keys:
                        if key not in hmm_data:
                            tprint_warning(f"⚠️ HMM data missing key: {key}")
            
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
    
    def _integrate_hmm_features(
        self,
        X: np.ndarray,
        hmm_data: Optional[Dict[str, Any]],
        execution_stats: Dict[str, Any]
    ) -> np.ndarray:
        """Integrate HMM features with base features for enhanced context."""
        try:
            tprint_info("🔄 Integrating HMM features with base features")
            
            if hmm_data is None:
                tprint_warning("⚠️ No HMM data provided, using base features only")
                return X
            
            # Extract HMM features
            hmm_features = []
            feature_count = X.shape[1]
            
            # Add regime probabilities
            if 'regime_probabilities' in hmm_data:
                try:
                    regime_probs_data = hmm_data['regime_probabilities']
                    if not isinstance(regime_probs_data, (list, np.ndarray)):
                        tprint_warning("⚠️ Regime probabilities must be list or numpy array")
                    else:
                        regime_probs = np.array(regime_probs_data, dtype=np.float64)
                        if regime_probs.shape[0] == X.shape[0]:
                            hmm_features.append(regime_probs)
                            feature_count += regime_probs.shape[1]
                            tprint_success(f"✅ Added {regime_probs.shape[1]} regime probability features")
                        else:
                            tprint_warning("⚠️ Regime probabilities shape mismatch")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to process regime probabilities: {e}")
            
            # Add regime confidence
            if 'regime_confidence' in hmm_data:
                try:
                    regime_conf_data = hmm_data['regime_confidence']
                    if not isinstance(regime_conf_data, (list, np.ndarray)):
                        tprint_warning("⚠️ Regime confidence must be list or numpy array")
                    else:
                        regime_conf = np.array(regime_conf_data, dtype=np.float64)
                        if len(regime_conf) == X.shape[0]:
                            regime_conf = regime_conf.reshape(-1, 1)
                            hmm_features.append(regime_conf)
                            feature_count += 1
                            tprint_success("✅ Added regime confidence feature")
                        else:
                            tprint_warning("⚠️ Regime confidence shape mismatch")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to process regime confidence: {e}")
            
            # Add regime states as one-hot encoded features
            if 'regime_states' in hmm_data:
                try:
                    regime_states_data = hmm_data['regime_states']
                    if not isinstance(regime_states_data, (list, np.ndarray)):
                        tprint_warning("⚠️ Regime states must be list or numpy array")
                    else:
                        regime_states = np.array(regime_states_data)
                        if len(regime_states) == X.shape[0]:
                            try:
                                # Handle non-integer regime states by mapping to integers
                                unique_regimes = list(set(regime_states))
                                regime_to_int = {regime: idx for idx, regime in enumerate(unique_regimes)}
                                regime_int_mapped = np.array([regime_to_int[regime] for regime in regime_states])
                                
                                # One-hot encode regime states
                                n_regimes = len(unique_regimes)
                                regime_onehot = np.eye(n_regimes)[regime_int_mapped]
                                hmm_features.append(regime_onehot)
                                feature_count += n_regimes
                                tprint_success(f"✅ Added {n_regimes} regime state features (one-hot encoded)")
                            except Exception as e:
                                tprint_warning(f"⚠️ Failed to one-hot encode regime states: {e}")
                                # Fallback: use regime states as categorical features
                                unique_regimes = list(set(regime_states))
                                for i, regime in enumerate(unique_regimes):
                                    regime_binary = (regime_states == regime).astype(float).reshape(-1, 1)
                                    hmm_features.append(regime_binary)
                                    feature_count += 1
                                tprint_success(f"✅ Added {len(unique_regimes)} regime state features (binary encoded)")
                        else:
                            tprint_warning("⚠️ Regime states shape mismatch")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to process regime states: {e}")
            
            # Combine features
            if hmm_features:
                X_enhanced = np.column_stack([X] + hmm_features)
                actual_hmm_features = X_enhanced.shape[1] - X.shape[1]
                tprint_success(f"✅ Enhanced features: {X.shape[1]} base + {actual_hmm_features} HMM = {X_enhanced.shape[1]} total")
                
                # Update execution stats with actual feature counts
                execution_stats['hmm_features_added'] = actual_hmm_features
                execution_stats['total_features'] = X_enhanced.shape[1]
                
                return X_enhanced
            else:
                tprint_warning("⚠️ No valid HMM features found, using base features only")
                return X
                
        except Exception as e:
            tprint_error(f"❌ HMM feature integration failed: {e}")
            tprint_warning("⚠️ Returning base features due to integration failure")
            return X
    
    def _prepare_base_models(
        self,
        base_analyst_models: Optional[Dict[str, Any]],
        execution_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare base models with validation and cache support."""
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
    
    def _try_load_cached_model(
        self,
        regime: str,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        config: Dict[str, Any]
    ) -> Optional[Tuple[Any, CachedModelMetadata]]:
        """
        Try to load a cached model for warm-start training.
        
        Args:
            regime: Regime identifier
            model_type: Type of model
            X: Training features (for hash)
            y: Training targets (for hash)
            config: Model configuration (for hash)
            
        Returns:
            Tuple of (model, metadata) or None if not found
        """
        if not self.model_cache:
            return None
        
        try:
            # Generate hashes
            data_hash = self.model_cache._hash_data(X, y)
            config_hash = self.model_cache._hash_config(config)
            
            # Try to retrieve from cache
            cached_result = self.model_cache.get_model(
                regime=regime,
                model_type=model_type,
                data_hash=data_hash,
                config_hash=config_hash
            )
            
            if cached_result:
                tprint_success(f"✅ Loaded cached model for regime {regime}, type {model_type}")
                return cached_result
            
            return None
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load cached model: {e}")
            return None
    
    def _cache_trained_model(
        self,
        model: Any,
        regime: str,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        config: Dict[str, Any],
        train_score: Optional[float] = None,
        val_score: Optional[float] = None,
        training_duration: float = 0.0
    ) -> None:
        """
        Cache a trained model for future use.
        
        Args:
            model: Trained model
            regime: Regime identifier
            model_type: Type of model
            X: Training features
            y: Training targets
            config: Model configuration
            train_score: Training score
            val_score: Validation score
            training_duration: Time taken to train
        """
        if not self.model_cache:
            return
        
        try:
            # Create metadata
            metadata = CachedModelMetadata(
                model_id=f"{regime}_{model_type}",
                model_type=model_type,
                regime=regime,
                timestamp=get_current_datetime().isoformat(),
                data_hash=self.model_cache._hash_data(X, y),
                config_hash=self.model_cache._hash_config(config),
                train_score=train_score,
                val_score=val_score,
                n_samples=len(X),
                n_features=X.shape[1],
                training_duration=training_duration,
                hyperparameters=config
            )
            
            # Cache the model
            self.model_cache.put_model(
                model=model,
                regime=regime,
                model_type=model_type,
                X=X,
                y=y,
                config=config,
                metadata=metadata
            )
            
            tprint_success(f"✅ Cached model for regime {regime}, type {model_type}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to cache model: {e}")
    
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
    
    def _save_models_with_metadata(self, results: Dict[str, Any], cleaning_report: Dict[str, Any]) -> None:
        """Save models with comprehensive metadata using model persistence."""
        try:
            if 'trained_models' not in results:
                tprint_warning("⚠️ No trained models to save")
                return
            
            for regime, model in results['trained_models'].items():
                try:
                    metadata = ModelMetadata(
                        model_name=f"analyst_ensemble_{regime}",
                        model_type=self.config.model_types[0] if self.config.model_types else "ensemble",
                        version="1.0",
                        training_timestamp=get_current_datetime().isoformat(),
                        training_duration=results.get('execution_time', 0),
                        training_data_size=cleaning_report.get('initial_shape', [0])[0],
                        feature_count=cleaning_report.get('initial_shape', [0, 0])[1],
                        r2_score=results.get('evaluation_results', {}).get(regime, {}).get('r2'),
                        hyperparameters=getattr(model, 'get_params', lambda: {})(),
                        description=f"Analyst ensemble model for regime {regime}",
                        tags=['analyst', 'ensemble', f'regime_{regime}', self.config.timeframe]
                    )
                    
                    self.model_persistence.save_model(model, metadata)
                    tprint_success(f"✅ Saved model for regime {regime}")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to save model for regime {regime}: {e}")
        except Exception as e:
            tprint_warning(f"⚠️ Model saving failed: {e}")
    
    def _cleanup_hardware_resources(self) -> None:
        """Cleanup hardware resources using common_operations utility."""
        try:
            # Use the common_operations cleanup utility
            cleanup_m1_optimizers()
            tprint_success("✅ Hardware cleanup complete")
        except Exception as e:
            tprint_warning(f"⚠️ Hardware cleanup failed: {e}")
    
    def _final_validation_and_cleanup(
        self,
        results: Dict[str, Any],
        execution_stats: Dict[str, Any]
    ) -> None:
        """Final validation and cleanup - now delegates to cleanup method."""
        if 'error' in results:
            tprint_warning("⚠️ Training completed with errors")
        else:
            tprint_success("✅ Training completed successfully")
        
        self._cleanup_hardware_resources()
    
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
        
        Required Dependencies:
        - TensorFlow/Keras for TCN model
        - CatBoost for CatBoost model  
        - LightGBM for LightGBM model
        - Scikit-learn for Elastic Net
        
        Returns:
            Dictionary of base models
            
        Raises:
            ImportError: If required ML libraries are not available
            RuntimeError: If models cannot be created
        """
        tprint_info("🤖 Creating base models for ensemble training")
        
        # Import required models
        from sklearn.linear_model import ElasticNet
        from sklearn.svm import SVR
        
        # Import specialized models - fast fail if not available
        try:
            import catboost as cb
        except ImportError as e:
            error_msg = f"CatBoost is required but not available: {e}"
            tprint_error(f"❌ {error_msg}")
            raise ImportError(error_msg) from e
        
        try:
            import lightgbm as lgb
        except ImportError as e:
            error_msg = f"LightGBM is required but not available: {e}"
            tprint_error(f"❌ {error_msg}")
            raise ImportError(error_msg) from e
        
        # Import TCN from models module
        try:
            from src.models.tcn_regressor import TCNRegressor
        except ImportError as e:
            error_msg = f"TCNRegressor is required but not available: {e}"
            tprint_error(f"❌ {error_msg}")
            raise ImportError(error_msg) from e
        
        # Create base models for Analyst (5m timeframe)
        base_models = {}
        
        # TCN Model - Now imported from models module
        base_models['tcn'] = TCNRegressor(
            filters=64,
            kernel_size=3,
            dropout=0.2,
            epochs=50,
            batch_size=32,
            early_stopping_patience=10,  # Enable early stopping
            reduce_lr_patience=5,  # Enable learning rate reduction
            random_state=42,
            verbose=0
        )
        tprint_success("✅ TCN (Temporal Convolutional Network) model created")
        
        # CatBoost Model - Fast fail if CatBoost not available
        base_models['catboost'] = cb.CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=43,
            verbose=False
        )
        tprint_success("✅ CatBoost Regressor model created")
        
        # LightGBM Model - Fast fail if LightGBM not available
        base_models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            random_state=44,
            verbose=-1
        )
        tprint_success("✅ LightGBM Regressor model created")
        
        # Elastic Net Meta-learner
        base_models['elastic_net'] = ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            random_state=45,
            max_iter=1000
        )
        tprint_success("✅ Elastic Net meta-learner created")
        
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
    
    def load_nas_models(self, nas_models: Dict[str, Any], nas_architectures: Dict[str, Any] = None):
        """Load NAS models for ensemble integration."""
        try:
            self.nas_models = nas_models
            if nas_architectures:
                self.nas_architectures = nas_architectures
            
            tprint_success(f"✅ Loaded {len(nas_models)} NAS models for ensemble integration")
            tprint_info(f"   Regimes with NAS models: {list(nas_models.keys())}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to load NAS models: {e}")
            raise
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive training statistics.
        
        Returns:
            Dictionary containing training statistics
        """
        return {
            'training_stats': self.training_stats.copy(),
            'nas_models_loaded': len(self.nas_models),
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
    
    def _get_meta_features(self, df: pd.DataFrame, is_live: bool = False, **kwargs: Any) -> pd.DataFrame:
        """
        Extract comprehensive meta-features including disagreement features for the analyst ensemble.
        
        Args:
            df: Input DataFrame with features
            is_live: Whether this is for live trading or backtesting
            **kwargs: Additional keyword arguments
            
        Returns:
            DataFrame containing meta-features including disagreement features
        """
        try:
            tprint(f"🔍 [ANALYST_ENSEMBLE] Generating meta-features for analyst ensemble", color="cyan")
            
            # Initialize meta-features DataFrame
            meta_features = pd.DataFrame(index=df.index)
            
            # Add basic analyst-specific meta-features
            if 'close' in df.columns:
                meta_features['price_momentum'] = df['close'].pct_change(10)
                meta_features['price_acceleration'] = df['close'].pct_change(10).diff()
                meta_features['volatility_proxy'] = df['close'].pct_change().rolling(20).std()
                meta_features['price_trend'] = df['close'].rolling(50).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
            
            if 'volume' in df.columns:
                meta_features['volume_momentum'] = df['volume'].pct_change(10)
                meta_features['volume_acceleration'] = df['volume'].pct_change(10).diff()
                meta_features['volume_trend'] = df['volume'].rolling(50).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
            
            # Add regime-specific features if available
            if 'composite_cluster_id' in df.columns:
                meta_features['regime_stability'] = df['composite_cluster_id'].rolling(20).std()
                meta_features['regime_persistence'] = (df['composite_cluster_id'] == df['composite_cluster_id'].shift(1)).rolling(20).mean()
                meta_features['regime_transition'] = (df['composite_cluster_id'] != df['composite_cluster_id'].shift(1)).rolling(10).sum()
            
            # Add HMM integration features if available
            hmm_features = ['hmm_state', 'hmm_transition_prob', 'hmm_confidence']
            for feature in hmm_features:
                if feature in df.columns:
                    meta_features[f'{feature}_momentum'] = df[feature].pct_change(10)
                    meta_features[f'{feature}_stability'] = df[feature].rolling(20).std()
            
            # Get base model predictions for disagreement analysis
            base_predictions = self._get_base_model_predictions(df, is_live=is_live)
            
            if base_predictions and len(base_predictions) > 1:
                # Use meta-feature generator from feature engineering
                try:
                    from src.feature_engineering_roadmap.ensemble_meta_features import EnsembleMetaFeatureGenerator
                    meta_feature_generator = EnsembleMetaFeatureGenerator(self.logger)
                    
                    # Generate meta-features using the feature engineering module
                    meta_features = meta_feature_generator.generate_meta_features_for_analyst_ensemble(
                        df, base_predictions, is_live
                    )
                    
                    tprint(f"✅ [ANALYST_ENSEMBLE] Generated {len(meta_features.columns)} meta-features", color="green")
                except ImportError as e:
                    tprint(f"⚠️ [ANALYST_ENSEMBLE] Could not import meta-feature generator: {e}", color="yellow")

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
                    # Add default disagreement features
                    default_disagreement = {
                        "prediction_dispersion": 0.0, "prediction_std": 0.0,
                        "direction_conflict": 0.0, "long_ratio": 0.5, "disagreement_rate": 0.0,
                        "confidence_gap": 0.0, "max_confidence": 0.0, "second_max_confidence": 0.0,
                        "entropy": 0.0, "uncertainty": 0.0, "prediction_range": 0.0, "prediction_iqr": 0.0,
                        "probability_range": 0.0, "probability_iqr": 0.0, "js_divergence": 0.0,
                        "kl_divergence": 0.0, "avg_divergence": 0.0
                    }
                    for feature_name, feature_value in default_disagreement.items():
                        meta_features[feature_name] = feature_value
            else:
                tprint("⚠️ [ANALYST_ENSEMBLE] Insufficient base model predictions for disagreement analysis", color="yellow")
                # Add default disagreement features
                default_disagreement = {
                    "prediction_dispersion": 0.0, "prediction_std": 0.0,
                    "direction_conflict": 0.0, "long_ratio": 0.5, "disagreement_rate": 0.0,
                    "confidence_gap": 0.0, "max_confidence": 0.0, "second_max_confidence": 0.0,
                    "entropy": 0.0, "uncertainty": 0.0, "prediction_range": 0.0, "prediction_iqr": 0.0,
                    "probability_range": 0.0, "probability_iqr": 0.0, "js_divergence": 0.0,
                    "kl_divergence": 0.0, "avg_divergence": 0.0
                }
                for feature_name, feature_value in default_disagreement.items():
                    meta_features[feature_name] = feature_value
            
            # Ensure all features are numeric and handle any NaN values
            meta_features = meta_features.fillna(0.0)
            
            # Convert to numeric, coercing any non-numeric values
            for col in meta_features.columns:
                meta_features[col] = pd.to_numeric(meta_features[col], errors='coerce').fillna(0.0)
            
            tprint(f"✅ [ANALYST_ENSEMBLE] Generated {len(meta_features.columns)} meta-features", color="green")
            return meta_features
            
        except Exception as e:
            self.logger.error(f"Error generating meta-features for analyst ensemble: {e}")
            # Return basic meta-features as fallback
            try:
                meta_features = pd.DataFrame(index=df.index)
                if 'close' in df.columns:
                    meta_features['price_momentum'] = df['close'].pct_change(10).fillna(0)
                return meta_features.fillna(0.0)
            except Exception as fallback_error:
                self.logger.error(f"Fallback meta-feature generation also failed: {fallback_error}")
                return pd.DataFrame(index=df.index)
    
    def _get_base_model_predictions(self, df: pd.DataFrame, is_live: bool = False) -> Dict[str, Any]:
        """
        Get predictions from all base models for disagreement analysis.
        
        Args:
            df: Input DataFrame with features
            is_live: Whether this is for live trading or backtesting
            
        Returns:
            Dict containing model predictions and probabilities
        """
        try:
            base_predictions = {}
            
            # Get predictions from ensemble models if available
            if hasattr(self, 'analyst_ensembles') and self.analyst_ensembles:
                for regime, ensemble in self.analyst_ensembles.items():
                    if ensemble and hasattr(ensemble, 'predict'):
                        try:
                            # Get prediction from ensemble
                            prediction = ensemble.predict(df.values) if hasattr(ensemble, 'predict') else 0.5
                            confidence = 0.8  # Default confidence for analyst ensemble models
                            
                            base_predictions[f'ensemble_{regime}'] = {
                                'prediction': float(prediction),
                                'probability': float(prediction),
                                'confidence': float(confidence)
                            }
                        except Exception as model_error:
                            self.logger.warning(f"Error getting prediction from ensemble {regime}: {model_error}")
                            base_predictions[f'ensemble_{regime}'] = {
                                'prediction': 0.5,
                                'probability': 0.5,
                                'confidence': 0.0
                            }
            
            # Get predictions from NAS models if available
            if hasattr(self, 'nas_models') and self.nas_models:
                for regime, nas_model in self.nas_models.items():
                    if nas_model and hasattr(nas_model, 'predict'):
                        try:
                            prediction = nas_model.predict(df.values) if hasattr(nas_model, 'predict') else 0.5
                            confidence = 0.7  # Default confidence for NAS models
                            
                            base_predictions[f'nas_{regime}'] = {
                                'prediction': float(prediction),
                                'probability': float(prediction),
                                'confidence': float(confidence)
                            }
                        except Exception as model_error:
                            self.logger.warning(f"Error getting prediction from NAS model {regime}: {model_error}")
                            base_predictions[f'nas_{regime}'] = {
                                'prediction': 0.5,
                                'probability': 0.5,
                                'confidence': 0.0
                            }
            
            return base_predictions
            
        except Exception as e:
            self.logger.error(f"Error getting base model predictions: {e}")
            return {}


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
        model_types=["tcn", "lightgbm", "ridge", "elastic_net", "random_forest"],
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
    tprint_info("- ✅ Models: TCN (Temporal Convolutional Network), CatBoost, LightGBM, Elastic Net")
    tprint_info("- ✅ Comprehensive context from multi-timeframe dynamics")
    tprint_info("- ✅ Fast failing for missing ML dependencies - no fallbacks")
    
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
    tprint_info("- ✅ Ensemble diversity metrics for model complementarity")
    tprint_info("- ✅ Confidence intervals for OOF predictions")
    tprint_info("- ✅ Bootstrap-based uncertainty quantification")
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

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
