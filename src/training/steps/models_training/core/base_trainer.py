"""
Base Trainer - Unified Training Architecture

This module provides the abstract base trainer class that consolidates
common training functionality across all training components.

Key Features:
- Unified training interface for all model types
- Common training patterns and lifecycle management
- Standardized configuration and validation
- Performance monitoring and checkpointing
- Error handling and recovery mechanisms
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_debug, tprint_performance, tprint_data_format, tprint_data_preview, LogLevel
from src.utils.common_operations import (
    safe_divide, safe_correlation, safe_mean, safe_std, safe_float, safe_int
)
from src.utils.common_utilities import calculate_data_quality_metrics, get_dataframe_info
from src.utils.math_validation import validate_finite, validate_positive, validate_range
from src.utils.hardware.integrated_hardware_manager import (
    get_integrated_hardware_manager, IntegratedHardwareConfig,
    process_market_data, process_ml_training_data, process_backtesting_data
)
from src.utils.hardware.unified_hardware_manager import (
    get_unified_hardware_manager, WorkloadType, OptimizationLevel
)
from src.utils.hardware.optimization_decorators import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked
)
from src.utils.hardware.memory_optimized_decorators import (
    memory_optimized, gc_optimized, comprehensive_memory_optimization,
    MemoryOptimizationLevel
)
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
from src.utils.kline_parquet import KlinesParquetManager
from src.core.decorators import handles_errors, traced, log_execution_time


class TrainingRole(Enum):
    """Training roles in the system."""
    ANALYST = "analyst"
    TACTICIAN = "tactician"
    ENSEMBLE = "ensemble"


class ModelType(Enum):
    """Types of ML models."""
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    LINEAR = "linear"


@dataclass
class TrainingConfig:
    """Unified training configuration."""
    # Core configuration
    role: TrainingRole
    model_types: List[ModelType]
    timeframe: str = "15m"
    symbol: str = "ETHUSDT"
    
    # Training parameters
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    random_seed: Optional[int] = None
    
    # Model-specific parameters
    enable_hyperparameter_optimization: bool = True
    enable_ensemble: bool = True
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Performance parameters
    max_training_time: Optional[float] = None  # seconds
    memory_limit_mb: Optional[int] = None
    
    # Feature configuration
    feature_selection_method: str = "multi_objective"
    max_features: int = 100
    correlation_threshold: float = 0.85
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Result of training operation."""
    success: bool
    model: Optional[Any] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validation operation."""
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result of prediction operation."""
    success: bool
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    confidence_scores: Optional[np.ndarray] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTrainer(ABC):
    """
    Abstract base trainer for all training components.
    
    This class provides a unified interface for training different types of models
    across different roles (Analyst, Tactician, Ensemble) while maintaining
    consistent patterns for configuration, validation, and error handling.
    """
    
    def __init__(self, config: TrainingConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the base trainer.
        
        Args:
            config: Training configuration
            logger: Logger instance (optional)
        """
        self.config = config
        self.logger = logger or system_logger.getChild(f"{self.__class__.__name__}")
        
        # Training state
        self._training_state = {
            'initialized': False,
            'training_started': False,
            'training_completed': False,
            'model_created': False,
            'best_model_saved': False
        }
        
        # Performance tracking
        self._performance_metrics = {
            'training_time': 0.0,
            'validation_time': 0.0,
            'prediction_time': 0.0,
            'memory_usage_mb': 0.0,
            'cpu_usage_percent': 0.0
        }
        
        # Model state
        self._model_state = {
            'model': None,
            'best_model': None,
            'training_history': [],
            'validation_history': [],
            'checkpoints': []
        }
        
        # Initialize enhanced hardware managers
        self._integrated_hardware_manager = None
        self._unified_hardware_manager = None
        self._parquet_manager = None
        
        tprint_info(f"🔧 Initializing {self.__class__.__name__} for {config.role.value}")
        self.logger.info(f"Initialized {self.__class__.__name__} for {config.role.value}")
    
    async def _create_model(self, model_type: ModelType) -> Optional[Any]:
        """
        Create a model instance for the given model type.
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Model instance or None if creation fails
        """
        try:
            tprint_debug(f"🔧 Creating {model_type.value} model...")
            
            if model_type == ModelType.ANALYST:
                return self._create_analyst_model()
            elif model_type == ModelType.TACTICIAN:
                return self._create_tactician_model()
            elif model_type == ModelType.ENSEMBLE:
                return self._create_ensemble_model()
            else:
                tprint_warning(f"⚠️ Unknown model type: {model_type}")
                return None
                
        except Exception as e:
            tprint_error(f"❌ Failed to create {model_type.value} model: {e}")
            return None
    
    async def _preprocess_data(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess training data and targets.
        
        Args:
            data: Training data
            targets: Target variables
            
        Returns:
            Tuple of (processed_data, processed_targets)
        """
        try:
            tprint_debug("🔄 Preprocessing training data...")
            
            # Handle missing values
            if data.isnull().any().any():
                tprint_debug("🔧 Handling missing values...")
                data = data.fillna(data.mean())
            
            # Handle infinite values
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(data.mean())
            
            # Ensure targets are available
            if targets is None:
                # Try to infer targets from data
                target_columns = [col for col in data.columns if 'target' in col.lower() or 'label' in col.lower()]
                if target_columns:
                    targets = data[target_columns[0]]
                    data = data.drop(columns=target_columns)
                    tprint_debug(f"🔧 Inferred targets from column: {target_columns[0]}")
                else:
                    # Create dummy targets for unsupervised learning
                    targets = pd.Series([0] * len(data), index=data.index)
                    tprint_warning("⚠️ No targets found, using dummy targets")
            
            # Ensure targets are numeric
            if not pd.api.types.is_numeric_dtype(targets):
                tprint_debug("🔧 Converting targets to numeric...")
                targets = pd.to_numeric(targets, errors='coerce')
                targets = targets.fillna(0)
            
            # Remove any remaining non-numeric columns from data
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data = data[numeric_columns]
            
            tprint_success(f"✅ Preprocessed data: {data.shape[0]} samples, {data.shape[1]} features")
            return data, targets
            
        except Exception as e:
            tprint_error(f"❌ Data preprocessing failed: {e}")
            raise
    
    def _get_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from trained model.
        
        Args:
            model: Trained model
            
        Returns:
            Dictionary of feature importance scores or None
        """
        try:
            # Check if model has feature_importances_ attribute
            if hasattr(model, 'feature_importances_'):
                return dict(zip(
                    getattr(self, '_feature_names', [f'feature_{i}' for i in range(len(model.feature_importances_))]),
                    model.feature_importances_
                ))
            
            # Check if model has coef_ attribute (linear models)
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if coef.ndim > 1:
                    coef = coef[0]  # Take first class for multi-class
                return dict(zip(
                    getattr(self, '_feature_names', [f'feature_{i}' for i in range(len(coef))]),
                    coef
                ))
            
            # Check if model has feature_importances_ method
            elif hasattr(model, 'get_feature_importance'):
                return model.get_feature_importance()
            
            else:
                tprint_debug("🔧 Model does not support feature importance extraction")
                return None
                
        except Exception as e:
            tprint_warning(f"⚠️ Could not extract feature importance: {e}")
            return None

    async def train(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> TrainingResult:
        """
        Train the model with given data.
        
        Args:
            data: Training data
            targets: Target variables (optional, can be inferred from data)
            
        Returns:
            Training result with model and metrics
        """
        try:
            tprint_info("🚀 Starting model training...")
            self.logger.info("Starting model training...")
            
            start_time = time.time()
            
            # Preprocess data
            processed_data, processed_targets = self._preprocess_data(data, targets)
            
            # Initialize training state
            self._training_state['training_started'] = True
            self._training_state['training_completed'] = False
            
            # Train each model type
            all_models = {}
            all_metrics = {}
            best_model = None
            best_score = -float('inf')
            
            for model_type in self.config.model_types:
                tprint_info(f"🔧 Training {model_type.value} model...")
                
                # Create model
                model = self._create_model(model_type)
                if model is None:
                    tprint_warning(f"⚠️ Failed to create {model_type.value} model, skipping...")
                    continue
                
                # Train model
                model_result = await self._train_single_model(
                    model, processed_data, processed_targets, model_type
                )
                
                if model_result['success']:
                    all_models[model_type.value] = model_result['model']
                    all_metrics[model_type.value] = model_result['metrics']
                    
                    # Track best model
                    primary_metric = self.config.custom_params.get('primary_metric', 'accuracy')
                    if primary_metric in model_result['metrics']:
                        score = model_result['metrics'][primary_metric]
                        if score > best_score:
                            best_score = score
                            best_model = model_result['model']
                    
                    tprint_success(f"✅ {model_type.value} model trained successfully")
                else:
                    tprint_error(f"❌ {model_type.value} model training failed: {model_result.get('error', 'Unknown error')}")
            
            # Calculate training time
            training_time = time.time() - start_time
            self._performance_metrics['training_time'] = training_time
            
            # Update state
            self._training_state['training_completed'] = True
            self._model_state['model'] = best_model
            self._model_state['best_model'] = best_model
            
            # Get feature importance if available
            feature_importance = None
            if best_model is not None:
                feature_importance = self._get_feature_importance(best_model)
            
            # Create result
            result = TrainingResult(
                success=len(all_models) > 0,
                model=best_model,
                metrics=all_metrics.get(list(all_models.keys())[0], {}) if all_models else {},
                training_time=training_time,
                validation_metrics=all_metrics,
                feature_importance=feature_importance,
                metadata={
                    'models_trained': list(all_models.keys()),
                    'best_model_type': list(all_models.keys())[0] if all_models else None,
                    'best_score': best_score
                }
            )
            
            if result.success:
                tprint_success(f"✅ Training completed successfully in {training_time:.2f}s")
                self.logger.info(f"Training completed successfully in {training_time:.2f}s")
            else:
                tprint_error("❌ Training failed - no models were successfully trained")
                result.error_message = "No models were successfully trained"
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Training failed: {e}")
            self.logger.error(f"Training failed: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e),
                training_time=time.time() - start_time if 'start_time' in locals() else 0.0
            )
    
    async def validate(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> ValidationResult:
        """
        Validate the trained model.
        
        Args:
            data: Validation data
            targets: Target variables (optional)
            
        Returns:
            Validation result with metrics
        """
        try:
            tprint_info("🔍 Starting model validation...")
            self.logger.info("Starting model validation...")
            
            start_time = time.time()
            
            # Check if model is trained
            if not self._training_state['training_completed']:
                return ValidationResult(
                    success=False,
                    error_message="Model not trained yet"
                )
            
            # Preprocess validation data
            processed_data, processed_targets = self._preprocess_data(data, targets)
            
            # Get the best model
            model = self._model_state['best_model']
            if model is None:
                return ValidationResult(
                    success=False,
                    error_message="No trained model available"
                )
            
            # Make predictions
            predictions = await self._predict_with_model(model, processed_data)
            
            # Calculate validation metrics
            metrics = self._calculate_validation_metrics(
                processed_targets, predictions, processed_data
            )
            
            # Calculate validation time
            validation_time = time.time() - start_time
            self._performance_metrics['validation_time'] = validation_time
            
            tprint_success(f"✅ Validation completed in {validation_time:.2f}s")
            self.logger.info(f"Validation completed in {validation_time:.2f}s")
            
            return ValidationResult(
                success=True,
                metrics=metrics,
                predictions=predictions,
                metadata={
                    'validation_time': validation_time,
                    'data_shape': processed_data.shape,
                    'predictions_shape': predictions.shape if predictions is not None else None
                }
            )
            
        except Exception as e:
            tprint_error(f"❌ Validation failed: {e}")
            self.logger.error(f"Validation failed: {e}")
            return ValidationResult(
                success=False,
                error_message=str(e)
            )
    
    async def predict(self, data: pd.DataFrame) -> PredictionResult:
        """
        Make predictions with the trained model.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Prediction result
        """
        try:
            tprint_info("🔮 Making predictions...")
            self.logger.info("Making predictions...")
            
            start_time = time.time()
            
            # Check if model is trained
            if not self._training_state['training_completed']:
                return PredictionResult(
                    success=False,
                    error_message="Model not trained yet"
                )
            
            # Preprocess data (without targets)
            processed_data, _ = self._preprocess_data(data, None)
            
            # Get the best model
            model = self._model_state['best_model']
            if model is None:
                return PredictionResult(
                    success=False,
                    error_message="No trained model available"
                )
            
            # Make predictions
            predictions = await self._predict_with_model(model, processed_data)
            
            # Calculate confidence scores if possible
            confidence_scores = self._calculate_confidence_scores(model, processed_data, predictions)
            
            # Calculate probabilities if possible
            probabilities = self._calculate_probabilities(model, processed_data)
            
            # Calculate prediction time
            prediction_time = time.time() - start_time
            self._performance_metrics['prediction_time'] = prediction_time
            
            tprint_success(f"✅ Predictions completed in {prediction_time:.2f}s")
            self.logger.info(f"Predictions completed in {prediction_time:.2f}s")
            
            return PredictionResult(
                success=True,
                predictions=predictions,
                probabilities=probabilities,
                confidence_scores=confidence_scores,
                metadata={
                    'prediction_time': prediction_time,
                    'data_shape': processed_data.shape,
                    'predictions_shape': predictions.shape if predictions is not None else None
                }
            )
            
        except Exception as e:
            tprint_error(f"❌ Prediction failed: {e}")
            self.logger.error(f"Prediction failed: {e}")
            return PredictionResult(
                success=False,
                error_message=str(e)
            )
    
    def _create_model(self, model_type: ModelType) -> Any:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Model instance
        """
        try:
            tprint_debug(f"🔧 Creating {model_type.value} model...")
            
            if model_type == ModelType.LIGHTGBM:
                return self._create_lightgbm_model()
            elif model_type == ModelType.CATBOOST:
                return self._create_catboost_model()
            elif model_type == ModelType.NEURAL_NETWORK:
                return self._create_neural_network_model()
            elif model_type == ModelType.ENSEMBLE:
                return self._create_ensemble_model()
            elif model_type == ModelType.LINEAR:
                return self._create_linear_model()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            tprint_error(f"❌ Failed to create {model_type.value} model: {e}")
            self.logger.error(f"Failed to create {model_type.value} model: {e}")
            return None
    
    def _get_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from model.
        
        Args:
            model: Trained model
            
        Returns:
            Feature importance dictionary
        """
        try:
            if model is None:
                return None
            
            # Try different methods based on model type
            if hasattr(model, 'feature_importances_'):
                # Tree-based models (LightGBM, CatBoost, etc.)
                importances = model.feature_importances_
                feature_names = getattr(model, 'feature_name_', None)
                
                if feature_names is not None and len(feature_names) == len(importances):
                    return dict(zip(feature_names, importances))
                else:
                    # Use generic feature names
                    return {f"feature_{i}": imp for i, imp in enumerate(importances)}
            
            elif hasattr(model, 'coef_'):
                # Linear models
                coef = model.coef_
                if coef.ndim == 1:
                    # Binary classification or regression
                    return {f"feature_{i}": abs(coef[i]) for i in range(len(coef))}
                else:
                    # Multi-class classification
                    # Use L2 norm of coefficients
                    importance = np.linalg.norm(coef, axis=0)
                    return {f"feature_{i}": imp for i, imp in enumerate(importance)}
            
            elif hasattr(model, 'named_steps'):
                # Pipeline models
                # Try to get feature importance from the last step
                last_step = model.named_steps[list(model.named_steps.keys())[-1]]
                return self._get_feature_importance(last_step)
            
            else:
                tprint_warning(f"⚠️ Cannot extract feature importance from {type(model).__name__}")
                return None
                
        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract feature importance: {e}")
            self.logger.warning(f"Failed to extract feature importance: {e}")
            return None
    
    @handles_errors(
        exceptions=(ValueError, AttributeError, RuntimeError),
        default_return=False,
        context="trainer initialization"
    )
    async def initialize(self) -> bool:
        """Initialize the trainer."""
        try:
            tprint_info("🔧 Initializing trainer...")
            self.logger.info("Initializing trainer...")
            
            # Validate configuration
            if not self._validate_config():
                tprint_error("❌ Configuration validation failed")
                return False
            
            # Initialize hardware optimizers
            await self._initialize_hardware_optimizers()
            
            # Initialize model if auto-initialization is enabled
            if self.config.custom_params.get('auto_initialize_model', False):
                await self._initialize_models()
            
            self._training_state['initialized'] = True
            tprint_success("✅ Trainer initialized successfully")
            self.logger.info("✅ Trainer initialized successfully")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Trainer initialization failed: {e}")
            self.logger.error(f"❌ Trainer initialization failed: {e}")
            return False
    
    async def _initialize_hardware_optimizers(self):
        """Initialize enhanced hardware managers."""
        try:
            tprint_debug("🔧 Initializing enhanced hardware managers...")
            
            # Initialize integrated hardware manager with ML training configuration
            integrated_config = IntegratedHardwareConfig(
                enable_automatic_optimization=True,
                enable_caching=True,
                enable_memory_monitoring=True,
                enable_performance_tracking=True,
                memory_limit_gb=8.0,
                cache_memory_limit_mb=1024.0
            )
            self._integrated_hardware_manager = get_integrated_hardware_manager(integrated_config)
            
            # Initialize unified hardware manager for workload optimization
            self._unified_hardware_manager = get_unified_hardware_manager()
            
            # Initialize parquet manager
            self._parquet_manager = KlinesParquetManager()
            
            tprint_success("✅ Enhanced hardware managers initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ Hardware manager initialization failed: {e}")
            self.logger.warning(f"Hardware manager initialization failed: {e}")
    
    def _validate_config(self) -> bool:
        """Validate training configuration."""
        try:
            tprint_debug("🔍 Validating configuration...")
            
            # Validate required fields
            if not self.config.role:
                tprint_error("❌ Training role is required")
                return False
            
            if not self.config.model_types:
                tprint_error("❌ At least one model type is required")
                return False
            
            # Validate training parameters
            if not validate_range(self.config.validation_split, 0.0, 1.0):
                tprint_error("❌ Validation split must be between 0 and 1")
                return False
            
            if self.config.cross_validation_folds < 2:
                tprint_error("❌ Cross-validation folds must be at least 2")
                return False
            
            # Validate feature parameters
            if not validate_positive(self.config.max_features):
                tprint_error("❌ Max features must be positive")
                return False
            
            if not validate_range(self.config.correlation_threshold, 0.0, 1.0):
                tprint_error("❌ Correlation threshold must be between 0 and 1")
                return False
            
            tprint_success("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Configuration validation failed: {e}")
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    async def _initialize_models(self) -> bool:
        """Initialize models for training."""
        try:
            self.logger.info("Initializing models...")
            
            for model_type in self.config.model_types:
                model = self._create_model(model_type)
                if model is None:
                    self.logger.error(f"Failed to create {model_type.value} model")
                    return False
                
                # Store model in state
                model_key = f"{model_type.value}_model"
                self._model_state[model_key] = model
            
            self._training_state['model_created'] = True
            self.logger.info("✅ Models initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            return False
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=None,
        context="data preprocessing"
    )
    @comprehensive_memory_optimization(
        optimization_level=MemoryOptimizationLevel.AGGRESSIVE,
        enable_caching=True,
        enable_chunking=True,
        enable_gc=True,
        enable_pools=True
    )
    def _preprocess_data(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data for training using our utilities.
        
        Args:
            data: Input data
            targets: Target variables
            
        Returns:
            Preprocessed data and targets
        """
        try:
            tprint_info("🔧 Preprocessing data...")
            self.logger.info("Preprocessing data...")
            
            # Memory usage is now tracked by enhanced hardware managers
            
            # Preview raw input data and validate format
            from src.utils.tprint import tprint_data_preview
            tprint_data_preview(data, "Raw input data", max_rows=5, level="INFO")
            tprint_data_format(data, "Raw input data", level=LogLevel.DEBUG)
            
            # Calculate data quality metrics
            quality_metrics = calculate_data_quality_metrics(data)
            tprint_debug(f"📊 Data quality metrics: {quality_metrics}")
            
            # Preview data after quality checks and validate format
            tprint_data_preview(data, "Data after quality checks", max_rows=5, level="DEBUG")
            tprint_data_format(data, "Data after quality checks", level=LogLevel.DEBUG)
            
            # Handle missing values using safe operations
            if data.isnull().any().any():
                tprint_warning("⚠️ Found missing values, filling with median")
                data = data.fillna(data.median())
            
            # Handle infinite values using safe operations
            if np.isinf(data).any().any():
                tprint_warning("⚠️ Found infinite values, replacing with finite values")
                data = data.replace([np.inf, -np.inf], np.nan)
                data = data.fillna(data.median())
            
            # Optimize memory usage using integrated hardware manager
            if self._integrated_hardware_manager:
                data = self._integrated_hardware_manager.process_data_with_optimization(
                    data, WorkloadType.ML_TRAINING
                )
                tprint_debug("🧠 Enhanced memory optimization applied")
                # Preview data after memory optimization
                tprint_data_preview(data, "Data after memory optimization", max_rows=5, level="DEBUG")
            
            # Feature selection if enabled
            if self.config.max_features < len(data.columns):
                selected_features = self._select_features(data, targets)
                data = data[selected_features]
                tprint_info(f"📊 Selected {len(selected_features)} features")
                # Preview data after feature selection
                tprint_data_preview(data, "Data after feature selection", max_rows=5, level="DEBUG")
            
            # Extract targets if not provided
            if targets is None:
                target_columns = ['target', 'y', 'label']
                for col in target_columns:
                    if col in data.columns:
                        targets = data[col]
                        data = data.drop(columns=[col])
                        # Preview extracted targets
                        tprint_data_preview(targets, f"Extracted targets from {col}", max_rows=10, level="INFO")
                        break
                
                if targets is None:
                    raise ValueError("No target column found in data")
            
            # Validate finite values in targets
            if targets is not None:
                validate_finite(targets, "targets")
            
            # Preview final preprocessed data
            tprint_data_preview(data, "Final preprocessed data", max_rows=5, level="INFO")
            if targets is not None:
                tprint_data_preview(targets, "Final preprocessed targets", max_rows=10, level="INFO")
            
            tprint_success(f"✅ Data preprocessed: {data.shape[0]} samples, {data.shape[1]} features")
            
            return data, targets
            
        except Exception as e:
            tprint_error(f"❌ Data preprocessing failed: {e}")
            self.logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def _select_features(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> List[str]:
        """
        Select features based on configuration.
        
        Args:
            data: Input data
            targets: Target variables
            
        Returns:
            List of selected feature names
        """
        try:
            if self.config.feature_selection_method == "correlation":
                return self._select_features_by_correlation(data)
            elif self.config.feature_selection_method == "variance":
                return self._select_features_by_variance(data)
            elif self.config.feature_selection_method == "mutual_info":
                return self._select_features_by_mutual_info(data, targets)
            else:
                # Default: select top features by correlation with target
                if targets is not None:
                    correlations = data.corrwith(targets).abs().sort_values(ascending=False)
                    return correlations.head(self.config.max_features).index.tolist()
                else:
                    return data.columns[:self.config.max_features].tolist()
                    
        except Exception as e:
            self.logger.warning(f"Feature selection failed, using all features: {e}")
            return data.columns.tolist()
    
    def _select_features_by_correlation(self, data: pd.DataFrame) -> List[str]:
        """Select features by correlation threshold."""
        corr_matrix = data.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features to drop
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.config.correlation_threshold)]
        
        # Select remaining features
        selected = [col for col in data.columns if col not in to_drop]
        return selected[:self.config.max_features]
    
    def _select_features_by_variance(self, data: pd.DataFrame) -> List[str]:
        """Select features by variance."""
        variances = data.var().sort_values(ascending=False)
        return variances.head(self.config.max_features).index.tolist()
    
    def _select_features_by_mutual_info(self, data: pd.DataFrame, targets: pd.Series) -> List[str]:
        """Select features by mutual information."""
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            if targets is None:
                return data.columns[:self.config.max_features].tolist()
            
            # Calculate mutual information
            mi_scores = mutual_info_regression(data, targets)
            feature_scores = pd.Series(mi_scores, index=data.columns).sort_values(ascending=False)
            
            return feature_scores.head(self.config.max_features).index.tolist()
            
        except ImportError:
            self.logger.warning("sklearn not available, using correlation-based selection")
            return self._select_features_by_correlation(data)
    
    def _update_performance_metrics(self, operation: str, duration: float, memory_usage: float = 0.0):
        """Update performance metrics."""
        self._performance_metrics[f'{operation}_time'] = duration
        if memory_usage > 0:
            self._performance_metrics['memory_usage_mb'] = memory_usage
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        return {
            'trainer_type': self.__class__.__name__,
            'config': self.config.__dict__,
            'training_state': self._training_state.copy(),
            'performance_metrics': self._performance_metrics.copy(),
            'model_state': {
                'model_created': self._training_state['model_created'],
                'training_completed': self._training_state['training_completed'],
                'best_model_saved': self._training_state['best_model_saved']
            }
        }
    
    def get_required_dependencies(self) -> List[str]:
        """Get list of required dependencies."""
        return ['pandas', 'numpy', 'scikit-learn']
    
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get component processing capabilities."""
        return {
            'supports_parallel_processing': False,
            'supports_checkpointing': True,
            'supports_early_stopping': True,
            'supports_ensemble': self.config.enable_ensemble,
            'memory_efficient': True,
            'gpu_acceleration': False
        }
    
    def estimate_processing_time(self, data_size: int) -> float:
        """Estimate processing time for given data size."""
        base_time = 5.0  # seconds
        size_factor = data_size / 1000
        model_factor = len(self.config.model_types)
        
        return base_time * size_factor * model_factor
    
    def get_memory_requirements(self, data_size: int) -> Dict[str, float]:
        """Get memory requirements for processing."""
        base_memory = 200  # MB
        data_memory = data_size * 0.001  # Rough estimate
        model_memory = len(self.config.model_types) * 100  # MB per model
        
        return {
            'estimated_memory_mb': base_memory + data_memory + model_memory,
            'peak_memory_mb': (base_memory + data_memory + model_memory) * 1.5
        }
    
    # Helper methods for the implemented abstract methods
    
    async def _train_single_model(self, model: Any, data: pd.DataFrame, targets: pd.Series, model_type: ModelType) -> Dict[str, Any]:
        """Train a single model and return results."""
        try:
            start_time = time.time()
            
            # Split data for validation
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                data, targets, 
                test_size=self.config.validation_split,
                random_state=self.config.random_seed
            )
            
            # Train model
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
            else:
                raise ValueError(f"Model {type(model).__name__} does not have fit method")
            
            # Make predictions on validation set
            if hasattr(model, 'predict'):
                val_predictions = model.predict(X_val)
            else:
                raise ValueError(f"Model {type(model).__name__} does not have predict method")
            
            # Calculate metrics
            metrics = self._calculate_validation_metrics(y_val, val_predictions, X_val)
            
            training_time = time.time() - start_time
            
            return {
                'success': True,
                'model': model,
                'metrics': metrics,
                'training_time': training_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model': None,
                'metrics': {},
                'training_time': 0.0
            }
    
    async def _predict_with_model(self, model: Any, data: pd.DataFrame) -> np.ndarray:
        """Make predictions with a trained model."""
        try:
            if hasattr(model, 'predict'):
                return model.predict(data)
            else:
                raise ValueError(f"Model {type(model).__name__} does not have predict method")
        except Exception as e:
            tprint_error(f"❌ Prediction failed: {e}")
            raise
    
    def _calculate_validation_metrics(self, y_true: pd.Series, y_pred: np.ndarray, X: pd.DataFrame) -> Dict[str, float]:
        """Calculate validation metrics."""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
            
            metrics = {}
            
            # Determine if classification or regression
            unique_values = len(np.unique(y_true))
            is_classification = unique_values <= 20  # Heuristic for classification
            
            if is_classification:
                # Classification metrics
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                try:
                    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                except:
                    # Handle edge cases
                    metrics['precision'] = 0.0
                    metrics['recall'] = 0.0
                    metrics['f1'] = 0.0
            else:
                # Regression metrics
                metrics['mse'] = mean_squared_error(y_true, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['r2'] = r2_score(y_true, y_pred)
                
                # Additional regression metrics
                mae = np.mean(np.abs(y_true - y_pred))
                metrics['mae'] = mae
                
                # Mean absolute percentage error
                mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
                metrics['mape'] = mape
            
            return metrics
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to calculate metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_confidence_scores(self, model: Any, data: pd.DataFrame, predictions: np.ndarray) -> Optional[np.ndarray]:
        """Calculate confidence scores for predictions."""
        try:
            # Try to get prediction probabilities
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(data)
                if proba.ndim == 2:
                    # Multi-class: use max probability as confidence
                    return np.max(proba, axis=1)
                else:
                    return proba
            elif hasattr(model, 'decision_function'):
                # SVM or similar: use decision function values
                scores = model.decision_function(data)
                if scores.ndim == 1:
                    return np.abs(scores)
                else:
                    return np.max(scores, axis=1)
            else:
                # Fallback: use prediction variance or simple confidence
                return np.ones(len(predictions)) * 0.5
                
        except Exception as e:
            tprint_debug(f"Could not calculate confidence scores: {e}")
            return None
    
    def _calculate_probabilities(self, model: Any, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Calculate prediction probabilities."""
        try:
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(data)
            else:
                return None
        except Exception as e:
            tprint_debug(f"Could not calculate probabilities: {e}")
            return None
    
    def _create_lightgbm_model(self):
        """Create LightGBM model."""
        try:
            import lightgbm as lgb
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': self.config.random_seed
            }
            
            # Update with custom parameters
            params.update(self.config.custom_params.get('lightgbm_params', {}))
            
            return lgb.LGBMRegressor(**params)
            
        except ImportError:
            tprint_warning("⚠️ LightGBM not available, using fallback")
            return None
        except Exception as e:
            tprint_error(f"❌ Failed to create LightGBM model: {e}")
            return None
    
    def _create_catboost_model(self):
        """Create CatBoost model."""
        try:
            import catboost as cb
            
            params = {
                'iterations': 100,
                'learning_rate': 0.1,
                'depth': 6,
                'verbose': False,
                'random_seed': self.config.random_seed
            }
            
            # Update with custom parameters
            params.update(self.config.custom_params.get('catboost_params', {}))
            
            return cb.CatBoostRegressor(**params)
            
        except ImportError:
            tprint_warning("⚠️ CatBoost not available, using fallback")
            return None
        except Exception as e:
            tprint_error(f"❌ Failed to create CatBoost model: {e}")
            return None
    
    def _create_neural_network_model(self):
        """Create neural network model."""
        try:
            from sklearn.neural_network import MLPRegressor
            
            params = {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate': 'constant',
                'learning_rate_init': 0.001,
                'max_iter': 200,
                'random_state': self.config.random_seed
            }
            
            # Update with custom parameters
            params.update(self.config.custom_params.get('neural_network_params', {}))
            
            return MLPRegressor(**params)
            
        except Exception as e:
            tprint_error(f"❌ Failed to create neural network model: {e}")
            return None
    
    def _create_ensemble_model(self):
        """Create ensemble model."""
        try:
            from sklearn.ensemble import VotingRegressor
            from sklearn.linear_model import LinearRegression
            
            # Create base models
            base_models = []
            
            # Try to add different model types
            lgb_model = self._create_lightgbm_model()
            if lgb_model:
                base_models.append(('lightgbm', lgb_model))
            
            cb_model = self._create_catboost_model()
            if cb_model:
                base_models.append(('catboost', cb_model))
            
            # Always add linear regression as fallback
            base_models.append(('linear', LinearRegression()))
            
            if len(base_models) < 2:
                tprint_warning("⚠️ Not enough models for ensemble, using single model")
                return base_models[0][1] if base_models else None
            
            return VotingRegressor(base_models)
            
        except Exception as e:
            tprint_error(f"❌ Failed to create ensemble model: {e}")
            return None
    
    def _create_linear_model(self):
        """Create linear model."""
        try:
            from sklearn.linear_model import LinearRegression
            
            params = {
                'fit_intercept': True,
                'normalize': False,
                'copy_X': True,
                'n_jobs': -1
            }
            
            # Update with custom parameters
            params.update(self.config.custom_params.get('linear_params', {}))
            
            return LinearRegression(**params)
            
        except Exception as e:
            tprint_error(f"❌ Failed to create linear model: {e}")
            return None
    
    def _create_analyst_model(self) -> Optional[Any]:
        """Create an analyst model instance."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            
            model_type = self.config.custom_params.get('analyst_model_type', 'random_forest')
            
            if model_type == 'random_forest':
                return RandomForestClassifier(
                    n_estimators=self.config.custom_params.get('n_estimators', 100),
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == 'logistic_regression':
                return LogisticRegression(
                    random_state=42,
                    max_iter=1000
                )
            elif model_type == 'svm':
                return SVC(
                    random_state=42,
                    probability=True
                )
            else:
                tprint_warning(f"⚠️ Unknown analyst model type: {model_type}, using RandomForest")
                return RandomForestClassifier(random_state=42, n_jobs=-1)
                
        except Exception as e:
            tprint_error(f"❌ Failed to create analyst model: {e}")
            return None
    
    def _create_tactician_model(self) -> Optional[Any]:
        """Create a tactician model instance."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.neural_network import MLPClassifier
            from sklearn.linear_model import RidgeClassifier
            
            model_type = self.config.custom_params.get('tactician_model_type', 'gradient_boosting')
            
            if model_type == 'gradient_boosting':
                return GradientBoostingClassifier(
                    n_estimators=self.config.custom_params.get('n_estimators', 100),
                    random_state=42
                )
            elif model_type == 'neural_network':
                return MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    random_state=42,
                    max_iter=1000
                )
            elif model_type == 'ridge':
                return RidgeClassifier(random_state=42)
            else:
                tprint_warning(f"⚠️ Unknown tactician model type: {model_type}, using GradientBoosting")
                return GradientBoostingClassifier(random_state=42)
                
        except Exception as e:
            tprint_error(f"❌ Failed to create tactician model: {e}")
            return None
    
    def _create_ensemble_model(self) -> Optional[Any]:
        """Create an ensemble model instance."""
        try:
            from sklearn.ensemble import VotingClassifier, BaggingClassifier
            
            model_type = self.config.custom_params.get('ensemble_model_type', 'voting')
            
            if model_type == 'voting':
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
                
                estimators = [
                    ('rf', RandomForestClassifier(random_state=42, n_jobs=-1)),
                    ('lr', LogisticRegression(random_state=42, max_iter=1000))
                ]
                return VotingClassifier(estimators, voting='soft')
            
            elif model_type == 'bagging':
                from sklearn.tree import DecisionTreeClassifier
                return BaggingClassifier(
                    DecisionTreeClassifier(random_state=42),
                    n_estimators=10,
                    random_state=42
                )
            else:
                tprint_warning(f"⚠️ Unknown ensemble model type: {model_type}, using Voting")
                from sklearn.ensemble import RandomForestClassifier, VotingClassifier
                return VotingClassifier([
                    ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
                ])
                
        except Exception as e:
            tprint_error(f"❌ Failed to create ensemble model: {e}")
            return None
