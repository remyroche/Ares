"""
Base Trainer - Enhanced with Comprehensive BaseStep Utilities

This module provides the abstract base trainer class that consolidates
common training functionality across all training components with comprehensive
BaseStep utility integration.

Key Features:
- Unified training interface for all model types
- Common training patterns and lifecycle management
- Standardized configuration and validation
- Performance monitoring and checkpointing
- Error handling and recovery mechanisms
- Comprehensive BaseStep utility integration
- Advanced logging and data visualization
- Hardware optimization and memory management
- Data quality validation and cleaning
- Model persistence and caching
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

from src.training.steps.base_step import BaseStep
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


class BaseTrainer(BaseStep, ABC):
    """
    Abstract base trainer for all training components with comprehensive BaseStep utility integration.
    
    This class provides a unified interface for training different types of models
    across different roles (Analyst, Tactician, Ensemble) while maintaining
    consistent patterns for configuration, validation, error handling, and utility integration.
    """
    
    def __init__(self, config: TrainingConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the base trainer with comprehensive utilities.
        
        Args:
            config: Training configuration
            logger: Logger instance (optional)
        """
        # Initialize BaseStep with trainer-specific name
        super().__init__(f"{self.__class__.__name__.lower()}", config.__dict__ if hasattr(config, '__dict__') else config)
        
        self.config = config
        self.logger = logger or self.logger
        
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
    
    @abstractmethod
    async def train(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> TrainingResult:
        """
        Train the model with given data.
        
        Args:
            data: Training data
            targets: Target variables (optional, can be inferred from data)
            
        Returns:
            Training result with model and metrics
        """
        pass
    
    @abstractmethod
    async def validate(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> ValidationResult:
        """
        Validate the trained model.
        
        Args:
            data: Validation data
            targets: Target variables (optional)
            
        Returns:
            Validation result with metrics
        """
        pass
    
    @abstractmethod
    async def predict(self, data: pd.DataFrame) -> PredictionResult:
        """
        Make predictions with the trained model.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Prediction result
        """
        pass
    
    @abstractmethod
    def _create_model(self, model_type: ModelType) -> Any:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Model instance
        """
        pass
    
    @abstractmethod
    def _get_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from model.
        
        Args:
            model: Trained model
            
        Returns:
            Feature importance dictionary
        """
        pass
    
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
    
    def _extract_and_validate_training_data(self, data: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Extract and validate training data using BaseStep utilities."""
        try:
            # Extract data
            X_train = data.get('X_train')
            y_train = data.get('y_train')
            
            if X_train is None or y_train is None:
                self.tprint_error("❌ Missing required data: X_train and y_train")
                return None, None
            
            # Convert to pandas if needed using safe operations
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train)
            if not isinstance(y_train, pd.Series):
                y_train = pd.Series(y_train)
            
            # Validate data using BaseStep utilities
            if not self._validate_dataframe_columns(X_train, []):
                self.tprint_error("❌ Invalid training features")
                return None, None
            
            # Data preview using BaseStep utilities
            self.tprint_data_summary(X_train, "Training Features", max_rows=5)
            self.tprint_data_summary(y_train, "Training Targets", max_rows=5)
            
            return X_train, y_train
            
        except Exception as e:
            self.tprint_error(f"❌ Data extraction failed: {e}")
            return None, None
    
    def _analyze_data_quality(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Analyze data quality using BaseStep utilities."""
        try:
            if self.data_quality:
                # Use data quality utilities
                quality_metrics = self.data_quality['calculate_quality_metrics'](X_train, y_train)
                self.tprint_validation_result(quality_metrics, "Data Quality Analysis")
            else:
                # Fallback analysis
                self.tprint_info(f"📊 Training data shape: {X_train.shape}")
                self.tprint_info(f"📊 Target data shape: {y_train.shape}")
                self.tprint_info(f"📊 Missing values in features: {X_train.isnull().sum().sum()}")
                self.tprint_info(f"📊 Missing values in targets: {y_train.isnull().sum()}")
        except Exception as e:
            self.tprint_warning(f"⚠️ Data quality analysis failed: {e}")
    
    def _optimize_training_data(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Optimize training data using hardware utilities."""
        try:
            if self.hardware_utils and 'optimize_dataframe' in self.hardware_utils:
                X_train = self.hardware_utils['optimize_dataframe'](X_train)
                self.tprint_success("✅ Training data optimized for hardware")
            return X_train, y_train
        except Exception as e:
            self.tprint_warning(f"⚠️ Data optimization failed: {e}")
            return X_train, y_train
    
    def _analyze_training_performance(self, result: Dict[str, Any]) -> None:
        """Analyze training performance using BaseStep utilities."""
        try:
            # Performance summary
            self.tprint_performance_summary({
                'training_time': result.get('training_time', 0.0),
                'success': result.get('success', False),
                'model_created': self._training_state['model_created'],
                'training_completed': self._training_state['training_completed']
            })
            
            # Memory usage analysis
            if self.hardware_utils and 'get_memory_usage' in self.hardware_utils:
                memory_usage = self.hardware_utils['get_memory_usage']()
                self.tprint_memory_usage(memory_usage)
            
        except Exception as e:
            self.tprint_warning(f"⚠️ Performance analysis failed: {e}")
    
    async def _save_training_artifacts(self, result: Dict[str, Any]) -> None:
        """Save training artifacts using BaseStep utilities."""
        try:
            # Save model using BaseStep utilities
            if result.get('model'):
                self._save_model(result['model'], f'{self.__class__.__name__.lower()}_model')
            
            # Save metrics using BaseStep utilities
            if result.get('metrics'):
                self._save_metadata(result['metrics'], 'training_metrics')
            
            # Save feature importance
            if result.get('feature_importance'):
                self._save_metadata(result['feature_importance'], 'feature_importance')
            
            self.tprint_success("✅ Training artifacts saved successfully")
            
        except Exception as e:
            self.tprint_error(f"❌ Artifact saving failed: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary using BaseStep utilities."""
        try:
            summary = {
                'trainer_type': self.__class__.__name__,
                'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config,
                'training_state': self._training_state.copy(),
                'performance_metrics': self._performance_metrics.copy(),
                'utility_availability': self._get_availability_status(),
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
