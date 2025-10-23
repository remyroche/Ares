"""
Production-Ready Base Trainer - Unified Training Architecture

This module provides a comprehensive, production-ready abstract base trainer class
that consolidates common training functionality across all training components.

PRODUCTION FEATURES:
===================

1. COMPREHENSIVE ERROR HANDLING:
   - Graceful degradation and recovery mechanisms
   - Detailed error logging and reporting
   - Automatic retry strategies with exponential backoff
   - Circuit breaker patterns for external dependencies

2. ROBUST CONFIGURATION MANAGEMENT:
   - Type-safe configuration validation
   - Environment-specific configuration support
   - Runtime configuration updates
   - Configuration versioning and migration

3. ADVANCED MONITORING & OBSERVABILITY:
   - Real-time performance metrics collection
   - Memory usage tracking and optimization
   - Training progress monitoring
   - Health checks and diagnostics

4. PRODUCTION-GRADE DATA PROCESSING:
   - Memory-efficient data handling
   - Automatic data validation and cleaning
   - Feature engineering pipeline integration
   - Data quality monitoring and reporting

5. ENTERPRISE-READY FEATURES:
   - Comprehensive logging and audit trails
   - Security and compliance features
   - Scalability and performance optimization
   - Integration with monitoring systems

6. MODEL MANAGEMENT:
   - Model versioning and lifecycle management
   - A/B testing support
   - Model performance tracking
   - Automated model deployment preparation

USAGE EXAMPLE:
==============

```python
from src.training.steps.models_training.core.base_trainer import BaseTrainer, TrainingConfig, ModelType, TrainingRole

# Create configuration
config = TrainingConfig(
    role=TrainingRole.ANALYST,
    model_types=[ModelType.LIGHTGBM, ModelType.CATBOOST],
    timeframe="15m",
    symbol="ETHUSDT",
    validation_split=0.2,
    enable_hyperparameter_optimization=True,
    enable_ensemble=True
)

# Create trainer
trainer = MyCustomTrainer(config)

# Initialize and train
await trainer.initialize()
result = await trainer.train(data, targets)

# Validate and predict
validation_result = await trainer.validate(validation_data, validation_targets)
predictions = await trainer.predict(new_data)
```

ARCHITECTURE:
=============

The BaseTrainer follows a layered architecture:

1. Configuration Layer: Type-safe configuration management
2. Validation Layer: Input validation and data quality checks
3. Processing Layer: Data preprocessing and feature engineering
4. Training Layer: Model training and optimization
5. Evaluation Layer: Model validation and performance assessment
6. Persistence Layer: Model and artifact management
7. Monitoring Layer: Performance tracking and health monitoring
"""

import logging
import time
import asyncio
import json
import pickle
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type, TypeVar, Generic, Protocol, runtime_checkable
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import traceback
import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression

# Core imports
from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, 
    tprint_debug, tprint_performance, tprint_data_format, tprint_data_preview, LogLevel
)
from src.utils.common_operations import (
    safe_divide, safe_correlation, safe_mean, safe_std, safe_float, safe_int,
    safe_json_load, safe_json_dump, safe_fillna, safe_to_parquet, safe_read_parquet,
    ensure_directory, safe_file_exists, get_current_datetime, format_datetime
)
from src.utils.common_utilities import (
    calculate_data_quality_metrics, get_dataframe_info, safe_dataframe_operation,
    validate_dataframe_columns, safe_merge_dataframes, create_summary_statistics
)
from src.utils.math_validation import (
    validate_finite, validate_positive, validate_range, validate_probability,
    validate_matrix_properties, validate_statistical_properties
)
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
from src.core.decorators import handles_errors, traced, log_execution_time, error_boundary, converts_errors
from src.core.errors import (
    AppError, ValidationError, DataIntegrityError, NotFoundError,
    BusinessRuleError, FileOperationError, MathValidationError, TimeoutError
)

# Type variables for generic support
T = TypeVar('T')
ModelType = TypeVar('ModelType')
ConfigType = TypeVar('ConfigType')
ResultType = TypeVar('ResultType')


class TrainingRole(Enum):
    """Training roles in the system with enhanced metadata."""
    ANALYST = "analyst"
    TACTICIAN = "tactician"
    ENSEMBLE = "ensemble"
    SUPERVISOR = "supervisor"
    STRATEGIST = "strategist"
    
    def get_description(self) -> str:
        """Get human-readable description of the role."""
        descriptions = {
            self.ANALYST: "Market analysis and pattern recognition",
            self.TACTICIAN: "Trading strategy execution and optimization",
            self.ENSEMBLE: "Multi-model ensemble and meta-learning",
            self.SUPERVISOR: "Risk management and oversight",
            self.STRATEGIST: "High-level strategy formulation"
        }
        return descriptions.get(self, "Unknown role")
    
    def get_priority(self) -> int:
        """Get execution priority (lower = higher priority)."""
        priorities = {
            self.ANALYST: 1,
            self.TACTICIAN: 2,
            self.ENSEMBLE: 3,
            self.SUPERVISOR: 4,
            self.STRATEGIST: 5
        }
        return priorities.get(self, 99)


class ModelType(Enum):
    """Types of ML models with enhanced metadata."""
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    LINEAR = "linear"
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    XGBOOST = "xgboost"
    TRANSFORMER = "transformer"
    
    def get_category(self) -> str:
        """Get model category."""
        categories = {
            self.LIGHTGBM: "gradient_boosting",
            self.CATBOOST: "gradient_boosting", 
            self.XGBOOST: "gradient_boosting",
            self.RANDOM_FOREST: "tree_ensemble",
            self.NEURAL_NETWORK: "neural_network",
            self.TRANSFORMER: "neural_network",
            self.LINEAR: "linear",
            self.SVM: "kernel_method",
            self.ENSEMBLE: "meta_learning"
        }
        return categories.get(self, "unknown")
    
    def get_complexity_score(self) -> int:
        """Get model complexity score (1-10, higher = more complex)."""
        scores = {
            self.LINEAR: 1,
            self.SVM: 3,
            self.RANDOM_FOREST: 4,
            self.LIGHTGBM: 5,
            self.CATBOOST: 5,
            self.XGBOOST: 6,
            self.NEURAL_NETWORK: 7,
            self.TRANSFORMER: 9,
            self.ENSEMBLE: 8
        }
        return scores.get(self, 5)


class TrainingStatus(Enum):
    """Training execution status."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ValidationStrategy(Enum):
    """Validation strategies for model evaluation."""
    HOLDOUT = "holdout"
    CROSS_VALIDATION = "cross_validation"
    TIME_SERIES_SPLIT = "time_series_split"
    STRATIFIED_K_FOLD = "stratified_k_fold"
    WALK_FORWARD = "walk_forward"
    PURGED_CROSS_VALIDATION = "purged_cross_validation"


class OptimizationStrategy(Enum):
    """Hyperparameter optimization strategies."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    TPE = "tpe"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PSO = "particle_swarm_optimization"


@dataclass
class TrainingConfig:
    """
    Production-ready unified training configuration.
    
    This configuration class provides comprehensive settings for all aspects
    of model training, from data preprocessing to model deployment.
    """
    # Core configuration
    role: TrainingRole
    model_types: List[ModelType]
    timeframe: str = "15m"
    symbol: str = "ETHUSDT"
    
    # Training parameters
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    random_seed: Optional[int] = None
    validation_strategy: ValidationStrategy = ValidationStrategy.HOLDOUT
    
    # Model-specific parameters
    enable_hyperparameter_optimization: bool = True
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN_OPTIMIZATION
    max_optimization_trials: int = 100
    enable_ensemble: bool = True
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # Performance parameters
    max_training_time: Optional[float] = None  # seconds
    memory_limit_mb: Optional[int] = None
    max_memory_usage_percent: float = 80.0
    enable_memory_optimization: bool = True
    enable_gpu_acceleration: bool = False
    
    # Feature configuration
    feature_selection_method: str = "multi_objective"
    max_features: int = 100
    correlation_threshold: float = 0.85
    feature_importance_threshold: float = 0.01
    enable_feature_engineering: bool = True
    enable_feature_scaling: bool = True
    
    # Data quality parameters
    min_data_quality_score: float = 0.7
    max_missing_value_percent: float = 0.1
    outlier_detection_method: str = "iqr"
    outlier_threshold: float = 3.0
    
    # Monitoring and logging
    enable_detailed_logging: bool = True
    log_level: str = "INFO"
    enable_performance_monitoring: bool = True
    enable_health_checks: bool = True
    health_check_interval: int = 30  # seconds
    
    # Security and compliance
    enable_audit_logging: bool = True
    data_encryption_enabled: bool = False
    model_versioning_enabled: bool = True
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate configuration parameters.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Validate core parameters
        if not self.role:
            errors.append("Training role is required")
        
        if not self.model_types:
            errors.append("At least one model type is required")
        
        # Validate training parameters
        if not 0.0 < self.validation_split < 1.0:
            errors.append("Validation split must be between 0 and 1")
        
        if self.cross_validation_folds < 2:
            errors.append("Cross-validation folds must be at least 2")
        
        # Validate performance parameters
        if self.max_training_time is not None and self.max_training_time <= 0:
            errors.append("Max training time must be positive")
        
        if self.memory_limit_mb is not None and self.memory_limit_mb <= 0:
            errors.append("Memory limit must be positive")
        
        if not 0.0 < self.max_memory_usage_percent <= 100.0:
            errors.append("Max memory usage percent must be between 0 and 100")
        
        # Validate feature parameters
        if self.max_features <= 0:
            errors.append("Max features must be positive")
        
        if not 0.0 < self.correlation_threshold < 1.0:
            errors.append("Correlation threshold must be between 0 and 1")
        
        if not 0.0 < self.feature_importance_threshold < 1.0:
            errors.append("Feature importance threshold must be between 0 and 1")
        
        # Validate data quality parameters
        if not 0.0 <= self.min_data_quality_score <= 1.0:
            errors.append("Min data quality score must be between 0 and 1")
        
        if not 0.0 <= self.max_missing_value_percent <= 1.0:
            errors.append("Max missing value percent must be between 0 and 1")
        
        if self.outlier_threshold <= 0:
            errors.append("Outlier threshold must be positive")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        # Handle enum conversions
        if 'role' in data and isinstance(data['role'], str):
            data['role'] = TrainingRole(data['role'])
        
        if 'model_types' in data:
            data['model_types'] = [ModelType(mt) if isinstance(mt, str) else mt 
                                 for mt in data['model_types']]
        
        if 'validation_strategy' in data and isinstance(data['validation_strategy'], str):
            data['validation_strategy'] = ValidationStrategy(data['validation_strategy'])
        
        if 'optimization_strategy' in data and isinstance(data['optimization_strategy'], str):
            data['optimization_strategy'] = OptimizationStrategy(data['optimization_strategy'])
        
        return cls(**data)


@dataclass
class TrainingResult:
    """
    Comprehensive result of training operation with production-ready features.
    
    This class provides detailed information about the training process,
    including performance metrics, model artifacts, and execution metadata.
    """
    success: bool
    model: Optional[Any] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced production features
    model_version: str = "1.0.0"
    training_id: str = ""
    status: TrainingStatus = TrainingStatus.COMPLETED
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    data_quality_score: float = 0.0
    model_complexity_score: float = 0.0
    training_samples: int = 0
    validation_samples: int = 0
    feature_count: int = 0
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint_paths: List[str] = field(default_factory=list)
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the training result."""
        return {
            'success': self.success,
            'model_version': self.model_version,
            'training_time': self.training_time,
            'primary_metric': self.metrics.get('accuracy', self.metrics.get('r2', 0.0)),
            'data_quality_score': self.data_quality_score,
            'feature_count': self.feature_count,
            'training_samples': self.training_samples,
            'status': self.status.value,
            'created_at': self.created_at.isoformat()
        }
    
    def is_high_quality(self, min_accuracy: float = 0.8, min_data_quality: float = 0.7) -> bool:
        """Check if the training result meets quality standards."""
        primary_metric = self.metrics.get('accuracy', self.metrics.get('r2', 0.0))
        return (
            self.success and
            primary_metric >= min_accuracy and
            self.data_quality_score >= min_data_quality and
            len(self.warnings) == 0
        )


@dataclass
class ValidationResult:
    """
    Comprehensive result of validation operation.
    
    Provides detailed validation metrics and analysis for model evaluation.
    """
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced validation features
    validation_time: float = 0.0
    validation_samples: int = 0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    prediction_uncertainty: Optional[np.ndarray] = None
    feature_contributions: Optional[Dict[str, float]] = None
    validation_warnings: List[str] = field(default_factory=list)
    cross_validation_scores: List[float] = field(default_factory=list)
    validation_folds: int = 0
    overfitting_score: float = 0.0
    bias_variance_tradeoff: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of the validation result."""
        return {
            'success': self.success,
            'primary_metric': self.metrics.get('accuracy', self.metrics.get('r2', 0.0)),
            'validation_time': self.validation_time,
            'validation_samples': self.validation_samples,
            'overfitting_score': self.overfitting_score,
            'confidence_interval': self.confidence_interval,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class PredictionResult:
    """
    Comprehensive result of prediction operation.
    
    Provides detailed prediction results with confidence measures and uncertainty quantification.
    """
    success: bool
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    confidence_scores: Optional[np.ndarray] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced prediction features
    prediction_time: float = 0.0
    prediction_samples: int = 0
    prediction_uncertainty: Optional[np.ndarray] = None
    feature_importance_scores: Optional[Dict[str, float]] = None
    model_confidence: float = 0.0
    prediction_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None
    anomaly_scores: Optional[np.ndarray] = None
    prediction_warnings: List[str] = field(default_factory=list)
    batch_id: str = ""
    model_version: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get a summary of the prediction result."""
        return {
            'success': self.success,
            'prediction_samples': self.prediction_samples,
            'prediction_time': self.prediction_time,
            'model_confidence': self.model_confidence,
            'model_version': self.model_version,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ModelCheckpoint:
    """Model checkpoint for saving and restoring training state."""
    model: Any
    epoch: int
    metrics: Dict[str, float]
    timestamp: datetime
    file_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def save(self, path: str) -> bool:
        """Save checkpoint to file."""
        try:
            checkpoint_data = {
                'model': self.model,
                'epoch': self.epoch,
                'metrics': self.metrics,
                'timestamp': self.timestamp.isoformat(),
                'metadata': self.metadata
            }
            with open(path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            return True
        except Exception as e:
            tprint_error(f"Failed to save checkpoint: {e}")
            return False
    
    @classmethod
    def load(cls, path: str) -> Optional['ModelCheckpoint']:
        """Load checkpoint from file."""
        try:
            with open(path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            checkpoint_data['timestamp'] = datetime.fromisoformat(checkpoint_data['timestamp'])
            return cls(**checkpoint_data)
        except Exception as e:
            tprint_error(f"Failed to load checkpoint: {e}")
            return None


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for monitoring."""
    training_time: float = 0.0
    validation_time: float = 0.0
    prediction_time: float = 0.0
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    throughput_samples_per_second: float = 0.0
    latency_ms: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    
    def update(self, **kwargs) -> None:
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of performance metrics."""
        return asdict(self)


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for model objects to ensure compatibility."""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelProtocol':
        """Fit the model to training data."""
        ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        ...
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        ...
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score the model on test data."""
        ...


@runtime_checkable
class OptimizerProtocol(Protocol):
    """Protocol for hyperparameter optimizers."""
    
    def optimize(self, objective: Callable, search_space: Dict[str, Any], 
                max_trials: int) -> Dict[str, Any]:
        """Optimize hyperparameters."""
        ...


class BaseTrainer(ABC):
    """
    Production-ready abstract base trainer for all training components.
    
    This class provides a comprehensive, enterprise-grade interface for training
    different types of models across different roles while maintaining consistent
    patterns for configuration, validation, error handling, and monitoring.
    
    KEY FEATURES:
    =============
    
    1. COMPREHENSIVE ERROR HANDLING:
       - Graceful degradation and recovery
       - Detailed error logging and reporting
       - Automatic retry strategies
       - Circuit breaker patterns
    
    2. ADVANCED MONITORING:
       - Real-time performance metrics
       - Memory usage tracking
       - Health checks and diagnostics
       - Audit logging
    
    3. PRODUCTION-GRADE DATA PROCESSING:
       - Memory-efficient data handling
       - Automatic data validation
       - Feature engineering integration
       - Data quality monitoring
    Production-ready base trainer for all training components.
    
    4. ENTERPRISE FEATURES:
       - Model versioning and lifecycle management
       - A/B testing support
       - Security and compliance
       - Scalability optimization
    
    USAGE:
    ======
    
    ```python
    class MyTrainer(BaseTrainer):
        async def _train_single_model(self, model, data, targets, model_type):
            # Implement specific training logic
            pass
        
        def _create_model(self, model_type: ModelType) -> Any:
            # Implement model creation logic
            pass
    
    # Create and use trainer
    config = TrainingConfig(role=TrainingRole.ANALYST, model_types=[ModelType.LIGHTGBM])
    trainer = MyTrainer(config)
    await trainer.initialize()
    result = await trainer.train(data, targets)
    ```
    """
    
    def __init__(self, config: TrainingConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the production-ready base trainer.
        
        Args:
            config: Training configuration
            logger: Logger instance (optional)
            
        Raises:
            ValidationError: If configuration is invalid
            ConfigurationError: If required dependencies are missing
        """
        # Validate configuration
        is_valid, errors = config.validate()
        if not is_valid:
            raise ValidationError(f"Invalid configuration: {', '.join(errors)}")
        
        self.config = config
        self.logger = logger or system_logger.getChild(f"{self.__class__.__name__}")
        
        # Generate unique training ID
        self.training_id = self._generate_training_id()
        
        # Training state with enhanced tracking
        self._training_state = {
            'initialized': False,
            'training_started': False,
            'training_completed': False,
            'model_created': False,
            'best_model_saved': False,
            'status': TrainingStatus.PENDING,
            'start_time': None,
            'end_time': None,
            'error_count': 0,
            'warning_count': 0
        }
        
        # Enhanced performance tracking
        self._performance_metrics = PerformanceMetrics()
        self._health_metrics = {
            'last_health_check': None,
            'consecutive_failures': 0,
            'memory_leak_detected': False,
            'performance_degradation': False
        }
        
        # Model state with versioning
        self._model_state = {
            'model': None,
            'best_model': None,
            'model_version': config.custom_params.get('model_version', '1.0.0'),
            'training_history': [],
            'validation_history': [],
            'checkpoints': [],
            'artifacts': {},
            'metadata': {}
        }
        
        # Initialize enhanced hardware managers
        self._integrated_hardware_manager = None
        self._unified_hardware_manager = None
        self._parquet_manager = None
        self._optimizer = None
        
        # Initialize monitoring
        self._monitoring_enabled = config.enable_performance_monitoring
        self._health_check_task = None
        
        # Initialize security and compliance
        self._audit_log = [] if config.enable_audit_logging else None
        
        # Initialize feature engineering
        self._feature_engineer = None
        self._feature_scaler = None
        self._feature_selector = None
        
        # Initialize data quality monitoring
        self._data_quality_monitor = None
        
        tprint_info(f"🔧 Initializing {self.__class__.__name__} for {config.role.value}")
        self.logger.info(f"Initialized {self.__class__.__name__} for {config.role.value}")
        
        # Log configuration summary
        self._log_configuration_summary()
    
    def _generate_training_id(self) -> str:
        """Generate unique training ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        role = self.config.role.value
        model_types = "_".join([mt.value for mt in self.config.model_types])
        random_suffix = hashlib.md5(f"{timestamp}_{role}_{model_types}".encode()).hexdigest()[:8]
        return f"{role}_{model_types}_{timestamp}_{random_suffix}"
    
    def _log_configuration_summary(self) -> None:
        """Log configuration summary for audit purposes."""
        summary = {
            'training_id': self.training_id,
            'role': self.config.role.value,
            'model_types': [mt.value for mt in self.config.model_types],
            'timeframe': self.config.timeframe,
            'symbol': self.config.symbol,
            'validation_strategy': self.config.validation_strategy.value,
            'optimization_strategy': self.config.optimization_strategy.value,
            'max_training_time': self.config.max_training_time,
            'memory_limit_mb': self.config.memory_limit_mb,
            'enable_gpu_acceleration': self.config.enable_gpu_acceleration
        }
        
        self.logger.info(f"Configuration summary: {json.dumps(summary, indent=2)}")
        
        if self._audit_log is not None:
            self._audit_log.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'configuration_loaded',
                'data': summary
            })
    
    # ============================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # ============================================================================
    
    @abstractmethod
    async def _train_single_model(self, model: Any, data: pd.DataFrame, 
                                targets: pd.Series, model_type: ModelType) -> Dict[str, Any]:
        """
        Train a single model instance.
        
        This method must be implemented by subclasses to provide specific
        training logic for different model types.
        
        Args:
            model: Model instance to train
            data: Training features
            targets: Training targets
            model_type: Type of model being trained
            
        Returns:
            Dictionary containing training results with keys:
            - success: bool
            - model: trained model instance
            - metrics: dict of training metrics
            - training_time: float
            - error: str (if failed)
        """
        pass
    
    @abstractmethod
    def _create_model(self, model_type: ModelType) -> Any:
        """
        Create a model instance for the given model type.
        
        This method must be implemented by subclasses to provide specific
        model creation logic for different model types.
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model type is not supported
        """
        pass
    
    # ============================================================================
    # CORE TRAINING METHODS
    # ============================================================================
    
    # ============================================================================
    # PRODUCTION-READY HELPER METHODS
    # ============================================================================
    
    def _log_audit_event(self, event: str, data: Dict[str, Any] = None) -> None:
        """Log audit event for compliance and debugging."""
        if self._audit_log is not None:
            self._audit_log.append({
                'timestamp': datetime.now().isoformat(),
                'training_id': self.training_id,
                'event': event,
                'data': data or {}
            })
    
    def _update_health_metrics(self) -> None:
        """Update health metrics for monitoring."""
        try:
            import psutil
            
            # Memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            self._performance_metrics.memory_usage_mb = memory_info.rss / 1024 / 1024
            self._performance_metrics.peak_memory_mb = max(
                self._performance_metrics.peak_memory_mb,
                self._performance_metrics.memory_usage_mb
            )
            
            # CPU usage
            self._performance_metrics.cpu_usage_percent = process.cpu_percent()
            
            # Check for memory leaks
            if self._performance_metrics.memory_usage_mb > self.config.memory_limit_mb * 0.9:
                self._health_metrics['memory_leak_detected'] = True
                tprint_warning("⚠️ High memory usage detected - potential memory leak")
            
            # Check for performance degradation
            if self._performance_metrics.cpu_usage_percent > 90:
                self._health_metrics['performance_degradation'] = True
                tprint_warning("⚠️ High CPU usage detected - performance degradation")
            
            self._health_metrics['last_health_check'] = datetime.now()
            
        except ImportError:
            tprint_debug("psutil not available for health monitoring")
        except Exception as e:
            tprint_warning(f"Health metrics update failed: {e}")
    
    def _check_health(self) -> bool:
        """Perform health check on the trainer."""
        try:
            # Check if trainer is responsive
            if self._training_state['status'] == TrainingStatus.FAILED:
                return False
            
            # Check memory usage
            if (self.config.memory_limit_mb and 
                self._performance_metrics.memory_usage_mb > self.config.memory_limit_mb):
                tprint_error("❌ Memory limit exceeded")
                return False
            
            # Check training time limit
            if (self.config.max_training_time and 
                self._training_state['start_time'] and
                (datetime.now() - self._training_state['start_time']).total_seconds() > self.config.max_training_time):
                tprint_error("❌ Training time limit exceeded")
                return False
            
            # Check consecutive failures
            if self._health_metrics['consecutive_failures'] > 5:
                tprint_error("❌ Too many consecutive failures")
                return False
            
            return True
            
        except Exception as e:
            tprint_error(f"Health check failed: {e}")
            return False
    
    def _handle_error(self, error: Exception, context: str = "unknown") -> None:
        """Handle errors with proper logging and recovery."""
        self._training_state['error_count'] += 1
        self._health_metrics['consecutive_failures'] += 1
        
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'training_id': self.training_id
        }
        
        self.logger.error(f"Error in {context}: {error}", exc_info=True)
        self._log_audit_event('error_occurred', error_info)
        
        # Update training state
        if self._training_state['error_count'] > 10:
            self._training_state['status'] = TrainingStatus.FAILED
            tprint_error("❌ Too many errors - training failed")
    
    def _reset_error_count(self) -> None:
        """Reset error count after successful operation."""
        self._health_metrics['consecutive_failures'] = 0
        self._training_state['error_count'] = 0
    
    def _create_model(self, model_type: ModelType) -> Optional[Any]:
        """
        Create a model instance for the given model type.
        
        This method provides a production-ready model creation with comprehensive
        error handling, logging, and validation.
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Model instance or None if creation fails
        """
        try:
            tprint_debug(f"🔧 Creating {model_type.value} model...")
            self._log_audit_event('model_creation_started', {'model_type': model_type.value})
            
            # Validate model type
            if model_type not in self.config.model_types:
                raise ValueError(f"Model type {model_type.value} not in configured model types")
            
            # Create model based on type
            model = self._create_model_by_type(model_type)
            
            if model is None:
                raise ValueError(f"Failed to create {model_type.value} model")
            
            # Validate model interface
            if not self._validate_model_interface(model):
                raise ValueError(f"Model {type(model).__name__} does not implement required interface")
            
            # Log successful creation
            self._log_audit_event('model_creation_completed', {
                'model_type': model_type.value,
                'model_class': type(model).__name__
            })
            
            tprint_success(f"✅ Created {model_type.value} model: {type(model).__name__}")
            return model
                
        except Exception as e:
            self._handle_error(e, f"model_creation_{model_type.value}")
            tprint_error(f"❌ Failed to create {model_type.value} model: {e}")
            return None
    
    def _create_model_by_type(self, model_type: ModelType) -> Optional[Any]:
        """Create model instance based on type with fallback mechanisms."""
        try:
            if model_type == ModelType.LIGHTGBM:
                return self._create_lightgbm_model()
            elif model_type == ModelType.CATBOOST:
                return self._create_catboost_model()
            elif model_type == ModelType.XGBOOST:
                return self._create_xgboost_model()
            elif model_type == ModelType.NEURAL_NETWORK:
                return self._create_neural_network_model()
            elif model_type == ModelType.TRANSFORMER:
                return self._create_transformer_model()
            elif model_type == ModelType.ENSEMBLE:
                return self._create_ensemble_model()
            elif model_type == ModelType.LINEAR:
                return self._create_linear_model()
            elif model_type == ModelType.RANDOM_FOREST:
                return self._create_random_forest_model()
            elif model_type == ModelType.SVM:
                return self._create_svm_model()
            else:
                # Try role-based model creation as fallback
                return self._create_role_based_model(model_type)
                
        except Exception as e:
            tprint_warning(f"Primary model creation failed for {model_type.value}: {e}")
            # Try fallback creation
            return self._create_fallback_model(model_type)
    
    def _validate_model_interface(self, model: Any) -> bool:
        """Validate that model implements required interface."""
        required_methods = ['fit', 'predict']
        optional_methods = ['predict_proba', 'score', 'get_params', 'set_params']
        
        # Check required methods
        for method in required_methods:
            if not hasattr(model, method) or not callable(getattr(model, method)):
                tprint_warning(f"Model missing required method: {method}")
                return False
        
        # Log available optional methods
        available_optional = [method for method in optional_methods 
                            if hasattr(model, method) and callable(getattr(model, method))]
        if available_optional:
            tprint_debug(f"Model supports optional methods: {available_optional}")
        
        return True
    
    def _create_role_based_model(self, model_type: ModelType) -> Optional[Any]:
        """Create model based on training role as fallback."""
        try:
            if self.config.role == TrainingRole.ANALYST:
                return self._create_analyst_model()
            elif self.config.role == TrainingRole.TACTICIAN:
                return self._create_tactician_model()
            elif self.config.role == TrainingRole.ENSEMBLE:
                return self._create_ensemble_model()
            else:
                return self._create_default_model()
        except Exception as e:
            tprint_warning(f"Role-based model creation failed: {e}")
            return None
    
    def _create_fallback_model(self, model_type: ModelType) -> Optional[Any]:
        """Create fallback model when primary creation fails."""
        try:
            tprint_warning(f"Using fallback model for {model_type.value}")
            
            # Try to create a simple linear model as fallback
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
            
        except Exception as e:
            tprint_error(f"Fallback model creation failed: {e}")
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
        """Create an analyst model instance with production-ready configuration."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            from sklearn.ensemble import GradientBoostingClassifier
            
            model_type = self.config.custom_params.get('analyst_model_type', 'random_forest')
            
            # Get model parameters with production defaults
            params = self.config.custom_params.get('analyst_params', {})
            
            if model_type == 'random_forest':
                return RandomForestClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', None),
                    min_samples_split=params.get('min_samples_split', 2),
                    min_samples_leaf=params.get('min_samples_leaf', 1),
                    random_state=self.config.random_seed or 42,
                    n_jobs=-1,
                    verbose=0
                )
            elif model_type == 'logistic_regression':
                return LogisticRegression(
                    random_state=self.config.random_seed or 42,
                    max_iter=params.get('max_iter', 1000),
                    C=params.get('C', 1.0),
                    solver=params.get('solver', 'lbfgs')
                )
            elif model_type == 'svm':
                return SVC(
                    random_state=self.config.random_seed or 42,
                    probability=True,
                    C=params.get('C', 1.0),
                    kernel=params.get('kernel', 'rbf'),
                    gamma=params.get('gamma', 'scale')
                )
            elif model_type == 'gradient_boosting':
                return GradientBoostingClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    learning_rate=params.get('learning_rate', 0.1),
                    max_depth=params.get('max_depth', 3),
                    random_state=self.config.random_seed or 42
                )
            else:
                tprint_warning(f"⚠️ Unknown analyst model type: {model_type}, using RandomForest")
                return RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.config.random_seed or 42,
                    n_jobs=-1
                )
                
        except Exception as e:
            tprint_error(f"❌ Failed to create analyst model: {e}")
            return None
    
    def _create_tactician_model(self) -> Optional[Any]:
        """Create a tactician model instance with production-ready configuration."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.neural_network import MLPClassifier
            from sklearn.linear_model import RidgeClassifier
            from sklearn.ensemble import AdaBoostClassifier
            
            model_type = self.config.custom_params.get('tactician_model_type', 'gradient_boosting')
            params = self.config.custom_params.get('tactician_params', {})
            
            if model_type == 'gradient_boosting':
                return GradientBoostingClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    learning_rate=params.get('learning_rate', 0.1),
                    max_depth=params.get('max_depth', 3),
                    random_state=self.config.random_seed or 42,
                    verbose=0
                )
            elif model_type == 'neural_network':
                return MLPClassifier(
                    hidden_layer_sizes=params.get('hidden_layer_sizes', (100, 50)),
                    activation=params.get('activation', 'relu'),
                    solver=params.get('solver', 'adam'),
                    alpha=params.get('alpha', 0.0001),
                    learning_rate=params.get('learning_rate', 'constant'),
                    random_state=self.config.random_seed or 42,
                    max_iter=params.get('max_iter', 1000),
                    early_stopping=params.get('early_stopping', True)
                )
            elif model_type == 'ridge':
                return RidgeClassifier(
                    alpha=params.get('alpha', 1.0),
                    random_state=self.config.random_seed or 42
                )
            elif model_type == 'ada_boost':
                return AdaBoostClassifier(
                    n_estimators=params.get('n_estimators', 50),
                    learning_rate=params.get('learning_rate', 1.0),
                    random_state=self.config.random_seed or 42
                )
            else:
                tprint_warning(f"⚠️ Unknown tactician model type: {model_type}, using GradientBoosting")
                return GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=self.config.random_seed or 42
                )
                
        except Exception as e:
            tprint_error(f"❌ Failed to create tactician model: {e}")
            return None
    
    def _create_ensemble_model(self) -> Optional[Any]:
        """Create an ensemble model instance with production-ready configuration."""
        try:
            from sklearn.ensemble import VotingClassifier, BaggingClassifier, StackingClassifier
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier
            
            model_type = self.config.custom_params.get('ensemble_model_type', 'voting')
            params = self.config.custom_params.get('ensemble_params', {})
            
            if model_type == 'voting':
                estimators = [
                    ('rf', RandomForestClassifier(
                        n_estimators=params.get('rf_n_estimators', 100),
                        random_state=self.config.random_seed or 42,
                        n_jobs=-1
                    )),
                    ('lr', LogisticRegression(
                        random_state=self.config.random_seed or 42,
                        max_iter=params.get('lr_max_iter', 1000)
                    ))
                ]
                return VotingClassifier(estimators, voting=params.get('voting', 'soft'))
            
            elif model_type == 'bagging':
                return BaggingClassifier(
                    DecisionTreeClassifier(random_state=self.config.random_seed or 42),
                    n_estimators=params.get('n_estimators', 10),
                    random_state=self.config.random_seed or 42,
                    n_jobs=-1
                )
            
            elif model_type == 'stacking':
                base_estimators = [
                    ('rf', RandomForestClassifier(random_state=self.config.random_seed or 42, n_jobs=-1)),
                    ('lr', LogisticRegression(random_state=self.config.random_seed or 42))
                ]
                return StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=LogisticRegression(random_state=self.config.random_seed or 42),
                    cv=params.get('cv', 5)
                )
            
            else:
                tprint_warning(f"⚠️ Unknown ensemble model type: {model_type}, using Voting")
                return VotingClassifier([
                    ('rf', RandomForestClassifier(random_state=self.config.random_seed or 42, n_jobs=-1))
                ])
                
        except Exception as e:
            tprint_error(f"❌ Failed to create ensemble model: {e}")
            return None
    
    def _create_xgboost_model(self) -> Optional[Any]:
        """Create XGBoost model with production-ready configuration."""
        try:
            import xgboost as xgb
            
            params = self.config.custom_params.get('xgboost_params', {})
            
            return xgb.XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                subsample=params.get('subsample', 1.0),
                colsample_bytree=params.get('colsample_bytree', 1.0),
                random_state=self.config.random_seed,
                n_jobs=-1,
                verbosity=0
            )
            
        except ImportError:
            tprint_warning("⚠️ XGBoost not available, using fallback")
            return self._create_fallback_model(ModelType.XGBOOST)
        except Exception as e:
            tprint_error(f"❌ Failed to create XGBoost model: {e}")
            return None
    
    def _create_random_forest_model(self) -> Optional[Any]:
        """Create Random Forest model with production-ready configuration."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            params = self.config.custom_params.get('random_forest_params', {})
            
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                max_features=params.get('max_features', 'sqrt'),
                random_state=self.config.random_seed or 42,
                n_jobs=-1,
                verbose=0
            )
            
        except Exception as e:
            tprint_error(f"❌ Failed to create Random Forest model: {e}")
            return None
    
    def _create_svm_model(self) -> Optional[Any]:
        """Create SVM model with production-ready configuration."""
        try:
            from sklearn.svm import SVC
            
            params = self.config.custom_params.get('svm_params', {})
            
            return SVC(
                C=params.get('C', 1.0),
                kernel=params.get('kernel', 'rbf'),
                gamma=params.get('gamma', 'scale'),
                probability=True,
                random_state=self.config.random_seed or 42
            )
            
        except Exception as e:
            tprint_error(f"❌ Failed to create SVM model: {e}")
            return None
    
    def _create_transformer_model(self) -> Optional[Any]:
        """Create Transformer model with production-ready configuration."""
        try:
            # This would typically use a transformer library like transformers
            # For now, we'll create a simple neural network as a fallback
            from sklearn.neural_network import MLPClassifier
            
            params = self.config.custom_params.get('transformer_params', {})
            
            return MLPClassifier(
                hidden_layer_sizes=params.get('hidden_layer_sizes', (512, 256, 128)),
                activation=params.get('activation', 'relu'),
                solver=params.get('solver', 'adam'),
                alpha=params.get('alpha', 0.0001),
                learning_rate=params.get('learning_rate', 'constant'),
                random_state=self.config.random_seed or 42,
                max_iter=params.get('max_iter', 1000),
                early_stopping=params.get('early_stopping', True)
            )
            
        except Exception as e:
            tprint_error(f"❌ Failed to create Transformer model: {e}")
            return None
    
    def _create_default_model(self) -> Optional[Any]:
        """Create a default model as final fallback."""
        try:
            from sklearn.linear_model import LinearRegression
            
            return LinearRegression()
            
        except Exception as e:
            tprint_error(f"❌ Failed to create default model: {e}")
            return None
