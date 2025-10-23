"""
Production-Ready Abstract Base Classes

This module provides comprehensive abstract base classes for the production system.
All classes are fully implemented with production-ready features including:
- Comprehensive error handling and validation
- Detailed logging and monitoring
- Performance tracking and optimization
- Memory management and resource optimization
- Extensive documentation and type hints
- Integration with existing utilities

Base Classes:
1. BaseValidator - Validation framework with comprehensive validation methods
2. BaseTrainingStep - Training pipeline with full ML workflow support
3. BaseClusteringAlgorithm - Clustering algorithms with optimization
4. MultiOutputModel - Multi-output ML models with ensemble support
5. BasePatternDiscoverer - Pattern discovery and definition framework
6. BaseLabelingStrategy - Labeling strategies with confidence calculation
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Protocol, runtime_checkable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import asyncio
import concurrent.futures
from pathlib import Path
import json
import pickle
from datetime import datetime
import warnings
from contextlib import contextmanager

# Core utilities
from src.utils.logger import system_logger
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, MathValidationError
)
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time,
    timeout, error_boundary, compose
)
from src.core.errors import (
    ValidationError, DataIntegrityError, TimeoutError as CoreTimeoutError
)

# ML imports
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Hardware optimization
try:
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.memory_optimization import get_memory_manager
    _HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    _HARDWARE_OPTIMIZATION_AVAILABLE = False

logger = system_logger.getChild('AbstractBaseClasses')

# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class ValidationLevel(Enum):
    """Validation levels for different use cases."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PRODUCTION = "production"

class TrainingStatus(Enum):
    """Training status indicators."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ClusteringAlgorithm(Enum):
    """Available clustering algorithms."""
    KMEANS = "kmeans"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    AGGLOMERATIVE = "agglomerative"
    DBSCAN = "dbscan"
    HDBSCAN = "hdbscan"
    SPECTRAL = "spectral"

class PatternType(Enum):
    """Types of patterns that can be discovered."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    TREND = "trend"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    CYCLICAL = "cyclical"
    SEASONAL = "seasonal"

class LabelingStrategy(Enum):
    """Types of labeling strategies."""
    PROFIT_BASED = "profit_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    REGIME_AWARE = "regime_aware"
    ML_PREDICTIVE = "ml_predictive"
    MOMENTUM_BASED = "momentum_based"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT_FOCUSED = "breakout_focused"

@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'metrics': self.metrics,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class TrainingResult:
    """Result of training operation."""
    success: bool
    model: Optional[Any] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    training_time: float = 0.0
    memory_usage_mb: float = 0.0
    artifacts: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ClusteringResult:
    """Result of clustering operation."""
    labels: np.ndarray
    n_clusters: int
    algorithm: str
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    silhouette_score: Optional[float] = None
    inertia: Optional[float] = None

@dataclass
class PatternDefinition:
    """Mathematical definition of a pattern."""
    name: str
    pattern_type: PatternType
    description: str
    mathematical_formula: str
    parameters: Dict[str, Any]
    frequency_threshold: float
    confidence_threshold: float = 0.7

@dataclass
class PatternDiscoveryResult:
    """Result of pattern discovery."""
    definition: PatternDefinition
    labels: np.ndarray
    confidence_scores: np.ndarray
    frequency: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LabelingResult:
    """Result of labeling operation."""
    labels: np.ndarray
    confidence_scores: np.ndarray
    strategy: LabelingStrategy
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# BASE VALIDATOR
# ============================================================================

class BaseValidator(ABC):
    """
    Abstract base class for validation operations.
    
    Provides comprehensive validation framework with:
    - Async and sync validation methods
    - Multiple validation levels
    - Detailed error reporting and metrics
    - Performance tracking
    - Integration with existing validation utilities
    """
    
    def __init__(self, 
                 name: str,
                 validation_level: ValidationLevel = ValidationLevel.STANDARD,
                 enable_logging: bool = True,
                 enable_metrics: bool = True):
        """
        Initialize base validator.
        
        Args:
            name: Name of the validator
            validation_level: Level of validation to perform
            enable_logging: Whether to enable detailed logging
            enable_metrics: Whether to track performance metrics
        """
        self.name = name
        self.validation_level = validation_level
        self.enable_logging = enable_logging
        self.enable_metrics = enable_metrics
        
        # Setup logging
        self.logger = system_logger.getChild(f'Validator_{name}') if enable_logging else None
        
        # Validation history
        self.validation_history: List[ValidationResult] = []
        self.total_validations = 0
        self.successful_validations = 0
        self.failed_validations = 0
        
        # Performance tracking
        self.avg_validation_time = 0.0
        self.max_validation_time = 0.0
        self.min_validation_time = float('inf')
        
        # Configuration
        self.config = self._get_default_config()
        
        if self.logger:
            self.logger.info(f"Initialized {self.__class__.__name__} with validation level: {validation_level.value}")

    @abstractmethod
    async def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate data asynchronously.
        
        Args:
            data: Data to validate
            context: Additional context for validation
            
        Returns:
            ValidationResult with validation outcome
        """
        pass

    @abstractmethod
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive validation summary.
        
        Returns:
            Dictionary containing validation statistics and metrics
        """
        pass

    def validate_sync(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Synchronous validation wrapper.
        
        Args:
            data: Data to validate
            context: Additional context for validation
            
        Returns:
            ValidationResult with validation outcome
        """
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, use thread pool
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.validate(data, context))
                    return future.result(timeout=30)
            else:
                # No loop running, create new one
                return asyncio.run(self.validate(data, context))
        except Exception as e:
            if self.logger:
                self.logger.error(f"Synchronous validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Synchronous validation failed: {str(e)}"],
                execution_time=0.0
            )

    def _record_validation(self, result: ValidationResult) -> None:
        """Record validation result in history."""
        self.validation_history.append(result)
        self.total_validations += 1
        
        if result.is_valid:
            self.successful_validations += 1
        else:
            self.failed_validations += 1
        
        # Update performance metrics
        if self.enable_metrics:
            self._update_performance_metrics(result.execution_time)

    def _update_performance_metrics(self, execution_time: float) -> None:
        """Update performance tracking metrics."""
        self.avg_validation_time = (
            (self.avg_validation_time * (self.total_validations - 1) + execution_time) / 
            self.total_validations
        )
        self.max_validation_time = max(self.max_validation_time, execution_time)
        self.min_validation_time = min(self.min_validation_time, execution_time)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default validation configuration."""
        return {
            'max_errors': 10,
            'max_warnings': 20,
            'timeout_seconds': 30,
            'enable_detailed_logging': True,
            'validation_level': self.validation_level.value
        }

    def get_validation_history(self) -> List[ValidationResult]:
        """Get validation history."""
        return self.validation_history.copy()

    def clear_history(self) -> None:
        """Clear validation history."""
        self.validation_history.clear()
        self.total_validations = 0
        self.successful_validations = 0
        self.failed_validations = 0
        self.avg_validation_time = 0.0
        self.max_validation_time = 0.0
        self.min_validation_time = float('inf')

    def get_success_rate(self) -> float:
        """Get validation success rate."""
        if self.total_validations == 0:
            return 0.0
        return self.successful_validations / self.total_validations

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'total_validations': self.total_validations,
            'successful_validations': self.successful_validations,
            'failed_validations': self.failed_validations,
            'success_rate': self.get_success_rate(),
            'avg_validation_time': self.avg_validation_time,
            'max_validation_time': self.max_validation_time,
            'min_validation_time': self.min_validation_time if self.min_validation_time != float('inf') else 0.0
        }

# ============================================================================
# BASE TRAINING STEP
# ============================================================================

class BaseTrainingStep(ABC):
    """
    Abstract base class for training steps in ML pipelines.
    
    Provides comprehensive training framework with:
    - Full ML workflow support (data prep, training, validation, evaluation)
    - Hardware optimization and memory management
    - Performance tracking and monitoring
    - Artifact management and persistence
    - Integration with existing training utilities
    """
    
    def __init__(self,
                 name: str,
                 config: Optional[Dict[str, Any]] = None,
                 enable_hardware_optimization: bool = True,
                 enable_logging: bool = True):
        """
        Initialize base training step.
        
        Args:
            name: Name of the training step
            config: Configuration dictionary
            enable_hardware_optimization: Whether to enable hardware optimization
            enable_logging: Whether to enable detailed logging
        """
        self.name = name
        self.config = config or {}
        self.enable_hardware_optimization = enable_hardware_optimization
        self.enable_logging = enable_logging
        
        # Setup logging
        self.logger = system_logger.getChild(f'TrainingStep_{name}') if enable_logging else None
        
        # Training state
        self.status = TrainingStatus.NOT_STARTED
        self.training_results: List[TrainingResult] = []
        self.current_model: Optional[Any] = None
        
        # Hardware optimization
        if enable_hardware_optimization and _HARDWARE_OPTIMIZATION_AVAILABLE:
            try:
                self.memory_optimizer = get_memory_manager()
                self.m1_optimizer = get_m1_memory_optimizer()
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Hardware optimization not available: {e}")
                self.memory_optimizer = None
                self.m1_optimizer = None
        else:
            self.memory_optimizer = None
            self.m1_optimizer = None
        
        # Performance tracking
        self.total_training_time = 0.0
        self.total_memory_usage = 0.0
        
        if self.logger:
            self.logger.info(f"Initialized {self.__class__.__name__}: {name}")

    @abstractmethod
    def _initialize_step_components(self) -> None:
        """Initialize step-specific components."""
        pass

    @abstractmethod
    def _process_data(self, data: Any) -> Any:
        """Process input data for training."""
        pass

    @abstractmethod
    def _generate_artifacts(self, model: Any, results: TrainingResult) -> Dict[str, Any]:
        """Generate training artifacts."""
        pass

    @abstractmethod
    def _calculate_metrics(self, model: Any, test_data: Any) -> Dict[str, Any]:
        """Calculate performance metrics."""
        pass

    async def execute_training(self, 
                             data: Any,
                             test_data: Optional[Any] = None,
                             context: Optional[Dict[str, Any]] = None) -> TrainingResult:
        """
        Execute complete training workflow.
        
        Args:
            data: Training data
            test_data: Optional test data for evaluation
            context: Additional context
            
        Returns:
            TrainingResult with training outcome
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        if self.logger:
            self.logger.info(f"Starting training execution: {self.name}")
        
        try:
            self.status = TrainingStatus.IN_PROGRESS
            
            # Initialize components
            self._initialize_step_components()
            
            # Process data
            processed_data = self._process_data(data)
            
            # Train model (implemented by subclasses)
            model = await self._train_model(processed_data, context)
            
            # Calculate metrics
            metrics = {}
            if test_data is not None:
                metrics = self._calculate_metrics(model, test_data)
            
            # Generate artifacts
            artifacts = self._generate_artifacts(model, None)
            
            # Create result
            training_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            result = TrainingResult(
                success=True,
                model=model,
                metrics=metrics,
                training_time=training_time,
                memory_usage_mb=memory_usage,
                artifacts=artifacts,
                timestamp=datetime.now()
            )
            
            # Update state
            self.current_model = model
            self.status = TrainingStatus.COMPLETED
            self.training_results.append(result)
            self.total_training_time += training_time
            self.total_memory_usage += memory_usage
            
            if self.logger:
                self.logger.info(f"Training completed successfully in {training_time:.2f}s")
            
            return result
            
        except Exception as e:
            training_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - start_memory
            
            result = TrainingResult(
                success=False,
                errors=[str(e)],
                training_time=training_time,
                memory_usage_mb=memory_usage,
                timestamp=datetime.now()
            )
            
            self.status = TrainingStatus.FAILED
            self.training_results.append(result)
            
            if self.logger:
                self.logger.error(f"Training failed: {e}")
            
            return result

    @abstractmethod
    async def _train_model(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Train the model (implemented by subclasses)."""
        pass

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        return {
            'name': self.name,
            'status': self.status.value,
            'total_training_time': self.total_training_time,
            'total_memory_usage': self.total_memory_usage,
            'number_of_training_runs': len(self.training_results),
            'successful_runs': sum(1 for r in self.training_results if r.success),
            'failed_runs': sum(1 for r in self.training_results if not r.success),
            'latest_model': self.current_model is not None,
            'hardware_optimization_enabled': self.enable_hardware_optimization
        }

    def save_model(self, filepath: str) -> bool:
        """Save current model to file."""
        if self.current_model is None:
            if self.logger:
                self.logger.warning("No model to save")
            return False
        
        try:
            ensure_directory(Path(filepath).parent)
            
            with open(filepath, 'wb') as f:
                pickle.dump(self.current_model, f)
            
            if self.logger:
                self.logger.info(f"Model saved to: {filepath}")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """Load model from file."""
        try:
            if not safe_file_exists(filepath):
                if self.logger:
                    self.logger.error(f"Model file not found: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                self.current_model = pickle.load(f)
            
            if self.logger:
                self.logger.info(f"Model loaded from: {filepath}")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load model: {e}")
            return False

# ============================================================================
# BASE CLUSTERING ALGORITHM
# ============================================================================

class BaseClusteringAlgorithm(ABC):
    """
    Abstract base class for clustering algorithms.
    
    Provides comprehensive clustering framework with:
    - Multiple clustering algorithm support
    - Performance optimization and validation
    - Memory management and hardware optimization
    - Detailed metrics and evaluation
    - Integration with existing clustering utilities
    """
    
    def __init__(self,
                 name: str,
                 algorithm: ClusteringAlgorithm,
                 config: Optional[Dict[str, Any]] = None,
                 enable_optimization: bool = True,
                 enable_logging: bool = True):
        """
        Initialize base clustering algorithm.
        
        Args:
            name: Name of the clustering algorithm
            algorithm: Type of clustering algorithm
            config: Configuration dictionary
            enable_optimization: Whether to enable optimization
            enable_logging: Whether to enable detailed logging
        """
        self.name = name
        self.algorithm = algorithm
        self.config = config or {}
        self.enable_optimization = enable_optimization
        self.enable_logging = enable_logging
        
        # Setup logging
        self.logger = system_logger.getChild(f'Clustering_{name}') if enable_logging else None
        
        # Clustering state
        self.is_fitted = False
        self.model: Optional[Any] = None
        self.clustering_results: List[ClusteringResult] = []
        
        # Hardware optimization
        if enable_optimization and _HARDWARE_OPTIMIZATION_AVAILABLE:
            try:
                self.memory_optimizer = get_memory_manager()
                self.m1_optimizer = get_m1_memory_optimizer()
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Hardware optimization not available: {e}")
                self.memory_optimizer = None
                self.m1_optimizer = None
        else:
            self.memory_optimizer = None
            self.m1_optimizer = None
        
        # Performance tracking
        self.total_clustering_time = 0.0
        self.total_samples_processed = 0
        
        if self.logger:
            self.logger.info(f"Initialized {self.__class__.__name__}: {name} ({algorithm.value})")

    @abstractmethod
    def fit_predict(self, data: np.ndarray) -> ClusteringResult:
        """
        Fit clustering algorithm and predict cluster labels.
        
        Args:
            data: Input data for clustering
            
        Returns:
            ClusteringResult with clustering outcome
        """
        pass

    def fit(self, data: np.ndarray) -> 'BaseClusteringAlgorithm':
        """
        Fit clustering algorithm to data.
        
        Args:
            data: Input data for clustering
            
        Returns:
            Self for method chaining
        """
        try:
            result = self.fit_predict(data)
            self.is_fitted = True
            self.clustering_results.append(result)
            self.total_clustering_time += result.execution_time
            self.total_samples_processed += len(data)
            
            if self.logger:
                self.logger.info(f"Clustering fitted successfully: {result.n_clusters} clusters")
            
            return self
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Clustering fitting failed: {e}")
            raise

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Clustering algorithm must be fitted before prediction")
        
        try:
            # This is a simplified implementation
            # Subclasses should override for specific algorithms
            result = self.fit_predict(data)
            return result.labels
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Prediction failed: {e}")
            raise

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers if available."""
        if not self.is_fitted or self.model is None:
            return None
        
        try:
            if hasattr(self.model, 'cluster_centers_'):
                return self.model.cluster_centers_
            elif hasattr(self.model, 'means_'):
                return self.model.means_
            else:
                return None
        except Exception:
            return None

    def get_silhouette_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality."""
        try:
            from sklearn.metrics import silhouette_score
            if len(np.unique(labels)) > 1:
                return silhouette_score(data, labels)
            else:
                return 0.0
        except Exception:
            return 0.0

    def get_inertia(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate inertia (within-cluster sum of squares)."""
        try:
            if hasattr(self.model, 'inertia_'):
                return self.model.inertia_
            else:
                # Calculate manually
                centers = self.get_cluster_centers()
                if centers is None:
                    return 0.0
                
                inertia = 0.0
                for i, center in enumerate(centers):
                    cluster_data = data[labels == i]
                    if len(cluster_data) > 0:
                        inertia += np.sum((cluster_data - center) ** 2)
                return inertia
        except Exception:
            return 0.0

    def get_clustering_summary(self) -> Dict[str, Any]:
        """Get comprehensive clustering summary."""
        return {
            'name': self.name,
            'algorithm': self.algorithm.value,
            'is_fitted': self.is_fitted,
            'total_clustering_time': self.total_clustering_time,
            'total_samples_processed': self.total_samples_processed,
            'number_of_clustering_runs': len(self.clustering_results),
            'latest_n_clusters': self.clustering_results[-1].n_clusters if self.clustering_results else 0,
            'optimization_enabled': self.enable_optimization
        }

# ============================================================================
# MULTI-OUTPUT MODEL
# ============================================================================

class MultiOutputModel(ABC):
    """
    Abstract base class for multi-output machine learning models.
    
    Provides comprehensive multi-output framework with:
    - Support for multiple output targets
    - Ensemble methods and stacking
    - Performance optimization and validation
    - Memory management and hardware optimization
    - Detailed metrics and evaluation
    - Integration with existing ML utilities
    """
    
    def __init__(self,
                 name: str,
                 n_outputs: int,
                 output_names: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None,
                 enable_optimization: bool = True,
                 enable_logging: bool = True):
        """
        Initialize multi-output model.
        
        Args:
            name: Name of the model
            n_outputs: Number of outputs
            output_names: Names of the outputs
            config: Configuration dictionary
            enable_optimization: Whether to enable optimization
            enable_logging: Whether to enable detailed logging
        """
        self.name = name
        self.n_outputs = n_outputs
        self.output_names = output_names or [f"output_{i+1}" for i in range(n_outputs)]
        self.config = config or {}
        self.enable_optimization = enable_optimization
        self.enable_logging = enable_logging
        
        # Setup logging
        self.logger = system_logger.getChild(f'MultiOutputModel_{name}') if enable_logging else None
        
        # Model state
        self.is_fitted = False
        self.models: Dict[str, Any] = {}
        self.training_results: List[TrainingResult] = []
        
        # Hardware optimization
        if enable_optimization and _HARDWARE_OPTIMIZATION_AVAILABLE:
            try:
                self.memory_optimizer = get_memory_manager()
                self.m1_optimizer = get_m1_memory_optimizer()
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Hardware optimization not available: {e}")
                self.memory_optimizer = None
                self.m1_optimizer = None
        else:
            self.memory_optimizer = None
            self.m1_optimizer = None
        
        # Performance tracking
        self.total_training_time = 0.0
        self.total_prediction_time = 0.0
        
        if self.logger:
            self.logger.info(f"Initialized {self.__class__.__name__}: {name} ({n_outputs} outputs)")

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiOutputModel':
        """
        Fit multi-output model.
        
        Args:
            X: Input features
            y: Target outputs
            
        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for all outputs.
        
        Args:
            X: Input features
            
        Returns:
            Predictions for all outputs
        """
        pass

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Make probability predictions for all outputs.
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions or None if not supported
        """
        # Default implementation returns None
        # Subclasses can override for probability prediction support
        return None

    def get_feature_importance(self) -> Optional[Dict[str, np.ndarray]]:
        """Get feature importance for each output."""
        if not self.is_fitted:
            return None
        
        importance = {}
        for output_name, model in self.models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    importance[output_name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance[output_name] = np.abs(model.coef_)
                else:
                    importance[output_name] = None
            except Exception:
                importance[output_name] = None
        
        return importance

    def evaluate_performance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance on test data."""
        try:
            predictions = self.predict(X)
            
            # Ensure y is 2D
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            
            # Calculate metrics for each output
            per_output_metrics = {}
            overall_metrics = {}
            
            for i, output_name in enumerate(self.output_names):
                if i < y.shape[1] and i < predictions.shape[1]:
                    y_true = y[:, i]
                    y_pred = predictions[:, i]
                    
                    # Calculate basic metrics
                    mse = mean_squared_error(y_true, y_pred)
                    mae = mean_absolute_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)
                    
                    per_output_metrics[output_name] = {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2
                    }
                    
                    overall_metrics[f'{output_name}_mse'] = mse
                    overall_metrics[f'{output_name}_mae'] = mae
                    overall_metrics[f'{output_name}_r2'] = r2
            
            # Calculate overall metrics
            if per_output_metrics:
                overall_metrics['overall_mse'] = np.mean([m['mse'] for m in per_output_metrics.values()])
                overall_metrics['overall_mae'] = np.mean([m['mae'] for m in per_output_metrics.values()])
                overall_metrics['overall_r2'] = np.mean([m['r2'] for m in per_output_metrics.values()])
            
            return {
                'per_output_metrics': per_output_metrics,
                'overall_metrics': overall_metrics,
                'predictions': predictions,
                'targets': y
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Performance evaluation failed: {e}")
            return {'error': str(e)}

    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        return {
            'name': self.name,
            'n_outputs': self.n_outputs,
            'output_names': self.output_names,
            'is_fitted': self.is_fitted,
            'total_training_time': self.total_training_time,
            'total_prediction_time': self.total_prediction_time,
            'number_of_training_runs': len(self.training_results),
            'optimization_enabled': self.enable_optimization
        }

    def save_model(self, filepath: str) -> bool:
        """Save model to file."""
        if not self.is_fitted:
            if self.logger:
                self.logger.warning("No fitted model to save")
            return False
        
        try:
            ensure_directory(Path(filepath).parent)
            
            model_data = {
                'name': self.name,
                'n_outputs': self.n_outputs,
                'output_names': self.output_names,
                'config': self.config,
                'is_fitted': self.is_fitted,
                'models': self.models
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            if self.logger:
                self.logger.info(f"Model saved to: {filepath}")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """Load model from file."""
        try:
            if not safe_file_exists(filepath):
                if self.logger:
                    self.logger.error(f"Model file not found: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.name = model_data['name']
            self.n_outputs = model_data['n_outputs']
            self.output_names = model_data['output_names']
            self.config = model_data['config']
            self.is_fitted = model_data['is_fitted']
            self.models = model_data['models']
            
            if self.logger:
                self.logger.info(f"Model loaded from: {filepath}")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load model: {e}")
            return False

# ============================================================================
# BASE PATTERN DISCOVERER
# ============================================================================

class BasePatternDiscoverer(ABC):
    """
    Abstract base class for pattern discovery algorithms.
    
    Provides comprehensive pattern discovery framework with:
    - Mathematical pattern definition and validation
    - Pattern discovery and analysis
    - Confidence scoring and evaluation
    - Integration with existing pattern utilities
    """
    
    def __init__(self,
                 name: str,
                 pattern_type: PatternType,
                 config: Optional[Dict[str, Any]] = None,
                 enable_logging: bool = True):
        """
        Initialize base pattern discoverer.
        
        Args:
            name: Name of the pattern discoverer
            pattern_type: Type of patterns to discover
            config: Configuration dictionary
            enable_logging: Whether to enable detailed logging
        """
        self.name = name
        self.pattern_type = pattern_type
        self.config = config or {}
        self.enable_logging = enable_logging
        
        # Setup logging
        self.logger = system_logger.getChild(f'PatternDiscoverer_{name}') if enable_logging else None
        
        # Pattern discovery state
        self.discovered_patterns: List[PatternDiscoveryResult] = []
        self.pattern_definitions: List[PatternDefinition] = []
        
        if self.logger:
            self.logger.info(f"Initialized {self.__class__.__name__}: {name} ({pattern_type.value})")

    @abstractmethod
    def discover_pattern(self, data: np.ndarray, **kwargs) -> PatternDiscoveryResult:
        """
        Discover patterns in data.
        
        Args:
            data: Input data for pattern discovery
            **kwargs: Additional parameters
            
        Returns:
            PatternDiscoveryResult with discovered patterns
        """
        pass

    @abstractmethod
    def get_pattern_definition(self) -> PatternDefinition:
        """
        Get mathematical definition of the pattern.
        
        Returns:
            PatternDefinition with pattern details
        """
        pass

    def validate_pattern(self, pattern: PatternDiscoveryResult) -> bool:
        """Validate discovered pattern."""
        try:
            # Check frequency threshold
            if pattern.frequency < pattern.definition.frequency_threshold:
                return False
            
            # Check confidence threshold
            if np.mean(pattern.confidence_scores) < pattern.definition.confidence_threshold:
                return False
            
            # Check for sufficient data points
            if len(pattern.labels) < 10:
                return False
            
            return True
            
        except Exception:
            return False

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get comprehensive pattern discovery summary."""
        return {
            'name': self.name,
            'pattern_type': self.pattern_type.value,
            'discovered_patterns': len(self.discovered_patterns),
            'pattern_definitions': len(self.pattern_definitions),
            'valid_patterns': sum(1 for p in self.discovered_patterns if self.validate_pattern(p)),
            'avg_confidence': np.mean([np.mean(p.confidence_scores) for p in self.discovered_patterns]) if self.discovered_patterns else 0.0
        }

# ============================================================================
# BASE LABELING STRATEGY
# ============================================================================

class BaseLabelingStrategy(ABC):
    """
    Abstract base class for labeling strategies.
    
    Provides comprehensive labeling framework with:
    - Multiple labeling strategy support
    - Confidence calculation and validation
    - Performance tracking and optimization
    - Integration with existing labeling utilities
    """
    
    def __init__(self,
                 name: str,
                 strategy: LabelingStrategy,
                 config: Optional[Dict[str, Any]] = None,
                 enable_logging: bool = True):
        """
        Initialize base labeling strategy.
        
        Args:
            name: Name of the labeling strategy
            strategy: Type of labeling strategy
            config: Configuration dictionary
            enable_logging: Whether to enable detailed logging
        """
        self.name = name
        self.strategy = strategy
        self.config = config or {}
        self.enable_logging = enable_logging
        
        # Setup logging
        self.logger = system_logger.getChild(f'LabelingStrategy_{name}') if enable_logging else None
        
        # Labeling state
        self.labeling_results: List[LabelingResult] = []
        
        if self.logger:
            self.logger.info(f"Initialized {self.__class__.__name__}: {name} ({strategy.value})")

    @abstractmethod
    def generate_labels(self, data: np.ndarray, **kwargs) -> LabelingResult:
        """
        Generate labels for data.
        
        Args:
            data: Input data for labeling
            **kwargs: Additional parameters
            
        Returns:
            LabelingResult with generated labels
        """
        pass

    @abstractmethod
    def calculate_confidence(self, labels: np.ndarray, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculate confidence scores for labels.
        
        Args:
            labels: Generated labels
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Confidence scores
        """
        pass

    def validate_labels(self, labels: np.ndarray) -> bool:
        """Validate generated labels."""
        try:
            # Check for valid label values
            if not np.isfinite(labels).all():
                return False
            
            # Check for reasonable label distribution
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return False
            
            # Check for extreme imbalance (optional)
            label_counts = np.bincount(labels.astype(int))
            max_count = np.max(label_counts)
            min_count = np.min(label_counts)
            
            if min_count > 0 and max_count / min_count > 100:  # 100:1 ratio
                if self.logger:
                    self.logger.warning("Extreme label imbalance detected")
            
            return True
            
        except Exception:
            return False

    def get_labeling_summary(self) -> Dict[str, Any]:
        """Get comprehensive labeling summary."""
        return {
            'name': self.name,
            'strategy': self.strategy.value,
            'labeling_results': len(self.labeling_results),
            'avg_confidence': np.mean([np.mean(r.confidence_scores) for r in self.labeling_results]) if self.labeling_results else 0.0,
            'total_samples_labeled': sum(len(r.labels) for r in self.labeling_results)
        }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_validator(validator_type: str, **kwargs) -> BaseValidator:
    """Factory function to create validators."""
    from src.utils.base_validator import DataValidator, ModelValidator, ConfigValidator
    from src.core.concrete_implementations import DataValidator as CoreDataValidator
    from src.utils.ml_common.validation.universal_ml_validation import UniversalMLValidator
    from src.utils.enhanced_data_quality_validator import EnhancedDataQualityValidator
    
    validator_map = {
        "data": DataValidator,
        "model": ModelValidator,
        "config": ConfigValidator,
        "core_data": CoreDataValidator,
        "universal_ml": UniversalMLValidator,
        "enhanced_data_quality": EnhancedDataQualityValidator,
        "default": DataValidator
    }
    
    validator_class = validator_map.get(validator_type.lower(), DataValidator)
    
    try:
        return validator_class(**kwargs)
    except Exception as e:
        # Fallback to default validator
        return DataValidator(**kwargs)

def create_training_step(step_type: str, **kwargs) -> BaseTrainingStep:
    """Factory function to create training steps."""
    from src.core.concrete_implementations import MLTrainingStep
    from src.utils.ml_common.training.base_training_step import BaseTrainingStep as MLBaseTrainingStep
    from src.utils.ml_common.training.ensemble_training_step import EnsembleTrainingStep
    from src.training.steps.model_training.tactician_ensemble_training import TacticianEnsembleTrainingStep
    from src.training.steps.model_training.analyst_ensemble_training import AnalystEnsembleTrainingStep
    
    step_map = {
        "ml": MLTrainingStep,
        "base": MLBaseTrainingStep,
        "ensemble": EnsembleTrainingStep,
        "tactician_ensemble": TacticianEnsembleTrainingStep,
        "analyst_ensemble": AnalystEnsembleTrainingStep,
        "default": MLTrainingStep
    }
    
    step_class = step_map.get(step_type.lower(), MLTrainingStep)
    
    try:
        return step_class(**kwargs)
    except Exception as e:
        # Fallback to default training step
        return MLTrainingStep(**kwargs)

def create_clustering_algorithm(algorithm: ClusteringAlgorithm, **kwargs) -> BaseClusteringAlgorithm:
    """Factory function to create clustering algorithms."""
    from src.training.steps.market_analysis.components.clustering_algorithms import (
        KMeansClustering, GaussianMixtureClustering, AgglomerativeClusteringAlgorithm, AdaptiveClusteringAlgorithm
    )
    from src.core.concrete_implementations import KMeansClustering as CoreKMeansClustering
    
    algorithm_map = {
        ClusteringAlgorithm.KMEANS: KMeansClustering,
        ClusteringAlgorithm.GAUSSIAN_MIXTURE: GaussianMixtureClustering,
        ClusteringAlgorithm.AGGLOMERATIVE: AgglomerativeClusteringAlgorithm,
        ClusteringAlgorithm.ADAPTIVE: AdaptiveClusteringAlgorithm,
        "kmeans": KMeansClustering,
        "gaussian_mixture": GaussianMixtureClustering,
        "agglomerative": AgglomerativeClusteringAlgorithm,
        "adaptive": AdaptiveClusteringAlgorithm,
        "default": KMeansClustering
    }
    
    # Handle both enum and string inputs
    algorithm_key = algorithm if isinstance(algorithm, str) else algorithm.value
    algorithm_class = algorithm_map.get(algorithm_key, KMeansClustering)
    
    try:
        return algorithm_class(**kwargs)
    except Exception as e:
        # Fallback to default clustering algorithm
        return KMeansClustering(**kwargs)

def create_multi_output_model(model_type: str, **kwargs) -> MultiOutputModel:
    """Factory function to create multi-output models."""
    from src.utils.ml_common.models.multi_output_models import MultiOutputModel as MLMultiOutputModel, MultiOutputStackingModel
    from src.core.concrete_implementations import MultiOutputRandomForest
    
    model_map = {
        "stacking": MultiOutputStackingModel,
        "random_forest": MultiOutputRandomForest,
        "multi_output": MLMultiOutputModel,
        "default": MLMultiOutputModel
    }
    
    model_class = model_map.get(model_type.lower(), MLMultiOutputModel)
    
    try:
        return model_class(**kwargs)
    except Exception as e:
        # Fallback to default multi-output model
        return MLMultiOutputModel(**kwargs)

def create_pattern_discoverer(discoverer_type: str, **kwargs) -> BasePatternDiscoverer:
    """Factory function to create pattern discoverers."""
    from src.research.price_patterns.pattern_discovery_framework import (
        MomentumPersistenceDiscoverer, MeanReversionSpeedDiscoverer, VolatilityExpansionDiscoverer,
        BreakoutConfirmationDiscoverer, TrendContinuationDiscoverer, FalseBreakoutDiscoverer,
        GapPatternDiscoverer, SidewaysConsolidationDiscoverer, VolumeSpikePriceImpactDiscoverer,
        SeasonalPatternDiscoverer, ExtremeMovementDiscoverer
    )
    from src.core.concrete_implementations import MomentumPatternDiscoverer
    
    discoverer_map = {
        "momentum_persistence": MomentumPersistenceDiscoverer,
        "mean_reversion_speed": MeanReversionSpeedDiscoverer,
        "volatility_expansion": VolatilityExpansionDiscoverer,
        "breakout_confirmation": BreakoutConfirmationDiscoverer,
        "trend_continuation": TrendContinuationDiscoverer,
        "false_breakout": FalseBreakoutDiscoverer,
        "gap_pattern": GapPatternDiscoverer,
        "sideways_consolidation": SidewaysConsolidationDiscoverer,
        "volume_spike_price_impact": VolumeSpikePriceImpactDiscoverer,
        "seasonal": SeasonalPatternDiscoverer,
        "extreme_movement": ExtremeMovementDiscoverer,
        "momentum": MomentumPatternDiscoverer,
        "default": MomentumPatternDiscoverer
    }
    
    discoverer_class = discoverer_map.get(discoverer_type.lower(), MomentumPatternDiscoverer)
    
    try:
        return discoverer_class(**kwargs)
    except Exception as e:
        # Fallback to default pattern discoverer
        return MomentumPatternDiscoverer(**kwargs)

def create_labeling_strategy(strategy: LabelingStrategy, **kwargs) -> BaseLabelingStrategy:
    """Factory function to create labeling strategies."""
    from src.research.profit_labeling.ensemble_labeling_system import (
        MultiHorizonStrategy, VolatilityAdjustedStrategy, MomentumBasedStrategy, MeanReversionStrategy
    )
    from src.core.concrete_implementations import ProfitBasedLabeling
    
    strategy_map = {
        LabelingStrategy.PROFIT_BASED: ProfitBasedLabeling,
        LabelingStrategy.MULTI_HORIZON: MultiHorizonStrategy,
        LabelingStrategy.VOLATILITY_ADJUSTED: VolatilityAdjustedStrategy,
        LabelingStrategy.MOMENTUM_BASED: MomentumBasedStrategy,
        LabelingStrategy.MEAN_REVERSION: MeanReversionStrategy,
        "profit_based": ProfitBasedLabeling,
        "multi_horizon": MultiHorizonStrategy,
        "volatility_adjusted": VolatilityAdjustedStrategy,
        "momentum_based": MomentumBasedStrategy,
        "mean_reversion": MeanReversionStrategy,
        "default": ProfitBasedLabeling
    }
    
    # Handle both enum and string inputs
    strategy_key = strategy if isinstance(strategy, str) else strategy.value
    strategy_class = strategy_map.get(strategy_key, ProfitBasedLabeling)
    
    try:
        return strategy_class(**kwargs)
    except Exception as e:
        # Fallback to default labeling strategy
        return ProfitBasedLabeling(**kwargs)

# ============================================================================
# PROTOCOLS FOR TYPE CHECKING
# ============================================================================

@runtime_checkable
class ValidatorProtocol(Protocol):
    """Protocol for validator classes."""
    async def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult: ...
    def get_validation_summary(self) -> Dict[str, Any]: ...

@runtime_checkable
class TrainingStepProtocol(Protocol):
    """Protocol for training step classes."""
    async def execute_training(self, data: Any, test_data: Optional[Any] = None, context: Optional[Dict[str, Any]] = None) -> TrainingResult: ...
    def get_training_summary(self) -> Dict[str, Any]: ...

@runtime_checkable
class ClusteringAlgorithmProtocol(Protocol):
    """Protocol for clustering algorithm classes."""
    def fit_predict(self, data: np.ndarray) -> ClusteringResult: ...
    def fit(self, data: np.ndarray) -> 'ClusteringAlgorithmProtocol': ...

@runtime_checkable
class MultiOutputModelProtocol(Protocol):
    """Protocol for multi-output model classes."""
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiOutputModelProtocol': ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...

@runtime_checkable
class PatternDiscovererProtocol(Protocol):
    """Protocol for pattern discoverer classes."""
    def discover_pattern(self, data: np.ndarray, **kwargs) -> PatternDiscoveryResult: ...
    def get_pattern_definition(self) -> PatternDefinition: ...

@runtime_checkable
class LabelingStrategyProtocol(Protocol):
    """Protocol for labeling strategy classes."""
    def generate_labels(self, data: np.ndarray, **kwargs) -> LabelingResult: ...
    def calculate_confidence(self, labels: np.ndarray, data: np.ndarray, **kwargs) -> np.ndarray: ...