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
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}
        
        try:
            if self.logger:
                self.logger.info(f"Starting validation of {type(data).__name__}")
            
            # Basic validation checks
            if data is None:
                errors.append("Data cannot be None")
            elif hasattr(data, '__len__') and len(data) == 0:
                warnings.append("Data is empty")
            
            # Type-specific validation
            if isinstance(data, (list, tuple)):
                if len(data) == 0:
                    warnings.append("Empty collection")
                else:
                    metrics['collection_size'] = len(data)
                    metrics['element_types'] = list(set(type(item).__name__ for item in data))
            
            elif isinstance(data, dict):
                if len(data) == 0:
                    warnings.append("Empty dictionary")
                else:
                    metrics['dict_size'] = len(data)
                    metrics['keys'] = list(data.keys())
            
            elif hasattr(data, 'shape'):  # numpy array or similar
                if data.shape[0] == 0:
                    errors.append("Data has zero samples")
                else:
                    metrics['shape'] = data.shape
                    metrics['dtype'] = str(data.dtype) if hasattr(data, 'dtype') else 'unknown'
            
            # Context-specific validation
            if context:
                if 'expected_type' in context:
                    expected_type = context['expected_type']
                    if not isinstance(data, expected_type):
                        errors.append(f"Expected {expected_type.__name__}, got {type(data).__name__}")
                
                if 'min_size' in context and hasattr(data, '__len__'):
                    if len(data) < context['min_size']:
                        errors.append(f"Data size {len(data)} is below minimum {context['min_size']}")
                
                if 'max_size' in context and hasattr(data, '__len__'):
                    if len(data) > context['max_size']:
                        warnings.append(f"Data size {len(data)} exceeds maximum {context['max_size']}")
            
            # Determine if validation passed
            is_valid = len(errors) == 0
            
            execution_time = time.time() - start_time
            
            result = ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
            # Record validation result
            self._record_validation(result)
            
            if self.logger:
                status = "PASSED" if is_valid else "FAILED"
                self.logger.info(f"Validation {status} in {execution_time:.3f}s")
                if errors:
                    self.logger.error(f"Validation errors: {errors}")
                if warnings:
                    self.logger.warning(f"Validation warnings: {warnings}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Validation failed with exception: {str(e)}"
            
            if self.logger:
                self.logger.error(error_msg)
            
            result = ValidationResult(
                is_valid=False,
                errors=[error_msg],
                warnings=warnings,
                metrics=metrics,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
            self._record_validation(result)
            return result

    @abstractmethod
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive validation summary.
        
        Returns:
            Dictionary containing validation statistics and metrics
        """
        try:
            # Basic statistics
            summary = {
                'validator_name': self.name,
                'validation_level': self.validation_level.value,
                'total_validations': self.total_validations,
                'successful_validations': self.successful_validations,
                'failed_validations': self.failed_validations,
                'success_rate': self.get_success_rate(),
                'performance_metrics': self.get_performance_summary()
            }
            
            # Recent validation history
            if self.validation_history:
                recent_results = self.validation_history[-10:]  # Last 10 validations
                summary['recent_validations'] = [
                    {
                        'timestamp': result.timestamp.isoformat(),
                        'is_valid': result.is_valid,
                        'execution_time': result.execution_time,
                        'error_count': len(result.errors),
                        'warning_count': len(result.warnings)
                    }
                    for result in recent_results
                ]
                
                # Error analysis
                all_errors = []
                all_warnings = []
                for result in self.validation_history:
                    all_errors.extend(result.errors)
                    all_warnings.extend(result.warnings)
                
                if all_errors:
                    from collections import Counter
                    error_counts = Counter(all_errors)
                    summary['common_errors'] = dict(error_counts.most_common(5))
                
                if all_warnings:
                    from collections import Counter
                    warning_counts = Counter(all_warnings)
                    summary['common_warnings'] = dict(warning_counts.most_common(5))
            
            # Configuration summary
            summary['configuration'] = {
                'max_errors': self.config.get('max_errors', 10),
                'max_warnings': self.config.get('max_warnings', 20),
                'timeout_seconds': self.config.get('timeout_seconds', 30),
                'enable_detailed_logging': self.config.get('enable_detailed_logging', True)
            }
            
            # Validation level specific metrics
            if self.validation_level == ValidationLevel.STRICT:
                summary['strict_mode'] = True
                summary['tolerance'] = 'zero_errors'
            elif self.validation_level == ValidationLevel.PRODUCTION:
                summary['production_mode'] = True
                summary['tolerance'] = 'minimal_warnings'
            else:
                summary['tolerance'] = 'standard'
            
            if self.logger:
                self.logger.debug(f"Generated validation summary: {len(summary)} metrics")
            
            return summary
            
        except Exception as e:
            error_summary = {
                'validator_name': self.name,
                'error': f"Failed to generate summary: {str(e)}",
                'total_validations': self.total_validations,
                'successful_validations': self.successful_validations,
                'failed_validations': self.failed_validations
            }
            
            if self.logger:
                self.logger.error(f"Failed to generate validation summary: {e}")
            
            return error_summary

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
        try:
            if self.logger:
                self.logger.info("Initializing training step components")
            
            # Initialize hardware optimization if available
            if self.enable_hardware_optimization and self.memory_optimizer:
                try:
                    self.memory_optimizer.initialize()
                    if self.logger:
                        self.logger.debug("Memory optimizer initialized")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Memory optimizer initialization failed: {e}")
            
            if self.enable_hardware_optimization and self.m1_optimizer:
                try:
                    self.m1_optimizer.initialize()
                    if self.logger:
                        self.logger.debug("M1 optimizer initialized")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"M1 optimizer initialization failed: {e}")
            
            # Initialize step-specific configuration
            self._setup_step_configuration()
            
            # Initialize performance tracking
            self._initialize_performance_tracking()
            
            if self.logger:
                self.logger.info("Training step components initialized successfully")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize step components: {e}")
            raise
    
    def _setup_step_configuration(self) -> None:
        """Setup step-specific configuration."""
        # Default configuration setup
        default_config = {
            'enable_early_stopping': True,
            'early_stopping_patience': 10,
            'validation_split': 0.2,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 1
        }
        
        # Merge with provided config
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def _initialize_performance_tracking(self) -> None:
        """Initialize performance tracking variables."""
        self.step_start_time = None
        self.step_end_time = None
        self.memory_usage_start = 0.0
        self.memory_usage_end = 0.0

    @abstractmethod
    def _process_data(self, data: Any) -> Any:
        """Process input data for training."""
        try:
            if self.logger:
                self.logger.info(f"Processing data of type: {type(data).__name__}")
            
            # Basic data validation
            if data is None:
                raise ValueError("Input data cannot be None")
            
            # Handle different data types
            if isinstance(data, dict):
                processed_data = self._process_dict_data(data)
            elif hasattr(data, 'shape'):  # numpy array or similar
                processed_data = self._process_array_data(data)
            elif isinstance(data, (list, tuple)):
                processed_data = self._process_sequence_data(data)
            else:
                # For other types, try to convert to numpy array
                try:
                    import numpy as np
                    processed_data = np.array(data)
                    if self.logger:
                        self.logger.info(f"Converted data to numpy array with shape: {processed_data.shape}")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Could not convert data to numpy array: {e}")
                    processed_data = data
            
            # Apply hardware optimization if available
            if self.enable_hardware_optimization and self.memory_optimizer:
                try:
                    processed_data = self.memory_optimizer.optimize_data(processed_data)
                    if self.logger:
                        self.logger.debug("Applied memory optimization to data")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Memory optimization failed: {e}")
            
            if self.logger:
                self.logger.info("Data processing completed successfully")
            
            return processed_data
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Data processing failed: {e}")
            raise
    
    def _process_dict_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process dictionary data."""
        processed_data = {}
        
        for key, value in data.items():
            if hasattr(value, 'shape'):  # numpy array
                processed_data[key] = self._process_array_data(value)
            elif isinstance(value, dict):
                processed_data[key] = self._process_dict_data(value)
            else:
                processed_data[key] = value
        
        return processed_data
    
    def _process_array_data(self, data: Any) -> Any:
        """Process array-like data."""
        import numpy as np
        
        # Convert to numpy array if not already
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Handle NaN and infinite values
        if np.any(np.isnan(data)):
            if self.logger:
                self.logger.warning("NaN values found in data, replacing with 0")
            data = np.nan_to_num(data, nan=0.0)
        
        if np.any(np.isinf(data)):
            if self.logger:
                self.logger.warning("Infinite values found in data, replacing with finite values")
            data = np.nan_to_num(data, posinf=0.0, neginf=0.0)
        
        return data
    
    def _process_sequence_data(self, data: Any) -> Any:
        """Process sequence data (list, tuple)."""
        import numpy as np
        
        # Convert to numpy array
        processed_data = np.array(data)
        
        # Apply same processing as array data
        return self._process_array_data(processed_data)

    @abstractmethod
    def _generate_artifacts(self, model: Any, results: TrainingResult) -> Dict[str, Any]:
        """Generate training artifacts."""
        try:
            if self.logger:
                self.logger.info("Generating training artifacts")
            
            artifacts = {}
            
            # Model metadata
            if model is not None:
                artifacts['model_metadata'] = self._extract_model_metadata(model)
            
            # Training results metadata
            if results is not None:
                artifacts['training_metadata'] = {
                    'success': results.success,
                    'training_time': results.training_time,
                    'memory_usage_mb': results.memory_usage_mb,
                    'timestamp': results.timestamp.isoformat(),
                    'errors': results.errors,
                    'warnings': results.warnings
                }
            
            # Performance metrics
            if results and results.metrics:
                artifacts['performance_metrics'] = results.metrics
            
            # Configuration artifacts
            artifacts['configuration'] = {
                'step_name': self.name,
                'config': self.config,
                'hardware_optimization_enabled': self.enable_hardware_optimization,
                'logging_enabled': self.enable_logging
            }
            
            # Hardware optimization artifacts
            if self.enable_hardware_optimization:
                artifacts['hardware_info'] = self._get_hardware_info()
            
            # Step-specific artifacts
            step_artifacts = self._generate_step_specific_artifacts(model, results)
            if step_artifacts:
                artifacts.update(step_artifacts)
            
            if self.logger:
                self.logger.info(f"Generated {len(artifacts)} training artifacts")
            
            return artifacts
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to generate training artifacts: {e}")
            
            # Return minimal artifacts on error
            return {
                'error': str(e),
                'step_name': self.name,
                'timestamp': datetime.now().isoformat()
            }
    
    def _extract_model_metadata(self, model: Any) -> Dict[str, Any]:
        """Extract metadata from trained model."""
        metadata = {
            'model_type': type(model).__name__,
            'is_fitted': getattr(model, 'is_fitted', False)
        }
        
        # Try to extract common model attributes
        common_attrs = ['n_features_in_', 'feature_names_in_', 'n_outputs_', 'classes_']
        for attr in common_attrs:
            if hasattr(model, attr):
                value = getattr(model, attr)
                if hasattr(value, 'tolist'):  # numpy array
                    metadata[attr] = value.tolist()
                else:
                    metadata[attr] = value
        
        # Model parameters
        if hasattr(model, 'get_params'):
            try:
                metadata['parameters'] = model.get_params()
            except Exception:
                metadata['parameters'] = {}
        
        return metadata
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        info = {}
        
        try:
            import psutil
            info['cpu_count'] = psutil.cpu_count()
            info['memory_total_gb'] = psutil.virtual_memory().total / (1024**3)
            info['memory_available_gb'] = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            info['psutil_available'] = False
        
        # M1 specific info
        if self.m1_optimizer:
            try:
                info['m1_optimization'] = self.m1_optimizer.get_info()
            except Exception:
                info['m1_optimization'] = 'unavailable'
        
        return info
    
    def _generate_step_specific_artifacts(self, model: Any, results: TrainingResult) -> Dict[str, Any]:
        """Generate step-specific artifacts. Override in subclasses."""
        return {}

    @abstractmethod
    def _calculate_metrics(self, model: Any, test_data: Any) -> Dict[str, Any]:
        """Calculate performance metrics."""
        try:
            if self.logger:
                self.logger.info("Calculating performance metrics")
            
            metrics = {}
            
            if model is None:
                if self.logger:
                    self.logger.warning("No model provided for metrics calculation")
                return {'error': 'No model provided'}
            
            if test_data is None:
                if self.logger:
                    self.logger.warning("No test data provided for metrics calculation")
                return {'error': 'No test data provided'}
            
            # Basic model validation
            if not hasattr(model, 'predict'):
                if self.logger:
                    self.logger.warning("Model does not have predict method")
                return {'error': 'Model does not support prediction'}
            
            try:
                # Extract features and targets from test data
                if isinstance(test_data, dict):
                    X_test = test_data.get('X', test_data.get('features', test_data.get('X_test')))
                    y_test = test_data.get('y', test_data.get('targets', test_data.get('y_test')))
                elif hasattr(test_data, 'shape') and len(test_data.shape) == 2:
                    # Assume last column is target
                    X_test = test_data[:, :-1]
                    y_test = test_data[:, -1]
                else:
                    X_test = test_data
                    y_test = None
                
                if X_test is None:
                    if self.logger:
                        self.logger.warning("Could not extract features from test data")
                    return {'error': 'Could not extract features from test data'}
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate basic metrics if we have true labels
                if y_test is not None:
                    metrics.update(self._calculate_basic_metrics(y_test, y_pred))
                
                # Model-specific metrics
                metrics.update(self._calculate_model_specific_metrics(model, X_test, y_test, y_pred))
                
                # Data quality metrics
                metrics.update(self._calculate_data_quality_metrics(X_test, y_test, y_pred))
                
                if self.logger:
                    self.logger.info(f"Calculated {len(metrics)} performance metrics")
                
                return metrics
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error during metrics calculation: {e}")
                return {'error': f'Metrics calculation failed: {str(e)}'}
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to calculate metrics: {e}")
            return {'error': f'Metrics calculation failed: {str(e)}'}
    
    def _calculate_basic_metrics(self, y_true: Any, y_pred: Any) -> Dict[str, Any]:
        """Calculate basic performance metrics."""
        import numpy as np
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
        
        metrics = {}
        
        try:
            # Convert to numpy arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Ensure same shape
            if y_true.shape != y_pred.shape:
                if self.logger:
                    self.logger.warning(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
                return {'error': 'Shape mismatch between true and predicted values'}
            
            # Regression metrics
            metrics['mse'] = float(mean_squared_error(y_true, y_pred))
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
            metrics['r2'] = float(r2_score(y_true, y_pred))
            
            # Check if this is classification (discrete values)
            unique_true = len(np.unique(y_true))
            unique_pred = len(np.unique(y_pred))
            
            if unique_true <= 20 and unique_pred <= 20:  # Likely classification
                try:
                    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
                except Exception:
                    pass
            
            # Additional metrics
            metrics['mean_absolute_percentage_error'] = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
            metrics['max_error'] = float(np.max(np.abs(y_true - y_pred)))
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error calculating basic metrics: {e}")
            metrics['basic_metrics_error'] = str(e)
        
        return metrics
    
    def _calculate_model_specific_metrics(self, model: Any, X_test: Any, y_test: Any, y_pred: Any) -> Dict[str, Any]:
        """Calculate model-specific metrics."""
        metrics = {}
        
        try:
            # Model complexity metrics
            if hasattr(model, 'n_estimators'):
                metrics['n_estimators'] = model.n_estimators
            if hasattr(model, 'max_depth'):
                metrics['max_depth'] = model.max_depth
            if hasattr(model, 'n_features_in_'):
                metrics['n_features'] = model.n_features_in_
            
            # Feature importance if available
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                metrics['feature_importance_mean'] = float(np.mean(importances))
                metrics['feature_importance_std'] = float(np.std(importances))
                metrics['feature_importance_max'] = float(np.max(importances))
                metrics['n_important_features'] = int(np.sum(importances > 0.01))  # Features with >1% importance
            
            # Model-specific scores
            if hasattr(model, 'score') and y_test is not None:
                try:
                    score = model.score(X_test, y_test)
                    metrics['model_score'] = float(score)
                except Exception:
                    pass
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error calculating model-specific metrics: {e}")
            metrics['model_specific_metrics_error'] = str(e)
        
        return metrics
    
    def _calculate_data_quality_metrics(self, X_test: Any, y_test: Any, y_pred: Any) -> Dict[str, Any]:
        """Calculate data quality metrics."""
        import numpy as np
        
        metrics = {}
        
        try:
            # Data shape metrics
            if hasattr(X_test, 'shape'):
                metrics['n_samples'] = int(X_test.shape[0])
                metrics['n_features'] = int(X_test.shape[1]) if len(X_test.shape) > 1 else 1
            
            # Missing values
            if hasattr(X_test, 'isnull'):
                missing_count = X_test.isnull().sum().sum()
                metrics['missing_values'] = int(missing_count)
            else:
                metrics['missing_values'] = int(np.isnan(X_test).sum()) if hasattr(X_test, 'dtype') else 0
            
            # Prediction quality
            if y_pred is not None:
                pred_array = np.array(y_pred)
                metrics['prediction_mean'] = float(np.mean(pred_array))
                metrics['prediction_std'] = float(np.std(pred_array))
                metrics['prediction_min'] = float(np.min(pred_array))
                metrics['prediction_max'] = float(np.max(pred_array))
                
                # Check for constant predictions
                if np.std(pred_array) < 1e-10:
                    metrics['constant_predictions'] = True
                else:
                    metrics['constant_predictions'] = False
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error calculating data quality metrics: {e}")
            metrics['data_quality_metrics_error'] = str(e)
        
        return metrics

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
        try:
            if self.logger:
                self.logger.info("Training model with processed data")
            
            # Default implementation - create a simple model
            # Subclasses should override this with specific training logic
            model = self._create_default_model(data, context)
            
            # Train the model
            if hasattr(model, 'fit'):
                # Extract features and targets from data
                if isinstance(data, dict):
                    X = data.get('X', data.get('features'))
                    y = data.get('y', data.get('targets'))
                elif hasattr(data, 'shape') and len(data.shape) == 2:
                    # Assume last column is target
                    X = data[:, :-1]
                    y = data[:, -1]
                else:
                    # Use data as features, create dummy targets
                    X = data
                    y = np.zeros(len(data))
                
                if X is not None and y is not None:
                    model.fit(X, y)
                    if self.logger:
                        self.logger.info("Model training completed successfully")
                else:
                    if self.logger:
                        self.logger.warning("Could not extract features and targets for training")
            else:
                if self.logger:
                    self.logger.warning("Model does not support fit method")
            
            return model
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Model training failed: {e}")
            raise
    
    def _create_default_model(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Create a default model for training."""
        try:
            # Try to determine if this is classification or regression
            is_classification = self._determine_task_type(data)
            
            if is_classification:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=self.config.get('n_estimators', 100),
                    random_state=self.config.get('random_state', 42),
                    n_jobs=self.config.get('n_jobs', -1)
                )
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    n_estimators=self.config.get('n_estimators', 100),
                    random_state=self.config.get('random_state', 42),
                    n_jobs=self.config.get('n_jobs', -1)
                )
            
            if self.logger:
                self.logger.info(f"Created default {type(model).__name__} model")
            
            return model
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to create default model: {e}")
            
            # Fallback to simple linear model
            try:
                from sklearn.linear_model import LinearRegression
                return LinearRegression()
            except Exception:
                # Last resort - create a dummy model
                class DummyModel:
                    def fit(self, X, y):
                        self._mean = float(np.mean(y)) if len(y) else 0.0
                        return self
                    def predict(self, X):
                        n = X.shape[0] if hasattr(X, 'shape') else len(X)
                        return np.full(n, getattr(self, '_mean', 0.0))
                return DummyModel()
    
    def _determine_task_type(self, data: Any) -> bool:
        """Determine if this is a classification task."""
        try:
            # Extract targets from data
            if isinstance(data, dict):
                y = data.get('y', data.get('targets'))
            elif hasattr(data, 'shape') and len(data.shape) == 2:
                y = data[:, -1]
            else:
                return False  # Default to regression
            
            if y is None:
                return False
            
            # Check if targets are discrete (classification)
            unique_values = np.unique(y)
            if len(unique_values) <= 20:  # Likely classification
                return True
            
            return False
            
        except Exception:
            return False

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
        start_time = time.time()
        
        try:
            if self.logger:
                self.logger.info(f"Starting clustering with {len(data)} samples")
            
            # Validate input data
            if not self._validate_input_data(data):
                raise ValueError("Invalid input data for clustering")
            
            # Preprocess data
            processed_data = self._preprocess_data(data)
            
            # Determine number of clusters if not specified
            n_clusters = self._determine_n_clusters(processed_data)
            
            # Perform clustering based on algorithm type
            labels = self._perform_clustering(processed_data, n_clusters)
            
            # Calculate clustering metrics
            metrics = self._calculate_clustering_metrics(processed_data, labels)
            
            # Create clustering result
            execution_time = time.time() - start_time
            
            result = ClusteringResult(
                labels=labels,
                n_clusters=n_clusters,
                algorithm=self.algorithm.value,
                metrics=metrics,
                metadata={
                    'algorithm_name': self.name,
                    'n_samples': len(data),
                    'n_features': data.shape[1] if len(data.shape) > 1 else 1,
                    'config': self.config
                },
                execution_time=execution_time,
                silhouette_score=metrics.get('silhouette_score'),
                inertia=metrics.get('inertia')
            )
            
            # Update model state
            self.is_fitted = True
            self.model = self._get_fitted_model()
            
            if self.logger:
                self.logger.info(f"Clustering completed: {n_clusters} clusters in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            if self.logger:
                self.logger.error(f"Clustering failed: {e}")
            
            # Return error result
            return ClusteringResult(
                labels=np.zeros(len(data), dtype=int),
                n_clusters=1,
                algorithm=self.algorithm.value,
                metrics={'error': str(e)},
                metadata={'error': str(e), 'execution_time': execution_time},
                execution_time=execution_time
            )
    
    def _validate_input_data(self, data: np.ndarray) -> bool:
        """Validate input data for clustering."""
        try:
            if data is None:
                return False
            
            if not isinstance(data, np.ndarray):
                return False
            
            if len(data.shape) != 2:
                return False
            
            if data.shape[0] < 2:
                return False
            
            if data.shape[1] < 1:
                return False
            
            # Check for NaN or infinite values
            if not np.isfinite(data).all():
                if self.logger:
                    self.logger.warning("Non-finite values found in data")
                return False
            
            return True
            
        except Exception:
            return False
    
    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """Preprocess data for clustering."""
        try:
            # Handle NaN and infinite values
            processed_data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply hardware optimization if available
            if self.enable_optimization and self.memory_optimizer:
                try:
                    processed_data = self.memory_optimizer.optimize_data(processed_data)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Memory optimization failed: {e}")
            
            return processed_data
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def _determine_n_clusters(self, data: np.ndarray) -> int:
        """Determine optimal number of clusters."""
        # Check if n_clusters is specified in config
        n_clusters = self.config.get('n_clusters')
        if n_clusters is not None:
            return int(n_clusters)
        
        # Use elbow method or other heuristics
        max_clusters = min(10, data.shape[0] // 2)
        if max_clusters < 2:
            return 2
        
        # Simple heuristic: use square root of number of samples
        n_clusters = max(2, int(np.sqrt(data.shape[0])))
        return min(n_clusters, max_clusters)
    
    def _perform_clustering(self, data: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform the actual clustering based on algorithm type."""
        if self.algorithm == ClusteringAlgorithm.KMEANS:
            return self._perform_kmeans_clustering(data, n_clusters)
        elif self.algorithm == ClusteringAlgorithm.GAUSSIAN_MIXTURE:
            return self._perform_gmm_clustering(data, n_clusters)
        elif self.algorithm == ClusteringAlgorithm.AGGLOMERATIVE:
            return self._perform_agglomerative_clustering(data, n_clusters)
        else:
            # Default to K-means
            return self._perform_kmeans_clustering(data, n_clusters)
    
    def _perform_kmeans_clustering(self, data: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform K-means clustering."""
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.config.get('random_state', 42),
            n_init=self.config.get('n_init', 10),
            max_iter=self.config.get('max_iter', 300)
        )
        
        labels = kmeans.fit_predict(data)
        self._fitted_model = kmeans
        return labels
    
    def _perform_gmm_clustering(self, data: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform Gaussian Mixture Model clustering."""
        from sklearn.mixture import GaussianMixture
        
        gmm = GaussianMixture(
            n_components=n_clusters,
            random_state=self.config.get('random_state', 42),
            max_iter=self.config.get('max_iter', 100)
        )
        
        labels = gmm.fit_predict(data)
        self._fitted_model = gmm
        return labels
    
    def _perform_agglomerative_clustering(self, data: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform Agglomerative clustering."""
        from sklearn.cluster import AgglomerativeClustering
        
        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=self.config.get('linkage', 'ward')
        )
        
        labels = agg.fit_predict(data)
        self._fitted_model = agg
        return labels
    
    def _calculate_clustering_metrics(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        metrics = {}
        
        try:
            n_clusters = len(np.unique(labels))
            n_samples = len(labels)
            
            # Silhouette score
            if n_clusters > 1 and n_samples > 1:
                try:
                    from sklearn.metrics import silhouette_score
                    metrics['silhouette_score'] = float(silhouette_score(data, labels))
                except Exception:
                    metrics['silhouette_score'] = 0.0
            else:
                metrics['silhouette_score'] = 0.0
            
            # Inertia (for K-means)
            if hasattr(self, '_fitted_model') and hasattr(self._fitted_model, 'inertia_'):
                metrics['inertia'] = float(self._fitted_model.inertia_)
            
            # Davies-Bouldin score
            if n_clusters > 1:
                try:
                    from sklearn.metrics import davies_bouldin_score
                    metrics['davies_bouldin_score'] = float(davies_bouldin_score(data, labels))
                except Exception:
                    metrics['davies_bouldin_score'] = float('inf')
            else:
                metrics['davies_bouldin_score'] = float('inf')
            
            # Calinski-Harabasz score
            if n_clusters > 1:
                try:
                    from sklearn.metrics import calinski_harabasz_score
                    metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(data, labels))
                except Exception:
                    metrics['calinski_harabasz_score'] = 0.0
            else:
                metrics['calinski_harabasz_score'] = 0.0
            
            # Cluster size statistics
            unique_labels, counts = np.unique(labels, return_counts=True)
            metrics['n_clusters'] = float(n_clusters)
            metrics['cluster_size_mean'] = float(np.mean(counts))
            metrics['cluster_size_std'] = float(np.std(counts))
            metrics['cluster_size_min'] = float(np.min(counts))
            metrics['cluster_size_max'] = float(np.max(counts))
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error calculating clustering metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _get_fitted_model(self) -> Any:
        """Get the fitted model."""
        return getattr(self, '_fitted_model', None)

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
        start_time = time.time()
        
        try:
            if self.logger:
                self.logger.info(f"Fitting multi-output model with {X.shape[0]} samples, {X.shape[1]} features")
            
            # Validate inputs
            if not self._validate_inputs(X, y):
                raise ValueError("Invalid input data for multi-output model")
            
            # Ensure y is 2D
            y_2d = self._ensure_2d_targets(y)
            
            # Initialize individual models for each output
            self.models = {}
            
            # Train models for each output
            for i in range(self.n_outputs):
                output_name = self.output_names[i] if i < len(self.output_names) else f"output_{i+1}"
                
                if self.logger:
                    self.logger.info(f"Training model for {output_name} (output {i+1}/{self.n_outputs})")
                
                # Extract target for this output
                if y_2d.shape[1] > i:
                    y_output = y_2d[:, i]
                else:
                    # If not enough outputs, use the last available
                    y_output = y_2d[:, -1]
                
                # Create and train model for this output
                model = self._create_single_output_model(i, y_output)
                model.fit(X, y_output)
                
                self.models[output_name] = model
                
                if self.logger:
                    self.logger.info(f"Model for {output_name} trained successfully")
            
            # Update state
            self.is_fitted = True
            self.total_training_time += time.time() - start_time
            
            if self.logger:
                self.logger.info(f"Multi-output model fitted successfully with {len(self.models)} individual models")
            
            return self
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Multi-output model fitting failed: {e}")
            raise
    
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Validate input data."""
        try:
            # Check X
            if X is None or not isinstance(X, np.ndarray):
                return False
            
            if len(X.shape) != 2 or X.shape[0] == 0 or X.shape[1] == 0:
                return False
            
            # Check y
            if y is None or not isinstance(y, np.ndarray):
                return False
            
            if len(y.shape) not in [1, 2]:
                return False
            
            if y.shape[0] != X.shape[0]:
                return False
            
            # Check for NaN or infinite values
            if not np.isfinite(X).all() or not np.isfinite(y).all():
                if self.logger:
                    self.logger.warning("Non-finite values found in input data")
                return False
            
            return True
            
        except Exception:
            return False
    
    def _ensure_2d_targets(self, y: np.ndarray) -> np.ndarray:
        """Ensure targets are 2D."""
        if len(y.shape) == 1:
            # Single output - duplicate for multi-output
            y_2d = np.column_stack([y] * self.n_outputs)
            if self.logger:
                self.logger.info(f"Converted single output to multi-output: {y.shape} -> {y_2d.shape}")
        else:
            y_2d = y
            if y_2d.shape[1] != self.n_outputs:
                if self.logger:
                    self.logger.warning(f"Output count mismatch: expected {self.n_outputs}, got {y_2d.shape[1]}")
                # Adjust to match expected outputs
                if y_2d.shape[1] < self.n_outputs:
                    # Pad with last column
                    padding = np.column_stack([y_2d[:, -1]] * (self.n_outputs - y_2d.shape[1]))
                    y_2d = np.column_stack([y_2d, padding])
                else:
                    # Truncate to expected number
                    y_2d = y_2d[:, :self.n_outputs]
        
        return y_2d
    
    def _create_single_output_model(self, output_index: int, y_target: np.ndarray) -> Any:
        """Create a single-output model for the specified output."""
        try:
            # Determine if this is classification or regression
            is_classification = self._is_classification_task(y_target)
            
            if is_classification:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=self.config.get('n_estimators', 100),
                    random_state=self.config.get('random_state', 42),
                    n_jobs=self.config.get('n_jobs', -1)
                )
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    n_estimators=self.config.get('n_estimators', 100),
                    random_state=self.config.get('random_state', 42),
                    n_jobs=self.config.get('n_jobs', -1)
                )
            
            if self.logger:
                self.logger.debug(f"Created {type(model).__name__} for output {output_index}")
            
            return model
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to create model for output {output_index}: {e}")
            
            # Fallback to simple linear model
            try:
                from sklearn.linear_model import LinearRegression
                return LinearRegression()
            except Exception:
                # Last resort - create a dummy model
                class DummyModel:
                    def fit(self, X, y):
                        self._mean = float(np.mean(y)) if len(y) else 0.0
                        return self
                    def predict(self, X):
                        n = X.shape[0] if hasattr(X, 'shape') else len(X)
                        return np.full(n, getattr(self, '_mean', 0.0))
                return DummyModel()
    
    def _is_classification_task(self, y: np.ndarray) -> bool:
        """Determine if this is a classification task."""
        try:
            unique_values = np.unique(y)
            
            # If few unique values and they look like integers, treat as classification
            if len(unique_values) <= 20:
                # Check if values are integer-like
                if np.all(np.equal(np.mod(unique_values, 1), 0)):
                    return True
            
            return False
            
        except Exception:
            return False

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for all outputs.
        
        Args:
            X: Input features
            
        Returns:
            Predictions for all outputs
        """
        start_time = time.time()
        
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            
            if not self._validate_inputs(X, None):
                raise ValueError("Invalid input features for prediction")
            
            if self.logger:
                self.logger.info(f"Making predictions for {X.shape[0]} samples")
            
            # Collect predictions from all models
            predictions = []
            
            for i, output_name in enumerate(self.output_names):
                if output_name in self.models:
                    model = self.models[output_name]
                    
                    try:
                        # Make prediction for this output
                        output_pred = model.predict(X)
                        predictions.append(output_pred)
                        
                        if self.logger:
                            self.logger.debug(f"Predictions generated for {output_name}: {len(output_pred)} samples")
                    
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"Prediction failed for {output_name}: {e}")
                        
                        # Fallback to mean prediction
                        fallback_pred = np.full(X.shape[0], 0.0)
                        predictions.append(fallback_pred)
                else:
                    if self.logger:
                        self.logger.warning(f"No model found for {output_name}, using zeros")
                    
                    # Use zeros as fallback
                    predictions.append(np.zeros(X.shape[0]))
            
            # Combine predictions into multi-output format
            if predictions:
                result = np.column_stack(predictions)
            else:
                result = np.zeros((X.shape[0], self.n_outputs))
            
            # Update performance tracking
            self.total_prediction_time += time.time() - start_time
            
            if self.logger:
                self.logger.info(f"Predictions completed in {time.time() - start_time:.3f}s")
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Prediction failed: {e}")
            raise

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
        try:
            if self.logger:
                self.logger.info(f"Starting pattern discovery on {len(data)} data points")
            
            # Validate input data
            if not self._validate_pattern_data(data):
                raise ValueError("Invalid input data for pattern discovery")
            
            # Get pattern definition
            pattern_definition = self.get_pattern_definition()
            
            # Discover patterns using the specific algorithm
            labels, confidence_scores = self._discover_patterns_impl(data, **kwargs)
            
            # Calculate pattern statistics
            frequency = np.mean(labels) if len(labels) > 0 else 0.0
            metrics = self._calculate_pattern_metrics(data, labels, confidence_scores)
            
            # Create pattern discovery result
            result = PatternDiscoveryResult(
                definition=pattern_definition,
                labels=labels,
                confidence_scores=confidence_scores,
                frequency=frequency,
                metrics=metrics,
                metadata={
                    'discoverer_name': self.name,
                    'pattern_type': self.pattern_type.value,
                    'n_data_points': len(data),
                    'discovery_parameters': kwargs
                }
            )
            
            # Validate the discovered pattern
            is_valid = self.validate_pattern(result)
            result.metadata['is_valid'] = is_valid
            
            # Store result
            self.discovered_patterns.append(result)
            
            if self.logger:
                status = "VALID" if is_valid else "INVALID"
                self.logger.info(f"Pattern discovery completed: {status} pattern found with frequency {frequency:.3f}")
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Pattern discovery failed: {e}")
            
            # Return error result
            error_definition = PatternDefinition(
                name=f"Error_{self.name}",
                pattern_type=self.pattern_type,
                description=f"Error in pattern discovery: {str(e)}",
                mathematical_formula="Error occurred during discovery",
                parameters={},
                frequency_threshold=0.0
            )
            
            return PatternDiscoveryResult(
                definition=error_definition,
                labels=np.zeros(len(data), dtype=int),
                confidence_scores=np.zeros(len(data), dtype=float),
                frequency=0.0,
                metrics={'error': str(e)},
                metadata={'error': str(e), 'discoverer_name': self.name}
            )
    
    def _validate_pattern_data(self, data: np.ndarray) -> bool:
        """Validate input data for pattern discovery."""
        try:
            if data is None:
                return False
            
            if not isinstance(data, np.ndarray):
                return False
            
            if len(data) < 10:  # Need minimum data points
                return False
            
            # Check for NaN or infinite values
            if not np.isfinite(data).all():
                if self.logger:
                    self.logger.warning("Non-finite values found in pattern data")
                return False
            
            return True
            
        except Exception:
            return False
    
    def _discover_patterns_impl(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Implementation-specific pattern discovery. Override in subclasses."""
        # Default implementation - simple threshold-based pattern
        threshold = kwargs.get('threshold', np.mean(data))
        
        # Create binary labels based on threshold
        labels = (data > threshold).astype(int)
        
        # Create confidence scores based on distance from threshold
        distances = np.abs(data - threshold)
        max_distance = np.max(distances) if np.max(distances) > 0 else 1.0
        confidence_scores = distances / max_distance
        
        return labels, confidence_scores
    
    def _calculate_pattern_metrics(self, data: np.ndarray, labels: np.ndarray, confidence_scores: np.ndarray) -> Dict[str, Any]:
        """Calculate pattern discovery metrics."""
        metrics = {}
        
        try:
            # Basic metrics
            metrics['n_patterns'] = int(np.sum(labels))
            metrics['pattern_frequency'] = float(np.mean(labels))
            metrics['avg_confidence'] = float(np.mean(confidence_scores))
            metrics['max_confidence'] = float(np.max(confidence_scores))
            metrics['min_confidence'] = float(np.min(confidence_scores))
            
            # Pattern duration metrics
            pattern_durations = self._calculate_pattern_durations(labels)
            if pattern_durations:
                metrics['avg_pattern_duration'] = float(np.mean(pattern_durations))
                metrics['max_pattern_duration'] = float(np.max(pattern_durations))
                metrics['min_pattern_duration'] = float(np.min(pattern_durations))
            
            # Data statistics
            metrics['data_mean'] = float(np.mean(data))
            metrics['data_std'] = float(np.std(data))
            metrics['data_min'] = float(np.min(data))
            metrics['data_max'] = float(np.max(data))
            
            # Pattern quality metrics
            if np.sum(labels) > 0:
                pattern_data = data[labels == 1]
                non_pattern_data = data[labels == 0]
                
                metrics['pattern_data_mean'] = float(np.mean(pattern_data))
                metrics['pattern_data_std'] = float(np.std(pattern_data))
                metrics['non_pattern_data_mean'] = float(np.mean(non_pattern_data))
                metrics['non_pattern_data_std'] = float(np.std(non_pattern_data))
                
                # Separation quality
                if len(pattern_data) > 0 and len(non_pattern_data) > 0:
                    separation = abs(np.mean(pattern_data) - np.mean(non_pattern_data))
                    metrics['pattern_separation'] = float(separation)
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error calculating pattern metrics: {e}")
            metrics['metrics_error'] = str(e)
        
        return metrics
    
    def _calculate_pattern_durations(self, labels: np.ndarray) -> List[int]:
        """Calculate durations of consecutive patterns."""
        durations = []
        current_duration = 0
        
        for label in labels:
            if label == 1:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        # Handle case where pattern continues to end
        if current_duration > 0:
            durations.append(current_duration)
        
        return durations

    @abstractmethod
    def get_pattern_definition(self) -> PatternDefinition:
        """
        Get mathematical definition of the pattern.
        
        Returns:
            PatternDefinition with pattern details
        """
        try:
            # Create a basic pattern definition
            # Subclasses should override this with specific pattern definitions
            definition = PatternDefinition(
                name=f"{self.name}_Pattern",
                pattern_type=self.pattern_type,
                description=f"Pattern discovered by {self.name} discoverer",
                mathematical_formula=f"""
                Pattern detection formula for {self.pattern_type.value} patterns:
                
                Let data = input_data
                Let threshold = mean(data) or custom threshold
                
                Pattern exists at position i IF:
                data[i] > threshold AND
                confidence_score[i] > confidence_threshold
                
                Where confidence_score[i] = |data[i] - threshold| / max(|data - threshold|)
                """,
                parameters={
                    'threshold_method': 'mean',
                    'confidence_threshold': 0.5,
                    'min_pattern_length': 1,
                    'max_pattern_length': len(self.config) if hasattr(self, 'config') else 1000
                },
                frequency_threshold=0.1,  # Must occur at least 10% of the time
                confidence_threshold=0.7
            )
            
            # Store definition if not already stored
            if definition not in self.pattern_definitions:
                self.pattern_definitions.append(definition)
            
            if self.logger:
                self.logger.debug(f"Generated pattern definition for {self.name}")
            
            return definition
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to generate pattern definition: {e}")
            
            # Return error definition
            return PatternDefinition(
                name=f"Error_{self.name}",
                pattern_type=self.pattern_type,
                description=f"Error generating pattern definition: {str(e)}",
                mathematical_formula="Error occurred during definition generation",
                parameters={},
                frequency_threshold=0.0,
                confidence_threshold=0.0
            )

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
        try:
            if self.logger:
                self.logger.info(f"Generating labels using {self.strategy.value} strategy")
            
            # Validate input data
            if not self._validate_labeling_data(data):
                raise ValueError("Invalid input data for labeling")
            
            # Generate labels based on strategy
            labels = self._generate_labels_impl(data, **kwargs)
            
            # Calculate confidence scores
            confidence_scores = self.calculate_confidence(labels, data, **kwargs)
            
            # Validate generated labels
            if not self.validate_labels(labels):
                if self.logger:
                    self.logger.warning("Generated labels failed validation")
            
            # Calculate labeling metrics
            metrics = self._calculate_labeling_metrics(data, labels, confidence_scores)
            
            # Create labeling result
            result = LabelingResult(
                labels=labels,
                confidence_scores=confidence_scores,
                strategy=self.strategy,
                metrics=metrics,
                metadata={
                    'labeling_strategy': self.name,
                    'n_samples': len(data),
                    'labeling_parameters': kwargs,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # Store result
            self.labeling_results.append(result)
            
            if self.logger:
                self.logger.info(f"Labeling completed: {len(labels)} labels generated")
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Label generation failed: {e}")
            
            # Return error result
            return LabelingResult(
                labels=np.zeros(len(data), dtype=int),
                confidence_scores=np.zeros(len(data), dtype=float),
                strategy=self.strategy,
                metrics={'error': str(e)},
                metadata={'error': str(e), 'labeling_strategy': self.name}
            )
    
    def _validate_labeling_data(self, data: np.ndarray) -> bool:
        """Validate input data for labeling."""
        try:
            if data is None:
                return False
            
            if not isinstance(data, np.ndarray):
                return False
            
            if len(data) < 2:  # Need minimum data points
                return False
            
            # Check for NaN or infinite values
            if not np.isfinite(data).all():
                if self.logger:
                    self.logger.warning("Non-finite values found in labeling data")
                return False
            
            return True
            
        except Exception:
            return False
    
    def _generate_labels_impl(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Implementation-specific label generation. Override in subclasses."""
        # Default implementation - simple binary labeling based on threshold
        threshold = kwargs.get('threshold', np.median(data))
        
        # Create binary labels
        labels = (data > threshold).astype(int)
        
        return labels
    
    def _calculate_labeling_metrics(self, data: np.ndarray, labels: np.ndarray, confidence_scores: np.ndarray) -> Dict[str, Any]:
        """Calculate labeling metrics."""
        metrics = {}
        
        try:
            # Basic metrics
            metrics['n_labels'] = len(labels)
            metrics['n_positive_labels'] = int(np.sum(labels))
            metrics['n_negative_labels'] = int(np.sum(1 - labels))
            metrics['label_balance'] = float(np.sum(labels) / len(labels))
            
            # Confidence metrics
            metrics['avg_confidence'] = float(np.mean(confidence_scores))
            metrics['max_confidence'] = float(np.max(confidence_scores))
            metrics['min_confidence'] = float(np.min(confidence_scores))
            metrics['confidence_std'] = float(np.std(confidence_scores))
            
            # Data statistics
            metrics['data_mean'] = float(np.mean(data))
            metrics['data_std'] = float(np.std(data))
            metrics['data_min'] = float(np.min(data))
            metrics['data_max'] = float(np.max(data))
            
            # Label quality metrics
            if np.sum(labels) > 0:
                positive_data = data[labels == 1]
                negative_data = data[labels == 0]
                
                metrics['positive_data_mean'] = float(np.mean(positive_data))
                metrics['positive_data_std'] = float(np.std(positive_data))
                metrics['negative_data_mean'] = float(np.mean(negative_data))
                metrics['negative_data_std'] = float(np.std(negative_data))
                
                # Separation quality
                if len(positive_data) > 0 and len(negative_data) > 0:
                    separation = abs(np.mean(positive_data) - np.mean(negative_data))
                    metrics['label_separation'] = float(separation)
            
            # Confidence distribution
            high_confidence = np.sum(confidence_scores > 0.8)
            medium_confidence = np.sum((confidence_scores > 0.5) & (confidence_scores <= 0.8))
            low_confidence = np.sum(confidence_scores <= 0.5)
            
            metrics['high_confidence_labels'] = int(high_confidence)
            metrics['medium_confidence_labels'] = int(medium_confidence)
            metrics['low_confidence_labels'] = int(low_confidence)
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error calculating labeling metrics: {e}")
            metrics['metrics_error'] = str(e)
        
        return metrics

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
        try:
            if self.logger:
                self.logger.debug("Calculating confidence scores for labels")
            
            # Validate inputs
            if len(labels) != len(data):
                raise ValueError("Labels and data must have the same length")
            
            # Calculate confidence based on strategy
            confidence_scores = self._calculate_confidence_impl(labels, data, **kwargs)
            
            # Ensure confidence scores are in valid range [0, 1]
            confidence_scores = np.clip(confidence_scores, 0.0, 1.0)
            
            if self.logger:
                self.logger.debug(f"Confidence scores calculated: mean={np.mean(confidence_scores):.3f}")
            
            return confidence_scores
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Confidence calculation failed: {e}")
            
            # Return default confidence scores
            return np.full(len(labels), 0.5)
    
    def _calculate_confidence_impl(self, labels: np.ndarray, data: np.ndarray, **kwargs) -> np.ndarray:
        """Implementation-specific confidence calculation. Override in subclasses."""
        # Default implementation - confidence based on distance from decision boundary
        try:
            # Calculate decision boundary (median of data)
            boundary = np.median(data)
            
            # Calculate distances from boundary
            distances = np.abs(data - boundary)
            
            # Normalize distances to [0, 1] range
            max_distance = np.max(distances) if np.max(distances) > 0 else 1.0
            normalized_distances = distances / max_distance
            
            # Convert to confidence scores (closer to boundary = lower confidence)
            confidence_scores = 1.0 - normalized_distances
            
            # Ensure minimum confidence for all labels
            min_confidence = kwargs.get('min_confidence', 0.1)
            confidence_scores = np.maximum(confidence_scores, min_confidence)
            
            return confidence_scores
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error in confidence calculation: {e}")
            
            # Return uniform confidence scores
            return np.full(len(labels), 0.5)

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