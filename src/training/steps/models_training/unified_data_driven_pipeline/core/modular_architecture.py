"""
ModularComponent Architecture for Models Training

This module provides the core ModularComponent architecture specifically designed
for machine learning model training workflows. It includes comprehensive functionality
for configuration management, state management, performance monitoring, and lifecycle
management optimized for ML training scenarios.

Key Features:
- ML-specific state management (model weights, training progress, validation metrics)
- Training-specific configuration management
- Performance monitoring for training metrics
- Model checkpointing and serialization
- Comprehensive error handling and logging
- Memory management for large datasets
- Training progress tracking
"""

import logging
import time
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import traceback
import gc

# Optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Core dependencies for ML training
try:
    import numpy as np
    import pandas as pd
    import torch
    import sklearn
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    ML_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some ML dependencies not available: {e}")
    ML_DEPENDENCIES_AVAILABLE = False

# Optional dependencies
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class ValidationLevel(Enum):
    """Validation levels for input data."""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"


class MetricType(Enum):
    """Types of performance metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    LOSS = "loss"
    CUSTOM = "custom"


class MetricLevel(Enum):
    """Levels of metric importance."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    PROCESSING = "processing"
    MEMORY = "memory"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    TRAINING = "training"
    MODEL = "model"
    DATA = "data"


@dataclass
class ErrorInfo:
    """Information about an error."""
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    timestamp: float = field(default_factory=time.time)
    component: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """Performance metric information."""
    name: str
    value: float
    metric_type: MetricType
    level: MetricLevel
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    errors: List[ErrorInfo] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_level: ValidationLevel = ValidationLevel.BASIC


class ModularComponent(ABC):
    """
    Abstract base class for modular components in the models training pipeline.
    
    This class provides comprehensive functionality for creating modular, reusable
    components specifically optimized for machine learning training workflows.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the ModularComponent.
        
        Args:
            name: Unique name for the component
            config: Configuration dictionary
            logger: Logger instance (optional)
        """
        self.name = name
        self.config = config or {}
        self.logger = logger or logging.getLogger(f"{__name__}.{name}")
        
        # State management
        self._state: Dict[str, Any] = {}
        self._initialized = False
        
        # Performance tracking
        self._performance_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_time': 0.0,
            'last_operation_time': 0.0,
            'training_epochs': 0,
            'validation_accuracy': 0.0,
            'model_convergence': False
        }
        
        # ML-specific state
        self._ml_state = {
            'model_weights': None,
            'training_progress': {},
            'validation_metrics': {},
            'best_model_state': None,
            'training_history': [],
            'experiment_id': None
        }
        
        # Configuration validation
        self._config_validators: Dict[str, Callable] = {}
        self._state_change_callbacks: List[Callable] = []
        self._config_change_callbacks: List[Callable] = []
        
        # Memory management
        self._memory_limit_mb = self.get_config('memory_limit_mb', 2048)
        self._slow_operation_threshold = self.get_config('slow_operation_threshold', 5.0)
        
        self.logger.info(f"Initialized ModularComponent: {name}")
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    def _initialize_resources(self) -> bool:
        """Initialize component-specific resources."""
        pass
    
    @abstractmethod
    def _cleanup_resources(self) -> None:
        """Cleanup component-specific resources."""
        pass
    
    @abstractmethod
    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data with component logic."""
        pass
    
    @abstractmethod
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for this component."""
        pass
    
    @abstractmethod
    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        """Validate data with component-specific rules."""
        pass
    
    # Core lifecycle methods
    
    def initialize(self) -> bool:
        """
        Initialize the component and its resources.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info(f"Initializing component: {self.name}")
            
            # Validate configuration
            if not self.validate_config():
                self.logger.error("Configuration validation failed")
                return False
            
            # Initialize resources
            if not self._initialize_resources():
                self.logger.error("Resource initialization failed")
                return False
            
            self._initialized = True
            self.set_state('initialized_at', time.time())
            self.logger.info(f"Component {self.name} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def process(self, data: Any, **kwargs) -> Any:
        """
        Process input data with comprehensive error handling.
        
        Args:
            data: Input data to process
            **kwargs: Additional keyword arguments
            
        Returns:
            Processed data
            
        Raises:
            RuntimeError: If component not initialized or processing fails
        """
        if not self._initialized:
            raise RuntimeError(f"Component {self.name} not initialized")
        
        start_time = time.time()
        
        try:
            # Validate input
            validation_result = self.validate_input(data)
            if not validation_result.is_valid:
                error_msg = f"Input validation failed: {[e.message for e in validation_result.errors]}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Check if component can process data
            if not self.can_process(data):
                raise RuntimeError(f"Component {self.name} cannot process the provided data")
            
            # Process data
            result = self._process_data(data, **kwargs)
            
            # Update performance stats
            processing_time = time.time() - start_time
            self._update_performance_stats(True, processing_time)
            
            self.logger.info(f"Data processed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_performance_stats(False, processing_time)
            self.logger.error(f"Processing failed: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def validate_input(self, data: Any) -> ValidationResult:
        """
        Comprehensive input validation with detailed results.
        
        Args:
            data: Data to validate
            
        Returns:
            ValidationResult with detailed validation information
        """
        errors = []
        warnings = []
        metadata = {}
        
        try:
            # Basic validation
            if data is None:
                errors.append(ErrorInfo(
                    message="Input data is None",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.VALIDATION
                ))
                return ValidationResult(False, errors, warnings, metadata)
            
            # Get validation rules
            rules = self._get_validation_rules()
            
            # Type validation
            if 'data_types' in rules:
                data_type = type(data).__name__
                if data_type not in rules['data_types']:
                    errors.append(ErrorInfo(
                        message=f"Invalid data type: {data_type}. Expected: {rules['data_types']}",
                        severity=ErrorSeverity.MEDIUM,
                        category=ErrorCategory.VALIDATION
                    ))
            
            # Size validation
            if hasattr(data, '__len__'):
                data_size = len(data)
                metadata['data_size'] = data_size
                
                if 'min_size' in rules and data_size < rules['min_size']:
                    warnings.append(f"Data size {data_size} is below minimum {rules['min_size']}")
                
                if 'max_size' in rules and data_size > rules['max_size']:
                    errors.append(ErrorInfo(
                        message=f"Data size {data_size} exceeds maximum {rules['max_size']}",
                        severity=ErrorSeverity.MEDIUM,
                        category=ErrorCategory.VALIDATION
                    ))
            
            # ML-specific validation for DataFrames
            if isinstance(data, pd.DataFrame):
                metadata['data_shape'] = data.shape
                metadata['data_columns'] = list(data.columns)
                
                # Check required columns
                if 'required_columns' in rules:
                    missing_columns = [col for col in rules['required_columns'] if col not in data.columns]
                    if missing_columns:
                        errors.append(ErrorInfo(
                            message=f"Missing required columns: {missing_columns}",
                            severity=ErrorSeverity.HIGH,
                            category=ErrorCategory.VALIDATION
                        ))
            
            # Component-specific validation
            component_validation = self._validate_component_specific(data)
            if 'errors' in component_validation:
                for error in component_validation['errors']:
                    errors.append(ErrorInfo(
                        message=error,
                        severity=ErrorSeverity.MEDIUM,
                        category=ErrorCategory.VALIDATION
                    ))
            
            if 'warnings' in component_validation:
                warnings.extend(component_validation['warnings'])
            
            if 'metadata' in component_validation:
                metadata.update(component_validation['metadata'])
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                metadata=metadata
            )
            
        except Exception as e:
            errors.append(ErrorInfo(
                message=f"Validation error: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.VALIDATION
            ))
            return ValidationResult(False, errors, warnings, metadata)
    
    def cleanup(self) -> None:
        """Cleanup resources and reset component state."""
        try:
            self.logger.info(f"Cleaning up component: {self.name}")
            
            # Cleanup resources
            self._cleanup_resources()
            
            # Clear state
            self.clear_state()
            
            # Reset performance stats
            self.reset_stats()
            
            # Reset initialization flag
            self._initialized = False
            
            self.logger.info(f"Component {self.name} cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            self.logger.error(traceback.format_exc())
    
    # Configuration management
    
    def get_config(self, key: str = None, default: Any = None) -> Any:
        """
        Get configuration value(s) with support for nested keys.
        
        Args:
            key: Configuration key (supports nested keys like 'parent.child')
            default: Default value if key not found
            
        Returns:
            Configuration value or entire config if no key provided
        """
        if key is None:
            return self.config.copy()
        
        # Support nested keys
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update component configuration with validation.
        
        Args:
            config: New configuration values
        """
        try:
            # Validate configuration keys
            for key in config.keys():
                if not isinstance(key, str):
                    raise ValueError(f"Configuration key must be string, got {type(key)}")
            
            # Update configuration
            self.config.update(config)
            
            # Trigger callbacks
            for callback in self._config_change_callbacks:
                try:
                    callback(config)
                except Exception as e:
                    self.logger.warning(f"Config change callback failed: {e}")
            
            self.logger.info(f"Configuration updated for {self.name}")
            
        except Exception as e:
            self.logger.error(f"Configuration update failed: {e}")
            raise
    
    def validate_config(self) -> bool:
        """
        Comprehensive configuration validation.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check required configuration parameters
            required_config = self.get_required_config()
            for key in required_config:
                if key not in self.config:
                    self.logger.error(f"Missing required configuration: {key}")
                    return False
            
            # Validate configuration values
            for key, value in self.config.items():
                if key in self._config_validators:
                    if not self._config_validators[key](value):
                        self.logger.error(f"Invalid configuration value for {key}: {value}")
                        return False
            
            # Component-specific validation
            if hasattr(self, '_validate_component_config'):
                if not self._validate_component_config():
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    # State management
    
    def set_state(self, key: str, value: Any) -> None:
        """
        Set component state with change tracking.
        
        Args:
            key: State key
            value: State value
        """
        if not isinstance(key, str):
            raise ValueError("State key must be string")
        
        previous_value = self._state.get(key)
        self._state[key] = value
        
        # Trigger callbacks
        for callback in self._state_change_callbacks:
            try:
                callback(key, value, previous_value)
            except Exception as e:
                self.logger.warning(f"State change callback failed: {e}")
        
        self.logger.debug(f"State updated: {key} = {type(value).__name__}")
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get component state with default fallback.
        
        Args:
            key: State key
            default: Default value if key not found
            
        Returns:
            State value or default
        """
        return self._state.get(key, default)
    
    def clear_state(self) -> None:
        """Clear all component state."""
        cleared_keys = list(self._state.keys())
        self._state.clear()
        self.logger.info(f"State cleared: {cleared_keys}")
    
    def get_all_state(self) -> Dict[str, Any]:
        """Get all component state."""
        return self._state.copy()
    
    def has_state(self, key: str) -> bool:
        """Check if state key exists."""
        return key in self._state
    
    def remove_state(self, key: str) -> Any:
        """Remove state key and return its value."""
        return self._state.pop(key, None)
    
    # ML-specific state management
    
    def set_ml_state(self, key: str, value: Any) -> None:
        """Set ML-specific state."""
        self._ml_state[key] = value
        self.logger.debug(f"ML state updated: {key} = {type(value).__name__}")
    
    def get_ml_state(self, key: str, default: Any = None) -> Any:
        """Get ML-specific state."""
        return self._ml_state.get(key, default)
    
    def update_training_progress(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Update training progress."""
        self._ml_state['training_progress'][epoch] = metrics
        self.set_state('current_epoch', epoch)
        self.set_state('current_metrics', metrics)
    
    def save_model_checkpoint(self, model_state: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state': model_state,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': time.time()
        }
        self._ml_state['training_history'].append(checkpoint)
        
        # Update best model if metrics improved
        if self._is_better_model(metrics):
            self._ml_state['best_model_state'] = checkpoint
            self.set_state('best_epoch', epoch)
            self.set_state('best_metrics', metrics)
    
    def _is_better_model(self, metrics: Dict[str, float]) -> bool:
        """Check if current model is better than best model."""
        if not self._ml_state['best_model_state']:
            return True
        
        best_metrics = self._ml_state['best_model_state']['metrics']
        # Simple comparison - can be overridden by subclasses
        return metrics.get('accuracy', 0) > best_metrics.get('accuracy', 0)
    
    # Performance monitoring
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self._performance_stats.copy()
        
        # Calculate rates
        total_ops = stats['total_operations']
        if total_ops > 0:
            stats['success_rate'] = stats['successful_operations'] / total_ops
            stats['failure_rate'] = stats['failed_operations'] / total_ops
            stats['avg_processing_time'] = stats['total_time'] / total_ops
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
            stats['avg_processing_time'] = 0.0
        
        # Add ML-specific metrics
        stats.update(self._ml_state)
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self._performance_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_time': 0.0,
            'last_operation_time': 0.0,
            'training_epochs': 0,
            'validation_accuracy': 0.0,
            'model_convergence': False
        }
        self.logger.info("Performance statistics reset")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get detailed performance analysis."""
        stats = self.get_performance_stats()
        
        # Calculate performance grade
        success_rate = stats['success_rate']
        if success_rate >= 0.95:
            grade = 'A'
        elif success_rate >= 0.85:
            grade = 'B'
        elif success_rate >= 0.70:
            grade = 'C'
        elif success_rate >= 0.50:
            grade = 'D'
        else:
            grade = 'F'
        
        # Generate recommendations
        recommendations = []
        if success_rate < 0.90:
            recommendations.append("Consider improving error handling and validation")
        if stats['avg_processing_time'] > self._slow_operation_threshold:
            recommendations.append("Consider optimizing processing performance")
        if stats['training_epochs'] == 0:
            recommendations.append("No training epochs recorded - check training process")
        
        return {
            'component_name': self.name,
            'performance_stats': stats,
            'performance_grade': grade,
            'recommendations': recommendations
        }
    
    # Lifecycle management
    
    def is_initialized(self) -> bool:
        """Check if component is initialized."""
        return self._initialized
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive component status."""
        return {
            'name': self.name,
            'initialized': self._initialized,
            'health': self._get_health_status(),
            'config': self.config.copy(),
            'performance_stats': self.get_performance_stats(),
            'state_keys': list(self._state.keys()),
            'ml_state_keys': list(self._ml_state.keys()),
            'dependencies': self.get_dependencies(),
            'capabilities': self.get_processing_capabilities()
        }
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get detailed health analysis."""
        stats = self.get_performance_stats()
        
        # Calculate overall health
        health_score = 0
        if self._initialized:
            health_score += 25
        if stats['success_rate'] > 0.8:
            health_score += 25
        if stats['total_operations'] > 0:
            health_score += 25
        if len(self._state) > 0:
            health_score += 25
        
        if health_score >= 90:
            overall_health = 'excellent'
        elif health_score >= 70:
            overall_health = 'good'
        elif health_score >= 50:
            overall_health = 'fair'
        else:
            overall_health = 'poor'
        
        return {
            'component_name': self.name,
            'overall_health': overall_health,
            'health_score': health_score,
            'initialization_status': self._initialized,
            'performance_metrics': stats,
            'configuration_status': self.validate_config(),
            'state_size': len(self._state),
            'ml_state_size': len(self._ml_state),
            'recommendations': self._get_health_recommendations(health_score)
        }
    
    def _get_health_status(self) -> str:
        """Get health status string."""
        if not self._initialized:
            return 'not_initialized'
        
        stats = self.get_performance_stats()
        if stats['success_rate'] < 0.5:
            return 'unhealthy'
        elif stats['success_rate'] < 0.8:
            return 'degraded'
        else:
            return 'healthy'
    
    def _get_health_recommendations(self, health_score: int) -> List[str]:
        """Get health recommendations based on score."""
        recommendations = []
        
        if health_score < 50:
            recommendations.append("Component needs immediate attention")
        if not self._initialized:
            recommendations.append("Initialize component before use")
        if self.get_performance_stats()['success_rate'] < 0.8:
            recommendations.append("Improve error handling and validation")
        
        return recommendations
    
    # Serialization
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize component for persistence."""
        return {
            'component_class': self.__class__.__name__,
            'name': self.name,
            'config': self.config.copy(),
            'state': self._state.copy(),
            'ml_state': self._ml_state.copy(),
            'performance_stats': self._performance_stats.copy(),
            'initialized': self._initialized,
            'timestamp': time.time(),
            'version': '1.0.0'
        }
    
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Deserialize component from persisted data."""
        try:
            self.name = data['name']
            self.config = data['config']
            self._state = data['state']
            self._ml_state = data['ml_state']
            self._performance_stats = data['performance_stats']
            self._initialized = data['initialized']
            
            self.logger.info(f"Component {self.name} deserialized successfully")
            
        except Exception as e:
            self.logger.error(f"Deserialization failed: {e}")
            raise
    
    def save_to_file(self, filepath: str) -> None:
        """Save component to JSON file."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            serialized_data = self.serialize()
            with open(filepath, 'w') as f:
                json.dump(serialized_data, f, indent=2, default=str)
            
            self.logger.info(f"Component saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save component: {e}")
            raise
    
    def load_from_file(self, filepath: str) -> None:
        """Load component from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.deserialize(data)
            self.logger.info(f"Component loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load component: {e}")
            raise
    
    # Safe processing
    
    def _safe_process(self, data: Any, **kwargs) -> Any:
        """Safely process data with comprehensive error handling."""
        try:
            # Pre-processing validation
            if not self._initialized:
                raise RuntimeError("Component not initialized")
            
            # Input validation
            validation_result = self.validate_input(data)
            if not validation_result.is_valid:
                raise ValueError(f"Input validation failed: {[e.message for e in validation_result.errors]}")
            
            # Capability checking
            if not self.can_process(data):
                raise RuntimeError("Component cannot process the provided data")
            
            # Memory requirement checking
            if not self._check_memory_usage(data):
                raise MemoryError("Insufficient memory for processing")
            
            # Process data
            start_time = time.time()
            result = self._process_data(data, **kwargs)
            processing_time = time.time() - start_time
            
            # Log operation
            self._log_operation("process", True, processing_time)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time if 'start_time' in locals() else 0
            self._log_operation("process", False, processing_time)
            self.logger.error(f"Safe processing failed: {e}")
            raise
    
    def _check_memory_usage(self, data: Any) -> bool:
        """Check if sufficient memory available."""
        try:
            if not PSUTIL_AVAILABLE:
                self.logger.warning("psutil not available, skipping memory check")
                return True
            
            # Estimate memory usage
            memory_usage = self.get_memory_requirements(data)
            estimated_memory = memory_usage.get('estimated_memory_mb', 0)
            
            # Check available memory
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            
            if estimated_memory > available_memory * 0.8:  # Use 80% of available memory
                self.logger.warning(f"Insufficient memory: {estimated_memory:.1f}MB required, {available_memory:.1f}MB available")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Memory check failed: {e}")
            return True  # Allow processing if check fails
    
    def _log_operation(self, operation: str, success: bool, processing_time: float) -> None:
        """Log operation details with appropriate level."""
        if success:
            if processing_time > self._slow_operation_threshold:
                self.logger.warning(f"Slow operation: {operation} took {processing_time:.2f}s")
            else:
                self.logger.info(f"Operation successful: {operation} in {processing_time:.2f}s")
        else:
            self.logger.error(f"Operation failed: {operation} after {processing_time:.2f}s")
    
    def _update_performance_stats(self, success: bool, processing_time: float) -> None:
        """Update performance statistics."""
        self._performance_stats['total_operations'] += 1
        self._performance_stats['total_time'] += processing_time
        self._performance_stats['last_operation_time'] = processing_time
        
        if success:
            self._performance_stats['successful_operations'] += 1
        else:
            self._performance_stats['failed_operations'] += 1
    
    def _validate_dependencies(self, dependencies: List[str]) -> bool:
        """Validate that all dependencies are available."""
        missing_deps = []
        
        for dep in dependencies:
            try:
                if dep == 'pandas':
                    import pandas
                elif dep == 'numpy':
                    import numpy
                elif dep == 'torch':
                    import torch
                elif dep == 'sklearn':
                    import sklearn
                elif dep == 'joblib':
                    import joblib
                else:
                    # Generic import
                    __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            self.logger.error(f"Missing dependencies: {missing_deps}")
            return False
        
        return True
    
    # Abstract methods with default implementations
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get comprehensive component metadata."""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'version': '1.0.0',
            'description': getattr(self, 'description', 'ModularComponent for models training'),
            'initialized': self._initialized,
            'config': self.config.copy(),
            'dependencies': self.get_dependencies(),
            'capabilities': self.get_processing_capabilities()
        }
    
    def get_dependencies(self) -> List[str]:
        """Get list of required dependencies."""
        return ['pandas', 'numpy', 'torch', 'sklearn']
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get expected output schema."""
        return {
            'type': 'dict',
            'description': 'Processed data with ML training results',
            'properties': {
                'model': {'type': 'object', 'description': 'Trained model'},
                'metrics': {'type': 'dict', 'description': 'Training metrics'},
                'metadata': {'type': 'dict', 'description': 'Additional metadata'}
            }
        }
    
    def get_required_config(self) -> List[str]:
        """Get required configuration parameters."""
        return []
    
    def can_process(self, data: Any) -> bool:
        """Check if component can process given data."""
        if data is None:
            return False
        
        if not self._initialized:
            return False
        
        # Check data type compatibility
        validation_result = self.validate_input(data)
        return validation_result.is_valid
    
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get component processing capabilities."""
        return {
            'input_types': ['pandas.DataFrame', 'numpy.ndarray', 'dict'],
            'output_types': ['dict'],
            'parallel_processing': False,
            'memory_efficient': True,
            'supports_checkpointing': True,
            'supports_validation': True
        }
    
    def estimate_processing_time(self, data: Any) -> float:
        """Estimate processing time for given data."""
        base_time = self.get_config('base_processing_time', 1.0)
        
        # Size-based factor
        if hasattr(data, '__len__'):
            size_factor = min(len(data) / 1000, 10.0)  # Cap at 10x
        else:
            size_factor = 1.0
        
        # Complexity factor
        complexity_factor = self.get_config('complexity_factor', 1.0)
        
        # Performance multiplier
        performance_multiplier = self.get_config('performance_multiplier', 1.0)
        
        return base_time * size_factor * complexity_factor * performance_multiplier
    
    def get_memory_requirements(self, data: Any) -> Dict[str, Any]:
        """Get memory requirements for processing data."""
        base_memory = 100  # MB
        
        # Calculate data memory usage
        if isinstance(data, pd.DataFrame):
            data_memory = data.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        elif isinstance(data, np.ndarray):
            data_memory = data.nbytes / (1024 * 1024)  # MB
        else:
            data_memory = 50  # Estimate for other types
        
        # Overhead factor
        overhead_factor = self.get_config('memory_overhead_factor', 2.0)
        
        estimated_memory = (base_memory + data_memory) * overhead_factor
        peak_memory = estimated_memory * 1.5  # 50% buffer
        
        return {
            'estimated_memory_mb': estimated_memory,
            'peak_memory_mb': peak_memory,
            'data_memory_mb': data_memory,
            'base_memory_mb': base_memory
        }


class ExampleModularComponent(ModularComponent):
    """Example implementation of ModularComponent for models training."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(name, config, logger)
        self.model_type = self.get_config('model_type', 'neural_network')
        self.training_config = self.get_config('training', {})
        self.version = "1.0.0"
        self.description = "Example ModularComponent for models training"
    
    def _initialize_resources(self) -> bool:
        """Initialize component-specific resources."""
        try:
            self.set_state('initialized_at', time.time())
            self.set_state('training_epoch', 0)
            self.set_state('best_accuracy', 0.0)
            self.set_ml_state('model_weights', None)
            return True
        except Exception as e:
            self.logger.error(f"Resource initialization failed: {e}")
            return False
    
    def _cleanup_resources(self) -> None:
        """Cleanup component-specific resources."""
        self.set_state('cleaned_up_at', time.time())
        self.set_ml_state('model_weights', None)
    
    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data with component logic."""
        # Simulate training process
        epochs = self.training_config.get('epochs', 10)
        
        for epoch in range(epochs):
            # Simulate training
            time.sleep(0.1)  # Simulate processing time
            
            # Simulate metrics
            accuracy = 0.5 + (epoch / epochs) * 0.4 + np.random.normal(0, 0.05)
            loss = 1.0 - (epoch / epochs) * 0.5 + np.random.normal(0, 0.1)
            
            metrics = {'accuracy': accuracy, 'loss': loss}
            self.update_training_progress(epoch, metrics)
            
            # Save checkpoint
            self.save_model_checkpoint({'weights': f'epoch_{epoch}'}, epoch, metrics)
        
        return {
            'model': {'type': self.model_type, 'trained': True},
            'metrics': self.get_ml_state('training_progress'),
            'metadata': {'epochs_trained': epochs, 'final_accuracy': accuracy}
        }
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for this component."""
        return {
            'min_size': 10,
            'max_size': 1000000,
            'required_attributes': ['required_column'],
            'data_types': ['pandas.DataFrame', 'dict']
        }
    
    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        """Validate data with component-specific rules."""
        errors = []
        warnings = []
        metadata = {}
        
        if isinstance(data, pd.DataFrame):
            if 'required_column' not in data.columns:
                errors.append("Missing required column")
            
            if len(data) < 10:
                warnings.append("Data size is small")
            
            metadata['data_shape'] = data.shape
        
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}


def create_modular_component(
    component_class: type,
    name: str,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> ModularComponent:
    """
    Factory function to create ModularComponent instances.
    
    Args:
        component_class: Component class to instantiate
        name: Component name
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Initialized ModularComponent instance
    """
    if not issubclass(component_class, ModularComponent):
        raise ValueError(f"Component class must inherit from ModularComponent")
    
    component = component_class(name, config, logger)
    
    if not component.initialize():
        raise RuntimeError(f"Failed to initialize component {name}")
    
    return component