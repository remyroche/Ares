"""
Modular Component Architecture for Backtesting Pipeline

This module provides a comprehensive modular component system specifically designed
for backtesting and trading strategy evaluation workflows. It includes the base
ModularComponent class and all supporting infrastructure.

Key Features:
- Backtesting-optimized component lifecycle management
- Portfolio and trade state tracking
- Strategy performance monitoring
- Risk management integration
- Comprehensive error handling and validation
- Serialization and persistence support
- Memory optimization for large datasets
- Real-time monitoring and health checks
"""

import time
import json
import logging
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path

# Backtesting-specific imports
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ValidationLevel(Enum):
    """Validation levels for input data."""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"


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
    BACKTESTING = "backtesting"
    RISK_MANAGEMENT = "risk_management"
    PORTFOLIO = "portfolio"


class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    TIMER = "timer"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


class MetricLevel(Enum):
    """Performance metric levels."""
    COMPONENT = "component"
    SYSTEM = "system"
    BACKTESTING = "backtesting"
    STRATEGY = "strategy"


@dataclass
class ErrorInfo:
    """Information about an error."""
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    timestamp: float = field(default_factory=time.time)
    component: str = ""
    traceback: str = ""


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
    warnings: List[ErrorInfo] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_level: ValidationLevel = ValidationLevel.BASIC


class ModularComponent(ABC):
    """
    Abstract base class for modular components in the backtesting pipeline.
    
    This class provides a comprehensive framework for creating reusable, testable,
    and maintainable components specifically optimized for backtesting workflows.
    
    Key Features:
    - Complete lifecycle management (initialize, process, cleanup)
    - Comprehensive input validation with backtesting-specific rules
    - State management for portfolio and trade tracking
    - Performance monitoring and health checks
    - Configuration management with validation
    - Serialization and persistence support
    - Memory optimization for large datasets
    - Error handling and recovery
    - Backtesting-specific capabilities
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the modular component.
        
        Args:
            name: Unique name for the component
            config: Configuration dictionary
            logger: Logger instance (creates default if None)
        """
        self.name = name
        self.config = config or {}
        self.logger = logger or logging.getLogger(f"{__name__}.{name}")
        
        # Component state
        self._initialized = False
        self._state = {}
        self._performance_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_time': 0.0,
            'success_rate': 0.0,
            'failure_rate': 0.0,
            'avg_processing_time': 0.0,
            'backtesting_specific': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'portfolio_value': 0.0
            }
        }
        
        # Backtesting-specific state
        self._portfolio_state = {
            'current_position': 0.0,
            'portfolio_value': 100000.0,
            'trade_history': [],
            'daily_returns': [],
            'risk_metrics': {},
            'strategy_parameters': {}
        }
        
        # Error tracking
        self._errors = []
        self._warnings = []
        
        # Component metadata
        self.version = "1.0.0"
        self.description = f"Modular Component: {name}"
        self.capabilities = self._get_default_capabilities()
        
        # Initialize with default configuration
        self._initialize_default_config()
    
    def _initialize_default_config(self) -> None:
        """Initialize default configuration for backtesting components."""
        default_config = {
            'validation_level': ValidationLevel.BASIC.value,
            'memory_limit_mb': 2048,
            'slow_operation_threshold': 10.0,
            'enable_performance_monitoring': True,
            'enable_state_persistence': True,
            'backtesting': {
                'initial_capital': 100000.0,
                'commission': 0.001,
                'slippage': 0.0005,
                'enable_risk_management': True,
                'max_position_size': 0.1,
                'stop_loss': 0.02,
                'take_profit': 0.04
            },
            'risk_management': {
                'max_drawdown': 0.15,
                'max_position_size': 0.1,
                'max_correlation': 0.7,
                'var_confidence': 0.95
            },
            'performance': {
                'enable_metrics': True,
                'enable_health_checks': True,
                'enable_serialization': True
            }
        }
        
        # Merge with provided config
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
            elif isinstance(value, dict) and isinstance(self.config[key], dict):
                self.config[key] = {**value, **self.config[key]}
    
    def _get_default_capabilities(self) -> Dict[str, Any]:
        """Get default capabilities for backtesting components."""
        return {
            'input_types': ['pandas.DataFrame', 'numpy.ndarray', 'dict', 'list'],
            'output_types': ['pandas.DataFrame', 'numpy.ndarray', 'dict'],
            'supports_parallel_processing': False,
            'supports_streaming': False,
            'memory_efficient': True,
            'backtesting_features': [
                'portfolio_tracking',
                'trade_execution',
                'risk_management',
                'performance_metrics',
                'strategy_optimization'
            ],
            'dependencies': ['pandas', 'numpy', 'vectorbt', 'matplotlib']
        }
    
    # Abstract Methods (Must be implemented by subclasses)
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the component and its resources.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """
        Process input data with comprehensive error handling.
        
        Args:
            data: Input data to process
            **kwargs: Additional processing parameters
            
        Returns:
            Processed data
        """
        pass
    
    @abstractmethod
    def validate_input(self, data: Any) -> ValidationResult:
        """
        Validate input data with comprehensive checks.
        
        Args:
            data: Data to validate
            
        Returns:
            ValidationResult with validation details
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources and reset component state."""
        pass
    
    @abstractmethod
    def get_component_info(self) -> Dict[str, Any]:
        """
        Get comprehensive component metadata.
        
        Returns:
            Dictionary with component information
        """
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """
        Get list of required dependencies.
        
        Returns:
            List of dependency names
        """
        pass
    
    @abstractmethod
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get expected output schema.
        
        Returns:
            Dictionary describing output schema
        """
        pass
    
    @abstractmethod
    def get_required_config(self) -> List[str]:
        """
        Get required configuration parameters.
        
        Returns:
            List of required config keys
        """
        pass
    
    @abstractmethod
    def can_process(self, data: Any) -> bool:
        """
        Check if component can process given data.
        
        Args:
            data: Data to check
            
        Returns:
            True if can process, False otherwise
        """
        pass
    
    @abstractmethod
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """
        Get component processing capabilities.
        
        Returns:
            Dictionary with capability information
        """
        pass
    
    @abstractmethod
    def estimate_processing_time(self, data: Any) -> float:
        """
        Estimate processing time for given data.
        
        Args:
            data: Data to estimate for
            
        Returns:
            Estimated time in seconds
        """
        pass
    
    @abstractmethod
    def get_memory_requirements(self, data: Any) -> Dict[str, Any]:
        """
        Get memory requirements for processing data.
        
        Args:
            data: Data to estimate for
            
        Returns:
            Dictionary with memory requirements
        """
        pass
    
    # Concrete Helper Methods (Available to all subclasses)
    
    def get_config(self, key: str = None, default: Any = None) -> Any:
        """
        Get configuration value(s) with support for nested keys.
        
        Args:
            key: Configuration key (supports nested keys like 'backtesting.initial_capital')
            default: Default value if key not found
            
        Returns:
            Configuration value or entire config if key is None
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
            config: Configuration updates
        """
        # Validate configuration keys
        for key in config.keys():
            if not isinstance(key, str):
                raise ValueError(f"Configuration keys must be strings, got {type(key)}")
        
        # Merge configuration
        self.config.update(config)
        
        # Trigger configuration change callback
        self._on_config_changed(config)
        
        self.logger.info(f"Configuration updated for component {self.name}")
    
    def validate_config(self) -> bool:
        """
        Validate component configuration.
        
        Returns:
            True if configuration is valid
        """
        try:
            # Check required configuration
            required_keys = self.get_required_config()
            for key in required_keys:
                if key not in self.config:
                    self.logger.error(f"Missing required configuration key: {key}")
                    return False
            
            # Validate configuration values
            for key, value in self.config.items():
                if not self._validate_config_value(key, value):
                    return False
            
            # Component-specific validation
            if hasattr(self, '_validate_component_config'):
                if not self._validate_component_config():
                    return False
            
            return True
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def _validate_config_value(self, key: str, value: Any) -> bool:
        """Validate a single configuration value."""
        try:
            # Basic type validation
            if key == 'validation_level':
                return value in [level.value for level in ValidationLevel]
            elif key == 'memory_limit_mb':
                return isinstance(value, (int, float)) and value > 0
            elif key == 'slow_operation_threshold':
                return isinstance(value, (int, float)) and value > 0
            elif key == 'backtesting':
                return isinstance(value, dict)
            elif key == 'risk_management':
                return isinstance(value, dict)
            elif key == 'performance':
                return isinstance(value, dict)
            
            return True
        except Exception:
            return False
    
    def set_state(self, key: str, value: Any) -> None:
        """
        Set component state with change tracking.
        
        Args:
            key: State key
            value: State value
        """
        if not isinstance(key, str):
            raise ValueError("State key must be a string")
        
        previous_value = self._state.get(key)
        self._state[key] = value
        
        # Track state changes
        self._on_state_changed(key, value, previous_value)
        
        self.logger.debug(f"State updated: {key} = {value}")
    
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
        self.logger.info(f"Cleared state keys: {cleared_keys}")
    
    def get_all_state(self) -> Dict[str, Any]:
        """Get all component state."""
        return self._state.copy()
    
    def has_state(self, key: str) -> bool:
        """Check if state key exists."""
        return key in self._state
    
    def remove_state(self, key: str) -> Any:
        """Remove state key and return its value."""
        return self._state.pop(key, None)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self._performance_stats.copy()
        
        # Calculate rates
        if stats['total_operations'] > 0:
            stats['success_rate'] = stats['successful_operations'] / stats['total_operations']
            stats['failure_rate'] = stats['failed_operations'] / stats['total_operations']
            stats['avg_processing_time'] = stats['total_time'] / stats['total_operations']
        
        # Add component-specific stats
        if hasattr(self, '_get_component_performance_stats'):
            component_stats = self._get_component_performance_stats()
            stats.update(component_stats)
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self._performance_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_time': 0.0,
            'success_rate': 0.0,
            'failure_rate': 0.0,
            'avg_processing_time': 0.0,
            'backtesting_specific': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'portfolio_value': 0.0
            }
        }
        self.logger.info("Performance statistics reset")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get detailed performance analysis."""
        stats = self.get_performance_stats()
        
        # Calculate performance grade
        grade = self._calculate_performance_grade(stats)
        
        # Generate recommendations
        recommendations = self._generate_performance_recommendations(stats)
        
        return {
            'component_name': self.name,
            'performance_stats': stats,
            'performance_grade': grade,
            'recommendations': recommendations,
            'timestamp': time.time()
        }
    
    def _calculate_performance_grade(self, stats: Dict[str, Any]) -> str:
        """Calculate performance grade based on statistics."""
        success_rate = stats.get('success_rate', 0)
        avg_time = stats.get('avg_processing_time', 0)
        
        if success_rate >= 0.95 and avg_time <= 1.0:
            return 'A'
        elif success_rate >= 0.90 and avg_time <= 5.0:
            return 'B'
        elif success_rate >= 0.80 and avg_time <= 10.0:
            return 'C'
        elif success_rate >= 0.70:
            return 'D'
        else:
            return 'F'
    
    def _generate_performance_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        success_rate = stats.get('success_rate', 0)
        avg_time = stats.get('avg_processing_time', 0)
        
        if success_rate < 0.90:
            recommendations.append("Consider improving error handling to increase success rate")
        
        if avg_time > 5.0:
            recommendations.append("Consider optimizing processing logic to reduce execution time")
        
        if stats.get('failed_operations', 0) > 0:
            recommendations.append("Review failed operations and improve error handling")
        
        return recommendations
    
    def is_initialized(self) -> bool:
        """Check if component is initialized."""
        return self._initialized
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive component status."""
        return {
            'name': self.name,
            'initialized': self._initialized,
            'health': self._calculate_health_status(),
            'config': self.config.copy(),
            'performance_stats': self.get_performance_stats(),
            'state_keys': list(self._state.keys()),
            'dependencies': self.get_dependencies(),
            'capabilities': self.capabilities.copy(),
            'version': self.version,
            'description': self.description
        }
    
    def _calculate_health_status(self) -> str:
        """Calculate component health status."""
        if not self._initialized:
            return 'uninitialized'
        
        stats = self.get_performance_stats()
        success_rate = stats.get('success_rate', 0)
        
        if success_rate >= 0.95:
            return 'healthy'
        elif success_rate >= 0.80:
            return 'warning'
        else:
            return 'critical'
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get detailed health analysis."""
        status = self.get_status()
        performance_stats = self.get_performance_stats()
        
        return {
            'component_name': self.name,
            'overall_health': status['health'],
            'initialization_status': self._initialized,
            'performance_metrics': performance_stats,
            'configuration_status': self.validate_config(),
            'state_size': len(self._state),
            'recommendations': self._generate_health_recommendations(status),
            'timestamp': time.time()
        }
    
    def _generate_health_recommendations(self, status: Dict[str, Any]) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        if not self._initialized:
            recommendations.append("Initialize component before use")
        
        if not status['configuration_status']:
            recommendations.append("Fix configuration issues")
        
        if status['health'] == 'critical':
            recommendations.append("Address critical performance issues")
        elif status['health'] == 'warning':
            recommendations.append("Monitor performance and address warnings")
        
        return recommendations
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize component for persistence."""
        return {
            'component_class': self.__class__.__name__,
            'name': self.name,
            'config': self.config.copy(),
            'state': self._state.copy(),
            'performance_stats': self._performance_stats.copy(),
            'initialized': self._initialized,
            'timestamp': time.time(),
            'version': self.version,
            'description': self.description,
            'capabilities': self.capabilities.copy()
        }
    
    def deserialize(self, data: Dict[str, Any]) -> None:
        """
        Deserialize component from persisted data.
        
        Args:
            data: Serialized component data
        """
        try:
            # Validate serialized data
            required_keys = ['component_class', 'name', 'config', 'state', 'performance_stats']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required key in serialized data: {key}")
            
            # Restore component state
            self.name = data['name']
            self.config = data['config']
            self._state = data['state']
            self._performance_stats = data['performance_stats']
            self._initialized = data['initialized']
            self.version = data.get('version', '1.0.0')
            self.description = data.get('description', f"Modular Component: {self.name}")
            self.capabilities = data.get('capabilities', self._get_default_capabilities())
            
            self.logger.info(f"Component {self.name} deserialized successfully")
            
        except Exception as e:
            self.logger.error(f"Deserialization failed: {e}")
            raise
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save component to JSON file.
        
        Args:
            filepath: Path to save file
        """
        try:
            # Create directory if needed
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Serialize and save
            data = self.serialize()
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Component saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save component to {filepath}: {e}")
            raise
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load component from JSON file.
        
        Args:
            filepath: Path to load file from
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.deserialize(data)
            self.logger.info(f"Component loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load component from {filepath}: {e}")
            raise
    
    def _safe_process(self, data: Any, **kwargs) -> Any:
        """
        Safely process data with comprehensive error handling.
        
        Args:
            data: Data to process
            **kwargs: Additional processing parameters
            
        Returns:
            Processed data
        """
        start_time = time.time()
        
        try:
            # Pre-processing validation
            if not self._initialized:
                raise RuntimeError("Component not initialized")
            
            # Input validation
            validation_result = self.validate_input(data)
            if not validation_result.is_valid:
                error_msg = f"Input validation failed: {[e.message for e in validation_result.errors]}"
                raise ValueError(error_msg)
            
            # Check if component can process data
            if not self.can_process(data):
                raise ValueError("Component cannot process this data")
            
            # Check memory requirements
            if not self._check_memory_usage(data):
                raise MemoryError("Insufficient memory for processing")
            
            # Process data
            result = self._process_data(data, **kwargs)
            
            # Update performance stats
            processing_time = time.time() - start_time
            self._update_performance_stats(True, processing_time)
            
            # Log operation
            self._log_operation("process", True, processing_time)
            
            return result
            
        except Exception as e:
            # Update performance stats
            processing_time = time.time() - start_time
            self._update_performance_stats(False, processing_time)
            
            # Log operation
            self._log_operation("process", False, processing_time)
            
            # Log error
            self.logger.error(f"Processing failed: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            
            raise
    
    def _check_memory_usage(self, data: Any) -> bool:
        """Check if sufficient memory available for processing."""
        try:
            memory_req = self.get_memory_requirements(data)
            estimated_memory = memory_req.get('estimated_mb', 0)
            memory_limit = self.get_config('memory_limit_mb', 2048)
            
            return estimated_memory <= memory_limit
        except Exception:
            # If memory check fails, allow processing
            return True
    
    def _update_performance_stats(self, success: bool, processing_time: float) -> None:
        """Update performance statistics."""
        self._performance_stats['total_operations'] += 1
        self._performance_stats['total_time'] += processing_time
        
        if success:
            self._performance_stats['successful_operations'] += 1
        else:
            self._performance_stats['failed_operations'] += 1
    
    def _log_operation(self, operation: str, success: bool, processing_time: float) -> None:
        """Log operation details with appropriate level."""
        if success:
            self.logger.info(f"Operation '{operation}' completed successfully in {processing_time:.3f}s")
        else:
            self.logger.error(f"Operation '{operation}' failed after {processing_time:.3f}s")
        
        # Log slow operations
        threshold = self.get_config('slow_operation_threshold', 10.0)
        if processing_time > threshold:
            self.logger.warning(f"Slow operation detected: '{operation}' took {processing_time:.3f}s")
    
    def _validate_dependencies(self, dependencies: List[str]) -> bool:
        """Validate that all dependencies are available."""
        try:
            for dep in dependencies:
                if dep == 'pandas':
                    import pandas
                elif dep == 'numpy':
                    import numpy
                elif dep == 'vectorbt':
                    import vectorbt
                elif dep == 'matplotlib':
                    import matplotlib
                else:
                    # Generic import
                    __import__(dep)
            return True
        except ImportError as e:
            self.logger.error(f"Dependency validation failed: {e}")
            return False
    
    # Abstract Helper Methods (Must be overridden by subclasses)
    
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
    
    # Optional Helper Methods (Can be overridden by subclasses)
    
    def _on_config_changed(self, config: Dict[str, Any]) -> None:
        """Called when configuration changes."""
        pass
    
    def _on_state_changed(self, key: str, value: Any, previous_value: Any) -> None:
        """Called when state changes."""
        pass
    
    def _get_component_performance_stats(self) -> Dict[str, Any]:
        """Get component-specific performance data."""
        return {}
    
    def _get_component_status(self) -> Dict[str, Any]:
        """Get component-specific status information."""
        return {}
    
    def _serialize_component_data(self) -> Dict[str, Any]:
        """Get component-specific data for serialization."""
        return {}
    
    def _deserialize_component_data(self, data: Dict[str, Any]) -> None:
        """Restore component-specific data from serialization."""
        pass
    
    def _validate_component_config(self) -> bool:
        """Validate component-specific configuration."""
        return True


class ExampleModularComponent(ModularComponent):
    """
    Example implementation of ModularComponent for backtesting.
    
    This class demonstrates how to implement all abstract methods
    and provides a template for creating custom backtesting components.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(name, config, logger)
        self.description = f"Example Backtesting Component: {name}"
    
    def initialize(self) -> bool:
        """Initialize the example component."""
        try:
            # Validate configuration
            if not self.validate_config():
                self.logger.error("Configuration validation failed")
                return False
            
            # Initialize resources
            if not self._initialize_resources():
                self.logger.error("Resource initialization failed")
                return False
            
            self._initialized = True
            self.logger.info(f"Component {self.name} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def process(self, data: Any, **kwargs) -> Any:
        """Process data with the example component."""
        return self._safe_process(data, **kwargs)
    
    def validate_input(self, data: Any) -> ValidationResult:
        """Validate input data for the example component."""
        errors = []
        warnings = []
        metadata = {}
        
        try:
            # Get validation rules
            rules = self._get_validation_rules()
            
            # Basic validation
            if data is None:
                errors.append(ErrorInfo(
                    message="Data cannot be None",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.VALIDATION
                ))
                return ValidationResult(False, errors, warnings, metadata)
            
            # Type validation
            data_types = rules.get('data_types', [])
            if data_types and not any(isinstance(data, eval(t)) for t in data_types):
                errors.append(ErrorInfo(
                    message=f"Invalid data type. Expected one of {data_types}",
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.VALIDATION
                ))
            
            # Size validation
            if hasattr(data, '__len__'):
                length = len(data)
                min_size = rules.get('min_data_points', 0)
                max_size = rules.get('max_data_points', float('inf'))
                
                if length < min_size:
                    warnings.append(ErrorInfo(
                        message=f"Data size {length} is below minimum {min_size}",
                        severity=ErrorSeverity.LOW,
                        category=ErrorCategory.VALIDATION
                    ))
                
                if length > max_size:
                    warnings.append(ErrorInfo(
                        message=f"Data size {length} exceeds maximum {max_size}",
                        severity=ErrorSeverity.MEDIUM,
                        category=ErrorCategory.VALIDATION
                    ))
                
                metadata['data_size'] = length
            
            # Component-specific validation
            component_validation = self._validate_component_specific(data)
            errors.extend([ErrorInfo(
                message=msg,
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.VALIDATION
            ) for msg in component_validation.get('errors', [])])
            warnings.extend([ErrorInfo(
                message=msg,
                severity=ErrorSeverity.LOW,
                category=ErrorCategory.VALIDATION
            ) for msg in component_validation.get('warnings', [])])
            metadata.update(component_validation.get('metadata', {}))
            
            return ValidationResult(len(errors) == 0, errors, warnings, metadata)
            
        except Exception as e:
            errors.append(ErrorInfo(
                message=f"Validation error: {e}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.VALIDATION
            ))
            return ValidationResult(False, errors, warnings, metadata)
    
    def cleanup(self) -> None:
        """Cleanup the example component."""
        try:
            self._cleanup_resources()
            self.clear_state()
            self.reset_stats()
            self._initialized = False
            self.logger.info(f"Component {self.name} cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information."""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'version': self.version,
            'description': self.description,
            'initialized': self._initialized,
            'config': self.config.copy(),
            'dependencies': self.get_dependencies(),
            'capabilities': self.capabilities.copy()
        }
    
    def get_dependencies(self) -> List[str]:
        """Get component dependencies."""
        return ['pandas', 'numpy', 'vectorbt', 'matplotlib']
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get output schema."""
        return {
            'type': 'dict',
            'description': 'Processed backtesting data',
            'properties': {
                'results': {'type': 'list', 'description': 'Processing results'},
                'metadata': {'type': 'dict', 'description': 'Processing metadata'}
            }
        }
    
    def get_required_config(self) -> List[str]:
        """Get required configuration keys."""
        return ['validation_level', 'memory_limit_mb']
    
    def can_process(self, data: Any) -> bool:
        """Check if component can process data."""
        if not self._initialized:
            return False
        
        if data is None:
            return False
        
        # Check data type compatibility
        supported_types = [pd.DataFrame, np.ndarray, dict, list]
        if not any(isinstance(data, t) for t in supported_types):
            return False
        
        # Check memory requirements
        return self._check_memory_usage(data)
    
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get processing capabilities."""
        return self.capabilities.copy()
    
    def estimate_processing_time(self, data: Any) -> float:
        """Estimate processing time."""
        base_time = self.get_config('base_processing_time', 1.0)
        
        # Size-based factor
        if hasattr(data, '__len__'):
            size_factor = len(data) / 1000.0
        else:
            size_factor = 1.0
        
        # Complexity factor
        complexity_factor = self.get_config('complexity_factor', 1.0)
        
        # Performance multiplier
        performance_multiplier = self.get_config('performance_multiplier', 1.0)
        
        return base_time * size_factor * complexity_factor * performance_multiplier
    
    def get_memory_requirements(self, data: Any) -> Dict[str, Any]:
        """Get memory requirements."""
        base_memory = 100  # MB
        
        if isinstance(data, pd.DataFrame):
            memory_usage = data.memory_usage(deep=True).sum() / (1024 * 1024)
        elif isinstance(data, np.ndarray):
            memory_usage = data.nbytes / (1024 * 1024)
        else:
            memory_usage = 50  # Estimate for other types
        
        overhead_factor = 1.5  # 50% overhead
        
        return {
            'estimated_mb': base_memory + memory_usage,
            'peak_mb': (base_memory + memory_usage) * overhead_factor,
            'data_mb': memory_usage,
            'overhead_mb': base_memory
        }
    
    # Abstract Helper Methods Implementation
    
    def _initialize_resources(self) -> bool:
        """Initialize example component resources."""
        try:
            self.set_state('initialized_at', time.time())
            self.set_state('processing_count', 0)
            return True
        except Exception as e:
            self.logger.error(f"Resource initialization failed: {e}")
            return False
    
    def _cleanup_resources(self) -> None:
        """Cleanup example component resources."""
        self.set_state('cleaned_up_at', time.time())
    
    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data with example logic."""
        # Increment processing count
        count = self.get_state('processing_count', 0)
        self.set_state('processing_count', count + 1)
        
        # Simple processing example
        if isinstance(data, pd.DataFrame):
            result = {
                'processed_data': data.copy(),
                'row_count': len(data),
                'column_count': len(data.columns),
                'processing_timestamp': time.time()
            }
        elif isinstance(data, np.ndarray):
            result = {
                'processed_data': data.copy(),
                'shape': data.shape,
                'dtype': str(data.dtype),
                'processing_timestamp': time.time()
            }
        else:
            result = {
                'processed_data': data,
                'type': type(data).__name__,
                'processing_timestamp': time.time()
            }
        
        return result
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for example component."""
        return {
            'min_data_points': 1,
            'max_data_points': 1000000,
            'data_types': ['pandas.DataFrame', 'numpy.ndarray', 'dict', 'list'],
            'required_keys': [],
            'optional_keys': []
        }
    
    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        """Validate data with example-specific rules."""
        errors = []
        warnings = []
        metadata = {}
        
        # Example validation logic
        if isinstance(data, dict):
            metadata['dict_keys'] = list(data.keys())
        elif hasattr(data, 'shape'):
            metadata['data_shape'] = data.shape
        
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}


def create_modular_component(
    component_class: type,
    name: str,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> ModularComponent:
    """
    Factory function to create modular components.
    
    Args:
        component_class: Component class to instantiate
        name: Component name
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Initialized component instance
    """
    if not issubclass(component_class, ModularComponent):
        raise ValueError(f"Component class must inherit from ModularComponent")
    
    component = component_class(name, config, logger)
    
    if not component.initialize():
        raise RuntimeError(f"Failed to initialize component {name}")
    
    return component


# Backtesting-specific utility functions

def create_backtesting_component(
    component_type: str,
    name: str,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> ModularComponent:
    """
    Create a backtesting-specific component.
    
    Args:
        component_type: Type of backtesting component
        name: Component name
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Initialized backtesting component
    """
    # Add backtesting-specific configuration
    backtesting_config = {
        'backtesting': {
            'initial_capital': 100000.0,
            'commission': 0.001,
            'slippage': 0.0005,
            'enable_risk_management': True
        },
        'risk_management': {
            'max_drawdown': 0.15,
            'max_position_size': 0.1,
            'max_correlation': 0.7
        }
    }
    
    if config:
        backtesting_config.update(config)
    
    # Create component based on type
    if component_type == 'example':
        return create_modular_component(ExampleModularComponent, name, backtesting_config, logger)
    else:
        raise ValueError(f"Unknown backtesting component type: {component_type}")


def validate_backtesting_environment() -> bool:
    """Validate that the backtesting environment is properly set up."""
    try:
        # Check required dependencies
        dependencies = ['pandas', 'numpy', 'vectorbt', 'matplotlib']
        for dep in dependencies:
            if dep == 'pandas':
                import pandas
            elif dep == 'numpy':
                import numpy
            elif dep == 'vectorbt':
                import vectorbt
            elif dep == 'matplotlib':
                import matplotlib
        
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False


# Export all public classes and functions
__all__ = [
    'ModularComponent',
    'ExampleModularComponent',
    'ValidationLevel',
    'ValidationResult',
    'ErrorInfo',
    'PerformanceMetric',
    'MetricType',
    'MetricLevel',
    'ErrorSeverity',
    'ErrorCategory',
    'create_modular_component',
    'create_backtesting_component',
    'validate_backtesting_environment'
]