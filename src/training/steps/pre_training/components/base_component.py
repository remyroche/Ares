"""
Base component for pre-training pipeline components.

This module provides base classes that inherit from ModularComponent
for the pre-training pipeline, maintaining backward compatibility
while adding advanced features.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
import logging

# Import ModularComponent and related classes
try:
    from src.training.steps.pre_training.unified_data_driven_pipeline.core.modular_architecture import (
        ModularComponent, ValidationResult, ErrorInfo, PerformanceMetric
    )
    MODULAR_COMPONENT_AVAILABLE = True
except ImportError:
    MODULAR_COMPONENT_AVAILABLE = False
    # Fallback for when ModularComponent is not available
    class ModularComponent(ABC):
        def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
            self.name = name
            self.config = config or {}
            self.logger = logger or logging.getLogger(name)

logger = logging.getLogger(__name__)

@dataclass
class ComponentConfig:
    """Configuration for pre-training components."""
    
    # Component identification
    component_name: str = "base_component"
    component_type: str = "pre_training"
    
    # Processing settings
    enabled: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    
    # Validation settings
    strict_validation: bool = True
    skip_validation: bool = False
    
    # Performance settings
    memory_limit_mb: int = 1024
    timeout_seconds: int = 300
    
    # Logging settings
    log_level: str = "INFO"
    verbose: bool = False
    
    # Custom parameters for component-specific configuration
    custom_params: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'component_name': self.component_name,
            'component_type': self.component_type,
            'enabled': self.enabled,
            'parallel_processing': self.parallel_processing,
            'max_workers': self.max_workers,
            'strict_validation': self.strict_validation,
            'skip_validation': self.skip_validation,
            'memory_limit_mb': self.memory_limit_mb,
            'timeout_seconds': self.timeout_seconds,
            'log_level': self.log_level,
            'verbose': self.verbose
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentConfig':
        """Create config from dictionary."""
        return cls(**data)

@dataclass
class ComponentResult:
    """Result from component processing."""
    
    # Processing status
    success: bool = True
    error_message: Optional[str] = None
    
    # Data results
    processed_data: Any = None
    metadata: Dict[str, Any] = None
    artifacts: Dict[str, Any] = None
    
    # Performance metrics
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Validation results
    validation_passed: bool = True
    validation_errors: List[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}
        if self.artifacts is None:
            self.artifacts = {}
        if self.validation_errors is None:
            self.validation_errors = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'error_message': self.error_message,
            'processed_data': self.processed_data,
            'metadata': self.metadata,
            'artifacts': self.artifacts,
            'processing_time': self.processing_time,
            'memory_usage_mb': self.memory_usage_mb,
            'validation_passed': self.validation_passed,
            'validation_errors': self.validation_errors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentResult':
        """Create result from dictionary."""
        return cls(**data)

class BaseComponent(ABC):
    """Base class for pre-training pipeline components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the component with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process the input data and return the result."""
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate the input data."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get the component configuration."""
        return self.config.copy()
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set the component configuration."""
        self.config.update(config)
    
    def log_info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)

class BasePreTrainingComponent(ModularComponent if MODULAR_COMPONENT_AVAILABLE else BaseComponent):
    """Base class for pre-training pipeline components with ModularComponent integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 name: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """Initialize the pre-training component."""
        if MODULAR_COMPONENT_AVAILABLE:
            # Use ModularComponent initialization
            component_name = name or self.__class__.__name__
            super().__init__(component_name, config, logger)
            self.component_type = "pre_training"
            # Set component type in config
            self.update_config({'component_type': self.component_type})
        else:
            # Fallback to original initialization
            super().__init__(config)
            self.component_type = "pre_training"
    
    def get_component_type(self) -> str:
        """Get the component type."""
        return self.component_type
    
    def is_ready(self) -> bool:
        """Check if the component is ready to process data."""
        if MODULAR_COMPONENT_AVAILABLE:
            return self.is_initialized()
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the component status."""
        if MODULAR_COMPONENT_AVAILABLE:
            # Use ModularComponent status with additional pre-training info
            status = super().get_status()
            status.update({
                'component_type': self.component_type,
                'is_ready': self.is_ready()
            })
            return status
        else:
            # Fallback to original status
            return {
                'component_type': self.component_type,
                'is_ready': self.is_ready(),
                'config': self.get_config()
            }
    
    # ModularComponent abstract method implementations
    if MODULAR_COMPONENT_AVAILABLE:
        def _initialize_resources(self) -> bool:
            """Initialize component-specific resources."""
            try:
                # Set up pre-training specific resources
                self.set_state('component_type', self.component_type)
                self.set_state('initialization_time', self.get_config('initialization_time'))
                return True
            except Exception as e:
                self.logger.error(f"Failed to initialize pre-training component resources: {e}")
                return False
        
        def _cleanup_resources(self) -> None:
            """Cleanup component-specific resources."""
            try:
                # Cleanup pre-training specific resources
                self.set_state('cleanup_time', self.get_config('cleanup_time'))
            except Exception as e:
                self.logger.error(f"Failed to cleanup pre-training component resources: {e}")
        
        def _process_data(self, data: Any, **kwargs) -> Any:
            """Process data using the component's process method."""
            # Delegate to the existing process method
            return self.process(data, **kwargs)
        
        def _get_validation_rules(self) -> Dict[str, Any]:
            """Get validation rules for pre-training components."""
            return {
                'min_data_size': self.get_config('min_data_size', 1),
                'max_data_size': self.get_config('max_data_size', 1000000),
                'required_columns': self.get_config('required_columns', []),
                'data_types': self.get_config('data_types', ['pandas.DataFrame'])
            }
        
        def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
            """Validate data with pre-training specific rules."""
            errors = []
            warnings = []
            metadata = {}
            
            # Use existing validate method if available
            if hasattr(self, 'validate'):
                try:
                    is_valid = self.validate(data)
                    if not is_valid:
                        errors.append("Component validation failed")
                except Exception as e:
                    errors.append(f"Validation error: {str(e)}")
            
            # Add pre-training specific validation
            if hasattr(data, 'shape'):
                data_size = data.shape[0] if len(data.shape) > 0 else 0
                min_size = self.get_config('min_data_size', 1)
                max_size = self.get_config('max_data_size', 1000000)
                
                if data_size < min_size:
                    errors.append(f"Data size {data_size} is below minimum {min_size}")
                elif data_size > max_size:
                    warnings.append(f"Data size {data_size} is above recommended maximum {max_size}")
                
                metadata['data_size'] = data_size
            
            return {'errors': errors, 'warnings': warnings, 'metadata': metadata}
        
        def get_required_config(self) -> List[str]:
            """Get required configuration parameters."""
            return ['component_name', 'component_type']
        
        def get_dependencies(self) -> List[str]:
            """Get component dependencies."""
            return ['pandas', 'numpy']
        
        def get_output_schema(self) -> Dict[str, Any]:
            """Get expected output schema."""
            return {
                'type': 'pandas.DataFrame',
                'required_columns': ['processed_data'],
                'optional_columns': ['metadata', 'artifacts']
            }
        
        def can_process(self, data: Any) -> bool:
            """Check if component can process the given data."""
            try:
                # Basic type checking
                if hasattr(data, 'shape'):  # DataFrame or array-like
                    return True
                elif isinstance(data, (list, tuple)):
                    return True
                else:
                    return False
            except Exception:
                return False
        
        def get_processing_capabilities(self) -> Dict[str, Any]:
            """Get processing capabilities."""
            return {
                'data_types': ['pandas.DataFrame', 'numpy.ndarray', 'list', 'tuple'],
                'max_data_size': self.get_config('max_data_size', 1000000),
                'parallel_processing': self.get_config('parallel_processing', True),
                'gpu_acceleration': self.get_config('gpu_acceleration', False)
            }
        
        def estimate_processing_time(self, data: Any) -> float:
            """Estimate processing time in seconds."""
            try:
                if hasattr(data, 'shape'):
                    data_size = data.shape[0] if len(data.shape) > 0 else 0
                else:
                    data_size = len(data) if hasattr(data, '__len__') else 1
                
                # Simple estimation based on data size
                base_time = 0.001  # 1ms base time
                size_factor = data_size * 0.000001  # 1μs per row
                return base_time + size_factor
            except Exception:
                return 1.0  # Default 1 second
        
        def get_memory_requirements(self, data: Any) -> Dict[str, Any]:
            """Get memory requirements for processing data."""
            try:
                if hasattr(data, 'memory_usage'):
                    # Pandas DataFrame
                    memory_usage = data.memory_usage(deep=True).sum()
                    estimated_memory = memory_usage * 2  # 2x for processing overhead
                elif hasattr(data, 'nbytes'):
                    # NumPy array
                    estimated_memory = data.nbytes * 2
                else:
                    # Fallback estimation
                    estimated_memory = 100 * 1024 * 1024  # 100MB default
                
                return {
                    'estimated_memory_mb': estimated_memory / (1024 * 1024),
                    'peak_memory_mb': estimated_memory * 1.5 / (1024 * 1024),
                    'recommended_memory_mb': estimated_memory * 2 / (1024 * 1024)
                }
            except Exception:
                return {
                    'estimated_memory_mb': 100.0,
                    'peak_memory_mb': 150.0,
                    'recommended_memory_mb': 200.0
                }
