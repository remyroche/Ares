"""
Base component for pre-training pipeline components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
import logging

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

class BasePreTrainingComponent(BaseComponent):
    """Base class for pre-training pipeline components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the pre-training component."""
        super().__init__(config)
        self.component_type = "pre_training"
    
    def get_component_type(self) -> str:
        """Get the component type."""
        return self.component_type
    
    def is_ready(self) -> bool:
        """Check if the component is ready to process data."""
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the component status."""
        return {
            'component_type': self.component_type,
            'is_ready': self.is_ready(),
            'config': self.get_config()
        }
