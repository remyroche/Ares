"""
Training-specific artifact manager utilities with comprehensive type safety.

This module re-exports the main artifact manager functions for training modules
with enhanced type safety and comprehensive tpritn logging.
"""

from typing import Dict, Any, Optional, List, TypeVar, Protocol, runtime_checkable, Final
from src.utils.artifact_manager import (
    ArtifactManager, 
    get_analyst_context, 
    setup_enhanced_artifact_manager, 
    get_pretraining_artifact_manager,
    ArtifactMetadata,
    OperationMetrics,
    CacheEntry,
    CompressionType,
    OperationType,
    RetryStrategy,
    RetryConfig,
    MemoryConfig
)
from src.utils.tprint import (
    tprint, tprint_success, tprint_info, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_progress, tprint_structured,
    tprint_exception, tprint_with_level, LogLevel, TPrintConfig
)

# Type definitions for better type safety
T = TypeVar('T')
ConfigType = TypeVar('ConfigType', bound=Dict[str, Any])

# Protocol definitions for better type checking
@runtime_checkable
class TrainingStep(Protocol):
    """Protocol for training steps."""
    def execute(self, config: Dict[str, Any]) -> Dict[str, Any]: ...
    def validate_config(self) -> None: ...
    def get_status(self) -> Dict[str, Any]: ...

def get_step_context_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get step context from configuration with comprehensive validation.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing step context
        
    Raises:
        TypeError: If config is not a dictionary
        ValueError: If config is empty
    """
    # Validate input parameters
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got: {type(config).__name__}")
    if not config:
        raise ValueError("config cannot be empty")
    
    tprint_info("🔍 Extracting step context from configuration")
    
    context = {
        'symbol': config.get('symbol', 'UNKNOWN'),
        'timeframe': config.get('timeframe', '15m'),
        'exchange': config.get('exchange', 'binance'),
        'execution_mode': config.get('execution_mode', 'light'),
        'step_name': config.get('step_name', 'unknown')
    }
    
    tprint_success(f"✅ Step context extracted: {context}")
    return context

def create_training_artifact_manager(config: Dict[str, Any]) -> ArtifactManager:
    """Create a training-specific artifact manager with enhanced configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured ArtifactManager instance for training
        
    Raises:
        TypeError: If config is not a dictionary
        ValueError: If configuration is invalid
    """
    # Validate input parameters
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got: {type(config).__name__}")
    
    tprint_info("🚀 Creating training-specific artifact manager")
    
    # Add training-specific configuration
    training_config = config.copy()
    training_config.update({
        'enable_compression': True,
        'enable_caching': True,
        'enable_memory_optimization': True,
        'max_cache_size_mb': 2048.0,  # Larger cache for training
        'max_memory_mb': 8000.0,  # More memory for training
        'enable_thread_safety': True,
        'compression': 'auto'
    })
    
    try:
        manager = ArtifactManager(config=training_config)
        tprint_success("✅ Training artifact manager created successfully")
        return manager
    except Exception as e:
        tprint_error(f"❌ Failed to create training artifact manager: {e}")
        raise

def validate_training_config(config: Dict[str, Any]) -> None:
    """Validate training configuration with comprehensive checks.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        TypeError: If config is not a dictionary
        ValueError: If configuration is invalid
    """
    # Validate input parameters
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got: {type(config).__name__}")
    if not config:
        raise ValueError("config cannot be empty")
    
    tprint_info("🔍 Validating training configuration")
    
    # Required fields for training
    required_fields = ['step_name', 'execution_mode', 'symbol', 'exchange']
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        raise ValueError(f"Missing required training configuration fields: {missing_fields}")
    
    # Validate field types
    type_validations = {
        'step_name': str,
        'execution_mode': str,
        'symbol': str,
        'exchange': str,
        'timeframe': str,
        'model': str
    }
    
    for field, expected_type in type_validations.items():
        if field in config and not isinstance(config[field], expected_type):
            raise TypeError(f"Config field '{field}' must be {expected_type.__name__}, got {type(config[field]).__name__}")
    
    tprint_success("✅ Training configuration validation passed")

def get_training_metrics(artifact_manager: ArtifactManager) -> Dict[str, Any]:
    """Get comprehensive training metrics from artifact manager.
    
    Args:
        artifact_manager: ArtifactManager instance
        
    Returns:
        Dictionary containing training metrics
        
    Raises:
        TypeError: If artifact_manager is not an ArtifactManager instance
    """
    if not isinstance(artifact_manager, ArtifactManager):
        raise TypeError(f"artifact_manager must be an ArtifactManager instance, got: {type(artifact_manager).__name__}")
    
    tprint_info("📊 Collecting training metrics")
    
    try:
        # Get comprehensive stats
        stats = artifact_manager.get_stats()
        performance_metrics = artifact_manager.get_performance_metrics()
        memory_analytics = artifact_manager.get_memory_analytics()
        
        training_metrics = {
            'stats': stats,
            'performance': performance_metrics,
            'memory': memory_analytics,
            'timestamp': datetime.now().isoformat()
        }
        
        tprint_success(f"✅ Training metrics collected: {len(training_metrics)} metric categories")
        return training_metrics
        
    except Exception as e:
        tprint_error(f"❌ Failed to collect training metrics: {e}")
        return {'error': str(e)}

def log_training_progress(step_name: str, progress: float, message: str = "") -> None:
    """Log training progress with structured information.
    
    Args:
        step_name: Name of the training step
        progress: Progress percentage (0.0 to 1.0)
        message: Optional progress message
        
    Raises:
        ValueError: If step_name is empty or progress is invalid
        TypeError: If parameter types are incorrect
    """
    # Validate input parameters
    if not isinstance(step_name, str) or not step_name.strip():
        raise ValueError(f"step_name must be a non-empty string, got: {step_name}")
    if not isinstance(progress, (int, float)):
        raise TypeError(f"progress must be a number, got: {type(progress).__name__}")
    if not 0.0 <= progress <= 1.0:
        raise ValueError(f"progress must be between 0.0 and 1.0, got: {progress}")
    if not isinstance(message, str):
        raise TypeError(f"message must be a string, got: {type(message).__name__}")
    
    progress_percentage = progress * 100
    
    if message:
        tprint_progress(int(progress_percentage), 100, f"{step_name}: {message}")
    else:
        tprint_progress(int(progress_percentage), 100, step_name)

def log_training_error(step_name: str, error: Exception, context: str = "") -> None:
    """Log training error with comprehensive context.
    
    Args:
        step_name: Name of the training step
        error: Exception that occurred
        context: Optional context information
        
    Raises:
        ValueError: If step_name is empty
        TypeError: If parameter types are incorrect
    """
    # Validate input parameters
    if not isinstance(step_name, str) or not step_name.strip():
        raise ValueError(f"step_name must be a non-empty string, got: {step_name}")
    if not isinstance(error, Exception):
        raise TypeError(f"error must be an Exception, got: {type(error).__name__}")
    if not isinstance(context, str):
        raise TypeError(f"context must be a string, got: {type(context).__name__}")
    
    error_message = f"Training step '{step_name}' failed"
    if context:
        error_message += f" in {context}"
    
    tprint_error(error_message)
    tprint_exception(error, context)

__all__ = [
    'ArtifactManager', 
    'get_analyst_context', 
    'setup_enhanced_artifact_manager', 
    'get_pretraining_artifact_manager',
    'get_step_context_from_config',
    'create_training_artifact_manager',
    'validate_training_config',
    'get_training_metrics',
    'log_training_progress',
    'log_training_error',
    # Re-export types for better IDE support
    'ArtifactMetadata',
    'OperationMetrics',
    'CacheEntry',
    'CompressionType',
    'OperationType',
    'RetryStrategy',
    'RetryConfig',
    'MemoryConfig',
    'LogLevel',
    'TPrintConfig'
]