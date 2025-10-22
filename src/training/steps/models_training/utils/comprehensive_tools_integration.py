"""
Comprehensive Tools Integration for Model Training

This module provides utility functions and decorators that make it easy to use
BaseStep comprehensive tools in model training components.

Key Features:
- Simplified access to BaseStep comprehensive tools
- Decorators for common training patterns
- Utility functions for data processing
- Performance monitoring helpers
- Error handling and logging utilities
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from functools import wraps
from dataclasses import dataclass

import pandas as pd
import numpy as np

from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, 
    tprint_debug, tprint_performance, tprint_data_format, LogLevel,
    tprint_operation_start, tprint_operation_end, tprint_data_preview,
    tprint_dict, tprint_list, tprint_dataframe_info, tprint_model_info,
    tprint_performance_summary, tprint_memory_usage, tprint_hardware_stats
)
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.hardware.optimization_decorators import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked
)
from src.utils.hardware.memory_optimized_decorators import (
    memory_optimized, comprehensive_memory_optimization, MemoryOptimizationLevel
)


@dataclass
class ComprehensiveToolsConfig:
    """Configuration for comprehensive tools integration."""
    enable_logging: bool = True
    enable_performance_monitoring: bool = True
    enable_memory_optimization: bool = True
    enable_hardware_optimization: bool = True
    enable_error_handling: bool = True
    log_level: str = "INFO"
    memory_optimization_level: str = "AGGRESSIVE"
    cache_enabled: bool = True


class ComprehensiveToolsIntegration:
    """
    Integration class for BaseStep comprehensive tools in model training.
    
    This class provides simplified access to all BaseStep utilities and
    common patterns for model training components.
    """
    
    def __init__(self, base_step_instance, config: Optional[ComprehensiveToolsConfig] = None):
        """
        Initialize comprehensive tools integration.
        
        Args:
            base_step_instance: Instance of BaseStep or GeneralizedModelTrainingBase
            config: Configuration for tools integration
        """
        self.base_step = base_step_instance
        self.config = config or ComprehensiveToolsConfig()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Initialize tool availability
        self._tool_availability = self.base_step._get_availability_status()
        self._log_tool_availability()
    
    def _log_tool_availability(self) -> None:
        """Log available comprehensive tools."""
        if self.config.enable_logging:
            available_count = sum(self._tool_availability.values())
            total_count = len(self._tool_availability)
            tprint_info(f"🔧 Comprehensive Tools: {available_count}/{total_count} available")
            
            if self.config.log_level == "DEBUG":
                for tool, available in self._tool_availability.items():
                    status = "✅" if available else "❌"
                    tprint_debug(f"  {status} {tool}")
    
    # ============================================================================
    # DATA PROCESSING UTILITIES
    # ============================================================================
    
    def process_data_with_comprehensive_tools(
        self, 
        data: pd.DataFrame, 
        operation: str = "preprocess",
        **kwargs
    ) -> pd.DataFrame:
        """
        Process data using comprehensive tools.
        
        Args:
            data: Input data
            operation: Processing operation
            **kwargs: Additional parameters
            
        Returns:
            Processed data
        """
        try:
            if self.config.enable_logging:
                tprint_operation_start(f"Data Processing: {operation}")
            
            # Use BaseStep data preview
            if self.config.enable_logging:
                tprint_data_preview(data, f"Input data for {operation}", max_rows=5)
            
            # Apply data processing based on operation
            if operation == "preprocess":
                processed_data = self._preprocess_data(data, **kwargs)
            elif operation == "clean":
                processed_data = self._clean_data(data, **kwargs)
            elif operation == "validate":
                processed_data = self._validate_data(data, **kwargs)
            elif operation == "optimize":
                processed_data = self._optimize_data(data, **kwargs)
            else:
                tprint_warning(f"⚠️ Unknown operation: {operation}")
                processed_data = data
            
            # Use BaseStep data preview for output
            if self.config.enable_logging:
                tprint_data_preview(processed_data, f"Output data for {operation}", max_rows=5)
                tprint_operation_end(f"Data Processing: {operation}", success=True)
            
            return processed_data
            
        except Exception as e:
            if self.config.enable_logging:
                tprint_operation_end(f"Data Processing: {operation}", success=False)
                tprint_error(f"❌ Data processing failed: {e}")
            raise
    
    def _preprocess_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Preprocess data using comprehensive tools."""
        # Handle missing values
        if data.isnull().any().any():
            tprint_warning("⚠️ Found missing values, filling with median")
            data = self.base_step._safe_dataframe_operation(data, "fillna", method="median")
        
        # Handle infinite values
        if np.isinf(data).any().any():
            tprint_warning("⚠️ Found infinite values, replacing with finite values")
            data = data.replace([np.inf, -np.inf], np.nan)
            data = self.base_step._safe_dataframe_operation(data, "fillna", method="median")
        
        # Optimize memory if enabled
        if self.config.enable_memory_optimization and self.base_step.hardware_utils:
            data = self.base_step.hardware_utils['optimize_dataframe'](data)
            tprint_debug("🧠 Memory optimization applied")
        
        return data
    
    def _clean_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Clean data using comprehensive tools."""
        # Use BaseStep data quality utilities if available
        if self.base_step.data_quality and 'DataCleaner' in self.base_step.data_quality:
            cleaner = self.base_step._get_data_cleaner()
            if cleaner:
                data = cleaner.clean(data)
                tprint_debug("🧹 Data cleaned using comprehensive tools")
        else:
            # Fallback cleaning
            data = self._preprocess_data(data)
        
        return data
    
    def _validate_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Validate data using comprehensive tools."""
        # Use BaseStep validation utilities
        required_columns = kwargs.get('required_columns', [])
        if required_columns:
            is_valid = self.base_step._validate_dataframe_columns(data, required_columns)
            if not is_valid:
                raise ValueError(f"Data validation failed: missing required columns {required_columns}")
        
        # Validate data types
        for col in data.columns:
            if data[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except:
                    pass
        
        tprint_debug("✅ Data validation passed")
        return data
    
    def _optimize_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Optimize data using comprehensive tools."""
        # Use BaseStep hardware optimization
        if self.config.enable_hardware_optimization and self.base_step.hardware_utils:
            data = self.base_step.hardware_utils['optimize_dataframe'](data)
            tprint_debug("⚡ Hardware optimization applied")
        
        # Use BaseStep memory optimization
        if self.config.enable_memory_optimization:
            # Convert to optimal dtypes
            for col in data.select_dtypes(include=['int64']).columns:
                data[col] = pd.to_numeric(data[col], downcast='integer')
            for col in data.select_dtypes(include=['float64']).columns:
                data[col] = pd.to_numeric(data[col], downcast='float')
            
            tprint_debug("💾 Memory optimization applied")
        
        return data
    
    # ============================================================================
    # MODEL MANAGEMENT UTILITIES
    # ============================================================================
    
    def save_model_with_comprehensive_tools(
        self, 
        model: Any, 
        model_name: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model using comprehensive tools.
        
        Args:
            model: Model to save
            model_name: Name for the model
            metadata: Additional metadata
            
        Returns:
            Path to saved model
        """
        try:
            if self.config.enable_logging:
                tprint_operation_start(f"Model Saving: {model_name}")
            
            # Create comprehensive metadata
            model_metadata = {
                'model_name': model_name,
                'model_type': type(model).__name__,
                'timestamp': time.strftime("%Y%m%d_%H%M%S"),
                'comprehensive_tools_used': True
            }
            
            if metadata:
                model_metadata.update(metadata)
            
            # Use BaseStep model saving
            model_path = self.base_step._save_model(model, model_name, model_metadata)
            
            # Log model information
            if self.config.enable_logging:
                tprint_model_info(model, f"Saved {model_name}")
                tprint_operation_end(f"Model Saving: {model_name}", success=True)
            
            return model_path
            
        except Exception as e:
            if self.config.enable_logging:
                tprint_operation_end(f"Model Saving: {model_name}", success=False)
                tprint_error(f"❌ Model saving failed: {e}")
            raise
    
    def load_model_with_comprehensive_tools(self, model_name: str) -> Any:
        """
        Load model using comprehensive tools.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model
        """
        try:
            if self.config.enable_logging:
                tprint_operation_start(f"Model Loading: {model_name}")
            
            # Use BaseStep model loading
            model = self.base_step._load_model(model_name)
            
            # Log model information
            if self.config.enable_logging:
                tprint_model_info(model, f"Loaded {model_name}")
                tprint_operation_end(f"Model Loading: {model_name}", success=True)
            
            return model
            
        except Exception as e:
            if self.config.enable_logging:
                tprint_operation_end(f"Model Loading: {model_name}", success=False)
                tprint_error(f"❌ Model loading failed: {e}")
            raise
    
    # ============================================================================
    # PERFORMANCE MONITORING UTILITIES
    # ============================================================================
    
    def monitor_performance(self, operation: str) -> Callable:
        """
        Decorator for performance monitoring.
        
        Args:
            operation: Name of the operation to monitor
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.config.enable_performance_monitoring:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                try:
                    if self.config.enable_logging:
                        tprint_operation_start(operation)
                    
                    result = func(*args, **kwargs)
                    
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    
                    # Log performance metrics
                    duration = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    if self.config.enable_logging:
                        tprint_performance(f"⏱️ {operation}: {duration:.2f}s, Memory: {memory_delta:+.1f}MB")
                        tprint_operation_end(operation, success=True)
                    
                    return result
                    
                except Exception as e:
                    if self.config.enable_logging:
                        tprint_operation_end(operation, success=False)
                        tprint_error(f"❌ {operation} failed: {e}")
                    raise
            
            return wrapper
        return decorator
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            # Use BaseStep performance utilities
            base_performance = self.base_step._get_performance_metrics()
            memory_analytics = self.base_step._get_memory_analytics()
            comprehensive_stats = self.base_step._get_comprehensive_stats()
            
            # Add tool availability
            tool_availability = self._tool_availability
            
            return {
                'base_performance': base_performance,
                'memory_analytics': memory_analytics,
                'comprehensive_stats': comprehensive_stats,
                'tool_availability': tool_availability,
                'config': self.config.__dict__
            }
            
        except Exception as e:
            tprint_error(f"❌ Failed to get performance summary: {e}")
            return {}
    
    def log_performance_summary(self) -> None:
        """Log comprehensive performance summary."""
        try:
            performance_summary = self.get_performance_summary()
            
            tprint_performance_summary(performance_summary['base_performance'])
            tprint_memory_usage(performance_summary['memory_analytics'])
            
            if self.base_step.hardware_utils:
                tprint_hardware_stats(performance_summary['comprehensive_stats'])
            
            tprint_dict(performance_summary['tool_availability'], "Tool Availability")
            
        except Exception as e:
            tprint_error(f"❌ Failed to log performance summary: {e}")
    
    # ============================================================================
    # ERROR HANDLING UTILITIES
    # ============================================================================
    
    def handle_errors(self, operation: str, default_return: Any = None) -> Callable:
        """
        Decorator for error handling.
        
        Args:
            operation: Name of the operation
            default_return: Default return value on error
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.config.enable_error_handling:
                    return func(*args, **kwargs)
                
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if self.config.enable_logging:
                        tprint_error(f"❌ {operation} failed: {e}")
                    self.logger.error(f"{operation} failed: {e}")
                    
                    if default_return is not None:
                        return default_return
                    raise
            
            return wrapper
        return decorator
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def get_available_tools(self) -> List[str]:
        """Get list of available comprehensive tools."""
        return [tool for tool, available in self._tool_availability.items() if available]
    
    def is_tool_available(self, tool_name: str) -> bool:
        """Check if a specific tool is available."""
        return self._tool_availability.get(tool_name, False)
    
    def get_tool_help(self) -> Dict[str, Any]:
        """Get help information for available tools."""
        return self.base_step._get_utility_help()


# ============================================================================
# DECORATORS FOR COMMON PATTERNS
# ============================================================================

def with_comprehensive_tools(
    enable_logging: bool = True,
    enable_performance_monitoring: bool = True,
    enable_memory_optimization: bool = True,
    enable_hardware_optimization: bool = True,
    enable_error_handling: bool = True
):
    """
    Decorator to add comprehensive tools integration to any method.
    
    Args:
        enable_logging: Enable comprehensive logging
        enable_performance_monitoring: Enable performance monitoring
        enable_memory_optimization: Enable memory optimization
        enable_hardware_optimization: Enable hardware optimization
        enable_error_handling: Enable error handling
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create comprehensive tools integration
            config = ComprehensiveToolsConfig(
                enable_logging=enable_logging,
                enable_performance_monitoring=enable_performance_monitoring,
                enable_memory_optimization=enable_memory_optimization,
                enable_hardware_optimization=enable_hardware_optimization,
                enable_error_handling=enable_error_handling
            )
            
            tools = ComprehensiveToolsIntegration(self, config)
            
            # Add tools to self for use in the method
            self.comprehensive_tools = tools
            
            # Call the original method
            return func(self, *args, **kwargs)
        
        return wrapper
    return decorator


def with_memory_optimization(level: str = "AGGRESSIVE"):
    """
    Decorator for memory optimization.
    
    Args:
        level: Memory optimization level
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            optimization_level = MemoryOptimizationLevel[level.upper()]
            
            @comprehensive_memory_optimization(
                optimization_level=optimization_level,
                enable_caching=True,
                enable_chunking=True,
                enable_gc=True,
                enable_pools=True
            )
            def optimized_func(*args, **kwargs):
                return func(*args, **kwargs)
            
            return optimized_func(*args, **kwargs)
        
        return wrapper
    return decorator


def with_performance_tracking(operation_name: str):
    """
    Decorator for performance tracking.
    
    Args:
        operation_name: Name of the operation to track
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                tprint_operation_start(operation_name)
                result = func(*args, **kwargs)
                tprint_operation_end(operation_name, success=True)
                return result
            except Exception as e:
                tprint_operation_end(operation_name, success=False)
                tprint_error(f"❌ {operation_name} failed: {e}")
                raise
            finally:
                duration = time.time() - start_time
                tprint_performance(f"⏱️ {operation_name}: {duration:.2f}s")
        
        return wrapper
    return decorator


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_comprehensive_tools_integration(
    base_step_instance, 
    config: Optional[ComprehensiveToolsConfig] = None
) -> ComprehensiveToolsIntegration:
    """
    Create comprehensive tools integration instance.
    
    Args:
        base_step_instance: BaseStep instance
        config: Configuration
        
    Returns:
        Comprehensive tools integration instance
    """
    return ComprehensiveToolsIntegration(base_step_instance, config)


def get_comprehensive_tools_status(base_step_instance) -> Dict[str, Any]:
    """
    Get comprehensive tools status.
    
    Args:
        base_step_instance: BaseStep instance
        
    Returns:
        Tools status dictionary
    """
    return {
        'utility_availability': base_step_instance._get_availability_status(),
        'performance_metrics': base_step_instance._get_performance_metrics(),
        'memory_analytics': base_step_instance._get_memory_analytics(),
        'comprehensive_stats': base_step_instance._get_comprehensive_stats()
    }


def print_comprehensive_tools_help(base_step_instance) -> None:
    """
    Print comprehensive tools help.
    
    Args:
        base_step_instance: BaseStep instance
    """
    base_step_instance._print_utility_help()