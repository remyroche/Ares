"""
Comprehensive logging configuration for features_common.

This module provides centralized logging configuration with extensive
tprint integration and error handling to ensure no silent failures.
"""

import logging
import sys
from typing import Optional, Dict, Any
from contextlib import contextmanager

# Import tprint utilities
try:
    from src.utils.tprint import tprint
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs):
        print(*args)

class FeaturesCommonLogger:
    """
    Centralized logger for features_common with extensive logging and error handling.
    """
    
    def __init__(self, name: str = "features_common", level: int = logging.INFO):
        """Initialize the logger."""
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(console_handler)
        
        # Statistics
        self.stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'warnings_issued': 0,
            'errors_handled': 0
        }
    
    def log_operation_start(self, operation_name: str, **kwargs) -> None:
        """Log the start of an operation."""
        self.stats['total_operations'] += 1
        
        if TPRINT_AVAILABLE:
            tprint(f"🔧 [{self.name}] Starting {operation_name}", color="cyan")
            if kwargs:
                tprint(f"   Parameters: {kwargs}", color="blue")
        else:
            self.logger.info(f"Starting {operation_name} with parameters: {kwargs}")
    
    def log_operation_success(self, operation_name: str, result_info: Optional[str] = None) -> None:
        """Log successful operation completion."""
        self.stats['successful_operations'] += 1
        
        if TPRINT_AVAILABLE:
            tprint(f"✅ [{self.name}] {operation_name} completed successfully", color="green")
            if result_info:
                tprint(f"   Result: {result_info}", color="green")
        else:
            self.logger.info(f"{operation_name} completed successfully. {result_info or ''}")
    
    def log_operation_failure(self, operation_name: str, error: Exception, 
                             fallback_attempted: bool = False) -> None:
        """Log operation failure."""
        self.stats['failed_operations'] += 1
        
        if TPRINT_AVAILABLE:
            tprint(f"❌ [{self.name}] {operation_name} failed: {error}", color="red")
            if fallback_attempted:
                tprint(f"   Fallback attempted", color="yellow")
        else:
            self.logger.error(f"{operation_name} failed: {error}")
            if fallback_attempted:
                self.logger.warning("Fallback attempted")
    
    def log_warning(self, message: str, component: str = "unknown") -> None:
        """Log a warning message."""
        self.stats['warnings_issued'] += 1
        
        if TPRINT_AVAILABLE:
            tprint(f"⚠️  [{self.name}] [{component}] {message}", color="yellow")
        else:
            self.logger.warning(f"[{component}] {message}")
    
    def log_error(self, message: str, component: str = "unknown") -> None:
        """Log an error message."""
        self.stats['errors_handled'] += 1
        
        if TPRINT_AVAILABLE:
            tprint(f"❌ [{self.name}] [{component}] {message}", color="red")
        else:
            self.logger.error(f"[{component}] {message}")
    
    def log_info(self, message: str, component: str = "unknown") -> None:
        """Log an info message."""
        if TPRINT_AVAILABLE:
            tprint(f"ℹ️  [{self.name}] [{component}] {message}", color="blue")
        else:
            self.logger.info(f"[{component}] {message}")
    
    def log_debug(self, message: str, component: str = "unknown") -> None:
        """Log a debug message."""
        if TPRINT_AVAILABLE:
            tprint(f"🔍 [{self.name}] [{component}] {message}", color="magenta")
        else:
            self.logger.debug(f"[{component}] {message}")
    
    def log_performance(self, operation_name: str, execution_time: float, 
                       optimization_used: str = "none") -> None:
        """Log performance metrics."""
        if TPRINT_AVAILABLE:
            tprint(f"📊 [{self.name}] {operation_name}: {execution_time:.4f}s ({optimization_used})", color="green")
        else:
            self.logger.info(f"{operation_name}: {execution_time:.4f}s ({optimization_used})")
    
    def log_validation(self, data_name: str, is_valid: bool, warnings: list) -> None:
        """Log validation results."""
        if is_valid:
            if warnings:
                if TPRINT_AVAILABLE:
                    tprint(f"✅ [{self.name}] {data_name} validation passed with warnings: {warnings}", color="yellow")
                else:
                    self.logger.warning(f"{data_name} validation passed with warnings: {warnings}")
            else:
                if TPRINT_AVAILABLE:
                    tprint(f"✅ [{self.name}] {data_name} validation passed", color="green")
                else:
                    self.logger.info(f"{data_name} validation passed")
        else:
            if TPRINT_AVAILABLE:
                tprint(f"❌ [{self.name}] {data_name} validation failed: {warnings}", color="red")
            else:
                self.logger.error(f"{data_name} validation failed: {warnings}")
    
    def log_optimization_decision(self, strategy: str, reason: str, data_size: int) -> None:
        """Log optimization strategy decision."""
        if TPRINT_AVAILABLE:
            tprint(f"🚀 [{self.name}] Optimization strategy: {strategy} (reason: {reason}, data_size: {data_size})", color="green")
        else:
            self.logger.info(f"Optimization strategy: {strategy} (reason: {reason}, data_size: {data_size})")
    
    def log_cache_operation(self, operation: str, cache_key: str, hit: bool = None) -> None:
        """Log cache operations."""
        if hit is True:
            if TPRINT_AVAILABLE:
                tprint(f"💾 [{self.name}] Cache HIT for {operation} (key: {cache_key[:8]}...)", color="green")
            else:
                self.logger.debug(f"Cache HIT for {operation} (key: {cache_key[:8]}...)")
        elif hit is False:
            if TPRINT_AVAILABLE:
                tprint(f"💾 [{self.name}] Cache MISS for {operation} (key: {cache_key[:8]}...)", color="yellow")
            else:
                self.logger.debug(f"Cache MISS for {operation} (key: {cache_key[:8]}...)")
        else:
            if TPRINT_AVAILABLE:
                tprint(f"💾 [{self.name}] Cache {operation} (key: {cache_key[:8]}...)", color="blue")
            else:
                self.logger.debug(f"Cache {operation} (key: {cache_key[:8]}...)")
    
    def log_vectorbt_operation(self, operation: str, data_size: int, success: bool, 
                              fallback_used: bool = False) -> None:
        """Log VectorBT operations."""
        if success:
            if TPRINT_AVAILABLE:
                tprint(f"🚀 [{self.name}] VectorBT {operation} successful (data_size: {data_size})", color="green")
            else:
                self.logger.info(f"VectorBT {operation} successful (data_size: {data_size})")
        else:
            if fallback_used:
                if TPRINT_AVAILABLE:
                    tprint(f"⚠️  [{self.name}] VectorBT {operation} failed, using fallback (data_size: {data_size})", color="yellow")
                else:
                    self.logger.warning(f"VectorBT {operation} failed, using fallback (data_size: {data_size})")
            else:
                if TPRINT_AVAILABLE:
                    tprint(f"❌ [{self.name}] VectorBT {operation} failed (data_size: {data_size})", color="red")
                else:
                    self.logger.error(f"VectorBT {operation} failed (data_size: {data_size})")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        stats = self.stats.copy()
        
        # Calculate success rate
        if stats['total_operations'] > 0:
            stats['success_rate'] = stats['successful_operations'] / stats['total_operations']
            stats['failure_rate'] = stats['failed_operations'] / stats['total_operations']
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset logging statistics."""
        self.stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'warnings_issued': 0,
            'errors_handled': 0
        }
    
    @contextmanager
    def operation_context(self, operation_name: str, **kwargs):
        """Context manager for operation logging."""
        self.log_operation_start(operation_name, **kwargs)
        start_time = None
        
        try:
            import time
            start_time = time.time()
            yield self
        except Exception as e:
            self.log_operation_failure(operation_name, e)
            raise
        else:
            if start_time:
                execution_time = time.time() - start_time
                self.log_performance(operation_name, execution_time)
            self.log_operation_success(operation_name)


# Global logger instance
_global_logger: Optional[FeaturesCommonLogger] = None

def get_logger(name: str = "features_common") -> FeaturesCommonLogger:
    """Get the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = FeaturesCommonLogger(name)
    return _global_logger

def set_logger(logger: FeaturesCommonLogger) -> None:
    """Set the global logger instance."""
    global _global_logger
    _global_logger = logger

def reset_logger() -> None:
    """Reset the global logger instance."""
    global _global_logger
    _global_logger = None

def log_operation(operation_name: str, **kwargs):
    """Decorator for automatic operation logging."""
    def decorator(func):
        def wrapper(*args, **func_kwargs):
            logger = get_logger()
            with logger.operation_context(operation_name, **kwargs):
                return func(*args, **func_kwargs)
        return wrapper
    return decorator