#!/usr/bin/env python3
"""
Optimisation Pipeline Decorators

Comprehensive decorators for the optimisation pipeline with:
- Data protection and validation
- Error handling and recovery
- Performance monitoring
- Operation logging and auditing
"""

import asyncio
import functools
import json
import logging
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.pipeline_protection_framework import (
    PipelineProtectionDecorator,
    OperationType,
    ValidationLevel,
    DataValidator,
    PipelineStateManager,
    PipelineMonitor,
    get_pipeline_protection,
    get_state_manager,
    get_monitor
)
from src.utils.common_operations import (
    ensure_directory,
    safe_file_exists,
    safe_json_dump,
    safe_json_load,
    format_datetime,
    get_current_datetime
)


class OptimisationDecorators:
    """Collection of decorators for optimisation pipeline operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("OptimisationDecorators")
        self.data_validator = DataValidator(config)
        
    def data_protection(self, 
                       validation_level: ValidationLevel = ValidationLevel.STANDARD,
                       backup_enabled: bool = True,
                       checksum_validation: bool = True):
        """Decorator for data protection operations."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                operation_name = f"{func.__module__}.{func.__name__}"
                start_time = time.time()
                
                self.logger.info(f"🛡️ Starting data protection for: {operation_name}")
                
                # Pre-operation data validation
                pre_validation = await self._validate_input_data(args, kwargs, validation_level)
                if not pre_validation["passed"]:
                    self.logger.error(f"❌ Pre-operation data validation failed: {pre_validation['error']}")
                    raise ValueError(f"Data validation failed: {pre_validation['error']}")
                
                # Create backup if enabled
                backup_paths = []
                if backup_enabled:
                    backup_paths = await self._create_data_backups(args, kwargs)
                
                try:
                    # Execute operation
                    result = await func(*args, **kwargs)
                    
                    # Post-operation data validation
                    post_validation = await self._validate_output_data(result, validation_level)
                    if not post_validation["passed"]:
                        self.logger.error(f"❌ Post-operation data validation failed: {post_validation['error']}")
                        # Restore from backup if validation failed
                        if backup_paths:
                            await self._restore_from_backups(backup_paths)
                        raise ValueError(f"Output data validation failed: {post_validation['error']}")
                    
                    # Validate checksums if enabled
                    if checksum_validation:
                        checksum_validation_result = await self._validate_checksums(result)
                        if not checksum_validation_result["passed"]:
                            self.logger.warning(f"⚠️ Checksum validation failed: {checksum_validation_result['error']}")
                    
                    duration = time.time() - start_time
                    self.logger.info(f"✅ Data protection completed for {operation_name} ({duration:.2f}s)")
                    
                    return result
                    
                except Exception as e:
                    # Restore from backup on error
                    if backup_paths:
                        await self._restore_from_backups(backup_paths)
                        self.logger.info("🔄 Restored data from backup after error")
                    
                    duration = time.time() - start_time
                    self.logger.error(f"❌ Data protection failed for {operation_name} ({duration:.2f}s): {str(e)}")
                    raise
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def error_handling(self, 
                      retry_count: int = 3,
                      retry_delay: float = 1.0,
                      exponential_backoff: bool = True,
                      critical_errors: Optional[List[str]] = None):
        """Decorator for error handling and recovery."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                operation_name = f"{func.__module__}.{func.__name__}"
                last_exception = None
                
                for attempt in range(retry_count + 1):
                    try:
                        if attempt > 0:
                            delay = retry_delay * (2 ** attempt) if exponential_backoff else retry_delay
                            self.logger.info(f"🔄 Retry attempt {attempt}/{retry_count} for {operation_name} (delay: {delay}s)")
                            await asyncio.sleep(delay)
                        
                        result = await func(*args, **kwargs)
                        
                        if attempt > 0:
                            self.logger.info(f"✅ Operation succeeded on retry {attempt} for {operation_name}")
                        
                        return result
                        
                    except Exception as e:
                        last_exception = e
                        error_type = type(e).__name__
                        error_message = str(e)
                        
                        # Check if this is a critical error that shouldn't be retried
                        if critical_errors and any(critical in error_message for critical in critical_errors):
                            self.logger.error(f"💥 Critical error in {operation_name}: {error_message}")
                            break
                        
                        self.logger.error(f"❌ Attempt {attempt + 1} failed for {operation_name}: {error_type}: {error_message}")
                        
                        if attempt == retry_count:
                            self.logger.error(f"💥 All retry attempts failed for {operation_name}")
                            break
                
                # Log final failure
                self.logger.exception(f"Final failure for {operation_name}")
                raise last_exception
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def performance_monitoring(self, 
                             log_performance: bool = True,
                             alert_threshold: float = 60.0,
                             memory_monitoring: bool = True):
        """Decorator for performance monitoring."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                operation_name = f"{func.__module__}.{func.__name__}"
                start_time = time.time()
                start_memory = self._get_memory_usage() if memory_monitoring else 0
                
                try:
                    result = await func(*args, **kwargs)
                    
                    duration = time.time() - start_time
                    end_memory = self._get_memory_usage() if memory_monitoring else 0
                    memory_delta = end_memory - start_memory
                    
                    # Log performance metrics
                    if log_performance:
                        self.logger.info(f"📊 Performance metrics for {operation_name}:")
                        self.logger.info(f"   Duration: {duration:.2f}s")
                        if memory_monitoring:
                            self.logger.info(f"   Memory usage: {memory_delta:.2f}MB")
                    
                    # Record metrics
                    monitor = get_monitor()
                    monitor.record_metric("operation_duration", duration, {"operation": operation_name})
                    if memory_monitoring:
                        monitor.record_metric("memory_usage", memory_delta, {"operation": operation_name})
                    
                    # Check for performance alerts
                    if duration > alert_threshold:
                        self.logger.warning(f"⚠️ Performance alert: {operation_name} took {duration:.2f}s (threshold: {alert_threshold}s)")
                        monitor.record_metric("performance_alerts", 1, {"operation": operation_name, "type": "duration"})
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    self.logger.error(f"❌ Performance monitoring for {operation_name} failed after {duration:.2f}s: {str(e)}")
                    
                    # Record error metrics
                    monitor = get_monitor()
                    monitor.record_metric("operation_errors", 1, {"operation": operation_name, "error": type(e).__name__})
                    
                    raise
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def operation_logging(self, 
                         log_level: str = "INFO",
                         include_args: bool = True,
                         include_result: bool = False,
                         audit_trail: bool = True):
        """Decorator for operation logging and auditing."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                operation_name = f"{func.__module__}.{func.__name__}"
                start_time = get_current_datetime()
                
                # Log operation start
                log_message = f"🚀 Starting operation: {operation_name}"
                if include_args and (args or kwargs):
                    log_message += f" with args: {args}, kwargs: {kwargs}"
                
                getattr(self.logger, log_level.lower())(log_message)
                
                # Create audit entry
                audit_entry = {
                    "operation": operation_name,
                    "start_time": start_time.isoformat(),
                    "args": str(args) if include_args else "hidden",
                    "kwargs": str(kwargs) if include_args else "hidden"
                }
                
                try:
                    result = await func(*args, **kwargs)
                    
                    end_time = get_current_datetime()
                    duration = (end_time - start_time).total_seconds()
                    
                    # Log operation completion
                    completion_message = f"✅ Completed operation: {operation_name} ({duration:.2f}s)"
                    if include_result:
                        completion_message += f" with result: {result}"
                    
                    getattr(self.logger, log_level.lower())(completion_message)
                    
                    # Update audit entry
                    audit_entry.update({
                        "end_time": end_time.isoformat(),
                        "duration": duration,
                        "success": True,
                        "result": str(result) if include_result else "hidden"
                    })
                    
                    return result
                    
                except Exception as e:
                    end_time = get_current_datetime()
                    duration = (end_time - start_time).total_seconds()
                    
                    # Log operation failure
                    failure_message = f"❌ Failed operation: {operation_name} ({duration:.2f}s): {str(e)}"
                    getattr(self.logger, log_level.lower())(failure_message)
                    
                    # Update audit entry
                    audit_entry.update({
                        "end_time": end_time.isoformat(),
                        "duration": duration,
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__
                    })
                    
                    raise
                
                finally:
                    # Save audit entry
                    if audit_trail:
                        await self._save_audit_entry(audit_entry)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    async def _validate_input_data(self, args: tuple, kwargs: dict, validation_level: ValidationLevel) -> Dict[str, Any]:
        """Validate input data."""
        try:
            # Check for DataFrame arguments
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    validation = self.data_validator.validate_dataframe(arg)
                    if not validation.passed:
                        return {"passed": False, "error": f"Input DataFrame validation failed: {validation}"}
            
            # Check for DataFrame in kwargs
            for key, value in kwargs.items():
                if isinstance(value, pd.DataFrame):
                    validation = self.data_validator.validate_dataframe(value)
                    if not validation.passed:
                        return {"passed": False, "error": f"Input DataFrame '{key}' validation failed: {validation}"}
            
            return {"passed": True, "message": "Input data validation passed"}
            
        except Exception as e:
            return {"passed": False, "error": f"Input data validation error: {str(e)}"}
    
    async def _validate_output_data(self, result: Any, validation_level: ValidationLevel) -> Dict[str, Any]:
        """Validate output data."""
        try:
            if isinstance(result, pd.DataFrame):
                validation = self.data_validator.validate_dataframe(result)
                if not validation.passed:
                    return {"passed": False, "error": f"Output DataFrame validation failed: {validation}"}
            
            elif isinstance(result, dict):
                if not result:
                    return {"passed": False, "error": "Output dictionary is empty"}
                
                # Check for error indicators
                if "error" in result or "errors" in result:
                    return {"passed": False, "error": "Output contains error indicators"}
            
            return {"passed": True, "message": "Output data validation passed"}
            
        except Exception as e:
            return {"passed": False, "error": f"Output data validation error: {str(e)}"}
    
    async def _create_data_backups(self, args: tuple, kwargs: dict) -> List[str]:
        """Create backups of data files."""
        backup_paths = []
        
        try:
            backup_dir = Path("data_cache/backups") / format_datetime(get_current_datetime(), "%Y%m%d_%H%M%S")
            ensure_directory(backup_dir)
            
            # Backup file paths from args and kwargs
            file_paths = []
            for arg in args:
                if isinstance(arg, (str, Path)) and str(arg).endswith(('.parquet', '.pkl', '.json')):
                    file_paths.append(str(arg))
            
            for value in kwargs.values():
                if isinstance(value, (str, Path)) and str(value).endswith(('.parquet', '.pkl', '.json')):
                    file_paths.append(str(value))
            
            # Create backups
            for file_path in file_paths:
                if safe_file_exists(file_path):
                    backup_path = backup_dir / Path(file_path).name
                    import shutil
                    shutil.copy2(file_path, backup_path)
                    backup_paths.append(str(backup_path))
            
            if backup_paths:
                self.logger.info(f"💾 Created {len(backup_paths)} data backups in {backup_dir}")
            
        except Exception as e:
            self.logger.exception(f"Error creating data backups: {e}")
        
        return backup_paths
    
    async def _restore_from_backups(self, backup_paths: List[str]) -> None:
        """Restore data from backups."""
        try:
            for backup_path in backup_paths:
                if safe_file_exists(backup_path):
                    # Determine original path (remove backup timestamp)
                    original_path = str(Path(backup_path).parent.parent / Path(backup_path).name)
                    import shutil
                    shutil.copy2(backup_path, original_path)
                    self.logger.info(f"🔄 Restored {original_path} from backup")
            
        except Exception as e:
            self.logger.exception(f"Error restoring from backups: {e}")
    
    async def _validate_checksums(self, result: Any) -> Dict[str, Any]:
        """Validate data checksums."""
        try:
            if isinstance(result, pd.DataFrame):
                # Calculate checksum
                df_str = result.to_string()
                checksum = hashlib.md5(df_str.encode()).hexdigest()
                
                # Store checksum for future validation
                checksum_file = Path("data_cache/checksums.json")
                checksums = safe_json_load(checksum_file) if safe_file_exists(checksum_file) else {}
                checksums[f"dataframe_{id(result)}"] = checksum
                safe_json_dump(checksums, checksum_file)
                
                return {"passed": True, "checksum": checksum}
            
            return {"passed": True, "message": "No checksum validation needed"}
            
        except Exception as e:
            return {"passed": False, "error": f"Checksum validation error: {str(e)}"}
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    async def _save_audit_entry(self, audit_entry: Dict[str, Any]) -> None:
        """Save audit entry to file."""
        try:
            audit_file = Path("data_cache/audit_trail.json")
            audit_trail = []
            
            if safe_file_exists(audit_file):
                audit_trail = safe_json_load(audit_file)
            
            audit_trail.append(audit_entry)
            
            # Keep only last 1000 entries
            if len(audit_trail) > 1000:
                audit_trail = audit_trail[-1000:]
            
            safe_json_dump(audit_trail, audit_file)
            
        except Exception as e:
            self.logger.exception(f"Error saving audit entry: {e}")


# Convenience decorators
def data_protection(validation_level: ValidationLevel = ValidationLevel.STANDARD, 
                   backup_enabled: bool = True,
                   checksum_validation: bool = True):
    """Convenience decorator for data protection."""
    decorators = OptimisationDecorators({})
    return decorators.data_protection(validation_level, backup_enabled, checksum_validation)


def error_handling(retry_count: int = 3, 
                  retry_delay: float = 1.0,
                  exponential_backoff: bool = True,
                  critical_errors: Optional[List[str]] = None):
    """Convenience decorator for error handling."""
    decorators = OptimisationDecorators({})
    return decorators.error_handling(retry_count, retry_delay, exponential_backoff, critical_errors)


def performance_monitoring(log_performance: bool = True,
                          alert_threshold: float = 60.0,
                          memory_monitoring: bool = True):
    """Convenience decorator for performance monitoring."""
    decorators = OptimisationDecorators({})
    return decorators.performance_monitoring(log_performance, alert_threshold, memory_monitoring)


def operation_logging(log_level: str = "INFO",
                     include_args: bool = True,
                     include_result: bool = False,
                     audit_trail: bool = True):
    """Convenience decorator for operation logging."""
    decorators = OptimisationDecorators({})
    return decorators.operation_logging(log_level, include_args, include_result, audit_trail)


# Combined decorators for common use cases
def protect_optimisation_operation(validation_level: ValidationLevel = ValidationLevel.CRITICAL,
                                  retry_count: int = 3,
                                  backup_enabled: bool = True):
    """Combined decorator for optimisation operations."""
    def decorator(func: Callable) -> Callable:
        # Apply multiple decorators
        func = data_protection(validation_level, backup_enabled)(func)
        func = error_handling(retry_count)(func)
        func = performance_monitoring()(func)
        func = operation_logging()(func)
        return func
    return decorator


def protect_data_operation(validation_level: ValidationLevel = ValidationLevel.STANDARD,
                          backup_enabled: bool = True):
    """Combined decorator for data operations."""
    def decorator(func: Callable) -> Callable:
        # Apply multiple decorators
        func = data_protection(validation_level, backup_enabled)(func)
        func = error_handling(retry_count=2)(func)
        func = operation_logging()(func)
        return func
    return decorator