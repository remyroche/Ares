#!/usr/bin/env python3
"""
Enhanced Pipeline Decorators

This module provides comprehensive decorators for the market analysis pipeline,
including data formatting, analysis protection, and access control decorators.
"""

import asyncio
import functools
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Core decorators
from src.core.decorators import (
    handles_errors,
    traced,
    validates,
    cached,
    log_execution_time,
    timeout,
    retry,
    circuit_breaker,
    audit_log,
    get_correlation_id,
    set_correlation_id,
)
from src.utils.common_operations import (
    get_current_datetime,
    format_datetime,
    ensure_directory,
    safe_json_dump,
    safe_json_load,
    validate_dataframe,
    validate_data_quality,
    get_logger,
    timed_operation,
)


class DataFormattingDecorator:
    """Decorator for ensuring proper data formatting and validation."""
    
    def __init__(self, 
                 required_columns: Optional[List[str]] = None,
                 data_types: Optional[Dict[str, type]] = None,
                 validation_rules: Optional[Dict[str, Any]] = None):
        """
        Initialize data formatting decorator.
        
        Args:
            required_columns: List of required column names
            data_types: Dictionary mapping column names to expected types
            validation_rules: Dictionary of validation rules
        """
        self.required_columns = required_columns or []
        self.data_types = data_types or {}
        self.validation_rules = validation_rules or {}
        self.logger = get_logger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Apply data formatting decorator to function."""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract data from arguments
            data = self._extract_data_from_args(args, kwargs)
            
            if data is not None:
                # Validate and format data
                formatted_data = await self._validate_and_format_data(data)
                
                # Replace data in arguments
                args, kwargs = self._replace_data_in_args(args, kwargs, formatted_data)
            
            # Execute function
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Extract data from arguments
            data = self._extract_data_from_args(args, kwargs)
            
            if data is not None:
                # Validate and format data
                formatted_data = self._validate_and_format_data_sync(data)
                
                # Replace data in arguments
                args, kwargs = self._replace_data_in_args(args, kwargs, formatted_data)
            
            # Execute function
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    def _extract_data_from_args(self, args: Tuple, kwargs: Dict) -> Optional[Any]:
        """Extract data from function arguments."""
        # Look for common data parameter names
        data_params = ['data', 'df', 'dataframe', 'input_data', 'price_data', 'volume_data']
        
        for param in data_params:
            if param in kwargs:
                return kwargs[param]
        
        # Look in positional arguments
        for arg in args:
            if hasattr(arg, 'columns') or hasattr(arg, 'shape'):  # Likely a DataFrame
                return arg
        
        return None
    
    def _replace_data_in_args(self, args: Tuple, kwargs: Dict, new_data: Any) -> Tuple[Tuple, Dict]:
        """Replace data in function arguments."""
        # Replace in kwargs
        for param in ['data', 'df', 'dataframe', 'input_data', 'price_data', 'volume_data']:
            if param in kwargs:
                kwargs[param] = new_data
                break
        
        # Replace in positional arguments (if it's a DataFrame-like object)
        new_args = []
        for arg in args:
            if hasattr(arg, 'columns') or hasattr(arg, 'shape'):
                new_args.append(new_data)
            else:
                new_args.append(arg)
        
        return tuple(new_args), kwargs
    
    async def _validate_and_format_data(self, data: Any) -> Any:
        """Validate and format data asynchronously."""
        try:
            import pandas as pd
            
            if isinstance(data, pd.DataFrame):
                # Validate required columns
                if self.required_columns:
                    missing_columns = set(self.required_columns) - set(data.columns)
                    if missing_columns:
                        raise ValueError(f"Missing required columns: {missing_columns}")
                
                # Validate data types
                for column, expected_type in self.data_types.items():
                    if column in data.columns:
                        if not isinstance(data[column].dtype, expected_type):
                            self.logger.warning(f"Column {column} has unexpected type: {data[column].dtype}")
                
                # Apply validation rules
                for rule_name, rule_config in self.validation_rules.items():
                    await self._apply_validation_rule(data, rule_name, rule_config)
                
                # Format data
                formatted_data = await self._format_dataframe(data)
                
                return formatted_data
            
            return data
            
        except Exception as e:
            self.logger.exception(f"Data validation failed: {e}")
            raise
    
    def _validate_and_format_data_sync(self, data: Any) -> Any:
        """Validate and format data synchronously."""
        try:
            import pandas as pd
            
            if isinstance(data, pd.DataFrame):
                # Validate required columns
                if self.required_columns:
                    missing_columns = set(self.required_columns) - set(data.columns)
                    if missing_columns:
                        raise ValueError(f"Missing required columns: {missing_columns}")
                
                # Validate data types
                for column, expected_type in self.data_types.items():
                    if column in data.columns:
                        if not isinstance(data[column].dtype, expected_type):
                            self.logger.warning(f"Column {column} has unexpected type: {data[column].dtype}")
                
                # Apply validation rules
                for rule_name, rule_config in self.validation_rules.items():
                    self._apply_validation_rule_sync(data, rule_name, rule_config)
                
                # Format data
                formatted_data = self._format_dataframe_sync(data)
                
                return formatted_data
            
            return data
            
        except Exception as e:
            self.logger.exception(f"Data validation failed: {e}")
            raise
    
    async def _apply_validation_rule(self, data: Any, rule_name: str, rule_config: Dict[str, Any]) -> None:
        """Apply a validation rule to data."""
        if rule_name == 'no_nan_ratio':
            max_ratio = rule_config.get('max_ratio', 0.1)
            nan_ratios = data.isna().sum() / len(data)
            high_nan_cols = nan_ratios[nan_ratios > max_ratio]
            if not high_nan_cols.empty:
                self.logger.warning(f"High NaN ratio in columns: {high_nan_cols.to_dict()}")
        
        elif rule_name == 'numeric_range':
            column = rule_config.get('column')
            min_val = rule_config.get('min_val')
            max_val = rule_config.get('max_val')
            if column and column in data.columns:
                out_of_range = (data[column] < min_val) | (data[column] > max_val)
                if out_of_range.any():
                    self.logger.warning(f"Values out of range in column {column}: {out_of_range.sum()} rows")
    
    def _apply_validation_rule_sync(self, data: Any, rule_name: str, rule_config: Dict[str, Any]) -> None:
        """Apply a validation rule to data synchronously."""
        if rule_name == 'no_nan_ratio':
            max_ratio = rule_config.get('max_ratio', 0.1)
            nan_ratios = data.isna().sum() / len(data)
            high_nan_cols = nan_ratios[nan_ratios > max_ratio]
            if not high_nan_cols.empty:
                self.logger.warning(f"High NaN ratio in columns: {high_nan_cols.to_dict()}")
        
        elif rule_name == 'numeric_range':
            column = rule_config.get('column')
            min_val = rule_config.get('min_val')
            max_val = rule_config.get('max_val')
            if column and column in data.columns:
                out_of_range = (data[column] < min_val) | (data[column] > max_val)
                if out_of_range.any():
                    self.logger.warning(f"Values out of range in column {column}: {out_of_range.sum()} rows")
    
    async def _format_dataframe(self, df: Any) -> Any:
        """Format DataFrame with standard operations."""
        try:
            import pandas as pd
            
            # Ensure proper index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                elif 'datetime' in df.columns:
                    df = df.set_index('datetime')
            
            # Sort by index
            df = df.sort_index()
            
            # Remove duplicates
            df = df.drop_duplicates()
            
            return df
            
        except Exception as e:
            self.logger.warning(f"DataFrame formatting failed: {e}")
            return df
    
    def _format_dataframe_sync(self, df: Any) -> Any:
        """Format DataFrame with standard operations synchronously."""
        try:
            import pandas as pd
            
            # Ensure proper index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                elif 'datetime' in df.columns:
                    df = df.set_index('datetime')
            
            # Sort by index
            df = df.sort_index()
            
            # Remove duplicates
            df = df.drop_duplicates()
            
            return df
            
        except Exception as e:
            self.logger.warning(f"DataFrame formatting failed: {e}")
            return df


class DataAnalysisProtectionDecorator:
    """Decorator for protecting data analysis operations."""
    
    def __init__(self, 
                 max_memory_mb: Optional[int] = None,
                 max_execution_time: Optional[int] = None,
                 allowed_operations: Optional[List[str]] = None,
                 forbidden_operations: Optional[List[str]] = None):
        """
        Initialize data analysis protection decorator.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            max_execution_time: Maximum execution time in seconds
            allowed_operations: List of allowed operations
            forbidden_operations: List of forbidden operations
        """
        self.max_memory_mb = max_memory_mb
        self.max_execution_time = max_execution_time
        self.allowed_operations = allowed_operations or []
        self.forbidden_operations = forbidden_operations or []
        self.logger = get_logger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Apply data analysis protection decorator to function."""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Check operation permissions
            if not self._check_operation_permissions(func.__name__):
                raise PermissionError(f"Operation {func.__name__} is not allowed")
            
            # Monitor memory usage
            memory_monitor = self._start_memory_monitoring()
            
            try:
                # Execute function with timeout if specified
                if self.max_execution_time:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.max_execution_time
                    )
                else:
                    result = await func(*args, **kwargs)
                
                # Check memory usage
                memory_usage = self._get_memory_usage(memory_monitor)
                if self.max_memory_mb and memory_usage > self.max_memory_mb:
                    self.logger.warning(f"High memory usage: {memory_usage}MB > {self.max_memory_mb}MB")
                
                return result
                
            except asyncio.TimeoutError:
                self.logger.error(f"Function {func.__name__} timed out after {self.max_execution_time}s")
                raise
            except Exception as e:
                self.logger.exception(f"Function {func.__name__} failed: {e}")
                raise
            finally:
                self._stop_memory_monitoring(memory_monitor)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Check operation permissions
            if not self._check_operation_permissions(func.__name__):
                raise PermissionError(f"Operation {func.__name__} is not allowed")
            
            # Monitor memory usage
            memory_monitor = self._start_memory_monitoring()
            
            try:
                # Execute function with timeout if specified
                if self.max_execution_time:
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"Function {func.__name__} timed out after {self.max_execution_time}s")
                    
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(self.max_execution_time)
                    
                    try:
                        result = func(*args, **kwargs)
                    finally:
                        signal.alarm(0)
                else:
                    result = func(*args, **kwargs)
                
                # Check memory usage
                memory_usage = self._get_memory_usage(memory_monitor)
                if self.max_memory_mb and memory_usage > self.max_memory_mb:
                    self.logger.warning(f"High memory usage: {memory_usage}MB > {self.max_memory_mb}MB")
                
                return result
                
            except Exception as e:
                self.logger.exception(f"Function {func.__name__} failed: {e}")
                raise
            finally:
                self._stop_memory_monitoring(memory_monitor)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    def _check_operation_permissions(self, operation_name: str) -> bool:
        """Check if operation is allowed."""
        # Check forbidden operations first
        if self.forbidden_operations and operation_name in self.forbidden_operations:
            return False
        
        # Check allowed operations
        if self.allowed_operations and operation_name not in self.allowed_operations:
            return False
        
        return True
    
    def _start_memory_monitoring(self) -> Dict[str, Any]:
        """Start memory monitoring."""
        try:
            import psutil
            process = psutil.Process()
            return {
                'process': process,
                'start_memory': process.memory_info().rss / 1024 / 1024,  # MB
                'start_time': time.time(),
            }
        except ImportError:
            return {'process': None, 'start_memory': 0, 'start_time': time.time()}
    
    def _get_memory_usage(self, memory_monitor: Dict[str, Any]) -> float:
        """Get current memory usage in MB."""
        try:
            if memory_monitor['process']:
                return memory_monitor['process'].memory_info().rss / 1024 / 1024
        except Exception:
            pass
        return 0.0
    
    def _stop_memory_monitoring(self, memory_monitor: Dict[str, Any]) -> None:
        """Stop memory monitoring."""
        try:
            if memory_monitor['process']:
                end_memory = memory_monitor['process'].memory_info().rss / 1024 / 1024
                memory_delta = end_memory - memory_monitor['start_memory']
                self.logger.info(f"Memory usage delta: {memory_delta:.2f}MB")
        except Exception:
            pass


class DataAccessProtectionDecorator:
    """Decorator for protecting data access operations."""
    
    def __init__(self, 
                 allowed_paths: Optional[List[str]] = None,
                 forbidden_paths: Optional[List[str]] = None,
                 require_authentication: bool = False,
                 audit_access: bool = True):
        """
        Initialize data access protection decorator.
        
        Args:
            allowed_paths: List of allowed file paths
            forbidden_paths: List of forbidden file paths
            require_authentication: Whether to require authentication
            audit_access: Whether to audit access
        """
        self.allowed_paths = allowed_paths or []
        self.forbidden_paths = forbidden_paths or []
        self.require_authentication = require_authentication
        self.audit_access = audit_access
        self.logger = get_logger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Apply data access protection decorator to function."""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Check authentication
            if self.require_authentication and not self._check_authentication():
                raise PermissionError("Authentication required")
            
            # Check path permissions
            paths = self._extract_paths_from_args(args, kwargs)
            for path in paths:
                if not self._check_path_permissions(path):
                    raise PermissionError(f"Access denied to path: {path}")
            
            # Audit access
            if self.audit_access:
                self._audit_access(func.__name__, paths)
            
            # Execute function
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Check authentication
            if self.require_authentication and not self._check_authentication():
                raise PermissionError("Authentication required")
            
            # Check path permissions
            paths = self._extract_paths_from_args(args, kwargs)
            for path in paths:
                if not self._check_path_permissions(path):
                    raise PermissionError(f"Access denied to path: {path}")
            
            # Audit access
            if self.audit_access:
                self._audit_access(func.__name__, paths)
            
            # Execute function
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    def _check_authentication(self) -> bool:
        """Check if user is authenticated."""
        # This is a simplified authentication check
        # In a real implementation, you would check JWT tokens, session data, etc.
        return True  # For now, always allow
    
    def _extract_paths_from_args(self, args: Tuple, kwargs: Dict) -> List[str]:
        """Extract file paths from function arguments."""
        paths = []
        
        # Look for common path parameter names
        path_params = ['path', 'file_path', 'data_dir', 'output_dir', 'input_dir']
        
        for param in path_params:
            if param in kwargs:
                paths.append(str(kwargs[param]))
        
        # Look in positional arguments
        for arg in args:
            if isinstance(arg, (str, Path)):
                paths.append(str(arg))
        
        return paths
    
    def _check_path_permissions(self, path: str) -> bool:
        """Check if path access is allowed."""
        path_obj = Path(path)
        
        # Check forbidden paths first
        for forbidden_path in self.forbidden_paths:
            if path_obj.match(forbidden_path) or str(path_obj).startswith(forbidden_path):
                return False
        
        # Check allowed paths
        if self.allowed_paths:
            for allowed_path in self.allowed_paths:
                if path_obj.match(allowed_path) or str(path_obj).startswith(allowed_path):
                    return True
            return False
        
        return True
    
    def _audit_access(self, function_name: str, paths: List[str]) -> None:
        """Audit data access."""
        audit_info = {
            'function': function_name,
            'paths': paths,
            'timestamp': format_datetime(get_current_datetime()),
            'correlation_id': get_correlation_id(),
        }
        
        self.logger.info(f"Data access audit: {audit_info}")


# Convenience decorator functions
def data_formatting(
    required_columns: Optional[List[str]] = None,
    data_types: Optional[Dict[str, type]] = None,
    validation_rules: Optional[Dict[str, Any]] = None,
) -> Callable:
    """Decorator for data formatting and validation."""
    decorator = DataFormattingDecorator(required_columns, data_types, validation_rules)
    return decorator


def data_analysis_protection(
    max_memory_mb: Optional[int] = None,
    max_execution_time: Optional[int] = None,
    allowed_operations: Optional[List[str]] = None,
    forbidden_operations: Optional[List[str]] = None,
) -> Callable:
    """Decorator for data analysis protection."""
    decorator = DataAnalysisProtectionDecorator(
        max_memory_mb, max_execution_time, allowed_operations, forbidden_operations
    )
    return decorator


def data_access_protection(
    allowed_paths: Optional[List[str]] = None,
    forbidden_paths: Optional[List[str]] = None,
    require_authentication: bool = False,
    audit_access: bool = True,
) -> Callable:
    """Decorator for data access protection."""
    decorator = DataAccessProtectionDecorator(
        allowed_paths, forbidden_paths, require_authentication, audit_access
    )
    return decorator


# Combined decorator for comprehensive protection
def comprehensive_pipeline_protection(
    required_columns: Optional[List[str]] = None,
    data_types: Optional[Dict[str, type]] = None,
    validation_rules: Optional[Dict[str, Any]] = None,
    max_memory_mb: Optional[int] = None,
    max_execution_time: Optional[int] = None,
    allowed_operations: Optional[List[str]] = None,
    forbidden_operations: Optional[List[str]] = None,
    allowed_paths: Optional[List[str]] = None,
    forbidden_paths: Optional[List[str]] = None,
    require_authentication: bool = False,
    audit_access: bool = True,
) -> Callable:
    """Combined decorator for comprehensive pipeline protection."""
    def decorator(func: Callable) -> Callable:
        # Apply all decorators in sequence
        func = data_formatting(required_columns, data_types, validation_rules)(func)
        func = data_analysis_protection(
            max_memory_mb, max_execution_time, allowed_operations, forbidden_operations
        )(func)
        func = data_access_protection(
            allowed_paths, forbidden_paths, require_authentication, audit_access
        )(func)
        
        # Add core decorators
        func = handles_errors(Exception, fallback=False)(func)
        func = traced(operation_name=func.__name__)(func)
        func = log_execution_time(func)
        func = audit_log(operation=func.__name__)(func)
        
        return func
    
    return decorator


if __name__ == "__main__":
    # Example usage
    @data_formatting(
        required_columns=['open', 'high', 'low', 'close', 'volume'],
        validation_rules={'no_nan_ratio': {'max_ratio': 0.1}}
    )
    @data_analysis_protection(max_memory_mb=1000, max_execution_time=300)
    @data_access_protection(allowed_paths=['data_cache/*'])
    async def example_function(data, output_path):
        """Example function with comprehensive protection."""
        print(f"Processing data with shape: {data.shape}")
        print(f"Output path: {output_path}")
        return True
    
    # Test the decorators
    import pandas as pd
    
    test_data = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [103, 104, 105],
        'volume': [1000, 1100, 1200]
    })
    
    asyncio.run(example_function(test_data, "data_cache/test_output.parquet"))