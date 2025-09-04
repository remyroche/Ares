#!/usr/bin/env python3
"""
Enhanced Error Handler

This module provides comprehensive error handling utilities that integrate with
the existing decorator system and provide fallback mechanisms for pipeline operations.
"""

import asyncio
import logging
import traceback
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
import json
from pathlib import Path

from src.utils.common_operations import (
    format_datetime, get_current_datetime, safe_file_exists, 
    ensure_directory, safe_json_dump, safe_json_load
)
from src.core.decorators import (
    handles_errors, error_boundary, timeout, retry
)
from src.core.domain.decorators import (
    monitor_step_execution, ensure_data_integrity
)

logger = logging.getLogger(__name__)

class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    async def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if this strategy can recover from the given error."""
        raise NotImplementedError
    
    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Attempt to recover from the error."""
        raise NotImplementedError

class RetryStrategy(ErrorRecoveryStrategy):
    """Retry strategy for transient errors."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("RetryStrategy", config)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.backoff_factor = config.get('backoff_factor', 2.0)
    
    async def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if error is retryable."""
        retryable_errors = [
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            OSError
        ]
        
        # Check if error type is retryable
        if any(isinstance(error, error_type) for error_type in retryable_errors):
            return True
        
        # Check if we haven't exceeded max retries
        retry_count = context.get('retry_count', 0)
        return retry_count < self.max_retries
    
    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Retry the operation with exponential backoff."""
        retry_count = context.get('retry_count', 0)
        delay = self.retry_delay * (self.backoff_factor ** retry_count)
        
        self.logger.info(f"🔄 Retrying operation (attempt {retry_count + 1}/{self.max_retries}) after {delay}s delay")
        await asyncio.sleep(delay)
        
        # Update context
        context['retry_count'] = retry_count + 1
        return context

class FallbackStrategy(ErrorRecoveryStrategy):
    """Fallback strategy for non-recoverable errors."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("FallbackStrategy", config)
        self.fallback_value = config.get('fallback_value', None)
        self.fallback_function = config.get('fallback_function', None)
    
    async def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Fallback can always be attempted."""
        return True
    
    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Execute fallback strategy."""
        self.logger.info("🔄 Executing fallback strategy")
        
        if self.fallback_function and callable(self.fallback_function):
            try:
                if asyncio.iscoroutinefunction(self.fallback_function):
                    return await self.fallback_function(context)
                else:
                    return self.fallback_function(context)
            except Exception as fallback_error:
                self.logger.exception(f"❌ Fallback function failed: {fallback_error}")
                return self.fallback_value
        
        return self.fallback_value

class DataRecoveryStrategy(ErrorRecoveryStrategy):
    """Data recovery strategy for data-related errors."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("DataRecoveryStrategy", config)
        self.backup_paths = config.get('backup_paths', [])
        self.data_repair_methods = config.get('data_repair_methods', [])
    
    async def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if this is a data-related error."""
        data_errors = [
            FileNotFoundError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            ValueError  # For data validation errors
        ]
        
        return any(isinstance(error, error_type) for error_type in data_errors)
    
    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Attempt data recovery."""
        self.logger.info("🔄 Attempting data recovery")
        
        # Try backup data sources
        for backup_path in self.backup_paths:
            if safe_file_exists(backup_path):
                self.logger.info(f"📁 Found backup data: {backup_path}")
                return backup_path
        
        # Try data repair methods
        for repair_method in self.data_repair_methods:
            try:
                if callable(repair_method):
                    result = await repair_method(context) if asyncio.iscoroutinefunction(repair_method) else repair_method(context)
                    if result:
                        self.logger.info("🔧 Data repair successful")
                        return result
            except Exception as repair_error:
                self.logger.warning(f"⚠️ Data repair method failed: {repair_error}")
        
        self.logger.error("❌ Data recovery failed")
        return None

class EnhancedErrorHandler:
    """Enhanced error handler with multiple recovery strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.EnhancedErrorHandler")
        self.strategies = []
        self.error_log = []
        
        # Initialize recovery strategies
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize error recovery strategies."""
        strategy_configs = self.config.get('recovery_strategies', {})
        
        # Add retry strategy
        if strategy_configs.get('retry', {}).get('enabled', True):
            self.strategies.append(RetryStrategy(strategy_configs.get('retry', {})))
        
        # Add fallback strategy
        if strategy_configs.get('fallback', {}).get('enabled', True):
            self.strategies.append(FallbackStrategy(strategy_configs.get('fallback', {})))
        
        # Add data recovery strategy
        if strategy_configs.get('data_recovery', {}).get('enabled', True):
            self.strategies.append(DataRecoveryStrategy(strategy_configs.get('data_recovery', {})))
    
    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        operation: Optional[Callable] = None
    ) -> Any:
        """Handle error with recovery strategies."""
        try:
            self.logger.error(f"💥 Error occurred: {type(error).__name__}: {error}")
            
            # Log error details
            error_info = {
                'timestamp': get_current_datetime(),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context,
                'traceback': traceback.format_exc()
            }
            self.error_log.append(error_info)
            
            # Try recovery strategies in order
            for strategy in self.strategies:
                try:
                    if await strategy.can_recover(error, context):
                        self.logger.info(f"🔄 Trying recovery strategy: {strategy.name}")
                        result = await strategy.recover(error, context)
                        
                        if result is not None:
                            self.logger.info(f"✅ Recovery successful with strategy: {strategy.name}")
                            
                            # Retry operation if provided
                            if operation and callable(operation):
                                try:
                                    if asyncio.iscoroutinefunction(operation):
                                        return await operation(**context)
                                    else:
                                        return operation(**context)
                                except Exception as retry_error:
                                    self.logger.warning(f"⚠️ Operation retry failed: {retry_error}")
                                    continue
                            
                            return result
                        else:
                            self.logger.warning(f"⚠️ Recovery strategy {strategy.name} returned None")
                    else:
                        self.logger.debug(f"⏭️ Skipping strategy {strategy.name} (cannot recover)")
                        
                except Exception as strategy_error:
                    self.logger.exception(f"❌ Recovery strategy {strategy.name} failed: {strategy_error}")
                    continue
            
            # All recovery strategies failed
            self.logger.error("❌ All recovery strategies failed")
            return None
            
        except Exception as handler_error:
            self.logger.exception(f"💥 Error handler itself failed: {handler_error}")
            return None
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of handled errors."""
        if not self.error_log:
            return {'total_errors': 0, 'error_types': {}}
        
        error_types = {}
        for error_info in self.error_log:
            error_type = error_info['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_log),
            'error_types': error_types,
            'recent_errors': self.error_log[-5:] if len(self.error_log) > 5 else self.error_log
        }
    
    def save_error_log(self, file_path: str):
        """Save error log to file."""
        try:
            ensure_directory(Path(file_path).parent)
            safe_json_dump(self.error_log, file_path, indent=2)
            self.logger.info(f"💾 Error log saved to: {file_path}")
        except Exception as e:
            self.logger.exception(f"❌ Failed to save error log: {e}")

def create_error_handler_decorator(
    config: Optional[Dict[str, Any]] = None,
    operation_name: Optional[str] = None
):
    """Create an error handler decorator."""
    if config is None:
        config = {
            'recovery_strategies': {
                'retry': {'enabled': True, 'max_retries': 3, 'retry_delay': 1.0},
                'fallback': {'enabled': True, 'fallback_value': None},
                'data_recovery': {'enabled': True, 'backup_paths': []}
            }
        }
    
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            error_handler = EnhancedErrorHandler(config)
            context = {
                'function_name': func.__name__,
                'operation_name': operation_name or func.__name__,
                'args': str(args)[:100],  # Truncate for logging
                'kwargs': str(kwargs)[:100]
            }
            
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as error:
                return await error_handler.handle_error(error, context, func)
        
        def sync_wrapper(*args, **kwargs):
            error_handler = EnhancedErrorHandler(config)
            context = {
                'function_name': func.__name__,
                'operation_name': operation_name or func.__name__,
                'args': str(args)[:100],
                'kwargs': str(kwargs)[:100]
            }
            
            try:
                return func(*args, **kwargs)
            except Exception as error:
                # For sync functions, we need to run the async handler
                return asyncio.run(error_handler.handle_error(error, context, func))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

class PipelineErrorHandler:
    """Specialized error handler for pipeline operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.PipelineErrorHandler")
        self.error_handler = EnhancedErrorHandler(config)
        self.pipeline_state = {}
    
    async def handle_pipeline_error(
        self,
        error: Exception,
        step_name: str,
        pipeline_state: Dict[str, Any],
        operation: Optional[Callable] = None
    ) -> Any:
        """Handle pipeline-specific errors."""
        try:
            context = {
                'step_name': step_name,
                'pipeline_state': pipeline_state,
                'error_timestamp': get_current_datetime(),
                'retry_count': pipeline_state.get(f'{step_name}_retry_count', 0)
            }
            
            # Update pipeline state
            self.pipeline_state.update(pipeline_state)
            self.pipeline_state[f'{step_name}_error_count'] = self.pipeline_state.get(f'{step_name}_error_count', 0) + 1
            
            # Handle the error
            result = await self.error_handler.handle_error(error, context, operation)
            
            if result is not None:
                # Mark step as recovered
                self.pipeline_state[f'{step_name}_recovered'] = True
                self.pipeline_state[f'{step_name}_recovery_timestamp'] = get_current_datetime()
            
            return result
            
        except Exception as handler_error:
            self.logger.exception(f"💥 Pipeline error handler failed: {handler_error}")
            return None
    
    def get_pipeline_health(self) -> Dict[str, Any]:
        """Get pipeline health status."""
        health_status = {
            'total_steps': len(self.pipeline_state),
            'error_counts': {},
            'recovered_steps': [],
            'failed_steps': []
        }
        
        for key, value in self.pipeline_state.items():
            if key.endswith('_error_count'):
                step_name = key.replace('_error_count', '')
                health_status['error_counts'][step_name] = value
            elif key.endswith('_recovered') and value:
                step_name = key.replace('_recovered', '')
                health_status['recovered_steps'].append(step_name)
        
        # Identify failed steps (high error count, no recovery)
        for step_name, error_count in health_status['error_counts'].items():
            if error_count > 3 and step_name not in health_status['recovered_steps']:
                health_status['failed_steps'].append(step_name)
        
        return health_status

# Export main classes and functions
__all__ = [
    'ErrorRecoveryStrategy',
    'RetryStrategy',
    'FallbackStrategy', 
    'DataRecoveryStrategy',
    'EnhancedErrorHandler',
    'create_error_handler_decorator',
    'PipelineErrorHandler'
]