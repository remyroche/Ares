from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""Enhanced Error Handling System.

This module provides comprehensive error handling with detailed function-level error reporting.
"""
import logging
import traceback
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List

from src.utils.comprehensive_function_logger import (
    log_step_functions, log_important_calls, log_all_calls,
    log_internal_call, log_step_progress, log_data_operation
)

import asyncio
import numpy as np
import time

class EnhancedErrorHandler:
    """Enhanced error handling system with detailed function-level error reporting."""

    @log_important_calls
    def __init__(self, logger: Any = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: List[Dict[str, Any]] = []
        self.error_patterns: Dict[str, int] = {}
        self.function_error_counts: Dict[str, int] = {}
        
    def handle_function_error(self, function_name: str, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle and analyze function-level errors with detailed reporting."""
        try:
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'function_name': function_name,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'error_details': {
                    'module': getattr(error, '__module__', 'unknown'),
                    'args': getattr(error, 'args', ()),
                    'filename': getattr(error, '__traceback__', {}).get('tb_frame', {}).get('f_code', {}).get('co_filename', 'unknown') if hasattr(error, '__traceback__') else 'unknown',
                    'line_number': getattr(error, '__traceback__', {}).get('tb_lineno', 'unknown') if hasattr(error, '__traceback__') else 'unknown'
                },
                'context': context or {},
                'stack_trace': traceback.format_exc(),
                'severity': self._determine_error_severity(error),
                'recovery_suggestions': self._generate_recovery_suggestions(error, function_name)
            }
            
            # Update error patterns and counts
            error_key = f"{function_name}:{type(error).__name__}"
            self.error_patterns[error_key] = self.error_patterns.get(error_key, 0) + 1
            self.function_error_counts[function_name] = self.function_error_counts.get(function_name, 0) + 1
            
            # Add to error history
            self.error_history.append(error_info)
            
            # Log detailed error information
            self._log_detailed_error(error_info)
            
            return error_info
            
        except Exception as e:
            self.logger.error(f"❌ Failed to handle error in EnhancedErrorHandler: {e}")
            return {}

    @log_all_calls
    def _determine_error_severity(self, error: Exception) -> str:
        """Determine error severity based on error type and context."""
        critical_errors = (SystemError, MemoryError, OSError, RuntimeError)
        warning_errors = (UserWarning, DeprecationWarning, FutureWarning)
        
        if isinstance(error, critical_errors):
            return "CRITICAL"
        elif isinstance(error, warning_errors):
            return "WARNING"
        elif isinstance(error, (ValueError, TypeError, AttributeError)):
            return "ERROR"
        else:
            return "UNKNOWN"

    @log_all_calls
    def _generate_recovery_suggestions(self, error: Exception, function_name: str) -> List[str]:
        """Generate recovery suggestions based on error type and function."""
        suggestions = []
        
        if isinstance(error, FileNotFoundError):
            suggestions.extend([
                "Check if the file path exists and is accessible",
                "Verify file permissions",
                "Ensure the directory structure is correct"
            ])
        elif isinstance(error, ValueError):
            suggestions.extend([
                "Validate input parameters before processing",
                "Check data types and ranges",
                "Review data format and structure"
            ])
        elif isinstance(error, MemoryError):
            suggestions.extend([
                "Consider processing data in smaller chunks",
                "Optimize memory usage",
                "Check for memory leaks"
            ])
        elif isinstance(error, TimeoutError):
            suggestions.extend([
                "Increase timeout values",
                "Optimize function performance",
                "Consider asynchronous processing"
            ])
        elif isinstance(error, ImportError):
            suggestions.extend([
                "Check if required modules are installed",
                "Verify import paths",
                "Update dependencies"
            ])
        
        # Function-specific suggestions
        if 'labeling' in function_name.lower():
            suggestions.extend([
                "Verify input data quality and format",
                "Check labeling configuration parameters",
                "Ensure sufficient data for labeling"
            ])
        elif 'regime' in function_name.lower():
            suggestions.extend([
                "Verify regime data availability",
                "Check regime detection parameters",
                "Ensure regime labels are properly formatted"
            ])
        
        return suggestions

    @log_all_calls
    def _log_detailed_error(self, error_info: Dict[str, Any]) -> None:
        """Log detailed error information."""
        try:
            self.logger.error(f"🚨 DETAILED ERROR REPORT")
            self.logger.error(f"   Function: {error_info['function_name']}")
            self.logger.error(f"   Error Type: {error_info['error_type']}")
            self.logger.error(f"   Severity: {error_info['severity']}")
            self.logger.error(f"   Message: {error_info['error_message']}")
            self.logger.error(f"   Timestamp: {error_info['timestamp']}")
            
            if error_info['context']:
                self.logger.error(f"   Context: {error_info['context']}")
            
            if error_info['recovery_suggestions']:
                self.logger.error(f"   Recovery Suggestions:")
                for i, suggestion in enumerate(error_info['recovery_suggestions'], 1):
                    self.logger.error(f"     {i}. {suggestion}")
            
            # Log stack trace at debug level to avoid cluttering logs
            self.logger.debug(f"   Stack Trace:\n{error_info['stack_trace']}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log detailed error: {e}")
    
    def generate_error_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive error summary report."""
        try:
            if not self.error_history:
                return {'total_errors': 0, 'message': 'No errors recorded'}
            
            # Analyze error patterns
            error_type_counts = {}
            severity_counts = {}
            function_error_summary = {}
            
            for error in self.error_history:
                error_type = error['error_type']
                severity = error['severity']
                function_name = error['function_name']
                
                error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                if function_name not in function_error_summary:
                    function_error_summary[function_name] = {
                        'total_errors': 0,
                        'error_types': {},
                        'severities': {}
                    }
                
                function_error_summary[function_name]['total_errors'] += 1
                function_error_summary[function_name]['error_types'][error_type] = \
                    function_error_summary[function_name]['error_types'].get(error_type, 0) + 1
                function_error_summary[function_name]['severities'][severity] = \
                    function_error_summary[function_name]['severities'].get(severity, 0) + 1
            
            # Find most common errors
            most_common_error_type = max(error_type_counts.items(), key=lambda x: x[1])[0] if error_type_counts else None
            most_error_prone_function = max(function_error_summary.items(), key=lambda x: x[1]['total_errors'])[0] if function_error_summary else None
            
            return {
                'total_errors': len(self.error_history),
                'error_type_counts': error_type_counts,
                'severity_counts': severity_counts,
                'function_error_summary': function_error_summary,
                'most_common_error_type': most_common_error_type,
                'most_error_prone_function': most_error_prone_function,
                'error_patterns': self.error_patterns,
                'recent_errors': self.error_history[-5:] if len(self.error_history) > 5 else self.error_history
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate error summary report: {e}")
            return {}
    
    def log_error_summary_report(self, report: Dict[str, Any]) -> None:
        """Log comprehensive error summary report."""
        try:
            if report.get('total_errors', 0) == 0:
                self.logger.info("✅ No errors recorded during execution")
                return
            
            self.logger.info("📊 ERROR SUMMARY REPORT")
            self.logger.info("=" * 40)
            self.logger.info(f"Total Errors: {report['total_errors']}")
            
            # Error type summary
            if report.get('error_type_counts'):
                self.logger.info(f"\nError Types:")
                for error_type, count in sorted(report['error_type_counts'].items(), key = lambda x: x[1], reverse = True):
                    self.logger.info(f"  - {error_type}: {count} occurrences")
            
            # Severity summary
            if report.get('severity_counts'):
                self.logger.info(f"\nSeverity Distribution:")
                for severity, count in sorted(report['severity_counts'].items(), key = lambda x: x[1], reverse = True):
                    self.logger.info(f"  - {severity}: {count} occurrences")
            
            # Function error summary
            if report.get('function_error_summary'):
                self.logger.info(f"\nFunction Error Summary:")
                for function_name, summary in sorted(report['function_error_summary'].items(), 
                                                   key = lambda x: x[1]['total_errors'], reverse = True):
                    self.logger.info(f"  - {function_name}: {summary['total_errors']} errors")
                    for error_type, count in summary['error_types'].items():
                        self.logger.info(f"    * {error_type}: {count}")
            
            # Most problematic areas
            if report.get('most_common_error_type'):
                self.logger.info(f"\nMost Common Error Type: {report['most_common_error_type']}")
            
            if report.get('most_error_prone_function'):
                self.logger.info(f"Most Error-Prone Function: {report['most_error_prone_function']}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log error summary report: {e}")

def enhanced_error_handler(handler: EnhancedErrorHandler):
    """Decorator for enhanced error handling with detailed reporting."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function_name': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys()),
                    'timestamp': datetime.now().isoformat()
                }
                handler.handle_function_error(func.__name__, e, context)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function_name': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys()),
                    'timestamp': datetime.now().isoformat()
                }
                handler.handle_function_error(func.__name__, e, context)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator