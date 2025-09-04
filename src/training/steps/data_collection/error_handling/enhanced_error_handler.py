#!/usr/bin/env python3
"""
Enhanced Error Handling System for Data Collection Pipeline

This module provides comprehensive error handling with proper exception types,
recovery mechanisms, and detailed error reporting.
"""

import asyncio
import logging
import traceback
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass, asdict
from enum import Enum
import json

from src.utils.common_operations import (
    get_current_datetime,
    format_datetime,
    safe_json_dump,
    safe_json_load
)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ErrorCategory(Enum):
    """Error categories."""
    DATA_QUALITY = "DATA_QUALITY"
    NETWORK = "NETWORK"
    STORAGE = "STORAGE"
    VALIDATION = "VALIDATION"
    PROCESSING = "PROCESSING"
    CONFIGURATION = "CONFIGURATION"
    PERMISSION = "PERMISSION"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


class RecoveryStrategy(Enum):
    """Recovery strategies for errors."""
    RETRY = "RETRY"
    SKIP = "SKIP"
    FALLBACK = "FALLBACK"
    ABORT = "ABORT"
    MANUAL_INTERVENTION = "MANUAL_INTERVENTION"


@dataclass
class ErrorContext:
    """Context information for errors."""
    operation: str
    step_name: str
    symbol: str
    exchange: str
    data_dir: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


@dataclass
class ErrorReport:
    """Comprehensive error report."""
    error_id: str
    timestamp: str
    severity: ErrorSeverity
    category: ErrorCategory
    error_type: str
    error_message: str
    context: ErrorContext
    stack_trace: str
    recovery_strategy: RecoveryStrategy
    retry_count: int
    resolved: bool = False
    resolution_time: Optional[str] = None
    resolution_notes: Optional[str] = None


class DataCollectionError(Exception):
    """Base exception for data collection errors."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: Optional[ErrorContext] = None,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.ABORT,
        **kwargs
    ):
        super().__init__(message)
        self.severity = severity
        self.category = category
        self.context = context
        self.recovery_strategy = recovery_strategy
        self.additional_info = kwargs


class DataQualityError(DataCollectionError):
    """Exception for data quality issues."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATA_QUALITY,
            context=context,
            recovery_strategy=RecoveryStrategy.SKIP,
            **kwargs
        )


class NetworkError(DataCollectionError):
    """Exception for network-related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            context=context,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs
        )


class StorageError(DataCollectionError):
    """Exception for storage-related errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.STORAGE,
            context=context,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs
        )


class ValidationError(DataCollectionError):
    """Exception for validation errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
            context=context,
            recovery_strategy=RecoveryStrategy.SKIP,
            **kwargs
        )


class ProcessingError(DataCollectionError):
    """Exception for data processing errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.PROCESSING,
            context=context,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs
        )


class ConfigurationError(DataCollectionError):
    """Exception for configuration errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.CONFIGURATION,
            context=context,
            recovery_strategy=RecoveryStrategy.ABORT,
            **kwargs
        )


class PermissionError(DataCollectionError):
    """Exception for permission errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.PERMISSION,
            context=context,
            recovery_strategy=RecoveryStrategy.ABORT,
            **kwargs
        )


class TimeoutError(DataCollectionError):
    """Exception for timeout errors."""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.TIMEOUT,
            context=context,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs
        )


class EnhancedErrorHandler:
    """Enhanced error handler with comprehensive error management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.error_reports: List[ErrorReport] = []
        self.error_log_file = Path(config.get('error_log_file', 'logs/error_reports.json'))
        self.max_retry_attempts = config.get('max_retry_attempts', 3)
        self.retry_delays = config.get('retry_delays', [1, 2, 4])  # seconds
        
        # Ensure error log directory exists
        self.error_log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        retry_count: int = 0
    ) -> ErrorReport:
        """Handle an error and create a comprehensive error report."""
        try:
            # Generate unique error ID
            error_id = self._generate_error_id(error, context)
            
            # Determine error characteristics
            severity = self._determine_severity(error)
            category = self._determine_category(error)
            recovery_strategy = self._determine_recovery_strategy(error, retry_count)
            
            # Create error report
            error_report = ErrorReport(
                error_id=error_id,
                timestamp=format_datetime(get_current_datetime()),
                severity=severity,
                category=category,
                error_type=type(error).__name__,
                error_message=str(error),
                context=context,
                stack_trace=traceback.format_exc(),
                recovery_strategy=recovery_strategy,
                retry_count=retry_count
            )
            
            # Store error report
            self.error_reports.append(error_report)
            
            # Log error
            self._log_error(error_report)
            
            # Save to file
            self._save_error_report(error_report)
            
            return error_report
            
        except Exception as e:
            self.logger.exception(f"Error in error handler: {e}")
            # Create a minimal error report
            return ErrorReport(
                error_id="error_handler_failed",
                timestamp=format_datetime(get_current_datetime()),
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.UNKNOWN,
                error_type=type(error).__name__,
                error_message=str(error),
                context=context,
                stack_trace=traceback.format_exc(),
                recovery_strategy=RecoveryStrategy.ABORT,
                retry_count=retry_count
            )
    
    async def execute_with_error_handling(
        self,
        operation: Callable,
        context: ErrorContext,
        *args,
        **kwargs
    ) -> Any:
        """Execute an operation with comprehensive error handling and retry logic."""
        retry_count = 0
        
        while retry_count <= self.max_retry_attempts:
            try:
                # Execute operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                # If we get here, operation succeeded
                if retry_count > 0:
                    self.logger.info(f"Operation succeeded after {retry_count} retries: {context.operation}")
                
                return result
                
            except Exception as error:
                # Handle the error
                error_report = self.handle_error(error, context, retry_count)
                
                # Determine next action based on recovery strategy
                if error_report.recovery_strategy == RecoveryStrategy.ABORT:
                    self.logger.error(f"Aborting operation due to error: {context.operation}")
                    raise error
                
                elif error_report.recovery_strategy == RecoveryStrategy.SKIP:
                    self.logger.warning(f"Skipping operation due to error: {context.operation}")
                    return None
                
                elif error_report.recovery_strategy == RecoveryStrategy.RETRY:
                    if retry_count < self.max_retry_attempts:
                        retry_count += 1
                        delay = self.retry_delays[min(retry_count - 1, len(self.retry_delays) - 1)]
                        self.logger.info(f"Retrying operation in {delay} seconds (attempt {retry_count}/{self.max_retry_attempts})")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        self.logger.error(f"Max retry attempts reached for operation: {context.operation}")
                        raise error
                
                elif error_report.recovery_strategy == RecoveryStrategy.FALLBACK:
                    self.logger.warning(f"Using fallback for operation: {context.operation}")
                    return await self._execute_fallback(operation, context, *args, **kwargs)
                
                elif error_report.recovery_strategy == RecoveryStrategy.MANUAL_INTERVENTION:
                    self.logger.critical(f"Manual intervention required for operation: {context.operation}")
                    raise error
        
        # Should not reach here
        raise Exception(f"Unexpected error in execute_with_error_handling for {context.operation}")
    
    def _generate_error_id(self, error: Exception, context: ErrorContext) -> str:
        """Generate a unique error ID."""
        import hashlib
        
        error_info = f"{type(error).__name__}_{context.operation}_{context.step_name}_{context.symbol}_{context.exchange}"
        return hashlib.md5(error_info.encode()).hexdigest()[:12]
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on error type."""
        if isinstance(error, ConfigurationError):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (DataQualityError, StorageError, PermissionError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (NetworkError, ValidationError, TimeoutError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _determine_category(self, error: Exception) -> ErrorCategory:
        """Determine error category based on error type."""
        if isinstance(error, DataQualityError):
            return ErrorCategory.DATA_QUALITY
        elif isinstance(error, NetworkError):
            return ErrorCategory.NETWORK
        elif isinstance(error, StorageError):
            return ErrorCategory.STORAGE
        elif isinstance(error, ValidationError):
            return ErrorCategory.VALIDATION
        elif isinstance(error, ProcessingError):
            return ErrorCategory.PROCESSING
        elif isinstance(error, ConfigurationError):
            return ErrorCategory.CONFIGURATION
        elif isinstance(error, PermissionError):
            return ErrorCategory.PERMISSION
        elif isinstance(error, TimeoutError):
            return ErrorCategory.TIMEOUT
        else:
            return ErrorCategory.UNKNOWN
    
    def _determine_recovery_strategy(
        self,
        error: Exception,
        retry_count: int
    ) -> RecoveryStrategy:
        """Determine recovery strategy based on error type and retry count."""
        if isinstance(error, DataCollectionError):
            # Use the strategy defined in the exception
            if retry_count >= self.max_retry_attempts:
                return RecoveryStrategy.ABORT
            return error.recovery_strategy
        
        # Default strategies for standard exceptions
        if isinstance(error, (ConnectionError, TimeoutError)):
            return RecoveryStrategy.RETRY
        elif isinstance(error, (FileNotFoundError, PermissionError)):
            return RecoveryStrategy.ABORT
        elif isinstance(error, (ValueError, TypeError)):
            return RecoveryStrategy.SKIP
        else:
            return RecoveryStrategy.ABORT
    
    def _log_error(self, error_report: ErrorReport) -> None:
        """Log error with appropriate level."""
        log_message = (
            f"Error {error_report.error_id}: {error_report.error_type} | "
            f"Severity: {error_report.severity.value} | "
            f"Category: {error_report.category.value} | "
            f"Operation: {error_report.context.operation} | "
            f"Step: {error_report.context.step_name} | "
            f"Symbol: {error_report.context.symbol} | "
            f"Exchange: {error_report.context.exchange} | "
            f"Retry: {error_report.retry_count} | "
            f"Strategy: {error_report.recovery_strategy.value}"
        )
        
        if error_report.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_report.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_report.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _save_error_report(self, error_report: ErrorReport) -> None:
        """Save error report to file."""
        try:
            # Convert to dict for JSON serialization
            report_dict = asdict(error_report)
            
            # Load existing reports
            existing_reports = []
            if self.error_log_file.exists():
                try:
                    existing_reports = safe_json_load(self.error_log_file)
                except Exception:
                    existing_reports = []
            
            # Add new report
            existing_reports.append(report_dict)
            
            # Keep only last 1000 reports to prevent file from growing too large
            if len(existing_reports) > 1000:
                existing_reports = existing_reports[-1000:]
            
            # Save back to file
            safe_json_dump(existing_reports, self.error_log_file)
            
        except Exception as e:
            self.logger.exception(f"Error saving error report: {e}")
    
    async def _execute_fallback(
        self,
        operation: Callable,
        context: ErrorContext,
        *args,
        **kwargs
    ) -> Any:
        """Execute fallback operation."""
        # This would be implemented based on specific fallback strategies
        # For now, return None as a placeholder
        self.logger.warning(f"No fallback implemented for operation: {context.operation}")
        return None
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors."""
        if not self.error_reports:
            return {"message": "No errors reported"}
        
        # Count by severity
        severity_counts = {}
        for report in self.error_reports:
            severity = report.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count by category
        category_counts = {}
        for report in self.error_reports:
            category = report.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Count by operation
        operation_counts = {}
        for report in self.error_reports:
            operation = report.context.operation
            operation_counts[operation] = operation_counts.get(operation, 0) + 1
        
        return {
            "total_errors": len(self.error_reports),
            "severity_counts": severity_counts,
            "category_counts": category_counts,
            "operation_counts": operation_counts,
            "recent_errors": [
                {
                    "error_id": r.error_id,
                    "timestamp": r.timestamp,
                    "severity": r.severity.value,
                    "category": r.category.value,
                    "operation": r.context.operation,
                    "message": r.error_message
                }
                for r in self.error_reports[-10:]  # Last 10 errors
            ]
        }
    
    def print_error_summary(self) -> None:
        """Print a formatted error summary."""
        summary = self.get_error_summary()
        
        print("\n" + "="*80)
        print("📊 ERROR HANDLING SUMMARY")
        print("="*80)
        print(f"Total Errors: {summary['total_errors']}")
        
        if 'severity_counts' in summary:
            print("\nSeverity Breakdown:")
            for severity, count in summary['severity_counts'].items():
                print(f"  {severity}: {count}")
        
        if 'category_counts' in summary:
            print("\nCategory Breakdown:")
            for category, count in summary['category_counts'].items():
                print(f"  {category}: {count}")
        
        if 'operation_counts' in summary:
            print("\nOperation Breakdown:")
            for operation, count in summary['operation_counts'].items():
                print(f"  {operation}: {count}")
        
        if 'recent_errors' in summary and summary['recent_errors']:
            print("\nRecent Errors:")
            for error in summary['recent_errors']:
                print(f"  {error['timestamp']} | {error['severity']} | {error['category']} | {error['operation']} | {error['message']}")
        
        print("="*80)


# Export main classes and functions
__all__ = [
    'ErrorSeverity',
    'ErrorCategory',
    'RecoveryStrategy',
    'ErrorContext',
    'ErrorReport',
    'DataCollectionError',
    'DataQualityError',
    'NetworkError',
    'StorageError',
    'ValidationError',
    'ProcessingError',
    'ConfigurationError',
    'PermissionError',
    'TimeoutError',
    'EnhancedErrorHandler'
]