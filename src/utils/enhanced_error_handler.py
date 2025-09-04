"""
Enhanced Error Handler with Recovery Mechanisms

This module provides comprehensive error handling including:
- Error classification and categorization
- Automatic recovery strategies
- Error context preservation
- Retry mechanisms with exponential backoff
- Error reporting and notification
- Graceful degradation strategies
"""
import asyncio
import json
import logging
import time
import traceback
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

from src.core.decorators import handles_errors, log_call, traced
from src.utils.logger import system_logger
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, get_current_datetime, format_datetime
)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'

class ErrorCategory(Enum):
    """Error categories."""
    DATA_ERROR = 'data_error'
    CONFIGURATION_ERROR = 'configuration_error'
    NETWORK_ERROR = 'network_error'
    MEMORY_ERROR = 'memory_error'
    DISK_ERROR = 'disk_error'
    VALIDATION_ERROR = 'validation_error'
    TRAINING_ERROR = 'training_error'
    SYSTEM_ERROR = 'system_error'

class RecoveryStrategy(Enum):
    """Recovery strategies."""
    RETRY = 'retry'
    FALLBACK = 'fallback'
    SKIP = 'skip'
    RESTART = 'restart'
    MANUAL_INTERVENTION = 'manual_intervention'

@dataclass
class ErrorContext:
    """Error context information."""
    error_type: str
    error_message: str
    error_category: ErrorCategory
    error_severity: ErrorSeverity
    timestamp: datetime
    function_name: str
    file_path: str
    line_number: int
    stack_trace: str
    context_data: Dict[str, Any]
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3

@dataclass
class RecoveryAction:
    """Recovery action definition."""
    strategy: RecoveryStrategy
    action: Callable
    max_attempts: int = 3
    backoff_multiplier: float = 2.0
    initial_delay: float = 1.0
    success_criteria: Optional[Callable] = None

class EnhancedErrorHandler:
    """Enhanced error handler with comprehensive recovery mechanisms."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize enhanced error handler."""
        self.config = config
        self.logger = system_logger.getChild('EnhancedErrorHandler')
        
        # Error tracking
        self.error_history: List[ErrorContext] = []
        self.recovery_history: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
        
        # Recovery configuration
        self.recovery_config = config.get('recovery', {})
        self.max_retry_attempts = self.recovery_config.get('max_retry_attempts', 3)
        self.retry_backoff_multiplier = self.recovery_config.get('retry_backoff_multiplier', 2.0)
        self.retry_initial_delay = self.recovery_config.get('retry_initial_delay', 1.0)
        
        # Error classification rules
        self.error_classification_rules = self._initialize_error_classification_rules()
        
        # Recovery strategies
        self.recovery_strategies = self._initialize_recovery_strategies()

    def _initialize_error_classification_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize error classification rules."""
        return {
            'FileNotFoundError': {
                'category': ErrorCategory.DATA_ERROR,
                'severity': ErrorSeverity.MEDIUM,
                'recovery_strategy': RecoveryStrategy.RETRY
            },
            'PermissionError': {
                'category': ErrorCategory.SYSTEM_ERROR,
                'severity': ErrorSeverity.HIGH,
                'recovery_strategy': RecoveryStrategy.MANUAL_INTERVENTION
            },
            'MemoryError': {
                'category': ErrorCategory.MEMORY_ERROR,
                'severity': ErrorSeverity.HIGH,
                'recovery_strategy': RecoveryStrategy.FALLBACK
            },
            'ValueError': {
                'category': ErrorCategory.VALIDATION_ERROR,
                'severity': ErrorSeverity.MEDIUM,
                'recovery_strategy': RecoveryStrategy.SKIP
            },
            'KeyError': {
                'category': ErrorCategory.CONFIGURATION_ERROR,
                'severity': ErrorSeverity.MEDIUM,
                'recovery_strategy': RecoveryStrategy.FALLBACK
            },
            'ConnectionError': {
                'category': ErrorCategory.NETWORK_ERROR,
                'severity': ErrorSeverity.MEDIUM,
                'recovery_strategy': RecoveryStrategy.RETRY
            },
            'TimeoutError': {
                'category': ErrorCategory.NETWORK_ERROR,
                'severity': ErrorSeverity.MEDIUM,
                'recovery_strategy': RecoveryStrategy.RETRY
            }
        }

    def _initialize_recovery_strategies(self) -> Dict[RecoveryStrategy, RecoveryAction]:
        """Initialize recovery strategies."""
        return {
            RecoveryStrategy.RETRY: RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                action=self._retry_with_backoff,
                max_attempts=self.max_retry_attempts,
                backoff_multiplier=self.retry_backoff_multiplier,
                initial_delay=self.retry_initial_delay
            ),
            RecoveryStrategy.FALLBACK: RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                action=self._fallback_strategy,
                max_attempts=1
            ),
            RecoveryStrategy.SKIP: RecoveryAction(
                strategy=RecoveryStrategy.SKIP,
                action=self._skip_strategy,
                max_attempts=1
            ),
            RecoveryStrategy.RESTART: RecoveryAction(
                strategy=RecoveryStrategy.RESTART,
                action=self._restart_strategy,
                max_attempts=1
            ),
            RecoveryStrategy.MANUAL_INTERVENTION: RecoveryAction(
                strategy=RecoveryStrategy.MANUAL_INTERVENTION,
                action=self._manual_intervention_strategy,
                max_attempts=1
            )
        }

    @handles_errors(Exception, fallback=None, log_level="ERROR")
    @log_call
    @traced
    def handle_error(
        self, 
        error: Exception, 
        context: Dict[str, Any] = None,
        recovery_enabled: bool = True
    ) -> Optional[Any]:
        """Handle error with comprehensive recovery mechanisms."""
        self.logger.error(f"🚨 Handling error: {type(error).__name__}: {str(error)}")
        
        try:
            # Create error context
            error_context = self._create_error_context(error, context or {})
            
            # Classify error
            classification = self._classify_error(error)
            error_context.error_category = classification['category']
            error_context.error_severity = classification['severity']
            
            # Log error
            self._log_error(error_context)
            
            # Track error
            self._track_error(error_context)
            
            # Attempt recovery if enabled
            if recovery_enabled:
                recovery_result = self._attempt_recovery(error_context, classification)
                if recovery_result['success']:
                    self.logger.info(f"✅ Error recovery successful: {error_context.error_type}")
                    return recovery_result['result']
                else:
                    self.logger.error(f"❌ Error recovery failed: {recovery_result['reason']}")
            
            # If no recovery or recovery failed, handle gracefully
            return self._handle_graceful_degradation(error_context)
            
        except Exception as e:
            self.logger.error(f"❌ Error handler failed: {e}")
            return None

    def _create_error_context(self, error: Exception, context: Dict[str, Any]) -> ErrorContext:
        """Create comprehensive error context."""
        tb = traceback.extract_tb(error.__traceback__)
        frame = tb[-1] if tb else None
        
        return ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            error_category=ErrorCategory.SYSTEM_ERROR,  # Default
            error_severity=ErrorSeverity.MEDIUM,  # Default
            timestamp=get_current_datetime(),
            function_name=frame.filename if frame else 'unknown',
            file_path=frame.filename if frame else 'unknown',
            line_number=frame.lineno if frame else 0,
            stack_trace=traceback.format_exc(),
            context_data=context,
            recovery_attempts=0,
            max_recovery_attempts=self.max_retry_attempts
        )

    def _classify_error(self, error: Exception) -> Dict[str, Any]:
        """Classify error based on type and context."""
        error_type = type(error).__name__
        
        # Check classification rules
        if error_type in self.error_classification_rules:
            return self.error_classification_rules[error_type]
        
        # Default classification
        return {
            'category': ErrorCategory.SYSTEM_ERROR,
            'severity': ErrorSeverity.MEDIUM,
            'recovery_strategy': RecoveryStrategy.SKIP
        }

    def _log_error(self, error_context: ErrorContext) -> None:
        """Log error with comprehensive information."""
        log_data = {
            'error_type': error_context.error_type,
            'error_message': error_context.error_message,
            'error_category': error_context.error_category.value,
            'error_severity': error_context.error_severity.value,
            'timestamp': format_datetime(error_context.timestamp),
            'function_name': error_context.function_name,
            'file_path': error_context.file_path,
            'line_number': error_context.line_number,
            'context_data': error_context.context_data
        }
        
        # Log based on severity
        if error_context.error_severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"🚨 CRITICAL ERROR: {log_data}")
        elif error_context.error_severity == ErrorSeverity.HIGH:
            self.logger.error(f"❌ HIGH SEVERITY ERROR: {log_data}")
        elif error_context.error_severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"⚠️ MEDIUM SEVERITY ERROR: {log_data}")
        else:
            self.logger.info(f"ℹ️ LOW SEVERITY ERROR: {log_data}")

    def _track_error(self, error_context: ErrorContext) -> None:
        """Track error for analysis and reporting."""
        # Add to error history
        self.error_history.append(error_context)
        
        # Update error counts
        error_key = f"{error_context.error_type}_{error_context.error_category.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Keep only last 1000 errors to prevent memory issues
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]

    def _attempt_recovery(self, error_context: ErrorContext, classification: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt error recovery based on classification."""
        recovery_strategy = classification.get('recovery_strategy', RecoveryStrategy.SKIP)
        
        if recovery_strategy not in self.recovery_strategies:
            return {
                'success': False,
                'reason': f'Unknown recovery strategy: {recovery_strategy}'
            }
        
        recovery_action = self.recovery_strategies[recovery_strategy]
        
        try:
            result = recovery_action.action(error_context, recovery_action)
            
            # Record recovery attempt
            recovery_record = {
                'timestamp': format_datetime(get_current_datetime()),
                'error_type': error_context.error_type,
                'recovery_strategy': recovery_strategy.value,
                'success': True,
                'result': result
            }
            self.recovery_history.append(recovery_record)
            
            return {
                'success': True,
                'result': result,
                'strategy': recovery_strategy.value
            }
            
        except Exception as e:
            recovery_record = {
                'timestamp': format_datetime(get_current_datetime()),
                'error_type': error_context.error_type,
                'recovery_strategy': recovery_strategy.value,
                'success': False,
                'error': str(e)
            }
            self.recovery_history.append(recovery_record)
            
            return {
                'success': False,
                'reason': f'Recovery failed: {e}'
            }

    def _retry_with_backoff(self, error_context: ErrorContext, recovery_action: RecoveryAction) -> Any:
        """Retry operation with exponential backoff."""
        self.logger.info(f"🔄 Attempting retry with backoff for {error_context.error_type}")
        
        for attempt in range(recovery_action.max_attempts):
            try:
                # Wait before retry (except for first attempt)
                if attempt > 0:
                    delay = recovery_action.initial_delay * (recovery_action.backoff_multiplier ** (attempt - 1))
                    self.logger.info(f"⏳ Waiting {delay:.2f}s before retry attempt {attempt + 1}")
                    time.sleep(delay)
                
                # Attempt to re-execute the original operation
                # This would need to be passed from the original context
                if 'original_function' in error_context.context_data:
                    original_function = error_context.context_data['original_function']
                    original_args = error_context.context_data.get('original_args', ())
                    original_kwargs = error_context.context_data.get('original_kwargs', {})
                    
                    result = original_function(*original_args, **original_kwargs)
                    
                    # Check success criteria if provided
                    if recovery_action.success_criteria:
                        if not recovery_action.success_criteria(result):
                            continue
                    
                    self.logger.info(f"✅ Retry successful on attempt {attempt + 1}")
                    return result
                else:
                    self.logger.warning("⚠️ No original function found for retry")
                    return None
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Retry attempt {attempt + 1} failed: {e}")
                if attempt == recovery_action.max_attempts - 1:
                    raise e
        
        return None

    def _fallback_strategy(self, error_context: ErrorContext, recovery_action: RecoveryAction) -> Any:
        """Implement fallback strategy."""
        self.logger.info(f"🔄 Attempting fallback strategy for {error_context.error_type}")
        
        # Implement fallback based on error type
        if error_context.error_type == 'FileNotFoundError':
            # Try alternative file paths
            if 'file_path' in error_context.context_data:
                original_path = error_context.context_data['file_path']
                fallback_paths = [
                    original_path.replace('.parquet', '.csv'),
                    original_path.replace('_consolidated', ''),
                    f"backup_{original_path}"
                ]
                
                for fallback_path in fallback_paths:
                    if Path(fallback_path).exists():
                        self.logger.info(f"✅ Found fallback file: {fallback_path}")
                        return fallback_path
        
        elif error_context.error_type == 'MemoryError':
            # Reduce memory usage
            self.logger.info("🔄 Implementing memory reduction fallback")
            return {'memory_reduced': True, 'chunk_size': 1000}
        
        elif error_context.error_type == 'KeyError':
            # Use default values
            self.logger.info("🔄 Using default values fallback")
            return {'use_defaults': True}
        
        return None

    def _skip_strategy(self, error_context: ErrorContext, recovery_action: RecoveryAction) -> Any:
        """Skip the problematic operation."""
        self.logger.info(f"⏭️ Skipping operation due to {error_context.error_type}")
        return {'skipped': True, 'reason': error_context.error_message}

    def _restart_strategy(self, error_context: ErrorContext, recovery_action: RecoveryAction) -> Any:
        """Restart the operation or system."""
        self.logger.info(f"🔄 Restarting operation due to {error_context.error_type}")
        return {'restarted': True, 'timestamp': format_datetime(get_current_datetime())}

    def _manual_intervention_strategy(self, error_context: ErrorContext, recovery_action: RecoveryAction) -> Any:
        """Request manual intervention."""
        self.logger.critical(f"🚨 Manual intervention required for {error_context.error_type}")
        
        # Create intervention request
        intervention_request = {
            'timestamp': format_datetime(get_current_datetime()),
            'error_type': error_context.error_type,
            'error_message': error_context.error_message,
            'error_category': error_context.error_category.value,
            'error_severity': error_context.error_severity.value,
            'context_data': error_context.context_data,
            'stack_trace': error_context.stack_trace
        }
        
        # Save intervention request
        intervention_file = f"intervention_request_{int(time.time())}.json"
        safe_json_dump(intervention_request, intervention_file, indent=2)
        
        return {'manual_intervention_requested': True, 'request_file': intervention_file}

    def _handle_graceful_degradation(self, error_context: ErrorContext) -> Any:
        """Handle graceful degradation when recovery fails."""
        self.logger.warning(f"⚠️ Implementing graceful degradation for {error_context.error_type}")
        
        # Return safe defaults based on error category
        if error_context.error_category == ErrorCategory.DATA_ERROR:
            return {'data_available': False, 'using_cached_data': True}
        elif error_context.error_category == ErrorCategory.CONFIGURATION_ERROR:
            return {'using_default_config': True}
        elif error_context.error_category == ErrorCategory.NETWORK_ERROR:
            return {'offline_mode': True}
        elif error_context.error_category == ErrorCategory.MEMORY_ERROR:
            return {'reduced_functionality': True}
        else:
            return {'degraded_mode': True}

    @handles_errors(Exception, fallback={}, log_level="ERROR")
    @log_call
    @traced
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        self.logger.info("📊 Generating error statistics...")
        
        try:
            # Calculate statistics
            total_errors = len(self.error_history)
            error_by_category = {}
            error_by_severity = {}
            error_by_type = {}
            
            for error in self.error_history:
                # Count by category
                category = error.error_category.value
                error_by_category[category] = error_by_category.get(category, 0) + 1
                
                # Count by severity
                severity = error.error_severity.value
                error_by_severity[severity] = error_by_severity.get(severity, 0) + 1
                
                # Count by type
                error_type = error.error_type
                error_by_type[error_type] = error_by_type.get(error_type, 0) + 1
            
            # Recovery statistics
            total_recoveries = len(self.recovery_history)
            successful_recoveries = sum(1 for r in self.recovery_history if r.get('success', False))
            recovery_rate = successful_recoveries / total_recoveries if total_recoveries > 0 else 0
            
            statistics = {
                'error_summary': {
                    'total_errors': total_errors,
                    'errors_by_category': error_by_category,
                    'errors_by_severity': error_by_severity,
                    'errors_by_type': error_by_type
                },
                'recovery_summary': {
                    'total_recovery_attempts': total_recoveries,
                    'successful_recoveries': successful_recoveries,
                    'recovery_rate': recovery_rate
                },
                'error_counts': self.error_counts,
                'recent_errors': self.error_history[-10:] if self.error_history else [],
                'recent_recoveries': self.recovery_history[-10:] if self.recovery_history else []
            }
            
            self.logger.info(f"✅ Generated error statistics: {total_errors} total errors, {recovery_rate:.2%} recovery rate")
            return statistics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate error statistics: {e}")
            return {'error': f"Statistics generation failed: {e}"}

    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    def export_error_report(self, output_path: str) -> bool:
        """Export comprehensive error report."""
        self.logger.info(f"📤 Exporting error report to: {output_path}")
        
        try:
            report = {
                'report_metadata': {
                    'generated_at': format_datetime(get_current_datetime()),
                    'handler_version': '1.0.0',
                    'total_errors_processed': len(self.error_history)
                },
                'error_statistics': self.get_error_statistics(),
                'error_history': [
                    {
                        'error_type': error.error_type,
                        'error_message': error.error_message,
                        'error_category': error.error_category.value,
                        'error_severity': error.error_severity.value,
                        'timestamp': format_datetime(error.timestamp),
                        'function_name': error.function_name,
                        'file_path': error.file_path,
                        'line_number': error.line_number,
                        'context_data': error.context_data,
                        'recovery_attempts': error.recovery_attempts
                    }
                    for error in self.error_history
                ],
                'recovery_history': self.recovery_history
            }
            
            safe_json_dump(report, output_path, indent=2)
            
            self.logger.info(f"✅ Error report exported successfully: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export error report: {e}")
            return False

# Global error handler instance
_global_error_handler: Optional[EnhancedErrorHandler] = None

def get_global_error_handler() -> Optional[EnhancedErrorHandler]:
    """Get global error handler instance."""
    return _global_error_handler

def set_global_error_handler(handler: EnhancedErrorHandler) -> None:
    """Set global error handler instance."""
    global _global_error_handler
    _global_error_handler = handler

def handle_error_globally(error: Exception, context: Dict[str, Any] = None) -> Optional[Any]:
    """Handle error using global error handler."""
    if _global_error_handler:
        return _global_error_handler.handle_error(error, context)
    else:
        # Fallback to basic logging
        logger = system_logger.getChild('GlobalErrorHandler')
        logger.error(f"Unhandled error: {type(error).__name__}: {str(error)}")
        return None