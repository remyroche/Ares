"""
Enhanced Error Handling System for Step03.

This module provides comprehensive error handling with:
1. Detailed error categorization and analysis
2. Automatic error recovery strategies
3. Error context preservation
4. Detailed error reporting
5. Error pattern detection
6. Performance impact analysis
"""
import asyncio
import functools
import inspect
import logging
import sys
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from .compose import P, R, uniform_wrapper
# Optional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

error_context: ContextVar[Dict[str, Any]] = ContextVar('error_context', default={})
error_history: ContextVar[List[Dict[str, Any]]] = ContextVar('error_history', default=[])

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'

class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = 'validation'
    DATA_QUALITY = 'data_quality'
    PERFORMANCE = 'performance'
    RESOURCE = 'resource'
    NETWORK = 'network'
    AUTHENTICATION = 'authentication'
    AUTHORIZATION = 'authorization'
    BUSINESS_LOGIC = 'business_logic'
    SYSTEM = 'system'
    UNKNOWN = 'unknown'

class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = 'retry'
    FALLBACK = 'fallback'
    SKIP = 'skip'
    ABORT = 'abort'
    MANUAL_INTERVENTION = 'manual_intervention'

@dataclass
class ErrorContext:
    """Comprehensive error context information."""
    error_id: str
    timestamp: datetime
    function_name: str
    module_name: str
    error_type: str
    error_message: str
    error_category: ErrorCategory
    severity: ErrorSeverity
    stack_trace: str
    function_parameters: Dict[str, Any] = field(default_factory = dict)
    local_variables: Dict[str, Any] = field(default_factory = dict)
    system_state: Dict[str, Any] = field(default_factory = dict)
    recovery_attempts: List[Dict[str, Any]] = field(default_factory = list)
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_successful: bool = False
    performance_impact: Dict[str, Any] = field(default_factory = dict)
    related_errors: List[str] = field(default_factory = list)

@dataclass
class ErrorReport:
    """Comprehensive error report."""
    report_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_errors: int = 0
    errors_by_category: Dict[str, int] = field(default_factory = dict)
    errors_by_severity: Dict[str, int] = field(default_factory = dict)
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    performance_impact_total: float = 0.0
    error_patterns: List[Dict[str, Any]] = field(default_factory = list)
    recommendations: List[str] = field(default_factory = list)
    detailed_errors: List[ErrorContext] = field(default_factory = list)

class EnhancedErrorHandler:
    """Enhanced error handling system with comprehensive analysis and recovery."""

    def __init__(self, enable_automatic_recovery: bool = True, enable_error_pattern_detection: bool = True, enable_performance_impact_analysis: bool = True, max_recovery_attempts: int = 3, recovery_timeout_seconds: float = 30.0, log_level: str='INFO', generate_error_reports: bool = True, error_report_path: Optional[str]=None) -> None:
        """
        Initialize the enhanced error handler.
        
        Args:
            enable_automatic_recovery: Enable automatic error recovery
            enable_error_pattern_detection: Enable error pattern detection
            enable_performance_impact_analysis: Enable performance impact analysis
            max_recovery_attempts: Maximum number of recovery attempts
            recovery_timeout_seconds: Timeout for recovery operations
            log_level: Logging level for error handling
            generate_error_reports: Generate detailed error reports
            error_report_path: Path to save error reports
        """
        self.enable_automatic_recovery = enable_automatic_recovery
        self.enable_error_pattern_detection = enable_error_pattern_detection
        self.enable_performance_impact_analysis = enable_performance_impact_analysis
        self.max_recovery_attempts = max_recovery_attempts
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.log_level = log_level
        self.generate_error_reports = generate_error_reports
        self.error_report_path = error_report_path
        self.logger = logging.getLogger(f'{__name__}.EnhancedErrorHandler')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.error_patterns = self._initialize_error_patterns()
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.error_history: List[ErrorContext] = []

    def _initialize_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize error patterns for classification."""
        return {'validation_error': {'patterns': ['validation', 'invalid', 'required', 'missing', 'format'], 'category': ErrorCategory.VALIDATION, 'severity': ErrorSeverity.MEDIUM, 'recovery_strategy': RecoveryStrategy.RETRY}, 'data_quality_error': {'patterns': ['empty', 'null', 'nan', 'corrupt', 'incomplete'], 'category': ErrorCategory.DATA_QUALITY, 'severity': ErrorSeverity.HIGH, 'recovery_strategy': RecoveryStrategy.FALLBACK}, 'performance_error': {'patterns': ['timeout', 'slow', 'memory', 'cpu', 'resource'], 'category': ErrorCategory.PERFORMANCE, 'severity': ErrorSeverity.MEDIUM, 'recovery_strategy': RecoveryStrategy.RETRY}, 'resource_error': {'patterns': ['memory', 'disk', 'connection', 'file', 'permission'], 'category': ErrorCategory.RESOURCE, 'severity': ErrorSeverity.HIGH, 'recovery_strategy': RecoveryStrategy.FALLBACK}, 'network_error': {'patterns': ['connection', 'network', 'timeout', 'unreachable', 'refused'], 'category': ErrorCategory.NETWORK, 'severity': ErrorSeverity.MEDIUM, 'recovery_strategy': RecoveryStrategy.RETRY}, 'authentication_error': {'patterns': ['auth', 'login', 'credential', 'token', 'unauthorized'], 'category': ErrorCategory.AUTHENTICATION, 'severity': ErrorSeverity.HIGH, 'recovery_strategy': RecoveryStrategy.MANUAL_INTERVENTION}, 'authorization_error': {'patterns': ['permission', 'forbidden', 'access', 'denied'], 'category': ErrorCategory.AUTHORIZATION, 'severity': ErrorSeverity.HIGH, 'recovery_strategy': RecoveryStrategy.MANUAL_INTERVENTION}, 'business_logic_error': {'patterns': ['business', 'logic', 'rule', 'constraint', 'violation'], 'category': ErrorCategory.BUSINESS_LOGIC, 'severity': ErrorSeverity.MEDIUM, 'recovery_strategy': RecoveryStrategy.SKIP}, 'system_error': {'patterns': ['system', 'internal', 'fatal', 'critical', 'exception'], 'category': ErrorCategory.SYSTEM, 'severity': ErrorSeverity.CRITICAL, 'recovery_strategy': RecoveryStrategy.ABORT}}

    def _initialize_recovery_strategies(self) -> Dict[RecoveryStrategy, Callable]:
        """Initialize recovery strategy implementations."""
        return {RecoveryStrategy.RETRY: self._retry_strategy, RecoveryStrategy.FALLBACK: self._fallback_strategy, RecoveryStrategy.SKIP: self._skip_strategy, RecoveryStrategy.ABORT: self._abort_strategy, RecoveryStrategy.MANUAL_INTERVENTION: self._manual_intervention_strategy}

    def _classify_error(self, error: Exception) -> Tuple[ErrorCategory, ErrorSeverity, RecoveryStrategy]:
        """Classify error based on patterns and context."""
        error_message = str(error).lower()
        error_type = type(error).__name__.lower()
        for pattern_name, pattern_info in self.error_patterns.items():
            for pattern in pattern_info['patterns']:
                if pattern in error_message or pattern in error_type:
                    return (pattern_info['category'], pattern_info['severity'], pattern_info['recovery_strategy'])
        return (ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM, RecoveryStrategy.RETRY)

    def _create_error_context(self, error: Exception, func: Callable, args: tuple, kwargs: dict, local_vars: Optional[Dict[str, Any]]=None) -> ErrorContext:
        """Create comprehensive error context."""
        error_id = str(uuid.uuid4())
        category, severity, recovery_strategy = self._classify_error(error)
        function_parameters = {}
        try:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            function_parameters = dict(bound_args.arguments)
        except Exception:
            function_parameters = {'args': str(args), 'kwargs': str(kwargs)}
        system_state = {'timestamp': datetime.now().isoformat(), 'python_version': sys.version, 'platform': sys.platform, 'memory_usage': self._get_memory_usage(), 'cpu_usage': self._get_cpu_usage()}
        return ErrorContext(error_id = error_id, timestamp = datetime.now(), function_name = func.__name__, module_name = func.__module__, error_type = type(error).__name__, error_message = str(error), error_category = category, severity = severity, stack_trace = traceback.format_exc(), function_parameters = function_parameters, local_variables = local_vars or {}, system_state = system_state, recovery_strategy = recovery_strategy)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            process = psutil.Process()
            return process.cpu_percent()
        except Exception:
            return 0.0

    def _log_error_context(self, error_context: ErrorContext) -> None:
        """Log comprehensive error context."""
        self.logger.error(f'💥 ERROR DETECTED: {error_context.function_name}')
        self.logger.error(f'   🆔 Error ID: {error_context.error_id}')
        self.logger.error(f'   📍 Location: {error_context.module_name}.{error_context.function_name}')
        self.logger.error(f'   🏷️ Type: {error_context.error_type}')
        self.logger.error(f'   📝 Message: {error_context.error_message}')
        self.logger.error(f'   📊 Category: {error_context.error_category.value}')
        self.logger.error(f'   ⚠️ Severity: {error_context.severity.value}')
        self.logger.error(f'   🔄 Recovery Strategy: {error_context.recovery_strategy.value}')
        self.logger.error(f'   ⏰ Timestamp: {error_context.timestamp.isoformat()}')
        if error_context.function_parameters:
            self.logger.error(f'   📋 Parameters ({len(error_context.function_parameters)}):')
            for param_name, param_value in list(error_context.function_parameters.items())[:5]:
                self.logger.error(f'      - {param_name}: {str(param_value)[:100]}...')
            if len(error_context.function_parameters) > 5:
                self.logger.error(f'      ... and {len(error_context.function_parameters) - 5} more parameters')
        if error_context.system_state:
            self.logger.error(f'   🖥️ System State:')
            self.logger.error(f"      - Memory: {error_context.system_state.get('memory_usage', 0):.2f} MB")
            self.logger.error(f"      - CPU: {error_context.system_state.get('cpu_usage', 0):.2f}%")
        self.logger.error(f'   📍 Stack Trace:')
        for line in error_context.stack_trace.split('\n')[-10:]:
            if line.strip():
                self.logger.error(f'      {line}')

    def _retry_strategy(self, error_context: ErrorContext, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Retry strategy implementation."""
        self.logger.info(f'🔄 Attempting retry strategy for error {error_context.error_id}')
        for attempt in range(self.max_recovery_attempts):
            try:
                self.logger.info(f'   🔄 Retry attempt {attempt + 1}/{self.max_recovery_attempts}')
                if attempt > 0:
                    import time
                    time.sleep(min(2 ** attempt, 10))
                if asyncio.iscoroutinefunction(func):
                    return asyncio.run(func(*args, **kwargs))
                else:
                    return func(*args, **kwargs)
            except Exception as retry_error:
                self.logger.warning(f'   ⚠️ Retry attempt {attempt + 1} failed: {retry_error}')
                if attempt == self.max_recovery_attempts - 1:
                    self.logger.error(f'   ❌ All retry attempts failed for error {error_context.error_id}')
                    raise retry_error
        return None

    def _fallback_strategy(self, error_context: ErrorContext, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Fallback strategy implementation."""
        self.logger.info(f'🔄 Attempting fallback strategy for error {error_context.error_id}')
        fallback_value = self._get_fallback_value(func, error_context)
        if fallback_value is not None:
            self.logger.info(f'   ✅ Fallback value provided: {type(fallback_value).__name__}')
            return fallback_value
        else:
            self.logger.error(f'   ❌ No fallback value available for error {error_context.error_id}')
            raise error_context

    def _skip_strategy(self, error_context: ErrorContext, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Skip strategy implementation."""
        self.logger.info(f'⏭️ Skipping function execution due to error {error_context.error_id}')
        return None

    def _abort_strategy(self, error_context: ErrorContext, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Abort strategy implementation."""
        self.logger.critical(f'🛑 Aborting execution due to critical error {error_context.error_id}')
        raise error_context

    def _manual_intervention_strategy(self, error_context: ErrorContext, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Manual intervention strategy implementation."""
        self.logger.critical(f'🚨 Manual intervention required for error {error_context.error_id}')
        self.logger.critical(f'   📧 Please contact system administrator')
        self.logger.critical(f'   🆔 Error ID: {error_context.error_id}')
        raise error_context

    def _get_fallback_value(self, func: Callable, error_context: ErrorContext) -> Any:
        """Get appropriate fallback value based on function signature."""
        try:
            sig = inspect.signature(func)
            return_type = sig.return_annotation
            if return_type == bool:
                return False
            elif return_type == int:
                return 0
            elif return_type == float:
                return 0.0
            elif return_type == str:
                return ''
            elif return_type == list:
                return []
            elif return_type == dict:
                return {}
            elif return_type == tuple:
                return ()
            else:
                return None
        except Exception:
            return None

    def _detect_error_patterns(self) -> List[Dict[str, Any]]:
        """Detect error patterns in recent error history."""
        if not self.enable_error_pattern_detection:
            return []
        patterns = []
        function_errors = {}
        for error in self.error_history[-50:]:
            key = f'{error.module_name}.{error.function_name}'
            if key not in function_errors:
                function_errors[key] = []
            function_errors[key].append(error)
        for function_name, errors in function_errors.items():
            if len(errors) >= 3:
                patterns.append({'pattern_type': 'frequent_errors', 'function_name': function_name, 'error_count': len(errors), 'error_categories': [e.error_category.value for e in errors], 'severity': max((e.severity.value for e in errors)), 'recommendation': f'Function {function_name} has {len(errors)} recent errors'})
        return patterns

    def _generate_error_report(self, error_context: ErrorContext) -> None:
        """Generate detailed error report."""
        if not self.generate_error_reports:
            return
        try:
            self.error_history.append(error_context)
            patterns = self._detect_error_patterns()
            report = ErrorReport(report_id = str(uuid.uuid4()), start_time = datetime.now(), total_errors = len(self.error_history), detailed_errors=[error_context], error_patterns = patterns)
            for error in self.error_history:
                category = error.error_category.value
                severity = error.severity.value
                report.errors_by_category[category] = report.errors_by_category.get(category, 0) + 1
                report.errors_by_severity[severity] = report.errors_by_severity.get(severity, 0) + 1
            if self.error_report_path:
                report_file = Path(self.error_report_path) / f'error_report_{error_context.error_id}.json'
                report_file.parent.mkdir(parents = True, exist_ok = True)
                import json

                report_data = {'report_id': report.report_id, 'start_time': report.start_time.isoformat(), 'total_errors': report.total_errors, 'errors_by_category': report.errors_by_category, 'errors_by_severity': report.errors_by_severity, 'error_patterns': report.error_patterns, 'detailed_errors': [{'error_id': e.error_id, 'timestamp': e.timestamp.isoformat(), 'function_name': e.function_name, 'module_name': e.module_name, 'error_type': e.error_type, 'error_message': e.error_message, 'error_category': e.error_category.value, 'severity': e.severity.value, 'recovery_strategy': e.recovery_strategy.value if e.recovery_strategy else None, 'recovery_successful': e.recovery_successful} for e in report.detailed_errors]}
                with open(report_file, 'w') as f:
                    json.dump(report_data, f, indent = 2, default = str)
                self.logger.info(f'📊 Error report saved to: {report_file}')
        except Exception as e:
            self.logger.error(f'Failed to generate error report: {e}')

    def handle_errors(self, func: Callable[P, R]) -> Callable[P, R]:
        """Main decorator for enhanced error handling."""

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return self._handle_sync_errors(func, args, kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return await self._handle_async_errors(func, args, kwargs)
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    def _handle_sync_errors(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Handle errors in synchronous functions."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return self._process_error(e, func, args, kwargs)

    async def _handle_async_errors(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Handle errors in asynchronous functions."""
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            return self._process_error(e, func, args, kwargs)

    def _process_error(self, error: Exception, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Process error with comprehensive handling."""
        error_context = self._create_error_context(error, func, args, kwargs)
        self._log_error_context(error_context)
        self._generate_error_report(error_context)
        if self.enable_automatic_recovery and error_context.recovery_strategy:
            try:
                recovery_func = self.recovery_strategies[error_context.recovery_strategy]
                result = recovery_func(error_context, func, args, kwargs)
                error_context.recovery_successful = True
                self.logger.info(f'✅ Recovery successful for error {error_context.error_id}')
                return result
            except Exception as recovery_error:
                error_context.recovery_successful = False
                self.logger.error(f'❌ Recovery failed for error {error_context.error_id}: {recovery_error}')
        raise error

def handle_errors_enhanced(enable_automatic_recovery: bool = True, enable_error_pattern_detection: bool = True, enable_performance_impact_analysis: bool = True, max_recovery_attempts: int = 3, recovery_timeout_seconds: float = 30.0, log_level: str='INFO', generate_error_reports: bool = True, error_report_path: Optional[str]=None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator factory for enhanced error handling.
    
    Args:
        enable_automatic_recovery: Enable automatic error recovery
        enable_error_pattern_detection: Enable error pattern detection
        enable_performance_impact_analysis: Enable performance impact analysis
        max_recovery_attempts: Maximum number of recovery attempts
        recovery_timeout_seconds: Timeout for recovery operations
        log_level: Logging level for error handling
        generate_error_reports: Generate detailed error reports
        error_report_path: Path to save error reports
    
    Returns:
        Decorator function for enhanced error handling
    """
    handler = EnhancedErrorHandler(enable_automatic_recovery = enable_automatic_recovery, enable_error_pattern_detection = enable_error_pattern_detection, enable_performance_impact_analysis = enable_performance_impact_analysis, max_recovery_attempts = max_recovery_attempts, recovery_timeout_seconds = recovery_timeout_seconds, log_level = log_level, generate_error_reports = generate_error_reports, error_report_path = error_report_path)
    return handler.handle_errors

def handle_step03_errors(func: Callable[P, R]) -> Callable[P, R]:
    """Specialized decorator for step03 functions with enhanced error handling."""
    return handle_errors_enhanced(enable_automatic_recovery = True, enable_error_pattern_detection = True, enable_performance_impact_analysis = True, max_recovery_attempts = 3, recovery_timeout_seconds = 30.0, log_level='INFO', generate_error_reports = True, error_report_path='logs/step03_errors')(func)

def handle_critical_errors(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator for critical functions with maximum error handling."""
    return handle_errors_enhanced(enable_automatic_recovery = True, enable_error_pattern_detection = True, enable_performance_impact_analysis = True, max_recovery_attempts = 5, recovery_timeout_seconds = 60.0, log_level='DEBUG', generate_error_reports = True, error_report_path='logs/critical_errors')(func)

def handle_errors_with_retry(func: Callable[P, R]) -> Callable[P, R]:
    """Lightweight decorator for error handling with retry only."""
    return handle_errors_enhanced(enable_automatic_recovery = True, enable_error_pattern_detection = False, enable_performance_impact_analysis = False, max_recovery_attempts = 3, recovery_timeout_seconds = 10.0, log_level='WARNING', generate_error_reports = False)(func)