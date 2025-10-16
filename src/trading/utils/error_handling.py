"""
Trading Error Handling

Comprehensive error handling system for trading operations
with no fallbacks and important warnings for critical failures.
"""

import logging
import traceback
from typing import Any, Dict, Optional, Callable, Union, Type
from functools import wraps
from datetime import datetime
from enum import Enum

from src.utils.logger import system_logger
from src.utils.tprint import tprint_error, tprint_warning, tprint_info, tprint_structured, LogLevel

logger = system_logger.getChild('TradingErrorHandling')

class TradingErrorSeverity(Enum):
    """Error severity levels for trading operations."""
    CRITICAL = "critical"  # System-breaking errors that require immediate attention
    HIGH = "high"         # Trading-stopping errors
    MEDIUM = "medium"     # Component-level errors
    LOW = "low"          # Recoverable errors
    WARNING = "warning"   # Non-critical issues

class TradingError(Exception):
    """Base trading error class."""

    def __init__(
        self,
        message: str,
        error_code: str = "TRADING_ERROR",
        severity: TradingErrorSeverity = TradingErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.context = context or {}
        self.original_exception = original_exception
        self.timestamp = datetime.now()

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging."""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'original_exception': str(self.original_exception) if self.original_exception else None,
            'traceback': traceback.format_exc() if self.original_exception else None
        }

class RegimeDetectionError(TradingError):
    """Regime detection specific errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="REGIME_DETECTION_ERROR",
            severity=kwargs.get('severity', TradingErrorSeverity.HIGH),
            **kwargs
        )

class SignalGenerationError(TradingError):
    """Signal generation specific errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="SIGNAL_GENERATION_ERROR",
            severity=kwargs.get('severity', TradingErrorSeverity.HIGH),
            **kwargs
        )

class PositionSizingError(TradingError):
    """Position sizing specific errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="POSITION_SIZING_ERROR",
            severity=kwargs.get('severity', TradingErrorSeverity.HIGH),
            **kwargs
        )

class ExecutionError(TradingError):
    """Order execution specific errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="EXECUTION_ERROR",
            severity=kwargs.get('severity', TradingErrorSeverity.CRITICAL),
            **kwargs
        )

class DataCollectionError(TradingError):
    """Data collection specific errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="DATA_COLLECTION_ERROR",
            severity=kwargs.get('severity', TradingErrorSeverity.MEDIUM),
            **kwargs
        )

class ConfigurationError(TradingError):
    """Configuration specific errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            severity=kwargs.get('severity', TradingErrorSeverity.CRITICAL),
            **kwargs
        )

class ValidationError(TradingError):
    """Validation specific errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            severity=kwargs.get('severity', TradingErrorSeverity.HIGH),
            **kwargs
        )

def trading_error_handler(
    error_types: Union[Type[Exception], tuple] = Exception,
    severity: TradingErrorSeverity = TradingErrorSeverity.MEDIUM,
    raise_on_error: bool = True,
    log_traceback: bool = True,
    context_extractor: Optional[Callable] = None
):
    """
    Comprehensive error handler decorator for trading operations.

    Args:
        error_types: Exception types to catch
        severity: Error severity level
        raise_on_error: Whether to re-raise the error after handling
        log_traceback: Whether to log full traceback
        context_extractor: Function to extract context from function args
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except error_types as e:
                await _handle_trading_error(
                    e, func, args, kwargs, severity,
                    raise_on_error, log_traceback, context_extractor
                )

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                _handle_trading_error_sync(
                    e, func, args, kwargs, severity,
                    raise_on_error, log_traceback, context_extractor
                )

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

async def _handle_trading_error(
    error: Exception,
    func: Callable,
    args: tuple,
    kwargs: dict,
    severity: TradingErrorSeverity,
    raise_on_error: bool,
    log_traceback: bool,
    context_extractor: Optional[Callable]
):
    """Handle trading error asynchronously."""
    # Extract context
    context = {}
    if context_extractor:
        try:
            context = context_extractor(*args, **kwargs)
        except Exception as ctx_error:
            logger.warning(f"Failed to extract context: {ctx_error}")

    # Create trading error
    if isinstance(error, TradingError):
        trading_error = error
    else:
        trading_error = TradingError(
            message=str(error),
            severity=severity,
            context=context,
            original_exception=error
        )

    # Log error based on severity
    await _log_trading_error(trading_error, func, log_traceback)

    # Handle critical errors
    if severity == TradingErrorSeverity.CRITICAL:
        await _handle_critical_error(trading_error, func)

    # Re-raise if requested
    if raise_on_error:
        raise trading_error

def _handle_trading_error_sync(
    error: Exception,
    func: Callable,
    args: tuple,
    kwargs: dict,
    severity: TradingErrorSeverity,
    raise_on_error: bool,
    log_traceback: bool,
    context_extractor: Optional[Callable]
):
    """Handle trading error synchronously."""
    # Extract context
    context = {}
    if context_extractor:
        try:
            context = context_extractor(*args, **kwargs)
        except Exception as ctx_error:
            logger.warning(f"Failed to extract context: {ctx_error}")

    # Create trading error
    if isinstance(error, TradingError):
        trading_error = error
    else:
        trading_error = TradingError(
            message=str(error),
            severity=severity,
            context=context,
            original_exception=error
        )

    # Log error based on severity
    _log_trading_error_sync(trading_error, func, log_traceback)

    # Handle critical errors
    if severity == TradingErrorSeverity.CRITICAL:
        _handle_critical_error_sync(trading_error, func)

    # Re-raise if requested
    if raise_on_error:
        raise trading_error

async def _log_trading_error(
    error: TradingError,
    func: Callable,
    log_traceback: bool
):
    """Log trading error with appropriate severity."""
    error_dict = error.to_dict()
    error_dict['function'] = func.__name__
    error_dict['module'] = func.__module__

    # Print based on severity
    if error.severity == TradingErrorSeverity.CRITICAL:
        tprint_error(f"🚨 CRITICAL ERROR in {func.__name__}: {error.message}")
        tprint_structured(error_dict, LogLevel.ERROR)

        # Also log to system logger
        logger.critical(f"CRITICAL TRADING ERROR: {error}")
        if log_traceback and error.original_exception:
            logger.critical(f"Traceback: {traceback.format_exc()}")

    elif error.severity == TradingErrorSeverity.HIGH:
        tprint_error(f"❌ HIGH SEVERITY ERROR in {func.__name__}: {error.message}")
        logger.error(f"HIGH SEVERITY TRADING ERROR: {error}")
        if log_traceback and error.original_exception:
            logger.error(f"Traceback: {traceback.format_exc()}")

    elif error.severity == TradingErrorSeverity.MEDIUM:
        tprint_warning(f"⚠️ MEDIUM SEVERITY ERROR in {func.__name__}: {error.message}")
        logger.warning(f"MEDIUM SEVERITY TRADING ERROR: {error}")

    elif error.severity == TradingErrorSeverity.LOW:
        tprint_info(f"ℹ️ LOW SEVERITY ERROR in {func.__name__}: {error.message}")
        logger.info(f"LOW SEVERITY TRADING ERROR: {error}")

    else:  # WARNING
        tprint_warning(f"⚠️ WARNING in {func.__name__}: {error.message}")
        logger.warning(f"TRADING WARNING: {error}")

def _log_trading_error_sync(
    error: TradingError,
    func: Callable,
    log_traceback: bool
):
    """Log trading error synchronously."""
    error_dict = error.to_dict()
    error_dict['function'] = func.__name__
    error_dict['module'] = func.__module__

    # Print based on severity
    if error.severity == TradingErrorSeverity.CRITICAL:
        tprint_error(f"🚨 CRITICAL ERROR in {func.__name__}: {error.message}")
        tprint_structured(error_dict, LogLevel.ERROR)

        # Also log to system logger
        logger.critical(f"CRITICAL TRADING ERROR: {error}")
        if log_traceback and error.original_exception:
            logger.critical(f"Traceback: {traceback.format_exc()}")

    elif error.severity == TradingErrorSeverity.HIGH:
        tprint_error(f"❌ HIGH SEVERITY ERROR in {func.__name__}: {error.message}")
        logger.error(f"HIGH SEVERITY TRADING ERROR: {error}")
        if log_traceback and error.original_exception:
            logger.error(f"Traceback: {traceback.format_exc()}")

    elif error.severity == TradingErrorSeverity.MEDIUM:
        tprint_warning(f"⚠️ MEDIUM SEVERITY ERROR in {func.__name__}: {error.message}")
        logger.warning(f"MEDIUM SEVERITY TRADING ERROR: {error}")

    elif error.severity == TradingErrorSeverity.LOW:
        tprint_info(f"ℹ️ LOW SEVERITY ERROR in {func.__name__}: {error.message}")
        logger.info(f"LOW SEVERITY TRADING ERROR: {error}")

    else:  # WARNING
        tprint_warning(f"⚠️ WARNING in {func.__name__}: {error.message}")
        logger.warning(f"TRADING WARNING: {error}")

async def _handle_critical_error(error: TradingError, func: Callable):
    """Handle critical errors that require immediate attention."""
    tprint_error("🚨 CRITICAL TRADING ERROR DETECTED!")
    tprint_error("🛑 TRADING OPERATIONS MAY BE COMPROMISED!")
    tprint_error(f"🔍 Error in function: {func.__name__}")
    tprint_error(f"💥 Error message: {error.message}")

    # Log critical error details
    logger.critical("=" * 80)
    logger.critical("CRITICAL TRADING ERROR - IMMEDIATE ATTENTION REQUIRED")
    logger.critical("=" * 80)
    logger.critical(f"Function: {func.__name__}")
    logger.critical(f"Module: {func.__module__}")
    logger.critical(f"Error: {error}")
    logger.critical(f"Context: {error.context}")
    logger.critical("=" * 80)

    # In a production system, you might want to:
    # 1. Send alerts to monitoring systems
    # 2. Stop trading operations
    # 3. Notify administrators
    # 4. Save state for recovery

def _handle_critical_error_sync(error: TradingError, func: Callable):
    """Handle critical errors synchronously."""
    tprint_error("🚨 CRITICAL TRADING ERROR DETECTED!")
    tprint_error("🛑 TRADING OPERATIONS MAY BE COMPROMISED!")
    tprint_error(f"🔍 Error in function: {func.__name__}")
    tprint_error(f"💥 Error message: {error.message}")

    # Log critical error details
    logger.critical("=" * 80)
    logger.critical("CRITICAL TRADING ERROR - IMMEDIATE ATTENTION REQUIRED")
    logger.critical("=" * 80)
    logger.critical(f"Function: {func.__name__}")
    logger.critical(f"Module: {func.__module__}")
    logger.critical(f"Error: {error}")
    logger.critical(f"Context: {error.context}")
    logger.critical("=" * 80)

def require_no_fallback(message: str = "Operation failed with no fallback available"):
    """
    Decorator that ensures no fallback behavior is used.
    Raises TradingError immediately on any exception.
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                raise TradingError(
                    message=f"{message}: {str(e)}",
                    error_code="NO_FALLBACK_ERROR",
                    severity=TradingErrorSeverity.HIGH,
                    original_exception=e
                )

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                raise TradingError(
                    message=f"{message}: {str(e)}",
                    error_code="NO_FALLBACK_ERROR",
                    severity=TradingErrorSeverity.HIGH,
                    original_exception=e
                )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

def critical_operation(func):
    """
    Decorator for critical trading operations that must succeed.
    Any failure is treated as a critical error.
    """
    return trading_error_handler(
        severity=TradingErrorSeverity.CRITICAL,
        raise_on_error=True,
        log_traceback=True
    )(func)

def warn_on_failure(message: str = "Operation completed with warnings"):
    """
    Decorator that logs warnings but doesn't raise exceptions.
    Use sparingly and only for non-critical operations.
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                tprint_warning(f"⚠️ {message}: {str(e)}")
                logger.warning(f"{message}: {str(e)}")
                return None

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                tprint_warning(f"⚠️ {message}: {str(e)}")
                logger.warning(f"{message}: {str(e)}")
                return None

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

# Context extractors for common trading operations
def extract_symbol_context(*args, **kwargs) -> Dict[str, Any]:
    """Extract symbol context from function arguments."""
    context = {}

    # Look for symbol in args and kwargs
    if args and isinstance(args[0], str):
        context['symbol'] = args[0]
    elif 'symbol' in kwargs:
        context['symbol'] = kwargs['symbol']

    return context

def extract_market_data_context(*args, **kwargs) -> Dict[str, Any]:
    """Extract market data context from function arguments."""
    context = {}

    # Look for market data
    for arg in args:
        if hasattr(arg, 'shape') and len(arg.shape) == 2:  # Likely DataFrame
            context['data_shape'] = arg.shape
            context['data_columns'] = list(arg.columns) if hasattr(arg, 'columns') else None
            break

    if 'market_data' in kwargs:
        data = kwargs['market_data']
        if hasattr(data, 'shape'):
            context['data_shape'] = data.shape
            context['data_columns'] = list(data.columns) if hasattr(data, 'columns') else None

    return context
