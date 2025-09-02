#!/usr/bin/env python3
"""
Observability Module

This module provides comprehensive observability capabilities including:
- Sentry integration for error tracking and monitoring
- OpenTelemetry integration for distributed tracing and logging
- Centralized logging configuration
- Performance monitoring hooks
"""

import logging
import os
from typing import Any, Optional

from src.utils.warning_symbols import failed


logger = logging.getLogger(__name__)


def init_sentry(dsn: Optional[str] = None) -> bool:
    """Initialize Sentry SDK for error tracking and monitoring.
    
    Args:
        dsn: Sentry DSN (if not provided, will try to get from SENTRY_DSN env var)
        
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    try:
        if dsn is None:
            dsn = os.getenv("SENTRY_DSN")
        
        if not dsn:
            logger.info("No Sentry DSN provided, skipping Sentry initialization")
            return False
        
        # Import Sentry SDK
        import sentry_sdk
        from sentry_sdk.integrations.aiohttp import AioHttpIntegration
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration
        
        # Configure logging integration
        sentry_logging = LoggingIntegration(
            level=logging.INFO,
            event_level=logging.ERROR,
        )
        
        # Initialize Sentry
        sentry_sdk.init(
            dsn=dsn,
            environment=os.getenv("SENTRY_ENV", "production"),
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0")),
            profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.0")),
            integrations=[
                sentry_logging,
                AioHttpIntegration(),
                FastApiIntegration()
            ],
            send_default_pii=False,
        )
        
        logger.info("✅ Sentry initialized successfully")
        return True
        
    except Exception as exc:  # pragma: no cover
        logger.error(f"Failed to initialize Sentry: {exc}")
        print(failed(f"Failed to initialize Sentry: {exc}"))
        return False


def init_otlp_logging(endpoint: Optional[str] = None) -> bool:
    """Initialize OpenTelemetry logging exporter.
    
    Args:
        endpoint: OTLP endpoint (if not provided, will try to get from OTEL_EXPORTER_OTLP_ENDPOINT env var)
        
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    try:
        if endpoint is None:
            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        
        if not endpoint:
            logger.info("No OTLP endpoint provided, skipping OpenTelemetry logging initialization")
            return False
        
        # Import OpenTelemetry modules
        from opentelemetry import _logs as otel_logs
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import Resource
        
        # Create resource
        resource = Resource.create(
            {"service.name": os.getenv("OTEL_SERVICE_NAME", "ares-bot")},
        )
        
        # Create provider and exporter
        provider = LoggerProvider(resource=resource)
        exporter = OTLPLogExporter(endpoint=endpoint)
        
        # Add processor
        provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
        
        # Set global logger provider
        otel_logs.set_logger_provider(provider)
        
        logger.info("✅ OpenTelemetry logging exporter initialized successfully")
        return True
        
    except Exception as exc:  # pragma: no cover
        logger.error(f"Failed to initialize OTLP logging: {exc}")
        print(failed(f"Failed to initialize OTLP logging: {exc}"))
        return False


def init_observability() -> dict:
    """Initialize all observability components.
    
    Returns:
        dict: Status of each component initialization
    """
    logger.info("🚀 Initializing observability components...")
    
    results = {
        "sentry": False,
        "otlp_logging": False,
        "overall_success": False
    }
    
    try:
        # Initialize Sentry
        results["sentry"] = init_sentry()
        
        # Initialize OpenTelemetry logging
        results["otlp_logging"] = init_otlp_logging()
        
        # Determine overall success
        results["overall_success"] = any([results["sentry"], results["otlp_logging"]])
        
        if results["overall_success"]:
            logger.info("✅ Observability initialization completed successfully")
        else:
            logger.warning("⚠️ No observability components were initialized successfully")
        
        return results
        
    except Exception as e:
        logger.exception(f"❌ Error during observability initialization: {e}")
        results["overall_success"] = False
        return results


def get_observability_status() -> dict:
    """Get current status of observability components.
    
    Returns:
        dict: Current status information
    """
    status = {
        "sentry": {
            "enabled": bool(os.getenv("SENTRY_DSN")),
            "environment": os.getenv("SENTRY_ENV", "not_set"),
            "traces_sample_rate": os.getenv("SENTRY_TRACES_SAMPLE_RATE", "not_set"),
            "profiles_sample_rate": os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "not_set")
        },
        "otlp_logging": {
            "enabled": bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")),
            "service_name": os.getenv("OTEL_SERVICE_NAME", "not_set"),
            "endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "not_set")
        },
        "logging": {
            "level": logging.getLogger().getEffectiveLevel(),
            "handlers_count": len(logging.getLogger().handlers)
        }
    }
    
    return status


def configure_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = False,
    log_file_path: Optional[str] = None
) -> None:
    """Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        enable_console: Whether to enable console logging
        enable_file: Whether to enable file logging
        log_file_path: Path to log file (required if enable_file is True)
    """
    try:
        # Set log level
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logging.basicConfig(level=numeric_level)
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Default format
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        formatter = logging.Formatter(format_string)
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if enable_file and log_file_path:
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        logger.info(f"✅ Logging configured successfully (level: {level})")
        
    except Exception as e:
        logger.exception(f"❌ Error configuring logging: {e}")


def create_span_context(operation_name: str, **attributes) -> dict:
    """Create a span context for distributed tracing.
    
    Args:
        operation_name: Name of the operation being traced
        **attributes: Additional attributes for the span
        
    Returns:
        dict: Span context information
    """
    try:
        # Check if OpenTelemetry is available
        try:
            from opentelemetry import trace
            tracer = trace.get_tracer(__name__)
            
            with tracer.start_as_current_span(operation_name, attributes=attributes) as span:
                span_context = {
                    "trace_id": format(span.get_span_context().trace_id, "032x"),
                    "span_id": format(span.get_span_context().span_id, "016x"),
                    "operation_name": operation_name,
                    "attributes": attributes,
                    "active": True
                }
                return span_context
                
        except ImportError:
            # Fallback if OpenTelemetry is not available
            span_context = {
                "trace_id": "00000000000000000000000000000000",
                "span_id": "0000000000000000",
                "operation_name": operation_name,
                "attributes": attributes,
                "active": False,
                "note": "OpenTelemetry not available, using fallback"
            }
            return span_context
            
    except Exception as e:
        logger.exception(f"❌ Error creating span context: {e}")
        return {
            "error": str(e),
            "operation_name": operation_name,
            "active": False
        }


def log_performance_metric(
    metric_name: str,
    value: float,
    unit: str = "count",
    tags: Optional[dict] = None,
    timestamp: Optional[float] = None
) -> bool:
    """Log a performance metric for monitoring.
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        unit: Unit of measurement
        tags: Additional tags/metadata
        timestamp: Timestamp (defaults to current time)
        
    Returns:
        bool: True if logging was successful, False otherwise
    """
    try:
        if timestamp is None:
            import time
            timestamp = time.time()
        
        if tags is None:
            tags = {}
        
        # Log the metric
        logger.info(
            f"📊 METRIC: {metric_name}={value} {unit} "
            f"tags={tags} timestamp={timestamp}"
        )
        
        # If OpenTelemetry metrics are available, send there too
        try:
            from opentelemetry import metrics
            meter = metrics.get_meter(__name__)
            
            # Create or get counter
            counter = meter.create_counter(
                name=metric_name,
                description=f"Performance metric: {metric_name}",
                unit=unit
            )
            
            # Record the value
            counter.add(value, tags)
            
        except ImportError:
            # OpenTelemetry metrics not available, continue with logging only
            pass
        
        return True
        
    except Exception as e:
        logger.exception(f"❌ Error logging performance metric: {e}")
        return False


def log_error_with_context(
    error: Exception,
    context: str = "unknown",
    additional_data: Optional[dict] = None,
    severity: str = "ERROR"
) -> None:
    """Log an error with additional context for better debugging.
    
    Args:
        error: The exception that occurred
        context: Context where the error occurred
        additional_data: Additional data to log
        severity: Log severity level
    """
    try:
        log_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "severity": severity
        }
        
        if additional_data:
            log_data.update(additional_data)
        
        # Log with appropriate level
        if severity.upper() == "CRITICAL":
            logger.critical(f"🚨 CRITICAL ERROR in {context}: {error}", extra=log_data)
        elif severity.upper() == "ERROR":
            logger.error(f"❌ Error in {context}: {error}", extra=log_data)
        elif severity.upper() == "WARNING":
            logger.warning(f"⚠️ Warning in {context}: {error}", extra=log_data)
        else:
            logger.info(f"ℹ️ Info in {context}: {error}", extra=log_data)
        
        # If Sentry is available, capture the exception
        try:
            import sentry_sdk
            if sentry_sdk.Hub.current.client:
                sentry_sdk.capture_exception(error)
        except ImportError:
            pass
        
    except Exception as e:
        # Fallback logging if the structured logging fails
        logger.exception(f"Failed to log error with context: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    configure_logging(level="DEBUG", enable_console=True)
    
    # Initialize observability
    init_results = init_observability()
    print(f"Observability initialization results: {init_results}")
    
    # Get status
    status = get_observability_status()
    print(f"Observability status: {status}")
    
    # Test span context creation
    span_ctx = create_span_context("test_operation", test_attr="value")
    print(f"Span context: {span_ctx}")
    
    # Test performance metric logging
    log_performance_metric("test_metric", 42.0, "requests", {"service": "test"})
    
    # Test error logging
    try:
        raise ValueError("Test error for observability")
    except Exception as e:
        log_error_with_context(e, "test_context", {"test_data": "value"})
