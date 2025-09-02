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
from typing import Any, Optional, Dict, List

# Dependency imports with graceful fallbacks
try:
    from src.utils.warning_symbols import failed
    WARNING_SYMBOLS_AVAILABLE = True
except ImportError:
    WARNING_SYMBOLS_AVAILABLE = False
    # Create a fallback warning function
    def failed(message: str) -> str:
        return f"❌ {message}"

try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

try:
    from opentelemetry import _logs as otel_logs
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.resources import Resource
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

try:
    from opentelemetry import trace
    from opentelemetry import metrics
    OPENTELEMETRY_TRACE_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_TRACE_AVAILABLE = False


logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


class DependencyError(Exception):
    """Raised when required dependencies are not available."""
    pass


def validate_observability_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate and sanitize observability configuration.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Validated and sanitized configuration
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    if config is None:
        config = {}
    
    # Default configuration
    default_config = {
        "sentry": {
            "enabled": bool(os.getenv("SENTRY_DSN")),
            "dsn": os.getenv("SENTRY_DSN"),
            "environment": os.getenv("SENTRY_ENV", "production"),
            "traces_sample_rate": float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0")),
            "profiles_sample_rate": float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.0")),
            "send_default_pii": False
        },
        "opentelemetry": {
            "enabled": bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")),
            "endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            "service_name": os.getenv("OTEL_SERVICE_NAME", "ares-bot"),
            "traces_enabled": True,
            "metrics_enabled": True,
            "logs_enabled": True
        },
        "logging": {
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "format": os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            "enable_console": True,
            "enable_file": False,
            "log_file_path": None,
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "backup_count": 5
        }
    }
    
    # Merge with provided config
    if "sentry" in config:
        default_config["sentry"].update(config["sentry"])
    
    if "opentelemetry" in config:
        default_config["opentelemetry"].update(config["opentelemetry"])
    
    if "logging" in config:
        default_config["logging"].update(config["logging"])
    
    # Validate Sentry configuration
    sentry_config = default_config["sentry"]
    if sentry_config["enabled"] and not sentry_config["dsn"]:
        raise ConfigurationError("Sentry enabled but no DSN provided")
    
    if not (0.0 <= sentry_config["traces_sample_rate"] <= 1.0):
        raise ConfigurationError("Sentry traces_sample_rate must be between 0.0 and 1.0")
    
    if not (0.0 <= sentry_config["profiles_sample_rate"] <= 1.0):
        raise ConfigurationError("Sentry profiles_sample_rate must be between 0.0 and 1.0")
    
    # Validate OpenTelemetry configuration
    otel_config = default_config["opentelemetry"]
    if otel_config["enabled"] and not otel_config["endpoint"]:
        raise ConfigurationError("OpenTelemetry enabled but no endpoint provided")
    
    # Validate logging configuration
    logging_config = default_config["logging"]
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if logging_config["level"].upper() not in valid_levels:
        raise ConfigurationError(f"Invalid log level: {logging_config['level']}. Valid: {valid_levels}")
    
    if logging_config["enable_file"] and not logging_config["log_file_path"]:
        raise ConfigurationError("File logging enabled but no log file path provided")
    
    return default_config


def check_dependencies() -> Dict[str, bool]:
    """Check availability of required dependencies.
    
    Returns:
        Dictionary mapping dependency names to availability status
    """
    dependencies = {
        "sentry_sdk": SENTRY_AVAILABLE,
        "opentelemetry": OPENTELEMETRY_AVAILABLE,
        "opentelemetry_trace": OPENTELEMETRY_TRACE_AVAILABLE,
        "warning_symbols": WARNING_SYMBOLS_AVAILABLE,
        "logging": True,  # Built-in
        "os": True,       # Built-in
        "time": True      # Built-in
    }
    
    return dependencies


def init_sentry(dsn: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize Sentry SDK for error tracking and monitoring.
    
    Args:
        dsn: Sentry DSN (if not provided, will try to get from SENTRY_DSN env var)
        config: Configuration dictionary
        
    Returns:
        bool: True if initialization was successful, False otherwise
        
    Raises:
        DependencyError: If Sentry SDK is not available
    """
    try:
        if not SENTRY_AVAILABLE:
            logger.warning("Sentry SDK not available, skipping Sentry initialization")
            return False
        
        if dsn is None:
            dsn = os.getenv("SENTRY_DSN")
        
        if not dsn:
            logger.info("No Sentry DSN provided, skipping Sentry initialization")
            return False
        
        # Validate configuration
        if config is None:
            config = {}
        
        sentry_config = config.get("sentry", {})
        
        # Configure logging integration
        sentry_logging = sentry_sdk.integrations.logging.LoggingIntegration(
            level=getattr(logging, sentry_config.get("level", "INFO")),
            event_level=getattr(logging, sentry_config.get("event_level", "ERROR")),
        )
        
        # Initialize Sentry
        sentry_sdk.init(
            dsn=dsn,
            environment=sentry_config.get("environment", os.getenv("SENTRY_ENV", "production")),
            traces_sample_rate=sentry_config.get("traces_sample_rate", float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0"))),
            profiles_sample_rate=sentry_config.get("profiles_sample_rate", float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.0"))),
            integrations=[
                sentry_logging,
                sentry_sdk.integrations.aiohttp.AioHttpIntegration(),
                sentry_sdk.integrations.fastapi.FastApiIntegration()
            ],
            send_default_pii=sentry_config.get("send_default_pii", False),
        )
        
        logger.info("✅ Sentry initialized successfully")
        return True
        
    except Exception as exc:  # pragma: no cover
        logger.error(f"Failed to initialize Sentry: {exc}")
        if WARNING_SYMBOLS_AVAILABLE:
            print(failed(f"Failed to initialize Sentry: {exc}"))
        else:
            print(f"❌ Failed to initialize Sentry: {exc}")
        return False


def init_otlp_logging(endpoint: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize OpenTelemetry logging exporter.
    
    Args:
        endpoint: OTLP endpoint (if not provided, will try to get from OTEL_EXPORTER_OTLP_ENDPOINT env var)
        config: Configuration dictionary
        
    Returns:
        bool: True if initialization was successful, False otherwise
        
    Raises:
        DependencyError: If OpenTelemetry is not available
    """
    try:
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry not available, skipping OTLP logging initialization")
            return False
        
        if endpoint is None:
            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        
        if not endpoint:
            logger.info("No OTLP endpoint provided, skipping OpenTelemetry logging initialization")
            return False
        
        # Validate configuration
        if config is None:
            config = {}
        
        otel_config = config.get("opentelemetry", {})
        
        # Create resource
        resource = Resource.create(
            {"service.name": otel_config.get("service_name", os.getenv("OTEL_SERVICE_NAME", "ares-bot"))},
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
        if WARNING_SYMBOLS_AVAILABLE:
            print(failed(f"Failed to initialize OTLP logging: {exc}"))
        else:
            print(f"❌ Failed to initialize OTLP logging: {exc}")
        return False


def init_observability(config: Optional[Dict[str, Any]] = None) -> dict:
    """Initialize all observability components.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        dict: Status of each component initialization
    """
    logger.info("🚀 Initializing observability components...")
    
    # Validate configuration
    try:
        validated_config = validate_observability_config(config)
    except ConfigurationError as e:
        logger.error(f"Configuration validation failed: {e}")
        return {
            "sentry": False,
            "otlp_logging": False,
            "overall_success": False,
            "error": f"Configuration error: {e}"
        }
    
    results = {
        "sentry": False,
        "otlp_logging": False,
        "overall_success": False
    }
    
    try:
        # Initialize Sentry
        if validated_config["sentry"]["enabled"]:
            results["sentry"] = init_sentry(config=validated_config)
        
        # Initialize OpenTelemetry logging
        if validated_config["opentelemetry"]["enabled"]:
            results["otlp_logging"] = init_otlp_logging(config=validated_config)
        
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
            "available": SENTRY_AVAILABLE,
            "environment": os.getenv("SENTRY_ENV", "not_set"),
            "traces_sample_rate": os.getenv("SENTRY_TRACES_SAMPLE_RATE", "not_set"),
            "profiles_sample_rate": os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "not_set")
        },
        "opentelemetry": {
            "enabled": bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")),
            "available": OPENTELEMETRY_AVAILABLE,
            "trace_available": OPENTELEMETRY_TRACE_AVAILABLE,
            "service_name": os.getenv("OTEL_SERVICE_NAME", "not_set"),
            "endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "not_set")
        },
        "logging": {
            "level": logging.getLogger().getEffectiveLevel(),
            "handlers_count": len(logging.getLogger().handlers)
        },
        "dependencies": check_dependencies()
    }
    
    return status


def configure_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = False,
    log_file_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        enable_console: Whether to enable console logging
        enable_file: Whether to enable file logging
        log_file_path: Path to log file (required if enable_file is True)
        config: Configuration dictionary
        
    Raises:
        ConfigurationError: If logging configuration is invalid
    """
    try:
        # Use configuration if provided
        if config and "logging" in config:
            logging_config = config["logging"]
            level = logging_config.get("level", level)
            format_string = logging_config.get("format", format_string)
            enable_console = logging_config.get("enable_console", enable_console)
            enable_file = logging_config.get("enable_file", enable_file)
            log_file_path = logging_config.get("log_file_path", log_file_path)
        
        # Validate level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level.upper() not in valid_levels:
            raise ConfigurationError(f"Invalid log level: {level}. Valid: {valid_levels}")
        
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
        raise ConfigurationError(f"Failed to configure logging: {e}")


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
        if OPENTELEMETRY_TRACE_AVAILABLE:
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
        else:
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
        if OPENTELEMETRY_TRACE_AVAILABLE:
            try:
                meter = metrics.get_meter(__name__)
                
                # Create or get counter
                counter = meter.create_counter(
                    name=metric_name,
                    description=f"Performance metric: {metric_name}",
                    unit=unit
                )
                
                # Record the value
                counter.add(value, tags)
                
            except Exception as e:
                logger.warning(f"Failed to send metric to OpenTelemetry: {e}")
        else:
            logger.debug("OpenTelemetry metrics not available, logging only")
        
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
        if SENTRY_AVAILABLE:
            try:
                if sentry_sdk.Hub.current.client:
                    sentry_sdk.capture_exception(error)
            except Exception as e:
                logger.warning(f"Failed to capture exception in Sentry: {e}")
        
    except Exception as e:
        # Fallback logging if the structured logging fails
        logger.exception(f"Failed to log error with context: {e}")


def create_observability_config(
    sentry_dsn: Optional[str] = None,
    otel_endpoint: Optional[str] = None,
    log_level: str = "INFO",
    enable_file_logging: bool = False,
    log_file_path: Optional[str] = None
) -> Dict[str, Any]:
    """Create a standard observability configuration.
    
    Args:
        sentry_dsn: Sentry DSN
        otel_endpoint: OpenTelemetry endpoint
        log_level: Logging level
        enable_file_logging: Whether to enable file logging
        log_file_path: Path to log file
        
    Returns:
        Configuration dictionary
    """
    config = {
        "sentry": {
            "enabled": bool(sentry_dsn),
            "dsn": sentry_dsn,
            "environment": os.getenv("SENTRY_ENV", "production"),
            "traces_sample_rate": 0.1,
            "profiles_sample_rate": 0.0,
            "send_default_pii": False
        },
        "opentelemetry": {
            "enabled": bool(otel_endpoint),
            "endpoint": otel_endpoint,
            "service_name": os.getenv("OTEL_SERVICE_NAME", "ares-bot"),
            "traces_enabled": True,
            "metrics_enabled": True,
            "logs_enabled": True
        },
        "logging": {
            "level": log_level,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "enable_console": True,
            "enable_file": enable_file_logging,
            "log_file_path": log_file_path,
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "backup_count": 5
        }
    }
    
    return config


# Example usage and testing
if __name__ == "__main__":
    try:
        # Check dependencies
        deps = check_dependencies()
        print(f"📦 Dependencies: {deps}")
        
        # Create configuration
        config = create_observability_config(
            sentry_dsn=os.getenv("SENTRY_DSN"),
            otel_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            log_level="DEBUG",
            enable_file_logging=False
        )
        
        # Configure logging
        configure_logging(config=config)
        
        # Initialize observability
        init_results = init_observability(config)
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
            
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
