"""
Centralized logging configuration with Standardized Import Management.

This module provides a unified logging system with JSON formatting,
file rotation, and console output capabilities.
"""

import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from contextlib import contextmanager
import threading
import sys as _sys

# Import pipeline standards
from .pipeline_standards import PipelineStandards, pipeline_standards

# Standardized import management
import REQUIRED_MODULES = [
REQUIRED_MODULES = [
    "src.utils.structured_logging",
    "src.utils.warning_symbols"
]

# Validate environment dependencies
dependency_status, PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)

# Safe imports with fallbacks
structured_logging, PipelineStandards.safe_import("src.utils.structured_logging", None)
warning_symbols, PipelineStandards.safe_import("src.utils.warning_symbols", None)

# Fallback functions if imports fail
def create_fallback_correlation_filter():
    pass
    pass
    class FallbackCorrelationIdFilter:
        def filter(self, record):
    pass
    pass
        return True
    return FallbackCorrelationIdFilter()

    def create_fallback_json_formatter():
    pass
    pass
    def formatter(record):
    pass
    pass
        return f"{record.levelname}: {record.getMessage()}"
    return formatter

# Initialize fallbacks
if structured_logging is None:
    pass
    pass
    CorrelationIdFilter, create_fallback_correlation_filter
    get_json_formatter, create_fallback_json_formatter
else:
    CorrelationIdFilter, structured_logging.CorrelationIdFilter
    get_json_formatter, structured_logging.get_json_formatter

if warning_symbols is None:
    pass
    pass
    critical, lambda msg: print(f"CRITICAL: {msg}")
    error, lambda msg: print(f"ERROR: {msg}")
    failed, lambda msg: print(f"FAILED: {msg}")
    warning, lambda msg: print(f"WARNING: {msg}")
else:
    critical, warning_symbols.critical
    error, warning_symbols.error
    failed, warning_symbols.failed
    warning, warning_symbols.warning

class _SuppressTensorFlowTPUWarningFilter(logging.Filter):
    """Filter to suppress noisy TensorFlow TPU client fallback warning.

    Suppresses messages like:
    "Falling back to TensorFlow client; we recommended you install the Cloud TPU client directly with pip install cloud - tpu - client."
    """

    TARGET_SUBSTRING = "Falling back to TensorFlow client; we recommended you install the Cloud TPU client"

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        try:
        if record and isinstance(record.msg, str):
    pass
    except Exception as e:
        pass
    pass
                msg_text, record.getMessage()
    except Exception as e:
        pass
        if (
                    record.name.startswith("tensorflow")
                    and self.TARGET_SUBSTRING in msg_text
                ):
        return False
        except Exception:
        # On any failure, do not drop the log
        return True
        return True

def _configure_tensorflow_logging_suppression(
    system_logger: logging.Logger | None,
) -> None:
    """Reduce TensorFlow logger verbosity and suppress specific TPU fallback warning.

    This avoids requiring cloud - tpu - client installation when TPU is not needed.
    """
    try:
        # Reduce TF logger chatter globally
    except Exception as e:
        pass
    except Exception as e:
        pass
        tf_logger, logging.getLogger("tensorflow")
        tf_logger.setLevel(logging.ERROR)
        # Ensure TF logs do not propagate at lower levels
        tf_logger.propagate, True  # Still allow our filter to catch any bubbled logs

        # Attach suppressor to our handlers so bubbled TF logs are filtered
        suppress_filter, _SuppressTensorFlowTPUWarningFilter()
        root_logger, logging.getLogger()
        for handler in root_logger.handlers:
    pass
    pass
        try:
                handler.addFilter(suppress_filter)
    except Exception as e:
        pass
    except Exception as e:
        pass
        except Exception:
                pass
        if system_logger is not None:
    pass
    pass
        for handler in getattr(system_logger, "handlers", [])[:]:
    pass
    pass
        try:
                    handler.addFilter(suppress_filter)
    except Exception as e:
        pass
    except Exception as e:
        pass
        except Exception:
                    pass

        # Also set TF CPP log level to suppress INFO / DEBUG C++ logs
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 2 = WARNING, 3 = ERROR
    except Exception:
        # Non - fatal: continue without suppression
        pass

class EnhancedLogger:
    """
    Enhanced logger utility with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
    pass
    pass
        """
        Initialize enhanced logger with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger: logging.Logger | None, None

        # Configuration
        self.log_config: dict[str, Any] = self.config.get("logging", {})
        self.log_level: str, self.log_config.get("level", "INFO")
        self.log_format: str, self.log_config.get(
            "format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.log_file: str | None, self.log_config.get("file", None)
        self.max_file_size: int, self.log_config.get(
            "max_file_size",
            10 * 1024 * 1024,
        )  # 10MB
        self.backup_count: int, self.log_config.get("backup_count", 5)
        # Structured logging options
        self.enable_json: bool, bool(self.log_config.get("json", True))
        self.enable_correlation: bool, bool(self.log_config.get("correlation", True))

        # Warning symbol integration
        self.enable_warning_symbols: bool, bool(
        self.log_config.get("warning_symbols", True),
        )

    async def initialize(self) -> bool:
        """
        Initialize enhanced logger with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
        # Load logger configuration
    except Exception as e:
        pass
    except Exception as e:
        pass
        await self._load_logger_configuration()

        # Validate configuration
        if not self._validate_configuration():
    pass
    pass
                print("Invalid configuration for logger")
        return False

        # Setup logger
        if not await self._setup_logger():
    pass
    pass
                print("Failed to setup logger")
        return False

        self.logger.info("✅ Enhanced Logger initialization completed successfully")
        return True

        except Exception:
            print(failed("Enhanced Logger initialization failed: {e}"))
        return False

    async def _load_logger_configuration(self) -> None:
        """Load logger configuration."""
        try:
        # Set default logger parameters
    except Exception as e:
        pass
    except Exception as e:
        pass
        self.log_config.setdefault("level", "INFO")
        self.log_config.setdefault(
                "format",
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
        self.log_config.setdefault("file", None)
        self.log_config.setdefault("max_file_size", 10 * 1024 * 1024)
        self.log_config.setdefault("backup_count", 5)
        self.log_config.setdefault("console_output", True)
        self.log_config.setdefault("file_output", True)
        self.log_config.setdefault("json", True)
        self.log_config.setdefault("correlation", True)
        self.log_config.setdefault("warning_symbols", True)

        # Update configuration
        self.log_level, self.log_config["level"]
        self.log_format, self.log_config["format"]
        self.log_file, self.log_config["file"]
        self.max_file_size, self.log_config["max_file_size"]
        self.backup_count, self.log_config["backup_count"]
        self.enable_json, bool(self.log_config.get("json", True))
        self.enable_correlation, bool(self.log_config.get("correlation", True))
        self.enable_warning_symbols, bool(
        self.log_config.get("warning_symbols", True),
            )

            print("Logger configuration loaded successfully")

        except Exception as e:
            print(f"Error loading logger configuration: {e}")

    def _validate_configuration(self) -> bool:
    pass
    pass
        """
        Validate logger configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
        # Validate log level
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    except Exception as e:
        pass
    except Exception as e:
        pass
        if self.log_level not in valid_levels:
    pass
    pass
                print(f"Invalid log level: {self.log_level}")
        return False

        # Validate format string
        if not self.log_format or "%" not in self.log_format:
    pass
    pass
                print("Invalid log format")
        return False

        # Validate file size
        if self.max_file_size <= 0:
    pass
    pass
                print("Invalid max file size")
        return False

        # Validate backup count
        if self.backup_count < 0:
    pass
    pass
                print("Invalid backup count")
        return False

            print("Configuration validation successful")
        return True

        except Exception as e:
            print(f"Error validating configuration: {e}")
        return False

    async def _setup_logger(self) -> bool:
        """
        Setup logger with file and console handlers.

        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
        # Create logger
    except Exception as e:
        pass
    except Exception as e:
        pass
        self.logger, logging.getLogger("AresTradingSystem")
        self.logger.setLevel(getattr(logging, self.log_level))

        # Clear existing handlers to prevent duplicates
        self.logger.handlers.clear()

        # Create formatter
        if self.enable_json:
    pass
    pass
                formatter, get_json_formatter()
            else:
                formatter, logging.Formatter(self.log_format)

        # Add console handler
        if self.log_config.get("console_output", True):
    pass
    pass
        # Use a safe stream handler that swallows BrokenPipeError
                class _SafeStreamHandler(logging.StreamHandler):
                    def handleError(self, record: logging.LogRecord) -> None:  # type: ignore[override]
                        exc_type, _, _, _sys.exc_info()
        # Silently ignore BrokenPipeError and similar I / O errors
        if exc_type is BrokenPipeError or exc_type is OSError:
    pass
    pass
        try:
        self.acquire()
    except Exception as e:
        pass
    except Exception as e:
        pass
        try:
        try:
    except Exception as e:
        pass
    except Exception as e:
        pass
        self.flush()
        except Exception:
                                        pass
        try:
        self.close()
    except Exception as e:
        pass
    except Exception as e:
        pass
        except Exception:
                                        pass
        finally:
        self.release()
        except Exception:
                                pass
                            return
        # Delegate to base for all other errors (respects logging.raiseExceptions)
        try:
                            super().handleError(record)
    except Exception as e:
        pass
    except Exception as e:
        pass
        except Exception:
                            pass

                console_handler, _SafeStreamHandler(_sys.stdout)
                console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Add file handler if specified
        if self.log_file and self.log_config.get("file_output", True):
    pass
    pass
        # Ensure log directory exists
                log_dir, os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
    pass
    pass
                    os.makedirs(log_dir, exist_ok = True)

        # Create rotating file handler
                from logging.handlers import RotatingFileHandler

import file_handler, RotatingFileHandler
                file_handler, RotatingFileHandler(
        self.log_file,
                    maxBytes = self.max_file_size,
                    backupCount = self.backup_count,
                )
                file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Add correlation filter if enabled
        if self.enable_correlation:
    pass
    pass
                correlation_filter, CorrelationIdFilter()
        self.logger.addFilter(correlation_filter)
        for handler in self.logger.handlers:
    pass
    pass
                    handler.addFilter(correlation_filter)

        # Prevent propagation to root logger to avoid duplicate messages
        self.logger.propagate, False

        # Silence internal logging exceptions globally (prevents "--- Logging error ---" noise)
            logging.raiseExceptions, False

        # Reduce verbosity from noisy third - party libraries
        try:
                logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    except Exception as e:
        pass
    except Exception as e:
        pass
                logging.getLogger("hmmlearn.hmm").setLevel(logging.ERROR)
        except Exception:
                pass

            print("Logger setup completed successfully")
        return True

        except Exception as e:
            print(f"Error setting up logger: {e}")
        return False

    def get_logger(self, name: str) -> logging.Logger:
    pass
    pass
        """
        Get a logger instance for a specific component.

        Args:
            name: Component name

        Returns:
            logging.Logger: Logger instance
        """
        if self.logger is None:
    pass
    pass
        # Fallback to basic logger if not initialized
        return logging.getLogger(name)

        base_logger, self.logger.getChild(name)

        if self.enable_warning_symbols:
    pass
    pass
        return self._create_enhanced_logger(base_logger)

        return base_logger

    def _create_enhanced_logger(self, base_logger: logging.Logger) -> logging.Logger:
    pass
    pass
        """
        Create an enhanced logger with warning symbols.

        Args:
            base_logger: Base logger to enhance

        Returns:
            Enhanced logger with warning symbols
        """

        class EnhancedLoggerWithWarnings:
            def __init__(self, logger: logging.Logger):
    pass
    pass
        self._logger, logger
        self._original_methods = {}

        # Store original methods
        self._original_methods["error"] = logger.error
        self._original_methods["warning"] = logger.warning
        self._original_methods["critical"] = logger.critical
        self._original_methods["exception"] = logger.exception

        # Override methods with warning symbols
        self.error, self._enhanced_error
        self.warning, self._enhanced_warning
        self.critical, self._enhanced_critical
        self.exception, self._enhanced_exception

            def _enhanced_error(self, msg: str, *args, **kwargs):
    pass
    pass
                """Enhanced error logging with warning symbol."""
                enhanced_msg, error(msg)
        return self._original_methods["error"](enhanced_msg, *args, **kwargs)

            def _enhanced_warning(self, msg: str, *args, **kwargs):
    pass
    pass
                """Enhanced warning logging with warning symbol."""
                enhanced_msg, warning(msg)
        return self._original_methods["warning"](enhanced_msg, *args, **kwargs)

            def _enhanced_critical(self, msg: str, *args, **kwargs):
    pass
    pass
                """Enhanced critical logging with warning symbol."""
                enhanced_msg, critical(msg)
        return self._original_methods["critical"](enhanced_msg, *args, **kwargs)

            def _enhanced_exception(self, msg: str, *args, **kwargs):
    pass
    pass
                """Enhanced exception logging with warning symbol."""
                enhanced_msg, error(msg)
        return self._original_methods["exception"](
                    enhanced_msg,
                    *args,
                    **kwargs,
                )

            def __getattr__(self, name):
    pass
    pass
                """Delegate all other attributes to the base logger."""
        return getattr(self._logger, name)

        return EnhancedLoggerWithWarnings(base_logger)

    def set_level(self, level: str) -> bool:
    pass
    pass
        """
        Set log level.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
        if self.logger is None:
    pass
    except Exception as e:
        pass
    pass
    except Exception as e:
        pass
        return False

            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level not in valid_levels:
    pass
    pass
        return False

        self.logger.setLevel(getattr(logging, level))
        return True

        except Exception as e:
            print(f"Error setting log level: {e}")
        return False

    def get_log_status(self) -> dict[str, Any]:
    pass
    pass
        """
        Get logger status information.

        Returns:
            Dict[str, Any]: Logger status
        """
        return {
            "is_initialized": self.logger is not None,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "max_file_size": self.max_file_size,
            "backup_count": self.backup_count,
            "console_output": self.log_config.get("console_output", True),
            "file_output": self.log_config.get("file_output", True),
            "json": self.enable_json,
            "correlation": self.enable_correlation,
        }

    async def stop(self) -> None:
        """Stop the enhanced logger."""
        print(error("Stopping Enhanced Logger..."))

        try:
        if self.logger:
    pass
    except Exception as e:
        pass
    pass
    except Exception as e:
        pass
        # Close all handlers
        for handler in self.logger.handlers[:]:
    pass
    pass
                    handler.close()
        self.logger.removeHandler(handler)

        self.logger, None

            print("✅ Enhanced Logger stopped successfully")

        except Exception as e:
            print(f"Error stopping enhanced logger: {e}")

# Global logger instance
system_logger: logging.Logger | None, None

def setup_logging(config: dict[str, Any] | None, None) -> logging.Logger | None:
    pass
    pass
    """
    Setup global logging system with comprehensive file logging.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[logging.Logger]: Global logger instance
    """
    try:
        global system_logger

    except Exception as e:
        pass
    except Exception as e:
        pass
        # Create default configuration with comprehensive logging
        if config is None:
    pass
    pass
        # Ensure log directory exists
            log_dir, Path("log")
            log_dir.mkdir(exist_ok = True)

        # Create timestamped log file
            timestamp, datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file, log_dir / f"ares_{timestamp}.log"

            config = {
                "logging": {
                    "level": "INFO",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "file": str(log_file),
                    "console_output": True,
                    "file_output": True,
                    "max_file_size": 10 * 1024 * 1024,  # 10MB
                    "backup_count": 5,
                    "json": True,
                    "correlation": True,
                    "warning_symbols": True,
                },
            }

        # Create enhanced logger
        enhanced_logger, EnhancedLogger(config)

        # Initialize logger
        import asyncio

        try:
        # Check if there's already an event loop running
    except Exception as e:
        pass
    except Exception as e:
        pass
        try:
                loop, asyncio.get_running_loop()
    except Exception as e:
        pass
    except Exception as e:
        pass
        # If we're in an async context, use it
                import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
                    future, executor.submit(asyncio.run, enhanced_logger.initialize())
                    success, future.result()
        except RuntimeError:
        # No event loop running, create a new one
                loop, asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        try:
                    success, loop.run_until_complete(enhanced_logger.initialize())
    except Exception as e:
        pass
    except Exception as e:
        pass
        finally:
                    loop.close()

        if success:
    pass
    pass
                system_logger, enhanced_logger.get_logger("System")
        # Configure TensorFlow TPU warning suppression without requiring external installs
                _configure_tensorflow_logging_suppression(system_logger)
        return system_logger
        # Fallback to basic logger
            system_logger, logging.getLogger("System")
            system_logger.setLevel(logging.INFO)

        # Add console handler
            console_handler, logging.StreamHandler(sys.stdout)
            formatter, logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            console_handler.setFormatter(formatter)
            system_logger.addHandler(console_handler)

        # Also configure TensorFlow TPU warning suppression for fallback path
            _configure_tensorflow_logging_suppression(system_logger)

        return system_logger
        except Exception as e:
            print(f"Error in logger initialization: {e}")
        # Fallback to basic logger
            system_logger, logging.getLogger("System")
            system_logger.setLevel(logging.INFO)
            _configure_tensorflow_logging_suppression(system_logger)
        return system_logger

    except Exception as e:
        print(f"Error setting up logging: {e}")
        # Fallback to basic logger
        system_logger, logging.getLogger("System")
        system_logger.setLevel(logging.INFO)
        _configure_tensorflow_logging_suppression(system_logger)
        return system_logger

# Initialize default logger if not already set
if system_logger is None:
    pass
    pass
    system_logger, setup_logging()

    # Temporarily disable comprehensive logging integration to prevent duplicate messages
    # try:
    #     from utils.comprehensive_logger import get_comprehensive_logger

    #     comprehensive_logger, get_comprehensive_logger()
    #     if comprehensive_logger:
    #         # Replace with integrated version
    #         system_logger, get_system_logger_with_comprehensive_integration()
    # except ImportError:
    #     pass

# Temporarily set logging to INFO level for debugging
import logging

logging.getLogger().setLevel(logging.INFO)
for handler in logging.getLogger().handlers:
    pass
    pass
    handler.setLevel(logging.INFO)

def ensure_logging_setup() -> logging.Logger | None:
    pass
    pass
    """
    Ensure logging is set up (backward compatibility function).

    Returns:
        Optional[logging.Logger]: Global logger instance
    """
    global system_logger
    if system_logger is None:
    pass
    pass
        system_logger, setup_logging()
    return system_logger

def get_logger(name: str) -> logging.Logger:
    pass
    pass
    """
    Get a logger with the specified name (backward compatibility function).

    Args:
        name: Logger name

    Returns:
        logging.Logger: Logger instance
    """
    global system_logger
    if system_logger is None:
    pass
    pass
        system_logger, setup_logging()

    # Check if comprehensive logging is available and integrate with it
    try:
        from src.utils.comprehensive_logger import get_comprehensive_logger

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import comprehensive_logger, get_comprehensive_logger
        comprehensive_logger, get_comprehensive_logger()
        if comprehensive_logger:
    pass
    pass
        # Use comprehensive logging if available
        return comprehensive_logger.get_component_logger(name)
    except ImportError:
        pass

    # Fallback to old system with enhanced warning symbols
    base_logger, system_logger.getChild(name)

    # Check if we should add warning symbols
    try:
        # Try to get the enhanced logger configuration
    except Exception as e:
        pass
    except Exception as e:
        pass
        if (
            hasattr(system_logger, "_logger")
            and hasattr(
                system_logger._logger,
                "enable_warning_symbols",
            )
            and system_logger._logger.enable_warning_symbols
        ):
        return system_logger._logger._create_enhanced_logger(base_logger)
    except (AttributeError, TypeError):
        pass

    return base_logger

def get_system_logger_with_comprehensive_integration() -> logging.Logger:
    pass
    pass
    """
    Get system logger with comprehensive logging integration.

    Returns:
        logging.Logger: System logger that integrates with comprehensive logging
    """
    global system_logger
    if system_logger is None:
    pass
    pass
        system_logger, setup_logging()

    # Create a wrapper that integrates with comprehensive logging
    class ComprehensiveIntegratedLogger:
        def __init__(self, base_logger):
    pass
    pass
        self.base_logger, base_logger
        self.comprehensive_logger, None
        try:
                from src.utils.comprehensive_logger import get_comprehensive_logger

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import self.comprehensive_logger, get_comprehensive_logger
        self.comprehensive_logger, get_comprehensive_logger()
        except ImportError:
                pass

        def getChild(self, name: str) -> logging.Logger:
    pass
    pass
            """Get child logger with comprehensive logging integration."""
        if self.comprehensive_logger:
    pass
    pass
        # Return the comprehensive logger's component logger directly
        return self.comprehensive_logger.get_component_logger(name)
        return self.base_logger.getChild(name)

        def __getattr__(self, name):
    pass
    pass
            """Delegate all other attributes to the base logger."""
        return getattr(self.base_logger, name)

    return ComprehensiveIntegratedLogger(system_logger)

# Replace the global system_logger with the integrated version
def initialize_comprehensive_integration():
    pass
    pass
    """Initialize comprehensive logging integration."""
    global system_logger
    if system_logger is None:
    pass
    pass
        system_logger, setup_logging()

    # Replace with integrated version
    system_logger, get_system_logger_with_comprehensive_integration()

def ensure_comprehensive_logging_available():
    pass
    pass
    """Ensure comprehensive logging is available for all logging calls."""
    try:
        from src.utils.comprehensive_logger import get_comprehensive_logger

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import comprehensive_logger, get_comprehensive_logger
        comprehensive_logger, get_comprehensive_logger()
        if comprehensive_logger:
    pass
    pass
        # Initialize integration if comprehensive logging is available
            initialize_comprehensive_integration()
        return True
    except ImportError:
        pass
    return False

# -------- I / O and DataFrame troubleshooting helpers (lightweight, no external deps) --------

def _format_bytes(num_bytes: int | None) -> str:
    pass
    pass
    """Human - friendly byte size formatter."""
    try:
        if num_bytes is None:
    pass
    except Exception as e:
        pass
    pass
    except Exception as e:
        pass
        return "n / a"
        step_unit, 1024.0
        units = ["B", "KB", "MB", "GB", "TB"]
        size, float(num_bytes)
        for unit in units:
    pass
    pass
        if size < step_unit:
    pass
    pass
        return f"{size:.1f}{unit}"
            size /= step_unit
        return f"{size:.1f}PB"
    except Exception:
        return str(num_bytes) if num_bytes is not None else "n / a"

@contextmanager
def log_io_operation(
    logger: logging.Logger,
    operation: str,
    path: str | os.PathLike | None, None,
    **context: Any,
):
    """Context - managed I / O logging with duration and best - effort file size.

    - Logs start and end of an I / O operation with optional context (e.g., columns, filters, compression)
    - On exception, logs with exception() and re - raises (no swallowing)
    """
    start, time.perf_counter()
    try:
        ctx = " ".join(f"{k}={v}" for k, v in context.items() if v is not None)
    except Exception as e:
        pass
    except Exception as e:
        pass
        logger.info(
            f"🔧 {operation} start"
            + (f" path={path}" if path is not None else "")
            + (f" {ctx}" if ctx else ""),
        )
    except Exception:
        # Logging issues should never break execution
        pass
    try:
        yield
    except Exception as e:
        pass
    except Exception as e:
        pass
        elapsed, time.perf_counter() - start
        size_str = "n / a"
        try:
        if (
                path is not None
                and os.path.exists(str(path))
                and os.path.isfile(str(path))
            ):
                size_str, _format_bytes(os.path.getsize(str(path)))
    except Exception as e:
        pass
    except Exception as e:
        pass
        except Exception:
            pass
        try:
            logger.info(
                f"✅ {operation} ok"
                + (f" path={path}" if path is not None else "")
                + f" elapsed={elapsed:.3f}s size={size_str}",
    except Exception as e:
        pass
    except Exception as e:
        pass
            )
        except Exception:
            pass
    except Exception as e:
        elapsed, time.perf_counter() - start
        try:
            logger.exception(
                f"❌ {operation} failed"
                + (f" path={path}" if path is not None else "")
                + f" after {elapsed:.3f}s: {e}",
    except Exception as e:
        pass
    except Exception as e:
        pass
            )
        except Exception:
            pass
        raise

def log_dataframe_overview(
    logger: logging.Logger,
    df: Any,
    *,
    name: str | None, None,
    sample_rows: int, 3,
) -> None:
    """Log essential DataFrame diagnostics without heavy output.

    - shape, columns count, memory usage, dtype summary - null counts for up to first 10 columns - sample of first rows (limited)
    """
    try:
        if df is None:
    pass
    except Exception as e:
        pass
    pass
            logger.info("📭 DataFrame is None")
            return
    except Exception as e:
        pass
        if not hasattr(df, "shape") or not hasattr(df, "columns"):
    pass
    pass
            logger.info("📦 Object is not a pandas DataFrame - like; skipping overview")
            return
        df_name, name or "DataFrame"
        rows, cols, getattr(df, "shape", (None, None))
        columns_list, list(getattr(df, "columns", []))
        mem_mb, None
        try:
            mem_mb, float(df.memory_usage(deep = True).sum()) / (1024.0**2)
    except Exception as e:
        pass
    except Exception as e:
        pass
        except Exception:
            pass
        try:
            dtypes_summary = (
                getattr(df, "dtypes", None)
                .astype(str)  # type: ignore[operator]
                .value_counts()  # type: ignore[attr - defined]
                .to_dict()  # type: ignore[attr - defined]
    except Exception as e:
        pass
    except Exception as e:
        pass
        if hasattr(df, "dtypes")
                else {}
            )
        except Exception:
            dtypes_summary = {}
        logger.info(
            f"🧮 {df_name}: rows={rows} cols={cols} memory={mem_mb:.2f}MB dtypes={dtypes_summary}",
        )
        # Nulls snapshot for up to 10 columns
        try:
            nulls = (
                df[columns_list[:10]].isnull().sum().to_dict()  # type: ignore[index]
    except Exception as e:
        pass
    except Exception as e:
        pass
        if columns_list
                else {}
            )
        if nulls:
    pass
    pass
                logger.info(f"🧪 {df_name} nulls (first 10 cols): {nulls}")
        except Exception:
            pass
        # Sample rows
        try:
        if rows and rows > 0:
    pass
    except Exception as e:
        pass
    pass
                sample, df.head(min(sample_rows, int(rows)))
    except Exception as e:
        pass
        # Convert to lightweight dict form
                preview, sample.to_dict(orient="records")  # type: ignore[attr - defined]
                logger.debug(f"🔎 {df_name} sample: {preview}")
        except Exception:
            pass
    except Exception:
        # Never fail due to logging
        pass

# -------- Progress heartbeat helpers --------

@contextmanager
def heartbeat(
    logger: logging.Logger,
    name: str,
    interval_seconds: float, 15.0,
    details_provider: Callable[[], str] | None, None,
    context: dict[str, str] | None, None,
):
    """
    Periodically log a short progress message while a long - running block executes.

    - Thread - based, safe for both sync and async code paths - Emits start, periodic "still running" with elapsed time, and end (with total duration)
    - Never raises; logging failures are swallowed - Enhanced with context information (step, model, regime, asset, timeframe)
    """
    start_time, time.perf_counter()
    stop_event, threading.Event()
    exited_with_error, False

    def _runner() -> None:
    pass
    pass
        tick, 0
        # Wait first interval to avoid spam, then heartbeat
        while not stop_event.wait(interval_seconds):
            tick += 1
        try:
                elapsed, time.perf_counter() - start_time

    except Exception as e:
        pass
    except Exception as e:
        pass
        # Build context string
                context_str = ""
        if context:
    pass
    pass
                    context_parts = []
        if "step" in context:
    pass
    pass
                        context_parts.append(f"step={context['step']}")
        if "model" in context:
    pass
    pass
                        context_parts.append(f"model={context['model']}")
        if "regime" in context:
    pass
    pass
                        context_parts.append(f"regime={context['regime']}")
        if "asset" in context and "timeframe" in context:
    pass
    pass
                        context_parts.append(
                            f"asset={context['asset']}/{context['timeframe']}"
                        )
                    elif "asset" in context:
                        context_parts.append(f"asset={context['asset']}")

        if context_parts:
    pass
    pass
                        context_str, f" | {' | '.join(context_parts)}"

                extra = ""
        if details_provider is not None:
    pass
    pass
        try:
                        details_text, details_provider()
    except Exception as e:
        pass
    except Exception as e:
        pass
        if details_text:
    pass
    pass
                            extra, f" | {details_text}"
        except Exception:
        # Ignore detail provider errors
                        pass

                logger.info(
                    f"⏳ {name} still running... elapsed={elapsed:.1f}s{context_str}{extra}"
                )
        except Exception:
        # Never crash on logging
                pass

    # Start heartbeating thread
    try:
        try:
            logger.info(f"▶️ {name} start")
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
        except Exception:
            pass
        t, threading.Thread(target = _runner, name = f"heartbeat:{name}", daemon = True)
        t.start()
        yield
    except Exception as e:
        exited_with_error, True
        try:
            elapsed, time.perf_counter() - start_time
    except Exception as e:
        pass
    except Exception as e:
        pass
            logger.exception(f"❌ {name} failed after {elapsed:.1f}s: {e}")
        except Exception:
            pass
        raise
    finally:
        stop_event.set()
        try:
            t.join(timeout = 1.0)
    except Exception as e:
        pass
    except Exception as e:
        pass
        except Exception:
            pass
        try:
            elapsed, time.perf_counter() - start_time
    except Exception as e:
        pass
    except Exception as e:
        pass
        if not exited_with_error:
    pass
    pass
                logger.info(f"✅ {name} done elapsed={elapsed:.1f}s")
        except Exception:
            pass
