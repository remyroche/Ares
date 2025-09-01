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
REQUIRED_MODULES = [
    "src.utils.structured_logging",
    "src.utils.warning_symbols"
]

# Validate environment dependencies
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)

# Safe imports with fallbacks
structured_logging = PipelineStandards.safe_import("src.utils.structured_logging", None)
warning_symbols = PipelineStandards.safe_import("src.utils.warning_symbols", None)

# Fallback functions if imports fail
# Initialize fallbacks
if structured_logging is None:
    CorrelationIdFilter = create_fallback_correlation_filter
    get_json_formatter = create_fallback_json_formatter
else:
    CorrelationIdFilter = structured_logging.CorrelationIdFilter
    get_json_formatter = structured_logging.get_json_formatter

if warning_symbols is None:
    critical = lambda msg: print(f"CRITICAL: {msg}")
    error = lambda msg: print(f"ERROR: {msg}")
    failed = lambda msg: print(f"FAILED: {msg}")
    warning = lambda msg: print(f"WARNING: {msg}")
else:
    critical = warning_symbols.critical
    error = warning_symbols.error
    failed = warning_symbols.failed
    warning = warning_symbols.warning


class _SuppressTensorFlowTPUWarningFilter(logging.Filter):
    """Filter to suppress noisy TensorFlow TPU client fallback warning.

    Suppresses messages like:
    "Falling back to TensorFlow client; we recommended you install the Cloud TPU client directly with pip install cloud-tpu-client."
    """

    TARGET_SUBSTRING = "Falling back to TensorFlow client; we recommended you install the Cloud TPU client"


def _configure_tensorflow_logging_suppression(
    system_logger: logging.Logger | None,
) -> None:
    """Reduce TensorFlow logger verbosity and suppress specific TPU fallback warning.

    This avoids requiring cloud-tpu-client installation when TPU is not needed.
    """
    try:
        # Reduce TF logger chatter globally
        tf_logger = logging.getLogger("tensorflow")
        tf_logger.setLevel(logging.ERROR)
        # Ensure TF logs do not propagate at lower levels
        tf_logger.propagate = True  # Still allow our filter to catch any bubbled logs

        # Attach suppressor to our handlers so bubbled TF logs are filtered
        suppress_filter = _SuppressTensorFlowTPUWarningFilter()
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            try:
                handler.addFilter(suppress_filter)
            except Exception:
                pass
        if system_logger is not None:
            for handler in getattr(system_logger, "handlers", [])[:]:
                try:
                    handler.addFilter(suppress_filter)
                except Exception:
                    pass

        # Also set TF CPP log level to suppress INFO/DEBUG C++ logs
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 2=WARNING, 3=ERROR
    except Exception:
        # Non-fatal: continue without suppression
        pass


class EnhancedLogger:
    """
    Enhanced logger utility with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize enhanced logger with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger: logging.Logger | None = None

        # Configuration
        self.log_config: dict[str, Any] = self.config.get("logging", {})
        self.log_level: str = self.log_config.get("level", "INFO")
        self.log_format: str = self.log_config.get(
            "format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.log_file: str | None = self.log_config.get("file", None)
        self.max_file_size: int = self.log_config.get(
            "max_file_size",
            10 * 1024 * 1024,
        )  # 10MB
        self.backup_count: int = self.log_config.get("backup_count", 5)
        # Structured logging options
        self.enable_json: bool = bool(self.log_config.get("json", True))
        self.enable_correlation: bool = bool(self.log_config.get("correlation", True))

        # Warning symbol integration
        self.enable_warning_symbols: bool = bool(
            self.log_config.get("warning_symbols", True),
        )

    async def _load_logger_configuration(self) -> None:
        """Load logger configuration."""
        try:
            # Set default logger parameters
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
            self.log_level = self.log_config["level"]
            self.log_format = self.log_config["format"]
            self.log_file = self.log_config["file"]
            self.max_file_size = self.log_config["max_file_size"]
            self.backup_count = self.log_config["backup_count"]
            self.enable_json = bool(self.log_config.get("json", True))
            self.enable_correlation = bool(self.log_config.get("correlation", True))
            self.enable_warning_symbols = bool(
                self.log_config.get("warning_symbols", True),
            )

            print("Logger configuration loaded successfully")

        except Exception as e:
            print(f"Error loading logger configuration: {e}")

    def _validate_configuration(self) -> bool:
        """
        Validate logger configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate log level
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if self.log_level not in valid_levels:
                print(f"Invalid log level: {self.log_level}")
                return False

            # Validate format string
            if not self.log_format or "%" not in self.log_format:
                print("Invalid log format")
                return False

            # Validate file size
            if self.max_file_size <= 0:
                print("Invalid max file size")
                return False

            # Validate backup count
            if self.backup_count < 0:
                print("Invalid backup count")
                return False

            print("Configuration validation successful")
            return True

        except Exception as e:
            print(f"Error validating configuration: {e}")
            return False

    def _create_enhanced_logger(self, base_logger: logging.Logger) -> logging.Logger:
        """
        Create an enhanced logger with warning symbols.

        Args:
            base_logger: Base logger to enhance

        Returns:
            Enhanced logger with warning symbols
        """

        class EnhancedLoggerWithWarnings:
            def __init__(self, logger: logging.Logger):
                self._logger = logger
                self._original_methods = {}

                # Store original methods
                self._original_methods["error"] = logger.error
                self._original_methods["warning"] = logger.warning
                self._original_methods["critical"] = logger.critical
                self._original_methods["exception"] = logger.exception

                # Override methods with warning symbols
                self.error = self._enhanced_error
                self.warning = self._enhanced_warning
                self.critical = self._enhanced_critical
                self.exception = self._enhanced_exception

        return EnhancedLoggerWithWarnings(base_logger)


# Global logger instance
system_logger: logging.Logger | None = None



# Initialize default logger if not already set
if system_logger is None:
    system_logger = setup_logging()

    # Temporarily disable comprehensive logging integration to prevent duplicate messages
    # try:
    #     from utils.comprehensive_logger import get_comprehensive_logger

    #     comprehensive_logger = get_comprehensive_logger()
    #     if comprehensive_logger:
    #         # Replace with integrated version
    #         system_logger = get_system_logger_with_comprehensive_integration()
    # except ImportError:
    #     pass

# Temporarily set logging to INFO level for debugging
import logging

logging.getLogger().setLevel(logging.INFO)
for handler in logging.getLogger().handlers:
    handler.setLevel(logging.INFO)





# Replace the global system_logger with the integrated version


# -------- I/O and DataFrame troubleshooting helpers (lightweight, no external deps) --------


def _format_bytes(num_bytes: int | None) -> str:
    """Human-friendly byte size formatter."""
    try:
        if num_bytes is None:
            return "n/a"
        step_unit = 1024.0
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(num_bytes)
        for unit in units:
            if size < step_unit:
                return f"{size:.1f}{unit}"
            size /= step_unit
        return f"{size:.1f}PB"
    except Exception:
        return str(num_bytes) if num_bytes is not None else "n/a"


@contextmanager


# -------- Progress heartbeat helpers --------


@contextmanager