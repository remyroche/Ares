"""Error Handler Component

Centralized error handling system for data quality checks.
Extracted from raw_data_quality_checker.py
"""

from datetime import datetime
from typing import Any, Optional, List
import traceback
import logging
import numpy as np
import time

from src.utils.logger import system_logger
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

class QualityCheckError(Exception):
    """Base exception for quality check errors."""
    pass

class ValidationError(QualityCheckError):
    """Exception for validation errors."""
    pass

class PreprocessingError(QualityCheckError):
    """Exception for preprocessing errors."""
    pass

class DataDownloadError(QualityCheckError):
    """Exception for data download errors."""
    pass

class ConfigurationError(QualityCheckError):
    """Exception for configuration errors."""
    pass

class ErrorHandler:
    """Centralized error handling for data quality checks.

    This class provides functionality for:
    - Consistent error handling across all components
    - Error categorization and logging
    - Error recovery strategies
    - Error reporting and metrics
    """
    @log_important_calls
    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.logger = system_logger.getChild("ErrorHandler")
        self.config = config or self._get_default_config()
        self.error_counts = {}
        self.error_history = []
    @log_all_calls
    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for error handling."""
        return {
            "error_handling": {
                "log_full_traceback": True,
                "max_error_history": 100,
                "retry_attempts": 3,
                "retry_delay_seconds": 1
            },
            "error_categories": {
                "validation": ["ValidationError", "ValueError", "TypeError"],
                "preprocessing": ["PreprocessingError", "KeyError", "IndexError"],
                "download": ["DataDownloadError", "ConnectionError", "TimeoutError"],
                "configuration": ["ConfigurationError", "KeyError"]
            }
        }

    def handle_validation_error(
        self,
        error: Exception,
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle validation errors consistently.

        Args:
            error: The exception that occurred
            context: Context information about the error

        Returns:
            Standardized error result dictionary
        """
        self._log_error(error, "validation", context)

        return {
            "validation_passed": False,
            "critical_issues": [f"Validation error: {str(error)}"],
            "warnings": [],
            "data_quality_score": 0.0,
            "error_type": type(error).__name__,
            "error_context": context,
            "timestamp": datetime.now().isoformat()
        }

    def handle_preprocessing_error(
        self,
        error: Exception,
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle preprocessing errors consistently.

        Args:
            error: The exception that occurred
            context: Context information about the error

        Returns:
            Standardized error result dictionary
        """
        self._log_error(error, "preprocessing", context)

        return {
            "preprocessing_successful": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_context": context,
            "timestamp": datetime.now().isoformat(),
            "fallback_applied": True
        }

    def handle_download_error(
        self,
        error: Exception,
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle data download errors consistently.

        Args:
            error: The exception that occurred
            context: Context information about the error

        Returns:
            Standardized error result dictionary
        """
        self._log_error(error, "download", context)

        return {
            "download_successful": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_context": context,
            "timestamp": datetime.now().isoformat(),
            "retry_recommended": True
        }

    def handle_configuration_error(
        self,
        error: Exception,
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle configuration errors consistently.

        Args:
            error: The exception that occurred
            context: Context information about the error

        Returns:
            Standardized error result dictionary
        """
        self._log_error(error, "configuration", context)

        return {
            "configuration_valid": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_context": context,
            "timestamp": datetime.now().isoformat(),
            "default_config_applied": True
        }
    @log_all_calls

    def _log_error(
        self,
        error: Exception,
        category: str,
        context: dict[str, Any]
    ) -> None:
        """Log error with appropriate level and details.

        Args:
            error: The exception that occurred
            category: Error category
            context: Context information
        """
        error_type = type(error).__name__
        error_message = str(error)

        # Update error counts
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1

        # Add to error history
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "category": category,
            "context": context
        }

        self.error_history.append(error_record)

        # Limit error history size
        max_history = self.config["error_handling"]["max_error_history"]
        if len(self.error_history) > max_history:
            self.error_history = self.error_history[-max_history:]

        # Log the error
        if category == "validation":
            self.logger.error(f"❌ Validation error: {error_type} - {error_message}")
        elif category == "preprocessing":
            self.logger.error(f"❌ Preprocessing error: {error_type} - {error_message}")
        elif category == "download":
            self.logger.error(f"❌ Download error: {error_type} - {error_message}")
        elif category == "configuration":
            self.logger.error(f"❌ Configuration error: {error_type} - {error_message}")
        else:
            self.logger.error(f"❌ Unknown error: {error_type} - {error_message}")

        # Log full traceback if configured
        if self.config["error_handling"]["log_full_traceback"]:
            self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")

        # Log context if available
        if context:
            self.logger.debug(f"Error context: {context}")

    def get_error_summary(self) -> dict[str, Any]:
        """Get a summary of errors that have occurred.

        Returns:
            Dictionary with error summary
        """
        total_errors = sum(self.error_counts.values())

        # Categorize errors
        error_categories = {}
        for error_type, count in self.error_counts.items():
            category = self._categorize_error(error_type)
            if category not in error_categories:
                error_categories[category] = 0
            error_categories[category] += count

        return {
            "total_errors": total_errors,
            "error_counts": self.error_counts,
            "error_categories": error_categories,
            "recent_errors": self.error_history[-10:] if self.error_history else [],
            "most_common_error": max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        }
    @log_all_calls

    def _categorize_error(self, error_type: str) -> str:
        """Categorize an error type.

        Args:
            error_type: The error type name

        Returns:
            Error category
        """
        error_categories = self.config["error_categories"]

        for category, error_types in error_categories.items():
            if error_type in error_types:
                return category

        return "unknown"

    def clear_error_history(self) -> None:
        """Clear the error history and counts."""
        self.error_counts.clear()
        self.error_history.clear()
        self.logger.info("✅ Error history cleared")

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if an operation should be retried.

        Args:
            error: The exception that occurred
            attempt: Current attempt number (1-based)

        Returns:
            True if should retry, False otherwise
        """
        max_attempts = self.config["error_handling"]["retry_attempts"]

        if attempt >= max_attempts:
            return False

        # Don't retry certain types of errors
        non_retryable_errors = [
            "ConfigurationError",
            "ValueError",
            "TypeError",
            "KeyError",
            "AttributeError"
        ]

        if type(error).__name__ in non_retryable_errors:
            return False

        return True

    def get_retry_delay(self, attempt: int) -> float:
        """Get the delay before retrying an operation.

        Args:
            attempt: Current attempt number (1-based)

        Returns:
            Delay in seconds
        """
        base_delay = self.config["error_handling"]["retry_delay_seconds"]
        # Exponential backoff
        return base_delay * (2 ** (attempt - 1))

    def create_error_result(
        self,
        message: str,
        context: dict[str, Any],
        error_type: str = "ValidationError"
    ) -> dict[str, Any]:
        """Create a standardized error result.

        Args:
            message: Error message
            context: Context information
            error_type: Type of error

        Returns:
            Standardized error result
        """
        return {
            "validation_passed": False,
            "critical_issues": [message],
            "warnings": [],
            "data_quality_score": 0.0,
            "error_type": error_type,
            "error_context": context,
            "timestamp": datetime.now().isoformat()
        }

    def wrap_with_error_handling(self, func, *args, **kwargs):
        """Wrap a function with error handling.

        Args:
            func: Function to wrap
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result or error result
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            context = {
                "function": func.__name__,
                "args": str(args)[:100],  # Truncate for logging
                "kwargs": str(kwargs)[:100]
            }

            if "validate" in func.__name__.lower():
                return self.handle_validation_error(e, context)
            elif "preprocess" in func.__name__.lower():
                return self.handle_preprocessing_error(e, context)
            elif "download" in func.__name__.lower():
                return self.handle_download_error(e, context)
            elif "config" in func.__name__.lower():
                return self.handle_configuration_error(e, context)
            else:
                return self.handle_validation_error(e, context)
