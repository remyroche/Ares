"""
Enhanced Error Handling for Training Steps with Comprehensive Logging and Recovery.

This module provides specialized error handling for training steps with:
- Detailed step-by-step logging
- Progress tracking
- Recovery strategies
- Performance monitoring
- Validation integration
"""

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Awaitable, Callable as TypingCallable, TypeVar, cast
import asyncio
import functools
import logging
import time
import traceback

from dataclasses import dataclass, field
from enum import Enum
import pandas as pd

# Type variables
T = TypeVar("T")
F = TypeVar("F", bound=TypingCallable[..., Any])


class StepStatus(Enum):
    """Status of training step execution."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    RECOVERED = "recovered"
    SKIPPED = "skipped"


@dataclass
class StepExecutionContext:
    """Context for step execution with detailed tracking."""

    step_name: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: StepStatus = StepStatus.NOT_STARTED
    error: Exception | None = None
    error_traceback: str | None = None
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    performance_metrics: dict[str, Any] = field(default_factory=dict)
    validation_results: dict[str, Any] = field(default_factory=dict)
    progress_messages: list[str] = field(default_factory=list)

    def add_progress(self, message: str) -> None:
        """Add a progress message with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        self.progress_messages.append(f"[{timestamp}] {message}")

    def set_error(self, error: Exception) -> None:
        """Set error details."""
        self.error = error
        self.error_traceback = traceback.format_exc()
        self.status = StepStatus.FAILED
        self.end_time = time.time()

    def mark_success(self) -> None:
        """Mark step as successful."""
        self.status = StepStatus.SUCCESS
        self.end_time = time.time()

    def mark_recovered(self) -> None:
        """Mark step as recovered."""
        self.status = StepStatus.RECOVERED
        self.end_time = time.time()


class TrainingStepErrorHandler:
    """Enhanced error handler specifically for training steps."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.execution_contexts: dict[str, StepExecutionContext] = {}
        self.global_start_time = time.time()

    def log_step_start(self, step_name: str, **kwargs: Any) -> None:
        """Log step start with context."""
        context = self.get_context(step_name)
        context.status = StepStatus.IN_PROGRESS
        context.add_progress(f"🚀 Starting {step_name}")

        # Log key parameters
        if kwargs:
            param_str = ", ".join([f"{k}={v}" for k, v in kwargs.items() if v is not None])
            context.add_progress(f"Parameters: {param_str}")
        else:
            param_str = ""

        self.logger.info(f"🚀 Step {step_name} started")
        if kwargs:
            self.logger.info(f"📋 {step_name} parameters: {param_str}")

    def log_step_progress(self, step_name: str, message: str, level: str = "info") -> None:
        """Log step progress with context."""
        context = self.get_context(step_name)
        context.add_progress(message)

        log_method = getattr(self.logger, level, self.logger.info)
        log_method(f"📊 {step_name}: {message}")

    def log_step_success(self, step_name: str, result: Any | None = None) -> None:
        """Log step success with context."""
        context = self.get_context(step_name)
        context.mark_success()
        duration = context.get_duration()

        context.add_progress(f"✅ Completed successfully in {duration:.2f}s")
        self.logger.info(
            f"✅ Step {step_name} completed successfully in {duration:.2f}s",
        )

        if result is not None:
            if isinstance(result, dict):
                result_summary = {k: v for k, v in result.items() if v is not None}
                context.add_progress(f"Results: {result_summary}")
                self.logger.info(f"📊 {step_name} results: {result_summary}")
            else:
                context.add_progress(f"Result: {result}")
                self.logger.info(f"📊 {step_name} result: {result}")

    def log_step_error(self, step_name: str, error: Exception, recovery_attempt: bool = False) -> None:
        """Log step error with context."""
        context = self.get_context(step_name)
        context.set_error(error)

        if recovery_attempt:
            context.recovery_attempts += 1
            context.add_progress(
                f"🔄 Recovery attempt {context.recovery_attempts}/{context.max_recovery_attempts}",
            )
            self.logger.warning(
                f"🔄 {step_name} recovery attempt {context.recovery_attempts}/{context.max_recovery_attempts}: {error}",
            )
        else:
            context.add_progress(f"❌ Failed: {error}")
            self.logger.error(f"❌ Step {step_name} failed: {error}")
            self.logger.error(
                f"📋 {step_name} error traceback:\n{context.error_traceback}",
            )

    def log_step_recovery(self, step_name: str, recovery_method: str) -> None:
        """Log step recovery with context."""
        context = self.get_context(step_name)
        context.mark_recovered()
        context.add_progress(f"🔄 Recovered using: {recovery_method}")
        self.logger.info(f"🔄 Step {step_name} recovered using: {recovery_method}")


# Global error handler instance
_global_handler = TrainingStepErrorHandler()


def get_training_error_handler() -> TrainingStepErrorHandler:
    """Get the global training error handler."""
    return _global_handler


def training_step_error_handler(
    step_name: str,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    default_return: T | None = None,
    enable_recovery: bool = True,
    max_recovery_attempts: int = 3,
    log_performance: bool = True,
    validate_output: bool = True,
) -> Callable[[F], F]:
    """
    Enhanced error handling decorator for training steps.

    Args:
        step_name: Name of the training step
        exceptions: Tuple of exceptions to catch
        default_return: Default return value on failure
        enable_recovery: Whether to enable recovery strategies
        max_recovery_attempts: Maximum number of recovery attempts
        log_performance: Whether to log performance metrics
        validate_output: Whether to validate step output
    """

    return decorator


def _validate_step_output(step_name: str, result: Any) -> dict[str, Any]:
    """Validate step output based on step type."""
    validation_result: dict[str, Any] = {"valid": True, "warnings": []}

    if step_name.startswith("step01_7"):
        # Validate HMM regime discovery output
        if isinstance(result, bool):
            if not result:
                validation_result["warnings"].append(
                    "Step returned False - may indicate failure",
                )
        elif isinstance(result, dict):
            if "status" in result and result["status"] != "SUCCESS":
                validation_result["warnings"].append(
                    f"Status indicates failure: {result['status']}",
                )

    elif step_name.startswith("step5"):
        # Validate model training output
        if isinstance(result, bool):
            if not result:
                validation_result["warnings"].append(
                    "Training step returned False - may indicate failure",
                )
        elif isinstance(result, dict):
            if "models" in result and not result["models"]:
                validation_result["warnings"].append("No models were trained")

    elif step_name.startswith("step6"):
        # Validate enhancement output
        if isinstance(result, bool) and not result:
            validation_result["warnings"].append(
                "Enhancement step returned False - may indicate failure",
            )

    return validation_result


async def _attempt_recovery(
    step_name: str,
    func: Callable[..., Awaitable[T | None]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    error: Exception,
) -> T | None:
    """Attempt recovery for async functions."""
    handler = get_training_error_handler()
    handler.get_context(step_name)

    # Simple recovery strategies based on step type
    if step_name.startswith("step01_7"):
        # For HMM regime discovery, try with different parameters
        handler.log_step_progress(
            step_name,
            "Attempting recovery with reduced complexity",
        )
        recovery_kwargs = kwargs.copy()
        recovery_kwargs["target_num_clusters"] = min(
            kwargs.get("target_num_clusters", 20),
            10,
        )
        recovery_kwargs["min_combination_frequency"] = max(
            kwargs.get("min_combination_frequency", 0.003),
            0.01,
        )
        try:
            return await func(*args, **recovery_kwargs)
        except Exception as recovery_error:  # noqa: BLE001
            handler.log_step_error(step_name, recovery_error, recovery_attempt=True)

    return None


def _attempt_sync_recovery(
    step_name: str,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    error: Exception,
) -> T | None:
    """Attempt recovery for sync functions."""
    handler = get_training_error_handler()
    handler.get_context(step_name)

    # Similar recovery strategies for sync functions
    if step_name.startswith("step01_7"):
        handler.log_step_progress(
            step_name,
            "Attempting recovery with reduced complexity",
        )
        recovery_kwargs = kwargs.copy()
        recovery_kwargs["target_num_clusters"] = min(
            kwargs.get("target_num_clusters", 20),
            10,
        )
        recovery_kwargs["min_combination_frequency"] = max(
            kwargs.get("min_combination_frequency", 0.003),
            0.01,
        )
        try:
            return func(*args, **recovery_kwargs)
        except Exception as recovery_error:  # noqa: BLE001
            handler.log_step_error(step_name, recovery_error, recovery_attempt=True)

    return None


@contextmanager
