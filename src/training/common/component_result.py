"""Shared component result data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


class ComponentError(Exception):
    """Domain specific exception raised by components."""


@dataclass(init=False)
class ComponentResult:
    """Result from a pipeline component execution."""

    success: bool
    artifacts: Dict[str, Any]
    metrics: Dict[str, float]
    warnings: List[str]
    error: Optional[Union[Exception, ComponentError]]
    execution_time: float
    metadata: Dict[str, Any]

    def __init__(
        self,
        success: bool,
        artifacts: Optional[Dict[str, Any]] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        execution_time: float = 0.0,
        metrics: Optional[Dict[str, float]] = None,
        warnings: Optional[List[str]] = None,
        error: Optional[Union[Exception, ComponentError]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        if error is not None and error_message is not None:
            raise ValueError("Provide either 'error' or 'error_message', not both.")

        artifacts = dict(artifacts or {})
        metadata = dict(metadata or {})
        metrics = dict(metrics or {})
        warnings = list(warnings or [])

        if error is None and error_message:
            error = ComponentError(error_message)

        self.success = success
        self.artifacts = artifacts
        self.metrics = metrics
        self.warnings = warnings
        self.error = error
        self.execution_time = execution_time
        self.metadata = metadata

        self.__post_init__()

    def __post_init__(self) -> None:
        if (self.success and self.error is not None) or (not self.success and self.error is None):
            raise ValueError(
                "ComponentResult invariant violated: 'error' must be None exactly when 'success' is True."
            )

    @property
    def error_message(self) -> Optional[str]:
        """Backward compatible view exposing the error message as a string."""

        if self.error is None:
            return None
        return str(self.error)

    @error_message.setter
    def error_message(self, value: Optional[str]) -> None:
        if value is None:
            self.error = None
            self.success = True
            return

        self.error = ComponentError(value)
        self.success = False


__all__ = ["ComponentError", "ComponentResult"]
