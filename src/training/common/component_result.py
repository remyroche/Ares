"""Shared component result data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union


class ComponentError(Exception):
    """Domain specific exception raised by components."""


@dataclass(init=False)
class ComponentResult:
    """Result from a pipeline component execution."""

    success: bool
    artifacts: Dict[str, Any]
    metrics: Dict[str, float]
    warnings: List[str]
    errors: List[ComponentError]
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
        warnings: Optional[Iterable[str]] = None,
        errors: Optional[Iterable[Union[str, ComponentError, Exception]]] = None,
        error: Optional[Union[Exception, ComponentError]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        if error is not None and error_message is not None:
            raise ValueError("Provide either 'error' or 'error_message', not both.")

        artifacts = dict(artifacts or {})
        metadata = dict(metadata or {})
        metrics = dict(metrics or {})
        normalized_warnings: List[str] = []
        for message in warnings or []:
            if message is None:
                continue
            text = str(message).strip()
            if text:
                normalized_warnings.append(text)

        converted_errors: List[ComponentError] = []
        seen_error_messages: set[str] = set()

        def _append_error(value: Union[str, ComponentError, Exception]) -> None:
            message = value
            if isinstance(value, ComponentError):
                component_error = value
                message_text = str(component_error)
            else:
                if isinstance(value, Exception):
                    message = str(value)
                message_text = str(message)
                component_error = ComponentError(message_text)
            if message_text in seen_error_messages:
                return
            seen_error_messages.add(message_text)
            converted_errors.append(component_error)

        if error is None and error_message:
            error = ComponentError(error_message)
            _append_error(error)
        elif error is not None:
            _append_error(error)

        if errors is not None:
            for item in errors:
                if item is None:
                    continue
                _append_error(item)

        self.success = success
        self.artifacts = artifacts
        self.metrics = metrics
        self.warnings = normalized_warnings
        self.errors = converted_errors
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
            if not self.errors:
                return None
            return str(self.errors[0])
        return str(self.error)

    @error_message.setter
    def error_message(self, value: Optional[str]) -> None:
        if value is None:
            self.error = None
            self.success = True
            self.errors.clear()
            return

        component_error = ComponentError(value)
        self.error = component_error
        self.success = False
        self.errors = [component_error]


__all__ = ["ComponentError", "ComponentResult"]
