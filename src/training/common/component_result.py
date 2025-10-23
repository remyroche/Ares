"""Shared component result data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

def _normalise_message(value: Optional[object]) -> str:
    """Return a user friendly string representation for ``value``."""

    if value is None:
        return ""
    text = str(value).strip()
    return text

@dataclass(init=False)
class ComponentError(Exception):
    """Structured error representation emitted by components."""

    code: str
    message: str
    details: Dict[str, Any]
    cause: Optional[BaseException]

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        *,
        details: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        normalized_message = _normalise_message(message)
        if not normalized_message:
            normalized_message = "Unspecified component error"
        super().__init__(normalized_message)
        self.code = (code or "UNKNOWN").strip() or "UNKNOWN"
        self.message = normalized_message
        self.details = dict(details or {})
        self.cause = cause

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON serialisable representation of the error."""

        payload: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }
        if self.cause is not None:
            payload["cause"] = repr(self.cause)
        return payload

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
        errors: Optional[Iterable[Union[str, ComponentError, Exception, Mapping[str, Any]]]] = None,
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
            text = _normalise_message(message)
            if text:
                normalized_warnings.append(text)

        converted_errors: List[ComponentError] = []
        seen_error_keys: set[tuple[str, str]] = set()

        def _coerce_error(value: Union[str, ComponentError, Exception, Mapping[str, Any]]) -> ComponentError:
            if isinstance(value, ComponentError):
                return value
            if isinstance(value, Mapping):
                code = _normalise_message(value.get("code")) or "UNKNOWN"
                message_text = _normalise_message(value.get("message"))
                details = value.get("details") if isinstance(value.get("details"), Mapping) else None
                return ComponentError(message_text or "Unspecified component error", code=code, details=details)
            if isinstance(value, Exception):
                message_text = _normalise_message(value)
                return ComponentError(message_text or value.__class__.__name__, cause=value)
            message_text = _normalise_message(value)
            return ComponentError(message_text or "Unspecified component error")

        def _append_error(value: Union[str, ComponentError, Exception, Mapping[str, Any]]) -> None:
            component_error = _coerce_error(value)
            dedupe_key = (component_error.code, component_error.message)
            if dedupe_key in seen_error_keys:
                return
            seen_error_keys.add(dedupe_key)
            converted_errors.append(component_error)

        if error is None and error_message:
            _append_error(ComponentError(error_message))
        elif error is not None:
            _append_error(error)

        if errors is not None:
            for item in errors:
                if item is None:
                    continue
                _append_error(item)

        if not success and not converted_errors:
            _append_error(ComponentError("Component failed without emitting an error"))

        self.success = success
        self.artifacts = artifacts
        self.metrics = metrics
        self.warnings = normalized_warnings
        self.errors = converted_errors
        self.error = converted_errors[0] if converted_errors else None
        self.execution_time = execution_time
        self.metadata = metadata

        self.__post_init__()

    def __post_init__(self) -> None:
        if self.success and self.errors:
            raise ValueError(
                "ComponentResult invariant violated: 'errors' must be empty when 'success' is True."
            )
        if not self.success and not self.errors:
            raise ValueError(
                "ComponentResult invariant violated: at least one error is required when 'success' is False."
            )
        if self.success:
            self.error = None

    @property
    def error_message(self) -> Optional[str]:
        """Backward compatible view exposing the error message as a string."""

        if not self.errors:
            return None
        return self.errors[0].message

    @error_message.setter
    def error_message(self, value: Optional[str]) -> None:
        if value is None:
            self.success = True
            self.errors.clear()
            self.error = None
            return

        component_error = ComponentError(value)
        self.success = False
        self.errors = [component_error]
        self.error = component_error

__all__ = ["ComponentError", "ComponentResult"]
