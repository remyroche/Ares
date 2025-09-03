from __future__ import annotations
"""
Base error hierarchy with codes and status mapping.

Provides a structured error system with machine-readable codes,
HTTP/gRPC status mappings, and rich context.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Standard error codes for the application."""

    # Client errors (4xx equivalent)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    RATE_LIMITED = "RATE_LIMITED"
    BAD_REQUEST = "BAD_REQUEST"

    # Server errors (5xx equivalent)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"

    # Business logic errors
    BUSINESS_RULE_VIOLATION = "BUSINESS_RULE_VIOLATION"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"

    # Data errors
    DATA_INTEGRITY_ERROR = "DATA_INTEGRITY_ERROR"
    DATA_NOT_FOUND = "DATA_NOT_FOUND"
    STALE_DATA = "STALE_DATA"


@dataclass
class AppError(Exception):
    """
    Base application error with structured information.

    All application errors should inherit from this class to ensure
    consistent error handling across different transports.
    """

    message: str
    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    status_code: int = 500
    details: dict[str, Any] = field(default_factory=dict)
    cause: Exception | None = None

    def __str__(self) -> str:
        """Human-readable error message."""
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "error": {
                "code": self.code.value,
                "message": self.message,
                "status_code": self.status_code,
            },
        }

        if self.details:
            result["error"]["details"] = self.details

        if self.cause:
            result["error"]["cause"] = str(self.cause)

        return result

    @property
    def is_client_error(self) -> bool:
        """Check if this is a client error (4xx range)."""
        return 400 <= self.status_code < 500

    @property
    def is_server_error(self) -> bool:
        """Check if this is a server error (5xx range)."""
        return 500 <= self.status_code < 600


# Specific error types

class ValidationError(AppError):
    """Input validation error."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value

        super().__init__(
            message=message,
            code=ErrorCode.VALIDATION_ERROR,
            status_code=400,
            details=details,
            **kwargs,
        )


class AuthenticationError(AppError):
    """Authentication failure."""

    def __init__(self, message: str = "Authentication required", **kwargs):
        super().__init__(
            message=message,
            code=ErrorCode.AUTHENTICATION_ERROR,
            status_code=401,
            **kwargs,
        )


class AuthorizationError(AppError):
    """Authorization failure."""

    def __init__(
        self,
        message: str = "Insufficient permissions",
        required_permission: str | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if required_permission:
            details["required_permission"] = required_permission

        super().__init__(
            message=message,
            code=ErrorCode.AUTHORIZATION_ERROR,
            status_code=403,
            details=details,
            **kwargs,
        )


class NotFoundError(AppError):
    """Resource not found."""

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id

        super().__init__(
            message=message,
            code=ErrorCode.NOT_FOUND,
            status_code=404,
            details=details,
            **kwargs,
        )


class ConflictError(AppError):
    """Resource conflict."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            code=ErrorCode.CONFLICT,
            status_code=409,
            **kwargs,
        )


class RateLimitError(AppError):
    """Rate limit exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if retry_after:
            details["retry_after"] = retry_after

        super().__init__(
            message=message,
            code=ErrorCode.RATE_LIMITED,
            status_code=429,
            details=details,
            **kwargs,
        )


class TimeoutError(AppError):
    """Operation timeout."""

    def __init__(
        self,
        message: str = "Operation timed out",
        timeout_seconds: float | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds

        super().__init__(
            message=message,
            code=ErrorCode.TIMEOUT,
            status_code=504,
            details=details,
            **kwargs,
        )


class ServiceUnavailableError(AppError):
    """Service temporarily unavailable."""

    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        service_name: str | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if service_name:
            details["service_name"] = service_name

        super().__init__(
            message=message,
            code=ErrorCode.SERVICE_UNAVAILABLE,
            status_code=503,
            details=details,
            **kwargs,
        )


class BusinessRuleError(AppError):
    """Business rule violation."""

    def __init__(
        self,
        message: str,
        rule_name: str | None = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if rule_name:
            details["rule_name"] = rule_name

        super().__init__(
            message=message,
            code=ErrorCode.BUSINESS_RULE_VIOLATION,
            status_code=422,
            details=details,
            **kwargs,
        )


class DataIntegrityError(AppError):
    """Data integrity violation."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            code=ErrorCode.DATA_INTEGRITY_ERROR,
            status_code=422,
            **kwargs,
        )
