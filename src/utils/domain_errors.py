"""
Domain-specific error types for normalization and validation.
"""
from __future__ import annotations

from typing import Any


class DomainError(Exception):
    """Base class for domain-specific errors raised by decorators.

    Contains a machine-readable "code" and an optional context payload
    for consistent error handling and logging.
    """

    def __init__(self, message: str, *, code: str = "domain_error", context: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.context = context or {}


class DataValidationError(DomainError):
    def __init__(self, message: str, *, context: dict[str, Any] | None = None):
        super().__init__(message, code="data_validation_error", context=context)


class SchemaValidationError(DomainError):
    def __init__(self, message: str, *, context: dict[str, Any] | None = None):
        super().__init__(message, code="schema_validation_error", context=context)


class VectorizationError(DomainError):
    def __init__(self, message: str, *, context: dict[str, Any] | None = None):
        super().__init__(message, code="vectorization_error", context=context)


class ExternalServiceError(DomainError):
    def __init__(self, message: str, *, context: dict[str, Any] | None = None):
        super().__init__(message, code="external_service_error", context=context)


class OperationTimeoutError(DomainError):
    def __init__(self, message: str, *, context: dict[str, Any] | None = None):
        super().__init__(message, code="operation_timeout", context=context)


class AuthenticationError(DomainError):
    def __init__(self, message: str, *, context: dict[str, Any] | None = None):
        super().__init__(message, code="authentication_error", context=context)


class AuthorizationError(DomainError):
    def __init__(self, message: str, *, context: dict[str, Any] | None = None):
        super().__init__(message, code="authorization_error", context=context)


class NotFoundError(DomainError):
    def __init__(self, message: str, *, context: dict[str, Any] | None = None):
        super().__init__(message, code="not_found", context=context)