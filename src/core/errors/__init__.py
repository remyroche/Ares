"""
Core errors package.

Provides a structured error hierarchy with consistent
error handling across different transports.
"""

# Base error types
from .base import (
    AppError,
    ErrorCode,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ConflictError,
    RateLimitError,
    TimeoutError,
    ServiceUnavailableError,
    BusinessRuleError,
    DataIntegrityError,
)

# Error mapping utilities
from .mapping import (
    ErrorMapper,
    error_mapper,
    map_exception,
    register_exception_mapping,
)

__all__ = [
    # Base errors
    "AppError",
    "ErrorCode",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "ConflictError",
    "RateLimitError",
    "TimeoutError",
    "ServiceUnavailableError",
    "BusinessRuleError",
    "DataIntegrityError",
    
    # Mapping
    "ErrorMapper",
    "error_mapper",
    "map_exception",
    "register_exception_mapping",
]