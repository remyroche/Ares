
"""

"""

# Base error types
from .base import (
    AppError,
    AuthenticationError,
    AuthorizationError,
    BusinessRuleError,
    ConflictError,
    DataIntegrityError,
    ErrorCode,
    FileOperationError,
    MathValidationError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    TimeoutError,
    ValidationError,
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
    "FileOperationError",
    "MathValidationError",
    # Mapping
    "ErrorMapper",
    "error_mapper",
    "map_exception",
    "register_exception_mapping",
]
