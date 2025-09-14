
"""

"""

# Base error types
from .base import (
    AppError,
    AuthenticationError,
    AuthorizationError,
    BusinessRuleError,
    ConfigurationError,
    ConflictError,
    DataIntegrityError,
    ErrorCode,
    FileOperationError,
    MathValidationError,
    ModelTrainingError,
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
    "ConfigurationError",
    "NotFoundError",
    "ConflictError",
    "RateLimitError",
    "TimeoutError",
    "ServiceUnavailableError",
    "BusinessRuleError",
    "DataIntegrityError",
    "FileOperationError",
    "MathValidationError",
    "ModelTrainingError",
    # Mapping
    "ErrorMapper",
    "error_mapper",
    "map_exception",
    "register_exception_mapping",
]
