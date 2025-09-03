"""
Exception mapping to transport-specific responses.

Maps internal exceptions to appropriate HTTP/gRPC/CLI responses.
"""

import logging
import traceback
from collections.abc import Callable
from typing import Any

from .base import (
    AppError,
    ErrorCode,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    ValidationError,
)
from .base import (
    TimeoutError as AppTimeoutError,
)

logger = logging.getLogger(__name__)


class ErrorMapper:
    """Maps exceptions to AppError instances and transport responses."""

    def __init__(self):
        self._exception_map: dict[type[Exception], Callable[[Exception], AppError]] = {
            # Standard Python exceptions
            ValueError: lambda e: ValidationError(str(e)),
            KeyError: lambda e: NotFoundError(f"Key not found: {e}"),
            TypeError: lambda e: ValidationError(f"Type error: {e}"),
            AttributeError: lambda e: ValidationError(f"Attribute error: {e}"),

            # Network/IO exceptions
            ConnectionError: lambda e: ServiceUnavailableError(
                "Connection failed", service_name="external",
            ),
            TimeoutError: lambda e: AppTimeoutError(str(e)),
            OSError: lambda e: AppError(
                f"System error: {e}",
                code=ErrorCode.INTERNAL_ERROR,
                status_code=500,
            ),
        }

        # Add pandas exceptions if available
        try:
            import pandas as pd
            self._exception_map.update({
                pd.errors.EmptyDataError: lambda e: ValidationError("Empty data provided"),
                pd.errors.ParserError: lambda e: ValidationError(f"Data parsing error: {e}"),
            })
        except ImportError:
            pass

        # Add numpy exceptions if available
        try:
            import numpy as np
            self._exception_map.update({
                np.linalg.LinAlgError: lambda e: ValidationError(f"Linear algebra error: {e}"),
            })
        except ImportError:
            pass

    def register_mapping(
        self,
        exception_type: type[Exception],
        mapper: Callable[[Exception], AppError],
    ) -> None:
        """Register a custom exception mapping."""
        self._exception_map[exception_type] = mapper

    def map_exception(self, exc: Exception) -> AppError:
        """Map an exception to an AppError."""
        # If already an AppError, return as-is
        if isinstance(exc, AppError):
            return exc

        # Check registered mappings
        for exc_type, mapper in self._exception_map.items():
            if isinstance(exc, exc_type):
                try:
                    return mapper(exc)
                except Exception as mapping_error:
                    logger.error(
                        f"Error mapping exception {exc}: {mapping_error}",
                        exc_info=True,
                    )
                    break

        # Default to internal error
        return AppError(
            message=f"Internal error: {type(exc).__name__}",
            code=ErrorCode.INTERNAL_ERROR,
            status_code=500,
            cause=exc,
        )

    def to_http_response(self, error: AppError) -> dict[str, Any]:
        """Convert AppError to HTTP response format."""
        response = {
            "status": error.status_code,
            "headers": {
                "Content-Type": "application/json",
            },
            "body": error.to_dict(),
        }

        # Add specific headers for certain errors
        if isinstance(error, RateLimitError) and "retry_after" in error.details:
            response["headers"]["Retry-After"] = str(error.details["retry_after"])

        return response

    def to_grpc_status(self, error: AppError) -> dict[str, Any]:
        """Convert AppError to gRPC status format."""
        # Map HTTP status codes to gRPC codes
        grpc_code_map = {
            400: 3,   # INVALID_ARGUMENT
            401: 16,  # UNAUTHENTICATED
            403: 7,   # PERMISSION_DENIED
            404: 5,   # NOT_FOUND
            409: 10,  # ABORTED
            429: 8,   # RESOURCE_EXHAUSTED
            500: 13,  # INTERNAL
            503: 14,  # UNAVAILABLE
            504: 4,   # DEADLINE_EXCEEDED
        }

        grpc_code = grpc_code_map.get(error.status_code, 2)  # Default to UNKNOWN

        return {
            "code": grpc_code,
            "message": error.message,
            "details": error.details,
        }

    def to_cli_output(self, error: AppError) -> dict[str, Any]:
        """Convert AppError to CLI output format."""
        # Use color codes for different error types
        color_map = {
            "client": "\033[33m",  # Yellow for client errors
            "server": "\033[31m",  # Red for server errors
            "reset": "\033[0m",
        }

        color = color_map["client"] if error.is_client_error else color_map["server"]

        output = {
            "exit_code": 1 if error.is_client_error else 2,
            "message": f"{color}Error: {error.message}{color_map['reset']}",
            "details": error.details,
        }

        if logger.isEnabledFor(logging.DEBUG) and error.cause:
            output["traceback"] = traceback.format_exception(
                type(error.cause),
                error.cause,
                error.cause.__traceback__,
            )

        return output


# Global mapper instance
error_mapper = ErrorMapper()


def map_exception(exc: Exception) -> AppError:
    """Convenience function to map exceptions."""
    return error_mapper.map_exception(exc)


def register_exception_mapping(
    exception_type: type[Exception],
    mapper: Callable[[Exception], AppError] | type[AppError],
) -> None:
    """
    Register a custom exception mapping.

    Args:
        exception_type: The exception type to map
        mapper: Either a function that maps the exception to AppError,
                or an AppError class to instantiate with str(exception)
    """
    if isinstance(mapper, type) and issubclass(mapper, AppError):
        # If mapper is an AppError class, create a lambda that instantiates it
        error_mapper.register_mapping(
            exception_type,
            lambda e: mapper(str(e)),
        )
    else:
        error_mapper.register_mapping(exception_type, mapper)
