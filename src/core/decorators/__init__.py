from __future__ import annotations

"""
Core decorators package.

Provides a unified, composable decorator system with consistent
behavior for both sync and async functions.
"""

# Core composition utilities
# Authentication/authorization decorators
from .auth import (
    AuthProvider,
    PermissionType,
    User,
    authenticated,
    get_current_user,
    owner_only,
    rate_limit,
    requires_permission,
    requires_role,
    set_auth_provider,
    set_current_user,
)

# Caching decorators
from .cache import (
    CachePolicy,
    cache_invalidate,
    cache_stats,
    cached,
    clear_request_cache,
    memoize,
)
from .compose import (
    compose,
    copy_decorator_metadata,
    ensure_async,
    ensure_sync,
    get_decorator_metadata,
    is_wrapped,
    mark_wrapped,
    set_decorator_metadata,
    uniform_wrapper,
)

# Error handling decorators
from .errors import (
    converts_errors,
    error_boundary,
    handles_errors,
)

# Logging decorators
from .logging import (
    audit_log,
    clear_correlation_id,
    get_correlation_id,
    log_call,
    log_execution_time,
    mask_sensitive_data,
    set_correlation_id,
)

# Retry and resilience decorators
from .retry_timeout import (
    CircuitBreaker,
    CircuitState,
    circuit_breaker,
    fallback,
    retry,
    retry_with_circuit_breaker,
    timeout,
)

# Tracing decorators
from .trace import (
    SpanKind,
    SpanStatus,
    create_trace,
    get_current_trace,
    get_trace_summary,
    set_current_trace,
    span_attribute,
    span_event,
    trace_method,
    traced,
)

# Validation decorators
from .validate import (
    validate_dataframe,
    validate_schema,
    validates,
)

__all__ = [
    # Compose
    "compose",
    "ensure_async",
    "ensure_sync",
    "uniform_wrapper",
    "is_wrapped",
    "mark_wrapped",
    "get_decorator_metadata",
    "set_decorator_metadata",
    "copy_decorator_metadata",
    # Errors
    "handles_errors",
    "error_boundary",
    "converts_errors",
    # Validation
    "validates",
    "validate_schema",
    "validate_dataframe",
    # Retry/resilience
    "retry",
    "timeout",
    "circuit_breaker",
    "retry_with_circuit_breaker",
    "fallback",
    "CircuitBreaker",
    "CircuitState",
    # Logging
    "log_call",
    "log_execution_time",
    "audit_log",
    "get_correlation_id",
    "set_correlation_id",
    "clear_correlation_id",
    "mask_sensitive_data",
    # Tracing
    "traced",
    "trace_method",
    "span_event",
    "span_attribute",
    "get_current_trace",
    "set_current_trace",
    "create_trace",
    "get_trace_summary",
    "SpanKind",
    "SpanStatus",
    # Caching
    "cached",
    "memoize",
    "cache_invalidate",
    "cache_stats",
    "clear_request_cache",
    "CachePolicy",
    # Auth
    "authenticated",
    "requires_role",
    "requires_permission",
    "owner_only",
    "rate_limit",
    "get_current_user",
    "set_current_user",
    "set_auth_provider",
    "User",
    "PermissionType",
    "AuthProvider",
]
