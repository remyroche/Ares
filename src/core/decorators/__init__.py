"""
Core decorators package.

Provides a unified, composable decorator system with consistent
behavior for both sync and async functions.
"""

# Core composition utilities
from .compose import (
    compose,
    ensure_async,
    ensure_sync,
    uniform_wrapper,
    is_wrapped,
    mark_wrapped,
    get_decorator_metadata,
    set_decorator_metadata,
    copy_decorator_metadata,
)

# Error handling decorators
from .errors import (
    handles_errors,
    error_boundary,
    converts_errors,
)

# Validation decorators
from .validate import (
    validates,
    validate_schema,
    validate_dataframe,
)

# Retry and resilience decorators
from .retry_timeout import (
    retry,
    timeout,
    circuit_breaker,
    retry_with_circuit_breaker,
    fallback,
    CircuitBreaker,
    CircuitState,
)

# Logging decorators
from .logging import (
    log_call,
    log_execution_time,
    audit_log,
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
    mask_sensitive_data,
)

# Tracing decorators
from .trace import (
    traced,
    trace_method,
    span_event,
    span_attribute,
    get_current_trace,
    set_current_trace,
    create_trace,
    get_trace_summary,
    SpanKind,
    SpanStatus,
)

# Caching decorators
from .cache import (
    cached,
    memoize,
    cache_invalidate,
    cache_stats,
    clear_request_cache,
    CachePolicy,
)

# Authentication/authorization decorators
from .auth import (
    authenticated,
    requires_role,
    requires_permission,
    owner_only,
    rate_limit,
    get_current_user,
    set_current_user,
    set_auth_provider,
    User,
    PermissionType,
    AuthProvider,
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