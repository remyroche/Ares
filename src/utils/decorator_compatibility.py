"""Backwards compatibility layer for existing decorator usage."""

import warnings
from collections.abc import Callable
from typing import Any, TypeVar

from .decorator_config import global_config

# Import registry and config
from .decorator_registry import decorator_registry

# Import new decorators
from .decorators import (
    auto_vectorize,
    guard_array_nan_inf,
    guard_dataframe_nulls,
    normalize_errors,
    pa_check_input,
    pa_check_io,
    pa_check_output,
    validate_call_or_runtime_types,
    with_tracing_span,
)

# Import new enhanced decorators
from .enhanced_decorators import (
    cached_validation,
    enhanced_validation,
    performance_monitor_v2,
    smart_error_recovery,
)

F = TypeVar("F", bound=Callable[..., Any])


def _deprecation_warning(old_name: str, new_name: str, removal_version: str = "3.0"):
    """Emit deprecation warning."""
    warnings.warn(
        f"Decorator '{old_name}' is deprecated and will be removed in version {removal_version}. "
        f"Use '{new_name}' instead.",
        DeprecationWarning,
        stacklevel=3,
    )


# Legacy decorator mappings for backwards compatibility
def validate_call(*args, **kwargs):
    """Deprecated: Use validate_call_or_runtime_types instead."""
    _deprecation_warning("validate_call", "validate_call_or_runtime_types")
    return validate_call_or_runtime_types(*args, **kwargs)


def check_input(*args, **kwargs):
    """Deprecated: Use pa_check_input instead."""
    _deprecation_warning("check_input", "pa_check_input")
    return pa_check_input(*args, **kwargs)


def check_output(*args, **kwargs):
    """Deprecated: Use pa_check_output instead."""
    _deprecation_warning("check_output", "pa_check_output")
    return pa_check_output(*args, **kwargs)


def check_io(*args, **kwargs):
    """Deprecated: Use pa_check_io instead."""
    _deprecation_warning("check_io", "pa_check_io")
    return pa_check_io(*args, **kwargs)


def vectorize(*args, **kwargs):
    """Deprecated: Use auto_vectorize instead."""
    _deprecation_warning("vectorize", "auto_vectorize")
    return auto_vectorize(*args, **kwargs)


def guard_nan_inf(*args, **kwargs):
    """Deprecated: Use guard_array_nan_inf instead."""
    _deprecation_warning("guard_nan_inf", "guard_array_nan_inf")
    return guard_array_nan_inf(*args, **kwargs)


def guard_nulls(*args, **kwargs):
    """Deprecated: Use guard_dataframe_nulls instead."""
    _deprecation_warning("guard_nulls", "guard_dataframe_nulls")
    return validates(*args, **kwargs)


def error_handler(*args, **kwargs):
    """Deprecated: Use normalize_errors instead."""
    _deprecation_warning("error_handler", "normalize_errors")
    return normalize_errors(*args, **kwargs)


def tracing(*args, **kwargs):
    """Deprecated: Use with_tracing_span instead."""
    _deprecation_warning("tracing", "with_tracing_span")
    return traced(*args, **kwargs)


# Enhanced decorator aliases for easier access
def smart_recovery(*args, **kwargs):
    """Alias for smart_error_recovery."""
    return smart_error_recovery(*args, **kwargs)


def cached(*args, **kwargs):
    """Alias for cached_validation."""
    return cached_validation(*args, **kwargs)


def validation(*args, **kwargs):
    """Alias for enhanced_validation."""
    return enhanced_validation(*args, **kwargs)


def performance(*args, **kwargs):
    """Alias for performance_monitor_v2."""
    return performance_monitor_v2(*args, **kwargs)


# Configuration helpers for backwards compatibility
def get_decorator_config():
    """Get global decorator configuration."""
    return global_config


def set_decorator_config(**kwargs):
    """Update global decorator configuration."""
    for key, value in kwargs.items():
        if hasattr(global_config, key):
            setattr(global_config, key, value)
        else:
            warnings.warn(f"Unknown configuration key: {key}", stacklevel=2)


def list_available_decorators(include_deprecated: bool = False):
    """List all available decorators."""
    return decorator_registry.list_decorators(include_deprecated=include_deprecated)


def get_decorator_usage_stats():
    """Get usage statistics for all decorators."""
    return decorator_registry.get_usage_stats()


def search_decorators(query: str):
    """Search decorators by name, description, or tags."""
    return decorator_registry.search(query)


# Legacy decorator factory for easy migration
def legacy_decorator_factory(legacy_name: str, new_name: str):
    """Create a legacy decorator that maps to a new one."""

    def decorator(*args, **kwargs):
        _deprecation_warning(legacy_name, new_name)
        # Import the new decorator dynamically to avoid circular imports
        if new_name == "smart_error_recovery":
            return smart_error_recovery(*args, **kwargs)
        if new_name == "cached_validation":
            return cached_validation(*args, **kwargs)
        if new_name == "enhanced_validation":
            return enhanced_validation(*args, **kwargs)
        if new_name == "performance_monitor_v2":
            return performance_monitor_v2(*args, **kwargs)
        # Fallback to importing from the main decorators module
        import importlib

        decorators_module = importlib.import_module("src.utils.decorators")
        new_decorator = getattr(decorators_module, new_name)
        return new_decorator(*args, **kwargs)

    return decorator


# Register legacy decorators in the registry for discovery
decorator_registry.register(
    name="validate_call",
    decorator=validate_call,
    version="1.0",
    description="Legacy decorator - use validate_call_or_runtime_types instead",
    tags=["legacy", "deprecated"],
    deprecated=True,
    aliases=["validate_call"],
)

decorator_registry.register(
    name="check_input",
    decorator=check_input,
    version="1.0",
    description="Legacy decorator - use pa_check_input instead",
    tags=["legacy", "deprecated"],
    deprecated=True,
    aliases=["check_input"],
)

decorator_registry.register(
    name="check_output",
    decorator=check_output,
    version="1.0",
    description="Legacy decorator - use pa_check_output instead",
    tags=["legacy", "deprecated"],
    deprecated=True,
    aliases=["check_output"],
)

# Export all decorators for backwards compatibility
__all__ = [
    # New enhanced decorators
    "smart_error_recovery",
    "cached_validation",
    "enhanced_validation",
    "performance_monitor_v2",
    # Legacy compatibility decorators
    "validate_call",
    "check_input",
    "check_output",
    "check_io",
    "vectorize",
    "guard_nan_inf",
    "guard_nulls",
    "error_handler",
    "tracing",
    # Enhanced decorator aliases
    "smart_recovery",
    "cached",
    "validation",
    "performance",
    # Configuration helpers
    "get_decorator_config",
    "set_decorator_config",
    "list_available_decorators",
    "get_decorator_usage_stats",
    "search_decorators",
    # Registry access
    "decorator_registry",
    "global_config",
]
