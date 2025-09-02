"""Backwards compatibility layer for existing decorator usage."""

import warnings
from typing import Any, Callable, TypeVar, Optional, List, Dict

# Import new decorators
from .decorators import (
    validate_call_or_runtime_types,
    pa_check_input,
    pa_check_output,
    pa_check_io,
    enforce_ndarray,
    auto_vectorize,
    guard_array_nan_inf,
    guard_dataframe_nulls,
    normalize_errors,
    with_tracing_span,
)

# Import new enhanced decorators
from .enhanced_decorators import (
    smart_error_recovery,
    cached_validation,
    enhanced_validation,
    performance_monitor_v2,
)

# Import registry and config
from .decorator_registry import decorator_registry
from .decorator_config import global_config

F = TypeVar('F', bound=Callable[..., Any])

def _deprecation_warning(old_name: str, new_name: str, removal_version: str = "2.0.0") -> None:
    """Emit deprecation warning."""
    warnings.warn(
        f"Decorator '{old_name}' is deprecated and will be removed in version {removal_version}. "
        f"Use '{new_name}' instead.",
        DeprecationWarning,
        stacklevel=3
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
    return guard_dataframe_nulls(*args, **kwargs)

def enforce_array(*args, **kwargs):
    """Deprecated: Use enforce_ndarray instead."""
    _deprecation_warning("enforce_array", "enforce_ndarray")
    return enforce_ndarray(*args, **kwargs)

def normalize_error(*args, **kwargs):
    """Deprecated: Use normalize_errors instead."""
    _deprecation_warning("normalize_error", "normalize_errors")
    return normalize_errors(*args, **kwargs)

def tracing_span(*args, **kwargs):
    """Deprecated: Use with_tracing_span instead."""
    _deprecation_warning("tracing_span", "with_tracing_span")
    return with_tracing_span(*args, **kwargs)

# Enhanced decorator compatibility
def smart_recovery(*args, **kwargs):
    """Deprecated: Use smart_error_recovery instead."""
    _deprecation_warning("smart_recovery", "smart_error_recovery")
    return smart_error_recovery(*args, **kwargs)

def cached_validate(*args, **kwargs):
    """Deprecated: Use cached_validation instead."""
    _deprecation_warning("cached_validate", "cached_validation")
    return cached_validation(*args, **kwargs)

def enhanced_validate(*args, **kwargs):
    """Deprecated: Use enhanced_validation instead."""
    _deprecation_warning("enhanced_validate", "enhanced_validation")
    return enhanced_validation(*args, **kwargs)

def performance_monitor(*args, **kwargs):
    """Deprecated: Use performance_monitor_v2 instead."""
    _deprecation_warning("performance_monitor", "performance_monitor_v2")
    return performance_monitor_v2(*args, **kwargs)

# Registry compatibility functions
def register_decorator(name: str, decorator: Callable, **metadata) -> None:
    """Register a decorator with the global registry."""
    decorator_registry.register(
        name=name,
        decorator=decorator,
        **metadata
    )

def get_decorator(name: str) -> Optional[Callable]:
    """Get a decorator from the global registry."""
    return decorator_registry.get(name)

def list_decorators() -> List[str]:
    """List all registered decorators."""
    return decorator_registry.list_decorators()

def is_deprecated(name: str) -> bool:
    """Check if a decorator is deprecated."""
    metadata = decorator_registry.get_metadata(name)
    return metadata.deprecated if metadata else False

# Configuration compatibility
def get_decorator_config() -> Dict[str, Any]:
    """Get global decorator configuration."""
    return global_config.get_config()

def set_decorator_config(config: Dict[str, Any]) -> None:
    """Set global decorator configuration."""
    global_config.update_config(config)

def reset_decorator_config() -> None:
    """Reset global decorator configuration to defaults."""
    global_config.reset_config()

# Migration helper functions
def migrate_decorator_usage(old_decorator_name: str, new_decorator_name: str) -> None:
    """Register a migration path for decorator usage."""
    old_decorator = globals().get(old_decorator_name)
    if old_decorator:
        # Register the old decorator as deprecated
        decorator_registry.register(
            name=old_decorator_name,
            decorator=old_decorator,
            deprecated=True,
            migration_target=new_decorator_name
        )

def get_migration_plan() -> Dict[str, str]:
    """Get the migration plan for deprecated decorators."""
    migration_plan = {}
    for name in decorator_registry.list_decorators():
        metadata = decorator_registry.get_metadata(name)
        if metadata and metadata.deprecated:
            # Try to get migration target from metadata
            migration_target = getattr(metadata, 'migration_target', None)
            if migration_target:
                migration_plan[name] = migration_target
    
    return migration_plan

# Auto-register all legacy decorators
def _register_legacy_decorators():
    """Register all legacy decorators for compatibility."""
    legacy_mappings = {
        "validate_call": "validate_call_or_runtime_types",
        "check_input": "pa_check_input",
        "check_output": "pa_check_output",
        "check_io": "pa_check_io",
        "vectorize": "auto_vectorize",
        "guard_nan_inf": "guard_array_nan_inf",
        "guard_nulls": "guard_dataframe_nulls",
        "enforce_array": "enforce_ndarray",
        "normalize_error": "normalize_errors",
        "tracing_span": "with_tracing_span",
        "smart_recovery": "smart_error_recovery",
        "cached_validate": "cached_validation",
        "enhanced_validate": "enhanced_validation",
        "performance_monitor": "performance_monitor_v2",
    }
    
    for old_name, new_name in legacy_mappings.items():
        if old_name in globals():
            migrate_decorator_usage(old_name, new_name)

# Initialize legacy decorator registration
_register_legacy_decorators()

# Export all legacy decorators for backwards compatibility
__all__ = [
    # Legacy decorators
    "validate_call",
    "check_input", 
    "check_output",
    "check_io",
    "vectorize",
    "guard_nan_inf",
    "guard_nulls",
    "enforce_array",
    "normalize_error",
    "tracing_span",
    "smart_recovery",
    "cached_validate",
    "enhanced_validate",
    "performance_monitor",
    
    # Registry functions
    "register_decorator",
    "get_decorator",
    "list_decorators",
    "is_deprecated",
    
    # Configuration functions
    "get_decorator_config",
    "set_decorator_config",
    "reset_decorator_config",
    
    # Migration functions
    "migrate_decorator_usage",
    "get_migration_plan",
]