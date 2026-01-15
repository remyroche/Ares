"""
Authentication package for exchanges.

This package provides shared authentication utilities and configurations
for various cryptocurrency exchanges.
"""

from .auth_manager import (
    AuthConfig,
    APIKeyPermission,
    AuthManager,
    auth_manager
)

# Re-export compatibility classes
try:
    from ..auth_compat import (
        AuthenticationManager,
        APIKeyManager,
        TimeSyncManager,
        SubaccountManager
    )
except ImportError:
    # If auth_compat isn't available, define stubs
    class AuthenticationManager:
        def __init__(self, *args, **kwargs) -> None: pass
        def register_auth_functions(self, *args, **kwargs): return None

    class APIKeyManager:
        def __init__(self, *args, **kwargs) -> None: pass

    class TimeSyncManager:
        def __init__(self, *args, **kwargs) -> None: pass

    class SubaccountManager:
        def __init__(self, *args, **kwargs) -> None: pass

__all__ = [
    "AuthConfig",
    "APIKeyPermission", 
    "AuthManager",
    "auth_manager",
    "AuthenticationManager",
    "APIKeyManager",
    "TimeSyncManager",
    "SubaccountManager"
]
