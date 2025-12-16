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

__all__ = [
    "AuthConfig",
    "APIKeyPermission", 
    "AuthManager",
    "auth_manager"
]
