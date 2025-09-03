from __future__ import annotations

"""
Authentication and authorization decorators.

Provides decorators for enforcing authentication and permission
policies in a framework-agnostic way.
"""

from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import A, Callableny, Optional

from src.core.errors.base import AuthenticationError, AuthorizationError

from .compose import P, R, uniform_wrapper
from .logging import get_correlation_id

# Context variable for current user
current_user_var: ContextVar[Optional["User"]] = ContextVar(
    "current_user", default=None
)


class PermissionType(Enum):
    """Types of permissions."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"


@dataclass
class User:
    """User model for authentication/authorization."""

    id: str
    username: str
    email: str | None = None
    roles: set[str] = None
    permissions: set[str] = None
    attributes: dict[str, Any] = None

    def __post_init__(self):
        if self.roles is None:
            self.roles = set()
        if self.permissions is None:
            self.permissions = set()
        if self.attributes is None:
            self.attributes = {}

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles

    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions

    def has_any_role(self, roles: list[str]) -> bool:
        """Check if user has any of the specified roles."""
        return any(self.has_role(role) for role in roles)

    def has_all_roles(self, roles: list[str]) -> bool:
        """Check if user has all specified roles."""
        return all(self.has_role(role) for role in roles)

    def has_any_permission(self, permissions: list[str]) -> bool:
        """Check if user has any of the specified permissions."""
        return any(self.has_permission(perm) for perm in permissions)

    def has_all_permissions(self, permissions: list[str]) -> bool:
        """Check if user has all specified permissions."""
        return all(self.has_permission(perm) for perm in permissions)


class AuthProvider(ABC):
    """Abstract authentication provider."""

    @abstractmethod
    def get_current_user(self) -> User | None:
        """Get the current authenticated user."""

    @abstractmethod
    def validate_token(self, token: str) -> User | None:
        """Validate a token and return the user."""


class SimpleAuthProvider(AuthProvider):
    """Simple auth provider using context variables."""

    def get_current_user(self) -> User | None:
        """Get user from context variable."""
        return current_user_var.get()

    def validate_token(self, token: str) -> User | None:
        """Simple token validation (override in production)."""
        # This is a placeholder - implement real token validation
        if token == "valid_token":
            return User(id="1", username="test_user")
        return None


# Global auth provider
_auth_provider: AuthProvider = SimpleAuthProvider()


def set_auth_provider(provider: AuthProvider) -> None:
    """Set the global authentication provider."""
    global _auth_provider
    _auth_provider = provider


def get_current_user() -> User | None:
    """Get the current authenticated user."""
    return _auth_provider.get_current_user()


def set_current_user(user: User | None) -> None:
    """Set the current user in context."""
    current_user_var.set(user)


def authenticated(
    *,
    optional: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Require authentication for the decorated function.

    Args:
        optional: If True, allow unauthenticated access

    Example:
        @authenticated()
        def get_profile() -> dict:
            user = get_current_user()
            return {"id": user.id, "username": user.username}

        @authenticated(optional=True)
        def get_public_data() -> dict:
            user = get_current_user()
            if user:
                return {"data": "private_data"}
            return {"data": "public_data"}
    """

    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        user = get_current_user()

        if not user and not optional:
            msg = "Authentication required"
            raise AuthenticationError(
                msg,
                details={
                    "function": func.__name__,
                    "correlation_id": get_correlation_id(),
                },
            )

        return func(*args, **kwargs)

    async def async_handler(
        func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        user = get_current_user()

        if not user and not optional:
            msg = "Authentication required"
            raise AuthenticationError(
                msg,
                details={
                    "function": func.__name__,
                    "correlation_id": get_correlation_id(),
                },
            )

        return await func(*args, **kwargs)

    return uniform_wrapper("authenticated", sync_handler, async_handler)


def requires_role(
    *roles: str,
    require_all: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Require specific roles for access.

    Args:
        *roles: Required roles
        require_all: If True, require all roles; if False, require any role

    Example:
        @requires_role("admin")
        def delete_user(user_id: str) -> bool:
            return database.delete_user(user_id)

        @requires_role("editor", "admin", require_all=False)
        def edit_content(content_id: str) -> dict:
            return database.update_content(content_id)
    """

    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        user = get_current_user()

        if not user:
            msg = "Authentication required"
            raise AuthenticationError(msg)

        has_required_roles = (
            user.has_all_roles(list(roles))
            if require_all
            else user.has_any_role(list(roles))
        )

        if not has_required_roles:
            msg = f"Missing required role(s): {', '.join(roles)}"
            raise AuthorizationError(
                msg,
                required_permission=f"role:{','.join(roles)}",
                details={
                    "required_roles": list(roles),
                    "user_roles": list(user.roles),
                    "require_all": require_all,
                },
            )

        return func(*args, **kwargs)

    async def async_handler(
        func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        user = get_current_user()

        if not user:
            msg = "Authentication required"
            raise AuthenticationError(msg)

        has_required_roles = (
            user.has_all_roles(list(roles))
            if require_all
            else user.has_any_role(list(roles))
        )

        if not has_required_roles:
            msg = f"Missing required role(s): {', '.join(roles)}"
            raise AuthorizationError(
                msg,
                required_permission=f"role:{','.join(roles)}",
                details={
                    "required_roles": list(roles),
                    "user_roles": list(user.roles),
                    "require_all": require_all,
                },
            )

        return await func(*args, **kwargs)

    role_desc = "all:" if require_all else "any:"
    return uniform_wrapper(
        f"requires_role({role_desc}{','.join(roles)})",
        sync_handler,
        async_handler,
    )


def requires_permission(
    *permissions: str,
    require_all: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Require specific permissions for access.

    Args:
        *permissions: Required permissions
        require_all: If True, require all permissions; if False, require any

    Example:
        @requires_permission("users.delete")
        def delete_user(user_id: str) -> bool:
            return database.delete_user(user_id)

        @requires_permission("content.read", "content.write", require_all=True)
        def update_content(content_id: str, data: dict) -> dict:
            return database.update_content(content_id, data)
    """

    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        user = get_current_user()

        if not user:
            msg = "Authentication required"
            raise AuthenticationError(msg)

        has_required_perms = (
            user.has_all_permissions(list(permissions))
            if require_all
            else user.has_any_permission(list(permissions))
        )

        if not has_required_perms:
            msg = f"Missing required permission(s): {', '.join(permissions)}"
            raise AuthorizationError(
                msg,
                required_permission=",".join(permissions),
                details={
                    "required_permissions": list(permissions),
                    "user_permissions": list(user.permissions),
                    "require_all": require_all,
                },
            )

        return func(*args, **kwargs)

    async def async_handler(
        func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        user = get_current_user()

        if not user:
            msg = "Authentication required"
            raise AuthenticationError(msg)

        has_required_perms = (
            user.has_all_permissions(list(permissions))
            if require_all
            else user.has_any_permission(list(permissions))
        )

        if not has_required_perms:
            msg = f"Missing required permission(s): {', '.join(permissions)}"
            raise AuthorizationError(
                msg,
                required_permission=",".join(permissions),
                details={
                    "required_permissions": list(permissions),
                    "user_permissions": list(user.permissions),
                    "require_all": require_all,
                },
            )

        return await func(*args, **kwargs)

    perm_desc = "all:" if require_all else "any:"
    return uniform_wrapper(
        f"requires_permission({perm_desc}{','.join(permissions)})",
        sync_handler,
        async_handler,
    )


def owner_only(
    owner_field: str = "user_id",
    param_name: str = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Ensure user can only access their own resources.

    Args:
        owner_field: Field name containing owner ID
        param_name: Parameter containing the resource (defaults to first arg)

    Example:
        @owner_only(owner_field="user_id")
        def get_user_data(user_data: dict) -> dict:
            # Will check user_data["user_id"] == current_user.id
            return user_data

        @owner_only(owner_field="owner_id", param_name="document")
        def update_document(doc_id: str, document: dict) -> dict:
            # Will check document["owner_id"] == current_user.id
            return database.update_document(doc_id, document)
    """

    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        user = get_current_user()

        if not user:
            msg = "Authentication required"
            raise AuthenticationError(msg)

        # Get the resource to check
        import inspect

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        if param_name:
            if param_name in kwargs:
                resource = kwargs[param_name]
            else:
                param_idx = params.index(param_name)
                resource = args[param_idx]
        else:
            # Default to first argument
            resource = args[0] if args else None

        if not resource:
            msg = "No resource to check ownership"
            raise ValueError(msg)

        # Check ownership
        if isinstance(resource, dict):
            owner_id = resource.get(owner_field)
        else:
            owner_id = getattr(resource, owner_field, None)

        if owner_id != user.id:
            msg = "You can only access your own resources"
            raise AuthorizationError(
                msg,
                details={
                    "owner_field": owner_field,
                    "owner_id": owner_id,
                    "user_id": user.id,
                },
            )

        return func(*args, **kwargs)

    async def async_handler(
        func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        user = get_current_user()

        if not user:
            msg = "Authentication required"
            raise AuthenticationError(msg)

        # Get the resource to check
        import inspect

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        if param_name:
            if param_name in kwargs:
                resource = kwargs[param_name]
            else:
                param_idx = params.index(param_name)
                resource = args[param_idx]
        else:
            # Default to first argument
            resource = args[0] if args else None

        if not resource:
            msg = "No resource to check ownership"
            raise ValueError(msg)

        # Check ownership
        if isinstance(resource, dict):
            owner_id = resource.get(owner_field)
        else:
            owner_id = getattr(resource, owner_field, None)

        if owner_id != user.id:
            msg = "You can only access your own resources"
            raise AuthorizationError(
                msg,
                details={
                    "owner_field": owner_field,
                    "owner_id": owner_id,
                    "user_id": user.id,
                },
            )

        return await func(*args, **kwargs)

    return uniform_wrapper(
        f"owner_only({owner_field})",
        sync_handler,
        async_handler,
    )


def rate_limit(
    *,
    calls: int = 10,
    period: float = 60.0,
    key_func: Callable[[], str] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Rate limit function calls per user.

    Args:
        calls: Maximum number of calls allowed
        period: Time period in seconds
        key_func: Function to generate rate limit key (defaults to user ID)

    Example:
        @rate_limit(calls=5, period=60.0)
        def send_email(to: str, subject: str) -> bool:
            # Max 5 emails per minute per user
            return email_service.send(to, subject)
    """
    from collections import defaultdict
    from time import time

    # Simple in-memory rate limiter (use Redis in production)
    call_times: dict[str, list[float]] = defaultdict(list)

    def get_rate_limit_key() -> str:
        """Get rate limit key for current context."""
        if key_func:
            return key_func()

        user = get_current_user()
        if user:
            return f"user:{user.id}"

        # Fall back to correlation ID
        return f"correlation:{get_correlation_id()}"

    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        key = get_rate_limit_key()
        current_time = time()

        # Clean old entries
        call_times[key] = [t for t in call_times[key] if current_time - t < period]

        # Check rate limit
        if len(call_times[key]) >= calls:
            oldest_call = min(call_times[key])
            retry_after = int(period - (current_time - oldest_call))

            from src.core.errors.base import RateLimitError

            msg = f"Rate limit exceeded: {calls} calls per {period}s"
            raise RateLimitError(
                msg,
                retry_after=retry_after,
                details={
                    "limit": calls,
                    "period": period,
                    "key": key,
                },
            )

        # Record this call
        call_times[key].append(current_time)

        return func(*args, **kwargs)

    async def async_handler(
        func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        key = get_rate_limit_key()
        current_time = time()

        # Clean old entries
        call_times[key] = [t for t in call_times[key] if current_time - t < period]

        # Check rate limit
        if len(call_times[key]) >= calls:
            oldest_call = min(call_times[key])
            retry_after = int(period - (current_time - oldest_call))

            from src.core.errors.base import RateLimitError

            msg = f"Rate limit exceeded: {calls} calls per {period}s"
            raise RateLimitError(
                msg,
                retry_after=retry_after,
                details={
                    "limit": calls,
                    "period": period,
                    "key": key,
                },
            )

        # Record this call
        call_times[key].append(current_time)

        return await func(*args, **kwargs)

    return uniform_wrapper(
        f"rate_limit({calls}/{period}s)",
        sync_handler,
        async_handler,
    )
