"""Central registry for optional market analysis dependencies.

The audit highlighted ad-hoc optional import patterns scattered across the
``market_analysis`` package. This module consolidates the logic so components can
query the availability of heavy or research-only dependencies without duplicating
boilerplate or silently swallowing import errors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class OptionalDependency:
    """Metadata describing the availability of an optional dependency."""

    name: str
    available: bool
    error: Optional[Exception] = None
    message: Optional[str] = None

    def require(self) -> None:
        """Raise a descriptive error if the dependency is unavailable."""

        if self.available:
            return
        detail = self.message or "Optional dependency is not installed"
        if self.error:
            detail = f"{detail}: {self.error}"
        raise ImportError(detail)


_registry: Dict[str, OptionalDependency] = {}


def register_optional_dependency(
    name: str,
    available: bool,
    *,
    error: Optional[Exception] = None,
    message: Optional[str] = None,
) -> OptionalDependency:
    """Register the availability of an optional dependency.

    Args:
        name: Human readable dependency identifier.
        available: ``True`` when the dependency import succeeded.
        error: Optional original ``ImportError`` for diagnostic context.
        message: Additional context that should be surfaced to callers.

    Returns:
        The :class:`OptionalDependency` instance stored in the registry so the
        caller can reuse it immediately.
    """

    dependency = OptionalDependency(
        name=name,
        available=available,
        error=error,
        message=message,
    )
    _registry[name] = dependency
    return dependency


def get_optional_dependency(name: str) -> OptionalDependency:
    """Retrieve dependency metadata previously registered with the registry."""

    return _registry.get(name, OptionalDependency(name=name, available=False))
