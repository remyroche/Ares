"""Utilities for managing optional dependencies in ml_common."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class OptionalDependencyError(RuntimeError):
    """Raised when an optional dependency is required but unavailable."""

    def __init__(self, package: str, install_hint: Optional[str] = None) -> None:
        message = f"Optional dependency '{package}' is required but not installed."
        if install_hint:
            message = f"{message} Install via: {install_hint}"
        super().__init__(message)
        self.package = package
        self.install_hint = install_hint


@dataclass(frozen=True)
class OptionalDependency:
    """Metadata describing a lazily imported optional dependency."""

    package: str
    import_target: Optional[str] = None
    install_hint: Optional[str] = None
    post_import: Optional[Callable[[object], None]] = None

    def load(self) -> object:
        """Attempt to import the dependency, logging failures explicitly."""
        target = self.import_target or self.package
        logger.debug("Attempting to import optional dependency '%s'", target)
        try:
            module = import_module(target)
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.exception("Failed to import optional dependency '%s'", target, exc_info=exc)
            raise OptionalDependencyError(self.package, self.install_hint) from exc

        if self.post_import:
            try:
                self.post_import(module)
            except Exception as exc:  # pragma: no cover - defensive logging path
                logger.exception("Post-import hook failed for '%s'", target, exc_info=exc)
                raise OptionalDependencyError(self.package, self.install_hint) from exc

        return module


class OptionalDependencyManager:
    """Registry and loader for optional dependencies with explicit error reporting."""

    def __init__(self) -> None:
        self._registry: Dict[str, OptionalDependency] = {}

    def register(self, dependency: OptionalDependency) -> None:
        if dependency.package in self._registry:
            logger.debug("Optional dependency '%s' already registered", dependency.package)
            return
        logger.debug("Registering optional dependency '%s'", dependency.package)
        self._registry[dependency.package] = dependency

    def is_available(self, package: str) -> bool:
        dependency = self._registry.get(package)
        if dependency is None:
            logger.debug("Optional dependency '%s' not registered", package)
            return False
        try:
            dependency.load()
        except OptionalDependencyError:
            return False
        return True

    def require(self, package: str) -> object:
        dependency = self._registry.get(package)
        if dependency is None:
            raise OptionalDependencyError(package)
        return dependency.load()


# Pre-register the standard optional ML packages used across the library.
manager = OptionalDependencyManager()
manager.register(
    OptionalDependency(
        package="optuna",
        install_hint="pip install optuna",
    )
)
manager.register(
    OptionalDependency(
        package="scikit-learn",
        import_target="sklearn",
        install_hint="pip install scikit-learn",
    )
)
manager.register(
    OptionalDependency(
        package="torch",
        install_hint="pip install torch",
    )
)

__all__ = [
    "OptionalDependencyError",
    "OptionalDependency",
    "OptionalDependencyManager",
    "manager",
]
