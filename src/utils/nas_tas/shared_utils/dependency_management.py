"""Centralised optional dependency tracking for NAS/TAS utilities."""

from __future__ import annotations

import importlib
import logging
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

__all__ = [
    "DependencyStatus",
    "DependencyManager",
    "dependency_manager",
]


@dataclass(frozen=True)
class DependencyStatus:
    """Runtime information about a dependency import attempt."""

    name: str
    available: bool
    error: Optional[str]
    checked_at: float
    install_hint: Optional[str] = None
    module: Optional[Any] = None

    def as_dict(self) -> Dict[str, Any]:
        """Serialize the dependency status for diagnostics."""
        data = asdict(self)
        # Modules are not JSON serialisable – drop them from summaries
        data.pop("module", None)
        return data


class DependencyManager:
    """Central registry that standardises optional dependency handling."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._statuses: Dict[str, DependencyStatus] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_status(
        self, name: str, available: bool, module: Optional[Any], error: Optional[BaseException], install_hint: Optional[str]
    ) -> DependencyStatus:
        status = DependencyStatus(
            name=name,
            available=available,
            module=module,
            error=None if error is None else f"{error.__class__.__name__}: {error}",
            checked_at=time.time(),
            install_hint=install_hint,
        )
        with self._lock:
            previous = self._statuses.get(name)
            self._statuses[name] = status
        if not available and (previous is None or previous.available):
            message = f"Optional dependency '{name}' is unavailable"
            if status.error:
                message += f": {status.error}"
            if install_hint:
                message += f". Install hint: {install_hint}"
            self._logger.warning(message)
        return status

    def _attempt_import(self, name: str, package: Optional[str]) -> Any:
        target = package or name
        return importlib.import_module(target)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def import_optional(self, name: str, package: Optional[str] = None, install_hint: Optional[str] = None) -> Optional[Any]:
        """Attempt to import an optional dependency and remember the outcome."""
        target = package or name
        with self._lock:
            status = self._statuses.get(target)
        if status and (status.available or status.error):
            return status.module

        try:
            module = self._attempt_import(name, package)
        except Exception as exc:  # noqa: BLE001 - explicit logging & propagation handled here
            self._record_status(target, False, None, exc, install_hint)
            return None

        self._record_status(target, True, module, None, install_hint)
        return module

    def require(self, name: str, package: Optional[str] = None, install_hint: Optional[str] = None) -> Any:
        """Import a mandatory dependency or raise a helpful runtime error."""
        module = self.import_optional(name, package=package, install_hint=install_hint)
        if module is None:
            hint = f" ({install_hint})" if install_hint else ""
            raise RuntimeError(f"Required dependency '{package or name}' is not available{hint}.")
        return module

    def is_available(self, name: str) -> bool:
        """Return whether the dependency import has succeeded."""
        with self._lock:
            status = self._statuses.get(name)
        return bool(status and status.available)

    def get_status_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return a serialisable snapshot of the dependency statuses."""
        with self._lock:
            items = list(self._statuses.items())
        return {name: status.as_dict() for name, status in items}


dependency_manager = DependencyManager()
