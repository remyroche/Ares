"""Utilities for managing imports within the training pipeline.

This module centralises common import patterns that previously lived inside
individual training steps.  The audit for the training pipeline highlighted a
couple of recurring problems:

*   Circular import risks because deeply nested modules were importing one
    another without coordination.
*   Fallback import patterns that silently swallowed missing dependencies and
    made it difficult to spot configuration problems.

To address those issues we expose a small collection of helper functions that
wrap :mod:`importlib` and provide deterministic behaviour.  The helpers keep
track of the active import stack so circular dependencies can be surfaced
immediately, and they provide structured error messages with remediation hints.
"""

from __future__ import annotations

import importlib
import logging
from contextlib import contextmanager
from types import ModuleType
from typing import Dict, Iterable, Iterator, MutableMapping, Optional, Tuple

logger = logging.getLogger(__name__)


class ImportCycleError(ImportError):
    """Raised when a circular import is detected while using the helpers."""


class DependencyNotFoundError(ImportError):
    """Raised when a required dependency cannot be imported."""


_import_stack: list[str] = []


@contextmanager
def _track_import(module_name: str) -> Iterator[None]:
    """Context manager used internally to detect circular imports."""

    if module_name in _import_stack:
        cycle = " -> ".join(_import_stack + [module_name])
        raise ImportCycleError(f"Circular import detected: {cycle}")

    _import_stack.append(module_name)
    try:
        yield
    finally:
        _import_stack.pop()


def import_module_safely(
    module_name: str,
    *,
    required: bool = False,
    package_hint: Optional[str] = None,
) -> Optional[ModuleType]:
    """Import a module with optional fast-fail behaviour.

    Args:
        module_name: Fully qualified module name passed to
            :func:`importlib.import_module`.
        required: When ``True`` an :class:`ImportError` will be raised if the
            module cannot be imported.
        package_hint: Optional message that will be appended to the error to
            help users resolve missing dependency issues (for example,
            ``"pip install numpy"``).

    Returns:
        The imported module or ``None`` when the module could not be imported
        and ``required`` is ``False``.
    """

    with _track_import(module_name):
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:  # pragma: no cover - defensive logging
            message = f"Failed to import '{module_name}'"
            if package_hint:
                message = f"{message}. Install with: {package_hint}"

            if required:
                raise DependencyNotFoundError(message) from exc

            logger.warning("%s", message)
            return None


def load_module_attributes(
    module: ModuleType,
    attribute_names: Iterable[str],
    module_name: str,
    *,
    required: bool = True,
) -> Tuple[Dict[str, object], Tuple[str, ...]]:
    """Load a set of attributes from a module in a safe and structured way."""

    attributes: Dict[str, object] = {}
    missing: list[str] = []

    for attr in attribute_names:
        if hasattr(module, attr):
            attributes[attr] = getattr(module, attr)
        else:
            missing.append(attr)

    if missing and required:
        joined = ", ".join(missing)
        raise DependencyNotFoundError(
            f"Module '{module_name}' is missing required attributes: {joined}"
        )

    return attributes, tuple(missing)


def ensure_dependencies(
    availability: MutableMapping[str, bool],
    *,
    error_message: Optional[str] = None,
) -> None:
    """Raise an informative error when required dependencies are missing."""

    missing = [name for name, available in availability.items() if not available]
    if not missing:
        return

    details = ", ".join(missing)
    if error_message is None:
        error_message = f"Missing required dependencies: {details}"
    else:
        error_message = f"{error_message}: {details}"

    raise DependencyNotFoundError(error_message)


__all__ = [
    "DependencyNotFoundError",
    "ImportCycleError",
    "ensure_dependencies",
    "import_module_safely",
    "load_module_attributes",
]

