"""Compatibility shim for enhanced variable selection utilities."""

from __future__ import annotations

from importlib import import_module

_MODULE_NAME = "src.utils.ml_common.cvlsa.enhanced_variable_selection"
_module = import_module(_MODULE_NAME)

__all__ = getattr(
    _module,
    "__all__",
    [name for name in dir(_module) if not name.startswith("_")],
)

globals().update({name: getattr(_module, name) for name in __all__})
