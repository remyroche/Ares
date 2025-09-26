"""Compatibility wrapper for the CVLSA architecture module.

This module preserves the legacy import path
``src.utils.ml_common.models.cvlsa_architecture`` while delegating all
functionality to the reorganised CVLSA package.
"""

from __future__ import annotations

from importlib import import_module

_MODULE_NAME = "src.utils.ml_common.cvlsa.cvlsa_architecture"
_module = import_module(_MODULE_NAME)

__all__ = getattr(
    _module,
    "__all__",
    [name for name in dir(_module) if not name.startswith("_")],
)

globals().update({name: getattr(_module, name) for name in __all__})
