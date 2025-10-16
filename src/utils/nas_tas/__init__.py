"""Neural Architecture Search and Trading Architecture Search (NAS/TAS).

This module provides comprehensive neural architecture search and trading
architecture search capabilities with extensive integration of utility modules
for optimal performance.

Key Features:
- Neural Architecture Search (NAS) for optimal model architectures
- Trading Architecture Search (TAS) for optimal trading strategies
- Extensive integration with common utilities for data processing
- M1 hardware optimization support
- Comprehensive logging and monitoring
- Advanced optimization algorithms (Grid Search + Bayesian TPE)
- Matrix operations for high-performance computations

Only the core engine exports are re-exported from ``src.utils.nas_tas`` because
the helper modules referenced by earlier versions of this file are not present
in the repository.  Importing modules that do not exist would cause
``ImportError`` during ``from src.utils import nas_tas``.  To keep the package
importable we load the core engines lazily when they are first accessed.
Additional utilities can be re-exported here once their implementations land in
the codebase.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict

__version__ = "1.0.0"
__author__ = "NAS/TAS Development Team"

__all__ = [
    "NASEngine",
    "TASEngine",
]

_EXPORT_MAP: Dict[str, str] = {
    "NASEngine": "src.utils.nas_tas.core.nas_engine",
    "TASEngine": "src.utils.nas_tas.core.tas_engine",
}

def __getattr__(name: str) -> Any:  # pragma: no cover - trivial wrapper
    """Lazily import NAS/TAS components on first attribute access."""
    if name in _EXPORT_MAP:
        module = import_module(_EXPORT_MAP[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
