"""Compatibility shim for the legacy LightGBM-based FeatureSelector.

Some training steps import ``FeatureSelector`` from ``src.feature_selection`` while the
implementation actually lives in the repository root under the ``feature_selection``
package. When the interpreter is launched with ``sys.path`` rooted inside ``src`` the
legacy package is not visible, which causes ``ImportError`` warnings during registry
initialisation. This shim ensures the root package is discoverable and re-exports
``FeatureSelector`` for downstream code.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

__all__ = ["FeatureSelector"]


def _load_legacy_feature_selector_module():
    """Load the legacy feature_selection module ensuring the repo root is on sys.path."""
    module_name = "feature_selection.feature_selection_with_lgbm"
    repo_root = Path(__file__).resolve().parents[2]
    legacy_pkg = repo_root / "feature_selection"

    if legacy_pkg.exists():
        root_str = str(repo_root)
        if root_str not in sys.path:
            # Prepend so the legacy package is preferred over any installed package
            sys.path.insert(0, root_str)

    return importlib.import_module(module_name)


try:
    _legacy_module = _load_legacy_feature_selector_module()
    FeatureSelector = getattr(_legacy_module, "FeatureSelector")
except Exception as exc:  # pragma: no cover - only triggered when module truly missing
    raise ImportError(
        "FeatureSelector shim could not import legacy feature_selection implementation"
    ) from exc
