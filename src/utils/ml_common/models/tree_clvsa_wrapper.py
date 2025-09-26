"""Lightweight Tree CLVSA compatibility wrapper.

The original implementation exposed a rich attention-based wrapper for tree
models. The current code base consolidates the advanced implementation under
the CVLSA package, but several modules still import the historical
``tree_clvsa_wrapper`` module. This shim provides a minimal, dependency-free
wrapper that preserves the public API expected by those modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TreeCLVSAConfig:
    """Configuration container for the compatibility wrapper."""

    attention_dim: int = 64
    use_temporal_attention: bool = True
    regime_aware: bool = True
    attention_dropout: float = 0
    feature_selection_method: str = "mutual_info"
    temporal_window_size: int = 20
    ensemble_attention: bool = True
    memory_efficient: bool = True


class TreeCLVSAWrapper:
    """Thin wrapper that simply delegates to the underlying estimator."""

    def __init__(self, base_model: Any, config: TreeCLVSAConfig) -> None:
        self.base_model = base_model
        self.config = config

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model.fit(*args, **kwargs)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model.predict(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - simple proxy
        return getattr(self.base_model, item)


def create_tree_clvsa_config(**overrides: Any) -> TreeCLVSAConfig:
    """Create a ``TreeCLVSAConfig`` with optional overrides."""

    return TreeCLVSAConfig(**overrides)


def create_tree_clvsa_wrapper(base_model: Any, config: TreeCLVSAConfig) -> TreeCLVSAWrapper:
    """Return a delegating wrapper that mirrors the original API."""

    return TreeCLVSAWrapper(base_model, config)


__all__ = [
    "TreeCLVSAConfig",
    "TreeCLVSAWrapper",
    "create_tree_clvsa_config",
    "create_tree_clvsa_wrapper",
]
