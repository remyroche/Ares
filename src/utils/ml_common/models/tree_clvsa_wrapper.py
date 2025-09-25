"""Tree-based CLVSA wrapper utilities.

This module provides a thin interoperability layer between the lightweight
CLVSA attention implementation that lives in
``src/training/steps/model_training/clvsa_attention_wrapper.py`` and the model
factory components that expect a reusable helper for wrapping tree models.

Historically a number of modules imported
``src.utils.ml_common.models.tree_clvsa_wrapper`` but the file did not exist in
the repository, forcing every caller to re-implement the same adapter logic.
The new module centralises that behaviour and ensures that *all* tree based
models can be wrapped in the CLVSA attention mechanism before being handed to
the rest of the NAS/TAS pipeline.

The wrapper is intentionally lightweight: when the CLVSA attention utilities
are unavailable we fall back to the raw base model to guarantee compatibility
in constrained environments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)

try:  # Import the lightweight CLVSA attention wrapper when available
    from src.training.steps.model_training.clvsa_attention_wrapper import (  # type: ignore
        CLVSAAttentionWrapper,
    )
    CLVSA_WRAPPER_AVAILABLE = True
except Exception as exc:  # pragma: no cover - optional dependency
    logger.warning("⚠️ CLVSA attention wrapper not available: %s", exc)
    CLVSA_WRAPPER_AVAILABLE = False
    CLVSAAttentionWrapper = None  # type: ignore[assignment]


@dataclass
class TreeCLVSAConfig:
    """Configuration for wrapping tree models with CLVSA attention."""

    attention_dim: int = 64
    use_temporal_attention: bool = True
    regime_aware: bool = True
    attention_dropout: float = 0.1
    temporal_window_size: int = 20
    feature_selection_method: str = "mutual_info"
    ensemble_attention: bool = True
    memory_efficient: bool = True


def create_tree_clvsa_config(**overrides: Any) -> TreeCLVSAConfig:
    """Create a :class:`TreeCLVSAConfig` instance with optional overrides."""

    config = TreeCLVSAConfig()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


class TreeCLVSAWrapper:
    """Wrap a tree model with the CLVSA attention pre-processing pipeline."""

    def __init__(self, base_model: Any, config: TreeCLVSAConfig):
        self.base_model = base_model
        self.config = config
        self._wrapper: Optional[CLVSAAttentionWrapper] = None

    def _ensure_wrapper(self) -> None:
        if self._wrapper is None:
            if not CLVSA_WRAPPER_AVAILABLE:
                raise RuntimeError("CLVSA attention wrapper is not available")

            self._wrapper = CLVSAAttentionWrapper(
                base_model=self.base_model,
                attention_dim=self.config.attention_dim,
                use_temporal_attention=self.config.use_temporal_attention,
                regime_aware=self.config.regime_aware,
                attention_dropout=self.config.attention_dropout,
            )

    # scikit-learn compatibility -------------------------------------------------
    def fit(self, X, y, sample_weight=None, regimes=None):
        if not CLVSA_WRAPPER_AVAILABLE:
            logger.debug("CLVSA wrapper unavailable, using raw base model")
            if sample_weight is not None:
                self.base_model.fit(X, y, sample_weight=sample_weight)
            else:
                self.base_model.fit(X, y)
            return self

        self._ensure_wrapper()
        assert self._wrapper is not None  # For type checking
        self._wrapper.fit(X, y, sample_weight=sample_weight, regimes=regimes)
        return self

    def predict(self, X):
        if not CLVSA_WRAPPER_AVAILABLE:
            return self.base_model.predict(X)

        self._ensure_wrapper()
        return self._wrapper.predict(X)  # type: ignore[return-value]

    def predict_proba(self, X):  # pragma: no cover - delegated call
        if not hasattr(self.base_model, "predict_proba"):
            raise AttributeError("Base model does not implement predict_proba")

        if not CLVSA_WRAPPER_AVAILABLE:
            return self.base_model.predict_proba(X)

        self._ensure_wrapper()
        return self._wrapper.base_model.predict_proba(X)  # type: ignore[attr-defined]

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = {"base_model": self.base_model, "config": self.config}
        if hasattr(self.base_model, "get_params") and deep:
            params.update({f"base__{k}": v for k, v in self.base_model.get_params().items()})
        return params

    def set_params(self, **params: Any) -> "TreeCLVSAWrapper":  # pragma: no cover - simple setter
        base_params = {k[len("base__"):]: v for k, v in params.items() if k.startswith("base__")}
        if base_params and hasattr(self.base_model, "set_params"):
            self.base_model.set_params(**base_params)

        if "config" in params and isinstance(params["config"], TreeCLVSAConfig):
            self.config = params["config"]

        return self


def create_tree_clvsa_wrapper(base_model: Any, config: TreeCLVSAConfig) -> Any:
    """Attempt to wrap ``base_model`` with the CLVSA attention wrapper."""

    if not CLVSA_WRAPPER_AVAILABLE:
        logger.warning("⚠️ CLVSA wrapper unavailable, returning base model unmodified")
        return base_model

    try:
        return TreeCLVSAWrapper(base_model, config)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("⚠️ Failed to create CLVSA wrapper (%s); using base model", exc)
        return base_model
