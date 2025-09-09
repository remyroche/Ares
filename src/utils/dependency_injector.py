from __future__ import annotations
"""Dependency injection façade for step modules.

This module provides a stable public import path for the UtilityDependencyInjector that
lives inside the (currently very large) step03_5 implementation.  Downstream code should
import the injector from here instead of reaching deep into step files.

Eventually the full implementation will be moved here.  For now we re-export the
existing class to avoid large immediate code moves while still giving us a single point
of reference.
"""

try:
    from src.training.steps.market_analysis.hmm_clustering.step03_5_final_regime_clustering import (
        UtilityDependencyInjector as _OldUtilityDependencyInjector,
    )
except ImportError as exc:  # pragma: no cover – should only fail in specialised build envs
    raise ImportError(
        "Could not import UtilityDependencyInjector from step03_5 module. "
        "Make sure that module is importable before using dependency_injector."
    ) from exc


class UtilityDependencyInjector(_OldUtilityDependencyInjector):
    """Temporary subclass façade.

    Inherits all behaviour from the original implementation.  Placed here so we can
    start importing `src.utils.dependency_injector.UtilityDependencyInjector`.
    Future refactors will move the full implementation into this module and remove the
    subclassing indirection.
    """

    pass