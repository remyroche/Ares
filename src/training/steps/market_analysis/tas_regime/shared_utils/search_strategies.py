"""TAS search strategy wrappers that rely on the unified NAS/TAS utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

from src.utils.nas_tas.shared_utils.search_strategies import (
    OptimizationResult,
    SearchStrategyConfig,
    SearchStrategyManager,
    create_search_strategy_manager,
)

logger = logging.getLogger(__name__)


class SearchStrategyType(Enum):
    """High level search modes for TAS callers."""

    AUTO = "auto"
    GRID = "grid"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"


class OptimizationObjective(Enum):
    """Placeholder objective enum retained for backwards compatibility."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class SearchConfig:
    """Minimal wrapper around :class:`SearchStrategyConfig`."""

    strategy_type: SearchStrategyType = SearchStrategyType.AUTO
    objective: OptimizationObjective = OptimizationObjective.MAXIMIZE
    strategy: SearchStrategyConfig = field(default_factory=SearchStrategyConfig)

    def as_shared_config(self) -> SearchStrategyConfig:
        """Return a shared NAS/TAS configuration instance."""

        if isinstance(self.strategy, SearchStrategyConfig):
            return self.strategy
        return SearchStrategyConfig(**(self.strategy or {}))


SearchResult = OptimizationResult


class SearchStrategies:
    """TAS-facing facade that delegates to :class:`SearchStrategyManager`."""

    def __init__(self, config: Optional[SearchConfig] = None) -> None:
        self.logger = logger.getChild("SearchStrategies")
        self.config = config or SearchConfig()
        self.manager: SearchStrategyManager = create_search_strategy_manager(self.config.as_shared_config())
        self.logger.info("✅ TAS SearchStrategies initialized using shared NAS/TAS manager")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def grid_search(self, objective: Callable[[Dict[str, Any]], float], search_space: Dict[str, Any]) -> SearchResult:
        return self.manager.optimize_with_strategy(objective, search_space, strategy="grid")

    def bayesian_optimization(self, objective: Callable[[Dict[str, Any]], float], search_space: Dict[str, Any]) -> SearchResult:
        return self.manager.optimize_with_strategy(objective, search_space, strategy="bayesian")

    def random_search(self, objective: Callable[[Dict[str, Any]], float], search_space: Dict[str, Any]) -> SearchResult:
        # The shared manager uses the evolutionary optimiser as the stochastic fallback.
        return self.manager.optimize_with_strategy(objective, search_space, strategy="evolutionary")

    def hybrid_search(self, objective: Callable[[Dict[str, Any]], float], search_space: Dict[str, Any]) -> SearchResult:
        return self.manager.optimize_with_strategy(objective, search_space, strategy="auto")

    def compare_strategies(self, objective: Callable[[Dict[str, Any]], float], search_space: Dict[str, Any]) -> Dict[str, SearchResult]:
        return self.manager.compare_strategies(objective, search_space)

    def cleanup(self) -> None:
        self.logger.debug("Cleanup invoked - no resources to release")

    def __enter__(self) -> "SearchStrategies":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.cleanup()


def create_search_strategies(config: Optional[SearchConfig] = None) -> SearchStrategies:
    """Factory helper retained for existing TAS workflows."""

    return SearchStrategies(config)


def grid_search(
    objective: Callable[[Dict[str, Any]], float],
    search_space: Dict[str, Any],
    config: Optional[SearchConfig] = None,
) -> SearchResult:
    with SearchStrategies(config) as strategies:
        return strategies.grid_search(objective, search_space)


def bayesian_optimization(
    objective: Callable[[Dict[str, Any]], float],
    search_space: Dict[str, Any],
    config: Optional[SearchConfig] = None,
) -> SearchResult:
    with SearchStrategies(config) as strategies:
        return strategies.bayesian_optimization(objective, search_space)


def random_search(
    objective: Callable[[Dict[str, Any]], float],
    search_space: Dict[str, Any],
    config: Optional[SearchConfig] = None,
) -> SearchResult:
    with SearchStrategies(config) as strategies:
        return strategies.random_search(objective, search_space)


def hybrid_search(
    objective: Callable[[Dict[str, Any]], float],
    search_space: Dict[str, Any],
    config: Optional[SearchConfig] = None,
) -> SearchResult:
    with SearchStrategies(config) as strategies:
        return strategies.hybrid_search(objective, search_space)


__all__ = [
    "SearchStrategies",
    "SearchConfig",
    "SearchResult",
    "SearchStrategyType",
    "OptimizationObjective",
    "create_search_strategies",
    "grid_search",
    "bayesian_optimization",
    "random_search",
    "hybrid_search",
]
