"""Pluggable search strategy abstractions for NAS/TAS systems."""
from __future__ import annotations

import json
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class Candidate:
    """A sampled candidate configuration."""

    params: Dict[str, Any]
    context: Any = None

    def cache_key(self) -> tuple:
        """Return a hashable representation for caching."""
        return tuple(sorted(self.params.items()))


@dataclass
class Evaluation:
    """Result of evaluating a candidate."""

    candidate: Candidate
    metrics: Dict[str, float]

    @property
    def score(self) -> float:
        """Primary score used for single-objective optimisation."""
        if not self.metrics:
            return float("nan")
        # Prefer explicit "score" key, fallback to first metric
        if "score" in self.metrics:
            return self.metrics["score"]
        return next(iter(self.metrics.values()))


@dataclass
class SearchState:
    """Mutable state shared between the search engine and the strategy."""

    iteration: int = 0
    best_evaluation: Optional[Evaluation] = None
    history: List[Evaluation] = field(default_factory=list)
    pareto_front: List[Evaluation] = field(default_factory=list)
    terminated: bool = False

    def register_evaluations(self, evaluations: Sequence[Evaluation]) -> None:
        """Update best evaluation and history with new results."""
        if not evaluations:
            return
        self.history.extend(evaluations)
        best_candidate = max(
            evaluations,
            key=lambda item: item.score,
        )
        if (
            self.best_evaluation is None
            or best_candidate.score > self.best_evaluation.score
        ):
            self.best_evaluation = best_candidate


class SearchStrategy(ABC):
    """Base class for pluggable search strategies."""

    name: str = "base"

    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed
        self.logger = logging.getLogger(self.__class__.__name__)
        self._rng = np.random.default_rng(random_seed)
        if random_seed is not None:
            random.seed(random_seed)

    def initialize(
        self,
        search_space: Dict[str, Any],
        objective: Callable[[Dict[str, Any]], Dict[str, float]],
        state: SearchState,
        config: Dict[str, Any],
    ) -> None:
        """Prepare the strategy for a new search run."""
        self.search_space = search_space
        self.objective = objective
        self.state = state
        self.config = config
        self.logger.debug(
            "Initialised strategy", extra={"strategy": self.name, "config": config}
        )

    @abstractmethod
    def sample_candidates(
        self, state: SearchState, n_candidates: int
    ) -> List[Candidate]:
        """Sample new candidates to evaluate."""

    @abstractmethod
    def update_state(
        self, state: SearchState, evaluations: Sequence[Evaluation]
    ) -> None:
        """Update internal state after evaluations have been completed."""

    def should_continue(self, state: SearchState) -> bool:
        """Determine whether the outer loop should continue sampling."""
        if state.terminated:
            return False
        max_iterations = self.config.get("max_iterations")
        if max_iterations is not None and state.iteration >= max_iterations:
            return False
        return True

    def finalize(self, state: SearchState) -> Dict[str, Any]:
        """Return a serialisable result summary."""
        best = state.best_evaluation
        return {
            "best_params": best.candidate.params if best else {},
            "best_metrics": best.metrics if best else {},
            "history": [
                {"params": ev.candidate.params, "metrics": ev.metrics}
                for ev in state.history
            ],
            "pareto_front": [
                {"params": ev.candidate.params, "metrics": ev.metrics}
                for ev in state.pareto_front
            ],
        }

    # Utility helpers shared by concrete strategies ---------------------------------
    def _sample_categorical(self, values: Sequence[Any]) -> Any:
        idx = self._rng.integers(0, len(values))
        return values[idx]

    def _sample_uniform(self, low: float, high: float, is_int: bool = False) -> Any:
        if is_int:
            return int(self._rng.integers(low, high + 1))
        return float(self._rng.uniform(low, high))

    def _iter_grid(self, param_values: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
        """Cartesian product generator used by the grid strategy."""
        keys = list(param_values.keys())
        if not keys:
            yield {}
            return

        def _product(index: int, current: Dict[str, Any]):
            if index == len(keys):
                yield dict(current)
                return
            key = keys[index]
            for value in param_values[key]:
                current[key] = value
                yield from _product(index + 1, current)
            current.pop(key, None)

        yield from _product(0, {})

    def dumps_state(self) -> str:
        """Serialise the strategy state for debugging or dashboards."""
        payload = {
            "strategy": self.name,
            "iteration": self.state.iteration if hasattr(self, "state") else None,
            "best_score": (
                self.state.best_evaluation.score
                if hasattr(self, "state") and self.state.best_evaluation
                else None
            ),
            "history_length": len(self.state.history)
            if hasattr(self, "state")
            else 0,
        }
        return json.dumps(payload)


class StrategyRegistry:
    """Registry for strategy plugins."""

    def __init__(self):
        self._strategies: Dict[str, Callable[..., SearchStrategy]] = {}

    def register(self, name: str, factory: Callable[..., SearchStrategy]) -> None:
        self._strategies[name] = factory

    def create(
        self, name: str, *, random_seed: Optional[int] = None, **kwargs: Any
    ) -> SearchStrategy:
        if name not in self._strategies:
            raise KeyError(f"Strategy '{name}' is not registered")
        return self._strategies[name](random_seed=random_seed, **kwargs)

    def available(self) -> List[str]:
        return sorted(self._strategies)
