"""Random and grid search implementations built on the strategy interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .base import Candidate, Evaluation, SearchState, SearchStrategy


@dataclass
class _GridIterator:
    """Simple iterator that yields cartesian product of parameter choices."""

    grid: Dict[str, Iterable[Any]]
    _generator: Optional[Iterable[Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        from itertools import product

        keys = list(self.grid)
        if not keys:
            self._generator = iter([{}])
            return
        values = [list(self.grid[key]) for key in keys]
        self._generator = (
            {key: combination[index] for index, key in enumerate(keys)}
            for combination in product(*values)
        )

    def next_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        assert self._generator is not None
        batch: List[Dict[str, Any]] = []
        try:
            for _ in range(batch_size):
                batch.append(next(self._generator))
        except StopIteration:
            pass
        return batch


class RandomSearchStrategy(SearchStrategy):
    """Simple random search with reproducibility support."""

    name = "random"

    def sample_candidates(
        self, state: SearchState, n_candidates: int
    ) -> List[Candidate]:
        candidates: List[Candidate] = []
        for _ in range(n_candidates):
            params: Dict[str, Any] = {}
            for param_name, spec in self.search_space.items():
                if isinstance(spec, dict):
                    param_type = spec.get("type", "float")
                    if param_type == "int":
                        params[param_name] = self._sample_uniform(
                            spec.get("low", 0), spec.get("high", 10), is_int=True
                        )
                    elif param_type == "float":
                        params[param_name] = self._sample_uniform(
                            spec.get("low", 0.0), spec.get("high", 1.0)
                        )
                    else:
                        params[param_name] = self._sample_categorical(
                            spec.get("choices", [])
                        )
                elif isinstance(spec, list):
                    params[param_name] = self._sample_categorical(spec)
                else:
                    params[param_name] = spec
            candidates.append(Candidate(params=params))
        return candidates

    def update_state(
        self, state: SearchState, evaluations: List[Evaluation]
    ) -> None:
        state.register_evaluations(evaluations)
        state.iteration += 1
        if self.config.get("max_iterations") is not None and state.iteration >= self.config.get(
            "max_iterations"
        ):
            state.terminated = True


class GridSearchStrategy(SearchStrategy):
    """Deterministic grid search strategy."""

    name = "grid"

    def initialize(
        self,
        search_space: Dict[str, Any],
        objective,
        state: SearchState,
        config: Dict[str, Any],
    ) -> None:  # type: ignore[override]
        super().initialize(search_space, objective, state, config)
        grid: Dict[str, Iterable[Any]] = {}
        for param_name, spec in search_space.items():
            if isinstance(spec, dict):
                if spec.get("type") == "int":
                    grid[param_name] = range(spec.get("low", 0), spec.get("high", 1) + 1)
                elif spec.get("type") == "float":
                    step = spec.get("step", 0.1)
                    low = spec.get("low", 0.0)
                    high = spec.get("high", 1.0)
                    grid[param_name] = [low + step * i for i in range(int((high - low) / step) + 1)]
                else:
                    grid[param_name] = spec.get("choices", [])
            else:
                grid[param_name] = spec
        self._iterator = _GridIterator(grid)

    def sample_candidates(
        self, state: SearchState, n_candidates: int
    ) -> List[Candidate]:
        batch = self._iterator.next_batch(n_candidates)
        return [Candidate(params=item) for item in batch]

    def update_state(
        self, state: SearchState, evaluations: List[Evaluation]
    ) -> None:
        state.register_evaluations(evaluations)
        state.iteration += 1
        if not evaluations:
            state.terminated = True

