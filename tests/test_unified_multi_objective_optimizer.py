import sys
import types
from pathlib import Path

if "numpy" not in sys.modules:
    np_stub = types.ModuleType("numpy")

    def _ensure_size(value, size):
        if size is None:
            return value
        if isinstance(size, int):
            return [value for _ in range(size)]
        if isinstance(size, tuple) and len(size) == 2:
            return [[value for _ in range(size[1])] for _ in range(size[0])]
        return value

    class _RandomNamespace:
        @staticmethod
        def uniform(low, high=None, size=None):
            if high is None:
                high = 1.0
            return _ensure_size((low + high) / 2.0, size)

        @staticmethod
        def normal(loc=0.0, scale=1.0, size=None):
            return _ensure_size(loc, size)

        @staticmethod
        def random(size=None):
            return _ensure_size(0.5, size)

        @staticmethod
        def choice(seq):
            return seq[0]

    np_stub.random = _RandomNamespace()

    def _clip(value, min_value, max_value):
        return max(min_value, min(value, max_value))

    def _argsort(sequence):
        return sorted(range(len(sequence)), key=sequence.__getitem__)

    np_stub.clip = _clip
    np_stub.argsort = _argsort
    np_stub.mean = lambda seq: sum(seq) / len(seq) if seq else 0.0
    np_stub.std = lambda seq: 0.0
    np_stub.asarray = lambda seq: list(seq)
    np_stub.array = lambda seq: list(seq)
    np_stub.inf = float("inf")
    np_stub.nan = float("nan")

    sys.modules["numpy"] = np_stub

if "pandas" not in sys.modules:
    pd_stub = types.ModuleType("pandas")

    class _Frame(dict):
        def __init__(self, data=None, **kwargs):
            super().__init__(data or {})

        def to_dict(self, *args, **kwargs):
            return dict(self)

    pd_stub.DataFrame = _Frame
    pd_stub.Series = _Frame
    pd_stub.Timestamp = lambda *args, **kwargs: args or kwargs
    pd_stub.Timedelta = lambda *args, **kwargs: args or kwargs
    pd_stub.isna = lambda value: value is None
    pd_stub.concat = lambda frames, axis=0, ignore_index=False: _Frame()
    sys.modules["pandas"] = pd_stub

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.utils.nas_tas.unified_multi_objective_optimizer import (
    ObjectiveType,
    ParetoSolution,
    UnifiedMultiObjectiveConfig,
    UnifiedMultiObjectiveOptimizer,
)


def _make_optimizer():
    config = UnifiedMultiObjectiveConfig(
        objectives=[
            ObjectiveType.SHARPE_RATIO,
            ObjectiveType.DOWNSIDE_DEVIATION,
            ObjectiveType.EXECUTION_LATENCY,
        ],
        objective_weights=[0.5, 0.25, 0.25],
        objective_directions=["maximize", "minimize", "minimize"],
        max_iterations=2,
        population_size=1,
    )
    return UnifiedMultiObjectiveOptimizer(config)


def test_dominance_respects_minimization_direction():
    optimizer = _make_optimizer()
    dominating = ParetoSolution(
        parameters={"depth": 4},
        objectives={
            ObjectiveType.SHARPE_RATIO: 1.0,
            ObjectiveType.DOWNSIDE_DEVIATION: 0.1,
            ObjectiveType.EXECUTION_LATENCY: 4.0,
        },
    )
    dominated = ParetoSolution(
        parameters={"depth": 6},
        objectives={
            ObjectiveType.SHARPE_RATIO: 0.8,
            ObjectiveType.DOWNSIDE_DEVIATION: 0.2,
            ObjectiveType.EXECUTION_LATENCY: 5.0,
        },
    )

    assert optimizer._dominates(dominating, dominated) is True
    assert optimizer._dominates(dominated, dominating) is False


def test_environmental_selection_uses_legacy_nsga(monkeypatch):
    optimizer = _make_optimizer()
    selected_population = {}

    class DummyLegacy:  # pragma: no cover - exercised via monkeypatch
        def __init__(self, objectives, population_size):
            selected_population['args'] = (objectives, population_size)
            self.population_size = population_size

        def optimize(self, population):
            selected_population['population'] = population
            # emulate legacy behaviour by returning the top-k candidates
            return population[: self.population_size]

    monkeypatch.setattr(
        'src.utils.nas_tas.unified_multi_objective_optimizer._LEGACY_NAS_NSGA',
        DummyLegacy,
        raising=False,
    )

    candidate_fast = ParetoSolution(
        parameters={'width': 8},
        objectives={
            ObjectiveType.SHARPE_RATIO: 0.8,
            ObjectiveType.DOWNSIDE_DEVIATION: 0.1,
            ObjectiveType.EXECUTION_LATENCY: 2.0,
        },
    )
    candidate_slow = ParetoSolution(
        parameters={'width': 64},
        objectives={
            ObjectiveType.SHARPE_RATIO: 1.0,
            ObjectiveType.DOWNSIDE_DEVIATION: 0.4,
            ObjectiveType.EXECUTION_LATENCY: 12.0,
        },
    )

    selected = optimizer._environmental_selection([[candidate_fast, candidate_slow]])

    assert selected_population['args'][0] == [obj.value for obj in optimizer.config.objectives]
    assert len(selected_population['population']) == 2
    assert selected_population['population'][0].solution is candidate_fast
    assert selected == [candidate_fast]


def test_weighted_solution_penalizes_minimize_objectives():
    optimizer = _make_optimizer()

    latency_heavy = ParetoSolution(
        parameters={"width": 32},
        objectives={
            ObjectiveType.SHARPE_RATIO: 1.0,
            ObjectiveType.DOWNSIDE_DEVIATION: 0.5,
            ObjectiveType.EXECUTION_LATENCY: 10.0,
        },
    )
    latency_light = ParetoSolution(
        parameters={"width": 16},
        objectives={
            ObjectiveType.SHARPE_RATIO: 0.9,
            ObjectiveType.DOWNSIDE_DEVIATION: 0.1,
            ObjectiveType.EXECUTION_LATENCY: 5.0,
        },
    )

    best = optimizer._find_best_weighted_solution([latency_heavy, latency_light])
    assert best is latency_light
