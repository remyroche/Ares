import types
import sys

if "cvxpy" not in sys.modules:
    DummyVariable = type("Variable", (), {})
    DummyConstraint = type("Constraint", (), {})

    class DummyProblem:
        def __init__(self, *args, **kwargs):
            self.status = None

        def solve(self, *args, **kwargs):
            self.status = "optimal"

    sys.modules["cvxpy"] = types.SimpleNamespace(
        Variable=DummyVariable,
        Constraint=DummyConstraint,
        Maximize=lambda *args, **kwargs: None,
        sum=lambda x: x,
        Problem=DummyProblem,
        CBC=None,
        OPTIMAL="optimal",
    )

import numpy as np
import pandas as pd
import types
import sys

if "pymc" not in sys.modules:
    pymc_stub = types.ModuleType("pymc")

    class _DummyModel:
        pass

    pymc_stub.Model = _DummyModel
    pymc_stub.sample = lambda *args, **kwargs: None

    sys.modules["pymc"] = pymc_stub

if "aesara" not in sys.modules:
    aesara_stub = types.ModuleType("aesara")
    tensor_stub = types.ModuleType("aesara.tensor")

    def _dummy_attr(*args, **kwargs):
        return None

    tensor_stub.__getattr__ = lambda name: _dummy_attr

    sys.modules["aesara"] = aesara_stub
    sys.modules["aesara.tensor"] = tensor_stub

if "arviz" not in sys.modules:
    sys.modules["arviz"] = types.ModuleType("arviz")

from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.phase1_probe import (
    Phase1HTFProbe,
)
from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.scoring_system import (
    AdaptiveScoringSystem,
)
from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.ehu_rih_assignment import (
    EHU_RIH_Assignment,
)
from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.pipeline import (
    PipelineConfig,
)


RNG = np.random.default_rng(42)


def _make_series(length: int, base_freq: int) -> pd.Series:
    index = pd.date_range(
        "2024-01-01",
        periods=length,
        freq=f"{base_freq}min",
    )
    return pd.Series(RNG.standard_normal(length), index=index)


def test_phase1_assignment_share_staleness_curves():
    config = PipelineConfig()
    scoring_system = AdaptiveScoringSystem(config.scoring, config.session)
    phase1_probe = Phase1HTFProbe(
        config.probe,
        config.session,
        scoring_system=scoring_system,
    )
    assignment = EHU_RIH_Assignment(config.assignment)

    test_cases = [
        ("p/price_ema10_pct", "trend_level_vol", 60),
        ("p/rsi14", "oscillators", 90),
        ("p/vwap_session_dist", "anchors", 45),
    ]

    base_freq = config.session.base_timeframe_minutes
    target = _make_series(240, base_freq)

    for base_feature, family, lookback in test_cases:
        feature = _make_series(240, base_freq)

        candidates = phase1_probe._score_candidate(
            htf_feature=feature,
            base_feature=base_feature,
            lookback=lookback,
            family=family,
            regime_segments={},
            targets=target,
        )

        assert candidates, "Expected Phase-1 to return at least one candidate"

        candidate = candidates[0]

        summary_phase1 = candidate.metadata["staleness_summary"]
        summary_assignment = assignment.staleness_calculator.get_summary(
            feature_name=base_feature,
            family=family,
            lookback=lookback,
            base_timeframe=base_freq,
        )

        assert summary_phase1 == summary_assignment
        assert np.isclose(candidate.staleness, summary_phase1.at_base)
