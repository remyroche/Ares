import sys
from types import SimpleNamespace

import pytest

# Provide a lightweight cvxpy stub so cross_timeframe_generation modules can be imported
sys.modules.setdefault(
    "cvxpy",
    SimpleNamespace(
        Variable=object,
        Parameter=object,
        Problem=object,
        Minimize=object,
        Constraint=object,
        sum=lambda *args, **kwargs: None,
    ),
)

from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.phase1_probe import (
    Phase1HTFProbe,
)
from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.scoring_system import (
    AdaptiveScoringSystem,
)


@pytest.fixture
def phase1_with_scoring():
    config = SimpleNamespace(
        coarse_grid_min=15,
        coarse_grid_max=298,
        meta_learning_range=0.05,
        base_timeframe_minutes=5,
    )
    scoring_system = AdaptiveScoringSystem(config)
    probe = Phase1HTFProbe(config, scoring_system=scoring_system)
    return probe, scoring_system


def test_phase1_probe_uses_adaptive_scoring_penalties(phase1_with_scoring):
    probe, scoring_system = phase1_with_scoring

    metrics = {
        "ic_oos": 0.12,
        "se_wild_bootstrap": 0.05,
        "cpu_p95": 1.5,
        "staleness": 0.3,
    }

    baseline_score = probe._calculate_utility_score(**metrics)

    # Change penalty configuration through the adaptive scoring system
    scoring_system.meta_learner.lambda_unc = 0.2
    scoring_system.meta_learner.lambda_cost = 0.1
    scoring_system.meta_learner.lambda_stale = 0.15

    updated_score = probe._calculate_utility_score(**metrics)
    expected_score = scoring_system.calculate_utility_score(**metrics)

    assert updated_score == pytest.approx(expected_score)
    assert updated_score != pytest.approx(baseline_score)
