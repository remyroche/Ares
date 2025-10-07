import numpy as np
import pandas as pd

from types import SimpleNamespace
import sys
import types

if "cvxpy" not in sys.modules:
    cvxpy_stub = types.ModuleType("cvxpy")

    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

        def __matmul__(self, other):  # pragma: no cover - unused but kept for safety
            return 0

    class _DummyProblem:
        def __init__(self, *args, **kwargs):
            self.status = "optimal"

        def solve(self, *args, **kwargs):
            return None

    cvxpy_stub.Variable = _Dummy
    cvxpy_stub.Parameter = _Dummy
    cvxpy_stub.Problem = _DummyProblem
    cvxpy_stub.Maximize = lambda *args, **kwargs: None
    cvxpy_stub.Minimize = lambda *args, **kwargs: None
    cvxpy_stub.sum = lambda *args, **kwargs: 0
    cvxpy_stub.CBC = "CBC"
    cvxpy_stub.OPTIMAL = "optimal"
    cvxpy_stub.Constraint = _Dummy

    sys.modules["cvxpy"] = cvxpy_stub

from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.evaluation import (
    WalkForwardEvaluation,
    EvaluationResult,
)


def _build_stubbed_evaluation():
    config = SimpleNamespace(walk_forward_folds=2, embargo_minutes=30, spa_test=False)
    evaluation = WalkForwardEvaluation(config)

    captured = {}

    def _validate(features, targets, regime_segments):
        captured["features"] = features
        captured["targets"] = targets
        return []

    evaluation.walk_forward_validator.validate_features = _validate
    evaluation.bootstrap_evaluator.calculate_confidence_intervals = lambda folds, metric: (0.0, 0.0, (0.0, 0.0))
    evaluation.ablation_evaluator.perform_ablation_study = lambda features, targets, groups: {}
    evaluation.regime_evaluator.evaluate_by_regime = lambda folds, segments: {}
    evaluation.spa_tester.test_spa = lambda folds: {}

    return evaluation, captured


def _build_materialized_feature(series):
    return SimpleNamespace(feature_series=series)


def test_evaluate_features_reconstructs_matrix_from_containers():
    evaluation, captured = _build_stubbed_evaluation()

    index = pd.date_range("2023-01-01", periods=5, freq="h")
    feature_series = pd.Series(np.arange(5.0), index=index)
    targets = pd.Series(np.linspace(0.0, 1.0, 5), index=index)

    materialized_htfs = {"feature_a": _build_materialized_feature(feature_series)}

    result = evaluation.evaluate_features(
        ["feature_a"],
        targets,
        materialized_htfs=materialized_htfs,
        interactions=[],
    )

    assert isinstance(result, EvaluationResult)
    assert "features" in captured
    assert "targets" in captured
    expected_df = pd.DataFrame({"feature_a": feature_series})
    pd.testing.assert_frame_equal(captured["features"], expected_df)
    pd.testing.assert_series_equal(captured["targets"], targets)
    assert result.metadata["n_features"] == 1


def test_evaluate_features_returns_default_when_matrix_empty():
    evaluation, captured = _build_stubbed_evaluation()

    targets = pd.Series(np.linspace(0.0, 1.0, 5), index=pd.date_range("2023-01-01", periods=5, freq="h"))

    result = evaluation.evaluate_features(["missing"], targets, materialized_htfs={}, interactions=[])

    assert isinstance(result, EvaluationResult)
    assert result.metadata["reason"] == "empty_feature_matrix"
    assert "features" not in captured
