import numpy as np
import pandas as pd
import pytest

from types import SimpleNamespace

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


def test_evaluate_features_accepts_callable_and_aligns_data():
    evaluation, captured = _build_stubbed_evaluation()

    index = pd.date_range("2023-01-01", periods=5, freq="h")
    feature_df = pd.DataFrame({"feature_a": np.arange(5.0)}, index=index)
    targets = pd.Series(np.arange(5.0), index=index)

    loader = lambda: feature_df

    result = evaluation.evaluate_features(["feature_a"], targets, feature_source=loader)

    assert isinstance(result, EvaluationResult)
    assert "features" in captured
    assert "targets" in captured
    pd.testing.assert_frame_equal(captured["features"], feature_df)
    pd.testing.assert_series_equal(captured["targets"], targets)


def test_evaluate_features_raises_for_missing_columns():
    evaluation, _ = _build_stubbed_evaluation()

    index = pd.date_range("2023-01-01", periods=5, freq="h")
    feature_df = pd.DataFrame({"feature_a": np.arange(5.0)}, index=index)
    targets = pd.Series(np.arange(5.0), index=index)

    with pytest.raises(ValueError) as excinfo:
        evaluation.evaluate_features(["feature_a", "missing"], targets, feature_source=feature_df)

    assert "missing" in str(excinfo.value)
