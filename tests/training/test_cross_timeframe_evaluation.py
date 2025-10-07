import importlib.util
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _load_evaluation_module():
    evaluation_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "training"
        / "steps"
        / "pre_training"
        / "interaction_feature_generator"
        / "cross_timeframe_generation"
        / "evaluation.py"
    )

    spec = importlib.util.spec_from_file_location(
        "cross_timeframe_evaluation", evaluation_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


evaluation_module = _load_evaluation_module()
WalkForwardEvaluation = evaluation_module.WalkForwardEvaluation
WalkForwardFold = evaluation_module.WalkForwardFold


def _create_config():
    return SimpleNamespace(walk_forward_folds=2, embargo_minutes=60, spa_test=False)


def test_evaluate_features_uses_provided_matrix(monkeypatch):
    config = _create_config()
    evaluation = WalkForwardEvaluation(config)

    index = pd.date_range("2022-01-01", periods=120, freq="min")
    feature_matrix = pd.DataFrame({
        "feat_a": np.linspace(0, 1, len(index)),
        "feat_b": np.linspace(1, 0, len(index)),
    }, index=index)
    targets = pd.Series(np.sin(np.linspace(0, np.pi, len(index))), index=index, name="target")

    calls = {"count": 0}

    def loader():
        calls["count"] += 1
        return feature_matrix

    captured = {}
    fold = WalkForwardFold(
        train_start=datetime.now(),
        train_end=datetime.now(),
        test_start=datetime.now(),
        test_end=datetime.now(),
        ic=0.15,
        mse=0.2,
        mae=0.1,
        r2=0.5,
        sharpe=0.3,
        max_drawdown=-0.05,
        metadata={"fold": 0},
    )

    def fake_validate(features, aligned_targets, regime_segments):
        captured["features"] = features.copy()
        captured["targets"] = aligned_targets.copy()
        return [fold]

    monkeypatch.setattr(evaluation.walk_forward_validator, "validate_features", fake_validate)

    result = evaluation.evaluate_features([
        "feat_a",
        "feat_b",
    ], targets, feature_data=loader)

    assert calls["count"] == 1
    pd.testing.assert_frame_equal(captured["features"], feature_matrix[["feat_a", "feat_b"]])
    pd.testing.assert_series_equal(captured["targets"], targets)
    assert result.metadata["n_features"] == 2
    assert len(result.walk_forward_results) == 1
    assert result.walk_forward_results[0]["ic"] == fold.ic


def test_evaluate_features_raises_on_missing_columns():
    config = _create_config()
    evaluation = WalkForwardEvaluation(config)

    index = pd.date_range("2022-01-01", periods=10, freq="min")
    feature_matrix = pd.DataFrame({"feat_a": np.arange(len(index))}, index=index)
    targets = pd.Series(np.arange(len(index)), index=index)

    with pytest.raises(ValueError, match="Missing features"):
        evaluation.evaluate_features([
            "feat_a",
            "feat_b",
        ], targets, feature_data=feature_matrix)
