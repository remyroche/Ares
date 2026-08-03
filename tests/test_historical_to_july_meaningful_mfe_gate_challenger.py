import inspect

import numpy as np
import pandas as pd
import pytest

from scripts.run_historical_to_july_meaningful_mfe_gate_challenger import (
    IDENTITY,
    _evaluate_selection,
    apply_calibrator,
    classification_metrics,
    fit_calibrator,
    freeze_historical_state,
    historical_folds,
    predict_model,
    select_features_nested,
)


def _historical_calendar() -> pd.DataFrame:
    timestamp = pd.date_range(
        "2026-05-01", "2026-07-19 23:00", freq="5min", tz="UTC"
    )
    return pd.DataFrame(
        {
            "__ts__": timestamp,
            "label_resolution_utc": timestamp + pd.Timedelta(hours=12),
        }
    )


def test_temporal_folds_require_strictly_resolved_training_labels():
    frame = _historical_calendar()
    folds = historical_folds(frame)
    assert len(folds) == 3
    for fold in folds:
        assert (
            frame.iloc[fold.train]["label_resolution_utc"]
            < fold.validation_start
        ).all()
        assert (
            frame.iloc[fold.validation]["__ts__"] >= fold.validation_start
        ).all()
        assert (frame.iloc[fold.validation]["__ts__"] < fold.validation_end).all()


def test_nested_feature_selection_uses_only_passed_training_positions():
    rows = 400
    rng = np.random.default_rng(7)
    target = np.r_[np.zeros(rows // 2), np.ones(rows // 2)]
    matrix = pd.DataFrame(
        {
            "stable_signal": target + rng.normal(0, 0.05, rows),
            "stable_signal_duplicate": target + rng.normal(0, 0.01, rows),
            "noise": rng.normal(size=rows),
            "noise_2": rng.normal(size=rows),
            "constant": 1.0,
        }
    )
    selected, screen = select_features_nested(
        matrix, target, np.arange(rows), count=3
    )
    assert "constant" not in selected
    assert "stable_signal" in selected or "stable_signal_duplicate" in selected
    assert not {
        "stable_signal",
        "stable_signal_duplicate",
    }.issubset(selected)
    assert screen.loc[screen["selected"], "feature"].tolist() == selected


def test_calibration_and_metrics_are_probability_safe():
    raw = np.linspace(0.05, 0.95, 200)
    target = (raw > 0.6).astype(int)
    model = fit_calibrator("sigmoid", raw, target)
    calibrated = apply_calibrator("sigmoid", model, raw)
    metrics = classification_metrics(target, calibrated)
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))
    assert metrics["auc"] > 0.99
    assert metrics["pr_auc"] > 0.99
    assert 0.0 <= metrics["ece_10"] <= 1.0


def test_soft_regressor_prediction_boundary_uses_predict():
    class Regressor:
        def predict(self, matrix):
            return np.repeat(0.4, len(matrix))

    prediction = predict_model(Regressor(), "lightgbm", pd.DataFrame({"x": [1, 2]}))
    np.testing.assert_allclose(prediction, [0.4, 0.4])


def test_top10_economics_is_one_pooled_global_ordering():
    rows = 20
    timestamp = pd.date_range("2026-07-20", periods=rows, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "candidate_id": [f"id-{index}" for index in range(rows)],
            "__ts__": timestamp,
            "__symbol__": [f"A{index}" for index in range(rows)],
            "side_name": ["long", "short"] * (rows // 2),
            "score": np.arange(rows),
            "execution_net_ev_12h": np.arange(rows) / 10_000,
            "meaningful_mfe_reached": [0, 1] * (rows // 2),
            "adverse_1atr_reached": [1, 0] * (rows // 2),
        }
    )
    result = _evaluate_selection(
        frame, arm="test", score_column="score", scope="pooled"
    )
    assert result["top10_rows"] == 2
    assert result["top10_net_ev_bps"] == pytest.approx(18.5)


def test_historical_freezer_has_no_current_outcome_argument():
    parameters = set(inspect.signature(freeze_historical_state).parameters)
    assert parameters == {
        "history",
        "matrix",
        "raw_features",
        "output_dir",
        "seed",
    }
    assert not any("current" in parameter or "july" in parameter for parameter in parameters)
