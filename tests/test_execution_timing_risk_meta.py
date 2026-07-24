from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.execution_ev_meta import FeatureProvenance
from extreme_price_movements.execution_timing_risk_meta import (
    EXECUTION_TIMING_RISK_BUNDLE_SCHEMA,
    ExecutionTimingRiskModelBundle,
    ExecutionTimingRiskTargetSpec,
    TimingRiskTrainerConfig,
    _ConstantBinaryClassifier,
    _fit_probability_calibrator,
    build_execution_timing_risk_targets,
    execution_timing_risk_metrics,
    predict_execution_timing_risk_bundle,
    timing_priority,
    validate_execution_timing_risk_feature_contract,
)


def _frame() -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "__ts__": times,
            "side_name": ["long", "short"] * 4,
            "execution_net_ev_12h": [0.02, -0.01, 0.03, 0.01, -0.02, 0.04, 0.01, -0.03],
            "execution_exit_hour": [2.0, 1.0, 4.0, 6.0, 1.0, 3.0, 5.0, 2.0],
            "execution_exit_reason": [
                "trailing",
                "full_stop",
                "trailing",
                "timeout",
                "adverse_exit",
                "trailing",
                "timeout",
                "timeout",
            ],
            "score_existing_alpha": np.linspace(0.1, 0.8, 8),
            "available_at": times,
        }
    )


def _provenance() -> dict[str, FeatureProvenance]:
    return {
        "score_existing_alpha": FeatureProvenance(
            family="alpha_score",
            source="frozen execution-EV alpha head",
            available_at_col="available_at",
        )
    }


def test_targets_keep_timing_conditional_on_non_loss_and_model_loss_separately() -> None:
    targets = build_execution_timing_risk_targets(_frame())
    np.testing.assert_allclose(
        targets["timing_target_hours"],
        [2.0, np.nan, 4.0, 6.0, np.nan, 3.0, 5.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        targets["loss_risk_target"].to_numpy(dtype=int), [0, 1, 0, 0, 1, 0, 0, 1]
    )


def test_targets_reject_unknown_reason_and_nonpositive_exit_hour() -> None:
    frame = _frame()
    frame.loc[0, "execution_exit_reason"] = "unknown"
    frame.loc[1, "execution_exit_hour"] = 0.0
    targets = build_execution_timing_risk_targets(frame)
    assert not bool(targets.loc[0, "timing_risk_target_valid"])
    assert not bool(targets.loc[1, "timing_risk_target_valid"])


def test_probability_calibrator_is_bounded_and_non_decreasing() -> None:
    raw = np.linspace(0.02, 0.98, 80)
    target = (raw > 0.65).astype(int)
    calibrator = _fit_probability_calibrator(raw, target)
    calibrated = calibrator.predict(raw)
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))
    assert np.all(np.diff(calibrated) >= -1e-12)


def test_feature_contract_rejects_realized_fields_and_late_inputs() -> None:
    frame = _frame()
    provenance = _provenance()
    assert validate_execution_timing_risk_feature_contract(frame, provenance) == ["score_existing_alpha"]
    bad = dict(provenance)
    bad["execution_exit_hour"] = FeatureProvenance(
        family="alpha_score", source="incorrect realized outcome", available_at_col="available_at"
    )
    with pytest.raises(ValueError, match="train-only"):
        validate_execution_timing_risk_feature_contract(frame, bad)
    late = frame.copy()
    late.loc[0, "available_at"] = late.loc[0, "__ts__"] + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="available after entry"):
        validate_execution_timing_risk_feature_contract(late, provenance)


def test_priority_is_bounded_and_metrics_include_required_scopes() -> None:
    frame = _frame()
    targets = build_execution_timing_risk_targets(frame)
    probability = np.asarray([0.1, 0.9, 0.2, 0.3, 0.8, 0.1, 0.2, 0.9])
    predicted_time = np.asarray([2.0, 11.0, 3.0, 5.0, 10.0, 3.0, 4.0, 12.0])
    predictions = pd.DataFrame(
        {
            "oof_predicted_time_hours": predicted_time,
            "oof_loss_probability": probability,
            "oof_timing_priority": timing_priority(probability, predicted_time),
        },
        index=frame.index,
    )
    assert predictions["oof_timing_priority"].between(0.0, 1.0).all()
    metrics = execution_timing_risk_metrics(
        frame,
        targets,
        predictions,
        config=TimingRiskTrainerConfig(),
        target_spec=ExecutionTimingRiskTargetSpec(),
    )
    assert {"overall", "side", "month"}.issubset(set(metrics["scope"]))
    overall = metrics.loc[metrics["scope"] == "overall"].iloc[0]
    assert overall["loss_auc"] == pytest.approx(1.0)
    assert np.isfinite(overall["loss_brier"])
    assert np.isfinite(overall["timing_mae_non_loss"])
    assert np.isfinite(overall["top10_realized_ev_mean"])


class _TimeModel:
    def predict(self, values: pd.DataFrame) -> np.ndarray:
        return np.full(len(values), 2.0)


def test_prediction_requires_no_realized_target_columns() -> None:
    frame = _frame().loc[:, ["__ts__", "side_name", "score_existing_alpha", "available_at"]]
    config = TimingRiskTrainerConfig()
    bundle = ExecutionTimingRiskModelBundle(
        schema=EXECUTION_TIMING_RISK_BUNDLE_SCHEMA,
        config=asdict(config),
        target_spec=ExecutionTimingRiskTargetSpec(),
        provenance=_provenance(),
        feature_names=("score_existing_alpha",),
        models={
            "long": {"features": ("score_existing_alpha",), "time_model": _TimeModel(), "risk_model": _ConstantBinaryClassifier(0.25)},
            "short": {"features": ("score_existing_alpha",), "time_model": _TimeModel(), "risk_model": _ConstantBinaryClassifier(0.25)},
        },
        report={},
        oof_predictions=pd.DataFrame(index=frame.index),
        oof_provenance=pd.DataFrame(index=frame.index),
    )
    scored = predict_execution_timing_risk_bundle(bundle, frame)
    assert set(scored) == {"predicted_time_hours", "loss_probability", "timing_priority"}
    np.testing.assert_allclose(scored["predicted_time_hours"], 2.0)
    np.testing.assert_allclose(scored["loss_probability"], 0.25)
