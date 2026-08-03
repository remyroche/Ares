from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.shared_regime_calibration import (
    CausalCalibrationError,
    fit_shared_bps_calibration,
    predict_shared_bps_calibration,
    prequential_shared_bps_calibration,
)


def _frame() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "__ts__": ts,
            "outcome_resolved_at": ts + pd.Timedelta(minutes=30),
            "side_name": ["long", "long", "short", "short"] * 2,
            "regime_state_p__calm": [0.9, 0.8, 0.2, 0.1, 0.8, 0.7, 0.3, 0.2],
            "regime_state_p__stress": [0.1, 0.2, 0.8, 0.9, 0.2, 0.3, 0.7, 0.8],
        }
    )


def test_c0_c1_c2_are_additive_common_bps_corrections() -> None:
    frame = _frame()
    raw = np.zeros(len(frame))
    # Global = +10 bps. Long residuals are higher and stress has an additional
    # effect, so C0/C1/C2 must be visibly different without fitting experts.
    target = np.array([25, 20, 0, -5, 22, 18, 2, -2], dtype=float)
    cutoff = "2024-01-02T00:00:00Z"
    c0 = fit_shared_bps_calibration(frame, raw, target, fit_before_utc=cutoff, mode="C0_global", min_global_rows=1)
    c1 = fit_shared_bps_calibration(frame, raw, target, fit_before_utc=cutoff, mode="C1_side", min_global_rows=1, side_shrink_rows=1)
    c2 = fit_shared_bps_calibration(
        frame, raw, target, fit_before_utc=cutoff, mode="C2_side_soft_regime",
        soft_regime_columns=["regime_state_p__calm", "regime_state_p__stress"],
        min_global_rows=1, side_shrink_rows=1, regime_shrink_rows=1, regime_weight_cap=0.5,
    )
    future = frame.copy()
    future["__ts__"] = future["__ts__"] + pd.Timedelta(days=2)
    p0 = predict_shared_bps_calibration(c0, future, raw)
    p1 = predict_shared_bps_calibration(c1, future, raw)
    detail = predict_shared_bps_calibration(c2, future, raw, return_details=True)

    assert np.allclose(p0, p0[0])
    assert p1[future.side_name.eq("long")].mean() > p1[future.side_name.eq("short")].mean()
    assert detail.loc[0, "calibrated_common_bps"] != detail.loc[1, "calibrated_common_bps"]
    assert set(detail.calibration_mode) == {"C2_side_soft_regime"}
    # The component carries corrections only—not a hidden model/expert route.
    assert not hasattr(c2, "models")
    assert not hasattr(c2, "experts")


def test_fit_rejects_current_or_future_outcome_resolution() -> None:
    frame = _frame()
    with pytest.raises(CausalCalibrationError, match="unresolved/current/future"):
        fit_shared_bps_calibration(
            frame, np.zeros(len(frame)), np.ones(len(frame)),
            fit_before_utc=frame.loc[4, "__ts__"], mode="C0_global", min_global_rows=1,
        )


def test_prediction_cannot_be_applied_before_fit_boundary() -> None:
    frame = _frame()
    calibrator = fit_shared_bps_calibration(
        frame, np.zeros(len(frame)), np.ones(len(frame)),
        fit_before_utc="2024-01-02", mode="C0_global", min_global_rows=1,
    )
    with pytest.raises(CausalCalibrationError, match="at/after its fit boundary"):
        predict_shared_bps_calibration(calibrator, frame, np.zeros(len(frame)))


def test_prequential_predictions_ignore_current_and_future_labels() -> None:
    frame = _frame()
    # One decision per day makes the strict prior-resolved boundary obvious.
    frame["__ts__"] = pd.date_range("2024-01-01", periods=len(frame), freq="D", tz="UTC")
    frame["outcome_resolved_at"] = frame["__ts__"] + pd.Timedelta(hours=12)
    raw = np.zeros(len(frame))
    target = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=float)
    mapped_a, audit = prequential_shared_bps_calibration(
        frame, raw, target, mode="C2_side_soft_regime",
        soft_regime_columns=["regime_state_p__calm", "regime_state_p__stress"],
        min_global_rows=2, side_shrink_rows=1, regime_shrink_rows=1,
    )
    changed = target.copy()
    changed[5:] += 100_000.0
    mapped_b, _ = prequential_shared_bps_calibration(
        frame, raw, changed, mode="C2_side_soft_regime",
        soft_regime_columns=["regime_state_p__calm", "regime_state_p__stress"],
        min_global_rows=2, side_shrink_rows=1, regime_shrink_rows=1,
    )
    # Mutating outcomes that resolve after the fourth decision cannot change
    # any earlier prequential calibration output.
    np.testing.assert_allclose(mapped_a[:5], mapped_b[:5])
    fitted = audit.loc[audit.status.eq("prior_resolved_hierarchical_calibration")]
    assert not fitted.empty
    assert (pd.to_datetime(fitted.max_resolution_utc, utc=True) < pd.to_datetime(fitted.anchor_utc, utc=True)).all()


def test_c2_rejects_invalid_or_non_simplex_soft_regime_contract() -> None:
    frame = _frame()
    frame.loc[0, "regime_state_p__stress"] = 0.4
    with pytest.raises(CausalCalibrationError, match="sum to one"):
        fit_shared_bps_calibration(
            frame, np.zeros(len(frame)), np.ones(len(frame)), fit_before_utc="2024-01-02",
            mode="C2_side_soft_regime", soft_regime_columns=["regime_state_p__calm", "regime_state_p__stress"],
            min_global_rows=1,
        )


def _affine_training_frame() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows: list[dict[str, object]] = []
    raw: list[float] = []
    target: list[float] = []
    start = pd.Timestamp("2024-01-01", tz="UTC")
    # Balanced, crisp training memberships make the known shared hierarchy
    # identifiable.  Application still accepts arbitrary soft memberships.
    for side, side_intercept, side_slope in (("long", 12.0, 0.15), ("short", -12.0, -0.15)):
        for calm, regime_intercept, regime_slope in ((1.0, 5.0, 0.10), (0.0, -5.0, -0.10)):
            for value in np.linspace(-100.0, 100.0, 80):
                ix = len(rows)
                rows.append({
                    "__ts__": start + pd.Timedelta(minutes=ix),
                    "outcome_resolved_at": start + pd.Timedelta(minutes=ix, seconds=30),
                    "side_name": side,
                    "regime_state_p__calm": calm,
                    "regime_state_p__stress": 1.0 - calm,
                })
                raw.append(value)
                target.append(20.0 + 1.4 * value + side_intercept + side_slope * value + regime_intercept + regime_slope * value)
    return pd.DataFrame(rows), np.asarray(raw), np.asarray(target)


def test_c3_affine_recovers_shared_slope_and_intercept_hierarchy() -> None:
    frame, raw, target = _affine_training_frame()
    calibrator = fit_shared_bps_calibration(
        frame, raw, target, fit_before_utc="2024-02-01", mode="C3_hierarchical_affine_soft_regime",
        soft_regime_columns=["regime_state_p__calm", "regime_state_p__stress"],
        min_global_rows=1, global_shrink_rows=1e-6, side_shrink_rows=1e-6,
        regime_shrink_rows=1e-6, regime_weight_cap=1.0,
    )
    assert calibrator.contract == "shared_hierarchical_affine_common_bps_v1"
    assert calibrator.global_correction_bps == pytest.approx(20.0, abs=1e-5)
    assert calibrator.global_slope == pytest.approx(1.4, abs=1e-5)
    assert calibrator.side_corrections_bps["long"] == pytest.approx(12.0, abs=1e-5)
    assert calibrator.side_slope_corrections["short"] == pytest.approx(-0.15, abs=1e-5)


def test_c3_global_affine_terms_are_strongly_shrunk_on_tiny_support() -> None:
    frame = _frame().iloc[:4].copy()
    raw = np.array([-10.0, -3.0, 3.0, 10.0])
    target = 100.0 + 3.0 * raw
    calibrator = fit_shared_bps_calibration(
        frame, raw, target, fit_before_utc="2024-01-02", mode="C3_hierarchical_affine_soft_regime",
        soft_regime_columns=["regime_state_p__calm", "regime_state_p__stress"], min_global_rows=1,
        global_shrink_rows=10_000.0, side_shrink_rows=10_000.0, regime_shrink_rows=10_000.0,
    )
    assert abs(calibrator.global_slope - 1.0) < 0.01
    assert abs(calibrator.global_correction_bps) < 1.0
    assert max(abs(value) for value in calibrator.side_slope_corrections.values()) < 0.01


def test_c3_soft_regime_prediction_is_a_mixture_not_a_route() -> None:
    frame, raw, target = _affine_training_frame()
    calibrator = fit_shared_bps_calibration(
        frame, raw, target, fit_before_utc="2024-02-01", mode="C3_hierarchical_affine_soft_regime",
        soft_regime_columns=["regime_state_p__calm", "regime_state_p__stress"],
        min_global_rows=1, global_shrink_rows=1e-6, side_shrink_rows=1e-6,
        regime_shrink_rows=1e-6, regime_weight_cap=1.0,
    )
    future = pd.DataFrame({
        "__ts__": pd.to_datetime(["2024-03-01", "2024-03-01", "2024-03-01"], utc=True),
        "side_name": ["long", "long", "long"],
        "regime_state_p__calm": [1.0, 0.0, 0.25],
        "regime_state_p__stress": [0.0, 1.0, 0.75],
    })
    prediction = predict_shared_bps_calibration(calibrator, future, np.array([40.0, 40.0, 40.0]))
    assert prediction[2] == pytest.approx(0.25 * prediction[0] + 0.75 * prediction[1], abs=1e-7)


def test_c3_prequential_outputs_do_not_use_future_labels() -> None:
    frame = _frame()
    frame["__ts__"] = pd.date_range("2024-01-01", periods=len(frame), freq="D", tz="UTC")
    frame["outcome_resolved_at"] = frame["__ts__"] + pd.Timedelta(hours=12)
    raw = np.linspace(-10.0, 10.0, len(frame))
    target = 5.0 + 1.5 * raw
    first, audit = prequential_shared_bps_calibration(
        frame, raw, target, mode="C3_hierarchical_affine_soft_regime",
        soft_regime_columns=["regime_state_p__calm", "regime_state_p__stress"], min_global_rows=2,
        global_shrink_rows=1.0, side_shrink_rows=1.0, regime_shrink_rows=1.0,
    )
    altered = target.copy()
    altered[5:] += 100_000.0
    second, _ = prequential_shared_bps_calibration(
        frame, raw, altered, mode="C3_hierarchical_affine_soft_regime",
        soft_regime_columns=["regime_state_p__calm", "regime_state_p__stress"], min_global_rows=2,
        global_shrink_rows=1.0, side_shrink_rows=1.0, regime_shrink_rows=1.0,
    )
    np.testing.assert_allclose(first[:5], second[:5])
    fitted = audit.loc[audit.status.eq("prior_resolved_hierarchical_calibration")]
    assert (pd.to_datetime(fitted.max_resolution_utc, utc=True) < pd.to_datetime(fitted.anchor_utc, utc=True)).all()
