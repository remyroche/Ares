"""Regression checks for feature-arm parity with the frozen D2 base."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_strict_r3_base_f1_session_funnel import _d2_weights, _strict_train
from scripts.select_strict_r3_base_context_features import _bin, _conditional_direction, _conditional_mi


def test_feature_arm_training_excludes_labels_resolved_in_reserve() -> None:
    cutoff = pd.Timestamp("2026-02-01T00:00:00Z")
    reserve_start = cutoff - pd.Timedelta(days=28)
    frame = pd.DataFrame({
        "candidate_id": ["old", "reserve_label", "late_decision"],
        "__decision_ts__": pd.to_datetime([
            "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-10T00:00:00Z",
        ], utc=True),
        "r3_label_available_ts": pd.to_datetime([
            "2026-01-02T00:00:00Z",
            "2026-01-10T00:00:00Z",
            "2026-01-11T00:00:00Z",
        ], utc=True),
        "r3_class": [0, 2, 1],
        "prequential_base_rank42": [.10, .95, .50],
    })

    train = _strict_train(frame, cutoff)

    assert train["candidate_id"].tolist() == ["old"]
    assert train["__decision_ts__"].lt(reserve_start).all()
    assert train["r3_label_available_ts"].lt(reserve_start).all()


def test_feature_arm_d2_weights_are_finite_and_mean_one() -> None:
    frame = pd.DataFrame({
        "r3_class": [0, 1, 2, 2],
        "prequential_base_rank42": [.10, .40, .70, .95],
    })

    weights, audit = _d2_weights(frame)

    assert np.isfinite(weights).all()
    assert np.isclose(weights.mean(), 1.0)
    assert audit["teacher_coverage"] == 1.0


def test_context_selector_conditions_on_base_rank_and_preserves_direction() -> None:
    base = np.repeat(np.arange(10, dtype=np.int16), 40)
    feature = np.tile(np.repeat(np.arange(10, dtype=np.int16), 4), 10)
    target = (feature >= 8).astype(np.int8)

    assert _conditional_mi(feature, base, target) > 0.0
    assert _conditional_direction(feature, base, target) > 0.0
    assert np.all(_bin(pd.Series(np.arange(100, dtype=float))) >= 0)
