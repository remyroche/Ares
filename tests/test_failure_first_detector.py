from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.unsupervised_regime_learning.failure_first_detector import (
    FailureFirstDetectorConfig,
    add_causal_bocpd_features,
    chronological_failure_first_oof,
    fit_failure_first_detector,
    validate_detector_features,
)


def _detector_rows(rows: int = 144) -> pd.DataFrame:
    timestamp = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    phase = np.arange(rows)
    state = np.where((phase // 12) % 3 == 0, "stable", "volatility_expansion")
    destination = np.roll(state, -3)
    destination[-3:] = state[-3:]
    transition = state != destination
    active = np.r_[False, state[1:] != state[:-1]]
    return pd.DataFrame(
        {
            "candidate_id": [f"candidate-{index:04d}" for index in range(rows)],
            "execution_decision_utc": timestamp,
            "side_name": np.where(phase % 2, "short", "long"),
            "market_volatility_state": np.sin(phase / 7.0),
            "model_entropy_state": np.cos(phase / 9.0),
            "target__transition_within_3h": transition,
            "target__active_transition": active,
            "target__current_failure_state": state,
            "target__destination_state_3h": destination,
            "transition_label_available_at": timestamp + pd.Timedelta("15h"),
        }
    )


def test_feature_contract_rejects_outcomes_and_excess_width() -> None:
    assert validate_detector_features(["market_volatility_state"], max_features=2) == [
        "market_volatility_state"
    ]
    with pytest.raises(ValueError, match="outcome-like"):
        validate_detector_features(["execution_net_ev_12h"])
    with pytest.raises(ValueError, match="exceeds"):
        validate_detector_features(["safe_a", "safe_b", "safe_c"], max_features=2)


def test_bocpd_is_prefix_causal() -> None:
    rows = _detector_rows(96)
    short = add_causal_bocpd_features(
        rows.iloc[:72],
        signal_columns=["market_volatility_state", "model_entropy_state"],
    )
    long = add_causal_bocpd_features(
        rows,
        signal_columns=["market_volatility_state", "model_entropy_state"],
    )
    columns = [
        "failure_bocpd_probability_max",
        "failure_bocpd_break_count",
        "failure_bocpd_break_intensity",
    ]
    np.testing.assert_allclose(
        short[columns].to_numpy(float),
        long.iloc[:72][columns].to_numpy(float),
        equal_nan=True,
    )


def test_bundle_uses_only_labels_available_before_boundary() -> None:
    rows = _detector_rows()
    config = FailureFirstDetectorConfig(
        min_train_rows=40,
        min_class_rows=2,
        max_iter=8,
        min_samples_leaf=5,
    )
    boundary = pd.Timestamp("2026-01-05T00:00:00Z")
    bundle = fit_failure_first_detector(
        rows,
        feature_columns=["market_volatility_state", "model_entropy_state"],
        train_end_exclusive=boundary,
        config=config,
    )
    assert pd.Timestamp(bundle.train_label_available_max) < boundary
    assert bundle.transition_head.model is not None
    assert bundle.active_head.model is not None
    assert bundle.current_state_head.model is not None
    assert bundle.destination_head.model is not None
    assert bundle.train_rows == int(
        (rows["transition_label_available_at"] < boundary).sum()
    )
    scored = bundle.score(rows.iloc[-12:])
    assert scored["p_transition_within_3h"].between(0.0, 1.0).all()
    assert scored["p_active_transition"].between(0.0, 1.0).all()
    destination = [
        name for name in scored if name.startswith("p_destination__")
    ]
    np.testing.assert_allclose(scored[destination].sum(axis=1), 1.0)


def test_chronological_oof_records_strict_label_cutoff() -> None:
    rows = _detector_rows(180)
    config = FailureFirstDetectorConfig(
        first_eval_time="2026-01-04T00:00:00Z",
        eval_hours=24,
        min_train_rows=40,
        min_class_rows=2,
        max_iter=8,
        min_samples_leaf=5,
    )
    predictions, bundles = chronological_failure_first_oof(
        rows,
        feature_columns=["market_volatility_state", "model_entropy_state"],
        config=config,
    )
    assert bundles
    assert not predictions.empty
    assert (
        pd.to_datetime(predictions["train_label_available_max"], utc=True)
        < pd.to_datetime(predictions["train_end_exclusive"], utc=True)
    ).all()
    assert predictions["p_failure_destination_3h"].between(0.0, 1.0).all()
