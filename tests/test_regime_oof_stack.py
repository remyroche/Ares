from __future__ import annotations

import pandas as pd
import pytest

from extreme_price_movements.regime_oof_stack import (
    RegimeOOFStackError,
    asof_join_regime_timeline,
    combine_regime_transition_feature_view,
    derive_soft_state_fields,
    exact_join_regime_outputs,
    matched_regime_transition_ablation_arms,
    period_q10_q50,
    qualify_category_stability,
    validate_combined_regime_transition_outputs,
    validate_regime_output_frame,
    validate_transition_output_frame,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "__ts__": pd.to_datetime(["2026-01-02T01:00Z", "2026-01-02T02:00Z"], utc=True),
            "__symbol__": ["BTC/USD:USD", "ETH/USD:USD"],
            "side_name": ["long", "short"],
        }
    )


def _outputs() -> pd.DataFrame:
    frame = _candidates()
    frame["regime_fold_id"] = "fold_1"
    frame["regime_train_end_utc"] = pd.Timestamp("2026-01-02T00:00Z")
    frame["regime_available_utc"] = frame["__ts__"]
    frame["regime_state_p__0"] = [0.8, 0.1]
    frame["regime_state_p__1"] = [0.2, 0.9]
    frame["regime_state_ood_score"] = [0.2, 0.4]
    return derive_soft_state_fields(frame)


def _transition_outputs() -> pd.DataFrame:
    frame = _candidates()
    frame["transition_fold_id"] = "transition_fold_1"
    frame["transition_train_end_utc"] = pd.Timestamp("2026-01-02T00:00Z")
    frame["transition_available_utc"] = frame["__ts__"]
    frame["transition_state_p__stable"] = [0.7, 0.2]
    frame["transition_state_p__active"] = [0.3, 0.8]
    frame["transition_state_ood_score"] = [0.1, 0.3]
    return derive_soft_state_fields(frame, probability_prefix="transition_state_p__")


def test_validate_regime_outputs_requires_provenance_simplex_and_identity() -> None:
    checked = validate_regime_output_frame(_outputs())
    assert checked["regime_state_id"].tolist() == ["0", "1"]
    broken = _outputs()
    broken.loc[0, "regime_state_id"] = "1"
    with pytest.raises(RegimeOOFStackError, match="argmax"):
        validate_regime_output_frame(broken)


def test_validate_regime_outputs_rejects_future_training_availability_and_outcomes() -> None:
    future_train = _outputs()
    future_train.loc[0, "regime_train_end_utc"] = future_train.loc[0, "__ts__"]
    with pytest.raises(RegimeOOFStackError, match="strictly before"):
        validate_regime_output_frame(future_train)
    future_available = _outputs()
    future_available.loc[0, "regime_available_utc"] = future_available.loc[0, "__ts__"] + pd.Timedelta(seconds=1)
    with pytest.raises(RegimeOOFStackError, match="at or before"):
        validate_regime_output_frame(future_available)
    leaked = _outputs()
    leaked["execution_net_ev_12h"] = 0.01
    with pytest.raises(RegimeOOFStackError, match="outcome-derived"):
        validate_regime_output_frame(leaked)


def test_exact_candidate_join_preserves_every_identity() -> None:
    joined = exact_join_regime_outputs(_candidates(), _outputs())
    assert len(joined) == 2
    assert joined["candidate_id"].tolist() == ["a", "b"]
    missing = _outputs().iloc[:1].copy()
    with pytest.raises(RegimeOOFStackError, match="lacks OOF"):
        exact_join_regime_outputs(_candidates(), missing)


def test_regime_and_transition_layers_are_distinct_and_combine_without_loss() -> None:
    regime = _outputs()
    transition = _transition_outputs()
    assert validate_transition_output_frame(transition)["transition_state_id"].tolist() == ["stable", "active"]
    with pytest.raises(RegimeOOFStackError, match="transition_fold_id"):
        validate_transition_output_frame(regime)
    with pytest.raises(RegimeOOFStackError, match="regime_fold_id"):
        validate_regime_output_frame(transition)
    combined = combine_regime_transition_feature_view(_candidates(), regime, transition)
    validated = validate_combined_regime_transition_outputs(combined)
    assert len(validated) == len(_candidates())
    assert {"regime_state_p__0", "transition_state_p__active"}.issubset(validated.columns)


def test_matched_ablation_arms_cover_separate_and_combined_layers() -> None:
    arms = {arm.name: arm for arm in matched_regime_transition_ablation_arms()}
    assert set(arms) == {"baseline", "regime_only", "transition_only", "regime_plus_transition"}
    assert not arms["baseline"].include_regime_state
    assert not arms["baseline"].include_transition_state
    assert arms["regime_only"].include_regime_state
    assert not arms["regime_only"].include_transition_state
    assert not arms["transition_only"].include_regime_state
    assert arms["regime_plus_transition"].include_regime_state
    assert arms["regime_plus_transition"].include_transition_state


def test_asof_join_is_backward_and_preserves_population() -> None:
    candidates = _candidates()
    timeline = pd.DataFrame(
        {
            "side_name": ["long", "short"],
            "regime_source_utc": pd.to_datetime(["2026-01-02T00:30Z", "2026-01-02T01:30Z"], utc=True),
            "regime_fold_id": ["f", "f"],
            "regime_train_end_utc": pd.to_datetime(["2026-01-01T23:00Z", "2026-01-02T00:30Z"], utc=True),
            "regime_available_utc": pd.to_datetime(["2026-01-02T00:30Z", "2026-01-02T01:30Z"], utc=True),
            "regime_context_score": [0.1, 0.2],
        }
    )
    joined = asof_join_regime_timeline(candidates, timeline, max_lag=pd.Timedelta(hours=1))
    assert joined["regime_source_utc"].tolist() == timeline["regime_source_utc"].tolist()
    assert joined["candidate_id"].tolist() == ["a", "b"]


def test_period_q10_q50_uses_utc_calendar_period_means() -> None:
    rows = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-01-05", "2026-01-06", "2026-01-12", "2026-01-13"], utc=True
            ),
            "net": [1.0, 3.0, 5.0, 7.0],
        }
    )
    result = period_q10_q50(rows, value_col="net", period_type="week")
    assert result["periods"] == 2
    assert result["q50"] == pytest.approx(4.0)
    assert result["q10"] == pytest.approx(2.4)


def test_category_stability_requires_support_and_consistent_effect() -> None:
    weeks = pd.date_range("2026-01-05", periods=12, freq="7D", tz="UTC")
    rows = []
    for timestamp in weeks:
        rows.extend(
            [{"__ts__": timestamp, "state": "regular", "ev": 0.0}] * 40
            + [{"__ts__": timestamp, "state": "good", "ev": 1.0}] * 40
        )
    result = qualify_category_stability(
        pd.DataFrame(rows), category_col="state", value_col="ev", min_rows=400, min_weeks=12, min_months=3
    ).set_index("category")
    assert bool(result.loc["good", "stable_category"])
    assert bool(result.loc["regular", "stable_category"])
