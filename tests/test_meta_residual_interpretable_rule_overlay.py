from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts import run_meta_residual_event_balanced_error_overlay as base
from scripts import run_meta_residual_interpretable_rule_overlay as overlay
from scripts.run_meta_residual_interpretable_rule_overlay import (
    GLOBAL_MARKET_EPISODE_RISK,
    GLOBAL_MARKET_EPISODE_RISK_PCT,
    SIDE_MARKET_EPISODE_RISK,
    SIDE_MARKET_EPISODE_RISK_PCT,
    PERIOD_STATE_FEATURES,
    TOP10_MARKET_PERIOD_EVENT_TARGET,
    TOP10_PERIOD_EVENT_TARGET,
    _calendar,
    _attach_market_episode_phases,
    _attach_market_period_targets,
    _attach_top10_period_targets,
    _add_episode_trajectory_features,
    _attach_train_only_adverse_subtype_features,
    _materialize_adverse_subtype_features,
    _attach_trajectory_reference,
    _episode_intervention_diagnostics,
    _episode_phase_labels,
    _event_intervention_report,
    _validate_oos_candidates,
    _period_state_frame,
    _daily_episode_state_frame,
    _attach_daily_market_context,
    _attach_daily_market_features,
    _fit_global_market_episode_context,
    _fit_side_market_episode_context,
    _load_pooled_market_clock,
    _shrunk_mechanism_reliability,
    _fit_mechanism_density_bins,
    _attach_fold_local_pooled_reliability,
)
from scripts.materialize_interpretable_overlay_forward_state import (
    _load_frozen_negative_panel,
    _materialize_global_market_episode_context,
    _materialize_side_market_episode_context,
    _materialize_period_state_features,
)


class _IdentityTransform:
    def transform(self, values):
        return values


class _ConstantRiskModel:
    def predict_proba(self, values):
        return pd.Series([0.73] * len(values)).to_numpy()


def test_pooled_market_clock_loads_primitives_then_derives_trajectories(tmp_path) -> None:
    """The raw causal panel must never be asked for derived trajectory names."""

    timestamps = pd.date_range("2026-01-01", periods=8, freq="6h", tz="UTC")
    path = tmp_path / "negative_residual_market.parquet"
    pd.DataFrame(
        {
            "negative_breadth_pct": np.linspace(0.1, 0.4, len(timestamps)),
            "unrelated": np.arange(len(timestamps)),
        },
        index=timestamps,
    ).to_parquet(path)
    config = base.Config(
        train_start="2026-01-01",
        train_end="2026-01-02",
        eval_end="2026-01-03",
    )
    train, valid, columns = _load_pooled_market_clock(
        path,
        config,
        [
            "negative_breadth_pct",
            "episode_traj__negative_breadth_pct__delta_6h",
        ],
    )
    assert columns == ["negative_breadth_pct"]
    assert list(train.columns) == ["__ts__", "negative_breadth_pct"]
    assert not valid.empty


def test_episode_phase_labels_are_local_and_outcome_only() -> None:
    days = pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "side_name": ["short"] * 5,
            "archetype_policy_key": ["short_default_clean_path"] * 5,
            "day": days,
            base.EVENT: [0, 1, 1, 0, 1],
        }
    )
    result = _episode_phase_labels(frame)
    assert result["episode_phase"].tolist() == ["normal", "onset", "persistent", "recovery", "onset"]
    assert result["episode_onset_target"].tolist() == [0, 1, 0, 0, 1]
    assert result["episode_persistent_target"].tolist() == [0, 0, 1, 0, 0]
    assert result["episode_recovery_target"].tolist() == [0, 0, 0, 1, 0]
    assert result.loc[1, "episode_block"] == result.loc[2, "episode_block"]
    assert result.loc[4, "episode_block"] > result.loc[2, "episode_block"]


def test_period_target_repeats_daily_state_not_individual_trade_loss() -> None:
    days = pd.to_datetime(
        ["2026-01-02", "2026-01-02", "2026-01-02", "2026-01-03"], utc=True
    )
    frame = pd.DataFrame(
        {
            "day": days,
            "side_name": ["short"] * 4,
            "archetype_policy_key": ["short_default_clean_path"] * 4,
            "parent_rank_v9": [0.91, 0.96, 0.84, 0.94],
            base.EVENT: [1, 1, 1, 0],
        }
    )
    frame = _episode_phase_labels(frame)
    result = _attach_top10_period_targets(frame, top10_floor=0.90)
    assert result[TOP10_PERIOD_EVENT_TARGET].tolist() == [1, 1, 0, 0]
    assert result[f"episode_onset_{TOP10_PERIOD_EVENT_TARGET}"].tolist() == [1, 1, 0, 0]
    assert result[f"episode_persistent_{TOP10_PERIOD_EVENT_TARGET}"].tolist() == [0, 0, 0, 0]


def test_episode_trajectory_features_are_causal_and_capture_persistent_transition() -> None:
    timestamps = pd.date_range("2026-01-01", periods=13, freq="6h", tz="UTC")
    state = pd.DataFrame(
        {
            "__ts__": timestamps,
            "negative_breadth_pct": np.arange(len(timestamps), dtype=np.float32),
        }
    )
    result = _add_episode_trajectory_features(state)
    last = result.iloc[-1]
    assert last["episode_traj__negative_breadth_pct__delta_48h"] == pytest.approx(8.0)
    assert last["episode_traj__negative_breadth_pct__trend_agreement_48h"] > 0.99
    assert last["episode_traj__negative_breadth_pct__trend_intensity_48h"] > 0.0
    assert last["episode_traj__negative_breadth_pct__state_variability_48h"] > 0.0

    # A later extreme observation cannot change a prior daily-open signature.
    future = pd.concat(
        [state, pd.DataFrame({"__ts__": [timestamps[-1] + pd.Timedelta(hours=6)], "negative_breadth_pct": [10_000.0]})],
        ignore_index=True,
    )
    with_future = _add_episode_trajectory_features(future).iloc[-2]
    for name in (
        "episode_traj__negative_breadth_pct__delta_48h",
        "episode_traj__negative_breadth_pct__trend_agreement_48h",
        "episode_traj__negative_breadth_pct__state_variability_48h",
    ):
        assert with_future[name] == pytest.approx(last[name])


def test_adverse_episode_subtypes_use_train_outcomes_only_for_fit(tmp_path) -> None:
    fit = pd.DataFrame(
        {
            "observable_a": np.r_[np.linspace(-2.0, -1.0, 6), np.linspace(1.0, 2.0, 6)],
            "observable_b": np.r_[np.linspace(-1.0, -0.5, 6), np.linspace(0.5, 1.0, 6)],
            "period_target": [1] * 6 + [0] * 6,
        }
    )
    # The score frame intentionally has no outcome/target column.
    score = pd.DataFrame(
        {"observable_a": [-1.5, 0.0, 1.5], "observable_b": [-0.7, 0.0, 0.7]}
    )
    fit_out, score_out, features, report, encoder = _attach_train_only_adverse_subtype_features(
        fit,
        score,
        ["observable_a", "observable_b"],
        target_column="period_target",
        seed=7,
    )
    assert report["fit_status"] == "ok"
    assert report["train_adverse_days"] == 6
    assert features
    assert all(name in fit_out and name in score_out for name in features)
    assert "period_target" not in score_out
    assert encoder is not None
    assert encoder["features"] == ["observable_a", "observable_b"]
    expected = _materialize_adverse_subtype_features(score, encoder)
    path = tmp_path / "subtype_encoder.joblib"
    joblib.dump(encoder, path)
    actual = _materialize_adverse_subtype_features(score, joblib.load(path))
    pd.testing.assert_frame_equal(actual, expected)


def test_adverse_subtype_prefix_and_daily_broadcast_are_frozen_context() -> None:
    fit = pd.DataFrame(
        {
            "observable_a": np.r_[np.linspace(-2.0, -1.0, 6), np.linspace(1.0, 2.0, 6)],
            "period_target": [1] * 6 + [0] * 6,
        }
    )
    score = pd.DataFrame({"observable_a": [-1.5, 1.5]})
    _, score_out, features, _, encoder = _attach_train_only_adverse_subtype_features(
        fit,
        score,
        ["observable_a"],
        target_column="period_target",
        seed=11,
        feature_prefix="market_adverse_subtype",
    )
    assert encoder is not None
    assert features == [
        "market_adverse_subtype_posterior_0",
        "market_adverse_subtype_posterior_1",
        "market_adverse_subtype_posterior_2",
        "market_adverse_subtype_max_posterior",
        "market_adverse_subtype_entropy",
        "market_adverse_subtype_neg_log_density",
    ]
    assert "period_target" not in score_out
    market = pd.DataFrame(
        {"__ts__": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 12:00"], utc=True)}
    )
    daily = pd.DataFrame(
        {
            "day": pd.to_datetime(["2026-01-01"], utc=True),
            **{name: [float(score_out.iloc[0][name])] for name in features},
        }
    )
    result = _attach_daily_market_features(market, daily, features)
    assert result[features].notna().all().all()


def test_mechanism_reliability_shrinks_sparse_local_cell_to_side_prior() -> None:
    train = pd.DataFrame(
        {
            "side_name": ["short"] * 11 + ["long"] * 10,
            "archetype_policy_key": ["local"] + ["peer"] * 10 + ["other"] * 10,
            "mechanism": [0] * 21,
            "target": [1] + [0] * 10 + [1] * 10,
        }
    )
    score = pd.DataFrame(
        {"side_name": ["short", "long"], "archetype_policy_key": ["local", "other"], "mechanism": [0, 0]}
    )
    result = _shrunk_mechanism_reliability(
        train, score, mechanism_column="mechanism", target_column="target", shrinkage_k=20.0
    )
    # One local adverse observation cannot dominate the short-side prior.
    assert result.loc[0, "mechanism_reliability_risk"] < 0.52
    assert result.loc[0, "mechanism_reliability_local_support"] == 1
    assert result.loc[1, "mechanism_reliability_risk"] > result.loc[0, "mechanism_reliability_risk"]


def test_pooled_mechanism_bins_use_train_edges_only() -> None:
    train = pd.DataFrame(
        {
            "market_adverse_subtype_neg_log_density": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] * 3,
            "market_adverse_subtype_entropy": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] * 3,
        }
    )
    score = pd.DataFrame(
        {
            "market_adverse_subtype_neg_log_density": [0.5, 99.0],
            "market_adverse_subtype_entropy": [0.05, 99.0],
        }
    )
    _, assigned, contract = _fit_mechanism_density_bins(train, score)
    assert contract["bins"] == 3
    assert assigned["pooled_mechanism_bin"].tolist()[0] >= 0
    assert assigned["pooled_mechanism_bin"].tolist()[1] == 8


def test_pooled_reliability_is_same_day_excluded_and_score_outcome_free() -> None:
    """Reliability must be an expanding train prior, never a daily label lookup."""

    days = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
    market = pd.DataFrame(
        {
            "__ts__": days,
            "market_adverse_subtype_neg_log_density": np.arange(len(days), dtype=np.float32),
            "market_adverse_subtype_entropy": np.linspace(0.0, 1.0, len(days), dtype=np.float32),
        }
    )
    pooled = pd.DataFrame(
        {
            "day": np.repeat(days[:6], 2),
            "side_name": ["short", "long"] * 6,
            "archetype_policy_key": ["short_default_clean_path", "other"] * 6,
            "parent_rank_v9": 0.95,
            TOP10_PERIOD_EVENT_TARGET: [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
        }
    )
    fit = pd.DataFrame({"day": days[2:6], "__ts__": days[2:6]})
    # Score state intentionally has no realised target/outcome columns.
    score = pd.DataFrame({"day": days[6:8], "__ts__": days[6:8]})
    fit_out, score_out, report = _attach_fold_local_pooled_reliability(
        fit,
        score,
        pooled,
        market,
        side="short",
        archetype="short_default_clean_path",
        target_column=TOP10_PERIOD_EVENT_TARGET,
        top10_floor=0.90,
        shrinkage_k=2.0,
    )
    assert report["status"] == "ok"
    assert len(fit_out) == len(fit)
    assert len(score_out) == len(score)
    assert "pooled_mechanism_bin" in fit_out
    assert "pooled_mechanism_bin" in score_out
    assert fit_out["mechanism_reliability_local_support"].iloc[0] == 2
    assert score_out["mechanism_reliability_global_support"].notna().all()
    assert TOP10_PERIOD_EVENT_TARGET not in score_out.columns

    # Altering the first fit day's own label cannot alter its feature value;
    # that outcome is excluded from the expanding prior at that decision time.
    altered = pooled.copy()
    altered.loc[altered["day"].eq(days[2]) & altered["side_name"].eq("short"), TOP10_PERIOD_EVENT_TARGET] = 1
    altered_fit, _, _ = _attach_fold_local_pooled_reliability(
        fit,
        score,
        altered,
        market,
        side="short",
        archetype="short_default_clean_path",
        target_column=TOP10_PERIOD_EVENT_TARGET,
        top10_floor=0.90,
        shrinkage_k=2.0,
    )
    assert altered_fit["mechanism_reliability_risk"].iloc[0] == pytest.approx(
        fit_out["mechanism_reliability_risk"].iloc[0]
    )


def test_pooled_reliability_arm_routes_prior_features_to_lgbm(monkeypatch) -> None:
    """The research arm must fit the established LGBM arm with frozen priors."""

    days = pd.date_range("2026-01-01", periods=92, freq="D", tz="UTC")
    event = pd.Series(range(len(days))).mod(9).eq(0).astype("int8").to_numpy()
    local = pd.DataFrame(
        {
            "__ts__": days,
            "day": days,
            "side_name": "short",
            "archetype_policy_key": "short_default_clean_path",
            "parent_rank_v9": 0.95,
            "ev_after_1pct": np.where(event > 0, -0.02, 0.01),
            "clean_exec": 1 - event,
            TOP10_PERIOD_EVENT_TARGET: event,
        }
    )
    for name in [*base.KEYS, base.EVENT, base.TARGET]:
        if name not in local:
            local[name] = 0
    pooled = pd.concat(
        [
            local.loc[:, ["day", "side_name", "archetype_policy_key", "parent_rank_v9", TOP10_PERIOD_EVENT_TARGET]],
            pd.DataFrame(
                {
                    "day": days,
                    "side_name": "long",
                    "archetype_policy_key": "other",
                    "parent_rank_v9": 0.95,
                    TOP10_PERIOD_EVENT_TARGET: np.roll(event, 2),
                }
            ),
        ],
        ignore_index=True,
    )
    market = pd.DataFrame(
        {
            "__ts__": days,
            "observable": np.linspace(0.0, 1.0, len(days), dtype=np.float32),
            "market_adverse_subtype_neg_log_density": np.linspace(0.0, 3.0, len(days), dtype=np.float32),
            "market_adverse_subtype_entropy": np.linspace(0.0, 1.0, len(days), dtype=np.float32),
        }
    )
    captured: list[tuple[str, list[str]]] = []

    def fake_screen(state, candidates, config, **kwargs):
        return ["observable"], pd.DataFrame()

    def fake_fit_arm(arm_name, fit, score, features, seed, **kwargs):
        captured.append((arm_name, list(features)))
        values = pd.to_numeric(score["observable"], errors="coerce").fillna(0.0).to_numpy(np.float32)
        return ({"features": list(features)}, values, values, np.sort(values), [], [])

    monkeypatch.setattr(base, "FOLD_STARTS", (days[70],))
    monkeypatch.setattr(base, "_screen_features", fake_screen)
    monkeypatch.setattr(overlay, "_fit_arm", fake_fit_arm)
    config = base.Config(
        train_start="2026-01-01", train_end="2026-04-03", eval_end="2026-04-04",
        min_train_rows=20, min_positive_rows=4, max_features=4,
    )
    oof, report, final, _, _ = overlay._fit_group_daily_episode_arm(
        local,
        local.iloc[0:0].copy(),
        ["observable"],
        "episode_lgbm_pooled_reliability",
        config,
        7,
        target_column=TOP10_PERIOD_EVENT_TARGET,
        period_control_mode="timestamp",
        market_train=market,
        market_valid=market.iloc[0:0].copy(),
        pooled_train=pooled,
        pooled_reliability_shrinkage_k=2.0,
    )
    assert not oof.empty
    assert final is None
    assert captured
    assert all(arm_name == "episode_lgbm" for arm_name, _ in captured)
    assert any("mechanism_reliability_risk" in features for _, features in captured)
    assert report["pooled_reliability_status"].eq("ok").any()


def test_market_period_target_requires_multiple_local_adverse_cells() -> None:
    days = pd.to_datetime(
        ["2026-01-02"] * 3 + ["2026-01-03"] * 2, utc=True
    )
    frame = pd.DataFrame(
        {
            "day": days,
            "side_name": ["short", "long", "short", "short", "long"],
            "archetype_policy_key": ["a", "b", "c", "a", "b"],
            "parent_rank_v9": [0.95, 0.93, 0.91, 0.94, 0.92],
            base.EVENT: [1, 1, 0, 1, 0],
        }
    )
    result = _attach_market_period_targets(frame, top10_floor=0.90)
    result = _attach_market_episode_phases(result)
    assert result["market_adverse_period"].tolist() == [1, 1, 1, 0, 0]
    assert result[TOP10_MARKET_PERIOD_EVENT_TARGET].tolist() == [1, 1, 1, 0, 0]
    assert result["market_episode_phase"].tolist()[:3] == ["onset"] * 3


def test_event_intervention_report_does_not_credit_normal_reallocation() -> None:
    day = pd.Timestamp("2026-01-02", tz="UTC")
    frame = pd.DataFrame(
        {
            "day": [day, day, day],
            "side_name": ["short", "short", "short"],
            "archetype_policy_key": ["a", "a", "b"],
            base.EVENT: [1, 1, 0],
            "ev_after_1pct": [-0.02, 0.01, 0.03],
            "clean_exec": [0.0, 1.0, 1.0],
        }
    )
    report, summary = _event_intervention_report(
        frame,
        parent_rank=[0.95, 0.96, 0.95],
        adjusted_rank=[0.95, 0.96, 0.89],
        flagged=[False, False, True],
        top10_floor=0.90,
    )
    assert summary["event_cells"] == 1
    assert summary["event_cells_intervened"] == 0
    assert report.loc[report[base.EVENT].eq(1), "intervened"].iloc[0] == False


def test_oos_validation_rejects_candidate_without_untouched_episode_action() -> None:
    candidate = pd.DataFrame(
        [{
            "side_name": "short",
            "archetype_policy_key": "short_mixed_clean_path",
            "model_arm": "episode_lgbm",
            "model_target": TOP10_PERIOD_EVENT_TARGET,
        }]
    )
    events = pd.DataFrame(
        {
            "side_name": ["short", "short"],
            "archetype_policy_key": ["short_mixed_clean_path"] * 2,
            "adverse_calendar_cell": [1, 1],
            "parent_rows": [10, 12],
            "overlay_rows": [10, 12],
            "intervened": [False, False],
            "ev_delta": [0.0, 0.0],
            "clean_precision_delta": [0.0, 0.0],
        }
    )
    result = _validate_oos_candidates(
        candidate,
        events,
        minimum_event_cells=2,
        minimum_intervention_recall=0.20,
        minimum_improved_cells=2,
    )
    assert result.loc[0, "oos_validated"] == False
    assert result.loc[0, "oos_validation_status"] == "fail_insufficient_untouched_intervention"


def test_oos_validation_requires_episode_improvement_not_normal_reallocation() -> None:
    candidate = pd.DataFrame(
        [{
            "side_name": "short",
            "archetype_policy_key": "short_mixed_clean_path",
            "model_arm": "episode_lgbm",
            "model_target": TOP10_PERIOD_EVENT_TARGET,
        }]
    )
    events = pd.DataFrame(
        {
            "side_name": ["short", "short", "short"],
            "archetype_policy_key": ["short_mixed_clean_path"] * 3,
            "adverse_calendar_cell": [1, 1, 0],
            "parent_rows": [10, 12, 20],
            "overlay_rows": [9, 11, 18],
            "intervened": [True, True, True],
            "ev_delta": [0.002, 0.003, 0.01],
            "clean_precision_delta": [0.01, 0.02, 0.0],
        }
    )
    result = _validate_oos_candidates(
        candidate,
        events,
        minimum_event_cells=2,
        minimum_intervention_recall=0.20,
        minimum_improved_cells=2,
    )
    assert result.loc[0, "oos_validated"] == True
    assert result.loc[0, "oos_event_cells_intervened"] == 2
    assert result.loc[0, "oos_event_cells_improved"] == 2


def test_episode_intervention_diagnostics_requires_episode_action() -> None:
    days = pd.to_datetime(
        ["2026-01-02", "2026-01-02", "2026-01-03"], utc=True
    )
    frame = pd.DataFrame(
        {
            "day": days,
            "parent_rank_v9": [0.95, 0.93, 0.94],
            "ev_after_1pct": [-0.02, 0.01, -0.01],
            base.EVENT: [1, 1, 1],
        }
    )
    result = _episode_intervention_diagnostics(
        frame,
        adjusted_rank=[0.89, 0.93, 0.94],
        top10_floor=0.90,
    )
    assert result["oof_event_cells"] == 2
    assert result["oof_event_cells_intervened"] == 1
    assert result["oof_event_cells_improved"] == 1
    assert result["oof_event_intervention_recall"] == 0.5


def test_calendar_returns_timestamp_mechanism_aggregation() -> None:
    day = pd.Timestamp("2026-01-02", tz="UTC")
    frame = pd.DataFrame(
        {
            "__ts__": [day, day],
            "side_name": ["short", "short"],
            "archetype_policy_key": ["a", "a"],
            "parent_rank_v9": [0.95, 0.92],
            "ev_after_1pct": [-0.01, 0.01],
            "clean_exec": [0.0, 1.0],
            base.EVENT: [1, 1],
            "negative_breadth_pct": [2.0, 3.0],
        }
    )
    result = _calendar(frame, frame, "parent_rank_v9")
    assert len(result) == 1
    assert result.loc[0, "selected_rows"] == 2
    assert result.loc[0, "adverse_calendar_cell"] == 1


def test_period_state_uses_top20_context_but_top10_period_label() -> None:
    timestamps = pd.to_datetime(
        ["2026-01-02 00:00", "2026-01-02 00:00", "2026-01-02 00:15"],
        utc=True,
    )
    context = pd.DataFrame(
        {
            "__ts__": timestamps,
            "day": timestamps.floor("D"),
            "observable": [1.0, 9.0, 5.0],
            "parent_rank_v9": [0.95, 0.82, 0.92],
            "score_meta_base_soft_label": [0.7, 0.2, 0.8],
            "hit_probability": [0.8, 0.3, 0.9],
            "ev_after_1pct": [0.02, -0.02, 0.01],
            "clean_exec": [1.0, 0.0, 1.0],
            TOP10_PERIOD_EVENT_TARGET: [1, 0, 0],
        }
    )
    decision = context.loc[context["parent_rank_v9"].ge(0.90)].copy()
    state = _period_state_frame(
        context,
        decision,
        ["observable", *PERIOD_STATE_FEATURES],
        target_column=TOP10_PERIOD_EVENT_TARGET,
        event_column=TOP10_PERIOD_EVENT_TARGET,
    )

    first = state.loc[state["__ts__"].eq(timestamps[0])].iloc[0]
    assert first["observable"] == 5.0
    assert first["period_context_rows"] == 2
    assert first["period_parent_rank_q90"] > 0.9
    assert first[TOP10_PERIOD_EVENT_TARGET] == 1
    assert len(state) == 2


def test_daily_episode_state_has_one_label_and_one_signature_per_day() -> None:
    day = pd.Timestamp("2026-01-02", tz="UTC")
    rows = pd.DataFrame(
        {
            "__ts__": [day + pd.Timedelta(minutes=15), day + pd.Timedelta(hours=6)],
            "day": [day, day],
            "parent_rank_v9": [0.96, 0.94],
            "ev_after_1pct": [-0.01, 0.01],
            "clean_exec": [0.0, 1.0],
            TOP10_PERIOD_EVENT_TARGET: [1, 1],
        }
    )
    market = pd.DataFrame(
        {
            "__ts__": [day],
            "negative_breadth_pct": [0.71],
        }
    )
    state = _daily_episode_state_frame(
        rows,
        ["negative_breadth_pct"],
        target_column=TOP10_PERIOD_EVENT_TARGET,
        event_column=TOP10_PERIOD_EVENT_TARGET,
        market_reference=market,
    )
    assert len(state) == 1
    assert state[TOP10_PERIOD_EVENT_TARGET].tolist() == [1]
    assert state["negative_breadth_pct"].iloc[0] == pytest.approx(0.71)


def test_daily_episode_state_normalizes_arrow_microsecond_timestamps() -> None:
    """Causal as-of joins must not depend on parquet timestamp resolution."""

    day = pd.Timestamp("2026-01-02", tz="UTC")
    rows = pd.DataFrame(
        {
            "__ts__": [day + pd.Timedelta(minutes=15)],
            "day": [day],
            "parent_rank_v9": [0.96],
            "ev_after_1pct": [0.01],
            "clean_exec": [1.0],
            TOP10_PERIOD_EVENT_TARGET: [0],
        }
    )
    market = pd.DataFrame(
        {
            "__ts__": pd.Series([day]).astype("datetime64[us, UTC]"),
            "negative_breadth_pct": [0.71],
        }
    )
    state = _daily_episode_state_frame(
        rows,
        ["negative_breadth_pct"],
        target_column=TOP10_PERIOD_EVENT_TARGET,
        event_column=TOP10_PERIOD_EVENT_TARGET,
        market_reference=market,
    )
    assert state["negative_breadth_pct"].iloc[0] == pytest.approx(0.71)


def test_global_market_context_broadcasts_daily_scores_without_outcomes() -> None:
    day = pd.Timestamp("2026-01-02", tz="UTC")
    market = pd.DataFrame(
        {
            "__ts__": [day, day + pd.Timedelta(hours=12), day + pd.Timedelta(days=1)],
            "negative_breadth_pct": [0.4, 0.5, 0.2],
        }
    )
    scores = pd.DataFrame(
        {
            "day": [day],
            GLOBAL_MARKET_EPISODE_RISK: [0.8],
            GLOBAL_MARKET_EPISODE_RISK_PCT: [0.95],
        }
    )
    result = _attach_daily_market_context(market, scores)
    assert result[GLOBAL_MARKET_EPISODE_RISK].iloc[:2].tolist() == pytest.approx([0.8, 0.8])
    assert pd.isna(result[GLOBAL_MARKET_EPISODE_RISK].iloc[2])
    assert "ev_after_1pct" not in result.columns
    assert "clean_exec" not in result.columns


def test_forward_global_market_context_is_daily_and_observable_only(tmp_path) -> None:
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    day = pd.Timestamp("2026-01-02", tz="UTC")
    frame = pd.DataFrame(
        {
            "__ts__": [day, day + pd.Timedelta(hours=12), day + pd.Timedelta(days=1)],
            "negative_breadth_pct": [0.2, 0.8, 0.3],
        }
    )
    bundle = {
        "features": ["negative_breadth_pct"],
        "robust": _IdentityTransform(),
        "model": _ConstantRiskModel(),
        "reference": pd.Series([0.1, 0.5, 0.9]).to_numpy(),
    }
    result, report = _materialize_global_market_episode_context(
        frame, [(overlay, bundle)]
    )
    assert report["enabled"] is True
    assert result[GLOBAL_MARKET_EPISODE_RISK].tolist() == pytest.approx([0.73] * 3)
    assert result[GLOBAL_MARKET_EPISODE_RISK_PCT].notna().all()
    assert "ev_after_1pct" not in result.columns


def test_global_market_episode_context_is_fitted_on_daily_chronological_states() -> None:
    train_days = pd.date_range("2025-04-01", "2026-03-31", freq="D", tz="UTC")
    valid_days = pd.date_range("2026-04-01", periods=14, freq="D", tz="UTC")

    def _rows(days: pd.DatetimeIndex) -> pd.DataFrame:
        event = (pd.Series(range(len(days))).mod(11).eq(0).to_numpy("int8"))
        return pd.DataFrame(
            {
                "__ts__": days,
                "parent_rank_v9": 0.95,
                "ev_after_1pct": event * -0.02 + (1 - event) * 0.01,
                "clean_exec": 1 - event,
                "market_adverse_period": event,
            }
        )

    train = _rows(train_days)
    valid = _rows(valid_days)
    market_train = train.loc[:, ["__ts__"]].copy()
    market_train["negative_breadth_pct"] = (
        train["market_adverse_period"] * 0.75
        + 0.05
        + (pd.Series(range(len(train))).mod(7).to_numpy() / 100.0)
    )
    market_valid = valid.loc[:, ["__ts__"]].copy()
    market_valid["negative_breadth_pct"] = (
        valid["market_adverse_period"] * 0.75
        + 0.05
        + (pd.Series(range(len(valid))).mod(7).to_numpy() / 100.0)
    )
    config = base.Config(train_end="2026-04-01", eval_end="2026-05-01", max_features=4)
    augmented_train, augmented_valid, oof, bundle, report = _fit_global_market_episode_context(
        train,
        valid,
        ["negative_breadth_pct"],
        config,
        market_train=market_train,
        market_valid=market_valid,
        seed=7,
    )
    assert not oof.empty
    assert bundle is not None
    assert GLOBAL_MARKET_EPISODE_RISK in augmented_train.columns
    assert GLOBAL_MARKET_EPISODE_RISK_PCT in augmented_valid.columns
    assert report["stage"].eq("oof").any()


def test_side_market_context_is_daily_and_never_crosses_side_outcomes() -> None:
    train_days = pd.date_range("2025-04-01", "2026-03-31", freq="D", tz="UTC")
    valid_days = pd.date_range("2026-04-01", periods=14, freq="D", tz="UTC")

    def _rows(days: pd.DatetimeIndex) -> pd.DataFrame:
        day_index = pd.Series(range(len(days)))
        long_event = day_index.mod(11).eq(0).to_numpy("int8")
        short_event = day_index.mod(13).eq(0).to_numpy("int8")
        return pd.concat(
            [
                pd.DataFrame({
                    "__ts__": days, "side_name": "long", "parent_rank_v9": 0.95,
                    "ev_after_1pct": long_event * -0.02 + (1 - long_event) * 0.01,
                    "clean_exec": 1 - long_event, base.SIDE_EVENT: long_event,
                }),
                pd.DataFrame({
                    "__ts__": days, "side_name": "short", "parent_rank_v9": 0.95,
                    "ev_after_1pct": short_event * -0.02 + (1 - short_event) * 0.01,
                    "clean_exec": 1 - short_event, base.SIDE_EVENT: short_event,
                }),
            ],
            ignore_index=True,
        )

    train = _rows(train_days)
    valid = _rows(valid_days)
    market_train = pd.DataFrame({
        "__ts__": train_days,
        "negative_breadth_pct": (
            pd.Series(range(len(train_days))).mod(11).eq(0).astype(float)
            + 0.05
            + pd.Series(range(len(train_days))).mod(7).to_numpy() / 100.0
        ),
    })
    market_valid = pd.DataFrame({
        "__ts__": valid_days,
        "negative_breadth_pct": (
            pd.Series(range(len(valid_days))).mod(11).eq(0).astype(float)
            + 0.05
            + pd.Series(range(len(valid_days))).mod(7).to_numpy() / 100.0
        ),
    })
    config = base.Config(train_end="2026-04-01", eval_end="2026-05-01", max_features=4)
    augmented_train, augmented_valid, oof, bundles, report = _fit_side_market_episode_context(
        train, valid, ["negative_breadth_pct"], config,
        market_train=market_train, market_valid=market_valid, seed=17,
    )
    assert set(bundles).issubset({"long", "short"})
    assert not oof.empty
    assert oof.groupby(["day", "side_name"], observed=True).size().eq(1).all()
    assert augmented_train[SIDE_MARKET_EPISODE_RISK].notna().any()
    assert augmented_valid[SIDE_MARKET_EPISODE_RISK_PCT].notna().any()
    assert report["side_name"].isin(["long", "short"]).all()


def test_forward_side_market_context_is_broadcast_only_to_matching_side(tmp_path) -> None:
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    day = pd.Timestamp("2026-01-02", tz="UTC")
    frame = pd.DataFrame(
        {
            "__ts__": [day, day + pd.Timedelta(hours=12)],
            "side_name": ["long", "short"],
            "negative_breadth_pct": [0.2, 0.8],
        }
    )
    bundle = {
        "features": ["negative_breadth_pct"],
        "robust": _IdentityTransform(),
        "model": _ConstantRiskModel(),
        "reference": pd.Series([0.1, 0.5, 0.9]).to_numpy(),
    }
    result, report = _materialize_side_market_episode_context(
        frame, [(overlay, "long", bundle), (overlay, "short", bundle)]
    )
    assert report["enabled"] is True
    assert result[SIDE_MARKET_EPISODE_RISK].tolist() == pytest.approx([0.73, 0.73])
    assert result[SIDE_MARKET_EPISODE_RISK_PCT].notna().all()
    assert "ev_after_1pct" not in result.columns


def test_episode_trajectory_features_use_only_values_before_each_horizon() -> None:
    timestamps = pd.to_datetime(
        ["2026-01-01 00:00", "2026-01-01 06:00", "2026-01-01 18:00", "2026-01-02 00:00"],
        utc=True,
    )
    state = pd.DataFrame(
        {
            "__ts__": timestamps,
            "negative_breadth_pct": [1.0, 3.0, 6.0, 7.0],
        }
    )
    result = _add_episode_trajectory_features(state)
    at_24h = result.loc[result["__ts__"].eq(timestamps[3])].iloc[0]
    assert at_24h["episode_traj__negative_breadth_pct__delta_6h"] == 1.0
    assert at_24h["episode_traj__negative_breadth_pct__delta_24h"] == 6.0
    assert abs(at_24h["episode_traj__negative_breadth_pct__velocity_accel_6h_vs_24h"] + 0.08333333) < 1e-6


def test_episode_trajectory_score_state_can_use_fit_history_without_future_values() -> None:
    fit_ts = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 06:00"], utc=True)
    score_ts = pd.to_datetime(["2026-01-01 12:00"], utc=True)
    fit = pd.DataFrame({"__ts__": fit_ts, "negative_breadth_pct": [1.0, 4.0]})
    score = pd.DataFrame({"__ts__": score_ts, "negative_breadth_pct": [10.0]})
    result = _add_episode_trajectory_features(score, history=fit)
    assert result.loc[0, "episode_traj__negative_breadth_pct__delta_6h"] == 6.0
    assert pd.isna(result.loc[0, "episode_traj__negative_breadth_pct__delta_24h"])


def test_episode_trajectory_rejects_stale_local_candidate_history() -> None:
    timestamps = pd.to_datetime(["2026-01-01 00:00", "2026-01-02 00:00"], utc=True)
    state = pd.DataFrame({"__ts__": timestamps, "negative_breadth_pct": [1.0, 7.0]})
    result = _add_episode_trajectory_features(state)
    assert pd.isna(result.loc[1, "episode_traj__negative_breadth_pct__delta_6h"])


def test_episode_trajectory_uses_full_market_clock_not_local_candidate_activity() -> None:
    local_timestamps = pd.to_datetime(["2026-01-01 00:00", "2026-01-02 00:00"], utc=True)
    reference_timestamps = pd.to_datetime(
        ["2026-01-01 00:00", "2026-01-01 06:00", "2026-01-01 18:00", "2026-01-02 00:00"], utc=True
    )
    state = pd.DataFrame(
        {"__ts__": local_timestamps, "negative_breadth_pct": [1.0, 7.0]}
    )
    reference = pd.DataFrame(
        {"__ts__": reference_timestamps, "negative_breadth_pct": [1.0, 4.0, 4.0, 10.0]}
    )
    result = _add_episode_trajectory_features(
        _attach_trajectory_reference(state, reference), history=reference
    )
    assert result.loc[1, "negative_breadth_pct"] == 10.0
    assert result.loc[1, "episode_traj__negative_breadth_pct__delta_6h"] == 6.0


def test_period_state_forward_materialization_matches_top20_context(tmp_path) -> None:
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "manifest.json").write_text(
        json.dumps({"period_state_contract": {"context_floor": 0.80}})
    )
    pd.DataFrame(
        [{
            "side_name": "short",
            "archetype_policy_key": "short_default_clean_path",
            "model_arm": "test",
        }]
    ).to_csv(overlay / "accepted_overlays.csv", index=False)
    joblib.dump(
        {
            "features": [
                "observable", "period_context_rows", "period_parent_rank_q90"
            ]
        },
        overlay / "model__test__short__short_default_clean_path.joblib",
    )
    timestamps = pd.to_datetime(
        ["2026-01-02 00:00", "2026-01-02 00:00", "2026-01-02 00:15"],
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "__ts__": timestamps,
            "__symbol__": ["A", "B", "A"],
            "side_name": ["short"] * 3,
            "archetype_policy_key": ["short_default_clean_path"] * 3,
            "parent_rank_v9": [0.95, 0.82, 0.92],
            "observable": [1.0, 9.0, 5.0],
        }
    )
    result, report = _materialize_period_state_features(frame, [overlay])
    first = result.loc[result["__symbol__"].eq("A")].iloc[0]
    assert first["observable"] == 5.0
    assert first["period_context_rows"] == 2
    assert first["period_parent_rank_q90"] > 0.9
    assert report["enabled"] is True


def test_period_state_forward_keeps_same_side_daily_context_separate(tmp_path) -> None:
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "manifest.json").write_text(
        json.dumps({"period_state_contract": {"context_floor": 0.80}, "episode_state_granularity": "daily_open"})
    )
    pd.DataFrame([
        {
            "side_name": "short",
            "archetype_policy_key": "short_default_clean_path",
            "model_arm": "test",
        }
    ]).to_csv(overlay / "accepted_overlays.csv", index=False)
    joblib.dump(
        {"features": [SIDE_MARKET_EPISODE_RISK, SIDE_MARKET_EPISODE_RISK_PCT]},
        overlay / "model__test__short__short_default_clean_path.joblib",
    )
    day = pd.Timestamp("2026-01-02", tz="UTC")
    frame = pd.DataFrame(
        {
            "__ts__": [day, day],
            "__symbol__": ["A", "B"],
            "side_name": ["short", "long"],
            "archetype_policy_key": ["short_default_clean_path", "other"],
            "parent_rank_v9": [0.95, 0.95],
            SIDE_MARKET_EPISODE_RISK: [0.82, 0.19],
            SIDE_MARKET_EPISODE_RISK_PCT: [0.88, 0.21],
        }
    )
    result, report = _materialize_period_state_features(frame, [overlay])
    selected = result.loc[result["side_name"].eq("short")].iloc[0]
    assert selected[SIDE_MARKET_EPISODE_RISK] == pytest.approx(0.82)
    assert selected[SIDE_MARKET_EPISODE_RISK_PCT] == pytest.approx(0.88)
    assert report["groups"][0]["side_context_features"] == [
        SIDE_MARKET_EPISODE_RISK,
        SIDE_MARKET_EPISODE_RISK_PCT,
    ]


def test_period_state_forward_materializes_causal_trajectory(tmp_path) -> None:
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "manifest.json").write_text(
        json.dumps({"period_state_contract": {"context_floor": 0.80}})
    )
    pd.DataFrame(
        [{
            "side_name": "short",
            "archetype_policy_key": "short_default_clean_path",
            "model_arm": "test",
        }]
    ).to_csv(overlay / "accepted_overlays.csv", index=False)
    feature = "episode_traj__negative_breadth_pct__delta_6h"
    joblib.dump({"features": [feature]}, overlay / "model__test__short__short_default_clean_path.joblib")
    timestamps = pd.to_datetime(["2026-01-02 00:00", "2026-01-02 06:00"], utc=True)
    frame = pd.DataFrame(
        {
            "__ts__": timestamps,
            "__symbol__": ["A", "A"],
            "side_name": ["short", "short"],
            "archetype_policy_key": ["short_default_clean_path"] * 2,
            "parent_rank_v9": [0.95, 0.95],
            "negative_breadth_pct": [1.0, 4.0],
        }
    )
    result, report = _materialize_period_state_features(frame, [overlay])
    assert pd.isna(result.loc[0, feature])
    assert result.loc[1, feature] == 3.0
    assert report["groups"][0]["trajectory_features"] == [feature]


def test_daily_open_period_state_uses_one_market_signature_for_the_day(tmp_path) -> None:
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "manifest.json").write_text(
        json.dumps(
            {
                "period_state_contract": {"context_floor": 0.80},
                "episode_state_granularity": "daily_open",
            }
        )
    )
    pd.DataFrame(
        [{
            "side_name": "short",
            "archetype_policy_key": "short_default_clean_path",
            "model_arm": "test",
        }]
    ).to_csv(overlay / "accepted_overlays.csv", index=False)
    joblib.dump(
        {"features": ["observable"]},
        overlay / "model__test__short__short_default_clean_path.joblib",
    )
    timestamps = pd.to_datetime(["2026-01-02 00:00", "2026-01-02 06:00"], utc=True)
    frame = pd.DataFrame(
        {
            "__ts__": timestamps,
            "__symbol__": ["A", "A"],
            "side_name": ["short", "short"],
            "archetype_policy_key": ["short_default_clean_path"] * 2,
            "parent_rank_v9": [0.95, 0.95],
            "observable": [1.0, 9.0],
        }
    )
    result, report = _materialize_period_state_features(frame, [overlay])
    assert result["observable"].tolist() == [1.0, 1.0]
    assert report["groups"][0]["state_granularity"] == "daily_open"


def test_frozen_negative_panel_is_timestamp_indexed_and_exact(tmp_path) -> None:
    timestamps = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    panel_path = tmp_path / "negative_panel.parquet"
    pd.DataFrame(
        {
            "negative_breadth_pct": [0.1, 0.2, 0.3],
            "flush_recovery_state": [0.0, 0.5, 1.0],
        },
        index=timestamps,
    ).to_parquet(panel_path)

    result = _load_frozen_negative_panel(
        panel_path,
        ["negative_breadth_pct", "flush_recovery_state"],
        start=timestamps[1],
        end=timestamps[2],
    )

    assert result["__ts__"].tolist() == timestamps[1:].tolist()
    assert result["negative_breadth_pct"].tolist() == [0.2, 0.3]
    assert result["flush_recovery_state"].tolist() == [0.5, 1.0]
