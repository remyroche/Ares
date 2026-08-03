from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.unsupervised_regime_learning.failure_first_hourly import (
    FailureFirstHourlyConfig,
    build_contiguous_failure_episodes,
    build_failure_episode_profiles,
    build_hourly_failure_calendar,
    build_hourly_state_transition_labels,
    extract_episode_windows,
    fit_failure_episode_taxonomy,
    select_pooled_global_top_k,
    validate_inference_feature_columns,
)


def _candidate_frame() -> pd.DataFrame:
    stamp = pd.Timestamp("2026-07-01T00:00:00Z")
    return pd.DataFrame(
        {
            "candidate_id": ["b", "a", "c", "d"],
            "__ts__": [stamp, stamp, stamp + pd.Timedelta(hours=1), stamp + pd.Timedelta(hours=1)],
            "side_name": ["long", "short", "long", "short"],
            "score": [0.90, 0.90, 0.99, 0.10],
            "execution_net_ev_12h": [-0.01, -0.02, 0.02, 0.01],
            "execution_label_end_utc": [stamp + pd.Timedelta(hours=12)] * 4,
        }
    )


def test_global_topk_is_not_per_timestamp_or_side_and_ties_use_candidate_id() -> None:
    frame = _candidate_frame()
    selected = select_pooled_global_top_k(frame, score_col="score", fraction=0.5)
    assert selected.sum() == 2
    # Global top two are c (0.99) and tie-broken a (0.90), not one per timestamp.
    assert set(frame.loc[selected, "candidate_id"]) == {"a", "c"}


def test_failure_calendar_keeps_outcomes_unavailable_until_every_selected_label_resolves() -> None:
    frame = _candidate_frame().iloc[:2].copy()
    frame.loc[1, "execution_label_end_utc"] = pd.NaT
    calendar = build_hourly_failure_calendar(frame, pd.Series([True, True]), config=FailureFirstHourlyConfig())
    assert len(calendar) == 2  # side-aware calendar has one row per selected side/hour
    # The fully resolved local cell is labelled; the other is unavailable,
    # rather than being silently treated as a non-failure.
    assert calendar["target__failure"].notna().sum() == 1
    assert calendar["target__failure"].isna().sum() == 1
    assert calendar["target__failure_label_resolution_utc"].isna().sum() == 1


def test_episode_windows_use_requested_offsets_and_keep_profiles_expost() -> None:
    stamp = pd.Timestamp("2026-07-01T00:00:00Z")
    calendar = pd.DataFrame(
        {"__ts__": [stamp, stamp + pd.Timedelta(hours=1), stamp + pd.Timedelta(hours=4)], "side_name": ["long"] * 3, "target__failure": [1.0, 1.0, 1.0], "expost__selected_net_ev_mean": [-.01, -.02, -.01]}
    )
    episodes = build_contiguous_failure_episodes(calendar, gap_hours=0)
    assert len(episodes) == 2
    state = pd.DataFrame({"__ts__": pd.date_range(stamp - pd.Timedelta(hours=48), stamp + pd.Timedelta(hours=12), freq="h", tz="UTC"), "side_name": "long", "state__volatility": np.arange(61, dtype=float)})
    windows = extract_episode_windows(state, episodes.iloc[:1], feature_columns=["state__volatility"])
    assert set(windows["expost__window_offset_hours"]) == {-48, -24, -12, -6, -3, 0, 3, 6, 12}
    profiles = build_failure_episode_profiles(windows, feature_columns=["state__volatility"])
    assert all(column.startswith("expost__") for column in profiles.columns)


def test_taxonomy_uses_failure_profiles_only_and_is_deterministic() -> None:
    profiles = pd.DataFrame(
        {
            "expost__episode_id": [f"failure-{idx}" for idx in range(8)],
            "expost__profile__state__funding__h+0": [-3, -2.8, -3.1, -2.9, 3, 2.9, 3.2, 2.8],
            "expost__profile__state__oi__h+3": [-2.9, -3.2, -2.8, -3.0, 2.8, 3.1, 3.0, 3.2],
        }
    )
    first = fit_failure_episode_taxonomy(profiles, min_clusters=2, max_clusters=2, random_state=7)
    second = fit_failure_episode_taxonomy(profiles, min_clusters=2, max_clusters=2, random_state=7)
    assert first.assignments["expost__episode_id"].tolist() == profiles["expost__episode_id"].tolist()
    assert first.assignments["expost__failure_cluster"].tolist() == second.assignments["expost__failure_cluster"].tolist()
    assert first.assignments["expost__failure_taxonomy_label"].str.startswith(("funding_transition", "leverage_repricing")).all()


def test_hourly_state_labels_are_side_local_and_future_labels_require_full_horizon() -> None:
    stamp = pd.Timestamp("2026-07-01T00:00:00Z")
    state = pd.DataFrame({"__ts__": pd.date_range(stamp, periods=5, freq="h", tz="UTC"), "side_name": "long", "state__id": [0, 0, 1, 1, 1]})
    labels = build_hourly_state_transition_labels(state, state_col="state__id", horizon_hours=3)
    first = labels.iloc[0]
    assert first["target__transition_within_3h"] == 1.0
    assert first["target__destination_state_3h"] == 1
    assert first["target__future_label_resolution_utc"] == stamp + pd.Timedelta(hours=3)
    assert labels.iloc[-1]["target__transition_within_3h"] != labels.iloc[-1]["target__transition_within_3h"]


def test_state_labels_respect_delayed_availability_and_origin_boundaries() -> None:
    stamp = pd.Timestamp("2026-07-01T00:00:00Z")
    state = pd.DataFrame(
        {
            "__ts__": [
                stamp,
                stamp + pd.Timedelta("1h"),
                stamp + pd.Timedelta("2h"),
                stamp + pd.Timedelta("3h"),
                stamp + pd.Timedelta("4h"),
            ],
            "side_name": "global",
            "evaluation_origin": ["old", "old", "new", "new", "new"],
            "state__id": ["stable", "stable", "failure", "failure", "stable"],
            "state_available_utc": [
                stamp + pd.Timedelta("12h"),
                stamp + pd.Timedelta("13h"),
                stamp + pd.Timedelta("14h"),
                stamp + pd.Timedelta("15h"),
                stamp + pd.Timedelta("16h"),
            ],
        }
    )
    early = build_hourly_state_transition_labels(
        state,
        state_col="state__id",
        state_available_col="state_available_utc",
        boundary_columns=("evaluation_origin",),
        horizon_hours=2,
        observed_through=stamp + pd.Timedelta("10h"),
    )
    assert early["target__current_state"].isna().all()
    assert early["target__transition_within_2h"].isna().all()

    resolved = build_hourly_state_transition_labels(
        state,
        state_col="state__id",
        state_available_col="state_available_utc",
        boundary_columns=("evaluation_origin",),
        horizon_hours=2,
        observed_through=stamp + pd.Timedelta("20h"),
    )
    first_new = resolved.loc[
        resolved["evaluation_origin"].eq("new")
    ].iloc[0]
    assert np.isnan(first_new["target__active_transition"])


def test_target_expost_and_future_columns_are_rejected_from_inference_features() -> None:
    assert validate_inference_feature_columns(["state__volatility", "mkt_state__breadth"]) == ("state__volatility", "mkt_state__breadth")
    for forbidden in ("target__failure", "expost__episode_id", "future__destination"):
        with pytest.raises(ValueError, match="target/expost/future"):
            validate_inference_feature_columns(["state__ok", forbidden])
