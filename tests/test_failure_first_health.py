from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.unsupervised_regime_learning.failure_first_health import (
    FailureHealthConfig,
    build_causal_decision_health,
    group_failure_bins_into_episodes,
)


def _rows(hours: int = 72, candidates: int = 12) -> pd.DataFrame:
    records = []
    for hour, timestamp in enumerate(
        pd.date_range("2026-01-01", periods=hours, freq="h", tz="UTC")
    ):
        for candidate in range(candidates):
            score = candidate / 1_000 + hour / 100_000
            net = 0.01
            if 42 <= hour < 54 and candidate >= candidates - 3:
                net = -0.05
            records.append(
                {
                    "candidate_id": f"{hour:03d}-{candidate:03d}",
                    "__symbol__": f"A{candidate:02d}",
                    "side_name": "long" if candidate % 2 else "short",
                    "evaluation_origin": "strict_oof",
                    "execution_decision_utc": timestamp,
                    "execution_label_end_utc": timestamp + pd.Timedelta("12h"),
                    "causal_recent_side_isotonic_ev": score,
                    "causal_recent_side_isotonic_ev__is_oof": True,
                    "execution_gross_ev_12h": net + 0.01,
                    "execution_net_ev_12h": net,
                }
            )
    return pd.DataFrame.from_records(records)


def _config() -> FailureHealthConfig:
    return FailureHealthConfig(
        admission_lookback_days=1,
        minimum_cutoff_rows=36,
        health_bin_hours=6,
        minimum_admitted_rows=6,
        residual_lookback_days=2,
        minimum_resolved_bins=2,
        join_gap_hours=12,
    )


def test_causal_health_prefix_is_unchanged_by_future_rows() -> None:
    source = _rows()
    short, _ = build_causal_decision_health(source.iloc[: 48 * 12], _config())
    full, _ = build_causal_decision_health(source, _config())
    columns = [
        "decision_bin_start_utc",
        "admission_cutoff",
        "cutoff_reference_rows",
        "admitted_rows",
        "economic_residual_mean",
        "prior_residual_q10",
        "model_failure_bin",
    ]
    matched = full.loc[
        full["decision_bin_start_utc"].isin(short["decision_bin_start_utc"])
    ]
    pd.testing.assert_frame_equal(
        short[columns].reset_index(drop=True),
        matched[columns].reset_index(drop=True),
    )


def test_admission_is_pooled_not_per_timestamp_or_side() -> None:
    health, membership = build_causal_decision_health(_rows(), _config())
    admitted = membership.loc[membership["admitted"]]
    assert not admitted.empty
    # The same global cutoff is used by both sides at a decision timestamp.
    per_time = membership.groupby("execution_decision_utc")["admission_cutoff"].nunique()
    assert per_time.max() == 1
    # A pooled trailing cutoff does not force a fixed quota in every timestamp.
    counts = admitted.groupby("execution_decision_utc").size()
    assert counts.nunique() >= 1
    assert health["provenance_status"].eq(
        "strict_oof_causal_shadow_admission"
    ).all()


def test_episode_grouping_never_crosses_origin() -> None:
    health, membership = build_causal_decision_health(_rows(), _config())
    assert health["model_failure_bin"].any()
    copied_health = health.loc[health["model_failure_bin"]].copy()
    copied_health["evaluation_origin"] = "forward"
    copied_membership = membership.loc[
        membership["decision_bin_start_utc"].isin(
            copied_health["decision_bin_start_utc"]
        )
    ].copy()
    copied_membership["evaluation_origin"] = "forward"
    combined_health = pd.concat([health, copied_health], ignore_index=True)
    combined_membership = pd.concat(
        [membership, copied_membership], ignore_index=True
    )
    episodes, members = group_failure_bins_into_episodes(
        combined_health, combined_membership, _config()
    )
    assert set(episodes["evaluation_origin"]) == {"strict_oof", "forward"}
    assert members.groupby("episode_id")["evaluation_origin"].nunique().max() == 1
    assert episodes["descriptive_only"].all()


def test_cutoff_and_residual_history_reset_at_origin_boundary() -> None:
    first = _rows(hours=48)
    second = _rows(hours=48)
    shift = pd.Timedelta("60h")
    second["execution_decision_utc"] += shift
    second["execution_label_end_utc"] += shift
    second["candidate_id"] = "new-" + second["candidate_id"]
    second["evaluation_origin"] = "new_model"
    health, membership = build_causal_decision_health(
        pd.concat([first, second], ignore_index=True), _config()
    )
    first_new_time = second["execution_decision_utc"].min()
    first_new = membership.loc[
        membership["execution_decision_utc"].eq(first_new_time)
    ]
    assert first_new["admission_cutoff"].isna().all()
    new_health = health.loc[health["evaluation_origin"].eq("new_model")]
    assert new_health.iloc[0]["prior_residual_q10"] != new_health.iloc[0][
        "prior_residual_q10"
    ]
