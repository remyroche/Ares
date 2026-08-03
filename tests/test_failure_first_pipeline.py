from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.unsupervised_regime_learning.failure_first_pipeline import (
    FailureFirstSufficiencyConfig,
    build_hourly_failure_state_targets,
    build_hourly_observable_state,
    choose_taxonomy_fit_cutoff,
    evaluate_failure_first_sufficiency,
    evaluate_taxonomy_bootstrap_stability,
    extract_failure_episode_windows,
    fit_frozen_failure_taxonomy,
    prepare_failure_first_sources,
)
from scripts.run_failure_first_regime_pipeline import (
    _detector_classification_report,
    _detector_promotion_gate,
    run,
)


def _sources(hours: int = 72, rows_per_hour: int = 10):
    stamps = pd.date_range("2026-01-01", periods=hours, freq="h", tz="UTC")
    decision = np.repeat(stamps.to_numpy(), rows_per_hour)
    row = np.arange(len(decision))
    candidate = [f"candidate-{index:06d}" for index in row]
    score = 0.01 + (row % rows_per_hour) / 1_000.0
    hour_index = row // rows_per_hour
    net = score + 0.02
    net[(hour_index >= 48) & (hour_index < 54)] = -0.08
    ledger = pd.DataFrame(
        {
            "candidate_id": candidate,
            "__ts__": pd.to_datetime(decision, utc=True) - pd.Timedelta("1h"),
            "__symbol__": [f"ASSET-{index % 7}" for index in row],
            "side_name": np.where(row % 2, "short", "long"),
            "execution_decision_utc": pd.to_datetime(decision, utc=True),
            "execution_label_end_utc": pd.to_datetime(decision, utc=True)
            + pd.Timedelta("12h"),
            "execution_gross_ev_12h": net + 0.01,
            "execution_net_ev_12h": net,
            "causal_recent_side_isotonic_ev": score,
            "causal_recent_side_isotonic_ev__is_oof": True,
            "causal_recent_side_isotonic_ev__is_forward_oos": False,
            "catboost__residual__without_hpo__all_features": score * 0.9,
            "evaluation_origin": "synthetic_oof",
        }
    )
    state = ledger.loc[
        :,
        [
            "candidate_id",
            "__ts__",
            "__symbol__",
            "side_name",
            "execution_decision_utc",
        ],
    ].copy()
    state["raw_state_source_utc_h0"] = (
        state["execution_decision_utc"] - pd.Timedelta("1h")
    )
    state["mkt_state__atr_slope__h0"] = np.sin(row / 19.0)
    state["mkt_state__atr_slope__h3"] = np.cos(row / 11.0)
    return ledger, state


def test_prepare_sources_is_strict_oof_h0_only_and_checks_availability() -> None:
    ledger, state = _sources()
    joined, features, audit = prepare_failure_first_sources(ledger, state)
    assert len(joined) == len(ledger)
    assert "mkt_state__atr_slope__h0" in features
    assert "mkt_state__atr_slope__h3" not in features
    assert audit["strict_oof_rows"] == len(ledger)

    broken = state.copy()
    broken.loc[0, "raw_state_source_utc_h0"] = (
        broken.loc[0, "execution_decision_utc"] + pd.Timedelta("1h")
    )
    with pytest.raises(ValueError, match="after decision"):
        prepare_failure_first_sources(ledger, broken)


def test_hourly_state_and_episode_windows_are_observable_and_exact() -> None:
    ledger, state = _sources()
    joined, features, _ = prepare_failure_first_sources(ledger, state)
    hourly, state_features = build_hourly_observable_state(
        joined, feature_columns=features
    )
    onset = hourly["execution_decision_utc"].iloc[50]
    episodes = pd.DataFrame(
        {
            "episode_id": ["episode-1"],
            "episode_onset_decision_utc": [onset],
            "episode_onset_available_utc": [onset + pd.Timedelta("18h")],
        }
    )
    windows = extract_failure_episode_windows(
        hourly, episodes, state_feature_columns=state_features
    )
    assert windows["offset_hours"].tolist() == [-48, -24, -12, -6, -3, 0, 3, 6, 12]
    assert windows.loc[windows["offset_hours"].le(0), "window_complete"].all()
    assert all(name.startswith("state__") for name in state_features)


def test_sufficiency_gate_fails_closed_for_three_episodes() -> None:
    health = pd.DataFrame(
        {
            "decision_bin_start_utc": pd.date_range(
                "2026-01-01", periods=100, freq="6h", tz="UTC"
            ),
            "model_failure_bin": [True] * 3 + [False] * 97,
        }
    )
    episodes = pd.DataFrame({"episode_id": ["a", "b", "c"]})
    windows = pd.DataFrame(
        {
            "episode_id": np.repeat(["a", "b", "c"], 9),
            "window_complete": True,
        }
    )
    gate = evaluate_failure_first_sufficiency(
        health,
        episodes,
        windows,
        profile_feature_count=30,
        config=FailureFirstSufficiencyConfig(),
    )
    assert gate["status"] == "INSUFFICIENT_SUPPORT"
    assert not gate["taxonomy_training_allowed"]
    assert not gate["detector_training_allowed"]


def test_source_join_accepts_explicit_retired_model_oos_flag() -> None:
    ledger, state = _sources(hours=12, rows_per_hour=4)
    ledger["retired_model_oos"] = False
    ledger.loc[ledger.index[:16], "retired_model_oos"] = True
    joined, _, audit = prepare_failure_first_sources(
        ledger,
        state,
        requested_market_features=["mkt_state__atr_slope__h0"],
        requested_health_features=[],
        score_valid_flag="retired_model_oos",
    )
    assert len(joined) == 16
    assert audit["strict_score_valid_flag"] == "retired_model_oos"
    assert audit["strict_oof_rows"] == 16


def test_sufficiency_gate_rejects_sparse_history_with_long_calendar_span() -> None:
    first = pd.date_range(
        "2025-01-01", periods=8, freq="6h", tz="UTC"
    )
    last = pd.date_range(
        "2025-08-01", periods=8, freq="6h", tz="UTC"
    )
    health = pd.DataFrame(
        {
            "decision_bin_start_utc": first.append(last),
            "model_failure_bin": False,
        }
    )
    gate = evaluate_failure_first_sufficiency(
        health,
        pd.DataFrame(columns=["episode_id"]),
        pd.DataFrame(columns=["episode_id", "window_complete"]),
        profile_feature_count=30,
        config=FailureFirstSufficiencyConfig(),
    )

    criteria = gate["criteria"]
    assert criteria["calendar_span_days"]["pass"]
    assert not criteria["observed_calendar_days"]["pass"]
    assert not criteria["maximum_calendar_gap_days"]["pass"]
    assert not gate["taxonomy_training_allowed"]


def test_sufficiency_gate_accepts_continuous_calendar_coverage_components() -> None:
    health = pd.DataFrame(
        {
            "decision_bin_start_utc": pd.date_range(
                "2025-01-01", periods=181 * 4, freq="6h", tz="UTC"
            ),
            "model_failure_bin": False,
        }
    )
    gate = evaluate_failure_first_sufficiency(
        health,
        pd.DataFrame(columns=["episode_id"]),
        pd.DataFrame(columns=["episode_id", "window_complete"]),
        profile_feature_count=30,
        config=FailureFirstSufficiencyConfig(),
    )

    criteria = gate["criteria"]
    assert criteria["observed_calendar_days"]["pass"]
    assert criteria["maximum_calendar_gap_days"]["pass"]


def test_frozen_taxonomy_uses_only_pre_cutoff_episodes_and_scores_later() -> None:
    episodes = pd.DataFrame(
        {
            "episode_id": [f"episode-{index:02d}" for index in range(40)],
            "episode_end_available_utc": pd.date_range(
                "2025-01-01", periods=40, freq="7d", tz="UTC"
            ),
        }
    )
    cluster = np.arange(40) % 5
    profiles = pd.DataFrame({"episode_id": episodes["episode_id"]})
    profile_columns = []
    for feature in range(10):
        name = f"expost__profile__market_feature_{feature}__onset"
        profiles[name] = cluster * 10.0 + feature + np.arange(40) / 1_000
        profile_columns.append(name)
    cutoff = choose_taxonomy_fit_cutoff(
        episodes, minimum_failure_episodes=30
    )
    bundle, assignments, selection, summary = fit_frozen_failure_taxonomy(
        profiles,
        episodes,
        profile_columns=profile_columns,
        fit_cutoff_utc=cutoff,
        method="kmeans",
        min_clusters=5,
        max_clusters=5,
        minimum_cluster_episodes=5,
        random_state=7,
    )
    assert len(bundle.train_episode_ids) == 30
    assert len(assignments) == 40
    assert selection["expost__cluster_support_pass"].all()
    assert summary["expost__cluster_episode_count"].min() >= 5
    assert (
        pd.to_datetime(assignments["taxonomy_label_available_utc"], utc=True)
        >= cutoff
    ).all()
    stability = evaluate_taxonomy_bootstrap_stability(
        bundle, profiles, repetitions=20
    )
    assert stability["repetitions"] == 20
    assert stability["train_episodes"] == 30
    assert 0.0 <= stability["median_adjusted_rand_index"] <= 1.0


def test_failure_state_targets_keep_taxonomy_availability_explicit() -> None:
    starts = pd.date_range("2026-01-01", periods=5, freq="6h", tz="UTC")
    health = pd.DataFrame(
        {
            "decision_bin_start_utc": starts,
            "bin_available_utc": starts + pd.Timedelta("18h"),
            "evaluation_origin": "origin-a",
            "model_failure_bin": [False, False, True, False, False],
        }
    )
    membership = pd.DataFrame(
        {
            "episode_id": ["episode-a"],
            "decision_bin_start_utc": [starts[2]],
            "evaluation_origin": ["origin-a"],
        }
    )
    taxonomy_available = starts[2] + pd.Timedelta("30h")
    assignments = pd.DataFrame(
        {
            "episode_id": ["episode-a"],
            "expost__failure_taxonomy_label": [
                "volatility_expansion__elevated__c00"
            ],
            "taxonomy_label_available_utc": [taxonomy_available],
        }
    )
    targets = build_hourly_failure_state_targets(
        health, membership, assignments
    )
    failure = targets.loc[
        targets["target__current_failure_state"].eq(
            "volatility_expansion__elevated__c00"
        )
    ]
    assert not failure.empty
    assert (
        pd.to_datetime(
            failure["target__current_state_label_resolution_utc"], utc=True
        )
        >= taxonomy_available
    ).all()
    assert (
        pd.to_datetime(targets["transition_label_available_at"], utc=True)
        >= pd.to_datetime(
            targets["target__current_state_label_resolution_utc"], utc=True
        )
    ).all()


def test_runner_publishes_descriptive_artifacts_but_skips_training(
    tmp_path: Path,
) -> None:
    ledger, state = _sources()
    ledger_path = tmp_path / "ledger.parquet"
    state_path = tmp_path / "state.parquet"
    output = tmp_path / "output"
    ledger.to_parquet(ledger_path, index=False)
    state.to_parquet(state_path, index=False)
    args = argparse.Namespace(
        ledger=ledger_path,
        state_source=state_path,
        rich_context=None,
        output_dir=output,
        minimum_cutoff_rows=30,
        minimum_admitted_rows=1,
        minimum_resolved_bins=2,
        minimum_failure_episodes=30,
        minimum_complete_window_episodes=24,
        minimum_failure_bins=30,
        minimum_span_days=180,
        minimum_profile_features=10,
    )
    result = run(args)
    assert result["status"] == "INSUFFICIENT_SUPPORT"
    assert not result["taxonomy_training_allowed"]
    assert (output / "decision_health_6h.parquet").exists()
    assert (output / "failure_episodes.parquet").exists()
    assert not (output / "detector_oof.parquet").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["no_forward_rows_used"]
    assert manifest["no_h1_h3_h6_h12_fields_used"]


def test_detector_promotion_requires_latest_coverage_and_positive_economics() -> None:
    classification = {
        "aggregate": {
            "transition_within_3h": {"roc_auc": 0.60},
            "active_transition": {"roc_auc": 0.60},
        },
        "latest_month_metrics": {
            "transition_within_3h": {
                "rows": 500,
                "positive_rows": 20,
                "roc_auc": 0.55,
            },
            "active_transition": {
                "rows": 500,
                "positive_rows": 10,
                "roc_auc": 0.55,
            },
        },
    }
    economics = {
        "aggregate": {
            "mapped_score": {"mean_net_ev_bps": -5.0},
            "failure_trust_adjusted_score": {"mean_net_ev_bps": -1.0},
        },
        "latest_month::2026-07": {
            "mapped_score": {"mean_net_ev_bps": -50.0},
            "failure_trust_adjusted_score": {"mean_net_ev_bps": -51.0},
        },
    }
    gate = _detector_promotion_gate(
        classification,
        economics,
        minimum_rows=1_000,
        minimum_positive_events=50,
    )
    assert gate["status"] == "REJECT"
    assert not gate["detector_promotion_allowed"]
    assert not gate["criteria"]["latest_transition_rows"]["pass"]
    assert not gate["criteria"]["latest_adjusted_net_bps"]["pass"]


def test_detector_report_includes_current_and_destination_state_metrics() -> None:
    predictions = pd.DataFrame(
        {
            "execution_decision_utc": pd.date_range(
                "2025-11-01", periods=4, freq="h", tz="UTC"
            ),
            "target__transition_within_3h": [0, 1, 0, 1],
            "p_transition_within_3h": [0.1, 0.8, 0.2, 0.7],
            "target__active_transition": [0, 0, 1, 1],
            "p_active_transition": [0.1, 0.2, 0.7, 0.8],
            "target__current_failure_state": [
                "stable",
                "failure",
                "stable",
                "failure",
            ],
            "predicted_current_failure_state": [
                "stable",
                "failure",
                "stable",
                "stable",
            ],
            "p_current_state__stable": [0.9, 0.2, 0.8, 0.6],
            "p_current_state__failure": [0.1, 0.8, 0.2, 0.4],
            "target__destination_state_3h": [
                "stable",
                "failure",
                "failure",
                "stable",
            ],
            "predicted_destination_state_3h": [
                "stable",
                "failure",
                "stable",
                "stable",
            ],
            "p_destination__stable": [0.9, 0.2, 0.6, 0.8],
            "p_destination__failure": [0.1, 0.8, 0.4, 0.2],
        }
    )
    report = _detector_classification_report(predictions)["aggregate"]
    assert report["current_failure_state"]["rows"] == 4
    assert report["current_failure_state"]["balanced_accuracy"] == 0.75
    assert report["destination_state_3h"]["rows"] == 4
    assert "multiclass_log_loss" in report["destination_state_3h"]
