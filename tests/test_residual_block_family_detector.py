from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import scripts.run_residual_block_family_detector as detector
from scripts.run_residual_block_family_detector import (
    _family_train_samples,
    _daily_start_features,
    _family_summary,
    _phase_event_days,
    _screen_features,
    _top_k_metrics,
)


def test_daily_start_features_exclude_future_values() -> None:
    daily = pd.DataFrame(
        {
            "day": pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC"),
            "side_name": "short",
            "archetype_policy_key": "short_default_clean_path",
            "feature": [1.0, 3.0, 100.0],
        }
    )
    result, features = _daily_start_features(daily)
    assert "onset__feature" in features
    assert result.loc[1, "prior2__feature"] == 1.0
    assert result.loc[1, "onset__feature"] == 2.0


def test_feature_screen_is_train_only_and_prefers_separation() -> None:
    samples = pd.DataFrame(
        {
            "target": [1, 1, 1, 0, 0, 0],
            "strong": [3, 4, 5, 0, 0, 1],
            "weak": [1, 1, 1, 1, 1, 1],
        }
    )
    assert _screen_features(samples, ["strong", "weak"], maximum=1) == ["strong"]


def test_top_k_metrics_are_daily_not_row_weighted() -> None:
    frame = pd.DataFrame({"risk": np.linspace(0.0, 1.0, 10), "event_start": [False] * 9 + [True]})
    metrics = _top_k_metrics(frame, fraction=.10)
    assert metrics["top10_selected_days"] == 1
    assert metrics["top10_precision"] == 1.0
    assert metrics["top10_block_recall"] == 1.0


def test_family_summary_requires_all_three_folds_to_repeat() -> None:
    report = pd.DataFrame(
        {
            "status": ["ok", "ok", "ok"],
            "family_source": ["onset_mechanism"] * 3,
            "side_name": ["long"] * 3,
            "archetype_policy_key": ["long_mixed"] * 3,
            "family": ["compression"] * 3,
            "fold_start": pd.date_range("2026-01-01", periods=3, freq="QS", tz="UTC"),
            "top01_lift": [1.0] * 3, "top01_fpr": [.01] * 3, "top01_block_recall": [0.0] * 3,
            "top03_lift": [1.0] * 3, "top03_fpr": [.03] * 3, "top03_block_recall": [0.0] * 3,
            "top05_lift": [3.0] * 3, "top05_fpr": [.05] * 3, "top05_block_recall": [1.0, 1.0, 0.0],
            "top10_lift": [1.0] * 3, "top10_fpr": [.10] * 3, "top10_block_recall": [0.0] * 3,
        }
    )
    summary = _family_summary(report)
    assert not summary.loc[0, "passes_top05_repetition_gate"]


def test_event_phases_keep_blocks_not_trade_rows() -> None:
    days = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    event = pd.DataFrame(
        {
            "day": days,
            "side_name": ["long"] * 4,
            "archetype_policy_key": ["long_mixed"] * 4,
            "event_block": ["event_001", "event_001", "normal", "event_002"],
        }
    )
    assert _phase_event_days(event, event_phase="onset")["day"].tolist() == [days[0], days[3]]
    assert _phase_event_days(event, event_phase="active")["day"].tolist() == [days[0], days[1], days[3]]
    assert _phase_event_days(event, event_phase="late")["day"].tolist() == [days[1]]


def test_active_samples_align_control_days_to_event_offsets() -> None:
    days = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
    events = pd.DataFrame(
        {
            "day": days,
            "side_name": ["long"] * 8,
            "archetype_policy_key": ["long_mixed"] * 8,
            "event_block": ["normal", "event_001", "event_001", "normal", "normal", "normal", "normal", "normal"],
        }
    )
    taxonomy = pd.DataFrame(
        {
            "side_name": ["long"], "archetype_policy_key": ["long_mixed"],
            "event_block": ["event_001"], "family": ["liquidation"],
        }
    )
    controls = pd.DataFrame(
        {
            "side_name": ["long"], "archetype_policy_key": ["long_mixed"],
            "event_block": ["event_001"], "event_start": [days[1]], "control_start": [days[4]],
        }
    )
    start_features = pd.DataFrame(
        {
            "day": days, "side_name": ["long"] * 8,
            "archetype_policy_key": ["long_mixed"] * 8, "state__x": range(8),
        }
    )
    result = _family_train_samples(
        taxonomy, controls, start_features, events, side="long", archetype="long_mixed",
        family="liquidation", family_column="family", event_phase="active",
    )
    assert result.loc[result["target"].eq(1), "day"].tolist() == [days[1], days[2]]
    assert result.loc[result["target"].eq(0), "day"].tolist() == [days[4], days[5]]


def test_family_samples_do_not_cross_join_reused_local_event_ids() -> None:
    days = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    events = pd.DataFrame(
        {
            "day": [days[0], days[1], days[0], days[1]],
            "side_name": ["long", "long", "short", "short"],
            "archetype_policy_key": ["long_mixed", "long_mixed", "short_default", "short_default"],
            "event_block": ["event_001", "normal", "event_001", "normal"],
        }
    )
    taxonomy = pd.DataFrame(
        {
            "side_name": ["long"],
            "archetype_policy_key": ["long_mixed"],
            "event_block": ["event_001"],
            "family": ["liquidation"],
        }
    )
    controls = pd.DataFrame(
        {
            "side_name": ["long"], "archetype_policy_key": ["long_mixed"],
            "event_block": ["event_001"], "event_start": [days[0]], "control_start": [days[1]],
        }
    )
    start_features = events.loc[:, ["day", "side_name", "archetype_policy_key"]].copy()
    start_features["state__x"] = np.arange(len(start_features), dtype=np.float32)
    result = _family_train_samples(
        taxonomy, controls, start_features, events, side="long", archetype="long_mixed",
        family="liquidation", family_column="family", event_phase="onset",
    )
    assert len(result) == 2
    assert result["side_name"].eq("long").all()
    assert result["archetype_policy_key"].eq("long_mixed").all()


def test_active_phase_runner_smoke_is_block_level_and_materializes(tmp_path, monkeypatch) -> None:
    days = pd.date_range("2026-01-01", periods=120, freq="D", tz="UTC")
    state = pd.DataFrame(
        {
            "__ts__": days,
            "__symbol__": ["BTC/USD:USD"] * len(days),
            "side_name": ["long"] * len(days),
            "archetype_policy_key": ["long_mixed"] * len(days),
            "selected_top30": True,
            "shock_12h": np.linspace(-1, 1, len(days), dtype=np.float32),
        }
    )
    state_path = tmp_path / "state.parquet"
    state.to_parquet(state_path)
    events = pd.DataFrame(
        {
            "day": days,
            "side_name": ["long"] * len(days),
            "archetype_policy_key": ["long_mixed"] * len(days),
            "adverse_calendar_cell": 0,
        }
    )
    # Three train blocks and two later blocks, all two days long, exercise the
    # active-day label and relative control-day alignment.
    for start in (20, 40, 60, 91, 105):
        events.loc[start : start + 1, "adverse_calendar_cell"] = 1
    calendar_path = tmp_path / "calendar.csv"
    events.to_csv(calendar_path, index=False)
    monkeypatch.setattr(
        detector,
        "FOLDS",
        ((pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-05-01", tz="UTC")),),
    )
    output = tmp_path / "out"
    args = SimpleNamespace(
        output=output,
        event_calendar=[calendar_path],
        state_artifact=[state_path],
        group=["long::long_mixed"],
        controls_per_block=2,
        max_features=4,
        min_family_blocks=3,
        seed=7,
        family_source="onset_mechanism",
        event_phase="active",
    )
    manifest = detector.run(args)
    assert manifest["event_phase"] == "active"
    report = pd.read_csv(output / "family_detector_oof_metrics.csv")
    assert not report.empty
    assert report["event_starts"].max() == 4
