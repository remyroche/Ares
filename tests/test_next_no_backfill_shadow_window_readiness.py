from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.audit_next_no_backfill_shadow_window_readiness import (
    annotate_coverage_gaps,
    build_readiness,
    feature_store_bounds,
    feature_store_hourly_coverage,
    update_config,
    write_readiness,
)


def _config() -> dict:
    return {
        "active_stack": {
            "rank_contract": "anchor_global_policy_rank_reference",
            "rank_scope": "global_over_time",
            "enabled_heads": ["short_asset", "short_boll"],
            "disabled_heads": ["long_bars", "long_dist"],
            "qfail_active": False,
            "market_state_threshold_controller_active": False,
        },
        "market_state_controller_validation": {
            "global_rank_threshold_controller_no_backfill_shadow_monitor": {
                "windows": [
                    {
                        "period_start": "2026-06-25T09:00:00+00:00",
                        "period_end": "2026-06-26T07:00:00+00:00",
                    }
                ]
            },
            "global_rank_threshold_controller_no_backfill_shadow_window_discovery": {
                "appendable_candidate_count": 0,
                "latest_discovered_window_end": "2026-06-26T07:00:00+00:00",
            },
        },
    }


def test_next_no_backfill_readiness_detects_insufficient_mature_history(tmp_path: Path) -> None:
    feature_store = tmp_path / "features" / "20260627_010000"
    feature_store.mkdir(parents=True)
    timestamps = pd.date_range(
        "2026-06-26T00:00:00Z",
        "2026-06-27T00:00:00Z",
        freq="1h",
    )
    pd.DataFrame({"ts": timestamps, "x": range(len(timestamps))}).to_parquet(
        feature_store / "symbol=BTC_USD:USD.parquet",
        index=False,
    )

    summary = build_readiness(
        config=_config(),
        config_path=tmp_path / "stack.json",
        data_root=tmp_path,
        feature_store_dir=feature_store,
        output_dir=tmp_path / "readiness",
        maturity_buffer_hours=16,
        target_window_hours=24,
        min_timestamp_count=3,
        min_feature_timestamp_coverage=0.95,
    )

    assert summary["status"] == "not_scoreable_yet"
    assert summary["feature_timestamp_max"] == "2026-06-27T00:00:00+00:00"
    assert summary["maturity_cutoff"] == "2026-06-26T08:00:00+00:00"
    assert summary["next_window_start"] == "2026-06-26T08:00:00+00:00"
    assert summary["mature_timestamp_count_available"] == 1
    assert summary["missing_feature_hours_for_min_window"] == 2
    assert summary["missing_feature_hours_for_full_window"] == 23
    assert summary["scoreable_min_window_now"] is False
    assert summary["scoreable_full_window_now"] is False
    assert summary["minimum_window_feature_coverage_ready"] is True
    assert summary["minimum_window_feature_coverage"]["min_feature_file_coverage"] == 1.0
    assert "insufficient_matured_timestamps_for_minimum_shadow_window" in summary["failures"]


def test_next_no_backfill_readiness_writes_and_updates_config(tmp_path: Path) -> None:
    feature_store = tmp_path / "features" / "20260627_030000"
    feature_store.mkdir(parents=True)
    timestamps = pd.date_range(
        "2026-06-26T08:00:00Z",
        "2026-06-27T03:00:00Z",
        freq="1h",
    )
    pd.DataFrame({"ts": timestamps, "x": range(len(timestamps))}).to_parquet(
        feature_store / "symbol=ETH_USD:USD.parquet",
        index=False,
    )
    config_path = tmp_path / "stack.json"
    config = _config()
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "readiness"

    summary = build_readiness(
        config=config,
        config_path=config_path,
        data_root=tmp_path,
        feature_store_dir=feature_store,
        output_dir=output_dir,
        maturity_buffer_hours=16,
        target_window_hours=24,
        min_timestamp_count=3,
        min_feature_timestamp_coverage=0.95,
    )
    write_readiness(summary, output_dir)
    update_config(config, config_path, summary)

    assert (output_dir / "next_no_backfill_shadow_window_readiness.json").exists()
    assert (output_dir / "next_no_backfill_shadow_window_readiness.csv").exists()
    assert (output_dir / "next_no_backfill_shadow_window_readiness_report.md").exists()
    assert (output_dir / "next_no_backfill_shadow_window_low_coverage_hours.csv").exists()
    updated = json.loads(config_path.read_text())
    stored = updated["market_state_controller_validation"][
        "global_rank_threshold_controller_no_backfill_next_window_readiness"
    ]
    assert stored["status"] == "scoreable_now"
    assert stored["scoreable_min_window_now"] is True
    assert stored["scoreable_full_window_now"] is False
    assert stored["mature_timestamp_count_available"] == 4
    assert stored["min_feature_timestamp_coverage"] == 0.95
    assert stored["minimum_window_feature_coverage_ready"] is True
    assert stored["minimum_window_feature_coverage"]["min_feature_file_coverage"] == 1.0
    assert stored["full_window_feature_coverage_ready"] is False
    assert (
        stored["needed_feature_timestamp_max_for_min_window"]
        == "2026-06-27T02:00:00+00:00"
    )
    assert (
        stored["needed_feature_timestamp_max_for_full_window"]
        == "2026-06-27T23:00:00+00:00"
    )


def test_next_no_backfill_readiness_counts_duckdb_feature_deltas(tmp_path: Path) -> None:
    import duckdb

    feature_store = tmp_path / "features" / "20260627_120000"
    feature_store.mkdir(parents=True)
    parquet_path = feature_store / "symbol=BTC_USD:USD.parquet"
    base_ts = pd.date_range(
        "2026-06-27T10:00:00Z",
        "2026-06-27T12:00:00Z",
        freq="1h",
    )
    pd.DataFrame({"ts": base_ts, "x": range(len(base_ts))}).to_parquet(
        parquet_path,
        index=False,
    )
    delta_path = feature_store / "symbol=BTC_USD:USD.parquet.deltas.duckdb"
    con = duckdb.connect(str(delta_path))
    delta = pd.DataFrame(
        {
            "ts": pd.date_range(
                "2026-06-27T13:00:00Z",
                "2026-06-27T14:00:00Z",
                freq="1h",
            ),
            "x": [3.0, 4.0],
            "__symbol__": ["BTC/USD:USD", "BTC/USD:USD"],
        }
    )
    con.register("incoming_delta", delta)
    con.execute("CREATE TABLE feature_deltas AS SELECT * FROM incoming_delta")
    con.close()

    bounds = feature_store_bounds(feature_store)
    coverage = feature_store_hourly_coverage(
        feature_store,
        start=pd.Timestamp("2026-06-27T12:00:00Z"),
        end=pd.Timestamp("2026-06-27T14:00:00Z"),
    )

    assert bounds["feature_timestamp_max"] == "2026-06-27T14:00:00+00:00"
    assert bounds["feature_row_count"] == 5
    assert coverage["min_feature_file_coverage"] == 1.0
    assert coverage["low_coverage_timestamp_count"] == 0


def test_next_no_backfill_readiness_reports_missing_files_by_low_coverage_hour(
    tmp_path: Path,
) -> None:
    feature_store = tmp_path / "features" / "20260627_030000"
    feature_store.mkdir(parents=True)
    pd.DataFrame(
        {
            "ts": pd.date_range(
                "2026-06-27T00:00:00Z",
                "2026-06-27T02:00:00Z",
                freq="1h",
            ),
            "x": [1.0, 2.0, 3.0],
        }
    ).to_parquet(feature_store / "symbol=BTC_USD:USD.parquet", index=False)
    pd.DataFrame(
        {
            "ts": [pd.Timestamp("2026-06-27T00:00:00Z")],
            "x": [4.0],
        }
    ).to_parquet(feature_store / "symbol=ETH_USD:USD.parquet", index=False)

    coverage = feature_store_hourly_coverage(
        feature_store,
        start=pd.Timestamp("2026-06-27T00:00:00Z"),
        end=pd.Timestamp("2026-06-27T02:00:00Z"),
    )

    assert coverage["coverage_feature_file_count"] == 2
    assert coverage["low_coverage_timestamp_count"] == 2
    assert coverage["min_feature_file_coverage"] == 0.5
    assert coverage["low_coverage_present_file_count_by_timestamp"][
        "2026-06-27T01:00:00+00:00"
    ] == 1
    assert coverage["low_coverage_missing_file_count_by_timestamp"][
        "2026-06-27T01:00:00+00:00"
    ] == 1
    assert coverage["low_coverage_missing_files_sample_by_timestamp"][
        "2026-06-27T01:00:00+00:00"
    ] == ["symbol=ETH_USD:USD.parquet"]
    assert coverage["low_coverage_present_files_sample_by_timestamp"][
        "2026-06-27T01:00:00+00:00"
    ] == ["symbol=BTC_USD:USD.parquet"]


def test_next_no_backfill_readiness_classifies_low_coverage_gap_types(
    tmp_path: Path,
) -> None:
    feature_store = tmp_path / "features" / "20260627_020000"
    feature_store.mkdir(parents=True)
    pd.DataFrame(
        {
            "ts": pd.date_range(
                "2026-06-27T01:00:00Z",
                "2026-06-27T02:00:00Z",
                freq="1h",
            ),
            "x": [1.0, 2.0],
        }
    ).to_parquet(feature_store / "symbol=BTC_USD:USD.parquet", index=False)
    pd.DataFrame(
        {
            "ts": [pd.Timestamp("2026-06-27T02:00:00Z")],
            "x": [3.0],
        }
    ).to_parquet(feature_store / "symbol=ETH_USD:USD.parquet", index=False)

    coverage = feature_store_hourly_coverage(
        feature_store,
        start=pd.Timestamp("2026-06-27T00:00:00Z"),
        end=pd.Timestamp("2026-06-27T03:00:00Z"),
    )
    annotated = annotate_coverage_gaps(
        coverage,
        feature_timestamp_max=pd.Timestamp("2026-06-27T02:00:00Z"),
        min_feature_timestamp_coverage=0.95,
    )

    assert annotated["low_coverage_gap_type_by_timestamp"][
        "2026-06-27T00:00:00+00:00"
    ] == "internal_total_gap"
    assert annotated["low_coverage_gap_type_by_timestamp"][
        "2026-06-27T01:00:00+00:00"
    ] == "internal_single_feature_gap"
    assert annotated["low_coverage_gap_type_by_timestamp"][
        "2026-06-27T03:00:00+00:00"
    ] == "tail_not_generated_yet"
    assert annotated["blocking_low_coverage_gap_type_counts"] == {
        "internal_total_gap": 1,
        "internal_single_feature_gap": 1,
        "tail_not_generated_yet": 1,
    }


def test_next_no_backfill_readiness_uses_latest_score_manifest_candidate_period(
    tmp_path: Path,
) -> None:
    feature_store = tmp_path / "features" / "20260628_160000"
    feature_store.mkdir(parents=True)
    timestamps = pd.date_range(
        "2026-06-26T20:00:00Z",
        "2026-06-28T16:00:00Z",
        freq="1h",
    )
    pd.DataFrame({"ts": timestamps, "x": range(len(timestamps))}).to_parquet(
        feature_store / "symbol=BTC_USD:USD.parquet",
        index=False,
    )
    score_dir = tmp_path / "reports" / "score_latest"
    score_dir.mkdir(parents=True)
    eval_candidates = score_dir / "eval_candidates.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-06-26T20:00:00Z",
                "2026-06-26T22:00:00Z",
                freq="1h",
            ),
            "symbol": ["BTC_USD", "ETH_USD", "SOL_USD"],
        }
    ).to_parquet(eval_candidates, index=False)
    (score_dir / "manifest.json").write_text(
        json.dumps({"eval_candidates": str(eval_candidates)}),
        encoding="utf-8",
    )
    config = _config()
    config["market_state_controller_validation"][
        "global_rank_threshold_controller_no_backfill_shadow_score_latest"
    ] = {"score_dir": str(score_dir)}

    summary = build_readiness(
        config=config,
        config_path=tmp_path / "stack.json",
        data_root=tmp_path,
        feature_store_dir=feature_store,
        output_dir=tmp_path / "readiness",
        maturity_buffer_hours=16,
        target_window_hours=24,
        min_timestamp_count=3,
        min_feature_timestamp_coverage=0.95,
    )

    assert summary["latest_scored_or_discovered_window_end"] == "2026-06-26T22:00:00+00:00"
    assert summary["next_window_start"] == "2026-06-26T23:00:00+00:00"
    assert summary["minimum_window_end"] == "2026-06-27T01:00:00+00:00"
    assert summary["scoreable_min_window_now"] is True


def test_next_no_backfill_readiness_uses_monitor_summary_window_metrics(
    tmp_path: Path,
) -> None:
    feature_store = tmp_path / "features" / "20260628_160000"
    feature_store.mkdir(parents=True)
    timestamps = pd.date_range(
        "2026-06-26T20:00:00Z",
        "2026-06-28T16:00:00Z",
        freq="1h",
    )
    pd.DataFrame({"ts": timestamps, "x": range(len(timestamps))}).to_parquet(
        feature_store / "symbol=BTC_USD:USD.parquet",
        index=False,
    )
    monitor_dir = tmp_path / "reports" / "monitor"
    monitor_dir.mkdir(parents=True)
    metrics_csv = monitor_dir / "no_backfill_shadow_window_metrics.csv"
    pd.DataFrame(
        {
            "period_end": [
                "2026-06-26T19:00:00+00:00",
                "2026-06-26T22:00:00+00:00",
            ]
        }
    ).to_csv(metrics_csv, index=False)
    summary_json = monitor_dir / "no_backfill_shadow_monitor_summary.json"
    summary_json.write_text(
        json.dumps({"window_metrics_csv": str(metrics_csv)}),
        encoding="utf-8",
    )
    config = _config()
    config["market_state_controller_validation"][
        "global_rank_threshold_controller_no_backfill_shadow_monitor"
    ] = {"summary_json": str(summary_json)}

    summary = build_readiness(
        config=config,
        config_path=tmp_path / "stack.json",
        data_root=tmp_path,
        feature_store_dir=feature_store,
        output_dir=tmp_path / "readiness",
        maturity_buffer_hours=16,
        target_window_hours=24,
        min_timestamp_count=3,
        min_feature_timestamp_coverage=0.95,
    )

    assert summary["latest_scored_or_discovered_window_end"] == "2026-06-26T22:00:00+00:00"
    assert summary["next_window_start"] == "2026-06-26T23:00:00+00:00"
