import json
from pathlib import Path

import pandas as pd

from scripts.report_strict_oos_repair_ranker_validation_input_preflight import build_preflight


def _write_table(path: Path, col: str, periods: list[str]) -> None:
    rows = []
    for period in periods:
        value = period if col == "period" else f"{period}-01 00:00:00+00:00"
        rows.append({col: value, "value": 1.0})
    if path.suffix == ".csv":
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        pd.DataFrame(rows).to_parquet(path, index=False)


def _manifest(tmp_path: Path, **overrides) -> Path:
    data = {
        "scope": "strict_oos_repair_ranker_frozen_profile_run",
        "history_periods": ["2026-04", "2026-05", "2026-06"],
        "validation_manifest": {"validation_periods": ["2026-07"]},
        "quality_labels_path": str(tmp_path / "quality.parquet"),
        "labels_path": str(tmp_path / "labels"),
        "predictions_path": str(tmp_path / "predictions.parquet"),
        "event_rows_path": str(tmp_path / "events.csv"),
        "feature_dir": str(tmp_path / "features"),
        "feature_list_csv": str(tmp_path / "features.csv"),
    }
    data.update(overrides)
    path = tmp_path / "runner_manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _write_ready_inputs(tmp_path: Path) -> None:
    (tmp_path / "labels").mkdir()
    (tmp_path / "features").mkdir()
    _write_table(tmp_path / "quality.parquet", "__ts__", ["2026-07"])
    _write_table(tmp_path / "labels" / "labels.parquet", "__ts__", ["2026-07"])
    _write_table(tmp_path / "predictions.parquet", "timestamp", ["2026-07"])
    _write_table(tmp_path / "features" / "BTC.parquet", "__ts__", ["2026-07"])
    _write_table(tmp_path / "events.csv", "period", ["2026-04", "2026-05", "2026-06"])
    (tmp_path / "features.csv").write_text("feature\nx\n", encoding="utf-8")


def test_preflight_ready_when_validation_and_history_inputs_exist(tmp_path):
    _write_ready_inputs(tmp_path)
    manifest = _manifest(tmp_path)

    result = build_preflight(
        runner_manifest_path=manifest,
        output_dir=tmp_path / "out",
    )

    assert result["ready_to_run_frozen_validation"] is True
    assert result["decision"] == "ready"
    assert result["blocking_inputs"] == []
    assert (tmp_path / "out" / "strict_oos_repair_ranker_validation_input_preflight.json").exists()


def test_preflight_blocks_when_prediction_validation_period_is_missing(tmp_path):
    _write_ready_inputs(tmp_path)
    _write_table(tmp_path / "predictions.parquet", "timestamp", ["2026-06"])
    manifest = _manifest(tmp_path)

    result = build_preflight(
        runner_manifest_path=manifest,
        output_dir=tmp_path / "out",
    )

    assert result["ready_to_run_frozen_validation"] is False
    pred_block = [row for row in result["blocking_inputs"] if row["role"] == "predictions"]
    assert pred_block
    assert pred_block[0]["missing_expected_periods"] == ["2026-07"]


def test_preflight_event_rows_require_history_not_validation_period(tmp_path):
    _write_ready_inputs(tmp_path)
    _write_table(tmp_path / "events.csv", "period", ["2026-04", "2026-05"])
    manifest = _manifest(tmp_path)

    result = build_preflight(
        runner_manifest_path=manifest,
        output_dir=tmp_path / "out",
    )

    event_block = [row for row in result["blocking_inputs"] if row["role"] == "event_rows_history"]
    assert event_block
    assert event_block[0]["missing_expected_periods"] == ["2026-06"]
    assert not any("2026-07" in row["missing_expected_periods"] for row in event_block)


def test_preflight_reads_feature_period_from_matrix_filename(tmp_path):
    _write_ready_inputs(tmp_path)
    feature_dir = tmp_path / "features"
    for path in feature_dir.glob("*.parquet"):
        path.unlink()
    pd.DataFrame({"x": [1.0, 2.0]}).to_parquet(feature_dir / "matrix_20260701T000000Z.parquet", index=False)
    manifest = _manifest(tmp_path)

    result = build_preflight(
        runner_manifest_path=manifest,
        output_dir=tmp_path / "out",
    )

    feature_row = [row for row in result["inputs"] if row["role"] == "feature_store"][0]
    assert feature_row["timestamp_col"] == "__path_date__"
    assert feature_row["period_counts"] == {"2026-07": 2}
    assert feature_row["passes"] is True


def test_preflight_ignores_live_latest_matrix_for_feature_store_readiness(tmp_path):
    _write_ready_inputs(tmp_path)
    feature_dir = tmp_path / "features"
    for path in feature_dir.glob("*.parquet"):
        path.unlink()
    _write_table(feature_dir / "BTC.parquet", "__ts__", ["2026-06"])
    live_dir = feature_dir / "_live_latest_matrix"
    live_dir.mkdir()
    pd.DataFrame({"x": [1.0, 2.0]}).to_parquet(
        live_dir / "matrix_20260701T000000Z.parquet",
        index=False,
    )
    manifest = _manifest(tmp_path)

    result = build_preflight(
        runner_manifest_path=manifest,
        output_dir=tmp_path / "out",
    )

    feature_row = [row for row in result["inputs"] if row["role"] == "feature_store"][0]
    assert feature_row["period_counts"] == {"2026-06": 1}
    assert feature_row["missing_expected_periods"] == ["2026-07"]
    assert feature_row["passes"] is False


def test_preflight_blocks_missing_validation_period_config(tmp_path):
    _write_ready_inputs(tmp_path)
    manifest = _manifest(tmp_path, validation_manifest={"validation_periods": []})

    result = build_preflight(
        runner_manifest_path=manifest,
        output_dir=tmp_path / "out",
    )

    assert result["ready_to_run_frozen_validation"] is False
    assert any(row["role"] == "validation_periods" for row in result["blocking_inputs"])
