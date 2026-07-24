from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import audit_packb_current_label_inventory as audit


def _row(side: str, candidate_id: str, signal: pd.Timestamp) -> pd.DataFrame:
    decision = signal.tz_localize("UTC") + pd.Timedelta(hours=1)
    return pd.DataFrame(
        {
            "candidate_id": [candidate_id],
            # The real established Pack-B signal field is legacy-naive and
            # interpreted as UTC; the downstream causal clocks are explicit.
            "__ts__": [signal],
            "__decision_ts__": [decision],
            "__entry_ts__": [decision],
            "__first_path_ts__": [decision],
            "side_name": [side],
            "__first_touch_round_trip_cost__": [0.01],
            "__first_touch_valid_path__": [1.0],
        }
    )


def _write_inventory(tmp_path: Path) -> Path:
    labels = tmp_path / "labels"
    labels.mkdir()
    entries = []
    for side in audit.CANONICAL_SIDES:
        for year, month in [(2025, month) for month in range(1, 13)] + [
            (2026, month) for month in range(1, 8)
        ]:
            name = f"train_global_{side}_5_{year}_{month:02d}.parquet"
            signal = pd.Timestamp(year=year, month=month, day=1)
            frame = _row(side, f"{side}-{year}-{month}-0", signal)
            if year == 2026 and month == 7:
                frame = pd.concat(
                    [
                        frame,
                        _row(
                            side,
                            f"{side}-{year}-{month}-1",
                            signal + pd.Timedelta(hours=1),
                        ),
                    ],
                    ignore_index=True,
                )
            frame.to_parquet(labels / name, index=False)
            entries.append({"file": name, "rows": 1})
    (labels / audit.BASE_AUDIT_FILENAME).write_text(
        json.dumps({"files": 38, "rows": 38, "per_file": entries}), encoding="utf-8"
    )
    (labels / audit.TAIL_APPEND_FILENAME).write_text(
        json.dumps(
            {
                "sides": [
                    {
                        "side": side,
                        "target": str(
                            labels / f"train_global_{side}_5_2026_07.parquet"
                        ),
                        "existing_rows": 1,
                        "appended_rows": 1,
                        "final_rows": 2,
                    }
                    for side in audit.CANONICAL_SIDES
                ]
            }
        ),
        encoding="utf-8",
    )
    # The monolithic short_7 source is intentionally present but explicitly
    # outside the causal-audit monthly inventory.
    _row("short", "short-monolithic", pd.Timestamp("2026-07-02")).to_parquet(
        labels / audit.EXCLUDED_MONOLITHIC_SHARD, index=False
    )
    return labels


def test_contract_only_uses_base_inventory_and_reconciles_july_tail(
    tmp_path: Path,
) -> None:
    labels = _write_inventory(tmp_path)

    report = audit.contract_only(labels_dir=labels)

    assert report["status"] == "CONTRACT_VALIDATED_SCHEMA_ONLY"
    assert report["inventory"]["canonical_monthly_files"] == 38
    assert report["inventory"]["expected_current_rows"] == 40
    assert report["inventory"]["excluded_unlisted_monolithic_files"] == [
        audit.EXCLUDED_MONOLITHIC_SHARD
    ]
    july = next(
        item
        for item in report["per_file"]
        if item["file"].endswith("long_5_2026_07.parquet")
    )
    assert july["metadata_rows"] == 2
    assert july["tail_reconciliation"]["appended_rows"] == 1
    assert (
        july["timestamp_storage_contract"]["__ts__"]
        == "legacy_naive_interpreted_as_utc"
    )


def test_streaming_audit_validates_rows_and_creates_immutable_report(
    tmp_path: Path,
) -> None:
    labels = _write_inventory(tmp_path)
    events: list[str] = []

    class Guard:
        def preflight(self, stage: str) -> None:
            events.append(stage)

        def checkpoint(self, stage: str) -> None:
            events.append(stage)

    output = tmp_path / "reports" / "current_inventory.json"
    report = audit.run_full_audit(
        labels_dir=labels, output_path=output, batch_rows=1, resource_guard=Guard()
    )

    assert report["status"] == "PASS"
    assert report["totals"]["rows"] == 40
    assert report["totals"]["duplicate_candidate_id"] == 0
    assert output.is_file()
    assert events[0] == "contract_only_schema_inventory"
    assert "before_full_label_scan" in events
    assert any(
        stage.startswith("before_batch:train_global_long_5_2025_01") for stage in events
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit.run_full_audit(
            labels_dir=labels, output_path=output, resource_guard=Guard()
        )


def test_streaming_audit_reports_duplicate_and_causal_failures(tmp_path: Path) -> None:
    labels = _write_inventory(tmp_path)
    first = labels / "train_global_long_5_2025_01.parquet"
    second = labels / "train_global_long_5_2025_02.parquet"
    broken = pd.read_parquet(second)
    broken.loc[0, "candidate_id"] = pd.read_parquet(first).loc[0, "candidate_id"]
    broken.loc[0, "__decision_ts__"] = pd.Timestamp("2025-02-01T03:00:00Z")
    broken.to_parquet(second, index=False)

    report = audit.run_full_audit(
        labels_dir=labels, output_path=tmp_path / "broken.json", batch_rows=1
    )

    assert report["status"] == "FAILED_INVARIANT_AUDIT"
    assert report["totals"]["duplicate_candidate_id"] == 1
    assert report["totals"]["bad_decision"] == 1


def test_resource_guard_can_abort_before_a_label_batch(tmp_path: Path) -> None:
    labels = _write_inventory(tmp_path)

    class Guard:
        def preflight(self, _stage: str) -> None:
            return None

        def checkpoint(self, stage: str) -> None:
            if stage.startswith("before_batch:"):
                raise RuntimeError("resource limit")

    with pytest.raises(RuntimeError, match="resource limit"):
        audit.run_full_audit(
            labels_dir=labels,
            output_path=tmp_path / "not_written.json",
            resource_guard=Guard(),
        )
    assert not (tmp_path / "not_written.json").exists()


def test_contract_only_rejects_unlisted_parquet_other_than_short_7(
    tmp_path: Path,
) -> None:
    labels = _write_inventory(tmp_path)
    _row("long", "unexpected", pd.Timestamp("2026-07-02")).to_parquet(
        labels / "unlisted.parquet", index=False
    )

    with pytest.raises(
        audit.PackBCurrentLabelAuditError, match="unexpected=unlisted.parquet"
    ):
        audit.contract_only(labels_dir=labels)


def test_contract_only_rejects_unreconciled_physical_row_count(tmp_path: Path) -> None:
    labels = _write_inventory(tmp_path)
    shard = labels / "train_global_long_5_2025_01.parquet"
    pd.concat(
        [pd.read_parquet(shard), _row("long", "extra", pd.Timestamp("2025-01-02"))],
        ignore_index=True,
    ).to_parquet(shard, index=False)

    with pytest.raises(audit.PackBCurrentLabelAuditError, match="metadata row counts"):
        audit.contract_only(labels_dir=labels)
