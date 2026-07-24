from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from extreme_price_movements.training_resource_guard import TrainingResourceGuardError

SCRIPT = Path(__file__).parents[1] / "scripts/materialize_packb_pre_march_population.py"
SPEC = importlib.util.spec_from_file_location("packb_population", SCRIPT)
assert SPEC and SPEC.loader
materializer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(materializer)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _decisions(path: Path) -> None:
    _write_json(
        path,
        {
            "schema_version": "full_pipeline_decisions_v1",
            "status": "LOCKED_BEFORE_NEW_TRAINING",
            "decisions": {
                "DEC-09": {
                    "feature_selection_hpo_resolution_cutoff_utc": "2026-03-01T00:00:00Z",
                    "decision_timestamp": "signal_timestamp + 1 hour",
                    "outer_folds": [list(item) for item in materializer.OUTER_FOLDS],
                    "packb_pre_march_inner_calendar": {
                        "ae_gmm_reference_signal_interval": [
                            "2025-01-01T00:00:00Z",
                            "2025-11-01T00:00:00Z",
                        ],
                        "feature_selection_validation_interval": [
                            "2025-11-01T00:00:00Z",
                            "2025-12-01T00:00:00Z",
                        ],
                        "hpo_validation_intervals": [
                            ["2025-12-01T00:00:00Z", "2026-01-01T00:00:00Z"],
                            ["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"],
                            ["2026-02-01T00:00:00Z", "2026-03-01T00:00:00Z"],
                        ],
                    },
                }
            },
        },
    )


def _source(tmp_path: Path, *, bad_decision: bool = False) -> tuple[Path, Path]:
    labels = tmp_path / "labels"
    labels.mkdir(parents=True)
    signals = [
        "2025-01-15T00:00:00Z",
        "2025-10-30T21:00:00Z",
        "2025-11-10T00:00:00Z",
        "2025-12-10T00:00:00Z",
        "2026-01-10T00:00:00Z",
        "2026-02-10T00:00:00Z",
    ]
    inventory = []
    for side in materializer.CANONICAL_SIDES:
        signal = pd.to_datetime(signals, utc=True)
        decision = signal + pd.Timedelta(hours=1)
        if bad_decision and side == "short":
            decision = pd.DatetimeIndex(
                [decision[0] - pd.Timedelta(minutes=1), *decision[1:]]
            )
        frame = pd.DataFrame(
            {
                "candidate_id": [f"{side}-{index}" for index in range(len(signal))],
                "side_name": side,
                "__ts__": signal,
                "__signal_ts__": signal,
                "__decision_ts__": decision,
                "__symbol__": "BTCUSDT",
            }
        )
        name = f"{side}.parquet"
        frame.to_parquet(labels / name, index=False)
        inventory.append({"file": name, "rows": len(frame)})
        inventory[-1]["expected_current_rows"] = len(frame)
    pd.DataFrame(
        {
            "candidate_id": ["excluded"],
            "side_name": ["short"],
            "__ts__": [pd.Timestamp("2025-01-01T00:00:00Z")],
            "__decision_ts__": [pd.Timestamp("2025-01-01T01:00:00Z")],
        }
    ).to_parquet(labels / "train_global_short_7.parquet", index=False)
    audit = labels / "causal_audit.json"
    _write_json(
        audit,
        {
            "schema": "packb_current_canonical_label_inventory_audit_v1",
            "status": "PASS",
            "mode": "streaming_full_audit",
            "per_file": inventory,
            "inventory": {
                "canonical_monthly_files": 2,
                "excluded_unlisted_monolithic_files": ["train_global_short_7.parquet"],
                "expected_current_rows": 12,
            },
            "totals": {"rows": 12},
        },
    )
    return labels, audit


def test_calendar_predicates_are_strict_at_boundaries() -> None:
    start, end = materializer.FS_WINDOW
    signal = pd.Series(
        pd.to_datetime(
            [
                "2025-10-30T22:00:00Z",
                "2025-10-30T23:00:00Z",
                "2025-11-01T00:00:00Z",
                "2025-11-30T23:00:00Z",
            ],
            utc=True,
        )
    )
    decision = signal + pd.Timedelta(hours=1)
    assert materializer.strict_train_mask(signal, decision, start).tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert materializer.strict_validation_mask(
        signal, decision, start, end
    ).tolist() == [False, False, True, True]
    feb_signal = pd.Series(
        pd.to_datetime(["2026-02-27T22:00:00Z", "2026-02-27T23:00:00Z"], utc=True)
    )
    assert materializer.strict_validation_mask(
        feb_signal, feb_signal + pd.Timedelta(hours=1), *materializer.HPO_WINDOWS[2]
    ).tolist() == [True, False]


def test_contract_only_writes_no_parquet_ledgers(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions.json"
    _decisions(decisions)
    labels, audit = _source(tmp_path)
    output = tmp_path / "contract"
    report = materializer.materialize(
        decisions_path=decisions,
        labels_dir=labels,
        causal_audit_path=audit,
        output_dir=output,
        contract_only=True,
    )
    assert report["status"] == "CONTRACT_ONLY"
    assert (output / "materialization_contract.json").is_file()
    assert not list(output.rglob("*.parquet"))


def test_streams_full_population_and_every_fixed_side_cohort(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions.json"
    _decisions(decisions)
    labels, audit = _source(tmp_path)
    output = tmp_path / "materialized"
    manifest = materializer.materialize(
        decisions_path=decisions,
        labels_dir=labels,
        causal_audit_path=audit,
        output_dir=output,
        batch_rows=2,
    )
    assert manifest["status"] == "MATERIALIZED_IMMUTABLE"
    assert manifest["ledgers"]["authorized_population"]["rows"] == 12
    assert (
        pq.ParquetFile(
            output / "authorized_pre_march_population.parquet"
        ).metadata.num_rows
        == 12
    )
    for side in materializer.CANONICAL_SIDES:
        assert (output / "cohorts" / side / "ae_reference.parquet").is_file()
        assert manifest["ledgers"][f"{side}/hpo_3_valid"]["rows"] == 1
    assert manifest["streaming"]["full_frame_load"] is False
    second = materializer.materialize(
        decisions_path=decisions,
        labels_dir=labels,
        causal_audit_path=audit,
        output_dir=tmp_path / "materialized_again",
        batch_rows=3,
    )
    assert (
        second["ledgers"]["authorized_population"]["identity_stream_sha256"]
        == manifest["ledgers"]["authorized_population"]["identity_stream_sha256"]
    )


def test_bad_timing_or_existing_output_fails_without_publication(
    tmp_path: Path,
) -> None:
    decisions = tmp_path / "decisions.json"
    _decisions(decisions)
    labels, audit = _source(tmp_path, bad_decision=True)
    output = tmp_path / "bad"
    with pytest.raises(Exception, match="decision"):
        materializer.materialize(
            decisions_path=decisions,
            labels_dir=labels,
            causal_audit_path=audit,
            output_dir=output,
        )
    assert not output.exists()
    labels, audit = _source(tmp_path / "good")
    output.mkdir()
    with pytest.raises(FileExistsError):
        materializer.materialize(
            decisions_path=decisions,
            labels_dir=labels,
            causal_audit_path=audit,
            output_dir=output,
        )


def test_resource_failure_does_not_create_output(tmp_path: Path) -> None:
    class FailingGuard:
        def preflight(self, stage: str) -> None:
            raise TrainingResourceGuardError(stage)

        def checkpoint(self, stage: str) -> None:
            return None

    with pytest.raises(TrainingResourceGuardError):
        materializer.materialize(
            output_dir=tmp_path / "out",
            contract_only=True,
            resource_guard=FailingGuard(),
        )
    assert not (tmp_path / "out").exists()
