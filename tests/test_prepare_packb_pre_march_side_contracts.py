from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from extreme_price_movements.training_resource_guard import TrainingResourceGuardError

SCRIPT = Path(__file__).parents[1] / "scripts/prepare_packb_pre_march_side_contracts.py"
SPEC = importlib.util.spec_from_file_location("packb_prepare", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


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
                    "signal_timestamp_purge_hours": 25,
                    "purge_hours": 12,
                    "outer_folds": [list(item) for item in runner.OUTER_FOLDS],
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


def _labels(tmp_path: Path, *, rows: int = 1) -> tuple[Path, Path]:
    labels = tmp_path / "labels"
    labels.mkdir()
    files = []
    for side in ("long", "short"):
        frame = pd.DataFrame(
            {
                "candidate_id": [f"{side}-{i}" for i in range(rows)],
                "__ts__": pd.date_range("2025-10-01", periods=rows, freq="h", tz="UTC"),
                "__decision_ts__": pd.date_range(
                    "2025-10-01 01:00", periods=rows, freq="h", tz="UTC"
                ),
                "side_name": side,
            }
        )
        name = f"{side}.parquet"
        frame.to_parquet(labels / name, index=False)
        files.append({"file": name, "rows": rows})
    audit = labels / "causal_audit.json"
    _write_json(
        audit, {"files": len(files), "rows": rows * len(files), "per_file": files}
    )
    return labels, audit


def test_fixed_calendar_and_strict_boundary_predicates() -> None:
    calendar = runner.locked_calendar()
    assert calendar["feature_selection_validation"] == [
        "2025-11-01T00:00:00+00:00",
        "2025-12-01T00:00:00+00:00",
    ]
    assert len(calendar["hpo_validations"]) == 3
    assert calendar["outer_folds"] == [list(item) for item in runner.OUTER_FOLDS]
    start, end = runner.FS_VALIDATION
    assert runner.strict_train_predicate(
        "2025-10-30T22:59:00Z",
        "2025-10-30T23:59:00Z",
        start,
    )
    assert not runner.strict_train_predicate(
        "2025-10-30T23:00:00Z",
        "2025-10-31T00:00:00Z",
        start,
    )
    assert not runner.strict_train_predicate(
        "2025-10-30T22:59:00Z",
        "2025-10-31T00:00:00Z",
        start,
    )
    assert runner.strict_validation_predicate(
        "2025-11-01T00:00:00Z",
        "2025-11-01T01:00:00Z",
        start,
        end,
    )
    assert runner.strict_validation_predicate(
        "2025-11-30T23:00:00Z",
        "2025-12-01T00:00:00Z",
        start,
        end,
    )
    assert not runner.strict_validation_predicate(
        "2025-12-01T00:00:00Z",
        "2025-12-01T01:00:00Z",
        start,
        end,
    )
    final_start, final_end = runner.HPO_VALIDATIONS[-1]
    assert runner.strict_validation_predicate(
        "2026-02-27T22:00:00Z",
        "2026-02-27T23:00:00Z",
        final_start,
        final_end,
    )
    assert not runner.strict_validation_predicate(
        "2026-02-27T23:00:00Z",
        "2026-02-28T00:00:00Z",
        final_start,
        final_end,
    )


def test_contract_only_publishes_no_fit_contract_after_preflight(
    tmp_path: Path,
) -> None:
    decisions = tmp_path / "decisions.json"
    _decisions(decisions)
    labels, audit = _labels(tmp_path)
    output = tmp_path / "out"
    report = runner.run_contract_only(
        decisions_path=decisions,
        labels_dir=labels,
        causal_audit_path=audit,
        output_dir=output,
        batch_rows=1,
    )
    assert report["status"] == "CONTRACT_ONLY_READY_FOR_REVIEW"
    assert report["fit_status"] == "NOT_IMPLEMENTED_NO_FALLBACK"
    assert (output / "preparation_contract.json").is_file()
    assert not list(output.glob("**/*.pkl"))


def test_contract_only_fails_closed_on_stale_causal_shard_and_publishes_nothing(
    tmp_path: Path,
) -> None:
    decisions = tmp_path / "decisions.json"
    _decisions(decisions)
    labels, audit = _labels(tmp_path, rows=2)
    payload = json.loads(audit.read_text())
    payload["rows"] = 2
    for item in payload["per_file"]:
        item["rows"] = 1
    _write_json(audit, payload)
    output = tmp_path / "out"
    with pytest.raises(
        runner.PackBPreparationError, match="stale causal audit row count"
    ):
        runner.run_contract_only(
            decisions_path=decisions,
            labels_dir=labels,
            causal_audit_path=audit,
            output_dir=output,
        )
    assert not (output / "preparation_contract.json").exists()


def test_side_stage_manifest_cli_requires_complete_six_file_bundle() -> None:
    values = [
        f"{side}:{stage}=/{side}-{stage}.json"
        for side in runner.CANONICAL_SIDES
        for stage in runner.stage_manifest.CANONICAL_STAGES
    ]

    parsed = runner._parse_side_stage_manifests(values)

    assert parsed is not None
    assert set(parsed) == {"long", "short"}
    assert set(parsed["long"]) == {"ae_gmm", "feature_selection", "hpo"}
    with pytest.raises(runner.PackBPreparationError, match="missing side-stage"):
        runner._parse_side_stage_manifests(values[:-1])


def test_resource_failure_and_fit_mode_are_fail_closed(tmp_path: Path) -> None:
    class FailingGuard:
        def preflight(self, stage: str) -> None:
            raise TrainingResourceGuardError(stage)

        def checkpoint(self, stage: str) -> None:
            return None

    with pytest.raises(TrainingResourceGuardError):
        runner.run_contract_only(
            output_dir=tmp_path / "out", resource_guard=FailingGuard()
        )
    with pytest.raises(NotImplementedError, match="NOT_IMPLEMENTED"):
        runner.run(contract_only=False, output_dir=tmp_path / "fit")
