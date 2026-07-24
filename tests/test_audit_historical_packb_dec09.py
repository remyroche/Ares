"""Focused tests for the read-only Pack-B DEC-09 provenance auditor."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/audit_historical_packb_dec09.py"
SPEC = importlib.util.spec_from_file_location("packb_audit", SCRIPT)
assert SPEC and SPEC.loader
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _decisions() -> dict:
    return {
        "schema_version": "full_pipeline_decisions_v1",
        "status": "LOCKED_BEFORE_NEW_TRAINING",
        "decisions": {
            "DEC-09": {
                "decision_timestamp": "signal_timestamp + 1 hour",
                "stage_label_horizons": audit_module.EXPECTED_HORIZONS,
                "train_authorization": audit_module.EXPECTED_AUTHORIZATION,
                "signal_timestamp_purge_hours": 25,
                "purge_hours": 12,
                "post_label_embargo_hours": 12,
                "outer_folds": [
                    list(fold) for fold in audit_module.EXPECTED_OUTER_FOLDS
                ],
                "feature_selection_hpo_resolution_cutoff_utc": "2026-03-01T00:00:00Z",
            }
        },
    }


def _sources(
    tmp_path: Path,
    *,
    scope: str = "per_side",
    rows: int = 100,
    inventory_rows: int = 100,
) -> tuple[Path, Path]:
    report_dir = tmp_path / "packb"
    _write_json(
        report_dir / "manifest.json",
        {
            "rows": rows,
            "feature_selection_calibration_fold": "2026-02-01_2026-03-01",
            "hpo_calibration_fold": "2026-02-01_2026-03-01",
            "model_side_scope": scope,
        },
    )
    _write_json(
        report_dir / "models/final_all_rows/ae_gmm_state/source_state_manifest.json",
        {
            "cycle_reference_end": "2026-02-28T23:00:00+00:00",
        },
    )
    labels = tmp_path / "labels"
    _write_json(
        labels / "labels_manifest.json",
        {
            "run_id": "labels-test",
            "datasets": {"one": {"rows": inventory_rows, "file": "one.parquet"}},
        },
    )
    return report_dir, labels


def test_parse_dec09_rejects_changed_packb_horizon(tmp_path: Path) -> None:
    decisions = _decisions()
    decisions["decisions"]["DEC-09"]["stage_label_horizons"] = {
        "packb_directional_base": "decision_timestamp + 12 hours"
    }
    path = tmp_path / "decisions.json"
    _write_json(path, decisions)
    with pytest.raises(ValueError, match="stage label horizons"):
        audit_module.parse_dec09(path)


def test_historical_post_cutoff_sources_are_blocked(tmp_path: Path) -> None:
    decisions_path = tmp_path / "decisions.json"
    _write_json(decisions_path, _decisions())
    report_dir, labels = _sources(tmp_path, rows=100, inventory_rows=99)
    report = json.loads((report_dir / "manifest.json").read_text())
    report["feature_selection_calibration_fold"] = "2026-06-30_2026-07-30"
    report["hpo_calibration_fold"] = "2026-06-30_2026-07-30"
    _write_json(report_dir / "manifest.json", report)
    state_path = (
        report_dir / "models/final_all_rows/ae_gmm_state/source_state_manifest.json"
    )
    _write_json(state_path, {"cycle_reference_end": "2026-06-25T23:00:00+00:00"})
    result = audit_module.audit_historical_packb(decisions_path, report_dir, labels)
    assert result["status"] == "BLOCKED_HISTORICAL_COMPARATOR_ONLY"
    assert {item["name"] for item in result["blockers"]} == {
        "feature_selection_calibration_fold_start",
        "hpo_calibration_fold_start",
        "ae_gmm_cycle_reference_end",
        "label_shard_inventory_matches_report_rows",
    }
    assert result["canonical_oof_reuse_allowed"] is False


def test_pooled_side_scope_is_a_blocker(tmp_path: Path) -> None:
    decisions_path = tmp_path / "decisions.json"
    _write_json(decisions_path, _decisions())
    report_dir, labels = _sources(tmp_path, scope="shared")
    result = audit_module.audit_historical_packb(decisions_path, report_dir, labels)
    assert [item["name"] for item in result["blockers"]] == [
        "model_side_scope_is_per_side"
    ]
    assert result["blockers"][0]["reason"] == "pooled_or_unknown_side_scope"


def test_stale_shard_inventory_is_a_blocker(tmp_path: Path) -> None:
    decisions_path = tmp_path / "decisions.json"
    _write_json(decisions_path, _decisions())
    report_dir, labels = _sources(tmp_path, rows=100, inventory_rows=101)
    result = audit_module.run(
        decisions_path=decisions_path, packb_report_dir=report_dir, labels_path=labels
    )
    assert [item["name"] for item in result["blockers"]] == [
        "label_shard_inventory_matches_report_rows"
    ]
    blocker = result["blockers"][0]
    assert blocker["report_rows"] == 100
    assert blocker["inventory_rows"] == 101
