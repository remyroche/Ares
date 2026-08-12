from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from extreme_price_movements.leaf_reasoning_portability_report import (
    DECISION_NAME,
    LeafReasoningPortabilityReportError,
    REPORT_NAME,
    STATUS,
    assemble_feature_leaf_reasoning_portability_report,
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_stage(root: Path, stage: str, *, lineage: dict[str, str] | None = None) -> Path:
    root.mkdir(parents=True)
    evidence = root / "evidence.json"
    evidence.write_text(json.dumps({"stage": stage, "sealed": True}, sort_keys=True), encoding="utf-8")
    payload: dict[str, object] = {
        "schema": "test_leaf_reasoning_stage_v1",
        "stage": stage,
        "status": "SEALED_DEVELOPMENT_EVIDENCE",
        "immutable_output": True,
        "artifact_state": "COMPLETE",
        "development_only": True,
        "development_evaluation_end_utc": "2024-11-01T00:00:00Z",
        "final_november_oos_consumed": False,
        "final_oos_used_for_selection": False,
        "sha256": {"evidence.json": _sha(evidence)},
    }
    if stage == "S":
        payload.update({
            "selected_arm": "C6",
            "successor": "S1",
            "terminal_decision": "COMPACT_REASONING_GENERATION_SELECTED",
            "stage_manifest_sha256": lineage,
        })
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def _sealed_stages(tmp_path: Path) -> dict[str, Path]:
    roots = {stage: _write_stage(tmp_path / stage, stage) for stage in ("A", "L", "H", "C")}
    hashes = {stage: _sha(root / "manifest.json") for stage, root in roots.items()}
    roots["S"] = _write_stage(tmp_path / "S", "S", lineage=hashes)
    return roots


def test_report_assembles_only_hash_bound_development_stage_artifacts(tmp_path: Path) -> None:
    stages = _sealed_stages(tmp_path / "inputs")
    result = assemble_feature_leaf_reasoning_portability_report(stages, tmp_path / "assembled")

    assert result.terminal_decision == "COMPACT_REASONING_GENERATION_SELECTED"
    assert result.report_path.name == REPORT_NAME
    report = result.report_path.read_text(encoding="utf-8")
    assert "final November OOS" in report
    decision = json.loads((result.output_dir / DECISION_NAME).read_text(encoding="utf-8"))
    assert decision["status"] == STATUS
    assert decision["final_november_oos_consumed"] is False
    assert decision["stage_manifest_sha256"]["H"] == _sha(stages["H"] / "manifest.json")
    manifest = json.loads((result.output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["sha256"]) == {REPORT_NAME, DECISION_NAME}


def test_report_fails_closed_on_missing_or_unsealed_prerequisites(tmp_path: Path) -> None:
    stages = _sealed_stages(tmp_path / "inputs")
    missing = {stage: root for stage, root in stages.items() if stage != "H"}
    with pytest.raises(LeafReasoningPortabilityReportError, match="exactly stages"):
        assemble_feature_leaf_reasoning_portability_report(missing, tmp_path / "missing")

    manifest_path = stages["H"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_state"] = "WRITING"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(LeafReasoningPortabilityReportError, match="not complete/sealed"):
        assemble_feature_leaf_reasoning_portability_report(stages, tmp_path / "unsealed")
    assert not (tmp_path / "unsealed").exists()


def test_report_rejects_final_oos_contamination_and_broken_successor_lineage(tmp_path: Path) -> None:
    stages = _sealed_stages(tmp_path / "inputs")
    h_manifest = stages["H"] / "manifest.json"
    payload = json.loads(h_manifest.read_text(encoding="utf-8"))
    payload["final_oos_labels_used"] = True
    h_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LeafReasoningPortabilityReportError, match="contaminated"):
        assemble_feature_leaf_reasoning_portability_report(stages, tmp_path / "contaminated")
    assert not (tmp_path / "contaminated").exists()

    stages = _sealed_stages(tmp_path / "other")
    s_manifest = stages["S"] / "manifest.json"
    payload = json.loads(s_manifest.read_text(encoding="utf-8"))
    payload["stage_manifest_sha256"]["C"] = "0" * 64
    s_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LeafReasoningPortabilityReportError, match="lineage hash"):
        assemble_feature_leaf_reasoning_portability_report(stages, tmp_path / "bad-lineage")
