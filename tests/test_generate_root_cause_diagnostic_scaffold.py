"""Focused fail-closed tests for the conclusion-free diagnostic scaffold."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pandas as pd
import pytest

from scripts.assemble_root_cause_final_pack import PackContractError, sha256
from scripts.generate_root_cause_diagnostic_scaffold import generate


REPO = Path(__file__).resolve().parents[1]


def _fixture_sources(tmp_path: Path) -> dict[str, Path]:
    helpers = runpy.run_path(str(REPO / "tests/test_assemble_root_cause_final_pack.py"))
    sources = helpers["_sources"](tmp_path)
    stage3_manifest_path = sources["stage3"] / "run_manifest.json"
    stage3_manifest = json.loads(stage3_manifest_path.read_text())

    execution_path = sources["stage56"] / "execution_waterfall.parquet"
    pd.DataFrame([{
        "record_type": "waterfall", "stage": "B_executable_entry_gross",
        "slice": "full_population", "score": "none", "rows": 3,
        "value_bps_per_candidate": -12.0, "status": "OBSERVED", "detail": "fixture",
    }]).to_parquet(execution_path)
    stage56_manifest_path = sources["stage56"] / "run_manifest.json"
    stage56_manifest = json.loads(stage56_manifest_path.read_text())
    stage56_manifest["outputs"]["execution_waterfall.parquet"] = sha256(execution_path)
    stage56_manifest_path.write_text(json.dumps(stage56_manifest, indent=2, sort_keys=True) + "\n")

    global_dir = tmp_path / "global"
    global_dir.mkdir()
    names = (
        "global_topk_learning_economics.parquet",
        "global_topk_learning_gaps.parquet",
        "causal_only_global_metric_concordance.parquet",
    )
    for name in names:
        pd.DataFrame([{"fixture": 1}]).to_parquet(global_dir / name)
    global_manifest = {
        "status": "COMPLETE_DIAGNOSTIC_ONLY",
        "selection_scope": "GLOBAL_TOP_K_NOT_PER_TIMESTAMP_OR_SIDE",
        "stage3_manifest_sha256": sha256(stage3_manifest_path),
        "runner": {
            "path": "scripts/materialize_root_cause_global_learning_economics.py",
            "sha256": sha256(REPO / "scripts/materialize_root_cause_global_learning_economics.py"),
        },
        "outputs_sha256": {name: sha256(global_dir / name) for name in names},
    }
    (global_dir / "run_manifest.json").write_text(json.dumps(global_manifest, indent=2, sort_keys=True) + "\n")
    sources["global"] = global_dir
    return sources


def _generate(sources: dict[str, Path], output: Path) -> dict:
    return generate(
        pointer_path=sources["pointer"],
        stage2_dir=sources["stage2"],
        stage3_dir=sources["stage3"],
        stage56_dir=sources["stage56"],
        global_learning_dir=sources["global"],
        output=output,
    )


def test_scaffold_is_pending_only_and_preserves_execution_evidence(tmp_path: Path) -> None:
    sources = _fixture_sources(tmp_path)
    output = tmp_path / "scaffold"
    manifest = _generate(sources, output)
    assert manifest["status"] == "SCAFFOLD_ONLY_NO_ECONOMIC_DIAGNOSIS_OR_PROMOTION"
    report = (output / "ROOT_CAUSE_DIAGNOSTIC_REPORT.md").read_text()
    assert "PENDING_HUMAN_EVIDENCE_SYNTHESIS" in report
    assert "TARGET_OR_POPULATION_FAILURE | PENDING_HUMAN_EVIDENCE_SYNTHESIS" in report
    waterfall = pd.read_parquet(output / "root_cause_waterfall.parquet")
    assert waterfall.diagnostic_interpretation.eq("PENDING_HUMAN_EVIDENCE_SYNTHESIS").all()
    assert waterfall.root_cause_rank.isna().all()
    assert waterfall.value_bps_per_candidate.iloc[0] == pytest.approx(-12.0)


def test_scaffold_fails_closed_on_invalid_global_or_missing_stage3(tmp_path: Path) -> None:
    sources = _fixture_sources(tmp_path)
    global_manifest_path = sources["global"] / "run_manifest.json"
    global_manifest = json.loads(global_manifest_path.read_text())
    global_manifest["stage3_manifest_sha256"] = "wrong"
    global_manifest_path.write_text(json.dumps(global_manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(PackContractError, match="Stage3 manifest digest"):
        _generate(sources, tmp_path / "bad-global")
    assert not (tmp_path / "bad-global").exists()

    sources = _fixture_sources(tmp_path / "missing")
    (sources["stage3"] / "run_manifest.json").unlink()
    with pytest.raises(PackContractError, match="required JSON is absent"):
        _generate(sources, tmp_path / "bad-stage3")
    assert not (tmp_path / "bad-stage3").exists()
