"""Focused sealing tests for the root-cause final-pack assembler."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.assemble_root_cause_final_pack import PackContractError, assemble


REPO = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _runner(script: str) -> dict[str, str]:
    path = REPO / script
    return {"path": script, "sha256": _sha(path)}


def _artifact(directory: Path, name: str, content: bytes) -> str:
    path = directory / name
    path.write_bytes(content)
    return _sha(path)


def _sources(tmp_path: Path) -> dict[str, Path]:
    stage0, stage1, stage2, stage3, stage56 = [tmp_path / name for name in ("stage0", "stage1", "stage2", "stage3", "stage56")]
    for directory in (stage0, stage1, stage2, stage3, stage56):
        directory.mkdir(parents=True)
    ledger_sha = _artifact(stage0, "diagnostic_row_ledger.parquet", b"canonical-ledger")
    stage0_manifest = {"runner": _runner("scripts/materialize_root_cause_diagnostic_substrate.py"), "outputs_sha256": {"diagnostic_row_ledger.parquet": ledger_sha}}
    _write_json(stage0 / "diagnostic_population_manifest.json", stage0_manifest)

    oracle_sha = _artifact(stage1, "oracle_ladder_results.parquet", b"oracle")
    stage1_manifest = {"runner": _runner("scripts/run_root_cause_oracle_ladder.py"), "ledger": str(stage0 / "diagnostic_row_ledger.parquet"), "ledger_sha256": ledger_sha, "outputs_sha256": {"oracle_ladder_results.parquet": oracle_sha}}
    _write_json(stage1 / "run_manifest.json", stage1_manifest)

    feature_sha = _artifact(stage2, "feature_information_results.parquet", b"features")
    stage2_manifest = {"code_sha256": _sha(REPO / "scripts/run_root_cause_feature_information_audit.py"), "inputs_sha256": {"ledger_sha256": ledger_sha}, "outputs_sha256": {"feature_information_results.parquet": feature_sha}}
    _write_json(stage2 / "run_manifest.json", stage2_manifest)

    efficiency_sha = _artifact(stage3, "model_learning_efficiency.parquet", b"efficiency")
    concordance_sha = _artifact(stage3, "metric_concordance.parquet", b"concordance")
    stage3_manifest = {
        "runner": _runner("scripts/run_root_cause_base_residual_learning.py"),
        "scope": "base_directional_alpha_and_stopped_gradient_gross_residual_only",
        "invariants": {"no_auxiliary_or_policy_layers": True},
        "inputs": {str(stage0 / "diagnostic_row_ledger.parquet"): ledger_sha},
        "outputs_sha256": {"model_learning_efficiency.parquet": efficiency_sha, "metric_concordance.parquet": concordance_sha},
    }
    _write_json(stage3 / "run_manifest.json", stage3_manifest)

    waterfall_sha = _artifact(stage56, "execution_waterfall.parquet", b"execution")
    regret_sha = _artifact(stage56, "policy_regret.parquet", b"policy")
    stage56_manifest = {
        "runner": _runner("scripts/run_root_cause_execution_policy_audit.py"),
        "input": str(stage0 / "diagnostic_row_ledger.parquet"), "input_sha256": ledger_sha,
        "architecture": ["base_directional_alpha", "stopped_gradient_residual"],
        "checks": {"action_head_disabled": True, "base_and_residual_reported_separately": True},
        "outputs": {"execution_waterfall.parquet": waterfall_sha, "policy_regret.parquet": regret_sha},
    }
    _write_json(stage56 / "run_manifest.json", stage56_manifest)

    pointer = tmp_path / "pointer.json"
    _write_json(pointer, {
        "stage0_substrate": str(stage0), "stage0_manifest_sha256": _sha(stage0 / "diagnostic_population_manifest.json"),
        "stage1_oracle_ladder": str(stage1), "stage1_manifest_sha256": _sha(stage1 / "run_manifest.json"),
    })
    root_waterfall = tmp_path / "root_cause_waterfall.parquet"; root_waterfall.write_bytes(b"final-waterfall")
    report = tmp_path / "ROOT_CAUSE_DIAGNOSTIC_REPORT.md"; report.write_text("# Supplied later\n")
    return {"pointer": pointer, "stage2": stage2, "stage3": stage3, "stage56": stage56, "waterfall": root_waterfall, "report": report}


def _assemble(sources: dict[str, Path], output: Path) -> dict:
    return assemble(pointer_path=sources["pointer"], stage2_dir=sources["stage2"], stage3_dir=sources["stage3"], stage56_dir=sources["stage56"], root_cause_waterfall=sources["waterfall"], diagnostic_report=sources["report"], output=output)


def test_assembler_copies_only_verified_terminal_artifacts(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    output = tmp_path / "pack"
    manifest = _assemble(sources, output)
    expected = {
        "ROOT_CAUSE_DIAGNOSTIC_REPORT.md", "root_cause_waterfall.parquet", "oracle_ladder_results.parquet",
        "feature_information_results.parquet", "model_learning_efficiency.parquet", "metric_concordance.parquet",
        "execution_waterfall.parquet", "policy_regret.parquet", "correctness_test_report.json", "run_manifest.json", "manifest.sha256",
    }
    assert expected == {path.name for path in output.iterdir()}
    assert manifest["two_head_scope"]["stage56_architecture"] == ["base_directional_alpha", "stopped_gradient_residual"]
    assert (output / "ROOT_CAUSE_DIAGNOSTIC_REPORT.md").read_text() == sources["report"].read_text()


def test_assembler_fails_closed_on_runner_or_two_head_scope_mismatch(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    stage56_manifest_path = sources["stage56"] / "run_manifest.json"
    stage56 = json.loads(stage56_manifest_path.read_text())
    stage56["architecture"].append("auxiliary_head")
    _write_json(stage56_manifest_path, stage56)
    with pytest.raises(PackContractError, match="two-head scope"):
        _assemble(sources, tmp_path / "bad-scope")
    assert not (tmp_path / "bad-scope").exists()

    sources = _sources(tmp_path / "runner")
    stage56_manifest_path = sources["stage56"] / "run_manifest.json"
    stage56 = json.loads(stage56_manifest_path.read_text())
    del stage56["runner"]
    _write_json(stage56_manifest_path, stage56)
    with pytest.raises(PackContractError, match="lacks runner"):
        _assemble(sources, tmp_path / "bad-runner")
    assert not (tmp_path / "bad-runner").exists()
