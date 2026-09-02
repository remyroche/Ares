#!/usr/bin/env python3
"""Assemble a sealed root-cause evidence pack without writing a diagnosis.

The packer only copies previously materialised artifacts.  It does not derive
economics, choose a root cause, or author ``ROOT_CAUSE_DIAGNOSTIC_REPORT.md``.
Both the report and the final root-cause waterfall must be supplied by a later,
separately reviewed diagnostic-authoring step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data_perp/artifacts"
DEFAULT_POINTER = ART / "root_cause_diagnostic_canonical_20260731.json"
DEFAULT_STAGE2 = ART / "root_cause_feature_information_20260731_v4"
DEFAULT_STAGE3 = ART / "root_cause_base_residual_learning_20260731_v1"
DEFAULT_STAGE56 = ART / "root_cause_execution_policy_audit_20260731_v4"
DEFAULT_OUTPUT = ART / "root_cause_final_pack_20260731_v1"
EXPECTED_STAGE3_SCOPE = "base_directional_alpha_and_stopped_gradient_gross_residual_only"
EXPECTED_TWO_HEAD_ARCHITECTURE = ("base_directional_alpha", "stopped_gradient_residual")


class PackContractError(RuntimeError):
    """A source is incomplete, mutated, or outside the approved two-head scope."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PackContractError(f"required JSON is absent: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise PackContractError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PackContractError(f"JSON object expected: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PackContractError(message)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _verify_manifest_output(directory: Path, manifest: Mapping[str, Any], name: str, manifest_key: str) -> Path:
    declared = manifest.get(manifest_key, {})
    _require(isinstance(declared, Mapping) and name in declared, f"{directory.name} manifest lacks output digest for {name}")
    source = directory / name
    _require(source.is_file(), f"{directory.name} output is absent: {name}")
    observed = sha256(source)
    _require(observed == declared[name], f"{directory.name}/{name} digest differs from its manifest")
    return source


def _verify_runner(manifest: Mapping[str, Any], *, name: str, legacy_code_key: str | None = None, expected_path: Path | None = None) -> dict[str, str]:
    """Verify the materialised runner, rejecting manifests that omit it."""
    if legacy_code_key is not None:
        expected = manifest.get(legacy_code_key)
        _require(isinstance(expected, str) and expected, f"{name} manifest lacks {legacy_code_key}")
        _require(expected_path is not None, f"internal packer binding missing for {name}")
        observed = sha256(expected_path)
        _require(observed == expected, f"{name} runner digest differs from the current bound source")
        return {"path": str(expected_path.relative_to(ROOT)), "sha256": observed}
    runner = manifest.get("runner")
    _require(isinstance(runner, Mapping), f"{name} manifest lacks runner path/digest; immutable pack assembly is unsafe")
    runner_path = runner.get("path")
    runner_sha = runner.get("sha256")
    _require(isinstance(runner_path, str) and isinstance(runner_sha, str), f"{name} runner provenance is incomplete")
    source = _resolve(runner_path)
    _require(source.is_file(), f"{name} runner source is absent: {source}")
    observed = sha256(source)
    _require(observed == runner_sha, f"{name} runner digest differs from the current bound source")
    return {"path": str(source.relative_to(ROOT)), "sha256": observed}


def _verify_ledger_input(manifest: Mapping[str, Any], *, name: str, ledger: Path, ledger_sha: str) -> None:
    """Require both the canonical ledger identity and its exact digest."""
    expected = ledger.resolve()
    if name == "stage1":
        _require(_resolve(manifest.get("ledger", "__missing__")) == expected, "Stage1 ledger path is not the pointer-pinned Stage0 ledger")
        _require(manifest.get("ledger_sha256") == ledger_sha, "Stage1 ledger digest differs from Stage0")
        return
    if name == "stage2":
        inputs = manifest.get("inputs_sha256")
        _require(isinstance(inputs, Mapping) and inputs.get("ledger_sha256") == ledger_sha, "Stage2 ledger digest differs from Stage0")
        return
    if name == "stage3":
        inputs = manifest.get("inputs")
        _require(isinstance(inputs, Mapping), "Stage3 manifest lacks input digests")
        matches = [(path, digest) for path, digest in inputs.items() if _resolve(path) == expected]
        _require(len(matches) == 1 and matches[0][1] == ledger_sha, "Stage3 canonical ledger path/digest differs from Stage0")
        return
    if name == "stage56":
        _require(_resolve(manifest.get("input", "__missing__")) == expected, "Stage5/6 ledger path is not the pointer-pinned Stage0 ledger")
        _require(manifest.get("input_sha256") == ledger_sha, "Stage5/6 ledger digest differs from Stage0")
        return
    raise AssertionError(name)


def _copy_checked(source: Path, destination: Path) -> str:
    shutil.copy2(source, destination)
    source_hash = sha256(source)
    output_hash = sha256(destination)
    _require(source_hash == output_hash, f"copy digest mismatch for {source.name}")
    return output_hash


def _atomic_directory(output: Path) -> Path:
    staging = output.with_name(f".{output.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    return staging


def assemble(
    *, pointer_path: Path = DEFAULT_POINTER, stage2_dir: Path = DEFAULT_STAGE2, stage3_dir: Path = DEFAULT_STAGE3,
    stage56_dir: Path = DEFAULT_STAGE56, root_cause_waterfall: Path | None = None,
    diagnostic_report: Path | None = None, output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Validate all stages and copy the exact terminal artifact names atomically."""
    _require(not output.exists(), f"refusing to overwrite immutable pack: {output}")
    _require(root_cause_waterfall is not None and root_cause_waterfall.is_file(), "root_cause_waterfall.parquet must be supplied by a later diagnostic-authoring step")
    _require(diagnostic_report is not None and diagnostic_report.is_file(), "ROOT_CAUSE_DIAGNOSTIC_REPORT.md must be supplied by a later diagnostic-authoring step")

    pointer = _read_json(pointer_path)
    stage0_dir = _resolve(pointer.get("stage0_substrate", "__missing__"))
    stage1_dir = _resolve(pointer.get("stage1_oracle_ladder", "__missing__"))
    stage0_manifest_path = stage0_dir / "diagnostic_population_manifest.json"
    stage1_manifest_path = stage1_dir / "run_manifest.json"
    _require(sha256(stage0_manifest_path) == pointer.get("stage0_manifest_sha256"), "canonical pointer Stage0 manifest digest mismatch")
    _require(sha256(stage1_manifest_path) == pointer.get("stage1_manifest_sha256"), "canonical pointer Stage1 manifest digest mismatch")
    stage0 = _read_json(stage0_manifest_path)
    stage1 = _read_json(stage1_manifest_path)
    stage2 = _read_json(stage2_dir / "run_manifest.json")
    stage3 = _read_json(stage3_dir / "run_manifest.json")
    stage56 = _read_json(stage56_dir / "run_manifest.json")

    ledger = stage0_dir / "diagnostic_row_ledger.parquet"
    ledger_sha = _verify_manifest_output(stage0_dir, stage0, "diagnostic_row_ledger.parquet", "outputs_sha256") and sha256(ledger)
    runner_provenance = {
        "stage0": _verify_runner(stage0, name="Stage0"),
        "stage1": _verify_runner(stage1, name="Stage1"),
        "stage2": _verify_runner(stage2, name="Stage2", legacy_code_key="code_sha256", expected_path=ROOT / "scripts/run_root_cause_feature_information_audit.py"),
        "stage3": _verify_runner(stage3, name="Stage3"),
        "stage56": _verify_runner(stage56, name="Stage5/6"),
    }
    for stage_name, manifest in (("stage1", stage1), ("stage2", stage2), ("stage3", stage3), ("stage56", stage56)):
        _verify_ledger_input(manifest, name=stage_name, ledger=ledger, ledger_sha=ledger_sha)

    _require(stage3.get("scope") == EXPECTED_STAGE3_SCOPE, "Stage3 scope is not the approved base-plus-stopped-gradient-residual scope")
    invariants = stage3.get("invariants", {})
    _require(isinstance(invariants, Mapping) and invariants.get("no_auxiliary_or_policy_layers") is True, "Stage3 two-head scope admits auxiliary or policy layers")
    architecture = tuple(stage56.get("architecture", ()))
    _require(architecture == EXPECTED_TWO_HEAD_ARCHITECTURE, f"Stage5/6 architecture violates the two-head scope: {architecture}")
    checks = stage56.get("checks", {})
    _require(isinstance(checks, Mapping) and checks.get("action_head_disabled") is True and checks.get("base_and_residual_reported_separately") is True, "Stage5/6 action/head-scope gates failed")

    copy_plan = {
        "oracle_ladder_results.parquet": _verify_manifest_output(stage1_dir, stage1, "oracle_ladder_results.parquet", "outputs_sha256"),
        "feature_information_results.parquet": _verify_manifest_output(stage2_dir, stage2, "feature_information_results.parquet", "outputs_sha256"),
        "model_learning_efficiency.parquet": _verify_manifest_output(stage3_dir, stage3, "model_learning_efficiency.parquet", "outputs_sha256"),
        "metric_concordance.parquet": _verify_manifest_output(stage3_dir, stage3, "metric_concordance.parquet", "outputs_sha256"),
        "execution_waterfall.parquet": _verify_manifest_output(stage56_dir, stage56, "execution_waterfall.parquet", "outputs"),
        "policy_regret.parquet": _verify_manifest_output(stage56_dir, stage56, "policy_regret.parquet", "outputs"),
        "root_cause_waterfall.parquet": root_cause_waterfall,
        "ROOT_CAUSE_DIAGNOSTIC_REPORT.md": diagnostic_report,
    }
    stage = _atomic_directory(output)
    try:
        output_hashes = {name: _copy_checked(source, stage / name) for name, source in copy_plan.items()}
        source_hashes = {name: sha256(source) for name, source in copy_plan.items()}
        correctness = {
            "schema": "root_cause_final_pack_correctness_v1",
            "status": "PASS",
            "diagnostic_authoring": "external input copied byte-for-byte; packer writes no economic diagnosis",
            "pointer": str(pointer_path),
            "stage_dirs": {"stage0": str(stage0_dir), "stage1": str(stage1_dir), "stage2": str(stage2_dir), "stage3": str(stage3_dir), "stage56": str(stage56_dir)},
            "source_manifest_hashes": {
                "canonical_pointer": sha256(pointer_path),
                "stage0": sha256(stage0_manifest_path),
                "stage1": sha256(stage1_manifest_path),
                "stage2": sha256(stage2_dir / "run_manifest.json"),
                "stage3": sha256(stage3_dir / "run_manifest.json"),
                "stage56": sha256(stage56_dir / "run_manifest.json"),
            },
            "ledger_sha256": ledger_sha,
            "runner_provenance": runner_provenance,
            "two_head_scope": {"stage3_scope": stage3["scope"], "stage56_architecture": list(architecture), "action_head_disabled": checks["action_head_disabled"]},
            "source_hashes": source_hashes,
            "output_hashes": output_hashes,
        }
        (stage / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2, sort_keys=True) + "\n")
        output_hashes["correctness_test_report.json"] = sha256(stage / "correctness_test_report.json")
        run_manifest = {
            "schema": "root_cause_final_pack_v1",
            "status": "ASSEMBLED_DIAGNOSTIC_ONLY_NO_PROMOTION",
            "runner": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256(Path(__file__))},
            "correctness_report_sha256": output_hashes["correctness_test_report.json"],
            **correctness,
        }
        (stage / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
        (stage / "manifest.sha256").write_text(sha256(stage / "run_manifest.json") + "\n")
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--stage2", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--stage3", type=Path, default=DEFAULT_STAGE3)
    parser.add_argument("--stage56", type=Path, default=DEFAULT_STAGE56)
    parser.add_argument("--root-cause-waterfall", type=Path, required=True)
    parser.add_argument("--diagnostic-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(assemble(pointer_path=args.pointer, stage2_dir=args.stage2, stage3_dir=args.stage3, stage56_dir=args.stage56, root_cause_waterfall=args.root_cause_waterfall, diagnostic_report=args.diagnostic_report, output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
