#!/usr/bin/env python3
"""Generate a sealed, conclusion-free root-cause report/waterfall scaffold.

The runner validates required evidence and renders only a pending-diagnosis
template. It never ranks root causes or invents unavailable execution values.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from scripts.assemble_root_cause_final_pack import (
    ART, DEFAULT_POINTER, DEFAULT_STAGE2, DEFAULT_STAGE3, DEFAULT_STAGE56,
    EXPECTED_STAGE3_SCOPE, EXPECTED_TWO_HEAD_ARCHITECTURE,
    ROOT, _read_json, _require, _resolve, _verify_ledger_input,
    _verify_manifest_output, _verify_runner, sha256,
)


DEFAULT_GLOBAL_LEARNING = ART / "root_cause_global_learning_economics_20260731_v1"
DEFAULT_OUTPUT = ART / "root_cause_diagnostic_scaffold_20260731_v1"
TERMINAL_CLASSES = (
    "TARGET_OR_POPULATION_FAILURE",
    "CAUSAL_FEATURE_INFORMATION_INSUFFICIENT",
    "ML_LEARNING_EFFICIENCY_FAILURE",
    "METRIC_SELECTION_MISALIGNMENT",
    "EXECUTION_TRANSFER_FAILURE",
    "COST_DRAG_FAILURE",
    "POLICY_CONVERSION_FAILURE",
)


def _atomic_directory(output: Path) -> Path:
    staging = output.with_name(f".{output.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    return staging


def _require_global(global_dir: Path, stage3_manifest_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest = _read_json(global_dir / "run_manifest.json")
    _require(manifest.get("status") == "COMPLETE_DIAGNOSTIC_ONLY", "global-learning output is not complete diagnostic evidence")
    _require(manifest.get("selection_scope") == "GLOBAL_TOP_K_NOT_PER_TIMESTAMP_OR_SIDE", "global-learning output is not exact global top-k")
    _require(manifest.get("stage3_manifest_sha256") == sha256(stage3_manifest_path), "global-learning Stage3 manifest digest differs from supplied Stage3 evidence")
    _verify_runner(manifest, name="Global-learning")
    names = (
        "global_topk_learning_economics.parquet",
        "global_topk_learning_gaps.parquet",
        "causal_only_global_metric_concordance.parquet",
    )
    return manifest, {name: _verify_manifest_output(global_dir, manifest, name, "outputs_sha256") for name in names}


def _render_report(*, sources: Mapping[str, Path], source_hashes: Mapping[str, str], runner_provenance: Mapping[str, Any]) -> str:
    registry = "\n".join(f"| {name} | {path} | {source_hashes[name]} |" for name, path in sources.items())
    pending = "\n".join(f"| {name} | PENDING_HUMAN_EVIDENCE_SYNTHESIS |" for name in TERMINAL_CLASSES)
    return f"""# Root-Cause Diagnostic Report — Evidence Scaffold

Status: SCAFFOLD_ONLY — NO ECONOMIC DIAGNOSIS OR PROMOTION DECISION

This deterministic document only registers sealed evidence. It does not rank
root causes, convert unavailable execution components into zero, or infer an
economic conclusion. A later human-reviewed authoring step must complete it.

## Verified source registry

| Evidence artifact | Source path | SHA-256 |
|---|---|---|
{registry}

## Locked terminal classifications

| Classification | Status |
|---|---|
{pending}

## Waterfall status

root_cause_waterfall.parquet preserves every original Stage-5 waterfall row
and column value, while adding provenance columns. It is evidence only: no
root-cause ranking, ideal/delayed entry synthesis, or inferred counterfactual
value.

## Scope gate

Validated runner provenance: {json.dumps(runner_provenance, sort_keys=True)}.
The approved architecture is base directional alpha plus stopped-gradient gross
residual only. Auxiliary, meta, timing, and action heads remain out of scope.
"""


def generate(
    *, pointer_path: Path = DEFAULT_POINTER, stage2_dir: Path = DEFAULT_STAGE2,
    stage3_dir: Path = DEFAULT_STAGE3, stage56_dir: Path = DEFAULT_STAGE56,
    global_learning_dir: Path = DEFAULT_GLOBAL_LEARNING, output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Validate evidence and render only a pending-diagnosis scaffold."""
    _require(not output.exists(), f"refusing to overwrite immutable scaffold: {output}")
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
    stage3_manifest_path = stage3_dir / "run_manifest.json"
    stage3 = _read_json(stage3_manifest_path)
    stage56 = _read_json(stage56_dir / "run_manifest.json")

    ledger = _verify_manifest_output(stage0_dir, stage0, "diagnostic_row_ledger.parquet", "outputs_sha256")
    ledger_sha = sha256(ledger)
    runner_provenance = {
        "stage0": _verify_runner(stage0, name="Stage0"),
        "stage1": _verify_runner(stage1, name="Stage1"),
        "stage2": _verify_runner(stage2, name="Stage2", legacy_code_key="code_sha256", expected_path=ROOT / "scripts/run_root_cause_feature_information_audit.py"),
        "stage3": _verify_runner(stage3, name="Stage3"),
        "stage56": _verify_runner(stage56, name="Stage5/6"),
    }
    for name, manifest in (("stage1", stage1), ("stage2", stage2), ("stage3", stage3), ("stage56", stage56)):
        _verify_ledger_input(manifest, name=name, ledger=ledger, ledger_sha=ledger_sha)
    _require(stage3.get("scope") == EXPECTED_STAGE3_SCOPE, "Stage3 scope is not the approved two-head scope")
    invariants = stage3.get("invariants", {})
    _require(isinstance(invariants, Mapping) and invariants.get("no_auxiliary_or_policy_layers") is True, "Stage3 admits auxiliary or policy layers")
    architecture = tuple(stage56.get("architecture", ()))
    _require(architecture == EXPECTED_TWO_HEAD_ARCHITECTURE, f"Stage5/6 violates two-head architecture: {architecture}")
    checks = stage56.get("checks", {})
    _require(isinstance(checks, Mapping) and checks.get("action_head_disabled") is True, "Stage5/6 action-head disable gate failed")

    global_manifest, global_outputs = _require_global(global_learning_dir, stage3_manifest_path)
    runner_provenance["global_learning"] = _verify_runner(global_manifest, name="Global-learning")
    sources = {
        "stage1_oracle_ladder_results": _verify_manifest_output(stage1_dir, stage1, "oracle_ladder_results.parquet", "outputs_sha256"),
        "stage2_feature_information_results": _verify_manifest_output(stage2_dir, stage2, "feature_information_results.parquet", "outputs_sha256"),
        "stage3_model_learning_efficiency": _verify_manifest_output(stage3_dir, stage3, "model_learning_efficiency.parquet", "outputs_sha256"),
        "stage3_metric_concordance": _verify_manifest_output(stage3_dir, stage3, "metric_concordance.parquet", "outputs_sha256"),
        "global_learning_economics": global_outputs["global_topk_learning_economics.parquet"],
        "global_learning_gaps": global_outputs["global_topk_learning_gaps.parquet"],
        "global_metric_concordance": global_outputs["causal_only_global_metric_concordance.parquet"],
        "stage56_execution_waterfall": _verify_manifest_output(stage56_dir, stage56, "execution_waterfall.parquet", "outputs"),
        "stage56_policy_regret": _verify_manifest_output(stage56_dir, stage56, "policy_regret.parquet", "outputs"),
    }
    source_hashes = {name: sha256(path) for name, path in sources.items()}
    execution = pd.read_parquet(sources["stage56_execution_waterfall"])
    needed = {"record_type", "stage", "slice", "score", "rows", "value_bps_per_candidate", "status", "detail"}
    missing = needed.difference(execution.columns)
    _require(not missing, f"Stage5/6 execution waterfall lacks required columns: {sorted(missing)}")
    root_waterfall = execution.copy()
    root_waterfall.insert(0, "source_stage", "stage5_6_execution_policy_audit")
    root_waterfall.insert(1, "source_artifact", str(sources["stage56_execution_waterfall"]))
    root_waterfall.insert(2, "source_sha256", source_hashes["stage56_execution_waterfall"])
    root_waterfall.insert(3, "diagnostic_interpretation", "PENDING_HUMAN_EVIDENCE_SYNTHESIS")
    root_waterfall.insert(4, "root_cause_rank", pd.NA)

    stage = _atomic_directory(output)
    try:
        root_waterfall.to_parquet(stage / "root_cause_waterfall.parquet", index=False)
        (stage / "ROOT_CAUSE_DIAGNOSTIC_REPORT.md").write_text(_render_report(sources=sources, source_hashes=source_hashes, runner_provenance=runner_provenance))
        output_hashes = {
            "root_cause_waterfall.parquet": sha256(stage / "root_cause_waterfall.parquet"),
            "ROOT_CAUSE_DIAGNOSTIC_REPORT.md": sha256(stage / "ROOT_CAUSE_DIAGNOSTIC_REPORT.md"),
        }
        correctness = {
            "schema": "root_cause_diagnostic_scaffold_correctness_v1",
            "status": "PASS_SCAFFOLD_ONLY_NO_DIAGNOSIS",
            "diagnosis_status": "PENDING_HUMAN_EVIDENCE_SYNTHESIS",
            "ledger_sha256": ledger_sha,
            "runner_provenance": runner_provenance,
            "source_hashes": source_hashes,
            "source_manifest_hashes": {
                "pointer": sha256(pointer_path), "stage0": sha256(stage0_manifest_path), "stage1": sha256(stage1_manifest_path),
                "stage2": sha256(stage2_dir / "run_manifest.json"), "stage3": sha256(stage3_manifest_path),
                "stage56": sha256(stage56_dir / "run_manifest.json"), "global_learning": sha256(global_learning_dir / "run_manifest.json"),
            },
            "output_hashes": output_hashes,
            "two_head_scope": {"stage3_scope": stage3["scope"], "stage56_architecture": list(architecture), "action_head_disabled": checks["action_head_disabled"]},
        }
        (stage / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2, sort_keys=True) + "\n")
        output_hashes["correctness_test_report.json"] = sha256(stage / "correctness_test_report.json")
        manifest = {
            **correctness,
            "schema": "root_cause_diagnostic_scaffold_v1",
            "status": "SCAFFOLD_ONLY_NO_ECONOMIC_DIAGNOSIS_OR_PROMOTION",
            "runner": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256(Path(__file__))},
            "correctness_report_sha256": output_hashes["correctness_test_report.json"],
        }
        (stage / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (stage / "manifest.sha256").write_text(sha256(stage / "run_manifest.json") + "\n")
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--stage2", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--stage3", type=Path, default=DEFAULT_STAGE3)
    parser.add_argument("--stage56", type=Path, default=DEFAULT_STAGE56)
    parser.add_argument("--global-learning", type=Path, default=DEFAULT_GLOBAL_LEARNING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(generate(pointer_path=args.pointer, stage2_dir=args.stage2, stage3_dir=args.stage3, stage56_dir=args.stage56, global_learning_dir=args.global_learning, output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
