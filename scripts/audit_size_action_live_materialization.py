#!/usr/bin/env python3
"""Audit whether a frozen size-action champion is deployable as a live scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SCORER_FILES = {
    "scorer_manifest": "size_action_live_scorer_manifest.json",
    "model_bundle": "size_action_live_scorer.joblib",
    "feature_contract": "size_action_live_feature_contract.json",
    "imputation_contract": "size_action_live_imputation.json",
    "policy_contract": "size_action_live_policy_contract.json",
}

RESEARCH_REQUIRED_RUN_FILES = [
    "manifest.json",
    "size_action_promotion_summary.csv",
    "size_action_replay_vs_label_audit.csv",
    "size_action_action_quality.csv",
    "size_action_schedules.csv",
    "size_action_gate_thresholds.csv",
    "size_action_selected_features.csv",
    "size_action_exact_panel.csv",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": bool(path.exists() and path.is_file()),
        "size_bytes": int(path.stat().st_size) if path.exists() and path.is_file() else None,
        "sha256": _sha256(path),
    }


def _resolve_run_dir(freeze_manifest: dict[str, Any], freeze_manifest_path: Path) -> Path | None:
    raw = freeze_manifest.get("run_dir")
    if not raw:
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return path
    cwd_relative = Path.cwd() / path
    if cwd_relative.exists():
        return cwd_relative
    return (freeze_manifest_path.parent / path).resolve()


def _scorer_paths(bundle_dir: Path) -> dict[str, Path]:
    return {name: bundle_dir / rel for name, rel in DEFAULT_SCORER_FILES.items()}


def _audit_scorer_manifest(
    scorer_manifest: dict[str, Any],
    *,
    freeze_manifest: dict[str, Any],
    run_manifest: dict[str, Any],
) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if not scorer_manifest:
        blockers.append("live_scorer_manifest_missing")
        return blockers, warnings

    required_fields = [
        "generated_by",
        "arm",
        "run_dir",
        "feature_columns",
        "model_artifacts",
        "imputation_policy",
        "fail_closed",
        "score_contract",
    ]
    for field in required_fields:
        if field not in scorer_manifest:
            blockers.append(f"live_scorer_manifest_missing_{field}")

    if scorer_manifest.get("arm") != freeze_manifest.get("arm"):
        blockers.append("live_scorer_arm_mismatch")
    if scorer_manifest.get("fail_closed") is not True:
        blockers.append("live_scorer_not_fail_closed")
    if not scorer_manifest.get("feature_columns"):
        blockers.append("live_scorer_empty_feature_columns")
    if not scorer_manifest.get("model_artifacts"):
        blockers.append("live_scorer_empty_model_artifacts")
    if scorer_manifest.get("coverage") != "full_arm":
        blockers.append("live_scorer_not_full_arm_coverage")
    missing_components = scorer_manifest.get("missing_components") or []
    if missing_components:
        blockers.append("live_scorer_missing_components")

    source_manifest = freeze_manifest.get("source_manifest") or {}
    expected_policy = source_manifest.get("policy_variant") or run_manifest.get("policy_variant")
    if expected_policy and scorer_manifest.get("policy_variant") not in {None, expected_policy}:
        blockers.append("live_scorer_policy_variant_mismatch")
    if scorer_manifest.get("mode") == "audit_or_fallback":
        warnings.append("live_scorer_declares_audit_or_fallback_mode")

    return blockers, warnings


def audit_materialization(
    freeze_manifest_path: Path,
    *,
    scorer_bundle_dir: Path | None = None,
) -> dict[str, Any]:
    freeze_manifest_path = freeze_manifest_path.resolve()
    freeze_manifest = _read_json(freeze_manifest_path)
    run_dir = _resolve_run_dir(freeze_manifest, freeze_manifest_path)
    if scorer_bundle_dir is None:
        scorer_bundle_dir = freeze_manifest_path.parent / "live_scorer"
    scorer_bundle_dir = scorer_bundle_dir.resolve()

    gate_status = freeze_manifest.get("gate_status") or {}
    run_manifest = _read_json(run_dir / "manifest.json") if run_dir is not None else {}

    run_files = {
        name: _file_record(run_dir / name) if run_dir is not None else {"path": name, "exists": False, "size_bytes": None, "sha256": None}
        for name in RESEARCH_REQUIRED_RUN_FILES
    }
    scorer_file_paths = _scorer_paths(scorer_bundle_dir)
    scorer_files = {name: _file_record(path) for name, path in scorer_file_paths.items()}
    scorer_manifest = _read_json(scorer_file_paths["scorer_manifest"])

    materialization_blockers: list[str] = []
    readiness_blockers: list[str] = []
    warnings: list[str] = []
    if not freeze_manifest:
        materialization_blockers.append("freeze_manifest_missing_or_invalid")
    if not gate_status.get("research_ready", False):
        readiness_blockers.append("frozen_champion_not_research_ready")
    if run_dir is None or not run_dir.exists():
        materialization_blockers.append("source_run_dir_missing")
    for name, record in run_files.items():
        if not record["exists"]:
            materialization_blockers.append(f"research_run_file_missing:{name}")
    for name, record in scorer_files.items():
        if not record["exists"]:
            materialization_blockers.append(f"live_scorer_file_missing:{name}")

    manifest_blockers, manifest_warnings = _audit_scorer_manifest(
        scorer_manifest,
        freeze_manifest=freeze_manifest,
        run_manifest=run_manifest,
    )
    materialization_blockers.extend(manifest_blockers)
    warnings.extend(manifest_warnings)

    if run_manifest and "outputs" in run_manifest and not scorer_manifest:
        warnings.append("source_run_outputs_are_replay_artifacts_not_live_scorer_artifacts")
    if gate_status.get("production_ready") is False:
        warnings.extend(str(x) for x in gate_status.get("production_blockers", []))

    materialized = not materialization_blockers
    all_blockers = sorted(set(materialization_blockers + readiness_blockers))
    payload = {
        "generated_by": "audit_size_action_live_materialization",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_manifest_path": str(freeze_manifest_path),
        "scorer_bundle_dir": str(scorer_bundle_dir),
        "arm": freeze_manifest.get("arm"),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "research_ready": bool(gate_status.get("research_ready", False)),
        "live_materialized": materialized,
        "production_ready": bool(materialized and gate_status.get("research_ready", False)),
        "blockers": all_blockers,
        "materialization_blockers": sorted(set(materialization_blockers)),
        "readiness_blockers": sorted(set(readiness_blockers)),
        "warnings": sorted(set(warnings)),
        "run_files": run_files,
        "scorer_files": scorer_files,
        "scorer_manifest": scorer_manifest,
        "run_manifest_summary": {
            "generated_by": run_manifest.get("generated_by"),
            "policy_variant": run_manifest.get("policy_variant"),
            "selected_arms": run_manifest.get("selected_arms"),
            "outputs": run_manifest.get("outputs"),
        },
    }
    return payload


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Size-Action Live Materialization Audit",
        "",
        f"Arm: `{payload.get('arm')}`",
        f"Freeze manifest: `{payload.get('freeze_manifest_path')}`",
        f"Scorer bundle: `{payload.get('scorer_bundle_dir')}`",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "## Readiness",
        "",
        f"- Research ready: `{payload.get('research_ready')}`",
        f"- Live materialized: `{payload.get('live_materialized')}`",
        f"- Production ready: `{payload.get('production_ready')}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = payload.get("blockers") or []
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    warnings = payload.get("warnings") or []
    if warnings:
        lines.extend(f"- `{warning}`" for warning in warnings)
    else:
        lines.append("- none")
    lines.extend(["", "## Live Scorer Files", "", "| file | exists | sha256 |", "|---|---:|---|"])
    for name, record in payload.get("scorer_files", {}).items():
        lines.append(f"| `{name}` | `{record['exists']}` | `{record['sha256']}` |")
    lines.extend(["", "## Research Run Files", "", "| file | exists | sha256 |", "|---|---:|---|"])
    for name, record in payload.get("run_files", {}).items():
        lines.append(f"| `{name}` | `{record['exists']}` | `{record['sha256']}` |")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--scorer-bundle-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = audit_materialization(args.freeze_manifest, scorer_bundle_dir=args.scorer_bundle_dir)
    (args.out_dir / "size_action_live_materialization_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    _write_markdown(args.out_dir / "size_action_live_materialization_audit.md", payload)
    print(
        {
            "out_dir": str(args.out_dir),
            "arm": payload.get("arm"),
            "research_ready": payload.get("research_ready"),
            "live_materialized": payload.get("live_materialized"),
            "production_ready": payload.get("production_ready"),
            "blockers": payload.get("blockers"),
        }
    )


if __name__ == "__main__":
    main()
