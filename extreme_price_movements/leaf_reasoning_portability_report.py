"""Assemble the sealed leaf-reasoning portability terminal decision.

This is deliberately a *consumer*, not a selector or a replay runner.  It
accepts only five already-sealed development artifacts (A/L/H/C/S), verifies
their declared file hashes and pre-November provenance, then republishes a
small hash-bound decision/report pair.  It never opens a candidate panel,
metrics table, or final-OOS result, so it cannot turn a reporting step into
post-selection tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import pandas as pd

from .leaf_reasoning_final_oos import DEVELOPMENT_CUTOFF
from .leaf_reasoning_meta_funnel import S_ALLOWED


SCHEMA = "feature_leaf_reasoning_portability_terminal_decision_v1"
STATUS = "SEALED_DEVELOPMENT_PORTABILITY_TERMINAL_DECISION"
REPORT_NAME = "FEATURE_LEAF_REASONING_PORTABILITY_REPORT.md"
DECISION_NAME = "FEATURE_LEAF_REASONING_PORTABILITY_TERMINAL_DECISION.json"
MANIFEST_NAME = "run_manifest.json"
STAGES = ("A", "L", "H", "C", "S")
_FINAL_OOS_BOOLEAN_KEYS = frozenset((
    "final_november_oos_consumed",
    "final_oos_consumed",
    "final_oos_used_for_selection",
    "final_oos_labels_used",
    "final_oos_used_for_feature_selection",
    "final_oos_used_for_hpo",
    "final_oos_used_for_tuning",
))


class LeafReasoningPortabilityReportError(ValueError):
    """Raised when an input is not a sealed, development-only stage artifact."""


@dataclass(frozen=True)
class SealedStageArtifact:
    """Validated immutable stage manifest used by the terminal assembler."""

    stage: str
    root: Path
    manifest_path: Path
    manifest_sha256: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class LeafReasoningPortabilityReportResult:
    """Published report and hash-bound terminal decision paths."""

    output_dir: Path
    report_path: Path
    decision_path: Path
    manifest_path: Path
    terminal_decision: str


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: object, *, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise LeafReasoningPortabilityReportError(f"{label} must be a finite UTC timestamp") from exc
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if pd.isna(timestamp):
        raise LeafReasoningPortabilityReportError(f"{label} must be a finite UTC timestamp")
    return timestamp


def _json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LeafReasoningPortabilityReportError(f"{label} must be a readable JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise LeafReasoningPortabilityReportError(f"{label} JSON artifact must be an object")
    return value


def _manifest_path(value: str | os.PathLike[str], *, stage: str) -> Path:
    source = Path(value).resolve()
    if source.is_file():
        return source
    if not source.is_dir():
        raise LeafReasoningPortabilityReportError(f"stage {stage} artifact is absent: {source}")
    for name in ("manifest.json", "run_manifest.json", "terminal_decision.json", "successor_decision.json"):
        candidate = source / name
        if candidate.is_file():
            return candidate
    raise LeafReasoningPortabilityReportError(
        f"stage {stage} artifact has no explicit immutable manifest under {source}"
    )


def _safe_child(path: object, *, root: Path, stage: str) -> Path:
    candidate = Path(str(path))
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise LeafReasoningPortabilityReportError(
            f"stage {stage} hash manifest names a file outside its artifact root"
        ) from exc
    return resolved


def _hash_declarations(payload: Mapping[str, Any], *, stage: str) -> Mapping[str, Any]:
    for name in ("sha256", "outputs", "outputs_sha256"):
        value = payload.get(name)
        if isinstance(value, Mapping) and value:
            return value
    raise LeafReasoningPortabilityReportError(
        f"stage {stage} is not sealed: it lacks non-empty declared output SHA-256 hashes"
    )


def _verify_hashes(payload: Mapping[str, Any], *, root: Path, stage: str) -> None:
    declarations = _hash_declarations(payload, stage=stage)
    for relative, expected_raw in declarations.items():
        expected = str(expected_raw).lower()
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            raise LeafReasoningPortabilityReportError(
                f"stage {stage} has an invalid declared SHA-256 for {relative!r}"
            )
        path = _safe_child(relative, root=root, stage=stage)
        if not path.is_file() or _sha256_file(path) != expected:
            raise LeafReasoningPortabilityReportError(
                f"stage {stage} sealed output hash mismatch: {path}"
            )


def _walk_final_oos_flags(value: object, *, path: str = "") -> list[str]:
    """Return any explicit final-OOS use flag that is not the boolean false."""

    failures: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            current = f"{path}.{key}" if path else str(key)
            if str(key) in _FINAL_OOS_BOOLEAN_KEYS and child is not False:
                failures.append(current)
            failures.extend(_walk_final_oos_flags(child, path=current))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            failures.extend(_walk_final_oos_flags(child, path=f"{path}[{index}]"))
    return failures


def _proves_declared_stage(payload: Mapping[str, Any], *, stage: str) -> bool:
    if str(payload.get("stage", payload.get("stage_name", ""))).upper() == stage:
        return True
    stages = payload.get("stages")
    if isinstance(stages, Mapping):
        return str(stages.get(stage, "")).lower() in {"complete", "completed", "sealed"}
    if isinstance(stages, (list, tuple)):
        return stage in {str(item).upper() for item in stages}
    # These two immutable producer contracts predate a one-letter stage field.
    if stage == "C" and str(payload.get("status", "")) == "STRICT_OOF_C5_C6_TAXONOMY_FINALIZED":
        return True
    if stage == "S" and bool(str(payload.get("terminal_decision", "")).strip()):
        return True
    return False


def _development_end(payload: Mapping[str, Any], *, stage: str) -> pd.Timestamp:
    for key in ("development_evaluation_end_utc", "evaluation_end_utc", "development_cutoff_utc"):
        if key in payload:
            end = _utc(payload[key], label=f"stage {stage}.{key}")
            if end != DEVELOPMENT_CUTOFF:
                raise LeafReasoningPortabilityReportError(
                    f"stage {stage} must end exactly at the final-OOS boundary "
                    f"{DEVELOPMENT_CUTOFF.isoformat()}"
                )
            return end
    raise LeafReasoningPortabilityReportError(
        f"stage {stage} does not explicitly bind its development evaluation end to the final-OOS boundary"
    )


def _validate_stage(stage: str, value: str | os.PathLike[str]) -> SealedStageArtifact:
    manifest_path = _manifest_path(value, stage=stage)
    root = manifest_path.parent.resolve()
    payload = _json(manifest_path, label=f"stage {stage} manifest")
    if payload.get("immutable_output") is not True:
        raise LeafReasoningPortabilityReportError(f"stage {stage} is not an immutable sealed output")
    if str(payload.get("artifact_state", "")).upper() != "COMPLETE":
        raise LeafReasoningPortabilityReportError(f"stage {stage} is not complete/sealed")
    if payload.get("development_only") is not True:
        raise LeafReasoningPortabilityReportError(f"stage {stage} must explicitly declare development_only=true")
    if payload.get("final_november_oos_consumed") is not False:
        raise LeafReasoningPortabilityReportError(
            f"stage {stage} must explicitly retain untouched final November OOS"
        )
    contaminated = _walk_final_oos_flags(payload)
    if contaminated:
        raise LeafReasoningPortabilityReportError(
            f"stage {stage} is contaminated by final-OOS use: {sorted(set(contaminated))}"
        )
    _development_end(payload, stage=stage)
    if not _proves_declared_stage(payload, stage=stage):
        raise LeafReasoningPortabilityReportError(f"stage {stage} manifest does not prove its declared stage role")
    _verify_hashes(payload, root=root, stage=stage)
    return SealedStageArtifact(
        stage=stage,
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=_sha256_file(manifest_path),
        payload=payload,
    )


def _validate_successor_lineage(stages: Mapping[str, SealedStageArtifact]) -> tuple[str, str, str]:
    successor = stages["S"].payload
    links = successor.get("stage_manifest_sha256", successor.get("predecessor_manifest_sha256"))
    if not isinstance(links, Mapping):
        raise LeafReasoningPortabilityReportError(
            "stage S must hash-bind the A/L/H/C stage manifests before a terminal decision can be assembled"
        )
    expected = set(STAGES).difference({"S"})
    normalized_links = {str(key).upper(): value for key, value in links.items()}
    actual = set(normalized_links)
    if actual != expected:
        raise LeafReasoningPortabilityReportError(
            f"stage S lineage must bind exactly {sorted(expected)}, got {sorted(actual)}"
    )
    for stage in sorted(expected):
        declared = str(normalized_links[stage]).lower()
        if declared != stages[stage].manifest_sha256:
            raise LeafReasoningPortabilityReportError(
                f"stage S lineage hash does not match sealed stage {stage}"
            )
    selected_arm = str(successor.get("selected_arm", "")).strip()
    terminal_decision = str(successor.get("terminal_decision", "")).strip()
    successor_name = str(successor.get("successor", "")).upper()
    if not selected_arm or not terminal_decision:
        raise LeafReasoningPortabilityReportError("stage S must declare selected_arm and a terminal_decision")
    if successor_name not in S_ALLOWED:
        raise LeafReasoningPortabilityReportError(f"stage S successor must be one of {S_ALLOWED}")
    return selected_arm, successor_name, terminal_decision


def _report_markdown(
    *, stages: Mapping[str, SealedStageArtifact], selected_arm: str, successor: str, terminal_decision: str,
) -> str:
    rows = "\n".join(
        f"| {stage} | `{item.manifest_path}` | `{item.manifest_sha256}` | development-only; final OOS untouched |"
        for stage, item in ((stage, stages[stage]) for stage in STAGES)
    )
    return "\n".join((
        "# Feature leaf-reasoning portability report",
        "",
        "## Terminal decision",
        "",
        f"`{terminal_decision}`",
        "",
        f"Selected development arm: `{selected_arm}`  ",
        f"Selected successor generation: `{successor}`",
        "",
        "This is a development-only immutable-artifact assembly. It did not read, select on, fit on, or replay final November OOS.",
        "",
        "## Sealed inputs",
        "",
        "| Stage | Manifest | SHA-256 | Final-OOS provenance |",
        "| --- | --- | --- | --- |",
        rows,
        "",
        "## Permitted next action",
        "",
        "A separate hash-bound finalization and one-time final-OOS replay may consume this decision. This report is not a replay authorization, fit, HPO, feature-selection, clustering, or policy result.",
        "",
    ))


def assemble_feature_leaf_reasoning_portability_report(
    stage_artifacts: Mapping[str, str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
) -> LeafReasoningPortabilityReportResult:
    """Write the final report/decision from five pre-sealed development stages.

    ``stage_artifacts`` must contain exactly A/L/H/C/S artifact roots (or
    explicit manifest paths).  The function fails before creating ``output``
    if any artifact is missing, partially written, hash-mismatched, outside
    the development cutoff, or has consumed final OOS.
    """

    supplied = {str(key).upper(): value for key, value in stage_artifacts.items()}
    if set(supplied) != set(STAGES):
        raise LeafReasoningPortabilityReportError(
            f"terminal report requires exactly stages {list(STAGES)}, got {sorted(supplied)}"
        )
    stages = {stage: _validate_stage(stage, supplied[stage]) for stage in STAGES}
    selected_arm, successor, terminal_decision = _validate_successor_lineage(stages)

    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable portability terminal report: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        report_path = temporary / REPORT_NAME
        decision_path = temporary / DECISION_NAME
        report_path.write_text(
            _report_markdown(
                stages=stages, selected_arm=selected_arm, successor=successor,
                terminal_decision=terminal_decision,
            ),
            encoding="utf-8",
        )
        decision = {
            "schema": SCHEMA,
            "status": STATUS,
            "immutable_output": True,
            "artifact_state": "COMPLETE",
            "development_only": True,
            "development_evaluation_end_utc": DEVELOPMENT_CUTOFF.isoformat(),
            "final_november_oos_consumed": False,
            "final_oos_used_for_selection": False,
            "selected_arm": selected_arm,
            "successor": successor,
            "terminal_decision": terminal_decision,
            "stage_manifest_sha256": {stage: stages[stage].manifest_sha256 for stage in STAGES},
            "contract": {
                "consumer_only": True,
                "final_oos": "not read, selected on, fit on, or replayed",
                "next_action": "separate hash-bound finalization and one-time final-OOS replay only",
            },
        }
        decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs = {
            REPORT_NAME: _sha256_file(report_path),
            DECISION_NAME: _sha256_file(decision_path),
        }
        manifest = {
            "schema": SCHEMA,
            "status": STATUS,
            "immutable_output": True,
            "artifact_state": "COMPLETE",
            "development_only": True,
            "development_evaluation_end_utc": DEVELOPMENT_CUTOFF.isoformat(),
            "final_november_oos_consumed": False,
            "final_oos_used_for_selection": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "terminal_decision_artifact": DECISION_NAME,
            "terminal_decision": terminal_decision,
            "selected_arm": selected_arm,
            "successor": successor,
            "source_stage_manifests": {
                stage: {
                    "path": str(stages[stage].manifest_path),
                    "sha256": stages[stage].manifest_sha256,
                }
                for stage in STAGES
            },
            "sha256": outputs,
        }
        (temporary / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return LeafReasoningPortabilityReportResult(
        output_dir=target,
        report_path=target / REPORT_NAME,
        decision_path=target / DECISION_NAME,
        manifest_path=target / MANIFEST_NAME,
        terminal_decision=terminal_decision,
    )


__all__ = [
    "DECISION_NAME",
    "LeafReasoningPortabilityReportError",
    "LeafReasoningPortabilityReportResult",
    "MANIFEST_NAME",
    "REPORT_NAME",
    "SCHEMA",
    "STATUS",
    "SealedStageArtifact",
    "assemble_feature_leaf_reasoning_portability_report",
]
