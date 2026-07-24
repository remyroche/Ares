#!/usr/bin/env python3
"""Fail-closed contract gate for the DEC-09 pre-March Pack-B preparation.

This runner intentionally does not train.  It locks the only admissible
feature-selection/HPO calendar, verifies an exact causal label inventory, and
can validate previously emitted side-local artifacts.  A later reviewed fit
implementation must consume this contract; it must not silently substitute a
historical Pack-B run or a pooled-side fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements import packb_pre_march_source_authorization as source_auth
from extreme_price_movements import packb_side_stage_manifest as stage_manifest
from extreme_price_movements.training_resource_guard import TrainingResourceGuard

RUNNER_SCHEMA = "packb_pre_march_side_contract_runner_v1"
CANONICAL_SIDES = ("long", "short")
RESOLUTION_CUTOFF = pd.Timestamp("2026-03-01T00:00:00Z")
DECISION_LAG = pd.Timedelta(hours=1)
BASE_LABEL_HORIZON = pd.Timedelta(hours=24)
SIGNAL_PURGE = pd.Timedelta(hours=25)
FS_VALIDATION = (
    pd.Timestamp("2025-11-01T00:00:00Z"),
    pd.Timestamp("2025-12-01T00:00:00Z"),
)
HPO_VALIDATIONS = (
    (pd.Timestamp("2025-12-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-02-01T00:00:00Z")),
    (pd.Timestamp("2026-02-01T00:00:00Z"), pd.Timestamp("2026-03-01T00:00:00Z")),
)
OUTER_FOLDS = (
    ("2026-04-01T00:00:00Z", "2026-05-01T00:00:00Z"),
    ("2026-05-01T00:00:00Z", "2026-06-01T00:00:00Z"),
    ("2026-06-01T00:00:00Z", "2026-07-01T00:00:00Z"),
    ("2026-07-01T00:00:00Z", "2026-07-11T00:00:00Z"),
)
DEFAULT_DECISIONS = ROOT / "config/full_pipeline_decisions_20260724.json"
DEFAULT_LABELS = (
    ROOT
    / "data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
)
DEFAULT_AUDIT = DEFAULT_LABELS / "causal_path_invariant_audit.json"
DEFAULT_FEATURE_STORE_INVENTORY = (
    ROOT / "docs/pipeline_roadmap/20260724/r0/migration_inventory.json"
)


class PackBPreparationError(ValueError):
    """Raised when canonical pre-March preparation cannot proceed safely."""


def _utc(value: Any, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise PackBPreparationError(f"{name} must include an explicit UTC offset")
    timestamp = timestamp.tz_convert("UTC")
    if pd.isna(timestamp):
        raise PackBPreparationError(f"{name} is not a valid timestamp")
    return timestamp


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackBPreparationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PackBPreparationError(f"JSON object required: {path}")
    return payload


def _source_revision() -> str:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise PackBPreparationError("cannot resolve source revision") from exc
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise PackBPreparationError("source revision is not a full Git commit SHA")
    if dirty:
        raise PackBPreparationError(
            "post-fit validation requires a clean tracked source revision"
        )
    return revision


def _feature_store_inventory_evidence(path: Path) -> dict[str, str]:
    payload = _json(path)
    inventory = payload.get("inventory")
    items = inventory.get("items") if isinstance(inventory, dict) else None
    if not isinstance(items, list):
        raise PackBPreparationError("feature-store inventory has no inventory.items")
    matches = [
        item
        for item in items
        if isinstance(item, dict) and item.get("id") == "canonical_feature_store"
    ]
    if len(matches) != 1:
        raise PackBPreparationError(
            "feature-store inventory must contain one canonical_feature_store"
        )
    digest = str(matches[0].get("sha256", "")).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise PackBPreparationError(
            "canonical feature-store inventory SHA-256 is invalid"
        )
    return {
        "feature_store_inventory_sha256": digest,
        "feature_store_inventory_evidence_sha256": _sha256(path),
        "feature_store_inventory_evidence_path": str(path),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


def locked_calendar() -> dict[str, Any]:
    """Return the literal, non-configurable DEC-09 preparation calendar."""
    windows = [("feature_selection", *FS_VALIDATION)] + [
        (f"hpo_{index}", start, end)
        for index, (start, end) in enumerate(HPO_VALIDATIONS, start=1)
    ]
    return {
        "resolution_cutoff_utc": RESOLUTION_CUTOFF.isoformat(),
        "base_label_resolution": "actual __decision_ts__ + 24 hours",
        "ae_gmm_reference_signal_interval": [
            "2025-01-01T00:00:00+00:00",
            "2025-11-01T00:00:00+00:00",
        ],
        "train_rule": "signal < validation_start - 25h AND decision == signal + 1h AND decision + 24h < validation_start",
        "validation_rule": "validation_start <= signal < validation_end AND decision == signal + 1h AND decision + 24h < resolution_cutoff",
        "feature_selection_validation": [item.isoformat() for item in FS_VALIDATION],
        "hpo_validations": [
            [item.isoformat() for item in window] for window in HPO_VALIDATIONS
        ],
        "outer_folds": [list(window) for window in OUTER_FOLDS],
        "windows": [
            {
                "name": name,
                "validation_start": start.isoformat(),
                "validation_end": end.isoformat(),
            }
            for window in windows
            for name, start, end in [window]
        ],
        "fallback": "FORBIDDEN_NO_POOLED_SIDE_OR_HISTORICAL_ARTIFACT_FALLBACK",
    }


def parse_locked_dec09(path: Path = DEFAULT_DECISIONS) -> dict[str, Any]:
    payload = _json(path)
    decision = payload.get("decisions", {}).get("DEC-09")
    if (
        payload.get("schema_version") != "full_pipeline_decisions_v1"
        or payload.get("status") != "LOCKED_BEFORE_NEW_TRAINING"
    ):
        raise PackBPreparationError("locked full_pipeline_decisions_v1 is required")
    if not isinstance(decision, dict):
        raise PackBPreparationError("decisions.DEC-09 is required")
    if (
        _utc(
            decision.get("feature_selection_hpo_resolution_cutoff_utc"),
            name="DEC-09 cutoff",
        )
        != RESOLUTION_CUTOFF
    ):
        raise PackBPreparationError("DEC-09 pre-March cutoff changed")
    if decision.get("decision_timestamp") != "signal_timestamp + 1 hour":
        raise PackBPreparationError("DEC-09 decision timestamp contract changed")
    if (
        decision.get("signal_timestamp_purge_hours") != 25
        or decision.get("purge_hours") != 12
    ):
        raise PackBPreparationError("DEC-09 purge contract changed")
    actual_outer = tuple(
        tuple(window)
        for window in decision.get("outer_folds", ())
        if isinstance(window, list)
    )
    if actual_outer != OUTER_FOLDS:
        raise PackBPreparationError("DEC-09 outer-fold calendar changed")
    inner = decision.get("packb_pre_march_inner_calendar")
    expected_inner = {
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
    }
    if not isinstance(inner, dict) or any(
        inner.get(key) != value for key, value in expected_inner.items()
    ):
        raise PackBPreparationError("DEC-09 pre-March inner calendar changed")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "decision": decision,
        "calendar": locked_calendar(),
    }


def strict_base_resolution(decision_timestamp: Any) -> pd.Timestamp:
    """Resolve the base label from the recorded decision time, never signal time."""
    return _utc(decision_timestamp, name="decision timestamp") + BASE_LABEL_HORIZON


def strict_train_predicate(
    signal_timestamp: Any,
    decision_timestamp: Any,
    validation_start: Any,
) -> bool:
    signal = _utc(signal_timestamp, name="signal timestamp")
    decision = _utc(decision_timestamp, name="decision timestamp")
    start = _utc(validation_start, name="validation start")
    return bool(
        decision == signal + DECISION_LAG
        and signal < start - SIGNAL_PURGE
        and strict_base_resolution(decision) < start
    )


def strict_validation_predicate(
    signal_timestamp: Any,
    decision_timestamp: Any,
    validation_start: Any,
    validation_end: Any,
) -> bool:
    signal = _utc(signal_timestamp, name="signal timestamp")
    decision = _utc(decision_timestamp, name="decision timestamp")
    start, end = (
        _utc(validation_start, name="validation start"),
        _utc(validation_end, name="validation end"),
    )
    return bool(
        decision == signal + DECISION_LAG
        and start <= signal < end
        and strict_base_resolution(decision) < RESOLUTION_CUTOFF
    )


def _validate_audit_row_counts(
    labels_dir: Path,
    causal_audit_path: Path,
    *,
    checkpoint: Any | None = None,
) -> None:
    """Reject tail-mutated shards before the streaming population preflight."""
    audit = _json(causal_audit_path)
    if audit.get("schema") == "packb_current_canonical_label_inventory_audit_v1":
        if audit.get("status") != "PASS" or audit.get("mode") != "streaming_full_audit":
            raise PackBPreparationError(
                "current canonical label audit must be a passing full audit"
            )
    entries = audit.get("per_file")
    if not isinstance(entries, list) or not entries:
        raise PackBPreparationError("causal audit has no per_file inventory")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - required project dependency.
        raise PackBPreparationError(
            "pyarrow is required for exact shard metadata"
        ) from exc
    total = 0
    for item in entries:
        if not isinstance(item, dict) or not isinstance(item.get("file"), str):
            raise PackBPreparationError("causal audit has invalid per_file entry")
        try:
            expected_value = (
                item["rows"] if "rows" in item else item["expected_current_rows"]
            )
            expected = int(expected_value)
        except (KeyError, TypeError, ValueError) as exc:
            raise PackBPreparationError(
                "causal audit per_file rows are invalid"
            ) from exc
        path = labels_dir / item["file"]
        if checkpoint is not None:
            checkpoint(f"before_audit_shard_metadata:{path.name}")
        if not path.is_file():
            raise PackBPreparationError(f"causal audit shard missing: {path.name}")
        actual = int(pq.ParquetFile(path).metadata.num_rows)
        if actual != expected:
            raise PackBPreparationError(
                f"stale causal audit row count for {path.name}: audit={expected} actual={actual}"
            )
        total += actual
    totals = audit.get("totals")
    declared_total = (
        totals.get("rows") if isinstance(totals, dict) else audit.get("rows", -1)
    )
    if int(declared_total) != total:
        raise PackBPreparationError(
            "causal audit total row count does not match its per_file inventory"
        )


def run_contract_only(
    *,
    decisions_path: Path = DEFAULT_DECISIONS,
    labels_dir: Path = DEFAULT_LABELS,
    causal_audit_path: Path = DEFAULT_AUDIT,
    output_dir: Path,
    batch_rows: int = source_auth.DEFAULT_BATCH_ROWS,
    side_stage_manifests: Mapping[str, Mapping[str, Path]] | None = None,
    authorized_population_ledger: Path | None = None,
    feature_store_inventory_manifest: Path = DEFAULT_FEATURE_STORE_INVENTORY,
    resource_guard: TrainingResourceGuard | None = None,
) -> dict[str, Any]:
    """Publish only a successful fixed-calendar/preflight contract; never fit."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PackBPreparationError(f"output directory must be empty: {output_dir}")
    guard = resource_guard or TrainingResourceGuard(
        disk_path=output_dir.parent,
        telemetry_path=output_dir.parent / f".{output_dir.name}.resource.jsonl",
    )
    guard.preflight("packb_pre_march_contract_preflight")
    dec09 = parse_locked_dec09(Path(decisions_path))
    _validate_audit_row_counts(
        Path(labels_dir),
        Path(causal_audit_path),
        checkpoint=guard.checkpoint,
    )
    population = source_auth.preflight_pre_march_packb_population(
        labels_dir=Path(labels_dir),
        causal_audit_path=Path(causal_audit_path),
        batch_rows=int(batch_rows),
        checkpoint=guard.checkpoint,
        scratch_dir=output_dir.parent,
    )
    expected_source_hashes: dict[str, str] | None = None
    fixed_calendar_sha256 = stage_manifest.canonical_json_sha256(dec09["calendar"])
    if side_stage_manifests is not None:
        if authorized_population_ledger is None:
            raise PackBPreparationError(
                "authorized_population_ledger is required with post-fit manifests"
            )
        population_ledger_path = Path(authorized_population_ledger)
        if not population_ledger_path.is_file():
            raise PackBPreparationError("authorized_population_ledger does not exist")
        feature_store_evidence = _feature_store_inventory_evidence(
            Path(feature_store_inventory_manifest)
        )
        revision = _source_revision()
        expected_source_hashes = {
            "dec09_decisions_sha256": dec09["sha256"],
            "canonical_shard_inventory_sha256": population["label_inventory"][
                "canonical_shard_inventory_sha256"
            ],
            "causal_audit_sha256": population["label_inventory"]["causal_audit_sha256"],
            "population_preflight_sha256": stage_manifest.canonical_json_sha256(
                population
            ),
            "authorized_population_ledger_sha256": _sha256(population_ledger_path),
            "feature_store_inventory_sha256": feature_store_evidence[
                "feature_store_inventory_sha256"
            ],
            "feature_store_inventory_evidence_sha256": feature_store_evidence[
                "feature_store_inventory_evidence_sha256"
            ],
        }
        verified = stage_manifest.validate_side_stage_manifest_bundle(
            side_stage_manifests,
            expected_source_revision=revision,
            expected_source_hashes=expected_source_hashes,
            expected_fixed_calendar_sha256=fixed_calendar_sha256,
        )
    else:
        feature_store_evidence = None
        revision = None
        verified = None
    guard.checkpoint("packb_pre_march_contract_complete")
    report = {
        "schema": RUNNER_SCHEMA,
        "status": "CONTRACT_ONLY_READY_FOR_REVIEW",
        "fit_status": "NOT_IMPLEMENTED_NO_FALLBACK",
        "canonical_oof_status": "NOT_STARTED",
        "dec09": dec09,
        "population_preflight": population,
        "fixed_calendar_sha256": fixed_calendar_sha256,
        "expected_source_hashes": expected_source_hashes,
        "source_revision": revision,
        "feature_store_inventory_evidence": feature_store_evidence,
        "side_artifact_verification": verified,
        "publication": "contract_only_json; no AE/GMM, feature, parameter, model, or OOF artifact was created",
    }
    _atomic_json(output_dir / "preparation_contract.json", report)
    return report


def run(*, contract_only: bool, **kwargs: Any) -> dict[str, Any]:
    if not contract_only:
        raise NotImplementedError(
            "NOT_IMPLEMENTED: side-local AE/GMM, FS, and HPO fitting requires a separately reviewed implementation; no fallback is permitted"
        )
    return run_contract_only(**kwargs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="Run the fixed-calendar and exact-inventory gate only.",
    )
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--causal-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--batch-rows", type=int, default=source_auth.DEFAULT_BATCH_ROWS
    )
    parser.add_argument(
        "--side-stage-manifest",
        action="append",
        default=[],
        metavar="SIDE:STAGE=PATH",
    )
    parser.add_argument("--authorized-population-ledger", type=Path)
    parser.add_argument(
        "--feature-store-inventory-manifest",
        type=Path,
        default=DEFAULT_FEATURE_STORE_INVENTORY,
    )
    return parser.parse_args(argv)


def _parse_side_stage_manifests(
    values: Sequence[str],
) -> dict[str, dict[str, Path]] | None:
    if not values:
        return None
    parsed: dict[str, dict[str, Path]] = {side: {} for side in CANONICAL_SIDES}
    for value in values:
        scope, separator, path = value.partition("=")
        side, colon, stage = scope.partition(":")
        if (
            separator != "="
            or colon != ":"
            or side not in CANONICAL_SIDES
            or stage not in stage_manifest.CANONICAL_STAGES
            or not path
        ):
            raise PackBPreparationError("--side-stage-manifest must be SIDE:STAGE=PATH")
        if stage in parsed[side]:
            raise PackBPreparationError(
                f"duplicate side-stage manifest: {side}:{stage}"
            )
        parsed[side][stage] = Path(path)
    for side in CANONICAL_SIDES:
        missing = sorted(set(stage_manifest.CANONICAL_STAGES) - set(parsed[side]))
        if missing:
            raise PackBPreparationError(
                f"missing side-stage manifests for {side}: {missing}"
            )
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(
            contract_only=bool(args.contract_only),
            decisions_path=args.decisions,
            labels_dir=args.labels_dir,
            causal_audit_path=args.causal_audit,
            output_dir=args.output_dir,
            batch_rows=int(args.batch_rows),
            side_stage_manifests=_parse_side_stage_manifests(args.side_stage_manifest),
            authorized_population_ledger=args.authorized_population_ledger,
            feature_store_inventory_manifest=args.feature_store_inventory_manifest,
        )
    except NotImplementedError as exc:
        print(
            json.dumps({"status": "NOT_IMPLEMENTED", "error": str(exc)}),
            file=sys.stderr,
        )
        return 3
    except (
        PackBPreparationError,
        source_auth.PackBSourceAuthorizationError,
        ValueError,
    ) as exc:
        print(
            json.dumps({"status": "BLOCKED_PRECONDITION_FAILED", "error": str(exc)}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
