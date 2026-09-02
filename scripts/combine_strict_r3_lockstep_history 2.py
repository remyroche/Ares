#!/usr/bin/env python3
"""Combine immutable strict-R3 lockstep ledgers without changing identities.

The canonical downstream trust layers need history before their reported
evaluation window.  This utility combines non-overlapping held ledgers and
their producer-local reference reserves while preserving every upstream,
conversion, geometry, and activation identity.  It does not refit models,
recompute scores, or join outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


HELD_FILE = "walkforward_scored_label_ledger.parquet"
REFERENCE_FILE = "immediate_calibration_reference_scores.parquet"
BLOCK_AUDIT_FILE = "lockstep_block_audit.parquet"
REFERENCE_IDENTITY = (
    "candidate_id",
    "conversion_bundle_sha256",
    "upstream_bundle_sha256",
    "calibration_activation_ts",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(directory: Path) -> dict[str, object]:
    path = directory / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text())
    if int(manifest.get("reference_window_days", -1)) != 28:
        raise ValueError(f"{directory} is not a prior-28 lockstep ledger")
    if int(manifest.get("held_percentile_operations", -1)) != 0:
        raise ValueError(f"{directory} contains held-window percentile operations")
    geometry = manifest.get("geometry", {})
    if not isinstance(geometry, dict) or geometry.get("refit_cadence") != "never":
        raise ValueError(f"{directory} does not use frozen geometry/K9")
    if manifest.get("outcomes_consumed_during_scoring") != []:
        raise ValueError(f"{directory} consumed outcomes during scoring")
    return manifest


def _utc(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in (
        "__decision_ts__", "calibration_activation_ts", "policy_label_available_ts",
    ):
        if column in output:
            output[column] = pd.to_datetime(output[column], utc=True, errors="coerce")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lockstep-dir", type=Path, action="append", required=True,
        help="Immutable lockstep directory; repeat in chronological order.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.lockstep_dir) < 2:
        raise ValueError("at least two --lockstep-dir inputs are required")
    if args.out_dir.exists():
        raise FileExistsError(f"immutable combined history exists: {args.out_dir}")

    directories = [path.resolve() for path in args.lockstep_dir]
    manifests = [_load_manifest(path) for path in directories]
    geometry_hashes = {
        str(manifest["geometry"]["parent_bundle_sha256"])  # type: ignore[index]
        for manifest in manifests
    }
    if len(geometry_hashes) != 1:
        raise ValueError("lockstep histories do not share one frozen Geometry/K9 identity")

    held_parts: list[pd.DataFrame] = []
    reference_parts: list[pd.DataFrame] = []
    block_audit_parts: list[pd.DataFrame] = []
    source_audit: list[dict[str, object]] = []
    for directory, manifest in zip(directories, manifests, strict=True):
        held_path = directory / HELD_FILE
        reference_path = directory / REFERENCE_FILE
        block_audit_path = directory / BLOCK_AUDIT_FILE
        if (
            not held_path.is_file()
            or not reference_path.is_file()
            or not block_audit_path.is_file()
        ):
            raise FileNotFoundError(f"{directory} lacks canonical lockstep ledgers")
        held = _utc(pd.read_parquet(held_path))
        reference = _utc(pd.read_parquet(reference_path))
        block_audit = _utc(pd.read_parquet(block_audit_path))
        if not held["stack_is_prequential"].fillna(False).astype(bool).all():
            raise ValueError(f"{directory} contains non-prequential held rows")
        reference_flag = "calibration_reference_oos_to_all_active_fits"
        if reference_flag not in reference:
            raise ValueError(f"{directory} reference rows lack the OOS-to-active-fits flag")
        if not reference[reference_flag].fillna(False).astype(bool).all():
            raise ValueError(f"{directory} contains non-OOS reference rows")
        held_parts.append(held)
        reference_parts.append(reference)
        block_audit_parts.append(block_audit)
        source_audit.append({
            "directory": str(directory),
            "manifest_sha256": _sha(directory / "run_manifest.json"),
            "held_sha256": _sha(held_path),
            "reference_sha256": _sha(reference_path),
            "block_audit_sha256": _sha(block_audit_path),
            "held_rows": int(len(held)),
            "reference_rows": int(len(reference)),
            "evaluation_start": manifest.get("evaluation_start"),
            "evaluation_end_exclusive": manifest.get("evaluation_end_exclusive"),
        })

    held = pd.concat(held_parts, ignore_index=True, sort=False).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    reference = pd.concat(reference_parts, ignore_index=True, sort=False).sort_values(
        ["calibration_activation_ts", "__decision_ts__", "candidate_id"], kind="stable",
    ).reset_index(drop=True)
    block_audit = pd.concat(block_audit_parts, ignore_index=True, sort=False).sort_values(
        ["cutoff"], kind="stable",
    ).reset_index(drop=True)
    if held["candidate_id"].astype(str).duplicated().any():
        raise ValueError("held history windows overlap candidate identities")
    missing_identity = sorted(set(REFERENCE_IDENTITY).difference(reference.columns))
    if missing_identity:
        raise ValueError(f"reference history lacks identity fields: {missing_identity}")
    if reference.duplicated(list(REFERENCE_IDENTITY)).any():
        raise ValueError("reference history duplicates a producer identity")
    if block_audit["cutoff"].duplicated().any():
        raise ValueError("lockstep histories overlap fold cutoffs")
    if block_audit["geometry_bundle_sha256"].astype(str).nunique() != 1:
        raise ValueError("combined fold audit changes Geometry/K9 semantics")
    block_audit["geometry_refit_cadence"] = "never"

    held_start = held["__decision_ts__"].min()
    held_end = held["__decision_ts__"].max()
    if pd.isna(held_start) or pd.isna(held_end):
        raise ValueError("combined held history has invalid timestamps")

    args.out_dir.mkdir(parents=True)
    held_path = args.out_dir / HELD_FILE
    reference_path = args.out_dir / REFERENCE_FILE
    block_audit_path = args.out_dir / BLOCK_AUDIT_FILE
    held.to_parquet(held_path, index=False, compression="zstd")
    reference.to_parquet(reference_path, index=False, compression="zstd")
    block_audit.to_parquet(block_audit_path, index=False, compression="zstd")
    pd.DataFrame(source_audit).to_parquet(
        args.out_dir / "source_history_audit.parquet", index=False,
    )
    manifest = {
        "schema": "strict_r3_combined_lockstep_history_v1",
        "purpose": "pre-evaluation history plus executable OOS evaluation ledger",
        "history_is_reported_as_executable_oos": False,
        "reference_window_days": 28,
        "held_percentile_operations": 0,
        "outcomes_consumed_during_scoring": [],
        "geometry": {
            "refit_cadence": "never",
            "parent_bundle_sha256": next(iter(geometry_hashes)),
        },
        "held_rows": int(len(held)),
        "reference_rows": int(len(reference)),
        "held_start": pd.Timestamp(held_start).isoformat(),
        "held_end": pd.Timestamp(held_end).isoformat(),
        "sources": source_audit,
        "held_ledger": str(held_path),
        "held_ledger_sha256": _sha(held_path),
        "reference_ledger": str(reference_path),
        "reference_ledger_sha256": _sha(reference_path),
        "block_audit": str(block_audit_path),
        "block_audit_sha256": _sha(block_audit_path),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
