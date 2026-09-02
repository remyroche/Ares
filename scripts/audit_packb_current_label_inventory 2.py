#!/usr/bin/env python3
"""Stream and attest the current canonical monthly Pack-B label inventory.

The historical ``causal_path_invariant_audit.json`` is a *base* inventory: it
names the 38 monthly ``*_5_YYYY_MM`` shards before the July tail append.  This
auditor resolves that immutable inventory first, applies only the declared
``label_tail_append_2026_07.json`` row-count amendment, and deliberately
excludes the unlisted monolithic ``train_global_short_7.parquet``.  It never
rewrites labels or an existing audit report.

``--contract-only`` reads Parquet metadata only.  The full audit streams a
small required-column projection in bounded batches and uses a temporary
SQLite primary-key index to detect duplicate candidate IDs without retaining
the label population in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.training_resource_guard import TrainingResourceGuard

AUDIT_SCHEMA = "packb_current_canonical_label_inventory_audit_v1"
BASE_AUDIT_FILENAME = "causal_path_invariant_audit.json"
TAIL_APPEND_FILENAME = "label_tail_append_2026_07.json"
EXCLUDED_MONOLITHIC_SHARD = "train_global_short_7.parquet"
CANONICAL_SIDES = ("long", "short")
BASE_LABEL_HORIZON = pd.Timedelta(hours=24)
DECISION_LAG = pd.Timedelta(hours=1)
ROUND_TRIP_COST = 0.01
SHARD_RE = re.compile(
    r"^train_global_(long|short)_5_(20\d{2})_(0[1-9]|1[0-2])\.parquet$"
)
REQUIRED_COLUMNS = (
    "candidate_id",
    "__ts__",
    "__decision_ts__",
    "__entry_ts__",
    "__first_path_ts__",
    "side_name",
    "__first_touch_round_trip_cost__",
    "__first_touch_valid_path__",
)

DEFAULT_LABELS = ROOT / (
    "data_perp/artifacts/20260720_s59_h5_signalclose_causal_"
    "trailing_cost100bps_labels_v2/labels"
)


class PackBCurrentLabelAuditError(ValueError):
    """Raised when the current canonical inventory cannot be attested safely."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackBCurrentLabelAuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PackBCurrentLabelAuditError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _existing_parent(path: Path) -> Path:
    candidate = Path(path)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _require_pyarrow() -> Any:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - test/runtime boundary.
        raise PackBCurrentLabelAuditError(
            "pyarrow is required for Pack-B audit"
        ) from exc
    return pa, pq


def _expected_monthly_inventory(base_audit: Mapping[str, Any]) -> dict[str, int]:
    entries = base_audit.get("per_file")
    if not isinstance(entries, list):
        raise PackBCurrentLabelAuditError("base causal audit has no per_file inventory")
    expected: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise PackBCurrentLabelAuditError(
                "base causal audit has malformed per_file entry"
            )
        name = entry.get("file")
        if not isinstance(name, str) or SHARD_RE.fullmatch(name) is None:
            raise PackBCurrentLabelAuditError(
                f"base causal audit has non-canonical monthly shard: {name!r}"
            )
        try:
            rows = int(entry.get("rows"))
        except (TypeError, ValueError) as exc:
            raise PackBCurrentLabelAuditError(
                f"base causal audit has invalid row count for {name}"
            ) from exc
        if rows < 0 or name in expected:
            raise PackBCurrentLabelAuditError(
                f"base causal audit has duplicate/invalid shard inventory entry: {name}"
            )
        expected[name] = rows
    if int(base_audit.get("files", -1)) != len(expected) or len(expected) != 38:
        raise PackBCurrentLabelAuditError(
            "base causal audit must declare exactly the 38 canonical monthly shards"
        )
    declared_rows = base_audit.get("rows")
    try:
        declared_rows = int(declared_rows)
    except (TypeError, ValueError) as exc:
        raise PackBCurrentLabelAuditError(
            "base causal audit has invalid total row count"
        ) from exc
    if declared_rows != sum(expected.values()):
        raise PackBCurrentLabelAuditError(
            "base causal audit total rows do not match its per_file inventory"
        )
    by_side = {side: 0 for side in CANONICAL_SIDES}
    for name in expected:
        by_side[SHARD_RE.fullmatch(name).group(1)] += 1  # type: ignore[union-attr]
    if by_side != {"long": 19, "short": 19}:
        raise PackBCurrentLabelAuditError(
            "base causal audit must contain 19 monthly shards per side"
        )
    return expected


def _tail_reconciliation(
    tail_append: Mapping[str, Any], *, base_rows: Mapping[str, int]
) -> dict[str, dict[str, Any]]:
    entries = tail_append.get("sides")
    if not isinstance(entries, list) or len(entries) != 2:
        raise PackBCurrentLabelAuditError(
            "tail append must contain exactly two side entries"
        )
    reconciled: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise PackBCurrentLabelAuditError("tail append has malformed side entry")
        side = str(entry.get("side", "")).strip().lower()
        target = Path(str(entry.get("target", ""))).name
        expected_target = f"train_global_{side}_5_2026_07.parquet"
        if (
            side not in CANONICAL_SIDES
            or target != expected_target
            or target not in base_rows
        ):
            raise PackBCurrentLabelAuditError(
                "tail append target is not a canonical July shard"
            )
        if side in reconciled:
            raise PackBCurrentLabelAuditError("tail append has duplicate side entry")
        try:
            existing = int(entry.get("existing_rows"))
            appended = int(entry.get("appended_rows"))
            final = int(entry.get("final_rows"))
        except (TypeError, ValueError) as exc:
            raise PackBCurrentLabelAuditError(
                f"tail append has invalid row counts for {side}"
            ) from exc
        if min(existing, appended, final) < 0 or existing != base_rows[target]:
            raise PackBCurrentLabelAuditError(
                f"tail append does not reconcile base row count for {target}"
            )
        if final != existing + appended:
            raise PackBCurrentLabelAuditError(
                f"tail append final row count is not existing + appended for {target}"
            )
        reconciled[target] = {
            "side": side,
            "base_audit_rows": existing,
            "appended_rows": appended,
            "expected_current_rows": final,
            "tail_manifest_entry": dict(entry),
        }
    if set(reconciled) != {
        "train_global_long_5_2026_07.parquet",
        "train_global_short_5_2026_07.parquet",
    }:
        raise PackBCurrentLabelAuditError(
            "tail append must reconcile both canonical July shards"
        )
    return reconciled


def _timestamp_storage_contract(schema: Any) -> dict[str, str]:
    pa, _pq = _require_pyarrow()
    contract: dict[str, str] = {}
    for column in ("__ts__", "__decision_ts__", "__entry_ts__", "__first_path_ts__"):
        field = schema.field(column)
        if not pa.types.is_timestamp(field.type):
            raise PackBCurrentLabelAuditError(f"{column} must be a timestamp column")
        timezone = field.type.tz
        if column == "__ts__" and timezone is None:
            # The established Pack-B shards persist this legacy signal clock as
            # naive timestamps.  Repository policy interprets it as UTC.
            contract[column] = "legacy_naive_interpreted_as_utc"
        elif timezone in {"UTC", "Etc/UTC"}:
            contract[column] = "explicit_utc"
        else:
            raise PackBCurrentLabelAuditError(
                f"{column} must be UTC or legacy-naive signal time, got {field.type}"
            )
    return contract


def contract_only(
    *,
    labels_dir: Path = DEFAULT_LABELS,
    base_causal_audit: Path | None = None,
    tail_append: Path | None = None,
    resource_guard: TrainingResourceGuard | None = None,
) -> dict[str, Any]:
    """Validate inventory, tail amendment, and Parquet schemas without row reads."""

    labels_dir = Path(labels_dir)
    base_causal_audit = Path(base_causal_audit or labels_dir / BASE_AUDIT_FILENAME)
    tail_append = Path(tail_append or labels_dir / TAIL_APPEND_FILENAME)
    if not labels_dir.is_dir():
        raise PackBCurrentLabelAuditError(
            f"labels directory does not exist: {labels_dir}"
        )
    if resource_guard is not None:
        resource_guard.preflight("contract_only_schema_inventory")

    base_payload = _json(base_causal_audit)
    base_rows = _expected_monthly_inventory(base_payload)
    tail_payload = _json(tail_append)
    tail_rows = _tail_reconciliation(tail_payload, base_rows=base_rows)
    expected_rows = dict(base_rows)
    expected_rows.update(
        {name: int(item["expected_current_rows"]) for name, item in tail_rows.items()}
    )
    actual = {path.name for path in labels_dir.glob("*.parquet") if path.is_file()}
    expected_names = set(expected_rows)
    missing = sorted(expected_names - actual)
    unlisted = sorted(actual - expected_names)
    unexpected = sorted(set(unlisted) - {EXCLUDED_MONOLITHIC_SHARD})
    if missing or unexpected:
        detail = []
        if missing:
            detail.append("missing=" + ", ".join(missing[:8]))
        if unexpected:
            detail.append("unexpected=" + ", ".join(unexpected[:8]))
        raise PackBCurrentLabelAuditError(
            "canonical monthly inventory mismatch: " + "; ".join(detail)
        )

    _pa, pq = _require_pyarrow()
    per_file: list[dict[str, Any]] = []
    for name in sorted(expected_names):
        if resource_guard is not None:
            resource_guard.checkpoint(f"before_schema:{name}")
        parquet = pq.ParquetFile(labels_dir / name)
        schema = parquet.schema_arrow
        columns = set(schema.names)
        missing_columns = sorted(set(REQUIRED_COLUMNS) - columns)
        if missing_columns:
            raise PackBCurrentLabelAuditError(
                f"canonical shard misses required audit columns: {name}: {missing_columns}"
            )
        matched = SHARD_RE.fullmatch(name)
        assert matched is not None
        per_file.append(
            {
                "file": name,
                "side_from_filename": matched.group(1),
                "month": f"{matched.group(2)}-{matched.group(3)}",
                "expected_current_rows": int(expected_rows[name]),
                "metadata_rows": int(parquet.metadata.num_rows),
                "metadata_row_count_matches_expected": int(parquet.metadata.num_rows)
                == int(expected_rows[name]),
                "timestamp_storage_contract": _timestamp_storage_contract(schema),
                "tail_reconciliation": tail_rows.get(name),
            }
        )
    metadata_mismatches = [
        item["file"]
        for item in per_file
        if not item["metadata_row_count_matches_expected"]
    ]
    if metadata_mismatches:
        raise PackBCurrentLabelAuditError(
            "canonical shard metadata row counts do not match the reconciled inventory: "
            + ", ".join(metadata_mismatches[:8])
        )
    return {
        "schema": AUDIT_SCHEMA,
        "status": "CONTRACT_VALIDATED_SCHEMA_ONLY",
        "mode": "contract_only_schema",
        "labels_dir": str(labels_dir),
        "base_causal_audit": {
            "path": str(base_causal_audit),
            "sha256": _sha256(base_causal_audit),
            "declared_files": int(base_payload["files"]),
            "declared_base_rows": int(
                base_payload.get("rows", sum(base_rows.values()))
            ),
        },
        "tail_append": {"path": str(tail_append), "sha256": _sha256(tail_append)},
        "inventory": {
            "canonical_monthly_files": len(expected_names),
            "excluded_unlisted_monolithic_files": [
                name for name in unlisted if name == EXCLUDED_MONOLITHIC_SHARD
            ],
            "expected_current_rows": int(sum(expected_rows.values())),
            "row_count_metadata_matches_expected": all(
                item["metadata_row_count_matches_expected"] for item in per_file
            ),
        },
        "per_file": per_file,
        "streaming_contract": {
            "full_scan_required_for": [
                "candidate_id_global_uniqueness",
                "row_timestamp_and_side_invariants",
                "original_causal_path_invariants",
                "derived_base_label_resolution_bounds",
            ],
            "full_scan_columns": list(REQUIRED_COLUMNS),
            "batch_materialization": "required-column projection only",
            "duplicate_index": "temporary SQLite primary key",
        },
    }


def _invalid_ids(values: pd.Series) -> pd.Series:
    ids = values.astype("string")
    return ids.isna() | ids.str.strip().eq("") | ids.ne(ids.str.strip())


def _audit_batch(frame: pd.DataFrame, *, expected_side: str) -> dict[str, Any]:
    signal = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    decision = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="coerce")
    entry = pd.to_datetime(frame["__entry_ts__"], utc=True, errors="coerce")
    first_path = pd.to_datetime(frame["__first_path_ts__"], utc=True, errors="coerce")
    valid_path = pd.to_numeric(frame["__first_touch_valid_path__"], errors="coerce")
    cost = pd.to_numeric(frame["__first_touch_round_trip_cost__"], errors="coerce")
    sides = frame["side_name"].astype("string").str.strip().str.lower()
    resolution = decision + BASE_LABEL_HORIZON
    return {
        "rows": int(len(frame)),
        "bad_candidate_id": int(_invalid_ids(frame["candidate_id"]).sum()),
        "bad_utc_timestamp": int(
            (signal.isna() | decision.isna() | entry.isna() | first_path.isna()).sum()
        ),
        "bad_decision": int((~decision.eq(signal + DECISION_LAG)).sum()),
        "bad_first_path": int((~first_path.ge(decision)).sum()),
        "bad_entry": int((~entry.ge(decision)).sum()),
        "bad_cost": int(
            (~np.isclose(cost, ROUND_TRIP_COST, rtol=0.0, atol=1e-9)).sum()
        ),
        "invalid_path": int((~valid_path.eq(1.0)).sum()),
        "bad_side_file_consistency": int((~sides.eq(expected_side)).sum()),
        "resolution_min_utc": resolution.min(),
        "resolution_max_utc": resolution.max(),
    }


def _merge_batch_counts(target: dict[str, Any], current: Mapping[str, Any]) -> None:
    for name in (
        "rows",
        "bad_candidate_id",
        "bad_utc_timestamp",
        "bad_decision",
        "bad_first_path",
        "bad_entry",
        "bad_cost",
        "invalid_path",
        "bad_side_file_consistency",
    ):
        target[name] += int(current[name])
    for name, comparator in (("resolution_min_utc", min), ("resolution_max_utc", max)):
        value = current[name]
        if pd.notna(value):
            target[name] = (
                value if target[name] is None else comparator(target[name], value)
            )


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    # Exclusive create is deliberate: an audit must remain immutable even if a
    # concurrent operator selects the same destination.
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def run_full_audit(
    *,
    labels_dir: Path = DEFAULT_LABELS,
    base_causal_audit: Path | None = None,
    tail_append: Path | None = None,
    output_path: Path,
    batch_rows: int = 50_000,
    resource_guard: TrainingResourceGuard | None = None,
) -> dict[str, Any]:
    """Run the bounded streaming audit and create one new immutable JSON report."""

    if not isinstance(batch_rows, int) or batch_rows < 1:
        raise PackBCurrentLabelAuditError("batch_rows must be a positive integer")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {output_path}")
    guard = resource_guard or TrainingResourceGuard(
        disk_path=_existing_parent(output_path)
    )
    contract = contract_only(
        labels_dir=labels_dir,
        base_causal_audit=base_causal_audit,
        tail_append=tail_append,
        resource_guard=guard,
    )
    guard.preflight("before_full_label_scan")
    _pa, pq = _require_pyarrow()
    totals = {
        name: 0
        for name in (
            "rows",
            "bad_candidate_id",
            "bad_utc_timestamp",
            "bad_decision",
            "bad_first_path",
            "bad_entry",
            "bad_cost",
            "invalid_path",
            "bad_side_file_consistency",
            "duplicate_candidate_id",
        )
    }
    scanned: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="packb-current-label-audit-",
        dir=_existing_parent(output_path),
    ) as temporary:
        index = sqlite3.connect(Path(temporary) / "candidate_ids.sqlite")
        index.execute("PRAGMA journal_mode=OFF")
        index.execute("PRAGMA synchronous=OFF")
        index.execute("CREATE TABLE candidate_ids (candidate_id TEXT PRIMARY KEY)")
        try:
            for contract_file in contract["per_file"]:
                name = str(contract_file["file"])
                side = str(contract_file["side_from_filename"])
                guard.checkpoint(f"before_shard:{name}")
                counts: dict[str, Any] = {
                    name: 0
                    for name in (
                        "rows",
                        "bad_candidate_id",
                        "bad_utc_timestamp",
                        "bad_decision",
                        "bad_first_path",
                        "bad_entry",
                        "bad_cost",
                        "invalid_path",
                        "bad_side_file_consistency",
                    )
                }
                counts["resolution_min_utc"] = None
                counts["resolution_max_utc"] = None
                parquet = pq.ParquetFile(Path(labels_dir) / name)
                for batch in parquet.iter_batches(
                    batch_size=batch_rows, columns=list(REQUIRED_COLUMNS)
                ):
                    guard.checkpoint(f"before_batch:{name}")
                    frame = batch.to_pandas()
                    current = _audit_batch(frame, expected_side=side)
                    _merge_batch_counts(counts, current)
                    invalid = _invalid_ids(frame["candidate_id"])
                    ids = frame.loc[~invalid, "candidate_id"].astype(str).tolist()
                    before = index.total_changes
                    index.executemany(
                        "INSERT OR IGNORE INTO candidate_ids(candidate_id) VALUES (?)",
                        ((candidate_id,) for candidate_id in ids),
                    )
                    duplicate = len(ids) - (index.total_changes - before)
                    totals["duplicate_candidate_id"] += int(duplicate)
                counts["row_count_matches_expected"] = counts["rows"] == int(
                    contract_file["expected_current_rows"]
                )
                counts["derived_base_label_resolution"] = "__decision_ts__ + 24 hours"
                counts["derived_base_label_resolution_min_utc"] = (
                    counts.pop("resolution_min_utc").isoformat()
                    if counts["resolution_min_utc"] is not None
                    else None
                )
                counts["derived_base_label_resolution_max_utc"] = (
                    counts.pop("resolution_max_utc").isoformat()
                    if counts["resolution_max_utc"] is not None
                    else None
                )
                for metric in totals:
                    if metric != "duplicate_candidate_id":
                        totals[metric] += int(counts[metric])
                scanned.append({**contract_file, **counts})
        finally:
            index.close()

    failures = {
        name: int(value)
        for name, value in totals.items()
        if name != "rows" and int(value) != 0
    }
    row_count_failures = [
        item["file"] for item in scanned if not item["row_count_matches_expected"]
    ]
    if row_count_failures:
        failures["row_count_mismatch_files"] = len(row_count_failures)
    report = {
        **contract,
        "status": "PASS" if not failures else "FAILED_INVARIANT_AUDIT",
        "mode": "streaming_full_audit",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "output_path": str(output_path),
        "contract_only": False,
        "per_file": scanned,
        "totals": totals,
        "failures": failures,
        "invariant_contract": {
            "decision": "__decision_ts__ == __ts__ + 1 hour",
            "base_label_resolution": "__decision_ts__ + 24 hours",
            "first_path": "__first_path_ts__ >= __decision_ts__",
            "entry": "__entry_ts__ >= __decision_ts__",
            "round_trip_cost": "__first_touch_round_trip_cost__ == 0.01",
            "valid_path": "__first_touch_valid_path__ == 1",
            "side": "side_name exactly matches the canonical shard filename side",
            "candidate_id": "nonblank and globally unique across all 38 listed shards",
        },
    }
    _write_immutable_json(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--base-causal-audit", type=Path)
    parser.add_argument("--tail-append", type=Path)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-rows", type=int, default=50_000)
    parser.add_argument("--resource-telemetry-path", type=Path)
    args = parser.parse_args(argv)
    if not args.contract_only and args.output is None:
        parser.error("--output is required unless --contract-only is set")
    guard = TrainingResourceGuard(
        disk_path=_existing_parent(args.output or args.labels_dir),
        telemetry_path=args.resource_telemetry_path,
    )
    try:
        if args.contract_only:
            report = contract_only(
                labels_dir=args.labels_dir,
                base_causal_audit=args.base_causal_audit,
                tail_append=args.tail_append,
                resource_guard=guard,
            )
        else:
            report = run_full_audit(
                labels_dir=args.labels_dir,
                base_causal_audit=args.base_causal_audit,
                tail_append=args.tail_append,
                output_path=args.output,
                batch_rows=args.batch_rows,
                resource_guard=guard,
            )
    except (PackBCurrentLabelAuditError, OSError) as exc:
        print(
            json.dumps({"status": "AUDIT_INPUT_ERROR", "error": str(exc)}),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
