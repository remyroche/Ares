#!/usr/bin/env python3
"""Seal the Stage-C F4/F5 non-admission decision from archived source schemas.

This is a read-only audit.  It deliberately does *not* manufacture an
availability timestamp from an hourly index, a filesystem timestamp, an API
request time, or a chosen delay.  If a future sidecar contains the complete
native source-clock contract, this audit fails closed and a dedicated as-of
adapter must be implemented and reviewed instead of silently retaining this
blocker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/stage_c_oi_funding_lineage_blocker_20260801_v1"
KRAKEN_ROOT = ROOT / "data_perp/exchanges/krakenfutures"

SCHEMA = "stage_c_oi_funding_lineage_blocker_v1"
DISPOSITION = "BLOCKED_NO_NATIVE_OBSERVATION_OR_AVAILABILITY_CLOCK"

# These are row-level, immutable source-lineage requirements.  A nominal
# hourly event/index timestamp is intentionally not a substitute for either
# source-observed or source-published/available time.
COMMON_REQUIRED_FIELDS = (
    "provider",
    "exchange",
    "market_id",
    "product_kind",
    "source_event_ts",
    "source_observed_ts",
    "available_ts",
    "ingested_ts",
    "source_revision",
    "raw_payload_sha256",
)
FUNDING_REQUIRED_FIELDS = (
    "funding_value_kind",
    "settlement_ts",
)
OI_REQUIRED_FIELDS = (
    "oi_unit",
    "unit_conversion_price_ts",
)
# Only these names count as native clocks.  ``ts``, ``timestamp``,
# ``fundingTimestamp``, or a parquet/Pandas index can be an event boundary but
# do not say when the value was observed or published.
NATIVE_OBSERVATION_COLUMNS = frozenset({"source_observed_ts", "observed_ts"})
NATIVE_AVAILABILITY_COLUMNS = frozenset({"available_ts", "source_published_ts", "published_ts"})
NOMINAL_TIMESTAMP_COLUMNS = frozenset({"ts", "timestamp", "__index_level_0__"})


@dataclass(frozen=True)
class SourceFamily:
    source_id: str
    feature_group: str
    family: str
    directory: Path
    value_column: str
    extra_required_fields: tuple[str, ...]

    @property
    def required_fields(self) -> tuple[str, ...]:
        return (*COMMON_REQUIRED_FIELDS, *self.extra_required_fields)


DEFAULT_SOURCES = (
    SourceFamily(
        "kraken_open_interest_hourly_archived_sidecar",
        "F4_oi_dynamics",
        "open_interest",
        KRAKEN_ROOT / "open_interest_hourly",
        "open_interest",
        OI_REQUIRED_FIELDS,
    ),
    SourceFamily(
        "kraken_funding_hourly_archived_sidecar",
        "F5_funding_crowding",
        "funding",
        KRAKEN_ROOT / "funding_hourly",
        "funding_rate",
        FUNDING_REQUIRED_FIELDS,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(paths: Iterable[Path], *, root: Path) -> str:
    """Hash both file identities and bytes so a renamed sidecar is visible."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        try:
            identity = str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            # Unit tests and external read-only audit copies can supply a
            # bounded temporary directory.  Its absolute identity is still
            # part of that test-local seal; production sources stay root-relative.
            identity = str(path.resolve())
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _schema_counts(files: Iterable[Path]) -> tuple[dict[tuple[str, ...], int], set[str], int]:
    patterns: dict[tuple[str, ...], int] = {}
    union: set[str] = set()
    rows = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        names = tuple(map(str, parquet.schema.names))
        patterns[names] = patterns.get(names, 0) + 1
        union.update(names)
        rows += int(parquet.metadata.num_rows)
    return patterns, union, rows


def audit_archived_sidecars(sources: Iterable[SourceFamily] = DEFAULT_SOURCES) -> pd.DataFrame:
    """Return a source-level inventory without reading or mutating values."""
    rows: list[dict[str, Any]] = []
    for source in sources:
        files = sorted(source.directory.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"archived Stage-C sidecar source is absent: {source.directory}")
        patterns, columns, source_rows = _schema_counts(files)
        native_observation = sorted(columns.intersection(NATIVE_OBSERVATION_COLUMNS))
        native_availability = sorted(columns.intersection(NATIVE_AVAILABILITY_COLUMNS))
        nominal_timestamps = sorted(columns.intersection(NOMINAL_TIMESTAMP_COLUMNS))
        missing = sorted(set(source.required_fields).difference(columns))
        if source.value_column not in columns:
            raise ValueError(f"{source.source_id} lacks its expected value column {source.value_column}")
        rows.append(
            {
                "source_id": source.source_id,
                "feature_group": source.feature_group,
                "family": source.family,
                "source_directory": str(source.directory),
                "file_count": len(files),
                "row_count": source_rows,
                "value_column": source.value_column,
                "column_union": json.dumps(sorted(columns)),
                "schema_patterns": json.dumps(
                    [
                        {"columns": list(pattern), "files": count}
                        for pattern, count in sorted(patterns.items())
                    ],
                    sort_keys=True,
                ),
                "nominal_timestamp_columns": json.dumps(nominal_timestamps),
                "native_observation_columns": json.dumps(native_observation),
                "native_availability_columns": json.dumps(native_availability),
                "missing_required_fields": json.dumps(missing),
                "source_event_clock_proven": False,
                "native_observation_clock_proven": bool(native_observation),
                "native_availability_clock_proven": bool(native_availability),
                "bounded_staleness_proven": False,
                "point_in_time_safe": False,
                "disposition": DISPOSITION,
                "timestamp_interpretation": (
                    "nominal hourly index/event label only; never interpreted as observed_ts, "
                    "available_ts, or a source-publication promise"
                ),
                "data_store_evidence": (
                    "data_store.py:3042-3043 and 4029-4033 reindex+ffill OI/funding; "
                    "data_store.py:3368-3423 exports OI with only value plus hour-floor index; "
                    "data_store.py:3496-3507 compacts auxiliary values with unlimited ffill"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("source_id", kind="stable").reset_index(drop=True)


def validate_blocked_inventory(inventory: pd.DataFrame) -> dict[str, Any]:
    required = {
        "source_id", "feature_group", "file_count", "row_count", "value_column",
        "native_observation_clock_proven", "native_availability_clock_proven",
        "bounded_staleness_proven", "point_in_time_safe", "disposition",
        "missing_required_fields", "nominal_timestamp_columns",
    }
    missing = sorted(required.difference(inventory.columns))
    if missing:
        raise ValueError(f"blocked-source inventory lacks {missing}")
    if inventory.source_id.duplicated().any() or set(inventory.feature_group) != {"F4_oi_dynamics", "F5_funding_crowding"}:
        raise ValueError("inventory must contain exactly one F4 and one F5 archived source")
    if (inventory.file_count <= 0).any() or (inventory.row_count <= 0).any():
        raise ValueError("archived source inventory has no readable sidecar rows")
    # If either native clock turns up, continuing to publish a non-admission
    # manifest would conceal a material source change.  Stop and require a
    # reviewed as-of adapter with finite source-specific staleness instead.
    if inventory.native_observation_clock_proven.any() or inventory.native_availability_clock_proven.any():
        raise ValueError("native observation/availability clock now exists; implement a reviewed bounded-staleness as-of adapter")
    if inventory.bounded_staleness_proven.any() or inventory.point_in_time_safe.any():
        raise ValueError("unproven archived source cannot claim bounded staleness or PIT safety")
    if not inventory.disposition.eq(DISPOSITION).all():
        raise ValueError("archived F4/F5 sources must retain the explicit blocked disposition")
    missing_fields = inventory.missing_required_fields.map(json.loads)
    for required_clock in ("source_observed_ts", "available_ts"):
        if not missing_fields.map(lambda values: required_clock in values).all():
            raise ValueError(f"inventory does not explicitly record missing {required_clock}")
    return {
        "passed": True,
        "disposition": DISPOSITION,
        "source_count": int(len(inventory)),
        "feature_groups": sorted(inventory.feature_group.tolist()),
        "required_for_future_adapter": {
            "row_identity": ["provider", "exchange", "market_id", "product_kind"],
            "native_clocks": ["source_event_ts", "source_observed_ts", "available_ts", "ingested_ts"],
            "revision": ["source_revision", "raw_payload_sha256"],
            "f4_oi": list(OI_REQUIRED_FIELDS),
            "f5_funding": list(FUNDING_REQUIRED_FIELDS),
            "join": "available_ts <= feature_cutoff_ts; finite source-specific maximum staleness; stale/missing rows rejected",
        },
    }


def _report(inventory: pd.DataFrame, validation: Mapping[str, Any]) -> str:
    lines = [
        "# Stage-C F4/F5 archived source-lineage blocker",
        "",
        "## Verdict",
        "",
        f"`{DISPOSITION}`",
        "",
        "The archived hourly sidecars have only nominal `ts` or a pandas index. "
        "Those fields are not native observation or publication clocks and must not be converted into them by an assumed delay.",
        "",
        "## Audited sources",
        "",
        "| Group | Source | Files | Rows | Nominal timestamp fields | Native observation fields | Native availability fields |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for row in inventory.to_dict("records"):
        lines.append(
            f"| {row['feature_group']} | `{row['source_id']}` | {row['file_count']:,} | {row['row_count']:,} | "
            f"{row['nominal_timestamp_columns']} | {row['native_observation_columns']} | {row['native_availability_columns']} |"
        )
    lines.extend(
        [
            "",
            "## Exact missing row-level fields",
            "",
            "Both F4 and F5 are missing: `provider`, `exchange`, `market_id`, `product_kind`, "
            "`source_event_ts`, `source_observed_ts`, `available_ts`, `ingested_ts`, `source_revision`, and `raw_payload_sha256`.",
            "",
            "F4 additionally lacks `oi_unit` and `unit_conversion_price_ts`. F5 additionally lacks "
            "`funding_value_kind` and `settlement_ts`; therefore a historical rate cannot be proved to be the last published/settled rate rather than a forecast or revision.",
            "",
            "Current ingestion also applies unbounded forward fill at `data_store.py:3042-3043`, "
            "`3496-3507`, and `4029-4033`. Repeated values cannot establish source freshness, so no finite staleness limit is provable from these artifacts.",
            "",
            "## Required repair before an adapter may exist",
            "",
            "Persist source-specific raw observations with the required fields above, then implement an as-of join "
            "using `available_ts <= feature_cutoff_ts`, a finite documented maximum staleness per source, stale-row rejection, "
            "and PF-linear-USD product/unit parity. This blocker will fail rather than automatically admitting a changed source.",
            "",
            f"Checks passed: `{str(validation['passed']).lower()}`.",
            "",
        ]
    )
    return "\n".join(lines)


def run(*, output: Path, sources: Iterable[SourceFamily] = DEFAULT_SOURCES) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"fresh output root required: {output}")
    sources = tuple(sources)
    inventory = audit_archived_sidecars(sources)
    validation = validate_blocked_inventory(inventory)
    stage = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
    try:
        inventory_path = stage / "archived_sidecar_schema_inventory.parquet"
        inventory.to_parquet(inventory_path, index=False, compression="zstd")
        blocked_path = stage / "blocked_source_manifest.json"
        _write_json(
            blocked_path,
            {
                "schema": SCHEMA,
                "status": DISPOSITION,
                "feature_admission_changed": False,
                "adapter_created": False,
                "reason": "no native source_observed_ts or available_ts exists in either archived sidecar schema",
                "validation": validation,
                "sources": inventory.to_dict(orient="records"),
            },
        )
        report_path = stage / "lineage_blocker_report.md"
        report_path.write_text(_report(inventory, validation), encoding="utf-8")
        source_files = [path for source in sources for path in sorted(source.directory.glob("*.parquet"))]
        manifest = {
            "schema": SCHEMA,
            "status": DISPOSITION,
            "read_only_source_audit": True,
            "feature_admission_changed": False,
            "adapter_created": False,
            "source_trees": {
                source.source_id: {
                    "directory": str(source.directory),
                    "files": len(list(source.directory.glob("*.parquet"))),
                    "sha256": _tree_sha256(sorted(source.directory.glob("*.parquet")), root=ROOT),
                }
                for source in sources
            },
            "code_sha256": _sha256(Path(__file__)),
            "outputs_sha256": {
                inventory_path.name: _sha256(inventory_path),
                blocked_path.name: _sha256(blocked_path),
                report_path.name: _sha256(report_path),
            },
        }
        _write_json(stage / "manifest.json", manifest)
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
