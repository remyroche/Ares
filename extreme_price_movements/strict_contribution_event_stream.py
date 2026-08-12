"""Reusable strict, time-ordered contribution event streams.

The canonical event store deliberately keeps contributions token-free and
candidate outcomes separate.  That is ideal for selection, but an exact
prequential health pass needs each contribution together with the *paired*
candidate's resolution data in feature-time and label-resolution order.

This immutable sidecar pays that pairing/sort once per already-bounded source
part.  Files are not globally concatenated: their ranges can overlap across
transports, so consumers perform a small k-way merge of the per-part streams.
The important boundary is that no consumer needs to join or externally sort a
70M-row side/head scope again.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:  # pragma: no cover - import failure is surfaced by the public builder
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = pq = None

from .strict_event_store import (
    CANDIDATE_COLUMNS,
    CONTRIBUTION_COLUMNS,
    StrictEventStore,
    StrictEventStoreError,
    _forbid_raw_leaf,
    _sha256,
    load_strict_event_store,
)


SCHEMA = "strict_leaf_reasoning_contribution_event_stream_v1"
STATUS = "STRICT_CONTRIBUTION_EVENT_STREAM_COMPLETED"
EVENT_COLUMNS = (
    "candidate_id", "decision_ts", "feature_generation_ts", "label_available_ts",
    "side_name", "head_name", "fold_id", "transport", "meta_partition",
    "feature_contract_sha256", "semantic_label", "head_prediction", "net_bps",
    "base_expected_bps", "asset", "rule_signature", "contribution_direction",
    "family_ensemble_tree_contribution",
)
_IDENTITY = ("candidate_id", "decision_ts", "side_name", "head_name", "fold_id", "transport", "meta_partition")
_SOURCE_PAIR_SCOPE = ("contract", "side", "head", "month", "meta_partition")
_SOURCE_PAIR_NEW_FIELDS = (
    "source_contribution_sha256",
    "source_candidate_identity_sha256",
    "source_contribution_identity_sha256",
)


@dataclass(frozen=True)
class StrictContributionEventStreams:
    root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    part_index: pd.DataFrame


def _require_arrow() -> None:
    if pa is None or pq is None:
        raise StrictEventStoreError("pyarrow is required for contribution-event stream materialisation")


def _literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _event_schema() -> "pa.Schema":
    _require_arrow()
    return pa.schema([
        pa.field("candidate_id", pa.dictionary(pa.int32(), pa.string())),
        pa.field("decision_ts", pa.timestamp("ns", tz="UTC")),
        pa.field("feature_generation_ts", pa.timestamp("ns", tz="UTC")),
        pa.field("label_available_ts", pa.timestamp("ns", tz="UTC")),
        pa.field("side_name", pa.dictionary(pa.int8(), pa.string())),
        pa.field("head_name", pa.dictionary(pa.int8(), pa.string())),
        pa.field("fold_id", pa.dictionary(pa.int16(), pa.string())),
        pa.field("transport", pa.dictionary(pa.int8(), pa.string())),
        pa.field("meta_partition", pa.dictionary(pa.int8(), pa.string())),
        pa.field("feature_contract_sha256", pa.dictionary(pa.int16(), pa.string())),
        pa.field("semantic_label", pa.float32()), pa.field("head_prediction", pa.float32()),
        pa.field("net_bps", pa.float32()), pa.field("base_expected_bps", pa.float32()),
        pa.field("asset", pa.dictionary(pa.int16(), pa.string())),
        pa.field("rule_signature", pa.dictionary(pa.int32(), pa.string())),
        pa.field("contribution_direction", pa.dictionary(pa.int8(), pa.string())),
        pa.field("family_ensemble_tree_contribution", pa.float32()),
    ])


def _source_part_index(store: StrictEventStore) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the two canonical physical datasets needed by this sidecar.

    The canonical store already seals each candidate/contribution part with a
    physical hash and a dataset-specific identity digest.  The sidecar must
    retain that one-to-one relation rather than trusting a path convention.
    """

    candidate = store.part_index.loc[store.part_index["dataset"].eq("candidate")].copy()
    contribution = store.part_index.loc[store.part_index["dataset"].eq("contribution")].copy()
    required = {
        "dataset", "path", "contract", "side", "head", "month", "meta_partition",
        "rows", "candidate_identity_sha256", "paired_contribution_path", "sha256",
    }
    missing = sorted(required.difference(store.part_index.columns))
    if missing or candidate.empty or contribution.empty:
        raise StrictEventStoreError(
            f"event store cannot prove paired source parts: missing={missing}, "
            f"candidate_parts={len(candidate)}, contribution_parts={len(contribution)}"
        )
    if candidate["path"].astype(str).duplicated().any() or contribution["path"].astype(str).duplicated().any():
        raise StrictEventStoreError("event store has duplicate physical source-part paths")
    return candidate, contribution


def _verify_source_pair_before_read(
    store: StrictEventStore,
    candidate_item: Any,
    *,
    contribution_index: pd.DataFrame,
) -> tuple[Path, Path, Mapping[str, Any], Mapping[str, Any]]:
    """Verify the immutable canonical pair immediately before decoding it.

    This deliberately hashes both source files *before* ``read_parquet``.
    It closes the gap between a verified store descriptor and a potentially
    modified physical file while a long sidecar build is in progress.
    """

    candidate_path_rel = str(candidate_item.path)
    contribution_path_rel = str(candidate_item.paired_contribution_path)
    if not contribution_path_rel or contribution_path_rel.lower() == "nan":
        raise StrictEventStoreError("event-store candidate part lacks its paired contribution path")
    matched = contribution_index.loc[
        contribution_index["path"].astype(str).eq(contribution_path_rel)
    ]
    if len(matched) != 1:
        raise StrictEventStoreError("event-store candidate does not map to exactly one contribution part")
    contribution_item = matched.iloc[0]
    if str(contribution_item["dataset"]) != "contribution":
        raise StrictEventStoreError("event-store paired source is not a contribution part")
    for column in _SOURCE_PAIR_SCOPE:
        if str(getattr(candidate_item, column)) != str(contribution_item[column]):
            raise StrictEventStoreError(f"event-store paired source crosses {column}")
    candidate_path = store.root / candidate_path_rel
    contribution_path = store.root / contribution_path_rel
    if not candidate_path.is_file() or not contribution_path.is_file():
        raise StrictEventStoreError("event-store paired source part is missing")
    # Hash first, then decode.  This is intentionally repeated for every
    # source pair rather than relying on a process-start validation pass.
    if _sha256(candidate_path) != str(candidate_item.sha256):
        raise StrictEventStoreError("event-store candidate source hash changed before sidecar read")
    if _sha256(contribution_path) != str(contribution_item["sha256"]):
        raise StrictEventStoreError("event-store contribution source hash changed before sidecar read")
    return candidate_path, contribution_path, candidate_item._asdict(), contribution_item.to_dict()


def _validate_event_physical_schema(path: Path, *, source: str) -> int:
    """Validate the exact compact Arrow physical contract without decoding rows."""

    parquet = pq.ParquetFile(path)
    actual = parquet.schema_arrow
    expected = _event_schema()
    if tuple(actual.names) != EVENT_COLUMNS:
        raise StrictEventStoreError(
            f"contribution event-stream physical types/schema has invalid columns: {source}"
        )
    for name in EVENT_COLUMNS:
        if actual.field(name).type != expected.field(name).type:
            raise StrictEventStoreError(
                "contribution event-stream physical types differ from the sealed contract: "
                f"{source}.{name}={actual.field(name).type}, expected={expected.field(name).type}"
            )
    return int(parquet.metadata.num_rows)


def _event_frame(candidate: pd.DataFrame, contribution: pd.DataFrame, *, source: str) -> pd.DataFrame:
    _forbid_raw_leaf(candidate.columns, source=f"{source} candidate")
    _forbid_raw_leaf(contribution.columns, source=f"{source} contribution")
    missing_candidate = sorted(set(CANDIDATE_COLUMNS).difference(candidate.columns))
    missing_contribution = sorted(set(CONTRIBUTION_COLUMNS).difference(contribution.columns))
    if missing_candidate or missing_contribution:
        raise StrictEventStoreError(
            f"paired event source is incomplete: candidate={missing_candidate}, contribution={missing_contribution}"
        )
    lookup = candidate.loc[:, list(CANDIDATE_COLUMNS)].copy()
    lookup = lookup.rename(columns={"decision_ts": "__ts__"})
    joined = contribution.loc[:, list(CONTRIBUTION_COLUMNS)].merge(
        lookup,
        on=["candidate_id", "__ts__", "side_name", "head_name", "fold_id", "transport", "meta_partition", "feature_contract_sha256"],
        how="left", validate="many_to_one", indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise StrictEventStoreError(f"{source} has a contribution without its paired strict candidate")
    joined = joined.drop(columns="_merge").rename(columns={"__ts__": "decision_ts"})
    for column in ("decision_ts", "feature_generation_ts", "label_available_ts"):
        joined[column] = pd.to_datetime(joined[column], utc=True, errors="coerce")
    if joined.loc[:, ["decision_ts", "feature_generation_ts", "label_available_ts"]].isna().any().any():
        raise StrictEventStoreError(f"{source} has invalid event timestamps")
    if not joined["feature_generation_ts"].le(joined["decision_ts"]).all() or not joined["label_available_ts"].ge(joined["decision_ts"]).all():
        raise StrictEventStoreError(f"{source} has invalid strict event timing")
    if joined.duplicated([* _IDENTITY, "rule_signature", "contribution_direction"]).any():
        raise StrictEventStoreError(f"{source} duplicates a collapsed candidate/family contribution")
    result = joined.loc[:, list(EVENT_COLUMNS)].copy()
    if not np.isfinite(result.loc[:, ["semantic_label", "head_prediction", "net_bps", "base_expected_bps", "family_ensemble_tree_contribution"]].to_numpy(dtype=float)).all():
        raise StrictEventStoreError(f"{source} has a non-finite numeric event value")
    return result


def _relative_path(kind: str, row: Any, counter: int) -> Path:
    return (
        Path(kind) / f"contract={row.contract}" / f"side={row.side}" / f"head={row.head}"
        / f"month={row.month}" / f"partition={row.meta_partition}" / f"part-{counter:06d}.parquet"
    )


def _write_event_part(frame: pd.DataFrame, *, path: Path, time_column: str) -> None:
    ordered = frame.sort_values(
        [time_column, "candidate_id", "fold_id", "transport", "meta_partition", "rule_signature", "contribution_direction"],
        kind="stable",
    ).reset_index(drop=True)
    table = pa.Table.from_pandas(ordered.loc[:, list(EVENT_COLUMNS)], preserve_index=False).cast(_event_schema(), safe=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", use_dictionary=True)


def build_strict_contribution_event_streams(
    event_store: StrictEventStore | str | Path,
    output_dir: str | Path,
    *,
    max_output_bytes: int = 6 * 1024**3,
    minimum_free_bytes: int = 12 * 1024**3,
) -> StrictContributionEventStreams:
    """Materialise immutable, candidate-paired contribution streams.

    Each output piece corresponds to exactly one sealed candidate/contribution
    source pair.  The write is atomic and all physical outputs are indexed and
    hashed before sealing.
    """

    _require_arrow()
    store = event_store if isinstance(event_store, StrictEventStore) else load_strict_event_store(
        event_store, verify_parts=False, verify_source=True,
    )
    target = Path(output_dir)
    if int(max_output_bytes) <= 0 or int(minimum_free_bytes) <= 0:
        raise StrictEventStoreError("contribution event-stream storage budgets must be positive")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite contribution event streams: {target}")
    candidate_index, contribution_index = _source_part_index(store)
    candidate_index = candidate_index.sort_values(["contract", "side", "head", "month", "path"], kind="stable")
    target.parent.mkdir(parents=True, exist_ok=True)
    available = int(shutil.disk_usage(target.parent).free)
    required = int(max_output_bytes) + int(minimum_free_bytes)
    if available < required:
        raise StrictEventStoreError(
            "insufficient free space for bounded contribution event streams: "
            f"available={available}, required={required}"
        )
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    index_rows: list[dict[str, Any]] = []
    counter = 0
    written_bytes = 0
    try:
        for item in candidate_index.itertuples(index=False):
            (
                candidate_path,
                contribution_path,
                candidate_source,
                contribution_source,
            ) = _verify_source_pair_before_read(
                store, item, contribution_index=contribution_index,
            )
            frame = _event_frame(
                pd.read_parquet(candidate_path), pd.read_parquet(contribution_path),
                source=str(item.path),
            )
            if len(frame) != int(item.rows) and not len(frame):
                raise StrictEventStoreError("paired contribution event frame is unexpectedly empty")
            for kind, timestamp in (("feature_event_order", "feature_generation_ts"), ("resolution_event_order", "label_available_ts")):
                relative = _relative_path(kind, item, counter)
                absolute = temporary / relative
                _write_event_part(frame, path=absolute, time_column=timestamp)
                written_bytes += int(absolute.stat().st_size)
                if written_bytes > int(max_output_bytes):
                    raise StrictEventStoreError(
                        "contribution event-stream output exceeds its declared storage budget: "
                        f"written={written_bytes}, max_output_bytes={int(max_output_bytes)}"
                    )
                index_rows.append({
                    "dataset": kind, "path": str(relative), "contract": str(item.contract),
                    "side": str(item.side), "head": str(item.head), "month": str(item.month),
                    "meta_partition": str(item.meta_partition), "rows": int(len(frame)),
                    "timestamp_column": timestamp,
                    "min_timestamp": pd.Timestamp(frame[timestamp].min()).isoformat(),
                    "max_timestamp": pd.Timestamp(frame[timestamp].max()).isoformat(),
                    "source_candidate_path": str(item.path),
                    "source_contribution_path": str(item.paired_contribution_path),
                    "source_candidate_sha256": str(candidate_source["sha256"]),
                    "source_contribution_sha256": str(contribution_source["sha256"]),
                    # These are dataset-specific physical-population
                    # digests.  A contribution part can contain multiple
                    # family rows for one candidate, so it is intentionally
                    # not expected to equal the paired candidate digest.
                    "source_candidate_identity_sha256": str(candidate_source["candidate_identity_sha256"]),
                    "source_contribution_identity_sha256": str(contribution_source["candidate_identity_sha256"]),
                    "sha256": _sha256(absolute),
                })
                counter += 1
        index = pd.DataFrame(index_rows)
        if index.empty or set(index["dataset"]) != {"feature_event_order", "resolution_event_order"}:
            raise StrictEventStoreError("contribution event-stream build produced an incomplete index")
        index_path = temporary / "contribution_event_stream_parts.parquet"
        index.to_parquet(index_path, index=False, compression="zstd")
        manifest = {
            "schema": SCHEMA, "status": STATUS,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_event_store_root": str(store.root),
            "source_event_store_manifest_sha256": _sha256(store.manifest_path),
            "source_strict_roots": list(store.manifest["source"]["strict_roots"]),
            "source_strict_root_manifest_sha256": dict(store.manifest["source"]["strict_root_manifest_sha256"]),
            "parts": index_path.name,
            "row_counts": {
                "feature_event_rows": int(index.loc[index["dataset"].eq("feature_event_order"), "rows"].sum()),
                "resolution_event_rows": int(index.loc[index["dataset"].eq("resolution_event_order"), "rows"].sum()),
                "physical_parts": int(len(index)),
            },
            "contract": {
                "event_columns": list(EVENT_COLUMNS),
                "ordering": "each immutable paired-source part is sorted by its declared timestamp; overlapping source parts are merged k-way by consumers",
                "history": "consumers must score all feature-generation events before applying events with equal label-resolution timestamps",
                "raw_leaf_ids": "rejected; only token-free rule-family signatures are present",
                "physical_types": "dictionary dimensions, float32 continuous values, UTC nanosecond timestamps",
                "source_pair_integrity": (
                    "each output index row records the canonical candidate/contribution paths, "
                    "physical hashes and dataset-specific identity hashes; both inputs are "
                    "hashed immediately before each decode"
                ),
            },
            "storage_guard": {
                "max_output_bytes": int(max_output_bytes),
                "minimum_free_bytes": int(minimum_free_bytes),
                "written_physical_bytes": int(written_bytes),
            },
            "sha256": {"parts": _sha256(index_path)},
        }
        manifest_path = temporary / "strict_contribution_event_stream_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_strict_contribution_event_streams(target, verify_parts=False)


def _validate_source_pair_correspondence(
    store: StrictEventStore,
    index: pd.DataFrame,
    *,
    verify_source_parts: bool,
) -> None:
    """Prove each sidecar part still points to its sealed canonical pair.

    Older v1 sidecars predate the contribution hash/identity columns.  They
    remain readable, but are explicitly treated as legacy: their pair is
    reconciled against the live sealed event-store index rather than against
    the additional per-row declarations made by current builders.
    """

    candidate_index, contribution_index = _source_part_index(store)
    candidate_by_path = candidate_index.set_index(candidate_index["path"].astype(str), drop=False)
    contribution_by_path = contribution_index.set_index(contribution_index["path"].astype(str), drop=False)
    new_fields = [field for field in _SOURCE_PAIR_NEW_FIELDS if field in index.columns]
    if new_fields and set(new_fields) != set(_SOURCE_PAIR_NEW_FIELDS):
        missing = sorted(set(_SOURCE_PAIR_NEW_FIELDS).difference(new_fields))
        raise StrictEventStoreError(
            f"contribution event-stream source-pair index has a partial integrity declaration: missing={missing}"
        )
    source_hashes_checked: set[str] = set()
    for item in index.itertuples(index=False):
        candidate_path_rel = str(item.source_candidate_path)
        contribution_path_rel = str(item.source_contribution_path)
        if candidate_path_rel not in candidate_by_path.index or contribution_path_rel not in contribution_by_path.index:
            raise StrictEventStoreError("contribution event-stream source pair is absent from the canonical event-store index")
        candidate_source = candidate_by_path.loc[candidate_path_rel]
        contribution_source = contribution_by_path.loc[contribution_path_rel]
        # Paths are unique by the canonical loader.  Preserve a defensive
        # check here because ``DataFrame.loc`` would otherwise return a frame.
        if isinstance(candidate_source, pd.DataFrame) or isinstance(contribution_source, pd.DataFrame):
            raise StrictEventStoreError("contribution event-stream source pair is ambiguous in the canonical event-store")
        if str(candidate_source["paired_contribution_path"]) != contribution_path_rel:
            raise StrictEventStoreError("contribution event-stream source contribution is not paired to its declared candidate")
        for column in _SOURCE_PAIR_SCOPE:
            if str(candidate_source[column]) != str(contribution_source[column]):
                raise StrictEventStoreError(f"canonical event-store source pair crosses {column}")
            if str(getattr(item, column)) != str(candidate_source[column]):
                raise StrictEventStoreError(f"contribution event-stream part scope differs from its source {column}")
        if int(item.rows) != int(contribution_source["rows"]):
            raise StrictEventStoreError("contribution event-stream row count differs from its source contribution part")
        if str(item.source_candidate_sha256) != str(candidate_source["sha256"]):
            raise StrictEventStoreError("contribution event-stream source candidate hash differs from canonical index")
        if new_fields:
            if str(item.source_contribution_sha256) != str(contribution_source["sha256"]):
                raise StrictEventStoreError("contribution event-stream source contribution hash differs from canonical index")
            if str(item.source_candidate_identity_sha256) != str(candidate_source["candidate_identity_sha256"]):
                raise StrictEventStoreError("contribution event-stream source candidate identity differs from canonical index")
            if str(item.source_contribution_identity_sha256) != str(contribution_source["candidate_identity_sha256"]):
                raise StrictEventStoreError("contribution event-stream source contribution identity differs from canonical index")
        if verify_source_parts:
            for source_path_rel, declared_hash in (
                (candidate_path_rel, str(candidate_source["sha256"])),
                (contribution_path_rel, str(contribution_source["sha256"])),
            ):
                cache_key = f"{source_path_rel}\x1f{declared_hash}"
                if cache_key in source_hashes_checked:
                    continue
                physical = store.root / source_path_rel
                if not physical.is_file() or _sha256(physical) != declared_hash:
                    raise StrictEventStoreError("canonical event-store source physical hash changed after sidecar sealing")
                source_hashes_checked.add(cache_key)


def load_strict_contribution_event_streams(
    root: str | Path, *, verify_parts: bool = True,
) -> StrictContributionEventStreams:
    _require_arrow()
    target = Path(root)
    manifest_path = target / "strict_contribution_event_stream_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrictEventStoreError(f"invalid contribution event-stream manifest: {target}") from exc
    if manifest.get("schema") != SCHEMA or manifest.get("status") != STATUS:
        raise StrictEventStoreError("contribution event stream is not complete")
    source = Path(str(manifest.get("source_event_store_root", "")))
    source_manifest = source / "strict_event_store_manifest.json"
    if not source_manifest.is_file() or _sha256(source_manifest) != str(manifest.get("source_event_store_manifest_sha256", "")):
        raise StrictEventStoreError("contribution event stream source event-store lineage changed")
    index_path = target / str(manifest.get("parts", ""))
    if not index_path.is_file() or _sha256(index_path) != str(manifest.get("sha256", {}).get("parts", "")):
        raise StrictEventStoreError("contribution event-stream index hash differs from its manifest")
    index = pd.read_parquet(index_path)
    required = {
        "dataset", "path", "contract", "side", "head", "month", "meta_partition", "rows",
        "timestamp_column", "source_candidate_path", "source_contribution_path",
        "source_candidate_sha256", "sha256",
    }
    if index.empty or not required.issubset(index.columns):
        raise StrictEventStoreError("contribution event-stream index is incomplete")
    if set(index["dataset"].astype(str)) != {"feature_event_order", "resolution_event_order"}:
        raise StrictEventStoreError("contribution event-stream index has an invalid event-order dataset")
    if not index["timestamp_column"].astype(str).isin({"feature_generation_ts", "label_available_ts"}).all():
        raise StrictEventStoreError("contribution event-stream index has an invalid timestamp column")
    canonical = load_strict_event_store(source, verify_parts=False, verify_source=True)
    _validate_source_pair_correspondence(canonical, index, verify_source_parts=verify_parts)
    if verify_parts:
        for item in index.itertuples(index=False):
            path = target / str(item.path)
            if not path.is_file() or _sha256(path) != str(item.sha256):
                raise StrictEventStoreError("contribution event-stream physical part differs from its index")
            rows = _validate_event_physical_schema(path, source=str(item.path))
            if rows != int(item.rows):
                raise StrictEventStoreError("contribution event-stream physical row count differs from its index")
            _forbid_raw_leaf(EVENT_COLUMNS, source=f"contribution event stream {item.path}")
    return StrictContributionEventStreams(root=target, manifest_path=manifest_path, manifest=manifest, part_index=index)


__all__ = [
    "EVENT_COLUMNS", "SCHEMA", "STATUS", "StrictContributionEventStreams",
    "build_strict_contribution_event_streams", "load_strict_contribution_event_streams",
]
