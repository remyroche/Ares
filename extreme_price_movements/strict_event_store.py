"""Immutable, reusable token-free event store for strict leaf-reasoning runs.

The strict base artifacts are expensive to validate and to decode into
rule-family contributions.  This module turns an already validated
``StrictOOFFamilyInputSpool`` into a physical store which can be consumed by
predecessor selection, H1--H5, C taxonomy and successor OOF work without
repeating that work.  It is deliberately a *data contract*, not a feature
generator: no raw leaf IDs are retained and the contribution dataset contains
no outcome fields.

Layout (every data file is immutable and hash-indexed)::

    strict_event_store_manifest.json
    candidate_parts/contract=<sha>/side=<side>/head=<head>/month=<yyyy-mm>/partition=<...>/...
    contribution_parts/contract=<sha>/side=<side>/head=<head>/month=<yyyy-mm>/partition=<...>/...
    score_order/contract=<sha>/side=<side>/head=<head>/month=<yyyy-mm>/partition=<...>/...
    resolution_order/contract=<sha>/side=<side>/head=<head>/month=<yyyy-mm>/partition=<...>/...

``score_order`` and ``resolution_order`` contain only candidate/head records,
ordered respectively by feature-generation and label-resolution time.  A
consumer joins a selected candidate part to its matching token-free
contribution part only when it actually needs family rows.  In particular,
predecessor selectors can inspect ``inner_oof`` part metadata and an as-of
cutoff without decoding unrelated contribution data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # The production strict-spool pipeline already depends on pyarrow.
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - clear failure in reduced environments
    pa = None
    pc = None
    ds = pq = None

from .causal_leaf_health import CausalLeafHealthError
from .causal_leaf_health_artifacts import StrictOOFFamilyInputSpool


SCHEMA = "strict_leaf_reasoning_event_store_v1"
STATUS = "STRICT_LEAF_REASONING_EVENT_STORE_COMPLETED"
SPOOL_SCHEMA = "strict_oof_family_input_spool_v1"
SPOOL_STATUS = "STRICT_OOF_FAMILY_INPUT_SPOOL_COMPLETED"
RAW_LEAF_TOKENS = ("leaf_token", "leaf_id", "leaf_assignment", "raw_leaf")

CANDIDATE_COLUMNS = (
    "candidate_id", "decision_ts", "feature_generation_ts", "label_available_ts",
    "side_name", "head_name", "fold_id", "transport", "meta_partition",
    "feature_contract_sha256", "semantic_label", "head_prediction", "net_bps",
    "base_expected_bps", "asset",
)
CONTRIBUTION_COLUMNS = (
    "candidate_id", "__ts__", "side_name", "head_name", "fold_id",
    "transport", "meta_partition", "feature_contract_sha256", "rule_signature",
    "contribution_direction", "family_ensemble_tree_contribution",
)
RAW_SPOOL_CONTRIBUTION_COLUMNS = (
    "candidate_id", "__ts__", "side_name", "fold_id", "head_name", "rule_signature",
    "contribution_direction", "family_ensemble_tree_contribution",
)
IDENTITY_COLUMNS = (
    "candidate_id", "decision_ts", "side_name", "head_name", "fold_id",
    "transport", "meta_partition",
)


class StrictEventStoreError(CausalLeafHealthError):
    """Raised when a strict event-store lineage or physical invariant fails."""


@dataclass(frozen=True)
class StrictEventStoreConfig:
    """Physical-only controls; changing them never changes the population."""

    compression: str = "zstd"
    max_rows_per_part: int = 500_000

    def validate(self) -> None:
        if str(self.compression).lower() not in {"zstd", "snappy", "gzip", "none"}:
            raise StrictEventStoreError("unsupported parquet compression")
        if int(self.max_rows_per_part) < 1:
            raise StrictEventStoreError("max_rows_per_part must be positive")


@dataclass(frozen=True)
class StrictEventStore:
    """Verified read-only event-store descriptor.

    Paths in ``part_index`` are relative to ``root`` and must be opened only
    after :func:`load_strict_event_store` has verified the sealed manifest.
    """

    root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]
    part_index: pd.DataFrame


def _require_arrow() -> None:
    if pa is None or pq is None or pc is None or ds is None:
        raise StrictEventStoreError("pyarrow is required for strict event-store materialisation")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, *, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrictEventStoreError(f"invalid {source}: {path}") from exc
    if not isinstance(payload, dict):
        raise StrictEventStoreError(f"{source} must be a JSON object: {path}")
    return payload


def _utc(values: pd.Series, *, source: str) -> pd.Series:
    result = pd.to_datetime(values, utc=True, errors="coerce")
    if result.isna().any():
        raise StrictEventStoreError(f"{source} has invalid or missing UTC timestamps")
    return result


def _forbid_raw_leaf(columns: Iterable[object], *, source: str) -> None:
    bad = sorted(
        str(name) for name in columns
        if any(token in str(name).lower() for token in RAW_LEAF_TOKENS)
    )
    if bad:
        raise StrictEventStoreError(f"{source} contains forbidden raw leaf identifiers: {bad}")


def _safe_part(value: object) -> str:
    """Safe deterministic partition component without relying on user IDs."""

    text = str(value)
    if not text or any(char in text for char in ("/", "\\", "\x00")):
        raise StrictEventStoreError("invalid scope value for physical partition")
    return text


def _identity_hash(frame: pd.DataFrame, *, timestamp: str) -> str:
    columns = ["candidate_id", timestamp, "side_name", "head_name", "fold_id", "transport", "meta_partition"]
    ordered = frame.loc[:, columns].copy()
    ordered[timestamp] = pd.to_datetime(ordered[timestamp], utc=True).astype("int64")
    for column in columns:
        if column != timestamp:
            ordered[column] = ordered[column].astype("string")
    ordered = ordered.sort_values(columns, kind="stable")
    digest = hashlib.sha256()
    for row in ordered.itertuples(index=False, name=None):
        digest.update("\x1f".join(map(str, row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_spool(spool_root: str | Path, *, verify_parts: bool = True) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    root = Path(spool_root)
    manifest_path = root / "strict_family_input_spool_manifest.json"
    manifest = _json(manifest_path, source="strict family input spool manifest")
    if manifest.get("schema") != SPOOL_SCHEMA or manifest.get("status") != SPOOL_STATUS:
        raise StrictEventStoreError("input is not a completed strict family input spool")
    roots = manifest.get("strict_roots")
    root_hashes = manifest.get("strict_root_manifest_sha256")
    if not isinstance(roots, list) or not roots or not isinstance(root_hashes, dict):
        raise StrictEventStoreError("strict family spool lacks strict-root manifest lineage")
    for raw in roots:
        strict_root = Path(str(raw))
        current = strict_root / "strict_oof_reasoning_manifest.json"
        declared = root_hashes.get(str(raw))
        if not current.is_file() or not isinstance(declared, str) or _sha256(current) != declared:
            raise StrictEventStoreError("input strict root manifest no longer matches sealed spool lineage")
    index_name = manifest.get("pair_index")
    if not isinstance(index_name, str):
        raise StrictEventStoreError("strict family spool lacks pair index")
    index_path = root / index_name
    if not index_path.is_file():
        raise StrictEventStoreError("strict family spool pair index is missing")
    index = pd.read_parquet(index_path)
    required = {
        "part", "candidate_part", "contribution_part", "candidate_rows", "contribution_rows",
        "candidate_sha256", "contribution_sha256",
    }
    missing = sorted(required.difference(index.columns))
    if missing:
        raise StrictEventStoreError(f"strict family spool index lacks {missing}")
    if index.empty or index["part"].duplicated().any():
        raise StrictEventStoreError("strict family spool index has empty or duplicate parts")
    if verify_parts:
        for row in index.itertuples(index=False):
            candidate = root / "candidate_parts" / str(row.candidate_part)
            contribution = root / "contribution_parts" / str(row.contribution_part)
            if not candidate.is_file() or not contribution.is_file():
                raise StrictEventStoreError("strict family spool is missing a paired input part")
            if _sha256(candidate) != str(row.candidate_sha256) or _sha256(contribution) != str(row.contribution_sha256):
                raise StrictEventStoreError("strict family spool part hash does not match its index")
    return root, manifest, index.sort_values("part", kind="stable").reset_index(drop=True)


def _normalise_candidate(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    _forbid_raw_leaf(frame.columns, source=source)
    missing = sorted(set(CANDIDATE_COLUMNS).difference(frame.columns))
    if missing:
        raise StrictEventStoreError(f"{source} candidate part lacks {missing}")
    work = frame.loc[:, list(CANDIDATE_COLUMNS)].copy()
    for column in ("decision_ts", "feature_generation_ts", "label_available_ts"):
        work[column] = _utc(work[column], source=f"{source}.{column}")
    for column in ("candidate_id", "side_name", "head_name", "fold_id", "transport", "meta_partition", "feature_contract_sha256", "asset"):
        work[column] = work[column].astype("string")
    if work[["candidate_id", "side_name", "head_name", "fold_id", "transport", "meta_partition", "feature_contract_sha256"]].isna().any().any():
        raise StrictEventStoreError(f"{source} has null candidate identity/scope")
    if work["candidate_id"].str.strip().eq("").any() or work["feature_contract_sha256"].str.strip().eq("").any():
        raise StrictEventStoreError(f"{source} has blank candidate identity/scope")
    if not work["side_name"].str.lower().isin(("long", "short")).all():
        raise StrictEventStoreError(f"{source} has an invalid side")
    if not work["meta_partition"].isin(("inner_oof", "outer_test")).all():
        raise StrictEventStoreError(f"{source} has an invalid strict partition")
    if not work["feature_generation_ts"].le(work["decision_ts"]).all():
        raise StrictEventStoreError(f"{source} has feature time after decision")
    if not work["label_available_ts"].ge(work["decision_ts"]).all():
        raise StrictEventStoreError(f"{source} has label availability before decision")
    numeric = ("semantic_label", "head_prediction", "net_bps", "base_expected_bps")
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
        if not np.isfinite(work[column].to_numpy(dtype=float)).all():
            raise StrictEventStoreError(f"{source}.{column} must be finite")
    if work.duplicated(list(IDENTITY_COLUMNS)).any():
        raise StrictEventStoreError(f"{source} duplicates candidate/head identity")
    return work


def _normalise_contribution(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    _forbid_raw_leaf(frame.columns, source=source)
    # A StrictOOFFamilyInputSpool intentionally stores raw token-free family
    # rows without transport/partition/contract duplication.  Those fields
    # are attached from its paired validated candidate part below.  A future
    # store/spool is also allowed to have already attached them.
    missing = sorted(set(RAW_SPOOL_CONTRIBUTION_COLUMNS).difference(frame.columns))
    if missing:
        raise StrictEventStoreError(f"{source} contribution part lacks {missing}")
    present = [column for column in CONTRIBUTION_COLUMNS if column in frame.columns]
    work = frame.loc[:, present].copy()
    work["__ts__"] = _utc(work["__ts__"], source=f"{source}.__ts__")
    for column in ("candidate_id", "side_name", "head_name", "fold_id", "transport", "meta_partition", "feature_contract_sha256", "rule_signature", "contribution_direction"):
        if column not in work.columns:
            continue
        work[column] = work[column].astype("string")
    if work.isna().any().any() or work["rule_signature"].str.strip().eq("").any():
        raise StrictEventStoreError(f"{source} has null/blank token-free family fields")
    if not work["contribution_direction"].isin(("positive", "negative")).all():
        raise StrictEventStoreError(f"{source} has an invalid family direction")
    work["family_ensemble_tree_contribution"] = pd.to_numeric(
        work["family_ensemble_tree_contribution"], errors="coerce"
    )
    if not np.isfinite(work["family_ensemble_tree_contribution"].to_numpy(dtype=float)).all():
        raise StrictEventStoreError(f"{source} has non-finite family contribution")
    if np.isclose(work["family_ensemble_tree_contribution"].to_numpy(dtype=float), 0.0).any():
        raise StrictEventStoreError(f"{source} retains zero family contribution")
    return work


def _attach_paired_candidate_provenance(contribution: pd.DataFrame, candidate: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Attach strict partition/contract only from the paired candidate part."""

    missing = [column for column in ("transport", "meta_partition", "feature_contract_sha256") if column not in contribution.columns]
    basic_key = ["candidate_id", "__ts__", "side_name", "head_name", "fold_id"]
    lookup = candidate.loc[:, ["candidate_id", "decision_ts", "side_name", "head_name", "fold_id", "transport", "meta_partition", "feature_contract_sha256"]].rename(columns={"decision_ts": "__ts__"})
    if lookup.duplicated(basic_key).any():
        raise StrictEventStoreError(f"{source} paired candidate provenance is ambiguous")
    if missing:
        contribution = contribution.merge(lookup, on=basic_key, how="left", validate="many_to_one", indicator=True)
        if not contribution["_merge"].eq("both").all():
            raise StrictEventStoreError(f"{source} token-free contribution cannot prove paired candidate provenance")
        contribution = contribution.drop(columns="_merge")
    else:
        # Even a richer future spool must prove that it did not introduce a
        # different scope while serialising the token-free contribution rows.
        check = contribution.merge(lookup, on=basic_key, how="left", validate="many_to_one", suffixes=("", "_candidate"), indicator=True)
        if not check["_merge"].eq("both").all():
            raise StrictEventStoreError(f"{source} token-free contribution cannot prove paired candidate provenance")
        for column in ("transport", "meta_partition", "feature_contract_sha256"):
            if not check[column].astype(str).eq(check[f"{column}_candidate"].astype(str)).all():
                raise StrictEventStoreError(f"{source} contribution crosses paired candidate {column}")
    if sorted(set(CONTRIBUTION_COLUMNS).difference(contribution.columns)):
        raise StrictEventStoreError(f"{source} could not attach full strict contribution provenance")
    return contribution.loc[:, list(CONTRIBUTION_COLUMNS)].copy()


def _validate_pair(candidate: pd.DataFrame, contribution: pd.DataFrame, *, source: str) -> None:
    contribution_identity = ["candidate_id", "__ts__", "side_name", "head_name", "fold_id", "transport", "meta_partition"]
    lookup = candidate.loc[:, list(IDENTITY_COLUMNS) + ["feature_contract_sha256"]].rename(columns={"decision_ts": "__ts__"})
    if lookup.duplicated(contribution_identity).any():
        raise StrictEventStoreError(f"{source} candidate identities are ambiguous")
    joined = contribution.merge(lookup, on=contribution_identity, how="left", validate="many_to_one", suffixes=("", "_candidate"), indicator=True)
    if not joined["_merge"].eq("both").all():
        raise StrictEventStoreError(f"{source} contribution cannot prove candidate/head identity")
    if not joined["feature_contract_sha256"].eq(joined["feature_contract_sha256_candidate"]).all():
        raise StrictEventStoreError(f"{source} contribution crosses candidate feature contract")


def _candidate_table(frame: pd.DataFrame) -> pa.Table:
    _require_arrow()
    schema = pa.schema([
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
        pa.field("semantic_label", pa.float32()),
        pa.field("head_prediction", pa.float32()),
        pa.field("net_bps", pa.float32()),
        pa.field("base_expected_bps", pa.float32()),
        pa.field("asset", pa.dictionary(pa.int16(), pa.string())),
    ])
    table = pa.Table.from_pandas(frame.loc[:, list(CANDIDATE_COLUMNS)], preserve_index=False)
    return table.cast(schema, safe=False)


def _contribution_table(frame: pd.DataFrame) -> pa.Table:
    _require_arrow()
    schema = pa.schema([
        pa.field("candidate_id", pa.dictionary(pa.int32(), pa.string())),
        pa.field("__ts__", pa.timestamp("ns", tz="UTC")),
        pa.field("side_name", pa.dictionary(pa.int8(), pa.string())),
        pa.field("head_name", pa.dictionary(pa.int8(), pa.string())),
        pa.field("fold_id", pa.dictionary(pa.int16(), pa.string())),
        pa.field("transport", pa.dictionary(pa.int8(), pa.string())),
        pa.field("meta_partition", pa.dictionary(pa.int8(), pa.string())),
        pa.field("feature_contract_sha256", pa.dictionary(pa.int16(), pa.string())),
        pa.field("rule_signature", pa.dictionary(pa.int32(), pa.string())),
        pa.field("contribution_direction", pa.dictionary(pa.int8(), pa.string())),
        pa.field("family_ensemble_tree_contribution", pa.float32()),
    ])
    table = pa.Table.from_pandas(frame.loc[:, list(CONTRIBUTION_COLUMNS)], preserve_index=False)
    return table.cast(schema, safe=False)


def _write_table(table: pa.Table, path: Path, *, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression=None if compression == "none" else compression, use_dictionary=True)


def _part_path(kind: str, *, contract: str, side: str, head: str, month: str, partition: str, counter: int) -> Path:
    return Path(kind) / f"contract={_safe_part(contract)}" / f"side={_safe_part(side)}" / f"head={_safe_part(head)}" / f"month={_safe_part(month)}" / f"partition={_safe_part(partition)}" / f"part-{counter:06d}.parquet"


def _add_part_index_row(rows: list[dict[str, Any]], *, absolute: Path, relative: Path, kind: str, frame: pd.DataFrame, timestamp: str, paired_contribution: str | None = None) -> None:
    path = relative
    rows.append({
        "dataset": kind,
        "path": str(path),
        "contract": str(frame["feature_contract_sha256"].iloc[0]),
        "side": str(frame["side_name"].iloc[0]),
        "head": str(frame["head_name"].iloc[0]),
        "month": str(pd.Timestamp(frame[timestamp].iloc[0]).strftime("%Y-%m")),
        "meta_partition": str(frame["meta_partition"].iloc[0]),
        "rows": int(len(frame)),
        "timestamp_column": timestamp,
        "min_timestamp": pd.Timestamp(frame[timestamp].min()).isoformat(),
        "max_timestamp": pd.Timestamp(frame[timestamp].max()).isoformat(),
        "min_label_available_ts": (
            pd.Timestamp(frame["label_available_ts"].min()).isoformat()
            if "label_available_ts" in frame.columns else None
        ),
        "max_label_available_ts": (
            pd.Timestamp(frame["label_available_ts"].max()).isoformat()
            if "label_available_ts" in frame.columns else None
        ),
        "candidate_identity_sha256": _identity_hash(frame, timestamp=timestamp),
        "paired_contribution_path": paired_contribution,
        "sha256": _sha256(absolute),
    })


def _write_candidate_slice(
    temporary: Path, part_rows: list[dict[str, Any]], *, candidate: pd.DataFrame,
    contribution: pd.DataFrame, contract: str, side: str, head: str, month: str, partition: str,
    counter: int, config: StrictEventStoreConfig,
) -> int:
    """Write matching monthly parts and two pre-sorted candidate streams."""

    candidate = candidate.sort_values(["candidate_id", "fold_id", "transport", "meta_partition"], kind="stable").reset_index(drop=True)
    contribution = contribution.sort_values(["candidate_id", "__ts__", "rule_signature", "contribution_direction"], kind="stable").reset_index(drop=True)
    cpath = _part_path("candidate_parts", contract=contract, side=side, head=head, month=month, partition=partition, counter=counter)
    fpath = _part_path("contribution_parts", contract=contract, side=side, head=head, month=month, partition=partition, counter=counter)
    _write_table(_candidate_table(candidate), temporary / cpath, compression=config.compression)
    _write_table(_contribution_table(contribution), temporary / fpath, compression=config.compression)
    _add_part_index_row(part_rows, absolute=temporary / cpath, relative=cpath, kind="candidate", frame=candidate, timestamp="decision_ts", paired_contribution=str(fpath))
    _add_part_index_row(part_rows, absolute=temporary / fpath, relative=fpath, kind="contribution", frame=contribution.rename(columns={"__ts__": "decision_ts"}), timestamp="decision_ts")
    return counter + 1


def _sql_literal(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _materialize_order_streams(
    temporary: Path, part_rows: list[dict[str, Any]], *, counter: int, config: StrictEventStoreConfig,
) -> int:
    """Write globally sorted reusable streams, once, by strict scope.

    Candidate files are already bounded physical parts.  DuckDB supplies the
    single external sort needed to turn them into globally ordered streams;
    downstream readers only concatenate the ordered parts and never sort the
    7M-row population again.  We split output at sorted-batch boundaries (and
    timestamp months for an inspectable layout), so ranges never overlap for a
    given ``contract/side/head/order`` stream.
    """

    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - spool pipeline requires it too
        raise StrictEventStoreError("duckdb is required to materialise globally ordered event streams") from exc
    candidate_index = pd.DataFrame(part_rows).loc[lambda item: item["dataset"].eq("candidate")].copy()
    scopes = candidate_index.loc[:, ["contract", "side", "head"]].drop_duplicates().sort_values(
        ["contract", "side", "head"], kind="stable"
    )
    candidate_root = str(temporary / "candidate_parts" / "**" / "*.parquet").replace("'", "''")
    fields = ", ".join(CANDIDATE_COLUMNS)
    for scope in scopes.itertuples(index=False):
        where = (
            f"feature_contract_sha256={_sql_literal(scope.contract)} AND "
            f"side_name={_sql_literal(scope.side)} AND head_name={_sql_literal(scope.head)}"
        )
        for kind, time_column in (("score_order", "feature_generation_ts"), ("resolution_order", "label_available_ts")):
            sql = (
                f"SELECT {fields} FROM read_parquet('{candidate_root}', hive_partitioning=false) "
                f"WHERE {where} ORDER BY {time_column}, candidate_id, fold_id, transport, meta_partition"
            )
            with duckdb.connect(database=":memory:") as connection:
                reader = connection.execute(sql).to_arrow_reader(batch_size=int(config.max_rows_per_part))
                for batch in reader:
                    ordered = batch.to_pandas()
                    if ordered.empty:
                        continue
                    ordered[time_column] = pd.to_datetime(ordered[time_column], utc=True, errors="coerce")
                    if ordered[time_column].isna().any():
                        raise StrictEventStoreError("ordered event stream has invalid timestamp")
                    # A DuckDB batch can cross a month; split it so the on-disk
                    # layout remains contract/side/head/month partitioned.
                    ordered["__order_month__"] = ordered[time_column].dt.strftime("%Y-%m")
                    for month, group in ordered.groupby("__order_month__", sort=False, observed=True):
                        group = group.drop(columns="__order_month__").reset_index(drop=True)
                        partition_values = group["meta_partition"].astype(str).unique()
                        # Stream ordering legitimately spans inner/outer.  The
                        # physical partition name records that fact while the
                        # candidate field preserves the exact original state.
                        partition = str(partition_values[0]) if len(partition_values) == 1 else "mixed"
                        path = _part_path(
                            kind, contract=str(scope.contract), side=str(scope.side), head=str(scope.head),
                            month=str(month), partition=partition, counter=counter,
                        )
                        _write_table(_candidate_table(group), temporary / path, compression=config.compression)
                        _add_part_index_row(
                            part_rows, absolute=temporary / path, relative=path, kind=kind,
                            frame=group, timestamp=time_column,
                        )
                        counter += 1
    return counter


def build_strict_event_store(
    spool_root: str | Path,
    output_dir: str | Path,
    *,
    config: StrictEventStoreConfig = StrictEventStoreConfig(),
) -> StrictEventStore:
    """Build a sealed reusable event store from a verified strict spool.

    The source spool is checked before every read.  Contributions are decoded
    exactly once during this build and only written after same candidate/head
    identity and feature-contract reconciliation.  No current strict or spool
    artifact is changed.
    """

    _require_arrow()
    config.validate()
    source_root, spool_manifest, spool_index = _read_spool(spool_root, verify_parts=True)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite strict event store: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    part_rows: list[dict[str, Any]] = []
    source_parts: list[dict[str, Any]] = []
    counter = 0
    try:
        for source in spool_index.itertuples(index=False):
            candidate_path = source_root / "candidate_parts" / str(source.candidate_part)
            contribution_path = source_root / "contribution_parts" / str(source.contribution_part)
            candidate = _normalise_candidate(pd.read_parquet(candidate_path), source=str(candidate_path))
            contribution = _normalise_contribution(pd.read_parquet(contribution_path), source=str(contribution_path))
            contribution = _attach_paired_candidate_provenance(
                contribution, candidate, source=f"spool part {source.part}",
            )
            _validate_pair(candidate, contribution, source=f"spool part {source.part}")
            source_parts.append({
                "spool_part": int(source.part),
                "candidate_path": str(source.candidate_part),
                "contribution_path": str(source.contribution_part),
                "candidate_sha256": str(source.candidate_sha256),
                "contribution_sha256": str(source.contribution_sha256),
                "candidate_rows": int(len(candidate)),
                "contribution_rows": int(len(contribution)),
            })
            candidate["__month__"] = candidate["decision_ts"].dt.strftime("%Y-%m")
            # Attach the decision-month once.  The previous straightforward
            # implementation re-merged the entire long contribution table
            # for every candidate month (often 6--10 times per artifact).
            # This one vectorised many-to-one merge is the important physical
            # build-time saving: monthly writes then filter an already mapped
            # token-free frame.
            contribution = contribution.merge(
                candidate.loc[:, list(IDENTITY_COLUMNS) + ["__month__"]].rename(columns={"decision_ts": "__ts__"}),
                on=["candidate_id", "__ts__", "side_name", "head_name", "fold_id", "transport", "meta_partition"],
                how="inner", validate="many_to_one",
            )
            if contribution.empty:
                raise StrictEventStoreError("spool contribution has no matching strict candidate after month attachment")
            # ``meta_partition`` is an additional physical subpartition.  It
            # makes predecessor-only source selection exact without opening a
            # contribution file that also contains outer-test evidence.
            scope_columns = ["feature_contract_sha256", "side_name", "head_name", "__month__", "meta_partition"]
            for (contract, side, head, month, partition), cgroup in candidate.groupby(scope_columns, sort=True, observed=True):
                cgroup = cgroup.drop(columns="__month__").copy()
                fgroup = contribution.loc[
                    contribution["__month__"].astype(str).eq(str(month))
                    & contribution["meta_partition"].astype(str).eq(str(partition))
                ].copy()
                if fgroup.empty:
                    raise StrictEventStoreError("monthly candidate group has no matching family contribution")
                # The physical row cap keeps all kinds aligned by candidate
                # groups.  A huge contribution group is still written as one
                # candidate-aligned partition rather than splitting family rows.
                chunks = [cgroup] if len(cgroup) <= int(config.max_rows_per_part) else [cgroup.iloc[start:start + int(config.max_rows_per_part)].copy() for start in range(0, len(cgroup), int(config.max_rows_per_part))]
                for chunk in chunks:
                    if len(chunks) == 1:
                        fchunk = fgroup
                    else:
                        chunk_key = chunk.loc[:, list(IDENTITY_COLUMNS)].rename(columns={"decision_ts": "__ts__"})
                        fchunk = fgroup.merge(chunk_key, on=["candidate_id", "__ts__", "side_name", "head_name", "fold_id", "transport", "meta_partition"], how="inner", validate="many_to_one")
                    counter = _write_candidate_slice(
                        temporary, part_rows, candidate=chunk, contribution=fchunk,
                        contract=str(contract), side=str(side), head=str(head), month=str(month), partition=str(partition),
                        counter=counter, config=config,
                    )
        counter = _materialize_order_streams(temporary, part_rows, counter=counter, config=config)
        part_index = pd.DataFrame(part_rows)
        if part_index.empty:
            raise StrictEventStoreError("strict event store received no candidate/head parts")
        # Every candidate part needs exactly one paired contribution and each
        # event-order stream needs a matching candidate identity population.
        candidates = part_index.loc[part_index["dataset"].eq("candidate")].copy()
        if candidates["candidate_identity_sha256"].duplicated().any() and len(candidates) != 1:
            # Identity hashes may repeat only if a source duplicated a strict
            # part; that is unsafe because physical paths would then look like
            # distinct observations downstream.
            raise StrictEventStoreError("event store contains duplicated candidate identity parts")
        for row in candidates.itertuples(index=False):
            if not isinstance(row.paired_contribution_path, str):
                raise StrictEventStoreError("candidate part lacks paired contribution path")
        index_path = temporary / "event_store_parts.parquet"
        part_index.to_parquet(index_path, index=False, compression=config.compression)
        source_path = temporary / "source_spool_parts.parquet"
        pd.DataFrame(source_parts).to_parquet(source_path, index=False, compression=config.compression)
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "status": STATUS,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
            "source": {
                "spool_root": str(source_root.resolve()),
                "spool_manifest": "strict_family_input_spool_manifest.json",
                "spool_manifest_sha256": _sha256(source_root / "strict_family_input_spool_manifest.json"),
                "strict_roots": list(spool_manifest["strict_roots"]),
                "strict_root_manifest_sha256": dict(spool_manifest["strict_root_manifest_sha256"]),
                "source_parts": source_path.name,
            },
            "parts": index_path.name,
            "row_counts": {
                "candidate_rows": int(candidates["rows"].sum()),
                "contribution_rows": int(part_index.loc[part_index["dataset"].eq("contribution"), "rows"].sum()),
                "score_rows": int(part_index.loc[part_index["dataset"].eq("score_order"), "rows"].sum()),
                "resolution_rows": int(part_index.loc[part_index["dataset"].eq("resolution_order"), "rows"].sum()),
                "physical_part_rows": int(len(part_index)),
            },
            "contract": {
                "candidate_columns": list(CANDIDATE_COLUMNS),
                "contribution_columns": list(CONTRIBUTION_COLUMNS),
                "identity": list(IDENTITY_COLUMNS),
                "partitioning": "contract / side / head / decision-month / strict-partition",
                "score_order": "candidate/head records sorted by feature_generation_ts, candidate_id, fold, transport, partition",
                "resolution_order": "candidate/head records sorted by label_available_ts, candidate_id, fold, transport, partition",
                "raw_leaf_ids": "rejected; no local leaf token, leaf assignment or raw leaf ID is persisted",
                "selection_safety": "contribution partitions have no semantic_label, net_bps, base_expected_bps or other outcome fields; cutoff readers select only inner_oof candidate/contribution parts before decode",
                "lineage": "sealed to strict root manifest hashes and source spool part hashes",
                "physical_types": "dictionary encoded string dimensions; float32 continuous values; UTC nanosecond timestamps",
            },
            "sha256": {
                "parts": _sha256(index_path),
                "source_parts": _sha256(source_path),
            },
        }
        manifest_path = temporary / "strict_event_store_manifest.json"
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    # Every physical file was hashed into the sealed index immediately before
    # the atomic rename.  Avoid a second multi-gigabyte full-file hash pass at
    # the end of a successful build; callers that consume an existing store
    # still verify physical hashes by default.
    return load_strict_event_store(target, verify_parts=False)


def load_strict_event_store(root: str | Path, *, verify_parts: bool = True, verify_source: bool = True) -> StrictEventStore:
    """Load and verify a sealed event store without materialising its tables."""

    _require_arrow()
    target = Path(root)
    manifest_path = target / "strict_event_store_manifest.json"
    manifest = _json(manifest_path, source="strict event store manifest")
    if manifest.get("schema") != SCHEMA or manifest.get("status") != STATUS:
        raise StrictEventStoreError("event store is not complete")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise StrictEventStoreError("event store lacks source lineage")
    if verify_source:
        spool_root = Path(str(source.get("spool_root", "")))
        expected = source.get("spool_manifest_sha256")
        spool_manifest = spool_root / "strict_family_input_spool_manifest.json"
        if not spool_manifest.is_file() or not isinstance(expected, str) or _sha256(spool_manifest) != expected:
            raise StrictEventStoreError("event store source spool manifest no longer matches sealed lineage")
        roots = source.get("strict_roots")
        hashes = source.get("strict_root_manifest_sha256")
        if not isinstance(roots, list) or not isinstance(hashes, Mapping):
            raise StrictEventStoreError("event store source lacks strict root hashes")
        for raw in roots:
            strict_manifest = Path(str(raw)) / "strict_oof_reasoning_manifest.json"
            if not strict_manifest.is_file() or _sha256(strict_manifest) != str(hashes.get(str(raw), "")):
                raise StrictEventStoreError("event store strict-root manifest changed after sealing")
    part_name = manifest.get("parts")
    expected_parts = manifest.get("sha256", {}).get("parts") if isinstance(manifest.get("sha256"), Mapping) else None
    index_path = target / str(part_name)
    if not index_path.is_file() or not isinstance(expected_parts, str) or _sha256(index_path) != expected_parts:
        raise StrictEventStoreError("event store part index hash differs from sealed manifest")
    parts = pd.read_parquet(index_path)
    required = {"dataset", "path", "contract", "side", "head", "month", "meta_partition", "rows", "timestamp_column", "min_timestamp", "max_timestamp", "min_label_available_ts", "max_label_available_ts", "candidate_identity_sha256", "paired_contribution_path", "sha256"}
    missing = sorted(required.difference(parts.columns))
    if missing or parts.empty:
        raise StrictEventStoreError(f"event store part index lacks {missing} or is empty")
    if verify_parts:
        for row in parts.itertuples(index=False):
            path = target / str(row.path)
            if not path.is_file():
                raise StrictEventStoreError(f"event store physical part is missing: {row.path}")
            if _sha256(path) != str(row.sha256):
                raise StrictEventStoreError(f"event store physical part hash differs from sealed index: {row.path}")
            schema = pq.ParquetFile(path).schema_arrow.names
            _forbid_raw_leaf(schema, source=f"event store part {row.path}")
            if str(row.dataset) == "contribution":
                missing_columns = sorted(set(CONTRIBUTION_COLUMNS).difference(schema))
                forbidden_outcomes = sorted({"semantic_label", "net_bps", "base_expected_bps"}.intersection(schema))
                if missing_columns or forbidden_outcomes:
                    raise StrictEventStoreError("event-store contribution physical contract is invalid")
            else:
                missing_columns = sorted(set(CANDIDATE_COLUMNS).difference(schema))
                if missing_columns:
                    raise StrictEventStoreError("event-store candidate physical contract is invalid")
    return StrictEventStore(root=target, manifest_path=manifest_path, manifest=manifest, part_index=parts)


def source_parts_for_cutoff(
    store: StrictEventStore | str | Path,
    cutoff_utc: str | pd.Timestamp,
    *,
    datasets: Sequence[str] = ("candidate", "contribution"),
) -> pd.DataFrame:
    """Return pre-cutoff inner-OOF physical parts without reading them.

    This is the performance boundary for predecessor-only selection.  It uses
    part index metadata only: source consumers may then decode exactly the
    returned pair paths.  The conservative condition is ``max label time <
    cutoff`` on the matching *candidate* part; no outer or unresolved rows
    can be included.
    """

    resolved = store if isinstance(store, StrictEventStore) else load_strict_event_store(store, verify_parts=False)
    cutoff = pd.to_datetime(cutoff_utc, utc=True, errors="coerce")
    if pd.isna(cutoff):
        raise StrictEventStoreError("cutoff_utc must be a finite UTC timestamp")
    requested = set(map(str, datasets))
    if not requested.issubset({"candidate", "contribution", "score_order", "resolution_order"}):
        raise StrictEventStoreError("unknown requested event-store dataset")
    parts = resolved.part_index
    candidate = parts.loc[parts["dataset"].eq("candidate")].copy()
    candidate_max = pd.to_datetime(candidate["max_label_available_ts"], utc=True, errors="coerce")
    # Candidate part was partitioned by decision month, but label time can
    # cross the boundary; this metadata filter remains correct by using the
    # actual maximum resolution time from its records.
    candidate = candidate.loc[candidate["meta_partition"].astype(str).eq("inner_oof") & candidate_max.lt(cutoff)].copy()
    eligible_paths = set(candidate["path"].astype(str))
    paired_paths = set(candidate["paired_contribution_path"].dropna().astype(str))
    output = parts.loc[
        (parts["dataset"].isin(requested))
        & (
            parts["path"].astype(str).isin(eligible_paths)
            | parts["path"].astype(str).isin(paired_paths)
        )
    ].copy()
    # Score/resolution parts share candidate physical identities.  Restrict
    # them via hash rather than a path convention.
    eligible_hashes = set(candidate["candidate_identity_sha256"].astype(str))
    output = pd.concat([
        output,
        parts.loc[
            parts["dataset"].isin(requested & {"score_order", "resolution_order"})
            & parts["candidate_identity_sha256"].astype(str).isin(eligible_hashes)
        ],
    ], ignore_index=True).drop_duplicates("path")
    return output.sort_values(["dataset", "contract", "side", "head", "month", "path"], kind="stable").reset_index(drop=True)


def iter_event_store_parts(
    store: StrictEventStore | str | Path,
    *,
    dataset: str,
    cutoff_utc: str | pd.Timestamp | None = None,
    columns: Sequence[str] | None = None,
) -> Iterator[tuple[Mapping[str, Any], pd.DataFrame]]:
    """Yield verified physical parts; cutoff selection happens before decode."""

    resolved = store if isinstance(store, StrictEventStore) else load_strict_event_store(store, verify_parts=False)
    if cutoff_utc is None:
        index = resolved.part_index.loc[resolved.part_index["dataset"].eq(str(dataset))].copy()
    else:
        index = source_parts_for_cutoff(resolved, cutoff_utc, datasets=(str(dataset),))
        index = index.loc[index["dataset"].eq(str(dataset))].copy()
    if str(dataset) in {"score_order", "resolution_order"}:
        # The materialiser externally sorts every scope and emits sequential,
        # non-overlapping sorted batches.  This index order is therefore a
        # k=1 append stream, not a fresh expensive dataframe sort.
        index["__min_ns__"] = pd.to_datetime(index["min_timestamp"], utc=True, errors="coerce").astype("int64")
        index = index.sort_values(["contract", "side", "head", "__min_ns__", "path"], kind="stable")
    else:
        index = index.sort_values(["contract", "side", "head", "month", "path"], kind="stable")
    for row in index.itertuples(index=False):
        path = resolved.root / str(row.path)
        yield row._asdict(), pd.read_parquet(path, columns=list(columns) if columns is not None else None)


def iter_predecessor_selection_pairs(
    store: StrictEventStore | str | Path,
    cutoff_utc: str | pd.Timestamp,
    *,
    contribution_columns: Sequence[str] = (
        "candidate_id", "__ts__", "side_name", "head_name", "fold_id", "transport",
        "meta_partition", "feature_contract_sha256", "rule_signature",
        "contribution_direction", "family_ensemble_tree_contribution",
    ),
) -> Iterator[tuple[Mapping[str, Any], pd.DataFrame, pd.DataFrame]]:
    """Yield exact pre-cutoff inner-OOF candidate/family inputs for selection.

    Candidate parts are read and filtered first.  A matching contribution part
    is never opened when no candidate in it is eligible; when it is opened,
    Arrow projects only the declared token-free family columns and filters to
    the selected candidate IDs.  This is intentionally separate from health
    because selection must never receive outcomes attached to family rows.
    """

    _require_arrow()
    resolved = store if isinstance(store, StrictEventStore) else load_strict_event_store(store, verify_parts=False)
    cutoff = pd.to_datetime(cutoff_utc, utc=True, errors="coerce")
    if pd.isna(cutoff):
        raise StrictEventStoreError("cutoff_utc must be a finite UTC timestamp")
    wanted = tuple(map(str, contribution_columns))
    if not set(wanted).issubset(CONTRIBUTION_COLUMNS):
        raise StrictEventStoreError("predecessor selection can read only token-free contribution columns")
    candidates = resolved.part_index.loc[
        resolved.part_index["dataset"].eq("candidate")
        & resolved.part_index["meta_partition"].astype(str).eq("inner_oof")
    ].sort_values(["contract", "side", "head", "month", "path"], kind="stable")
    for item in candidates.itertuples(index=False):
        candidate_path = resolved.root / str(item.path)
        candidate = pd.read_parquet(candidate_path)
        candidate["label_available_ts"] = pd.to_datetime(candidate["label_available_ts"], utc=True, errors="coerce")
        eligible = candidate.loc[candidate["label_available_ts"].lt(cutoff)].copy()
        if eligible.empty:
            continue
        contribution_relative = str(item.paired_contribution_path)
        contribution_path = resolved.root / contribution_relative
        if not contribution_path.is_file():
            raise StrictEventStoreError("eligible candidate part has no paired contribution part")
        candidate_ids = pa.array(eligible["candidate_id"].astype(str).unique())
        table = ds.dataset(contribution_path, format="parquet").to_table(
            columns=list(wanted), filter=pc.field("candidate_id").isin(candidate_ids),
        )
        contribution = table.to_pandas()
        if contribution.empty:
            raise StrictEventStoreError("eligible selection candidate has no token-free contribution evidence")
        allowed = eligible.loc[:, list(IDENTITY_COLUMNS)].rename(columns={"decision_ts": "__ts__"})
        contribution = contribution.merge(
            allowed, on=["candidate_id", "__ts__", "side_name", "head_name", "fold_id", "transport", "meta_partition"],
            how="inner", validate="many_to_one",
        )
        if contribution.empty:
            raise StrictEventStoreError("contribution scanner returned no exact eligible candidate identity")
        yield item._asdict(), eligible, contribution


__all__ = [
    "CANDIDATE_COLUMNS", "CONTRIBUTION_COLUMNS", "IDENTITY_COLUMNS", "SCHEMA", "STATUS",
    "StrictEventStore", "StrictEventStoreConfig", "StrictEventStoreError",
    "build_strict_event_store", "iter_event_store_parts", "iter_predecessor_selection_pairs",
    "load_strict_event_store", "source_parts_for_cutoff",
]
