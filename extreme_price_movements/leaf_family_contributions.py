"""Token-free, fold-local leaf-family contribution materialisation.

The strict base-reasoning artifact intentionally keeps opaque leaf tokens only
inside ``leaf_assignments.parquet`` and ``leaf_rule_catalog.parquet``.  This
module is the narrow lineage bridge for the one permitted join:

``same artifact assignment token -> same artifact catalog -> rule signature``.

It emits a long candidate/family table with *no* leaf token, tree column, or
raw LightGBM identifier.  A structural rule signature may recur across folds,
but contribution values are never looked up outside the artifact that emitted
the assignment.  This distinction matters: a leaf token is local to a fitted
tree while a rule signature is the safe cross-fold family description.

The reader is deliberately bounded.  It streams exactly one assignment column
at a time in row batches, spills mapped contribution rows into candidate-row
buckets, and then collapses each bucket before emitting the final table.  Peak
memory is therefore one identity table, one assignment batch, and one bounded
output bucket rather than the full wide assignment matrix or all tree rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator

import numpy as np
import pandas as pd

try:  # Production and tests use pyarrow; fail clearly if a minimal env does not.
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised only in minimal deployments
    pa = None
    pq = None


STRICT_STATUS = "MATERIALIZED_STRICT_OOF"
IDENTITY = ("candidate_id", "__ts__", "side_name")
SCOPE = ("head_name", "fold_id")
TREE_PREFIX = "leaf_assignment__"
INTERNAL_ROW = "__family_contribution_row_position__"
VALUE_COLUMN = "family_ensemble_tree_contribution"
OUTPUT_COLUMNS = (
    "candidate_id",
    "__ts__",
    "side_name",
    "fold_id",
    "head_name",
    "rule_signature",
    "contribution_direction",
    VALUE_COLUMN,
)
FORBIDDEN_OUTPUT_TOKENS = ("leaf_token", "leaf_id", "leaf_assignment", "raw_leaf")


class LeafFamilyContributionError(ValueError):
    """Raised when a strict per-head artifact cannot prove local lineage."""


@dataclass(frozen=True)
class LeafFamilyContributionConfig:
    """Hard memory and numerical controls for one strict artifact.

    ``max_rows_per_output_bucket`` bounds the pessimistic number of mapped
    selected-tree rows in one temporary bucket.  It is not a statistical
    setting and it does not change the emitted values.
    """

    assignment_batch_rows: int = 50_000
    max_rows_per_output_bucket: int = 250_000
    additive_atol: float = 1e-8
    additive_rtol: float = 1e-6

    def validate(self) -> None:
        if int(self.assignment_batch_rows) <= 0:
            raise LeafFamilyContributionError("assignment_batch_rows must be positive")
        if int(self.max_rows_per_output_bucket) <= 0:
            raise LeafFamilyContributionError("max_rows_per_output_bucket must be positive")
        if float(self.additive_atol) < 0.0 or float(self.additive_rtol) < 0.0:
            raise LeafFamilyContributionError("additive tolerances must be non-negative")


@dataclass(frozen=True)
class LeafFamilyContributionResult:
    """Materialised token-free family contribution table metadata."""

    artifact_dir: Path
    output_path: Path | None
    candidate_count: int
    contribution_row_count: int
    assignment_column_count: int
    family_contribution_total: float


def _require_pyarrow() -> None:
    if pa is None or pq is None:
        raise LeafFamilyContributionError(
            "pyarrow is required for bounded leaf-family contribution materialisation"
        )


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LeafFamilyContributionError(f"invalid strict reasoning manifest: {path}") from exc
    if not isinstance(value, dict):
        raise LeafFamilyContributionError("strict reasoning manifest must be a JSON object")
    return value


def _scope_from_manifest(manifest: dict[str, Any]) -> tuple[str, str, str]:
    if str(manifest.get("status")) != STRICT_STATUS:
        raise LeafFamilyContributionError("artifact is not MATERIALIZED_STRICT_OOF")
    head = str(manifest.get("head_name", "")).strip()
    side = str(manifest.get("side_name", "")).strip().lower()
    fold = str(manifest.get("fold_id", "")).strip()
    if not head or not fold or side not in {"long", "short"}:
        raise LeafFamilyContributionError("strict manifest has an invalid head/side/fold scope")
    return head, side, fold


def _utc(values: pd.Series, *, source: str) -> pd.Series:
    value = pd.to_datetime(values, utc=True, errors="coerce")
    if value.isna().any():
        raise LeafFamilyContributionError(f"{source} contains an invalid UTC timestamp")
    return value


def _identity_from_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    expected_side: str,
    expected_head: str | None = None,
    expected_fold: str | None = None,
) -> pd.DataFrame:
    required = [*IDENTITY]
    if expected_head is not None:
        required.append("head_name")
    if expected_fold is not None:
        required.append("fold_id")
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise LeafFamilyContributionError(f"{source} is missing required columns: {missing}")
    out = frame.loc[:, list(IDENTITY)].copy().reset_index(drop=True)
    out["candidate_id"] = out["candidate_id"].astype("string")
    out["__ts__"] = _utc(out["__ts__"], source=f"{source}.__ts__")
    out["side_name"] = out["side_name"].astype("string").str.lower()
    if out.isna().any().any() or out["candidate_id"].str.strip().eq("").any():
        raise LeafFamilyContributionError(f"{source} has a null or blank candidate identity")
    if not out["side_name"].eq(expected_side).all():
        raise LeafFamilyContributionError(f"{source} crosses the artifact side scope")
    if out.duplicated(list(IDENTITY)).any():
        raise LeafFamilyContributionError(f"{source} has duplicate candidate identities")
    if expected_head is not None:
        if not frame["head_name"].astype(str).eq(expected_head).all():
            raise LeafFamilyContributionError(f"{source} crosses the artifact head scope")
    if expected_fold is not None:
        if not frame["fold_id"].astype(str).eq(expected_fold).all():
            raise LeafFamilyContributionError(f"{source} crosses the artifact fold scope")
    return out


def _same_identity(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return (
        len(left) == len(right)
        and left["candidate_id"].astype("string").equals(right["candidate_id"].astype("string"))
        and left["__ts__"].astype("int64").equals(right["__ts__"].astype("int64"))
        and left["side_name"].astype("string").equals(right["side_name"].astype("string"))
    )


def _tree_column(model_slot: object, head_tree_slot: object) -> str:
    try:
        model = int(model_slot)
        tree = int(head_tree_slot)
    except (TypeError, ValueError) as exc:
        raise LeafFamilyContributionError("catalog model_slot/head_tree_slot must be integer-like") from exc
    if model < 0 or tree < 0:
        raise LeafFamilyContributionError("catalog model_slot/head_tree_slot must be non-negative")
    return f"{TREE_PREFIX}model_{model:02d}_head_tree_{tree:03d}"


def _catalog_lookup(
    catalog: pd.DataFrame,
    *,
    expected_head: str,
    expected_side: str,
    expected_fold: str,
    manifest: dict[str, Any],
) -> dict[str, tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray]]:
    required = {
        "head_name",
        "side_name",
        "fold_id",
        "model_slot",
        "head_tree_slot",
        "leaf_token",
        "rule_signature",
        "tree_leaf_value",
        "ensemble_tree_contribution",
    }
    missing = sorted(required.difference(catalog.columns))
    if missing:
        raise LeafFamilyContributionError(f"leaf rule catalog is missing required columns: {missing}")
    if catalog.empty:
        raise LeafFamilyContributionError("leaf rule catalog is empty")
    work = catalog.loc[:, list(required)].copy()
    if (
        not work["head_name"].astype(str).eq(expected_head).all()
        or not work["side_name"].astype(str).str.lower().eq(expected_side).all()
        or not work["fold_id"].astype(str).eq(expected_fold).all()
    ):
        raise LeafFamilyContributionError("leaf rule catalog crosses the manifest head/side/fold scope")
    if work["leaf_token"].isna().any():
        raise LeafFamilyContributionError("leaf rule catalog contains a null local token")
    if work["rule_signature"].isna().any() or work["rule_signature"].astype(str).str.strip().eq("").any():
        raise LeafFamilyContributionError("leaf rule catalog contains a blank rule signature")
    try:
        work["leaf_token"] = pd.to_numeric(work["leaf_token"], errors="raise").astype(np.uint64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LeafFamilyContributionError("leaf rule catalog local tokens must be unsigned integers") from exc
    contribution = pd.to_numeric(work["ensemble_tree_contribution"], errors="coerce").to_numpy(np.float64)
    leaf_value = pd.to_numeric(work["tree_leaf_value"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(contribution).all() or not np.isfinite(leaf_value).all():
        raise LeafFamilyContributionError("catalog additive values must be finite")
    model_hashes = manifest.get("provenance", {}).get("model_hashes")
    if not isinstance(model_hashes, list) or not model_hashes:
        raise LeafFamilyContributionError("strict manifest lacks the fitted ensemble lineage")
    expected_contribution = leaf_value / float(len(model_hashes))
    if not np.allclose(contribution, expected_contribution, rtol=1e-6, atol=1e-8):
        raise LeafFamilyContributionError(
            "catalog ensemble_tree_contribution does not reconcile to tree_leaf_value / ensemble size"
        )
    work["tree_column"] = [
        _tree_column(model, tree)
        for model, tree in zip(work["model_slot"], work["head_tree_slot"], strict=True)
    ]
    if work.duplicated(["tree_column", "leaf_token"]).any():
        raise LeafFamilyContributionError("catalog duplicates a local token within one assignment column")
    lookup: dict[str, tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray]] = {}
    for tree_column, cell in work.groupby("tree_column", sort=True, observed=True):
        ordered = cell.sort_values("leaf_token", kind="stable")
        token = pd.Index(ordered["leaf_token"].to_numpy(dtype=np.uint64, copy=False))
        if token.has_duplicates:
            raise LeafFamilyContributionError("catalog local token lookup is ambiguous")
        values = ordered["ensemble_tree_contribution"].to_numpy(dtype=np.float64, copy=False)
        direction = np.where(values > 0.0, "positive", "negative").astype(object)
        # Zero-valued leaves are exactly additive no-ops.  They are retained in
        # this local lookup for reconciliation, then excluded from the emitted
        # long table because they carry no positive/negative family evidence.
        lookup[str(tree_column)] = (
            token,
            ordered["rule_signature"].astype(str).to_numpy(dtype=object, copy=False),
            values,
            direction,
        )
    return lookup


def _read_assignment_identity(path: Path, *, head: str, side: str, fold: str) -> pd.DataFrame:
    _require_pyarrow()
    schema = set(pq.ParquetFile(path).schema.names)
    required = {*IDENTITY, *SCOPE}
    missing = sorted(required.difference(schema))
    if missing:
        raise LeafFamilyContributionError(f"leaf assignments are missing required columns: {missing}")
    identity = pd.read_parquet(path, columns=[*IDENTITY, *SCOPE])
    return _identity_from_frame(
        identity,
        source="leaf assignments",
        expected_side=side,
        expected_head=head,
        expected_fold=fold,
    )


def _assignment_columns(path: Path) -> list[str]:
    _require_pyarrow()
    columns = [name for name in pq.ParquetFile(path).schema.names if str(name).startswith(TREE_PREFIX)]
    if not columns:
        raise LeafFamilyContributionError("leaf assignments have no selected tree columns")
    return sorted(map(str, columns))


def _iter_assignment_batches(
    path: Path,
    *,
    assignment_column: str,
    batch_rows: int,
) -> Iterator[pd.DataFrame]:
    """Read one assignment column at a time; never materialise the wide table."""

    _require_pyarrow()
    columns = [*IDENTITY, *SCOPE, assignment_column]
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=int(batch_rows), columns=columns):
        yield batch.to_pandas()


def _output_schema() -> Any:
    _require_pyarrow()
    return pa.schema([
        pa.field(INTERNAL_ROW, pa.int64()),
        pa.field("candidate_id", pa.string()),
        pa.field("__ts__", pa.timestamp("ns", tz="UTC")),
        pa.field("side_name", pa.string()),
        pa.field("fold_id", pa.string()),
        pa.field("head_name", pa.string()),
        pa.field("rule_signature", pa.string()),
        pa.field("contribution_direction", pa.string()),
        pa.field(VALUE_COLUMN, pa.float64()),
    ])


def _assert_token_free(frame: pd.DataFrame, *, source: str) -> None:
    bad = [
        column for column in frame.columns
        if any(token in str(column).lower() for token in FORBIDDEN_OUTPUT_TOKENS)
    ]
    if bad:
        raise LeafFamilyContributionError(f"raw local leaf identifiers leaked into {source}: {bad}")


def _materialize(
    artifact_dir: Path,
    *,
    output_path: Path,
    config: LeafFamilyContributionConfig,
) -> LeafFamilyContributionResult:
    config.validate()
    _require_pyarrow()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite family contribution table: {output_path}")
    if not artifact_dir.is_dir():
        raise LeafFamilyContributionError(f"strict artifact directory does not exist: {artifact_dir}")
    manifest = _read_manifest(artifact_dir / "base_reasoning_manifest.json")
    head, side, fold = _scope_from_manifest(manifest)
    assignment_path = artifact_dir / "leaf_assignments.parquet"
    catalog_path = artifact_dir / "leaf_rule_catalog.parquet"
    if not assignment_path.is_file() or not catalog_path.is_file():
        raise LeafFamilyContributionError("strict artifact lacks leaf assignments or leaf rule catalog")

    identity = _read_assignment_identity(assignment_path, head=head, side=side, fold=fold)
    assignment_columns = _assignment_columns(assignment_path)
    catalog = pd.read_parquet(catalog_path)
    lookup = _catalog_lookup(
        catalog,
        expected_head=head,
        expected_side=side,
        expected_fold=fold,
        manifest=manifest,
    )
    if set(assignment_columns) != set(lookup):
        missing_catalog = sorted(set(assignment_columns).difference(lookup))
        orphan_catalog = sorted(set(lookup).difference(assignment_columns))
        raise LeafFamilyContributionError(
            "leaf assignment/catalog tree scope differs "
            f"(missing_catalog={missing_catalog}, orphan_catalog={orphan_catalog})"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent))
    staged_output = temporary / output_path.name
    parts = temporary / "mapped_contribution_parts"
    parts.mkdir()
    # A bucket receives at most one mapped contribution per selected tree for
    # each candidate.  Size it from that worst-case long-table expansion, not
    # merely from candidate rows; otherwise a 64-tree artifact could turn a
    # nominal 250k-row bucket into a multi-million-row groupby.
    mapped_row_upper_bound = int(len(identity)) * int(len(assignment_columns))
    bucket_count = max(
        1,
        int(np.ceil(mapped_row_upper_bound / int(config.max_rows_per_output_bucket))),
    )
    bucket_paths = [parts / f"bucket_{bucket:04d}.parquet" for bucket in range(bucket_count)]
    writers: dict[int, Any] = {}
    expected_per_candidate = np.zeros(len(identity), dtype=np.float64)
    emitted_rows = 0
    final_writer: Any = None
    try:
        # Only this local same-artifact loop ever sees opaque tokens.  Each
        # assignment field is streamed independently, so a 64-tree artifact
        # does not materialise a 64-column frame.
        for assignment_column in assignment_columns:
            token_index, signatures, contributions, directions = lookup[assignment_column]
            row_start = 0
            for raw_batch in _iter_assignment_batches(
                assignment_path,
                assignment_column=assignment_column,
                batch_rows=int(config.assignment_batch_rows),
            ):
                batch_identity = _identity_from_frame(
                    raw_batch,
                    source=f"leaf assignments {assignment_column}",
                    expected_side=side,
                    expected_head=head,
                    expected_fold=fold,
                )
                expected_identity = identity.iloc[row_start : row_start + len(batch_identity)].reset_index(drop=True)
                if not _same_identity(batch_identity, expected_identity):
                    raise LeafFamilyContributionError(
                        f"assignment identity changed while reading {assignment_column}"
                    )
                raw_token = raw_batch[assignment_column]
                if raw_token.isna().any():
                    raise LeafFamilyContributionError(f"{assignment_column} contains a null local token")
                try:
                    token = pd.to_numeric(raw_token, errors="raise").to_numpy(dtype=np.uint64, copy=False)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise LeafFamilyContributionError(
                        f"{assignment_column} contains a non-unsigned local token"
                    ) from exc
                position = token_index.get_indexer(token)
                if (position < 0).any():
                    raise LeafFamilyContributionError(
                        f"{assignment_column} contains a token absent from its same-artifact catalog"
                    )
                mapped_contribution = contributions[position]
                expected_per_candidate[row_start : row_start + len(token)] += mapped_contribution
                nonzero = mapped_contribution != 0.0
                if nonzero.any():
                    candidate_rows = np.arange(row_start, row_start + len(token), dtype=np.int64)[nonzero]
                    mapped = pd.DataFrame({
                        INTERNAL_ROW: candidate_rows,
                        "candidate_id": batch_identity.loc[nonzero, "candidate_id"].astype("string").to_numpy(copy=False),
                        "__ts__": batch_identity.loc[nonzero, "__ts__"].to_numpy(copy=False),
                        "side_name": side,
                        "fold_id": fold,
                        "head_name": head,
                        "rule_signature": signatures[position][nonzero],
                        "contribution_direction": directions[position][nonzero],
                        VALUE_COLUMN: mapped_contribution[nonzero],
                    })
                    # Each candidate always goes to one bucket across all tree
                    # columns, allowing an exact candidate/family aggregation
                    # without a full artifact-wide temporary dataframe.
                    buckets = candidate_rows % bucket_count
                    for bucket in np.unique(buckets):
                        chunk = mapped.loc[buckets == bucket]
                        writer = writers.get(int(bucket))
                        if writer is None:
                            writer = pq.ParquetWriter(
                                bucket_paths[int(bucket)], _output_schema(), compression="zstd"
                            )
                            writers[int(bucket)] = writer
                        writer.write_table(pa.Table.from_pandas(chunk, schema=_output_schema(), preserve_index=False))
                row_start += len(batch_identity)
            if row_start != len(identity):
                raise LeafFamilyContributionError(
                    f"{assignment_column} row count differs from artifact identity"
                )

        for writer in writers.values():
            writer.close()
        writers.clear()

        for bucket, path in enumerate(bucket_paths):
            expected_positions = np.arange(bucket, len(identity), bucket_count, dtype=np.int64)
            if path.exists():
                work = pd.read_parquet(path)
                required = {INTERNAL_ROW, *OUTPUT_COLUMNS}
                missing = sorted(required.difference(work.columns))
                if missing:
                    raise LeafFamilyContributionError(f"temporary contribution rows are missing {missing}")
                grouped = (
                    work.groupby(
                        [
                            INTERNAL_ROW,
                            "candidate_id",
                            "__ts__",
                            "side_name",
                            "fold_id",
                            "head_name",
                            "rule_signature",
                            "contribution_direction",
                        ],
                        sort=False,
                        observed=True,
                        as_index=False,
                    )[VALUE_COLUMN]
                    .sum()
                )
                actual = (
                    grouped.groupby(INTERNAL_ROW, sort=False, observed=True)[VALUE_COLUMN]
                    .sum()
                    .reindex(expected_positions, fill_value=0.0)
                    .to_numpy(dtype=np.float64)
                )
            else:
                grouped = pd.DataFrame(columns=[INTERNAL_ROW, *OUTPUT_COLUMNS])
                actual = np.zeros(len(expected_positions), dtype=np.float64)
            if not np.allclose(
                actual,
                expected_per_candidate[expected_positions],
                rtol=float(config.additive_rtol),
                atol=float(config.additive_atol),
            ):
                raise LeafFamilyContributionError(
                    "candidate-level family aggregation fails to reconstruct same-artifact additive values"
                )
            if grouped.empty:
                continue
            emitted = grouped.loc[:, list(OUTPUT_COLUMNS)].copy()
            _assert_token_free(emitted, source="family contribution output")
            table = pa.Table.from_pandas(emitted, preserve_index=False)
            if final_writer is None:
                final_writer = pq.ParquetWriter(staged_output, table.schema, compression="zstd")
            final_writer.write_table(table)
            emitted_rows += len(emitted)

        if final_writer is not None:
            final_writer.close()
            final_writer = None
        else:
            empty = pd.DataFrame({
                "candidate_id": pd.Series(dtype="string"),
                "__ts__": pd.Series(dtype="datetime64[ns, UTC]"),
                "side_name": pd.Series(dtype="string"),
                "fold_id": pd.Series(dtype="string"),
                "head_name": pd.Series(dtype="string"),
                "rule_signature": pd.Series(dtype="string"),
                "contribution_direction": pd.Series(dtype="string"),
                VALUE_COLUMN: pd.Series(dtype="float64"),
            })
            empty.to_parquet(staged_output, index=False, compression="zstd")
        os.replace(staged_output, output_path)
        return LeafFamilyContributionResult(
            artifact_dir=artifact_dir,
            output_path=output_path,
            candidate_count=int(len(identity)),
            contribution_row_count=int(emitted_rows),
            assignment_column_count=int(len(assignment_columns)),
            family_contribution_total=float(expected_per_candidate.sum()),
        )
    except Exception:
        if final_writer is not None:
            final_writer.close()
        for writer in writers.values():
            writer.close()
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def materialize_leaf_family_contributions(
    artifact_dir: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    config: LeafFamilyContributionConfig = LeafFamilyContributionConfig(),
) -> LeafFamilyContributionResult:
    """Write one immutable, token-free family contribution parquet table.

    ``output_path`` is a single Parquet table rather than a directory because
    it is normally consumed as a bounded input by the causal leaf-health
    materialiser.  Existing outputs are never overwritten.
    """

    return _materialize(
        Path(artifact_dir),
        output_path=Path(output_path),
        config=config,
    )


def extract_leaf_family_contributions(
    artifact_dir: str | os.PathLike[str],
    *,
    config: LeafFamilyContributionConfig = LeafFamilyContributionConfig(),
) -> pd.DataFrame:
    """Return the token-free long table for a small/interactive strict artifact.

    The extraction itself remains bounded; the returned frame is necessarily
    proportional to the requested final long table.  Large production callers
    should use :func:`materialize_leaf_family_contributions` instead.
    """

    source = Path(artifact_dir)
    temporary = Path(tempfile.mkdtemp(prefix=".leaf_family_contribution_extract_"))
    try:
        result = _materialize(
            source,
            output_path=temporary / "family_contributions.parquet",
            config=config,
        )
        value = pd.read_parquet(result.output_path)
        _assert_token_free(value, source="returned family contribution output")
        return value.loc[:, list(OUTPUT_COLUMNS)].copy()
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


__all__ = [
    "LeafFamilyContributionConfig",
    "LeafFamilyContributionError",
    "LeafFamilyContributionResult",
    "VALUE_COLUMN",
    "extract_leaf_family_contributions",
    "materialize_leaf_family_contributions",
]
