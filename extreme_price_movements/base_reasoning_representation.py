"""Compact, causal G1/G2/G3 representation from strict-OOF reasoning artifacts.

This is deliberately an *artifact consumer*, never a model trainer.  It reads
the fold-local base-reasoning trees produced by ``strict_oof_base_reasoning``
and the already-resolved base-prediction shards.  Leaf tokens are used only
inside one artifact to look up its structural rule signature; they are never
compared across artifacts and are never written to an output.

The resulting rows remain side/head/direction local.  G1 retains only safe
aggregate leaf summaries (never local leaf identifiers); G2 is a compact
recurrent-rule-family summary; G3 is the existing train-fold contribution
bundle, scaled by the contemporaneous recurrent-family strength. Historical
outcome fields are updated only after ``label_available_ts`` is strictly less
than a later row's ``feature_generation_ts``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

import numpy as np
import pandas as pd

try:  # Imported lazily in production environments without parquet support.
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised by callers without pyarrow
    pq = None


SCHEMA = "base_reasoning_representation_v2"
HEAD_CLASS_MAP = {"p_adverse": 0, "p_weak": 1, "p_clear": 2}
IDENTITY = ("candidate_id", "__ts__", "side_name")
FORBIDDEN_OUTPUT_TOKENS = ("leaf_token", "leaf_id", "leaf_assignment", "raw_leaf")


class BaseReasoningRepresentationError(ValueError):
    """Raised when an input cannot prove the compact strict-OOF contract."""


@dataclass(frozen=True)
class BaseReasoningRepresentationConfig:
    """Fixed bounded controls; changing them does not fit or evaluate a model."""

    batch_rows: int = 50_000
    max_bundle_components: int = 16
    # Bounds only transient vector work.  Assignment parquet remains the
    # authoritative artifact; this prevents a high-tree ensemble from turning
    # a nominal row batch into a multi-gigabyte dense matrix.
    max_vectorized_elements: int = 2_000_000

    def validate(self) -> None:
        if int(self.batch_rows) <= 0:
            raise BaseReasoningRepresentationError("batch_rows must be positive")
        if not 1 <= int(self.max_bundle_components) <= 64:
            raise BaseReasoningRepresentationError("max_bundle_components must be in [1, 64]")
        if int(self.max_vectorized_elements) <= 0:
            raise BaseReasoningRepresentationError("max_vectorized_elements must be positive")


@dataclass(frozen=True)
class BaseReasoningRepresentationResult:
    artifact_dir: Path
    row_count: int
    signature_count: int
    manifest: dict[str, Any]


def _utc(values: pd.Series, *, name: str) -> pd.Series:
    value = pd.to_datetime(values, utc=True, errors="coerce")
    if value.isna().any():
        raise BaseReasoningRepresentationError(f"{name} contains an invalid timestamp")
    return value


def _require_columns(frame: pd.DataFrame, columns: set[str], *, source: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise BaseReasoningRepresentationError(f"{source} is missing required columns: {missing}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaseReasoningRepresentationError(f"invalid strict reasoning manifest: {path}") from exc


def _validate_index(index_path: Path) -> pd.DataFrame:
    if not index_path.is_file():
        raise BaseReasoningRepresentationError(f"artifact index does not exist: {index_path}")
    index = pd.read_parquet(index_path)
    required = {"transport", "side_model", "head_name", "class_index", "fold_name", "fold_id", "artifact_dir", "strict_status"}
    _require_columns(index, required, source="artifact index")
    if index.empty:
        raise BaseReasoningRepresentationError("artifact index is empty")
    index = index.copy()
    index["side_model"] = index["side_model"].astype(str).str.lower()
    index["head_name"] = index["head_name"].astype(str)
    index["fold_name"] = index["fold_name"].astype(str)
    index["fold_id"] = index["fold_id"].astype(str)
    if not set(index.side_model).issubset({"long", "short"}):
        raise BaseReasoningRepresentationError("artifact index contains a non-long/short side")
    if not index.strict_status.astype(str).eq("MATERIALIZED_STRICT_OOF").all():
        raise BaseReasoningRepresentationError("artifact index contains a non-strict-OOF artifact")
    if index.duplicated(["transport", "side_model", "fold_id", "head_name"]).any():
        raise BaseReasoningRepresentationError("artifact index duplicates a side/fold/head artifact")
    for transport, cell in index.groupby("transport", observed=True, sort=True):
        sides = set(cell.side_model)
        if sides != {"long", "short"}:
            raise BaseReasoningRepresentationError(f"partial transport {transport!r}: expected long and short")
        expected_all_folds: set[str] | None = None
        for side, side_cell in cell.groupby("side_model", observed=True, sort=True):
            all_folds = set(side_cell.fold_id)
            if expected_all_folds is None:
                expected_all_folds = all_folds
            elif all_folds != expected_all_folds:
                raise BaseReasoningRepresentationError(f"partial transport {transport!r}: sides have different folds")
            for fold_id, fold_cell in side_cell.groupby("fold_id", observed=True, sort=True):
                got = dict(zip(fold_cell.head_name, fold_cell.class_index, strict=True))
                if got != HEAD_CLASS_MAP:
                    raise BaseReasoningRepresentationError(
                        f"partial transport {transport!r}/{side}/{fold_id}: expected all semantic heads"
                    )
        strict = cell.loc[~cell.fold_name.eq("outer")]
        if strict.empty:
            raise BaseReasoningRepresentationError(f"transport {transport!r} has no strict OOF folds")
        expected_folds: set[str] | None = None
        for side, side_cell in strict.groupby("side_model", observed=True, sort=True):
            folds = set(side_cell.fold_id)
            if expected_folds is None:
                expected_folds = folds
            elif folds != expected_folds:
                raise BaseReasoningRepresentationError(f"partial transport {transport!r}: sides have different strict folds")
        outer = cell.loc[cell.fold_name.eq("outer")]
        if outer.empty:
            raise BaseReasoningRepresentationError(f"transport {transport!r} has no outer base-OOF fold")
        expected_outer_folds: set[str] | None = None
        for side, side_cell in outer.groupby("side_model", observed=True, sort=True):
            folds = set(side_cell.fold_id)
            if expected_outer_folds is None:
                expected_outer_folds = folds
            elif folds != expected_outer_folds:
                raise BaseReasoningRepresentationError(f"partial transport {transport!r}: sides have different outer folds")
            if len(folds) != 1:
                raise BaseReasoningRepresentationError(f"transport {transport!r}/{side} must contain exactly one outer fold")
    return index


def _prediction_rows(path: Path, *, fold_id: str, side: str, batch_rows: int) -> pd.DataFrame:
    required = ["candidate_id", "decision_ts", "label_available_ts", "side_name", "fold_id", "net_bps", "feature_generation_ts"]
    if not path.is_file():
        raise BaseReasoningRepresentationError(f"missing strict-OOF prediction shard: {path}")
    if pq is None:
        value = pd.read_parquet(path, columns=required)
        pieces = [value]
    else:
        names = set(pq.ParquetFile(path).schema.names)
        missing = sorted(set(required).difference(names))
        if missing:
            raise BaseReasoningRepresentationError(f"prediction shard is missing {missing}: {path}")
        pieces = []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=int(batch_rows), columns=required):
            item = batch.to_pandas()
            item = item.loc[item["fold_id"].astype(str).eq(str(fold_id))]
            if not item.empty:
                pieces.append(item)
    if not pieces:
        raise BaseReasoningRepresentationError(f"prediction shard has no rows for fold {fold_id!r}")
    out = pd.concat(pieces, ignore_index=True)
    out["side_name"] = out["side_name"].astype(str).str.lower()
    if not out.side_name.eq(side).all():
        raise BaseReasoningRepresentationError("prediction shard crosses sides for a side-local fold")
    out["__ts__"] = _utc(out["decision_ts"], name="prediction decision_ts")
    out["feature_generation_ts"] = _utc(out["feature_generation_ts"], name="feature_generation_ts")
    out["label_available_ts"] = _utc(out["label_available_ts"], name="label_available_ts")
    if out.duplicated(list(IDENTITY)).any():
        raise BaseReasoningRepresentationError("prediction shard has duplicate candidate identities")
    if not out["feature_generation_ts"].is_monotonic_increasing:
        raise BaseReasoningRepresentationError("prediction shard must be ordered by feature_generation_ts; rows cannot be reordered independently of strict artifacts")
    if out["label_available_ts"].lt(out["feature_generation_ts"]).any():
        raise BaseReasoningRepresentationError("a prediction outcome resolves before its feature generation timestamp")
    return out


def _identity(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    _require_columns(frame, set(IDENTITY), source=source)
    out = frame.loc[:, list(IDENTITY)].copy()
    out["candidate_id"] = out["candidate_id"].astype(str)
    out["side_name"] = out["side_name"].astype(str).str.lower()
    out["__ts__"] = _utc(out["__ts__"], name=f"{source}.__ts__")
    if out.duplicated(list(IDENTITY)).any() or out.isna().any().any():
        raise BaseReasoningRepresentationError(f"{source} has duplicate or null identities")
    return out


def _assert_same_identity(left: pd.DataFrame, right: pd.DataFrame, *, source: str) -> None:
    if len(left) != len(right) or not left.loc[:, list(IDENTITY)].equals(right.loc[:, list(IDENTITY)]):
        raise BaseReasoningRepresentationError(f"{source} identity does not exactly match prediction shard")


def _catalog_lookup(catalog: pd.DataFrame, *, head: str, side: str, fold: str) -> tuple[dict[str, dict[int, tuple[str, float]]], pd.DataFrame]:
    required = {"head_name", "side_name", "fold_id", "model_slot", "head_tree_slot", "leaf_token", "rule_signature", "train_leaf_frequency", "path_depth", "unique_feature_count"}
    _require_columns(catalog, required, source="leaf rule catalog")
    work = catalog.copy()
    if not work.head_name.astype(str).eq(head).all() or not work.side_name.astype(str).str.lower().eq(side).all() or not work.fold_id.astype(str).eq(fold).all():
        raise BaseReasoningRepresentationError("leaf catalog side/head/fold disagrees with artifact index")
    if work.rule_signature.isna().any() or work.rule_signature.astype(str).str.strip().eq("").any():
        raise BaseReasoningRepresentationError("leaf catalog has a blank rule signature")
    work["tree_column"] = [f"leaf_assignment__model_{int(m):02d}_head_tree_{int(t):03d}" for m, t in zip(work.model_slot, work.head_tree_slot, strict=True)]
    if work.duplicated(["tree_column", "leaf_token"]).any():
        raise BaseReasoningRepresentationError("leaf catalog duplicates an opaque local token within a tree")
    lookups: dict[str, dict[int, tuple[str, float]]] = {}
    for column, cell in work.groupby("tree_column", observed=True, sort=True):
        lookups[str(column)] = {
            int(token): (str(signature), float(support))
            for token, signature, support in zip(cell.leaf_token, cell.rule_signature, cell.train_leaf_frequency, strict=True)
        }
    safe = work.loc[:, ["rule_signature", "train_leaf_frequency", "path_depth", "unique_feature_count"]].copy()
    safe["train_leaf_frequency"] = pd.to_numeric(safe["train_leaf_frequency"], errors="coerce").fillna(0.0).astype(np.float32)
    return lookups, safe


@dataclass(frozen=True)
class _VectorizedTreePlan:
    """One local tree's opaque-token lookup, held only while its fold runs.

    The arrays are deliberately local to one side/head/fold artifact.  A
    ``signature_code`` is an in-process index for causal aggregation, never a
    persisted model feature or cross-model leaf alignment key.
    """

    column: str
    tokens: np.ndarray
    signature_codes: np.ndarray
    supports: np.ndarray
    weights: np.ndarray


def _vectorized_tree_plans(
    lookup: dict[str, dict[int, tuple[str, float]]],
    *,
    catalog_folds: dict[str, set[str]],
    signature_codes: dict[str, int],
) -> list[_VectorizedTreePlan]:
    """Compile local token dictionaries into NumPy-searchable tree plans."""
    plans: list[_VectorizedTreePlan] = []
    for column in sorted(lookup):
        entries = sorted(lookup[column].items(), key=lambda item: item[0])
        try:
            tokens = np.asarray([token for token, _ in entries], dtype=np.uint64)
        except (OverflowError, TypeError, ValueError) as exc:
            raise BaseReasoningRepresentationError("leaf catalog contains a non-numeric local token") from exc
        signatures = [value[0] for _, value in entries]
        supports = np.asarray([value[1] for _, value in entries], dtype=np.float32)
        recurrence = np.asarray([len(catalog_folds[signature]) for signature in signatures], dtype=np.float64)
        # This matches the previous per-tree calculation before its float32
        # storage conversion.  Negative catalog support remains zero weight.
        weights = (np.maximum(supports.astype(np.float64), 0.0) * np.log1p(recurrence)).astype(np.float32)
        plans.append(_VectorizedTreePlan(
            column=str(column),
            tokens=tokens,
            signature_codes=np.asarray([signature_codes[signature] for signature in signatures], dtype=np.int32),
            supports=supports,
            weights=weights,
        ))
    return plans


def _map_assignment_positions(plan: _VectorizedTreePlan, values: np.ndarray) -> np.ndarray:
    """Vectorized local-token lookup with the old fail-closed semantics."""
    try:
        tokens = np.asarray(values, dtype=np.uint64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise BaseReasoningRepresentationError("local leaf assignment has no local catalog rule") from exc
    positions = np.searchsorted(plan.tokens, tokens)
    in_bounds = positions < len(plan.tokens)
    matches = np.zeros(len(tokens), dtype=bool)
    matches[in_bounds] = plan.tokens[positions[in_bounds]] == tokens[in_bounds]
    if not matches.all():
        raise BaseReasoningRepresentationError("local leaf assignment has no local catalog rule")
    return positions.astype(np.intp, copy=False)


def _effective_batch_rows(config: BaseReasoningRepresentationConfig, tree_count: int) -> int:
    """Cap local opaque-token matrices without weakening the configured bound."""
    return max(1, min(int(config.batch_rows), int(config.max_vectorized_elements) // max(int(tree_count), 1)))


def _schedule_pending_updates(
    *,
    due_heap: list[int],
    pending_updates: dict[int, dict[tuple[int, int], list[float]]],
    due_ns: np.ndarray,
    signature_codes: np.ndarray,
    direction_codes: np.ndarray,
    outcomes: np.ndarray,
) -> None:
    """Associatively coalesce same-release history updates.

    The old implementation pushed one heap item per row/tree occurrence.  For
    a fixed release timestamp, signature and direction, count/sum updates are
    associative and no intermediate value is observed.  Coalescing therefore
    preserves the strict-prior result while keeping the pending heap bounded by
    distinct release times.
    """
    if not len(due_ns):
        return
    flat_codes = signature_codes.reshape(-1)
    repeated_due = np.repeat(due_ns, signature_codes.shape[1]).astype(np.int64, copy=False)
    repeated_direction = np.repeat(direction_codes, signature_codes.shape[1]).astype(np.int8, copy=False)
    repeated_outcome = np.repeat(outcomes, signature_codes.shape[1]).astype(np.float64, copy=False)
    order = np.lexsort((flat_codes, repeated_direction, repeated_due))
    due_sorted = repeated_due[order]
    direction_sorted = repeated_direction[order]
    code_sorted = flat_codes[order]
    outcome_sorted = repeated_outcome[order]
    starts = np.r_[True, (due_sorted[1:] != due_sorted[:-1]) | (direction_sorted[1:] != direction_sorted[:-1]) | (code_sorted[1:] != code_sorted[:-1])]
    start_idx = np.flatnonzero(starts)
    counts = np.diff(np.r_[start_idx, len(order)]).astype(np.float64)
    totals = np.add.reduceat(outcome_sorted, start_idx)
    for index, count, total in zip(start_idx, counts, totals, strict=True):
        due = int(due_sorted[index])
        key = (int(code_sorted[index]), int(direction_sorted[index]))
        bucket = pending_updates.get(due)
        if bucket is None:
            bucket = {}
            pending_updates[due] = bucket
            heapq.heappush(due_heap, due)
        value = bucket.setdefault(key, [0.0, 0.0])
        value[0] += float(count)
        value[1] += float(total)


def _direction(balance: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(balance, dtype=np.float32) >= 0.0, "positive", "negative")


def _forbid_raw_leaf_columns(frame: pd.DataFrame, *, source: str) -> None:
    # G1's `leaf_assignment_count` is an aggregate count, not a local leaf
    # identifier.  It remains safe after head/direction qualification.  All
    # opaque assignment vectors/tokens stay forbidden.
    safe_aggregate = "base_reasoning__g1_leaf_assignment_count"
    bad = [
        name for name in frame.columns
        if not str(name).lower().startswith(safe_aggregate)
        and any(token in str(name).lower() for token in FORBIDDEN_OUTPUT_TOKENS)
    ]
    if bad:
        raise BaseReasoningRepresentationError(f"raw leaf identifiers leaked into {source}: {bad}")


def _new_writer(path: Path, frame: pd.DataFrame):
    if pq is None:
        return None
    import pyarrow as pa
    return pq.ParquetWriter(path, pa.Table.from_pandas(frame, preserve_index=False).schema, compression="zstd")


def _write_frame(writer: Any, path: Path, frame: pd.DataFrame) -> Any:
    if writer is None and pq is None:
        # This branch is only used on environments that can read but not stream
        # parquet; tests and production have pyarrow.
        if path.exists():
            prior = pd.read_parquet(path)
            pd.concat([prior, frame], ignore_index=True).to_parquet(path, index=False, compression="zstd")
        else:
            frame.to_parquet(path, index=False, compression="zstd")
        return None
    import pyarrow as pa
    if writer is None:
        writer = _new_writer(path, frame)
    writer.write_table(pa.Table.from_pandas(frame, preserve_index=False))
    return writer


def build_base_reasoning_representation(
    artifact_index: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    prediction_shards_root: str | os.PathLike[str] | None = None,
    config: BaseReasoningRepresentationConfig = BaseReasoningRepresentationConfig(),
) -> BaseReasoningRepresentationResult:
    """Build immutable compact G2/G3 features from a complete strict-OOF transport.

    ``destination`` must not exist.  Outcome summaries are causal streaming
    statistics: an event enters a signature history only after its declared
    label availability is strictly before the next feature-generation time.
    """
    config.validate()
    index_path = Path(artifact_index)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite compact reasoning artifact {destination}")
    index = _validate_index(index_path)
    root = index_path.parent
    prediction_root = Path(prediction_shards_root) if prediction_shards_root is not None else root / "base_prediction_shards"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    bundle_path, feature_path, signature_path = (temporary / name for name in (
        "contribution_bundle_features_oof.parquet", "base_reasoning_features_oof.parquet", "leaf_rule_signatures.parquet"))
    writers: dict[str, Any] = {"bundle": None, "features": None, "signatures": None}
    rows_written = 0
    signature_count = 0
    transport_counts: dict[str, int] = {}
    partition_counts: dict[str, int] = {"inner_oof": 0, "outer_test": 0}
    try:
        for transport, transport_index in index.groupby("transport", observed=True, sort=True):
            transport_counts[str(transport)] = 0
            # Side/head streams are independent, preventing accidental pooling.
            for side in ("long", "short"):
                side_index = transport_index.loc[transport_index.side_model.eq(side)].copy()
                strict_index = side_index.loc[~side_index.fold_name.eq("outer")]
                strict_shard_path = prediction_root / str(transport) / side / "strict_oof_predictions.parquet"
                outer_shard_path = prediction_root / str(transport) / side / "outer_predictions.parquet"
                expected_folds = set(strict_index.fold_id.astype(str))
                shard_folds = set(pd.read_parquet(strict_shard_path, columns=["fold_id"])["fold_id"].astype(str)) if strict_shard_path.exists() else set()
                if shard_folds != expected_folds:
                    raise BaseReasoningRepresentationError(
                        f"partial transport {transport!r}/{side}: strict prediction folds do not exactly match artifact index"
                    )
                expected_outer_folds = set(side_index.loc[side_index.fold_name.eq("outer"), "fold_id"].astype(str))
                outer_folds = set(pd.read_parquet(outer_shard_path, columns=["fold_id"])["fold_id"].astype(str)) if outer_shard_path.exists() else set()
                if outer_folds != expected_outer_folds:
                    raise BaseReasoningRepresentationError(
                        f"partial transport {transport!r}/{side}: outer prediction folds do not exactly match artifact index"
                    )
                for head, class_index in HEAD_CLASS_MAP.items():
                    stream = side_index.loc[side_index.head_name.eq(head)].sort_values("eval_start_utc", kind="stable")
                    history_counts = np.zeros((2, 0), dtype=np.float64)
                    history_totals = np.zeros((2, 0), dtype=np.float64)
                    pending_due: list[int] = []
                    pending_updates: dict[int, dict[tuple[int, int], list[float]]] = {}
                    catalog_folds: dict[str, set[str]] = {}
                    catalog_support: dict[str, list[float]] = {}
                    signature_codes: dict[str, int] = {}
                    for _, item in stream.iterrows():
                        fold = str(item.fold_id)
                        partition = "outer_test" if str(item.fold_name) == "outer" else "inner_oof"
                        artifact = root / str(item.artifact_dir)
                        manifest = _read_manifest(artifact / "base_reasoning_manifest.json")
                        if manifest.get("status") != "MATERIALIZED_STRICT_OOF" or str(manifest.get("head_name")) != head or str(manifest.get("side_name", "")).lower() != side or str(manifest.get("fold_id")) != fold:
                            raise BaseReasoningRepresentationError(f"artifact manifest lineage mismatch: {artifact}")
                        catalog = pd.read_parquet(artifact / "leaf_rule_catalog.parquet")
                        lookup, safe_catalog = _catalog_lookup(catalog, head=head, side=side, fold=fold)
                        # Catalog information was fitted before this fold's evaluation.
                        for signature, cell in safe_catalog.groupby("rule_signature", observed=True):
                            catalog_folds.setdefault(str(signature), set()).add(fold)
                            catalog_support.setdefault(str(signature), []).extend(cell.train_leaf_frequency.astype(float).tolist())
                        new_signatures = sorted(set(catalog_folds).difference(signature_codes))
                        if new_signatures:
                            first = len(signature_codes)
                            signature_codes.update({signature: first + offset for offset, signature in enumerate(new_signatures)})
                            history_counts = np.pad(history_counts, ((0, 0), (0, len(new_signatures))))
                            history_totals = np.pad(history_totals, ((0, 0), (0, len(new_signatures))))
                        plans = _vectorized_tree_plans(
                            lookup, catalog_folds=catalog_folds, signature_codes=signature_codes,
                        )
                        assignment = pd.read_parquet(artifact / "leaf_assignments.parquet")
                        bundle = pd.read_parquet(artifact / "contribution_bundle.parquet")
                        feature_path_source = artifact / "base_reasoning_features.parquet"
                        if pq is None:
                            available_feature_columns = set(pd.read_parquet(feature_path_source).columns)
                        else:
                            available_feature_columns = set(pq.ParquetFile(feature_path_source).schema.names)
                        g1_columns = sorted(
                            name for name in available_feature_columns
                            if str(name).startswith("base_reasoning__g1_")
                        )
                        if not g1_columns:
                            raise BaseReasoningRepresentationError("base reasoning features have no safe G1 aggregate fields")
                        features = pd.read_parquet(
                            feature_path_source,
                            columns=[*IDENTITY, "head_name", "fold_id", "base_reasoning__g3_balance", *g1_columns],
                        )
                        assignment_identity = _identity(assignment, source="leaf assignments")
                        bundle_identity = _identity(bundle, source="contribution bundle")
                        feature_identity = _identity(features, source="base reasoning features")
                        prediction = _prediction_rows(
                            outer_shard_path if partition == "outer_test" else strict_shard_path,
                            fold_id=fold, side=side, batch_rows=config.batch_rows,
                        )
                        prediction_identity = _identity(prediction, source="prediction shard")
                        for other, name in ((assignment_identity, "leaf assignments"), (bundle_identity, "contribution bundle"), (feature_identity, "base reasoning features")):
                            _assert_same_identity(other, prediction_identity, source=name)
                        if not plans or any(plan.column not in assignment for plan in plans):
                            raise BaseReasoningRepresentationError("leaf assignments do not cover the indexed local trees")
                        svd_columns = sorted(name for name in bundle if name.startswith("base_reasoning__g3_contribution_svd_"))[: int(config.max_bundle_components)]
                        if not svd_columns:
                            raise BaseReasoningRepresentationError("contribution bundle has no G3 SVD fields")
                        if not features.head_name.astype(str).eq(head).all() or not features.fold_id.astype(str).eq(fold).all():
                            raise BaseReasoningRepresentationError("feature sidecar head/fold mismatch")
                        balance = pd.to_numeric(features["base_reasoning__g3_balance"], errors="coerce").fillna(0.0).to_numpy(np.float32)
                        directions = _direction(balance)
                        g1_values = {
                            column: pd.to_numeric(features[column], errors="coerce").fillna(0.0).to_numpy(np.float32)
                            for column in g1_columns
                        }
                        # Fold-start signature audit is safe: it contains only previous resolved outcomes.
                        audit_rows: list[dict[str, Any]] = []
                        for signature in sorted(catalog_folds):
                            recurrence = len(catalog_folds[signature])
                            support = float(np.mean(catalog_support[signature]))
                            for direction in ("positive", "negative"):
                                direction_code = 0 if direction == "positive" else 1
                                code = signature_codes[signature]
                                count = history_counts[direction_code, code]
                                total = history_totals[direction_code, code]
                                audit_rows.append({"transport": str(transport), "side_name": side, "head_name": head, "fold_id": fold, "meta_partition": partition, "contribution_direction": direction, "rule_signature": signature, "recurrent_fold_count": np.int16(recurrence), "mean_train_leaf_frequency": np.float32(support), "prior_resolved_outcome_count": np.int32(count), "prior_resolved_outcome_mean": np.float32(total / count if count else 0.0)})
                        audit = pd.DataFrame(audit_rows)
                        _forbid_raw_leaf_columns(audit, source="leaf_rule_signatures")
                        writers["signatures"] = _write_frame(writers["signatures"], signature_path, audit)
                        signature_count += len(audit)
                        # Work by bounded row blocks; opaque-token mapping is vectorized
                        # once per tree/block instead of repeating DataFrame scalar access
                        # for every row and tree.
                        tree_count = len(plans)
                        work_batch_rows = _effective_batch_rows(config, tree_count)
                        outcome_values = pd.to_numeric(prediction.net_bps, errors="coerce").to_numpy(np.float64)
                        available_ns = prediction.label_available_ts.astype("int64", copy=False).to_numpy(np.int64)
                        feature_ns = prediction.feature_generation_ts.astype("int64", copy=False).to_numpy(np.int64)
                        positive_direction = balance >= 0.0
                        for start in range(0, len(prediction), work_batch_rows):
                            stop = min(start + work_batch_rows, len(prediction))
                            block_len = stop - start
                            positions = np.empty((block_len, tree_count), dtype=np.uint32)
                            for tree_index, plan in enumerate(plans):
                                positions[:, tree_index] = _map_assignment_positions(
                                    plan, assignment[plan.column].iloc[start:stop].to_numpy(copy=False),
                                )
                            # Flush strictly prior resolved outcomes before this timestamp group.
                            # Processing a full timestamp together prevents same-time feedback.
                            timestamps = feature_ns[start:stop]
                            weights = np.zeros(block_len, dtype=np.float32)
                            recurrent_counts = np.zeros(block_len, dtype=np.float32)
                            support_mean = np.zeros(block_len, dtype=np.float32)
                            prior_mean = np.zeros(block_len, dtype=np.float32)
                            prior_log_support = np.zeros(block_len, dtype=np.float32)
                            for current_ts in pd.unique(timestamps):
                                now = int(current_ts)
                                while pending_due and pending_due[0] < now:
                                    due = heapq.heappop(pending_due)
                                    for (code, direction_code), (count, total) in pending_updates.pop(due).items():
                                        history_counts[direction_code, code] += count
                                        history_totals[direction_code, code] += total
                                local = np.flatnonzero(timestamps == current_ts)
                                local_positions = positions[local]
                                local_positive = positive_direction[start + local]
                                local_codes = np.empty((len(local), tree_count), dtype=np.int32)
                                denominator = np.zeros(len(local), dtype=np.float32)
                                weighted_support = np.zeros(len(local), dtype=np.float32)
                                weighted_prior_mean = np.zeros(len(local), dtype=np.float32)
                                weighted_prior_log = np.zeros(len(local), dtype=np.float32)
                                for tree_index, plan in enumerate(plans):
                                    mapped = local_positions[:, tree_index]
                                    codes = plan.signature_codes[mapped]
                                    tree_weight = plan.weights[mapped]
                                    tree_support = plan.supports[mapped]
                                    local_codes[:, tree_index] = codes
                                    counts = np.where(local_positive, history_counts[0, codes], history_counts[1, codes])
                                    totals = np.where(local_positive, history_totals[0, codes], history_totals[1, codes])
                                    denominator += tree_weight
                                    weighted_support += tree_weight * tree_support
                                    weighted_prior_mean += tree_weight * np.divide(
                                        totals, counts, out=np.zeros_like(totals), where=counts > 0.0,
                                    ).astype(np.float32)
                                    weighted_prior_log += tree_weight * np.log1p(counts).astype(np.float32)
                                nonzero = denominator > 0.0
                                local_weight = np.divide(denominator, np.float32(tree_count), out=np.zeros_like(denominator), where=nonzero)
                                local_support = np.divide(weighted_support, denominator, out=np.zeros_like(denominator), where=nonzero)
                                local_prior_mean = np.divide(weighted_prior_mean, denominator, out=np.zeros_like(denominator), where=nonzero)
                                local_prior_log = np.divide(weighted_prior_log, denominator, out=np.zeros_like(denominator), where=nonzero)
                                sorted_codes = np.sort(local_codes, axis=1)
                                local_recurrent = 1 + np.count_nonzero(sorted_codes[:, 1:] != sorted_codes[:, :-1], axis=1)
                                weights[local] = local_weight
                                support_mean[local] = local_support
                                prior_mean[local] = local_prior_mean
                                prior_log_support[local] = local_prior_log
                                recurrent_counts[local] = local_recurrent.astype(np.float32)
                                global_local = start + local
                                finite = np.isfinite(outcome_values[global_local])
                                if finite.any():
                                    if (available_ns[global_local][finite] < int(current_ts)).any():
                                        raise BaseReasoningRepresentationError("outcome label is available before its feature generation timestamp")
                                    _schedule_pending_updates(
                                        due_heap=pending_due,
                                        pending_updates=pending_updates,
                                        due_ns=available_ns[global_local][finite],
                                        signature_codes=local_codes[finite],
                                        direction_codes=np.where(local_positive[finite], 0, 1),
                                        outcomes=outcome_values[global_local][finite],
                                    )
                            identity = prediction_identity.iloc[start:stop].reset_index(drop=True)
                            shared = identity.copy()
                            # Transport is row-level provenance, not merely a
                            # manifest attribute: downstream outer-meta joins
                            # must prove that no A/B reasoning row is crossed.
                            shared["transport"] = str(transport)
                            shared["head_name"] = head; shared["fold_id"] = fold; shared["meta_partition"] = partition; shared["contribution_direction"] = directions[start:stop]
                            g2 = pd.DataFrame({"base_reasoning__g2_recurrent_family_weight": weights, "base_reasoning__g2_recurrent_family_count": recurrent_counts, "base_reasoning__g2_recurrent_family_train_support": support_mean, "base_reasoning__g2_recurrent_family_prior_outcome_mean": prior_mean, "base_reasoning__g2_recurrent_family_prior_support_log1p": prior_log_support})
                            g1 = pd.DataFrame({column: value[start:stop] for column, value in g1_values.items()})
                            bundle_frame = shared.copy()
                            for column in svd_columns:
                                suffix = column.rsplit("_", 1)[-1]
                                raw = pd.to_numeric(bundle[column].iloc[start:stop], errors="coerce").fillna(0.0).to_numpy(np.float32)
                                bundle_frame[f"base_reasoning__g3_contribution_bundle_weighted_svd_{suffix}"] = (raw * weights).astype(np.float32)
                            output = pd.concat([shared, g1, g2, bundle_frame.drop(columns=list(shared.columns))], axis=1)
                            _forbid_raw_leaf_columns(bundle_frame, source="contribution bundle features")
                            _forbid_raw_leaf_columns(output, source="base reasoning features")
                            writers["bundle"] = _write_frame(writers["bundle"], bundle_path, bundle_frame)
                            writers["features"] = _write_frame(writers["features"], feature_path, output)
                            rows_written += len(output); transport_counts[str(transport)] += len(output); partition_counts[partition] += len(output)
        for writer in writers.values():
            if writer is not None:
                writer.close()
        if rows_written == 0 or signature_count == 0:
            raise BaseReasoningRepresentationError("no compact strict-OOF features were produced")
        outputs = {path.name: _sha256_file(path) for path in (bundle_path, feature_path, signature_path)}
        manifest = {"schema": SCHEMA, "status": "COMPACT_STRICT_OOF_BASE_REASONING_MATERIALIZED", "config": asdict(config), "contract": {"no_training_or_evaluation": True, "strict_transport_completeness": "both sides, every strict inner fold, one outer base-OOF fold, and p_adverse/p_weak/p_clear are required", "meta_partitions": "inner_oof rows are meta-training candidates; outer_test rows are held-out candidates and are explicitly never fed back into an earlier inner row", "leaf_alignment": "opaque local leaf tokens are used only for same-artifact catalog lookup and are never persisted or aligned", "families": "G1 safe per-head/direction leaf aggregates; G2 recurrent structural rule signatures; G3 train-fold contribution bundle only", "separation": "long/short, semantic head, and positive/negative contribution direction remain separate", "historical_outcomes": "only label_available_ts < feature_generation_ts enters a signature history", "numeric_storage": "feature values are float32 and parquet is written in bounded batches"}, "inputs": {"artifact_index": str(index_path), "prediction_shards_root": str(prediction_root), "transport_count": int(index.transport.nunique())}, "counts": {"feature_rows": int(rows_written), "signature_rows": int(signature_count), "rows_by_transport": transport_counts, "rows_by_meta_partition": partition_counts}, "outputs": outputs}
        (temporary / "base_reasoning_representation_manifest.json").write_text(json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        return BaseReasoningRepresentationResult(destination, rows_written, signature_count, manifest)
    except Exception:
        for writer in writers.values():
            if writer is not None:
                writer.close()
        for path in temporary.glob("*"):
            path.unlink(missing_ok=True)
        temporary.rmdir()
        raise
