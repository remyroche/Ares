"""Immutable, target-neutral Stage-I selector caches.

The expensive selector is repeated for the R3, scalar-S and ordinal-O target
families.  Feature availability, the numeric matrix, feature/feature Spearman
geometry and the chronological row ordering do not depend on those targets.
This module materialises those objects once and binds them to the exact row and
feature contracts.  Target labels, univariate scores, Relief hit/miss scores,
models and permutation scores are deliberately absent: they must be recomputed
for every target.

The cache is fail-closed.  It is not a best-effort accelerator: a requested
cache whose lineage differs raises instead of silently returning stale data.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd


SCHEMA = "stage_i_target_neutral_selector_cache_v1"
IDENTITY_COLUMNS = ("candidate_id", "__ts__", "__symbol__", "decision_ts")


def _fsync_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def atomic_cache_staging(root: str | Path, *, timeout_seconds: float = 30.0):
    """Publish a complete immutable cache directory under an exclusive lock."""
    root = Path(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = root.parent / f".{root.name}.publish.lock"
    deadline = time.monotonic() + float(timeout_seconds)
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.write(fd, f"pid={os.getpid()}\n".encode())
            os.fsync(fd)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"cache publisher lock timeout: {lock}")
            time.sleep(0.05)
    staging = root.parent / f".{root.name}.staging.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        if root.exists():
            yield None
            return
        staging.mkdir()
        yield staging
        _fsync_tree(staging)
        os.replace(staging, root)
        parent_fd = os.open(root.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if fd is not None:
            os.close(fd)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def array_sha256(value: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    header = canonical_sha256({"shape": list(arr.shape), "dtype": str(arr.dtype)})
    return sha256(header.encode("ascii") + arr.view(np.uint8).tobytes()).hexdigest()


def frame_identity_sha256(frame: pd.DataFrame) -> str:
    missing = [column for column in IDENTITY_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"target-neutral cache identity lacks columns: {missing}")
    work = frame.loc[:, list(IDENTITY_COLUMNS)].copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    work["decision_ts"] = pd.to_datetime(
        work["decision_ts"], utc=True, errors="raise"
    )
    if work["candidate_id"].isna().any() or work["candidate_id"].duplicated().any():
        raise ValueError("target-neutral cache requires unique finite candidate_id")
    return sha256(
        pd.util.hash_pandas_object(work, index=False)
        .to_numpy(dtype=np.uint64)
        .tobytes()
    ).hexdigest()


def _feature_matrix(frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    names = list(map(str, features))
    if len(names) != len(set(names)):
        raise ValueError("target-neutral cache feature names must be unique")
    missing = [feature for feature in names if feature not in frame]
    if missing:
        raise ValueError(f"target-neutral cache matrix lacks features: {missing[:20]}")
    return frame.loc[:, names].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float32, copy=True
    )


def _coverage_audit(matrix: np.ndarray, features: Sequence[str]) -> pd.DataFrame:
    finite = np.isfinite(matrix)
    rows = max(1, len(matrix))
    finite_rows = finite.sum(axis=0).astype(np.int64)
    nonconstant = np.zeros(matrix.shape[1], dtype=bool)
    variance = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    for index in range(matrix.shape[1]):
        values = matrix[finite[:, index], index]
        if len(values):
            variance[index] = float(np.var(values.astype(np.float64)))
            nonconstant[index] = bool(len(values) >= 2 and np.ptp(values) > 0.0)
    return pd.DataFrame(
        {
            "feature": list(map(str, features)),
            "finite_rows": finite_rows,
            "finite_rate": finite_rows / float(rows),
            "nonconstant": nonconstant,
            "variance": variance,
        }
    )


def _spearman_edges(
    matrix: np.ndarray,
    features: Sequence[str],
    *,
    row_ids: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    if not 0.0 < float(threshold) <= 1.0:
        raise ValueError("Spearman threshold must be in (0,1]")
    sample = matrix[np.asarray(row_ids, dtype=np.int64)]
    ranked = pd.DataFrame(sample, columns=list(map(str, features))).rank(
        pct=True
    ).fillna(0.5).to_numpy(dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.abs(np.corrcoef(ranked, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    left, right = np.where(np.triu(corr >= float(threshold), k=1))
    return pd.DataFrame(
        {
            "left_feature": [str(features[int(i)]) for i in left],
            "right_feature": [str(features[int(i)]) for i in right],
            "abs_spearman": corr[left, right].astype(np.float32),
        }
    )


def _time_spread_rows(identity: pd.DataFrame, cap: int) -> np.ndarray:
    n = len(identity)
    if cap <= 0 or n <= cap:
        return np.arange(n, dtype=np.int32)
    order = np.lexsort(
        (
            identity["candidate_id"].astype(str).to_numpy(),
            pd.to_datetime(identity["decision_ts"], utc=True).view("int64"),
        )
    )
    positions = np.linspace(0, n - 1, num=cap, dtype=np.int64)
    return np.sort(order[positions].astype(np.int32))


def _cohort_assignments(identity: pd.DataFrame, cohort_count: int) -> pd.DataFrame:
    if cohort_count < 2:
        raise ValueError("target-neutral cache requires at least two cohorts")
    ts = pd.to_datetime(identity["decision_ts"], utc=True, errors="raise")
    timestamp_ns = ts.astype("int64")
    unique = np.sort(timestamp_ns.unique())
    if len(unique) < cohort_count:
        raise ValueError("not enough distinct timestamps for chronological cohorts")
    boundaries = np.array_split(unique, cohort_count)
    assignment = np.full(len(identity), -1, dtype=np.int16)
    raw = timestamp_ns.to_numpy()
    for cohort, block in enumerate(boundaries):
        assignment[np.isin(raw, block)] = cohort
    if (assignment < 0).any():
        raise AssertionError("chronological cohort assignment is incomplete")
    return pd.DataFrame(
        {
            "candidate_id": identity["candidate_id"].astype(str).to_numpy(),
            "chronological_cohort": assignment,
        }
    )


@dataclass(frozen=True)
class TargetNeutralCache:
    root: Path
    manifest: Mapping[str, Any]
    matrix: np.ndarray
    coverage: pd.DataFrame
    spearman_edges: pd.DataFrame
    cohorts: pd.DataFrame


@dataclass(frozen=True)
class ReliefGeometryCache:
    """Target-free neighbour ordering for one deterministic Relief repeat."""

    root: Path
    manifest: Mapping[str, Any]
    standardized_matrix: np.ndarray
    anchor_ids: np.ndarray
    candidate_ids: np.ndarray
    candidate_distance_order: np.ndarray


def materialize_target_neutral_cache(
    root: str | Path,
    *,
    identity: pd.DataFrame,
    features: pd.DataFrame,
    feature_names: Sequence[str],
    selector_manifest_sha256: str,
    selector_feature_contract_sha256: str,
    selector_features_sha256: str,
    correlation_threshold: float = 0.95,
    correlation_rows: int = 2500,
    cohort_count: int = 3,
) -> TargetNeutralCache:
    """Create or exactly validate the reusable feature-only cache."""

    root = Path(root)
    names = list(map(str, feature_names))
    identity_hash = frame_identity_sha256(identity)
    matrix = _feature_matrix(features, names)
    matrix_hash = array_sha256(matrix)
    sample_rows = _time_spread_rows(identity, int(correlation_rows))
    request = {
        "schema": SCHEMA,
        "selector_manifest_sha256": str(selector_manifest_sha256),
        "selector_feature_contract_sha256": str(selector_feature_contract_sha256),
        "selector_features_sha256": str(selector_features_sha256),
        "identity_sha256": identity_hash,
        "rows": int(len(identity)),
        "feature_names": names,
        "feature_names_sha256": canonical_sha256(names),
        "matrix_sha256": matrix_hash,
        "correlation_threshold": float(correlation_threshold),
        "correlation_row_ids_sha256": array_sha256(sample_rows),
        "correlation_rows": int(len(sample_rows)),
        "cohort_count": int(cohort_count),
        "target_dependent_fields_cached": [],
        "full_population_spearman_semantics": (
            "target_neutral_diagnostic_only; MDA must use exact-training-scope "
            "materialize_scoped_spearman_groups"
        ),
    }
    request_sha = canonical_sha256(request)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        return load_target_neutral_cache(
            root,
            expected_request_sha256=request_sha,
            expected_matrix=matrix,
        )
    with atomic_cache_staging(root) as staging:
        if staging is None:
            return load_target_neutral_cache(
                root, expected_request_sha256=request_sha, expected_matrix=matrix
            )
        matrix_path = staging / "feature_matrix.npy"
        coverage_path = staging / "coverage_nonconstant_audit.parquet"
        edge_path = staging / "spearman_edges.parquet"
        cohort_path = staging / "chronological_cohort_identities.parquet"
        coverage = _coverage_audit(matrix, names)
        edges = _spearman_edges(
            matrix, names, row_ids=sample_rows, threshold=correlation_threshold
        )
        cohorts = _cohort_assignments(identity, cohort_count)
        np.save(matrix_path, matrix, allow_pickle=False)
        coverage.to_parquet(coverage_path, index=False, compression="zstd")
        edges.to_parquet(edge_path, index=False, compression="zstd")
        cohorts.to_parquet(cohort_path, index=False, compression="zstd")
        artifacts = {
            "feature_matrix.npy": sha256(matrix_path.read_bytes()).hexdigest(),
            "coverage_nonconstant_audit.parquet": sha256(coverage_path.read_bytes()).hexdigest(),
            "spearman_edges.parquet": sha256(edge_path.read_bytes()).hexdigest(),
            "chronological_cohort_identities.parquet": sha256(cohort_path.read_bytes()).hexdigest(),
        }
        manifest = {
            "schema": SCHEMA,
            "status": "complete",
            "request": request,
            "request_sha256": request_sha,
            "artifact_sha256": artifacts,
            "reuse_contract": {
                "reusable": [
                    "coverage", "nonconstant_audit", "numeric_feature_matrix",
                    "diagnostic_full_population_spearman_edges",
                    "chronological_cohort_identities",
                    "exact_training_scope_spearman_groups_via_scoped_cache",
                ],
                "must_recompute_per_target": [
                    "labels", "sample_weights", "univariate_scores",
                    "relief_hit_miss_scores", "model_fits", "permutation_scores",
                ],
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    return load_target_neutral_cache(
        root, expected_request_sha256=request_sha, expected_matrix=matrix
    )


def load_target_neutral_cache_for_contract(
    root: str | Path,
    *,
    identity: pd.DataFrame,
    feature_names: Sequence[str],
    selector_manifest_sha256: str,
    selector_feature_contract_sha256: str,
    selector_features_sha256: str,
    correlation_threshold: float = 0.95,
    cohort_count: int = 3,
) -> TargetNeutralCache:
    """Hot-load without rereading the source feature parquet.

    The source parquet digest is supplied by the already validated selector
    manifest/runner.  Row identity, ordered feature contract and all source
    artifact hashes must match the cold cache request.  The cached matrix's
    own semantic/content hashes are then reverified by
    :func:`load_target_neutral_cache`.
    """
    cache = load_target_neutral_cache(root)
    request = cache.manifest.get("request") or {}
    expected = {
        "selector_manifest_sha256": str(selector_manifest_sha256),
        "selector_feature_contract_sha256": str(selector_feature_contract_sha256),
        "selector_features_sha256": str(selector_features_sha256),
        "identity_sha256": frame_identity_sha256(identity),
        "feature_names_sha256": canonical_sha256(list(map(str, feature_names))),
        "correlation_threshold": float(correlation_threshold),
        "cohort_count": int(cohort_count),
    }
    drift = {
        key: {"expected": value, "observed": request.get(key)}
        for key, value in expected.items()
        if request.get(key) != value
    }
    if drift:
        raise ValueError(f"target-neutral selector cache hot-load lineage drift: {drift}")
    if int(request.get("rows", len(cache.matrix))) != len(identity):
        raise ValueError("target-neutral selector cache hot-load row-count drift")
    return cache


def load_target_neutral_cache(
    root: str | Path,
    *,
    expected_request_sha256: str | None = None,
    expected_matrix: np.ndarray | None = None,
) -> TargetNeutralCache:
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("target-neutral selector cache manifest is absent")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("target-neutral selector cache schema/status drift")
    if expected_request_sha256 is not None and manifest.get("request_sha256") != expected_request_sha256:
        raise ValueError("target-neutral selector cache request lineage drift")
    paths = {
        "feature_matrix.npy": root / "feature_matrix.npy",
        "coverage_nonconstant_audit.parquet": root / "coverage_nonconstant_audit.parquet",
        "spearman_edges.parquet": root / "spearman_edges.parquet",
        "chronological_cohort_identities.parquet": root / "chronological_cohort_identities.parquet",
    }
    expected = manifest.get("artifact_sha256") or {}
    for name, path in paths.items():
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected.get(name):
            raise ValueError(f"target-neutral selector cache artifact drift: {name}")
    matrix = np.load(paths["feature_matrix.npy"], mmap_mode="r", allow_pickle=False)
    request = manifest.get("request") or {}
    if array_sha256(matrix) != request.get("matrix_sha256"):
        raise ValueError("target-neutral selector cache numeric matrix drift")
    if expected_matrix is not None and array_sha256(expected_matrix) != request.get("matrix_sha256"):
        raise ValueError("target-neutral selector cache does not match requested matrix")
    coverage = pd.read_parquet(paths["coverage_nonconstant_audit.parquet"])
    edges = pd.read_parquet(paths["spearman_edges.parquet"])
    cohorts = pd.read_parquet(paths["chronological_cohort_identities.parquet"])
    if list(coverage["feature"].astype(str)) != list(request.get("feature_names", [])):
        raise ValueError("target-neutral selector cache feature order drift")
    return TargetNeutralCache(root, manifest, matrix, coverage, edges, cohorts)


def groups_from_cached_edges(
    cache: TargetNeutralCache, active_features: Sequence[str]
) -> list[list[int]]:
    """Rebuild exact connected components for the requested active subset."""
    names = list(map(str, active_features))
    positions = {name: index for index, name in enumerate(names)}
    parent = np.arange(len(names), dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return int(value)

    for row in cache.spearman_edges.itertuples(index=False):
        left, right = str(row.left_feature), str(row.right_feature)
        if left not in positions or right not in positions:
            continue
        a, b = find(positions[left]), find(positions[right])
        if a != b:
            parent[b] = a
    grouped: dict[int, list[int]] = {}
    for index in range(len(names)):
        grouped.setdefault(find(index), []).append(index)
    return [members for members in grouped.values() if len(members) > 1]


def materialize_scoped_spearman_groups(
    root: str | Path,
    *,
    cache: TargetNeutralCache,
    train_candidate_ids: Sequence[Any],
    active_features: Sequence[str],
    threshold: float,
    random_state: int,
    max_rows: int = 2500,
) -> tuple[list[list[int]], Mapping[str, Any]]:
    """Cache groups for one exact training cohort, never held-out rows.

    This is the MDA-safe Spearman hook.  It differs from the full-cache edge
    audit by binding the exact cohort's training identities and active feature
    order.  A later target may reuse it only with the identical training
    cohort and matrix.
    """
    root = Path(root)
    names = list(map(str, active_features))
    cached_names = list(map(str, cache.manifest["request"]["feature_names"]))
    missing = [name for name in names if name not in cached_names]
    if missing:
        raise ValueError(f"scoped Spearman features absent from cache: {missing[:20]}")
    cached_ids = cache.cohorts["candidate_id"].astype(str)
    lookup = pd.Series(np.arange(len(cached_ids), dtype=np.int64), index=cached_ids)
    requested = pd.Index(pd.Series(train_candidate_ids, dtype=object).astype(str))
    if requested.has_duplicates:
        raise ValueError("scoped Spearman training candidate ids are duplicated")
    positions = lookup.reindex(requested)
    if positions.isna().any():
        raise ValueError("scoped Spearman training identities are absent from cache")
    row_positions = positions.to_numpy(dtype=np.int64)
    rng = np.random.default_rng(int(random_state))
    if len(row_positions) > int(max_rows):
        selected = rng.choice(len(row_positions), size=int(max_rows), replace=False)
        row_positions = row_positions[selected]
    column_positions = [cached_names.index(name) for name in names]
    matrix = np.asarray(
        cache.matrix[row_positions[:, None], np.asarray(column_positions)[None, :]],
        dtype=np.float32,
    )
    request = {
        "schema": "stage_i_scoped_spearman_group_cache_v1",
        "parent_cache_request_sha256": cache.manifest["request_sha256"],
        "train_candidate_ids_sha256": canonical_sha256(list(map(str, requested))),
        "sampled_row_positions_sha256": array_sha256(row_positions),
        "active_features": names,
        "active_features_sha256": canonical_sha256(names),
        "matrix_sha256": array_sha256(matrix),
        "threshold": float(threshold),
        "random_state": int(random_state),
        "max_rows": int(max_rows),
        "training_rows_only": True,
    }
    request_sha = canonical_sha256(request)
    manifest_path = root / "manifest.json"
    groups_path = root / "groups.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("schema") != request["schema"]
            or manifest.get("status") != "complete"
            or manifest.get("request_sha256") != request_sha
            or not groups_path.is_file()
            or sha256(groups_path.read_bytes()).hexdigest()
            != (manifest.get("artifact_sha256") or {}).get("groups.json")
        ):
            raise ValueError("scoped Spearman cache lineage/artifact drift")
        return json.loads(groups_path.read_text())["groups"], manifest
    ranked = pd.DataFrame(matrix, columns=names).rank(pct=True).fillna(0.5)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.abs(np.corrcoef(ranked.to_numpy(np.float32), rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    parent = np.arange(len(names), dtype=np.int32)
    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return int(index)
    left, right = np.where(np.triu(corr >= float(threshold), k=1))
    for a, b in zip(left, right):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    components: dict[int, list[int]] = {}
    for index in range(len(names)):
        components.setdefault(find(index), []).append(index)
    groups = [members for members in components.values() if len(members) > 1]
    with atomic_cache_staging(root) as staging:
        if staging is None:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("request_sha256") != request_sha:
                raise ValueError("scoped Spearman cache lineage/artifact drift")
            return json.loads(groups_path.read_text())["groups"], manifest
        staged_groups = staging / "groups.json"
        staged_groups.write_text(json.dumps({"groups": groups}, sort_keys=True) + "\n")
        manifest = {
            "schema": request["schema"], "status": "complete",
            "request": request, "request_sha256": request_sha,
            "artifact_sha256": {
                "groups.json": sha256(staged_groups.read_bytes()).hexdigest()
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    return json.loads((root / "groups.json").read_text())["groups"], json.loads(
        (root / "manifest.json").read_text()
    )


def matrix_frame_from_cache(
    cache: TargetNeutralCache,
    *,
    candidate_ids: Sequence[Any],
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return an exact ordered matrix subset without re-reading parquet.

    The candidate-id lookup is one-to-one and fail-closed.  This is the narrow
    runner integration hook: target arms may select different valid rows while
    sharing the same immutable full-side feature matrix.
    """
    cached_ids = cache.cohorts["candidate_id"].astype(str)
    if cached_ids.duplicated().any():
        raise ValueError("target-neutral cache candidate ids are not unique")
    lookup = pd.Series(np.arange(len(cached_ids), dtype=np.int64), index=cached_ids)
    requested = pd.Index(pd.Series(candidate_ids, dtype=object).astype(str))
    if requested.has_duplicates:
        raise ValueError("requested target-arm candidate ids are duplicated")
    positions = lookup.reindex(requested)
    if positions.isna().any():
        missing = requested[positions.isna()][:10].tolist()
        raise ValueError(f"target-arm identities are absent from feature cache: {missing}")
    all_names = list(map(str, cache.manifest["request"]["feature_names"]))
    names = all_names if feature_names is None else list(map(str, feature_names))
    missing_features = [name for name in names if name not in all_names]
    if missing_features:
        raise ValueError(f"requested features are absent from cache: {missing_features[:20]}")
    column_positions = [all_names.index(name) for name in names]
    values = cache.matrix[
        np.asarray(positions, dtype=np.int64)[:, None],
        np.asarray(column_positions, dtype=np.int64)[None, :],
    ]
    return pd.DataFrame(np.asarray(values, dtype=np.float32), columns=names)


def _standardized_relief_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32).copy()
    median = np.nanmedian(arr, axis=0).astype(np.float32, copy=False)
    median[~np.isfinite(median)] = 0.0
    arr = np.where(np.isfinite(arr), arr, median[None, :]).astype(np.float32)
    q25 = np.nanpercentile(arr, 25.0, axis=0).astype(np.float32)
    q75 = np.nanpercentile(arr, 75.0, axis=0).astype(np.float32)
    scale = q75 - q25
    std = np.nanstd(arr, axis=0).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    return np.clip((arr - median[None, :]) / scale[None, :], -8.0, 8.0).astype(
        np.float32
    )


def materialize_relief_geometry_cache(
    root: str | Path,
    *,
    matrix: Any,
    feature_names: Sequence[str],
    work_row_ids: Any,
    training_candidate_ids: Sequence[Any],
    fold_lineage_sha256: str,
    random_state: int,
    anchor_max_rows: int,
    neighbor_candidate_rows: int,
) -> ReliefGeometryCache:
    """Cache the expensive feature-space distance ordering, never labels.

    ``work_row_ids`` is selected without target values (normally a
    chronological time-spread sample).  Labels are applied only by
    :func:`relief_scores_from_geometry`, so S/O/R3 targets share geometry but
    receive independent hit/miss scores.
    """

    root = Path(root)
    raw = np.asarray(matrix, dtype=np.float32)
    row_ids = np.asarray(work_row_ids, dtype=np.int64).reshape(-1)
    training_ids = pd.Index(pd.Series(training_candidate_ids, dtype=object).astype(str))
    if len(training_ids) != len(raw) or training_ids.has_duplicates:
        raise ValueError("Relief geometry requires unique ids for the exact training matrix")
    if (
        len(row_ids) < 4 or len(np.unique(row_ids)) != len(row_ids)
        or (row_ids < 0).any() or (row_ids >= len(raw)).any()
    ):
        raise ValueError("Relief geometry work-row identities are invalid")
    if len(str(fold_lineage_sha256)) != 64:
        raise ValueError("Relief geometry requires an exact fold-lineage SHA256")
    work = _standardized_relief_matrix(raw[row_ids])
    n = len(work)
    rng = np.random.default_rng(int(random_state))
    anchor_n = min(n, int(anchor_max_rows))
    anchors = (
        rng.choice(n, size=anchor_n, replace=False).astype(np.int32)
        if n > anchor_n else np.arange(n, dtype=np.int32)
    )
    candidate_n = min(n, max(int(neighbor_candidate_rows), anchor_n))
    candidates = (
        rng.choice(n, size=candidate_n, replace=False).astype(np.int32)
        if n > candidate_n else np.arange(n, dtype=np.int32)
    )
    candidates = np.unique(np.concatenate([candidates, anchors])).astype(np.int32)
    candidate_matrix = work[candidates]
    candidate_norm = np.einsum("ij,ij->i", candidate_matrix, candidate_matrix)
    order = np.empty((len(anchors), len(candidates)), dtype=np.int32)
    for position, anchor in enumerate(anchors):
        value = work[int(anchor)]
        distance = candidate_norm + float(np.dot(value, value)) - 2.0 * np.dot(
            candidate_matrix, value
        )
        distance = np.maximum(distance, 0.0)
        distance[candidates == int(anchor)] = np.inf
        order[position] = np.argsort(distance, kind="mergesort").astype(np.int32)
    request = {
        "schema": "stage_i_target_neutral_relief_geometry_v1",
        "matrix_sha256": array_sha256(raw),
        "feature_names_sha256": canonical_sha256(list(map(str, feature_names))),
        "work_row_ids_sha256": array_sha256(row_ids),
        "training_candidate_ids_sha256": canonical_sha256(list(training_ids)),
        "work_candidate_ids_sha256": canonical_sha256(
            training_ids[row_ids].astype(str).tolist()
        ),
        "fold_lineage_sha256": str(fold_lineage_sha256),
        "random_state": int(random_state),
        "anchor_max_rows": int(anchor_max_rows),
        "neighbor_candidate_rows": int(neighbor_candidate_rows),
        "standardization": "median_iqr_std_fallback_clip8",
        "labels_cached": False,
    }
    request_sha = canonical_sha256(request)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        return load_relief_geometry_cache(root, expected_request_sha256=request_sha)
    arrays = {
        "standardized_matrix.npy": work,
        "anchor_ids.npy": anchors,
        "candidate_ids.npy": candidates,
        "candidate_distance_order.npy": order,
    }
    with atomic_cache_staging(root) as staging:
        if staging is None:
            return load_relief_geometry_cache(root, expected_request_sha256=request_sha)
        for name, value in arrays.items():
            np.save(staging / name, value, allow_pickle=False)
        manifest = {
            "schema": request["schema"], "status": "complete",
            "request": request, "request_sha256": request_sha,
            "artifact_sha256": {
                name: sha256((staging / name).read_bytes()).hexdigest()
                for name in arrays
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    return load_relief_geometry_cache(root, expected_request_sha256=request_sha)


def load_relief_geometry_cache(
    root: str | Path, *, expected_request_sha256: str | None = None
) -> ReliefGeometryCache:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("schema") != "stage_i_target_neutral_relief_geometry_v1"
        or manifest.get("status") != "complete"
        or (
            expected_request_sha256 is not None
            and manifest.get("request_sha256") != expected_request_sha256
        )
    ):
        raise ValueError("Relief geometry cache contract drift")
    names = (
        "standardized_matrix.npy", "anchor_ids.npy", "candidate_ids.npy",
        "candidate_distance_order.npy",
    )
    arrays: list[np.ndarray] = []
    for name in names:
        path = root / name
        if (
            not path.is_file()
            or sha256(path.read_bytes()).hexdigest()
            != (manifest.get("artifact_sha256") or {}).get(name)
        ):
            raise ValueError(f"Relief geometry cache artifact drift: {name}")
        arrays.append(np.load(path, mmap_mode="r", allow_pickle=False))
    return ReliefGeometryCache(root, manifest, *arrays)


def relief_scores_from_geometry(
    cache: ReliefGeometryCache, target_labels: Any, *, neighbors: int
) -> np.ndarray:
    labels = np.asarray(target_labels).reshape(-1)
    matrix = cache.standardized_matrix
    if len(labels) != len(matrix):
        raise ValueError("Relief labels must align with cached work rows")
    if len(np.unique(labels)) < 2:
        return np.zeros(matrix.shape[1], dtype=np.float32)
    candidate_ids = np.asarray(cache.candidate_ids, dtype=np.int32)
    candidate = matrix[candidate_ids]
    scores = np.zeros(matrix.shape[1], dtype=np.float64)
    used = 0
    k = max(1, int(neighbors))
    for position, anchor in enumerate(np.asarray(cache.anchor_ids, dtype=np.int32)):
        ordered = candidate_ids[
            np.asarray(cache.candidate_distance_order[position], dtype=np.int32)
        ]
        ordered = ordered[ordered != int(anchor)]
        hit = ordered[labels[ordered] == labels[int(anchor)]][:k]
        miss = ordered[labels[ordered] != labels[int(anchor)]][:k]
        if not len(hit) or not len(miss):
            continue
        value = matrix[int(anchor)]
        scores += np.mean(np.abs(matrix[miss] - value), axis=0)
        scores -= np.mean(np.abs(matrix[hit] - value), axis=0)
        used += 1
    return (
        np.zeros(matrix.shape[1], dtype=np.float32)
        if not used else (scores / float(used)).astype(np.float32)
    )


__all__ = [
    "SCHEMA", "ReliefGeometryCache", "TargetNeutralCache", "array_sha256", "canonical_sha256",
    "atomic_cache_staging",
    "frame_identity_sha256", "groups_from_cached_edges",
    "load_relief_geometry_cache", "load_target_neutral_cache",
    "load_target_neutral_cache_for_contract",
    "materialize_relief_geometry_cache", "materialize_target_neutral_cache",
    "materialize_scoped_spearman_groups", "matrix_frame_from_cache",
    "relief_scores_from_geometry",
]
