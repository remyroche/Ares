"""Bounded covariance diagnostics for strict-OOF base-reasoning features.

This module is deliberately an *audit*, not a meta-model feature selector.  It
joins the materialised G1/G2/G3 scalar outputs for the semantic heads inside a
single already-issued side/fold, then reports coverage, variation and bounded
within-fold associations.  It never reads ``leaf_assignments.parquet`` and
therefore never treats a raw LightGBM leaf token as comparable information.

The fold-local G3 SVD coordinates are intentionally excluded.  Their axes are
fitted separately in every base-training fold, so comparing them across folds
would invent a common coordinate system that does not exist.  All reported
outcome associations are diagnostics only: this code performs no fitting,
ranking, thresholding, or feature selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCHEMA = "strict_oof_reasoning_covariance_v1"
IDENTITY = ("candidate_id", "__ts__", "side_name")
REASONING_PREFIX = "base_reasoning__"
G3_SVD_PREFIX = "base_reasoning__g3_contribution_svd_"
RAW_LEAF_MARKERS = ("leaf_token", "leaf_id", "raw_leaf")
VALID_BUNDLES = ("g1", "g2", "g3")


class StrictOOFReasoningCovarianceError(ValueError):
    """Raised when a materialised strict-OOF covariance contract is invalid."""


@dataclass(frozen=True)
class StrictOOFReasoningCovarianceConfig:
    """Resource bounds and numerical guards for a diagnostic-only audit.

    The pairwise subset is chosen without outcomes, by coverage and variance,
    with an equal deterministic allocation to every available ``head ×
    bundle``.  This makes the worst-case rank matrix bounded while preventing
    a wide G2 bucket bundle from silently crowding out G1 or G3.
    """

    max_pairwise_features: int = 96
    min_pairwise_rows: int = 100
    min_variance: float = 1e-12

    def validate(self) -> None:
        if int(self.max_pairwise_features) <= 1:
            raise StrictOOFReasoningCovarianceError("max_pairwise_features must exceed one")
        if int(self.min_pairwise_rows) <= 1:
            raise StrictOOFReasoningCovarianceError("min_pairwise_rows must exceed one")
        if not np.isfinite(float(self.min_variance)) or float(self.min_variance) < 0.0:
            raise StrictOOFReasoningCovarianceError("min_variance must be finite and non-negative")


@dataclass(frozen=True)
class StrictOOFReasoningCovarianceResult:
    """Strict-OOF feature diagnostics, without a candidate-level output ledger."""

    feature_summary: pd.DataFrame
    association_summary: pd.DataFrame
    pairwise_feature_selection: pd.DataFrame
    pairwise_correlation: pd.DataFrame
    bundle_summary: pd.DataFrame
    artifact_summary: pd.DataFrame
    artifact_count: int
    fold_count: int


def discover_strict_oof_reasoning_artifacts(inputs: Sequence[Path]) -> list[Path]:
    """Resolve directories/roots to strict-OOF per-head artifact directories."""

    found: set[Path] = set()
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_file() and path.name == "base_reasoning_manifest.json":
            found.add(path.parent.resolve())
        elif path.is_dir():
            found.update(item.parent.resolve() for item in path.rglob("base_reasoning_manifest.json"))
        else:
            raise FileNotFoundError(path)
    if not found:
        raise StrictOOFReasoningCovarianceError("no base_reasoning_manifest.json found")
    return sorted(found)


def _as_utc(frame: pd.DataFrame, column: str) -> pd.Series:
    value = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if value.isna().any():
        raise StrictOOFReasoningCovarianceError(f"{column} has invalid UTC timestamps")
    return value


def _normalise_identity(frame: pd.DataFrame, *, role: str, path: Path) -> pd.DataFrame:
    missing = [name for name in IDENTITY if name not in frame]
    if missing:
        raise StrictOOFReasoningCovarianceError(f"{role} lacks identity {missing}: {path}")
    result = frame.copy()
    result["candidate_id"] = result["candidate_id"].astype("string")
    result["__ts__"] = _as_utc(result, "__ts__")
    result["side_name"] = result["side_name"].astype("string").str.lower()
    if result.loc[:, list(IDENTITY)].isna().any().any() or result.duplicated(list(IDENTITY)).any():
        raise StrictOOFReasoningCovarianceError(f"{role} has null or duplicate identities: {path}")
    return result.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)


def _require_one(frame: pd.DataFrame, column: str, *, role: str, path: Path) -> str:
    if column not in frame:
        raise StrictOOFReasoningCovarianceError(f"{role} lacks {column}: {path}")
    values = frame[column].dropna().astype(str).unique().tolist()
    if len(values) != 1 or not values[0].strip():
        raise StrictOOFReasoningCovarianceError(f"{role} must have exactly one {column}: {path}")
    return str(values[0])


def _equal_numeric(left: pd.Series, right: pd.Series) -> bool:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    return bool(np.allclose(a, b, rtol=0.0, atol=1e-12, equal_nan=True))


def _bundle_for(column: str) -> str:
    for bundle in VALID_BUNDLES:
        if column.startswith(f"{REASONING_PREFIX}{bundle}_") or column.startswith(f"{bundle}_"):
            return bundle
    raise StrictOOFReasoningCovarianceError(f"not a recognised G1/G2/G3 scalar: {column}")


@dataclass(frozen=True)
class _HeadArtifact:
    path: Path
    side_name: str
    fold_id: str
    head_name: str
    feature_contract_sha256: str
    frame: pd.DataFrame
    scalar_columns: tuple[str, ...]


def _read_head_artifact(path: Path) -> _HeadArtifact:
    manifest_path = path / "base_reasoning_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "MATERIALIZED_STRICT_OOF":
        raise StrictOOFReasoningCovarianceError(f"not materialised strict OOF: {path}")
    for key in ("side_name", "fold_id", "head_name"):
        if not str(manifest.get(key, "")).strip():
            raise StrictOOFReasoningCovarianceError(f"manifest lacks {key}: {path}")
    try:
        contract = str(manifest["provenance"]["feature_contract_sha256"])
    except KeyError as exc:
        raise StrictOOFReasoningCovarianceError(f"manifest lacks feature-contract hash: {path}") from exc
    if not contract:
        raise StrictOOFReasoningCovarianceError(f"blank feature-contract hash: {path}")

    names = {
        "features": "base_reasoning_features.parquet",
        "predictions": "base_reasoning_predictions.parquet",
        "labels": "base_reasoning_labels.parquet",
    }
    missing = [name for name in names.values() if not (path / name).exists()]
    if missing:
        raise StrictOOFReasoningCovarianceError(f"artifact lacks {missing}: {path}")
    feature_schema = pq.ParquetFile(path / names["features"]).schema.names
    scalar_columns = tuple(
        column for column in feature_schema
        if column.startswith(REASONING_PREFIX) and not column.startswith(G3_SVD_PREFIX)
    )
    if not scalar_columns:
        raise StrictOOFReasoningCovarianceError(f"no comparable G1/G2/G3 scalars: {path}")
    forbidden = [column for column in scalar_columns if any(marker in column.lower() for marker in RAW_LEAF_MARKERS)]
    # g1_leaf_assignment_count is an aggregate count, not a raw identifier.
    forbidden = [column for column in forbidden if column != "base_reasoning__g1_leaf_assignment_count"]
    if forbidden:
        raise StrictOOFReasoningCovarianceError(f"raw leaf identifier leaked into feature table: {forbidden}")
    unknown = [column for column in scalar_columns if not any(column.startswith(f"{REASONING_PREFIX}{bundle}_") for bundle in VALID_BUNDLES)]
    if unknown:
        raise StrictOOFReasoningCovarianceError(f"unknown reasoning scalar bundle: {unknown}")

    features = _normalise_identity(
        pd.read_parquet(path / names["features"], columns=[*IDENTITY, "head_name", "fold_id", *scalar_columns]),
        role="features", path=path,
    )
    predictions = _normalise_identity(
        pd.read_parquet(path / names["predictions"], columns=[*IDENTITY, "base_prediction", "head_name", "class_index", "fold_id"]),
        role="predictions", path=path,
    )
    labels = _normalise_identity(
        pd.read_parquet(path / names["labels"], columns=[*IDENTITY, "label__r3_class", "label__net_bps", "head_name", "fold_id"]),
        role="labels", path=path,
    )
    expected_side = str(manifest["side_name"]).lower()
    expected_fold = str(manifest["fold_id"])
    expected_head = str(manifest["head_name"])
    for frame, role in ((features, "features"), (predictions, "predictions"), (labels, "labels")):
        if not frame["side_name"].eq(expected_side).all():
            raise StrictOOFReasoningCovarianceError(f"{role} side mismatches manifest: {path}")
        if _require_one(frame, "fold_id", role=role, path=path) != expected_fold:
            raise StrictOOFReasoningCovarianceError(f"{role} fold mismatches manifest: {path}")
        try:
            actual_head = _require_one(frame, "head_name", role=role, path=path)
        except StrictOOFReasoningCovarianceError as exc:
            raise StrictOOFReasoningCovarianceError(f"{role} head mismatches manifest: {path}") from exc
        if actual_head != expected_head:
            raise StrictOOFReasoningCovarianceError(f"{role} head mismatches manifest: {path}")
    if not features.loc[:, list(IDENTITY)].equals(predictions.loc[:, list(IDENTITY)]) or not features.loc[:, list(IDENTITY)].equals(labels.loc[:, list(IDENTITY)]):
        raise StrictOOFReasoningCovarianceError(f"feature/prediction/label identities differ: {path}")
    class_index = pd.to_numeric(predictions["class_index"], errors="coerce")
    if class_index.isna().any() or class_index.nunique() != 1:
        raise StrictOOFReasoningCovarianceError(f"predictions lack one valid class_index: {path}")
    declared_class = manifest.get("provenance", {}).get("class_index")
    if declared_class is not None and int(class_index.iloc[0]) != int(declared_class):
        raise StrictOOFReasoningCovarianceError(f"class_index mismatches manifest: {path}")

    frame = features.loc[:, list(IDENTITY)].copy()
    frame["base_prediction"] = pd.to_numeric(predictions["base_prediction"], errors="coerce")
    frame["net_bps"] = pd.to_numeric(labels["label__net_bps"], errors="coerce")
    frame["semantic_label"] = labels["label__r3_class"].eq(int(class_index.iloc[0])).astype(np.float32)
    for column in scalar_columns:
        frame[column] = pd.to_numeric(features[column], errors="coerce").astype(np.float32)
    return _HeadArtifact(path, expected_side, expected_fold, expected_head, contract, frame, scalar_columns)


def _correlation(left: np.ndarray, right: np.ndarray, *, min_rows: int, min_variance: float) -> tuple[int, float, float]:
    valid = np.isfinite(left) & np.isfinite(right)
    count = int(valid.sum())
    if count < int(min_rows):
        return count, np.nan, np.nan
    x = left[valid].astype(np.float64, copy=False)
    y = right[valid].astype(np.float64, copy=False)
    x_var = float(np.var(x))
    y_var = float(np.var(y))
    if x_var <= float(min_variance) or y_var <= float(min_variance):
        return count, np.nan, np.nan
    pearson = float(np.corrcoef(x, y)[0, 1])
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    if float(np.var(xr)) <= 0.0 or float(np.var(yr)) <= 0.0:
        return count, pearson, np.nan
    return count, pearson, float(np.corrcoef(xr, yr)[0, 1])


def _dense_correlation_matrix(values: np.ndarray, *, min_rows: int, min_variance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return guarded Pearson/Spearman matrices for a fully finite bounded set.

    Pairwise ranking is the costly part of this diagnostic.  For the common
    materialised case (all comparable scalar fields present), rank every column
    once and use two small matrix products rather than re-ranking every pair.
    The caller falls back to pairwise guards whenever missingness differs.
    """

    if values.ndim != 2 or not np.isfinite(values).all():
        raise StrictOOFReasoningCovarianceError("dense correlation requires a finite two-dimensional matrix")
    rows = int(values.shape[0])
    if rows < int(min_rows):
        shape = (values.shape[1], values.shape[1])
        return np.full(shape, rows, dtype=np.int64), np.full(shape, np.nan), np.full(shape, np.nan)
    work = values.astype(np.float64, copy=False)
    centered = work - work.mean(axis=0, keepdims=True)
    denominator = np.sqrt(np.sum(centered * centered, axis=0))
    pearson = np.full((work.shape[1], work.shape[1]), np.nan, dtype=np.float64)
    valid = denominator > np.sqrt(float(min_variance) * rows)
    if valid.any():
        pearson[np.ix_(valid, valid)] = (
            centered[:, valid].T @ centered[:, valid]
            / np.outer(denominator[valid], denominator[valid])
        )
    ranks = pd.DataFrame(work).rank(method="average").to_numpy(dtype=np.float64)
    ranks -= ranks.mean(axis=0, keepdims=True)
    rank_denominator = np.sqrt(np.sum(ranks * ranks, axis=0))
    spearman = np.full_like(pearson, np.nan)
    rank_valid = rank_denominator > 0.0
    if rank_valid.any():
        spearman[np.ix_(rank_valid, rank_valid)] = (
            ranks[:, rank_valid].T @ ranks[:, rank_valid]
            / np.outer(rank_denominator[rank_valid], rank_denominator[rank_valid])
        )
    return np.full((work.shape[1], work.shape[1]), rows, dtype=np.int64), pearson, spearman


def _feature_summary(frame: pd.DataFrame, metadata: dict[str, str], config: StrictOOFReasoningCovarianceConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in (name for name in frame if name.startswith(REASONING_PREFIX)):
        value = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = value[np.isfinite(value)]
        rows.append({
            **metadata,
            "feature_name": column,
            "head_name": metadata["feature_head"],
            "bundle": metadata["feature_bundle"],
            "rows": int(len(value)),
            "finite_rows": int(len(finite)),
            "coverage": float(len(finite) / max(len(value), 1)),
            "unique_finite_values": int(pd.Series(finite).nunique(dropna=True)),
            "variance": float(np.var(finite)) if len(finite) else np.nan,
            "standard_deviation": float(np.std(finite)) if len(finite) else np.nan,
            "eligible_pairwise": bool(len(finite) >= config.min_pairwise_rows and len(finite) and np.var(finite) > config.min_variance),
        })
    return pd.DataFrame(rows)


def _select_pairwise_features(summary: pd.DataFrame, config: StrictOOFReasoningCovarianceConfig) -> pd.DataFrame:
    """Use deterministic unsupervised quotas, preserving every G bundle."""

    work = summary.copy()
    cells = sorted(work[["feature_head", "feature_bundle"]].drop_duplicates().itertuples(index=False, name=None))
    quota = max(1, int(config.max_pairwise_features) // max(len(cells), 1))
    work = work.sort_values(
        ["feature_head", "feature_bundle", "eligible_pairwise", "coverage", "variance", "feature_name"],
        ascending=[True, True, False, False, False, True], kind="stable",
    )
    work["within_head_bundle_rank"] = work.groupby(["feature_head", "feature_bundle"], observed=True).cumcount() + 1
    work["selected_for_pairwise"] = work["eligible_pairwise"] & work["within_head_bundle_rank"].le(quota)
    # Fill spare capacity deterministically, still without outcomes.
    spare = int(config.max_pairwise_features) - int(work["selected_for_pairwise"].sum())
    if spare > 0:
        extras = work.loc[work["eligible_pairwise"] & ~work["selected_for_pairwise"]].sort_values(
            ["coverage", "variance", "feature_name"], ascending=[False, False, True], kind="stable",
        ).index[:spare]
        work.loc[extras, "selected_for_pairwise"] = True
    work["selection_reason"] = np.where(
        work["selected_for_pairwise"], "coverage_variance_quota", np.where(work["eligible_pairwise"], "bounded_out", "coverage_or_variance_guard")
    )
    return work


def _fold_diagnostics(artifacts: Sequence[_HeadArtifact], config: StrictOOFReasoningCovarianceConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    first = artifacts[0]
    if len({item.feature_contract_sha256 for item in artifacts}) != 1:
        raise StrictOOFReasoningCovarianceError(f"head artifacts have different feature contracts in {first.side_name}/{first.fold_id}")
    identity = first.frame.loc[:, list(IDENTITY)].copy()
    combined_columns: dict[str, np.ndarray] = {}
    net_bps = first.frame["net_bps"]
    feature_parts: list[pd.DataFrame] = []
    artifact_rows: list[dict[str, Any]] = []
    for item in sorted(artifacts, key=lambda value: value.head_name):
        if not identity.equals(item.frame.loc[:, list(IDENTITY)]):
            raise StrictOOFReasoningCovarianceError(f"head candidate identities differ in {item.side_name}/{item.fold_id}")
        if not _equal_numeric(net_bps, item.frame["net_bps"]):
            raise StrictOOFReasoningCovarianceError(f"net label differs between heads in {item.side_name}/{item.fold_id}")
        # The semantic heads are selected from different multiclass columns;
        # their base predictions are intentionally head-local, not expected to
        # equal a single scalar score.
        combined_columns[f"base_prediction__{item.head_name}"] = item.frame["base_prediction"].to_numpy(copy=False)
        semantic = f"semantic_label__{item.head_name}"
        combined_columns[semantic] = item.frame["semantic_label"].to_numpy(copy=False)
        prefixed: dict[str, np.ndarray] = {}
        for raw in item.scalar_columns:
            bundle = _bundle_for(raw)
            suffix = raw.removeprefix(REASONING_PREFIX)
            name = f"base_reasoning__{item.head_name}__{suffix}"
            if name in combined_columns or name in prefixed:
                raise StrictOOFReasoningCovarianceError(f"duplicate head-prefixed scalar {name}")
            prefixed[name] = item.frame[raw].to_numpy(copy=False)
        combined_columns.update(prefixed)
        for name in prefixed:
            bundle = _bundle_for(name.removeprefix(f"base_reasoning__{item.head_name}__"))
            metadata = {
                "side_name": item.side_name,
                "fold_id": item.fold_id,
                "feature_contract_sha256": item.feature_contract_sha256,
                "feature_head": item.head_name,
                "feature_bundle": bundle,
            }
            # Build a narrow one-column frame to keep the summary path simple.
            feature_parts.append(_feature_summary(pd.DataFrame({name: prefixed[name]}), metadata, config))
        artifact_rows.append({
            "artifact_dir": str(item.path), "side_name": item.side_name, "fold_id": item.fold_id,
            "head_name": item.head_name, "feature_contract_sha256": item.feature_contract_sha256,
            "rows": int(len(item.frame)), "scalar_feature_count": int(len(item.scalar_columns)),
            "g1_scalar_count": int(sum(_bundle_for(column) == "g1" for column in item.scalar_columns)),
            "g2_scalar_count": int(sum(_bundle_for(column) == "g2" for column in item.scalar_columns)),
            "g3_scalar_count": int(sum(_bundle_for(column) == "g3" for column in item.scalar_columns)),
            "g3_svd_excluded_count": int(sum(name.startswith(G3_SVD_PREFIX) for name in pq.ParquetFile(item.path / "base_reasoning_features.parquet").schema.names)),
        })
    # Construct the wide fold ledger in one operation; repeated insertion of
    # ~200 head-prefixed fields fragments the frame and defeats the memory cap.
    combined = pd.concat([identity, pd.DataFrame(combined_columns, index=identity.index)], axis=1)
    feature_summary = pd.concat(feature_parts, ignore_index=True)
    selection = _select_pairwise_features(feature_summary, config)
    selected = selection.loc[selection["selected_for_pairwise"], "feature_name"].tolist()
    targets = {"net_bps": net_bps.to_numpy(dtype=float)}
    targets.update({name: combined[name].to_numpy(dtype=float) for name in combined if name.startswith("base_prediction__")})
    targets.update({name: combined[name].to_numpy(dtype=float) for name in combined if name.startswith("semantic_label__")})
    associations: list[dict[str, Any]] = []
    for feature_head, cell in feature_summary.groupby("feature_head", observed=True, sort=True):
        feature_names = cell["feature_name"].tolist()
        target_names = [f"base_prediction__{feature_head}", f"semantic_label__{feature_head}", "net_bps"]
        all_names = [*feature_names, *target_names]
        matrix = np.column_stack([combined[name].to_numpy(dtype=float) if name in combined else targets[name] for name in all_names])
        dense = np.isfinite(matrix).all()
        if dense:
            counts, pearson_matrix, spearman_matrix = _dense_correlation_matrix(
                matrix, min_rows=config.min_pairwise_rows, min_variance=config.min_variance,
            )
        for feature_index, row in enumerate(cell.itertuples(index=False)):
            x = matrix[:, feature_index]
            for target_index, target_name in enumerate(target_names, start=len(feature_names)):
                if dense:
                    count = int(counts[feature_index, target_index])
                    pearson = float(pearson_matrix[feature_index, target_index])
                    spearman = float(spearman_matrix[feature_index, target_index])
                else:
                    count, pearson, spearman = _correlation(
                        x, targets[target_name], min_rows=config.min_pairwise_rows, min_variance=config.min_variance,
                    )
                associations.append({
                    "side_name": row.side_name, "fold_id": row.fold_id, "feature_contract_sha256": row.feature_contract_sha256,
                    "feature_name": row.feature_name, "feature_head": row.feature_head, "feature_bundle": row.feature_bundle,
                    "diagnostic_target": target_name, "pair_rows": count, "pearson": pearson, "spearman": spearman,
                    "diagnostic_only": True,
                })
    pairwise: list[dict[str, Any]] = []
    selected_values = combined.loc[:, selected].to_numpy(dtype=float)
    if len(selected) and np.isfinite(selected_values).all():
        pair_counts, pair_pearson, pair_spearman = _dense_correlation_matrix(
            selected_values, min_rows=config.min_pairwise_rows, min_variance=config.min_variance,
        )
        for left_index, left_name in enumerate(selected):
            for right_index, right_name in enumerate(selected[left_index + 1:], start=left_index + 1):
                pairwise.append({
                    "side_name": first.side_name, "fold_id": first.fold_id, "feature_contract_sha256": first.feature_contract_sha256,
                    "left_feature_name": left_name, "right_feature_name": right_name,
                    "pair_rows": int(pair_counts[left_index, right_index]),
                    "pearson": float(pair_pearson[left_index, right_index]),
                    "spearman": float(pair_spearman[left_index, right_index]),
                })
    else:
        # Missingness changes the rank population.  The bounded fallback keeps
        # the stated pairwise-valid semantics exact rather than imputing it.
        for left_index, left_name in enumerate(selected):
            x = combined[left_name].to_numpy(dtype=float)
            for right_name in selected[left_index + 1:]:
                count, pearson, spearman = _correlation(x, combined[right_name].to_numpy(dtype=float), min_rows=config.min_pairwise_rows, min_variance=config.min_variance)
                pairwise.append({
                    "side_name": first.side_name, "fold_id": first.fold_id, "feature_contract_sha256": first.feature_contract_sha256,
                    "left_feature_name": left_name, "right_feature_name": right_name, "pair_rows": count,
                    "pearson": pearson, "spearman": spearman,
                })
    bundle = feature_summary.merge(
        pd.DataFrame(associations).loc[lambda value: value.diagnostic_target.eq("base_prediction__" + value.feature_head)],
        on=["side_name", "fold_id", "feature_contract_sha256", "feature_name", "feature_head", "feature_bundle"], how="left", validate="one_to_one",
    ).groupby(["side_name", "fold_id", "feature_contract_sha256", "feature_head", "feature_bundle"], observed=True).agg(
        feature_count=("feature_name", "size"), eligible_pairwise_count=("eligible_pairwise", "sum"),
        selected_pairwise_count=("feature_name", lambda values: int(selection.set_index("feature_name").loc[list(values), "selected_for_pairwise"].sum())),
        coverage_mean=("coverage", "mean"), coverage_min=("coverage", "min"),
        variance_median=("variance", "median"), nonconstant_count=("eligible_pairwise", "sum"),
        own_base_prediction_abs_pearson_mean=("pearson", lambda values: float(np.nanmean(np.abs(values))) if np.isfinite(values).any() else np.nan),
    ).reset_index()
    return feature_summary, pd.DataFrame(associations), selection, pd.DataFrame(pairwise), bundle, pd.DataFrame(artifact_rows)


def analyze_strict_oof_reasoning_covariance(
    inputs: Sequence[Path], *, config: StrictOOFReasoningCovarianceConfig = StrictOOFReasoningCovarianceConfig(),
) -> StrictOOFReasoningCovarianceResult:
    """Audit strict-OOF G1/G2/G3 scalar covariance fold by fold, without fitting."""

    config.validate()
    parsed = [_read_head_artifact(path) for path in discover_strict_oof_reasoning_artifacts(inputs)]
    groups: dict[tuple[str, str], list[_HeadArtifact]] = {}
    for item in parsed:
        key = (item.side_name, item.fold_id)
        if any(other.head_name == item.head_name for other in groups.setdefault(key, [])):
            raise StrictOOFReasoningCovarianceError(f"duplicate side/fold/head artifact: {key}/{item.head_name}")
        groups[key].append(item)
    results = [_fold_diagnostics(value, config) for _, value in sorted(groups.items())]
    columns = ("feature_summary", "association_summary", "pairwise_feature_selection", "pairwise_correlation", "bundle_summary", "artifact_summary")
    tables = [pd.concat([result[index] for result in results], ignore_index=True) for index in range(len(columns))]
    return StrictOOFReasoningCovarianceResult(*tables, artifact_count=len(parsed), fold_count=len(groups))


def write_strict_oof_reasoning_covariance(
    result: StrictOOFReasoningCovarianceResult, output_dir: Path,
) -> Path:
    """Write a new immutable diagnostic artifact directory."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        tables = {
            "feature_summary.parquet": result.feature_summary,
            "association_summary.parquet": result.association_summary,
            "pairwise_feature_selection.parquet": result.pairwise_feature_selection,
            "pairwise_correlation.parquet": result.pairwise_correlation,
            "bundle_summary.parquet": result.bundle_summary,
            "artifact_summary.parquet": result.artifact_summary,
        }
        for name, table in tables.items():
            table.to_parquet(temporary / name, index=False, compression="zstd")
        manifest = {
            "schema": SCHEMA,
            "status": "COMPLETED_DIAGNOSTIC_ONLY",
            "artifact_count": int(result.artifact_count), "fold_count": int(result.fold_count),
            "contracts": {
                "inputs": "materialised strict-OOF per-head artifacts; validated manifest/table side, fold, head and candidate identity",
                "features": "all G1/G2/G3 scalar outputs joined with head prefixes",
                "g3": "fold-local contribution SVD coordinates excluded",
                "raw_leaf_tokens": "never read or used as an input",
                "pairwise": "bounded deterministic coverage/variance-only subset; Pearson and Spearman with support/variance guards",
                "outcomes": "base prediction, own semantic class label and net bps associations are diagnostic only",
                "fitting": "none",
            },
            "rows": {name: int(len(table)) for name, table in tables.items()},
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output_dir)
        return output_dir
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_strict_oof_reasoning_covariance(
    inputs: Sequence[Path], output_dir: Path, *, config: StrictOOFReasoningCovarianceConfig = StrictOOFReasoningCovarianceConfig(),
) -> Path:
    return write_strict_oof_reasoning_covariance(analyze_strict_oof_reasoning_covariance(inputs, config=config), output_dir)
