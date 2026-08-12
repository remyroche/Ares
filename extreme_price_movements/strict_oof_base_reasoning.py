"""Strict-OOF materialisation of base-model reasoning features.

This module is deliberately narrower than the regime research code.  It turns
one *already fitted*, side-local and head-local base-model fold into auditable
G1/G2/G3 outputs for the later meta layer:

* G1 -- local leaf assignment tokens plus train-fold leaf support;
* G2 -- decision-path rule signatures and compact path-shape features;
* G3 -- additive contribution summaries and a train-fold-fitted SVD bundle.

Raw LightGBM leaf indices are only used transiently.  Persisted assignments
are opaque tokens scoped to ``(head, side, fold, model, tree)``; the module
never compares or aligns a numeric leaf ID across models, folds, sides, or a
regime model.  The only cross-model description is the explicit G2 rule
signature, built from split paths rather than leaf numbers.

The caller must supply the actual train/evaluation split.  The materialiser
requires a strict chronological boundary and never accepts evaluation labels as
features.  It is therefore safe to invoke once for every base OOF fold and
concatenate only the returned evaluation features for meta training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .lgbm_archetype_features import (
    CONTRIB_ARCHETYPE_FEATURE_NAMES,
    contrib_summary_frame,
    fit_contrib_archetype_state,
    transform_contrib_archetype_features,
)


# v2 adds catalog-scoped selected-tree additive leaf values.  The values are
# only meaningful when joined to assignments from the same artifact and are
# the lineage boundary for later contribution-weighted family states.
STRICT_OOF_BASE_REASONING_SCHEMA = "strict_oof_base_reasoning_v2"
IDENTITY_COLUMNS = ("candidate_id", "__ts__", "side_name")
FEATURE_PREFIX = "base_reasoning__"


class StrictOOFBaseReasoningError(ValueError):
    """Raised when fold-local base-reasoning lineage is invalid."""


class StrictOOFContributionCacheCapacityError(StrictOOFBaseReasoningError):
    """The optional shared multiclass G3 cache would exceed its hard bound."""


@dataclass(frozen=True)
class StrictOOFBaseReasoningConfig:
    """Bounded output controls for one side/head OOF base fold.

    ``max_trees_per_model`` caps selected trees for the requested semantic head,
    not raw interleaved LightGBM tree columns.
    """

    max_trees_per_model: int = 64
    rule_signature_bucket_count: int = 16
    rule_threshold_band_count: int = 10
    rule_threshold_band_min_train_rows: int = 32
    contribution_components: int = 16
    contribution_method: str = "pred_contrib"
    contribution_batch_rows: int = 50_000
    write_leaf_assignments: bool = True
    write_rule_catalog: bool = True

    def validate(self) -> None:
        if int(self.max_trees_per_model) <= 0:
            raise StrictOOFBaseReasoningError("max_trees_per_model must be positive")
        if int(self.rule_signature_bucket_count) <= 0:
            raise StrictOOFBaseReasoningError(
                "rule_signature_bucket_count must be positive"
            )
        if int(self.rule_threshold_band_count) < 2:
            raise StrictOOFBaseReasoningError(
                "rule_threshold_band_count must be at least two"
            )
        if int(self.rule_threshold_band_min_train_rows) <= 0:
            raise StrictOOFBaseReasoningError(
                "rule_threshold_band_min_train_rows must be positive"
            )
        if int(self.contribution_components) <= 0:
            raise StrictOOFBaseReasoningError(
                "contribution_components must be positive"
            )
        if int(self.contribution_batch_rows) <= 0:
            raise StrictOOFBaseReasoningError(
                "contribution_batch_rows must be positive"
            )
        if str(self.contribution_method).strip().lower() != "pred_contrib":
            raise StrictOOFBaseReasoningError(
                "only LightGBM pred_contrib is supported; path proxies are not "
                "a substitute for the G3 additive bundle"
            )


@dataclass(frozen=True)
class StrictOOFBaseReasoningResult:
    """Evaluation-only rows and supporting fold-local audit tables."""

    features: pd.DataFrame
    predictions: pd.DataFrame
    labels: pd.DataFrame | None
    leaf_assignments: pd.DataFrame
    leaf_rule_catalog: pd.DataFrame
    contribution_bundle: pd.DataFrame
    manifest: dict[str, Any]
    artifact_dir: Path | None = None


@dataclass
class StrictOOFMulticlassContributionCache:
    """Per-fold, all-class additive contribution cache.

    This is deliberately a narrow, in-memory optimisation for the common
    three-class R3 base.  It contains only additive feature contributions,
    never leaf IDs, assignments, labels, or a fitted transform.  The cache is
    bound to the exact fitted model ensemble and train/evaluation matrices so
    callers cannot accidentally reuse a contribution block across folds.

    Arrays have shape ``[class, row, feature]`` and are read-only after
    construction.  They are normally in-memory, but an explicitly bounded
    cache may be backed by temporary ``float32`` memmaps.  That spill mode
    keeps exactly the same all-class values while preventing a large later
    chronological fold from silently reverting to three independent expensive
    ``pred_contrib`` passes.  Backing files are temporary implementation state,
    never a persisted reasoning artifact, and must be released after all three
    semantic heads have been materialised.
    """

    train_contributions: np.ndarray
    eval_contributions: np.ndarray
    class_count: int
    feature_names: tuple[str, ...]
    model_hashes: tuple[str, ...]
    train_matrix_fingerprint: str
    eval_matrix_fingerprint: str
    retained_bytes: int
    storage_mode: str = "in_memory"
    backing_paths: tuple[Path, ...] = ()
    backing_directory: Path | None = None
    closed: bool = False

    def release(self) -> None:
        """Close and remove temporary backing maps, if this cache spilled.

        The public materialiser only reads the cache; the strict runner owns
        its lifetime and calls this in a ``finally`` block once all explicit
        adverse/weak/clear slices have been produced.  Calling it more than
        once is harmless.  In-memory arrays deliberately remain ordinary
        process-owned arrays and require no special cleanup.
        """

        if self.closed:
            return
        for values in (self.train_contributions, self.eval_contributions):
            if isinstance(values, np.memmap):
                try:
                    values.flush()
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass
                mapping = getattr(values, "_mmap", None)
                if mapping is not None:
                    try:
                        mapping.close()
                    except Exception:  # pragma: no cover - best-effort cleanup
                        pass
        if self.backing_directory is not None:
            try:
                shutil.rmtree(self.backing_directory, ignore_errors=True)
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
        else:
            for path in self.backing_paths:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:  # pragma: no cover - best-effort cleanup
                    pass
        self.closed = True


@dataclass(frozen=True)
class _LeafRule:
    leaf_token: np.uint64
    rule_signature: str
    rule_raw_signature: str
    rule_structural_path_json: str
    rule_threshold_band_json: str
    rule_feature_signature: str
    rule_path_json: str
    path_depth: int
    unique_feature_count: int
    numeric_threshold_count: int
    right_branch_fraction: float
    signature_bucket: int
    tree_leaf_value: float


def _as_utc(values: Sequence[object] | pd.Series, *, role: str, expected: int) -> pd.Series:
    if len(values) != int(expected):
        raise StrictOOFBaseReasoningError(
            f"{role} length={len(values)} does not match rows={expected}"
        )
    result = pd.to_datetime(pd.Series(values, copy=False), utc=True, errors="coerce")
    if result.isna().any():
        raise StrictOOFBaseReasoningError(f"{role} must contain valid UTC timestamps")
    return result.reset_index(drop=True)


def _require_matrix_pair(train: pd.DataFrame, evaluate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(train, pd.DataFrame) or not isinstance(evaluate, pd.DataFrame):
        raise StrictOOFBaseReasoningError("train_matrix and eval_matrix must be DataFrames")
    if train.empty or evaluate.empty:
        raise StrictOOFBaseReasoningError("train_matrix and eval_matrix must be non-empty")
    if train.columns.duplicated().any() or evaluate.columns.duplicated().any():
        raise StrictOOFBaseReasoningError("base matrices cannot have duplicate columns")
    columns = list(map(str, train.columns))
    if columns != list(map(str, evaluate.columns)):
        raise StrictOOFBaseReasoningError(
            "train_matrix and eval_matrix must have the same ordered base feature contract"
        )
    # The matrices are the frozen base contract.  Do not silently fill a field
    # missing from a fold; LightGBM's own frozen missing-value routing remains
    # valid and is intentionally left untouched.
    return train.copy(deep=False), evaluate.copy(deep=False)


def _normalise_identity(
    identity: pd.DataFrame,
    *,
    eval_timestamps: pd.Series,
    side_name: str,
) -> pd.DataFrame:
    if not isinstance(identity, pd.DataFrame):
        raise StrictOOFBaseReasoningError("eval_identity must be a DataFrame")
    missing = [column for column in IDENTITY_COLUMNS if column not in identity]
    if missing:
        raise StrictOOFBaseReasoningError(f"eval_identity is missing {missing}")
    if len(identity) != len(eval_timestamps):
        raise StrictOOFBaseReasoningError("eval_identity is not aligned to eval_matrix")
    result = identity.loc[:, list(IDENTITY_COLUMNS)].copy().reset_index(drop=True)
    result["candidate_id"] = result["candidate_id"].astype("string")
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="coerce")
    result["side_name"] = result["side_name"].astype("string").str.lower()
    if result.isna().any().any() or result["candidate_id"].str.strip().eq("").any():
        raise StrictOOFBaseReasoningError("eval_identity contains a null or blank key")
    if result.duplicated(list(IDENTITY_COLUMNS)).any():
        raise StrictOOFBaseReasoningError("eval_identity contains duplicate candidate identities")
    if not result["side_name"].eq(str(side_name)).all():
        raise StrictOOFBaseReasoningError("eval_identity crosses sides in a side-local call")
    if not result["__ts__"].equals(eval_timestamps):
        raise StrictOOFBaseReasoningError(
            "eval_identity timestamps must exactly equal eval_timestamps"
        )
    return result


def _model_hash(model: Any) -> str:
    booster = getattr(model, "booster_", None)
    if booster is not None:
        try:
            value = booster.model_to_string(num_iteration=-1)
            return hashlib.sha256(value.encode("utf-8")).hexdigest()
        except Exception:
            pass
    return hashlib.sha256(repr(model).encode("utf-8", errors="ignore")).hexdigest()


def _token_u64(value: str) -> np.uint64:
    return np.uint64(int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big"))


def _rule_hash(payload: Any) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _fit_rule_threshold_bands(
    train_matrix: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    band_count: int,
    min_train_rows: int,
) -> dict[str, dict[str, Any]]:
    """Fit robust, fold-local numeric threshold bands from base-training rows.

    The bands describe a split threshold's relative position in the training
    distribution; they do not consume evaluation values.  Constant, sparse,
    and non-numeric features are retained as explicit structural states rather
    than falling back to raw threshold values.
    """

    state: dict[str, dict[str, Any]] = {}
    probabilities = np.linspace(0.0, 1.0, int(band_count) + 1, dtype=np.float64)[1:-1]
    for position, feature in enumerate(map(str, feature_names)):
        numeric = pd.to_numeric(train_matrix.iloc[:, position], errors="coerce").to_numpy(
            dtype=np.float64,
            copy=False,
        )
        finite = numeric[np.isfinite(numeric)]
        entry: dict[str, Any] = {"finite_train_rows": int(len(finite)), "cuts": []}
        if len(finite) < int(min_train_rows):
            entry["state"] = "numeric_low_support"
        elif float(np.min(finite)) == float(np.max(finite)):
            entry["state"] = "numeric_constant"
        else:
            cuts = np.unique(np.quantile(finite, probabilities)).astype(np.float64)
            if len(cuts) == 0:
                entry["state"] = "numeric_constant"
            else:
                entry["state"] = "numeric_quantile"
                entry["cuts"] = [float(value) for value in cuts.tolist()]
        state[feature] = entry
    return state


def _threshold_band_descriptor(
    *,
    feature: str,
    threshold: Any,
    threshold_kind: str,
    threshold_bands: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a raw-value-free descriptor for one split threshold."""

    if threshold_kind != "numeric":
        # Categorical values/codes themselves are deliberately excluded from a
        # recurrent signature.  Split-set cardinality preserves useful shape.
        cardinality = max(1, len([part for part in str(threshold).split("||") if part]))
        return {
            "threshold_band_state": "categorical_split",
            "threshold_band_index": int(cardinality),
            "threshold_band_count": int(cardinality),
        }
    entry = threshold_bands.get(feature, {})
    state = str(entry.get("state", "numeric_low_support"))
    cuts = np.asarray(entry.get("cuts", []), dtype=np.float64)
    if state != "numeric_quantile" or len(cuts) == 0:
        return {
            "threshold_band_state": state,
            "threshold_band_index": -1,
            "threshold_band_count": 0,
        }
    value = float(threshold)
    return {
        "threshold_band_state": "numeric_quantile",
        "threshold_band_index": int(np.searchsorted(cuts, value, side="right")),
        "threshold_band_count": int(len(cuts) + 1),
    }


def _banded_structural_rule_path(
    raw_path: Sequence[Mapping[str, Any]],
    *,
    threshold_bands: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Remove raw thresholds while retaining branch/rule geometry for G2."""

    structural: list[dict[str, Any]] = []
    for step in raw_path:
        threshold_kind = str(step["threshold_kind"])
        band = _threshold_band_descriptor(
            feature=str(step["feature"]),
            threshold=step.get("threshold"),
            threshold_kind=threshold_kind,
            threshold_bands=threshold_bands,
        )
        structural.append(
            {
                "feature": str(step["feature"]),
                "decision_type": str(step["decision_type"]),
                "threshold_kind": threshold_kind,
                **band,
                "branch": str(step["branch"]),
            }
        )
    return structural


def _threshold_band_state_fingerprint(state: Mapping[str, Mapping[str, Any]]) -> str:
    """Fingerprint the train-only band state for the fold provenance record."""

    return _rule_hash({str(feature): dict(value) for feature, value in sorted(state.items())})


def _model_trees_per_iteration(model: Any) -> int:
    """Return the LightGBM raw-tree stride (one for binary, C for multiclass)."""

    booster = getattr(model, "booster_", None)
    if booster is None:
        raise StrictOOFBaseReasoningError("every supplied model needs a LightGBM booster_")
    try:
        trees_per_iteration = int(booster.num_model_per_iteration())
    except Exception as exc:  # pragma: no cover - defensive LightGBM compatibility
        raise StrictOOFBaseReasoningError(
            "could not determine LightGBM trees per iteration"
        ) from exc
    if trees_per_iteration <= 0:
        raise StrictOOFBaseReasoningError("LightGBM trees per iteration must be positive")
    return trees_per_iteration


def _select_head_tree_indices(
    *,
    tree_total: int,
    trees_per_iteration: int,
    class_index: int,
    max_trees: int,
) -> list[int]:
    """Select one head's class-strided raw tree positions.

    Multiclass LightGBM stores class trees as ``[iter0-class0,
    iter0-class1, ...]``.  Selecting the first N raw columns for every head
    would silently materialise the wrong rules/leaves for clear/adverse/weak.
    """

    if trees_per_iteration > 1:
        if class_index < 0 or class_index >= trees_per_iteration:
            raise StrictOOFBaseReasoningError(
                "requested class_index is unavailable in the LightGBM tree stride"
            )
        first_tree = int(class_index)
    else:
        # Binary LightGBM has one raw tree per iteration; its sole tree stream
        # represents the positive class selected elsewhere in the score path.
        first_tree = 0
    return list(range(first_tree, int(tree_total), int(trees_per_iteration)))[: int(max_trees)]


def _leaf_rules_for_model(
    model: Any,
    *,
    feature_names: Sequence[str],
    head_name: str,
    side_name: str,
    fold_id: str,
    model_slot: int,
    class_index: int,
    max_trees: int,
    buckets: int,
    threshold_bands: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[tuple[int, int], _LeafRule], list[dict[str, Any]], list[int], int]:
    """Return local per-tree rules without exporting a raw leaf index."""

    booster = getattr(model, "booster_", None)
    if booster is None:
        raise StrictOOFBaseReasoningError("every supplied model needs a LightGBM booster_")
    try:
        tree_info = booster.dump_model().get("tree_info", [])
    except Exception as exc:  # pragma: no cover - defensive third-party failure
        raise StrictOOFBaseReasoningError("could not read LightGBM tree dump") from exc
    model_hash = _model_hash(model)
    selected = list(map(str, feature_names))
    rules: dict[tuple[int, int], _LeafRule] = {}
    catalog: list[dict[str, Any]] = []

    def walk(
        node: Mapping[str, Any],
        tree_index: int,
        head_tree_slot: int,
        path: list[dict[str, str]],
    ) -> None:
        if "leaf_index" in node:
            local_leaf = int(node["leaf_index"])
            try:
                leaf_value = float(node["leaf_value"])
            except (KeyError, TypeError, ValueError) as exc:
                raise StrictOOFBaseReasoningError(
                    "LightGBM leaf rule lacks a finite additive leaf value"
                ) from exc
            if not np.isfinite(leaf_value):
                raise StrictOOFBaseReasoningError("LightGBM leaf rule has a non-finite additive leaf value")
            feature_path = [step["feature"] for step in path]
            rule_path_json = json.dumps(path, separators=(",", ":"), ensure_ascii=True)
            structural_path = _banded_structural_rule_path(
                path,
                threshold_bands=threshold_bands,
            )
            rule_structural_path_json = json.dumps(
                structural_path,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            rule_threshold_band_json = json.dumps(
                [
                    {
                        "threshold_band_state": step["threshold_band_state"],
                        "threshold_band_index": step["threshold_band_index"],
                        "threshold_band_count": step["threshold_band_count"],
                    }
                    for step in structural_path
                ],
                separators=(",", ":"),
                ensure_ascii=True,
            )
            rule_signature = _rule_hash(structural_path)
            rule_raw_signature = _rule_hash(path)
            feature_signature = _rule_hash(feature_path)
            right_fraction = (
                float(np.mean([step["branch"] == "right" for step in path]))
                if path
                else 0.0
            )
            token = _token_u64(
                f"{head_name}|{side_name}|{fold_id}|{model_hash}|{model_slot}|{tree_index}|{local_leaf}"
            )
            rule = _LeafRule(
                leaf_token=token,
                rule_signature=rule_signature,
                rule_raw_signature=rule_raw_signature,
                rule_structural_path_json=rule_structural_path_json,
                rule_threshold_band_json=rule_threshold_band_json,
                rule_feature_signature=feature_signature,
                rule_path_json=rule_path_json,
                path_depth=len(path),
                unique_feature_count=len(set(feature_path)),
                numeric_threshold_count=sum(step["threshold_kind"] == "numeric" for step in path),
                right_branch_fraction=right_fraction,
                signature_bucket=int(_token_u64(rule_signature) % np.uint64(buckets)),
                tree_leaf_value=leaf_value,
            )
            rules[(tree_index, local_leaf)] = rule
            catalog.append(
                {
                    "head_name": head_name,
                    "side_name": side_name,
                    "fold_id": fold_id,
                    "model_slot": int(model_slot),
                    "model_hash": model_hash,
                    "tree_index": int(tree_index),
                    "head_tree_slot": int(head_tree_slot),
                    "trees_per_iteration": int(trees_per_iteration),
                    "leaf_token": token,
                    "rule_signature": rule_signature,
                    "rule_raw_signature": rule_raw_signature,
                    "rule_structural_path_json": rule_structural_path_json,
                    "rule_threshold_band_json": rule_threshold_band_json,
                    "rule_feature_signature": feature_signature,
                    "rule_path_json": rule_path_json,
                    "path_depth": int(rule.path_depth),
                    "unique_feature_count": int(rule.unique_feature_count),
                    "numeric_threshold_count": int(rule.numeric_threshold_count),
                    "right_branch_fraction": float(rule.right_branch_fraction),
                    "signature_bucket": int(rule.signature_bucket),
                    # This is the model-native additive tree output for the
                    # local leaf.  It is intentionally catalog-scoped: a
                    # downstream causal state builder may join it only to the
                    # matching same-artifact assignment, collapse by the
                    # structural rule signature, and then discard the token.
                    # It is neither a raw leaf identifier nor a cross-fold
                    # leaf alignment key.
                    "tree_leaf_value": float(rule.tree_leaf_value),
                }
            )
            return
        feature_idx = int(node.get("split_feature", -1))
        feature = selected[feature_idx] if 0 <= feature_idx < len(selected) else f"feature_{feature_idx}"
        threshold = node.get("threshold", "")
        decision_type = str(node.get("decision_type", "<="))
        threshold_kind = "numeric" if _is_finite_number(threshold) else "categorical"
        common = {
            "feature": feature,
            "decision_type": decision_type,
            # Preserve the LightGBM dump's complete threshold representation in
            # the audit path.  The structural signature never consumes it.
            "threshold": str(threshold),
            "threshold_kind": threshold_kind,
        }
        left = node.get("left_child")
        if isinstance(left, Mapping):
            walk(
                left,
                tree_index,
                head_tree_slot,
                [*path, {**common, "branch": "left"}],
            )
        right = node.get("right_child")
        if isinstance(right, Mapping):
            walk(
                right,
                tree_index,
                head_tree_slot,
                [*path, {**common, "branch": "right"}],
            )

    trees_per_iteration = _model_trees_per_iteration(model)
    selected_tree_indices = _select_head_tree_indices(
        tree_total=len(tree_info),
        trees_per_iteration=trees_per_iteration,
        class_index=class_index,
        max_trees=max_trees,
    )
    for head_tree_slot, tree_index in enumerate(selected_tree_indices):
        tree = tree_info[tree_index]
        root = tree.get("tree_structure", {}) if isinstance(tree, Mapping) else {}
        if isinstance(root, Mapping):
            walk(root, int(tree_index), int(head_tree_slot), [])
    return rules, catalog, selected_tree_indices, trees_per_iteration


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _leaf_ids(
    model: Any,
    matrix: pd.DataFrame,
    *,
    selected_tree_indices: Sequence[int],
    trees_per_iteration: int,
) -> np.ndarray:
    if not selected_tree_indices:
        raise StrictOOFBaseReasoningError("no selected LightGBM trees for requested head")
    max_raw_tree = int(max(selected_tree_indices))
    requested_iterations = int(np.ceil((max_raw_tree + 1) / int(trees_per_iteration)))
    try:
        leaf = np.asarray(
            model.predict(
                matrix,
                pred_leaf=True,
                num_iteration=requested_iterations,
            ),
            dtype=np.int64,
        )
    except TypeError:
        leaf = np.asarray(model.predict(matrix, pred_leaf=True), dtype=np.int64)
    except Exception as exc:  # pragma: no cover - third-party failure path
        raise StrictOOFBaseReasoningError("LightGBM leaf prediction failed") from exc
    if leaf.ndim == 1:
        leaf = leaf.reshape(len(matrix), 1)
    if leaf.shape[0] != len(matrix):
        raise StrictOOFBaseReasoningError("LightGBM leaf prediction is not row-aligned")
    if max_raw_tree >= leaf.shape[1]:
        raise StrictOOFBaseReasoningError(
            "LightGBM leaf prediction does not contain the requested head trees"
        )
    return leaf[:, list(map(int, selected_tree_indices))]


def _append_summary_features(
    feature_data: dict[str, np.ndarray],
    name: str,
    matrix: np.ndarray,
) -> None:
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return
    value = np.asarray(matrix, dtype=np.float32)
    feature_data[f"{FEATURE_PREFIX}{name}_mean"] = np.mean(value, axis=1).astype(np.float32)
    feature_data[f"{FEATURE_PREFIX}{name}_p10"] = np.percentile(value, 10.0, axis=1).astype(np.float32)
    feature_data[f"{FEATURE_PREFIX}{name}_p90"] = np.percentile(value, 90.0, axis=1).astype(np.float32)
    feature_data[f"{FEATURE_PREFIX}{name}_min"] = np.min(value, axis=1).astype(np.float32)
    feature_data[f"{FEATURE_PREFIX}{name}_max"] = np.max(value, axis=1).astype(np.float32)
    feature_data[f"{FEATURE_PREFIX}{name}_std"] = np.std(value, axis=1).astype(np.float32)


def _model_class_count(model: Any) -> int:
    """Return the fitted classifier's output count without inferring a head.

    A binary LightGBM booster has one tree-output per iteration but two
    probability classes, hence the ``max(..., 2)`` fallback below.  Multiclass
    sklearn estimators expose ``classes_`` (preferred) or ``n_classes_``.
    """

    classes = getattr(model, "classes_", None)
    if classes is not None:
        try:
            count = int(len(classes))
        except TypeError:
            count = 0
        if count >= 2:
            return count
    declared = getattr(model, "n_classes_", None)
    if declared is not None:
        try:
            count = int(declared)
        except (TypeError, ValueError):
            count = 0
        if count >= 2:
            return count
    booster = getattr(model, "booster_", None)
    if booster is not None:
        try:
            count = int(booster.num_model_per_iteration())
        except Exception:  # pragma: no cover - LightGBM version defensive path
            count = 0
        if count > 1:
            return count
        if count == 1:
            return 2
    raise StrictOOFBaseReasoningError(
        "could not determine the fitted classifier's class count; "
        "base reasoning requires a classifier with explicit outputs"
    )


def _resolve_class_index(
    models: Sequence[Any],
    *,
    head_name: str,
    class_index: int | None,
    head_class_map: Mapping[str, int] | None,
) -> tuple[int, list[int], str]:
    """Resolve a semantic head to one model output, never guessing multiclass."""

    mapped: int | None = None
    if head_class_map is not None:
        if not isinstance(head_class_map, Mapping):
            raise StrictOOFBaseReasoningError("head_class_map must be a mapping")
        if head_name in head_class_map:
            raw = head_class_map[head_name]
            if isinstance(raw, bool):
                raise StrictOOFBaseReasoningError("head_class_map indices must be integers")
            try:
                mapped = int(raw)
            except (TypeError, ValueError) as exc:
                raise StrictOOFBaseReasoningError(
                    "head_class_map indices must be integers"
                ) from exc
            if mapped != raw:
                raise StrictOOFBaseReasoningError("head_class_map indices must be integers")
    explicit: int | None = None
    if class_index is not None:
        if isinstance(class_index, bool):
            raise StrictOOFBaseReasoningError("class_index must be an integer")
        explicit = int(class_index)
        if explicit != class_index:
            raise StrictOOFBaseReasoningError("class_index must be an integer")
    if explicit is not None and mapped is not None and explicit != mapped:
        raise StrictOOFBaseReasoningError(
            "class_index conflicts with head_class_map for the requested head"
        )
    counts = [_model_class_count(model) for model in models]
    if len(set(counts)) != 1:
        raise StrictOOFBaseReasoningError(
            f"fitted base models disagree on class count: {counts}"
        )
    count = counts[0]
    resolved = explicit if explicit is not None else mapped
    if resolved is None:
        if count > 2:
            raise StrictOOFBaseReasoningError(
                "multiclass base reasoning requires an explicit class_index or "
                "a head_class_map entry for the requested head; class 1 is never "
                "assumed for multiclass outputs"
            )
        resolved = 1
        source = "binary_positive_class_default"
    else:
        source = "explicit_class_index" if explicit is not None else "head_class_map"
    if resolved < 0 or resolved >= count:
        raise StrictOOFBaseReasoningError(
            f"class_index={resolved} is outside fitted model class range [0, {count})"
        )
    return int(resolved), counts, source


def _single_model_class_prediction(
    model: Any,
    matrix: pd.DataFrame,
    *,
    class_index: int,
) -> np.ndarray:
    rows = len(matrix)
    try:
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(matrix), dtype=np.float32)
        else:
            proba = np.asarray(model.predict(matrix), dtype=np.float32)
    except Exception as exc:  # pragma: no cover - defensive third-party failure
        raise StrictOOFBaseReasoningError("base prediction failed") from exc
    if proba.ndim == 2:
        if proba.shape[0] != rows or class_index >= proba.shape[1]:
            raise StrictOOFBaseReasoningError(
                "base prediction output does not contain the requested class index"
            )
        prediction = proba[:, class_index]
    elif proba.ndim == 1 and class_index == 1:
        # LightGBM's binary one-dimensional output is the positive-class score.
        prediction = proba
    else:
        raise StrictOOFBaseReasoningError(
            "base prediction must expose a class axis for the requested head"
        )
    if len(prediction) != rows or not np.isfinite(prediction).all():
        raise StrictOOFBaseReasoningError("base prediction must be finite and row-aligned")
    return prediction.astype(np.float32, copy=False)


def _predict_base(
    models: Sequence[Any],
    matrix: pd.DataFrame,
    *,
    class_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = len(matrix)
    values: list[np.ndarray] = []
    for model in models:
        values.append(
            _single_model_class_prediction(model, matrix, class_index=class_index)
        )
    if not values:
        raise StrictOOFBaseReasoningError("at least one fitted base model is required")
    matrix_values = np.vstack(values).astype(np.float32, copy=False)
    return matrix_values.mean(axis=0).astype(np.float32), matrix_values.std(axis=0).astype(np.float32)


def _extract_class_contributions(
    raw: Any,
    *,
    rows: int,
    n_features: int,
    class_index: int,
) -> np.ndarray:
    """Select one LightGBM ``pred_contrib`` class block explicitly.

    LightGBM emits either ``[rows, classes * (features + bias)]`` or a
    three-dimensional equivalent for multiclass models.  A one-block binary
    output represents its positive class.  We deliberately reject the
    impossible class-0 contribution reconstruction rather than smuggling in a
    contribution bundle for another head.
    """

    contrib = np.asarray(raw, dtype=np.float32)
    block = int(n_features) + 1
    if contrib.ndim == 1:
        if rows != 1:
            raise StrictOOFBaseReasoningError("pred_contrib is not row-aligned")
        contrib = contrib.reshape(1, -1)
    if contrib.ndim == 3:
        if contrib.shape[0] == rows and contrib.shape[2] == block:
            if class_index >= contrib.shape[1]:
                raise StrictOOFBaseReasoningError(
                    "pred_contrib does not contain the requested multiclass output"
                )
            return contrib[:, class_index, :n_features]
        if contrib.shape[1] == rows and contrib.shape[2] == block:
            if class_index >= contrib.shape[0]:
                raise StrictOOFBaseReasoningError(
                    "pred_contrib does not contain the requested multiclass output"
                )
            return contrib[class_index, :, :n_features]
        raise StrictOOFBaseReasoningError("unsupported 3D LightGBM pred_contrib layout")
    if contrib.ndim != 2 or contrib.shape[0] != rows:
        raise StrictOOFBaseReasoningError("pred_contrib is not a row-aligned matrix")
    if contrib.shape[1] == block:
        if class_index != 1:
            raise StrictOOFBaseReasoningError(
                "a single-block binary pred_contrib output only represents class 1"
            )
        return contrib[:, :n_features]
    if contrib.shape[1] % block == 0:
        class_count = contrib.shape[1] // block
        if class_index >= class_count:
            raise StrictOOFBaseReasoningError(
                "pred_contrib does not contain the requested multiclass output"
            )
        return contrib.reshape(rows, class_count, block)[:, class_index, :n_features]
    raise StrictOOFBaseReasoningError("unsupported LightGBM pred_contrib column layout")


def _extract_all_class_contributions(
    raw: Any,
    *,
    rows: int,
    n_features: int,
    class_count: int,
) -> np.ndarray:
    """Return every multiclass LightGBM contribution block explicitly.

    This is intentionally not a generic binary shortcut.  The shared cache is
    only valid when a fitted multiclass model exposes every semantic class, so
    a one-block output is rejected rather than reconstructed or duplicated.
    """

    contrib = np.asarray(raw, dtype=np.float32)
    block = int(n_features) + 1
    if contrib.ndim == 1:
        if rows != 1:
            raise StrictOOFBaseReasoningError("pred_contrib is not row-aligned")
        contrib = contrib.reshape(1, -1)
    if contrib.ndim == 3:
        if contrib.shape[0] == rows and contrib.shape[1] == class_count and contrib.shape[2] == block:
            return contrib[:, :, :n_features]
        if contrib.shape[0] == class_count and contrib.shape[1] == rows and contrib.shape[2] == block:
            return np.moveaxis(contrib[:, :, :n_features], 0, 1)
        raise StrictOOFBaseReasoningError("unsupported multiclass pred_contrib 3D layout")
    if contrib.ndim != 2 or contrib.shape[0] != rows:
        raise StrictOOFBaseReasoningError("pred_contrib is not a row-aligned matrix")
    if contrib.shape[1] != int(class_count) * block:
        raise StrictOOFBaseReasoningError(
            "shared contribution cache requires every multiclass pred_contrib block"
        )
    return contrib.reshape(rows, int(class_count), block)[:, :, :n_features]


def _matrix_fingerprint(matrix: pd.DataFrame) -> str:
    """Hash an exact numeric matrix without exporting it to any artifact."""

    digest = hashlib.sha256()
    digest.update(json.dumps(list(map(str, matrix.columns)), separators=(",", ":")).encode("utf-8"))
    digest.update(json.dumps([str(dtype) for dtype in matrix.dtypes], separators=(",", ":")).encode("utf-8"))
    digest.update(str(matrix.shape).encode("ascii"))
    try:
        row_hashes = pd.util.hash_pandas_object(matrix, index=True, categorize=False)
    except Exception as exc:  # pragma: no cover - defensive pandas compatibility
        raise StrictOOFBaseReasoningError("could not fingerprint contribution-cache matrix") from exc
    digest.update(np.asarray(row_hashes.to_numpy(dtype=np.uint64, copy=False), dtype="<u8").tobytes())
    return digest.hexdigest()


def _accumulate_all_class_contributions(
    model: Any,
    matrix: pd.DataFrame,
    *,
    class_count: int,
    batch_rows: int,
    destination: np.ndarray,
) -> None:
    """Accumulate one model in bounded row batches into ``[class,row,feature]``."""

    expected = (int(class_count), len(matrix), matrix.shape[1])
    if destination.shape != expected or destination.dtype != np.float32:
        raise StrictOOFBaseReasoningError("contribution-cache destination shape is invalid")
    step = max(1, int(batch_rows))
    for start in range(0, len(matrix), step):
        stop = min(start + step, len(matrix))
        part = matrix.iloc[start:stop]
        try:
            raw = model.predict(part, pred_contrib=True)
        except Exception as exc:  # pragma: no cover - third-party failure path
            raise StrictOOFBaseReasoningError("LightGBM pred_contrib materialisation failed") from exc
        values = _extract_all_class_contributions(
            raw,
            rows=len(part),
            n_features=matrix.shape[1],
            class_count=class_count,
        )
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        destination[:, start:stop, :] += np.moveaxis(values, 1, 0)


def _bounded_contribution_batch_rows(
    *,
    requested_rows: int,
    class_count: int,
    n_features: int,
    max_working_bytes: int | None,
) -> int:
    """Return a bounded row batch for raw multiclass ``pred_contrib`` output.

    One LightGBM batch can temporarily contain its raw (usually float64)
    contribution payload, a float32 class reshape, and a finite-cleaned
    float32 payload.  Sixteen bytes per class/feature/row is deliberately
    conservative for that transient sequence.  It is only imposed for spill
    mode: the established in-memory cache retains its previous batch behaviour.
    """

    requested = max(1, int(requested_rows))
    if max_working_bytes is None:
        return requested
    budget = int(max_working_bytes)
    if budget <= 0:
        raise StrictOOFContributionCacheCapacityError(
            "multiclass contribution spill cache requires a positive working-memory bound"
        )
    bytes_per_row = 16 * int(class_count) * (int(n_features) + 1)
    allowed = max(1, budget // max(bytes_per_row, 1))
    return min(requested, allowed)


def _cache_arrays(
    *,
    class_count: int,
    train_rows: int,
    eval_rows: int,
    feature_count: int,
    retained_bytes: int,
    max_cache_bytes: int,
    spill_directory: str | os.PathLike[str] | None,
    max_spill_bytes: int | None,
) -> tuple[np.ndarray, np.ndarray, str, tuple[Path, ...], Path | None]:
    """Allocate exact all-class arrays in RAM or a bounded temporary memmap.

    The spill allocation is intentionally all-or-nothing and occurs only after
    both the configured disk budget and currently available filesystem space
    have been checked.  It is not a persisted artifact: the cache owner removes
    the unique scratch directory in ``release``.  A capacity miss remains a
    normal opportunity to use the established uncached implementation; actual
    filesystem failures deliberately propagate instead of silently changing
    the strict reasoning semantics.
    """

    shape_train = (int(class_count), int(train_rows), int(feature_count))
    shape_eval = (int(class_count), int(eval_rows), int(feature_count))
    if int(max_cache_bytes) > 0 and int(retained_bytes) <= int(max_cache_bytes):
        return (
            np.zeros(shape_train, dtype=np.float32),
            np.zeros(shape_eval, dtype=np.float32),
            "in_memory",
            (),
            None,
        )
    if spill_directory is None or max_spill_bytes is None or int(max_spill_bytes) <= 0:
        raise StrictOOFContributionCacheCapacityError(
            "multiclass contribution cache requires "
            f"{retained_bytes} retained bytes, exceeding configured memory bound "
            f"{int(max_cache_bytes)} and no bounded spill cache is available"
        )
    if int(retained_bytes) > int(max_spill_bytes):
        raise StrictOOFContributionCacheCapacityError(
            "multiclass contribution spill cache requires "
            f"{retained_bytes} bytes, exceeding configured spill bound {int(max_spill_bytes)}"
        )
    root = Path(spill_directory)
    root.mkdir(parents=True, exist_ok=True)
    available = int(shutil.disk_usage(root).free)
    if available < int(retained_bytes):
        raise StrictOOFContributionCacheCapacityError(
            "multiclass contribution spill cache requires "
            f"{retained_bytes} bytes but only {available} bytes are free"
        )
    scratch = Path(tempfile.mkdtemp(prefix=".strict_oof_contrib_", dir=str(root)))
    train_path = scratch / "train_contributions.f32.mmap"
    eval_path = scratch / "eval_contributions.f32.mmap"
    try:
        train = np.memmap(train_path, mode="w+", dtype=np.float32, shape=shape_train)
        evaluate = np.memmap(eval_path, mode="w+", dtype=np.float32, shape=shape_eval)
        # New maps are normally zero-filled by the OS.  This explicit fill is
        # required for the ensemble accumulation invariant and remains bounded
        # by the memmap instead of allocating a second dense array.
        train.fill(0.0)
        evaluate.fill(0.0)
        return train, evaluate, "disk_mmap", (train_path, eval_path), scratch
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise


def _arrays_all_finite(values: np.ndarray, *, batch_rows: int) -> bool:
    """Check a dense/memmapped contribution tensor without a full boolean copy."""

    if values.ndim != 3:
        return False
    step = max(1, int(batch_rows))
    for start in range(0, values.shape[1], step):
        stop = min(start + step, values.shape[1])
        if not np.isfinite(values[:, start:stop, :]).all():
            return False
    return True


def build_strict_oof_multiclass_contribution_cache(
    models: Sequence[Any],
    train_matrix: pd.DataFrame,
    eval_matrix: pd.DataFrame,
    *,
    batch_rows: int,
    max_cache_bytes: int,
    spill_directory: str | os.PathLike[str] | None = None,
    max_spill_bytes: int | None = None,
    spill_max_working_bytes: int | None = None,
) -> StrictOOFMulticlassContributionCache:
    """Compute all multiclass G3 contributions once for one strict OOF fold.

    The retained arrays are bounded before allocation.  If the exact all-class
    cache does not fit RAM, callers may explicitly opt into a separately
    bounded temporary memmap.  Both forms use the same ``[class,row,feature]``
    values and explicit class slices; the spill merely avoids re-running
    LightGBM's all-class contribution pass once for every semantic head.  A
    capacity exception remains a normal optimisation miss: callers can use the
    unchanged uncached path.
    """

    train, evaluate = _require_matrix_pair(train_matrix, eval_matrix)
    fitted = list(models)
    if not fitted:
        raise StrictOOFBaseReasoningError("at least one fitted base model is required")
    class_counts = [_model_class_count(model) for model in fitted]
    if len(set(class_counts)) != 1:
        raise StrictOOFBaseReasoningError(
            f"fitted base models disagree on class count: {class_counts}"
        )
    class_count = int(class_counts[0])
    if class_count <= 2:
        raise StrictOOFBaseReasoningError(
            "shared contribution cache is reserved for explicit multiclass heads"
        )
    retained_bytes = (
        int(class_count)
        * (len(train) + len(evaluate))
        * train.shape[1]
        * np.dtype(np.float32).itemsize
    )
    train_sum, eval_sum, storage_mode, backing_paths, backing_directory = _cache_arrays(
        class_count=class_count,
        train_rows=len(train),
        eval_rows=len(evaluate),
        feature_count=train.shape[1],
        retained_bytes=retained_bytes,
        max_cache_bytes=max_cache_bytes,
        spill_directory=spill_directory,
        max_spill_bytes=max_spill_bytes,
    )
    try:
        effective_batch_rows = _bounded_contribution_batch_rows(
            requested_rows=batch_rows,
            class_count=class_count,
            n_features=train.shape[1],
            max_working_bytes=(
                spill_max_working_bytes if storage_mode == "disk_mmap" else None
            ),
        )
        for model in fitted:
            _accumulate_all_class_contributions(
                model,
                train,
                class_count=class_count,
                batch_rows=effective_batch_rows,
                destination=train_sum,
            )
            _accumulate_all_class_contributions(
                model,
                evaluate,
                class_count=class_count,
                batch_rows=effective_batch_rows,
                destination=eval_sum,
            )
        divisor = np.float32(len(fitted))
        train_sum /= divisor
        eval_sum /= divisor
        if not _arrays_all_finite(train_sum, batch_rows=effective_batch_rows) or not _arrays_all_finite(
            eval_sum, batch_rows=effective_batch_rows
        ):
            raise StrictOOFBaseReasoningError("shared contribution cache contains non-finite values")
        if isinstance(train_sum, np.memmap):
            train_sum.flush()
        if isinstance(eval_sum, np.memmap):
            eval_sum.flush()
        train_sum.setflags(write=False)
        eval_sum.setflags(write=False)
        return StrictOOFMulticlassContributionCache(
            train_contributions=train_sum,
            eval_contributions=eval_sum,
            class_count=class_count,
            feature_names=tuple(map(str, train.columns)),
            model_hashes=tuple(_model_hash(model) for model in fitted),
            train_matrix_fingerprint=_matrix_fingerprint(train),
            eval_matrix_fingerprint=_matrix_fingerprint(evaluate),
            retained_bytes=retained_bytes,
            storage_mode=storage_mode,
            backing_paths=backing_paths,
            backing_directory=backing_directory,
        )
    except Exception:
        failed = StrictOOFMulticlassContributionCache(
            train_contributions=train_sum,
            eval_contributions=eval_sum,
            class_count=class_count,
            feature_names=(),
            model_hashes=(),
            train_matrix_fingerprint="",
            eval_matrix_fingerprint="",
            retained_bytes=retained_bytes,
            storage_mode=storage_mode,
            backing_paths=backing_paths,
            backing_directory=backing_directory,
        )
        failed.release()
        raise


def _contributions_from_cache(
    cache: StrictOOFMulticlassContributionCache,
    *,
    models: Sequence[Any],
    train_matrix: pd.DataFrame,
    eval_matrix: pd.DataFrame,
    class_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate an exact fold cache and return one explicit semantic slice."""

    if not isinstance(cache, StrictOOFMulticlassContributionCache):
        raise StrictOOFBaseReasoningError("contribution_cache has an invalid type")
    if cache.closed:
        raise StrictOOFBaseReasoningError("contribution_cache has already been released")
    feature_names = tuple(map(str, train_matrix.columns))
    if cache.feature_names != feature_names:
        raise StrictOOFBaseReasoningError("contribution_cache feature contract does not match this fold")
    expected_hashes = tuple(_model_hash(model) for model in models)
    if cache.model_hashes != expected_hashes:
        raise StrictOOFBaseReasoningError("contribution_cache model ensemble does not match this fold")
    if cache.train_matrix_fingerprint != _matrix_fingerprint(train_matrix):
        raise StrictOOFBaseReasoningError("contribution_cache train matrix does not match this fold")
    if cache.eval_matrix_fingerprint != _matrix_fingerprint(eval_matrix):
        raise StrictOOFBaseReasoningError("contribution_cache evaluation matrix does not match this fold")
    shape_train = (int(cache.class_count), len(train_matrix), train_matrix.shape[1])
    shape_eval = (int(cache.class_count), len(eval_matrix), eval_matrix.shape[1])
    if cache.train_contributions.shape != shape_train or cache.eval_contributions.shape != shape_eval:
        raise StrictOOFBaseReasoningError("contribution_cache array shapes do not match this fold")
    if class_index < 0 or class_index >= int(cache.class_count):
        raise StrictOOFBaseReasoningError("contribution_cache lacks the explicitly selected class")
    train_values = cache.train_contributions[int(class_index)]
    eval_values = cache.eval_contributions[int(class_index)]
    if not _arrays_all_finite(
        train_values[None, :, :], batch_rows=50_000
    ) or not _arrays_all_finite(eval_values[None, :, :], batch_rows=50_000):
        raise StrictOOFBaseReasoningError("contribution_cache selected class is non-finite")
    return train_values, eval_values


def _predict_contrib_for_class(
    model: Any,
    matrix: pd.DataFrame,
    *,
    class_index: int,
    batch_rows: int,
) -> np.ndarray:
    """Predict one explicit contribution head in bounded row batches."""

    parts: list[np.ndarray] = []
    for start in range(0, len(matrix), max(1, int(batch_rows))):
        part = matrix.iloc[start : start + max(1, int(batch_rows))]
        try:
            raw = model.predict(part, pred_contrib=True)
        except Exception as exc:  # pragma: no cover - defensive third-party failure
            raise StrictOOFBaseReasoningError("LightGBM pred_contrib materialisation failed") from exc
        values = _extract_class_contributions(
            raw,
            rows=len(part),
            n_features=matrix.shape[1],
            class_index=class_index,
        )
        parts.append(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0))
    if not parts:
        raise StrictOOFBaseReasoningError("pred_contrib received an empty matrix")
    return np.vstack(parts).astype(np.float32, copy=False)


def _mean_contrib_matrix_for_class(
    models: Sequence[Any],
    matrix: pd.DataFrame,
    *,
    class_index: int,
    batch_rows: int,
) -> np.ndarray:
    mats = [
        _predict_contrib_for_class(
            model,
            matrix,
            class_index=class_index,
            batch_rows=batch_rows,
        )
        for model in models
    ]
    if not mats:
        raise StrictOOFBaseReasoningError("at least one fitted base model is required")
    shape = (len(matrix), matrix.shape[1])
    if any(mat.shape != shape for mat in mats):
        raise StrictOOFBaseReasoningError(
            "contribution matrix does not match the frozen base feature contract"
        )
    return np.mean(np.stack(mats, axis=0), axis=0).astype(np.float32)


def _coerce_prediction(values: Sequence[float] | np.ndarray | None, *, fallback: np.ndarray) -> np.ndarray:
    if values is None:
        return fallback
    out = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(out) != len(fallback) or not np.isfinite(out).all():
        raise StrictOOFBaseReasoningError("eval_predictions must be finite and aligned")
    return out


def _labels_frame(identity: pd.DataFrame, labels: pd.DataFrame | Mapping[str, Sequence[Any]] | None) -> pd.DataFrame | None:
    if labels is None:
        return None
    raw = pd.DataFrame(labels).reset_index(drop=True)
    if len(raw) != len(identity):
        raise StrictOOFBaseReasoningError("eval_labels must be aligned to eval_matrix")
    if raw.columns.duplicated().any():
        raise StrictOOFBaseReasoningError("eval_labels has duplicate columns")
    result = identity.copy()
    for column in raw.columns:
        name = f"label__{str(column).strip()}"
        if name in result:
            raise StrictOOFBaseReasoningError(f"duplicate label output {name}")
        result[name] = raw[column].to_numpy(copy=False)
    return result


def _sha256_frame(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(map(str, frame.columns))).encode("utf-8"))
    digest.update(str(len(frame)).encode("ascii"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    """Hash large parquet artifacts without loading them into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_manifest(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _safe_manifest(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_manifest(item) for item in value]
    return value


def _write_artifacts(
    destination: Path,
    *,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    labels: pd.DataFrame | None,
    assignments: pd.DataFrame,
    catalog: pd.DataFrame,
    contribution_bundle: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite base-reasoning artifact {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        outputs: dict[str, str] = {}
        tables: list[tuple[str, pd.DataFrame]] = [
            ("base_reasoning_features.parquet", features),
            ("base_reasoning_predictions.parquet", predictions),
            ("leaf_assignments.parquet", assignments),
            ("leaf_rule_catalog.parquet", catalog),
            ("contribution_bundle.parquet", contribution_bundle),
        ]
        if labels is not None:
            tables.append(("base_reasoning_labels.parquet", labels))
        for name, frame in tables:
            path = temporary / name
            frame.to_parquet(path, index=False, compression="zstd")
            outputs[name] = _sha256_file(path)
        stored = {**manifest, "outputs": outputs}
        (temporary / "base_reasoning_manifest.json").write_text(
            json.dumps(_safe_manifest(stored), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        return destination
    except Exception:
        for path in temporary.glob("*"):
            path.unlink(missing_ok=True)
        temporary.rmdir()
        raise


def materialize_strict_oof_base_reasoning(
    models: Sequence[Any],
    train_matrix: pd.DataFrame,
    eval_matrix: pd.DataFrame,
    *,
    head_name: str,
    side_name: str,
    fold_id: str | int,
    train_timestamps: Sequence[object] | pd.Series,
    eval_timestamps: Sequence[object] | pd.Series,
    eval_identity: pd.DataFrame,
    eval_predictions: Sequence[float] | np.ndarray | None = None,
    eval_labels: pd.DataFrame | Mapping[str, Sequence[Any]] | None = None,
    train_targets: Sequence[float] | np.ndarray | None = None,
    class_index: int | None = None,
    head_class_map: Mapping[str, int] | None = None,
    contribution_cache: StrictOOFMulticlassContributionCache | None = None,
    artifact_dir: str | os.PathLike[str] | None = None,
    config: StrictOOFBaseReasoningConfig = StrictOOFBaseReasoningConfig(),
) -> StrictOOFBaseReasoningResult:
    """Materialise strict-OOF G1/G2/G3 base reasoning for one fold.

    ``models`` must have been fitted exclusively before the evaluation interval.
    A multiclass base requires ``class_index`` or a matching ``head_class_map``
    entry.  The caller must invoke the materialiser once for each semantic
    clear/adverse/weak head; a multiclass output is never implicitly routed to
    class one.
    ``train_matrix`` is used only for fold-local support and contribution-SVD
    fitting.  ``contribution_cache`` is an optional, exact-model-and-matrix
    validated all-class cache.  It changes only the number of ``pred_contrib``
    calls; the selected semantic class, train-only SVD fit, and returned G3
    values remain the same.  Returned features are evaluation-only and contain
    no labels.
    ``eval_labels`` are retained in a separate audit artifact so downstream
    feature selection cannot accidentally consume them.
    """

    config.validate()
    head = str(head_name).strip()
    side = str(side_name).strip().lower()
    fold = str(fold_id).strip()
    if not head or not fold or side not in {"long", "short"}:
        raise StrictOOFBaseReasoningError("head_name/fold_id must be nonblank and side must be long or short")
    fitted = list(models)
    if not fitted:
        raise StrictOOFBaseReasoningError("at least one fitted base model is required")
    selected_class, class_counts, class_selection_source = _resolve_class_index(
        fitted,
        head_name=head,
        class_index=class_index,
        head_class_map=head_class_map,
    )
    train, evaluate = _require_matrix_pair(train_matrix, eval_matrix)
    train_ts = _as_utc(train_timestamps, role="train_timestamps", expected=len(train))
    eval_ts = _as_utc(eval_timestamps, role="eval_timestamps", expected=len(evaluate))
    if train_ts.max() >= eval_ts.min():
        raise StrictOOFBaseReasoningError(
            "strict OOF violation: maximum train timestamp must precede minimum eval timestamp"
        )
    identity = _normalise_identity(eval_identity, eval_timestamps=eval_ts, side_name=side)
    labels = _labels_frame(identity, eval_labels)
    if train_targets is not None and len(np.asarray(train_targets).reshape(-1)) != len(train):
        raise StrictOOFBaseReasoningError("train_targets must be aligned to train_matrix")
    train_target = (
        np.asarray(train_targets, dtype=np.float32).reshape(-1)
        if train_targets is not None
        else None
    )

    feature_names = list(map(str, train.columns))
    threshold_bands = _fit_rule_threshold_bands(
        train,
        feature_names,
        band_count=int(config.rule_threshold_band_count),
        min_train_rows=int(config.rule_threshold_band_min_train_rows),
    )
    row_count = len(evaluate)
    feature_data: dict[str, np.ndarray] = {}
    assignment = identity.copy()
    catalog_rows: list[dict[str, Any]] = []
    catalog_support: dict[int, dict[str, float]] = {}
    support_values: list[np.ndarray] = []
    surprisal_values: list[np.ndarray] = []
    depth_values: list[np.ndarray] = []
    unique_feature_values: list[np.ndarray] = []
    numeric_threshold_values: list[np.ndarray] = []
    right_branch_values: list[np.ndarray] = []
    signature_bucket = np.zeros((row_count, int(config.rule_signature_bucket_count)), dtype=np.float32)
    tree_selection_by_model: list[dict[str, Any]] = []

    for model_slot, model in enumerate(fitted):
        rules, catalog, selected_tree_indices, trees_per_iteration = _leaf_rules_for_model(
            model,
            feature_names=feature_names,
            head_name=head,
            side_name=side,
            fold_id=fold,
            model_slot=model_slot,
            class_index=selected_class,
            max_trees=int(config.max_trees_per_model),
            buckets=int(config.rule_signature_bucket_count),
            threshold_bands=threshold_bands,
        )
        train_leaf = _leaf_ids(
            model,
            train,
            selected_tree_indices=selected_tree_indices,
            trees_per_iteration=trees_per_iteration,
        )
        eval_leaf = _leaf_ids(
            model,
            evaluate,
            selected_tree_indices=selected_tree_indices,
            trees_per_iteration=trees_per_iteration,
        )
        tree_count = min(len(selected_tree_indices), train_leaf.shape[1], eval_leaf.shape[1])
        if tree_count <= 0:
            raise StrictOOFBaseReasoningError("a supplied model has no materialisable trees")
        tree_selection_by_model.append(
            {
                "model_slot": int(model_slot),
                "trees_per_iteration": int(trees_per_iteration),
                "selected_model_tree_indices": list(
                    map(int, selected_tree_indices[:tree_count])
                ),
            }
        )
        catalog_rows.extend(catalog)
        for head_tree_slot, tree_index in enumerate(selected_tree_indices[:tree_count]):
            train_ids = train_leaf[:, head_tree_slot]
            eval_ids = eval_leaf[:, head_tree_slot]
            unique_ids, counts = np.unique(train_ids, return_counts=True)
            for train_leaf_id, train_count in zip(unique_ids, counts):
                rule = rules.get((tree_index, int(train_leaf_id)))
                if rule is None:
                    raise StrictOOFBaseReasoningError(
                        "training leaf has no local rule catalog entry; model/tree contract changed"
                    )
                train_mask = train_ids == train_leaf_id
                catalog_support[int(rule.leaf_token)] = {
                    "train_leaf_count": float(train_count),
                    "train_leaf_frequency": float(train_count / max(len(train), 1)),
                    "train_target_mean": (
                        float(np.nanmean(train_target[train_mask]))
                        if train_target is not None
                        else float("nan")
                    ),
                }
            position = np.searchsorted(unique_ids, eval_ids)
            position = np.clip(position, 0, max(len(unique_ids) - 1, 0))
            found = unique_ids[position] == eval_ids
            local_counts = np.zeros(row_count, dtype=np.float32)
            local_counts[found] = counts[position[found]]
            local_freq = local_counts / max(float(len(train)), 1.0)
            support_values.append(local_freq)
            surprisal_values.append(-np.log(np.maximum(local_freq, 1e-12)).astype(np.float32))

            tokens = np.zeros(row_count, dtype=np.uint64)
            depth = np.zeros(row_count, dtype=np.float32)
            unique_feature = np.zeros(row_count, dtype=np.float32)
            numeric_threshold = np.zeros(row_count, dtype=np.float32)
            right_branch = np.zeros(row_count, dtype=np.float32)
            for leaf_id in np.unique(eval_ids):
                rule = rules.get((tree_index, int(leaf_id)))
                if rule is None:
                    raise StrictOOFBaseReasoningError(
                        "evaluation leaf has no local rule catalog entry; model/tree contract changed"
                    )
                rows = eval_ids == leaf_id
                tokens[rows] = rule.leaf_token
                depth[rows] = np.float32(rule.path_depth)
                unique_feature[rows] = np.float32(rule.unique_feature_count)
                numeric_threshold[rows] = np.float32(rule.numeric_threshold_count)
                right_branch[rows] = np.float32(rule.right_branch_fraction)
                signature_bucket[rows, rule.signature_bucket] += 1.0
            assignment[
                f"leaf_assignment__model_{model_slot:02d}_head_tree_{head_tree_slot:03d}"
            ] = tokens
            depth_values.append(depth)
            unique_feature_values.append(unique_feature)
            numeric_threshold_values.append(numeric_threshold)
            right_branch_values.append(right_branch)

    if not support_values:
        raise StrictOOFBaseReasoningError("no strict-OOF leaf assignments were materialised")
    support = np.vstack(support_values).T.astype(np.float32)
    surprisal = np.vstack(surprisal_values).T.astype(np.float32)
    _append_summary_features(feature_data, "g1_leaf_train_frequency", support)
    _append_summary_features(feature_data, "g1_leaf_surprisal", surprisal)
    feature_data[f"{FEATURE_PREFIX}g1_leaf_low_frequency_fraction"] = np.mean(
        support <= 0.01, axis=1
    ).astype(np.float32)
    feature_data[f"{FEATURE_PREFIX}g1_leaf_assignment_count"] = np.full(
        row_count, support.shape[1], dtype=np.float32
    )

    _append_summary_features(feature_data, "g2_path_depth", np.vstack(depth_values).T)
    _append_summary_features(
        feature_data, "g2_unique_path_feature_count", np.vstack(unique_feature_values).T
    )
    _append_summary_features(
        feature_data, "g2_numeric_threshold_count", np.vstack(numeric_threshold_values).T
    )
    _append_summary_features(
        feature_data, "g2_right_branch_fraction", np.vstack(right_branch_values).T
    )
    signature_bucket /= np.float32(max(support.shape[1], 1))
    for bucket in range(signature_bucket.shape[1]):
        feature_data[f"{FEATURE_PREFIX}g2_rule_signature_bucket_{bucket:02d}"] = signature_bucket[:, bucket]
    feature_data[f"{FEATURE_PREFIX}g2_rule_signature_bucket_entropy"] = (
        -np.sum(
            np.where(
                signature_bucket > 0.0,
                signature_bucket * np.log(signature_bucket + 1e-12),
                0.0,
            ),
            axis=1,
        )
        / np.log(max(signature_bucket.shape[1], 2))
    ).astype(np.float32)

    if contribution_cache is None:
        contribution_train = _mean_contrib_matrix_for_class(
            fitted,
            train,
            class_index=selected_class,
            batch_rows=int(config.contribution_batch_rows),
        )
        contribution_eval = _mean_contrib_matrix_for_class(
            fitted,
            evaluate,
            class_index=selected_class,
            batch_rows=int(config.contribution_batch_rows),
        )
    else:
        contribution_train, contribution_eval = _contributions_from_cache(
            contribution_cache,
            models=fitted,
            train_matrix=train,
            eval_matrix=evaluate,
            class_index=selected_class,
        )
    if contribution_train.shape != (len(train), len(feature_names)) or contribution_eval.shape != (row_count, len(feature_names)):
        raise StrictOOFBaseReasoningError("contribution matrix does not match the frozen base feature contract")
    summary = contrib_summary_frame(contribution_eval).reset_index(drop=True)
    abs_contrib = np.abs(contribution_eval)
    total_abs = abs_contrib.sum(axis=1) + 1e-12
    sorted_abs = np.sort(abs_contrib, axis=1)[:, ::-1]
    for column in summary.columns:
        feature_data[f"{FEATURE_PREFIX}g3_{column}"] = summary[column].to_numpy(dtype=np.float32)
    feature_data[f"{FEATURE_PREFIX}g3_top1_abs_share"] = (sorted_abs[:, 0] / total_abs).astype(np.float32)
    feature_data[f"{FEATURE_PREFIX}g3_top3_abs_share"] = (
        sorted_abs[:, : min(3, sorted_abs.shape[1])].sum(axis=1) / total_abs
    ).astype(np.float32)
    feature_data[f"{FEATURE_PREFIX}g3_balance"] = np.clip(
        contribution_eval.sum(axis=1) / total_abs, -1.0, 1.0
    ).astype(np.float32)
    feature_data[f"{FEATURE_PREFIX}g3_material_feature_count"] = np.sum(
        abs_contrib / total_abs[:, None] >= 0.01, axis=1
    ).astype(np.float32)

    contribution_state = fit_contrib_archetype_state(
        contribution_train,
        feature_names,
        n_components=int(config.contribution_components),
        random_state=0,
    )
    compact = transform_contrib_archetype_features(contribution_eval, contribution_state)
    contribution_bundle = identity.copy()
    for column in CONTRIB_ARCHETYPE_FEATURE_NAMES:
        renamed = f"{FEATURE_PREFIX}g3_contribution_svd_{column.rsplit('_', 1)[-1]}"
        value = compact[column].to_numpy(dtype=np.float32)
        feature_data[renamed] = value
        contribution_bundle[renamed] = value

    model_prediction, model_prediction_std = _predict_base(
        fitted,
        evaluate,
        class_index=selected_class,
    )
    base_prediction = _coerce_prediction(eval_predictions, fallback=model_prediction)
    predictions = identity.copy()
    predictions["base_prediction"] = base_prediction
    predictions["base_model_prediction"] = model_prediction
    predictions["base_model_prediction_std"] = model_prediction_std
    predictions["head_name"] = head
    predictions["class_index"] = np.int16(selected_class)
    predictions["fold_id"] = fold
    for table in (assignment, contribution_bundle):
        table["head_name"] = head
        table["fold_id"] = fold

    features = identity.copy()
    features["head_name"] = head
    features["fold_id"] = fold
    for name in sorted(feature_data):
        features[name] = np.nan_to_num(
            feature_data[name], nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32, copy=False)
    if labels is not None:
        labels["head_name"] = head
        labels["fold_id"] = fold
    catalog = pd.DataFrame(catalog_rows)
    if not catalog.empty:
        for column in ("train_leaf_count", "train_leaf_frequency", "train_target_mean"):
            catalog[column] = catalog["leaf_token"].map(
                lambda token, key=column: catalog_support.get(int(token), {}).get(key, np.nan)
            ).astype(np.float32)
        catalog = catalog.drop_duplicates(
            ["head_name", "side_name", "fold_id", "model_slot", "tree_index", "leaf_token"],
            keep="first",
        ).sort_values(["model_slot", "tree_index", "leaf_token"], kind="stable").reset_index(drop=True)
        catalog["ensemble_tree_contribution"] = (
            pd.to_numeric(catalog["tree_leaf_value"], errors="coerce") / float(len(fitted))
        ).astype(np.float32)
        if not np.isfinite(catalog["ensemble_tree_contribution"].to_numpy(float)).all():
            raise StrictOOFBaseReasoningError("catalog leaf contributions must be finite")

    manifest: dict[str, Any] = {
        "schema": STRICT_OOF_BASE_REASONING_SCHEMA,
        "head_name": head,
        "side_name": side,
        "fold_id": fold,
        "status": "MATERIALIZED_STRICT_OOF",
        "config": asdict(config),
        "contract": {
            "evaluation_features_are_label_free": True,
            "strict_time_boundary": "max(train_timestamps) < min(eval_timestamps)",
            "leaf_alignment": "opaque leaf tokens are scoped to head/side/fold/model/tree; no raw leaf IDs are aligned across models or folds",
            "rule_signatures": "G2 hashes structural split paths plus train-fold robust threshold bands; raw split thresholds remain only in rule_path_json and are never used for recurrence",
            "contribution_bundle": "G3 uses the explicitly selected LightGBM pred_contrib class; SVD is fit on train fold only",
            "family_contribution_lineage": "selected-tree additive leaf values are catalog-scoped and may be joined only to same-artifact assignments before collapsing to structural rule signatures",
            "multiclass_head_selection": "multiclass requires an explicit class_index or matching head_class_map; each semantic head is materialised separately",
            "training_label_use": "optional train_targets appear only as leaf-catalog diagnostics and never as evaluation features",
            "latent_regime_inputs": False,
        },
        "provenance": {
            "train_start_utc": train_ts.min(),
            "train_end_utc": train_ts.max(),
            "eval_start_utc": eval_ts.min(),
            "eval_end_utc": eval_ts.max(),
            "feature_contract": feature_names,
            "feature_contract_sha256": hashlib.sha256(
                json.dumps(feature_names, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "rule_threshold_banding": {
                "method": "train_fold_numeric_quantiles",
                "band_count": int(config.rule_threshold_band_count),
                "min_train_rows": int(config.rule_threshold_band_min_train_rows),
                "feature_state_fingerprint": _threshold_band_state_fingerprint(
                    threshold_bands
                ),
                "feature_count": int(len(threshold_bands)),
            },
            "model_hashes": [_model_hash(model) for model in fitted],
            "model_class_counts": class_counts,
            "head_tree_selection": tree_selection_by_model,
            "class_index": int(selected_class),
            "class_selection_source": class_selection_source,
            "head_class_map": (
                {str(key): int(value) for key, value in head_class_map.items()}
                if head_class_map is not None
                else None
            ),
            "base_prediction_source": "caller_supplied" if eval_predictions is not None else "mean_fitted_model_prediction",
            "eval_labels_present": labels is not None,
            "train_targets_present": train_targets is not None,
        },
        "rows": {
            "train": int(len(train)),
            "eval": int(row_count),
            "leaf_assignment_columns": int(len(assignment.columns) - len(IDENTITY_COLUMNS) - 2),
            "leaf_rule_catalog_rows": int(len(catalog)),
            "feature_columns": int(len(features.columns) - len(IDENTITY_COLUMNS) - 2),
            "contribution_bundle_columns": int(len(contribution_bundle.columns) - len(IDENTITY_COLUMNS) - 2),
        },
        "in_memory_table_hashes": {
            "features": _sha256_frame(features),
            "predictions": _sha256_frame(predictions),
            "leaf_assignments": _sha256_frame(assignment),
            "leaf_rule_catalog": _sha256_frame(catalog),
            "contribution_bundle": _sha256_frame(contribution_bundle),
        },
    }
    if contribution_cache is not None:
        manifest["provenance"]["contribution_cache"] = {
            "mode": "shared_multiclass_per_fold",
            "class_count": int(contribution_cache.class_count),
            "retained_bytes": int(contribution_cache.retained_bytes),
            "selected_class_slice": int(selected_class),
            "contains_raw_leaf_identifiers": False,
        }
        if contribution_cache.storage_mode != "in_memory":
            # Scratch paths are intentionally omitted: they disappear after
            # the fold and are not part of the immutable reasoning artifact.
            manifest["provenance"]["contribution_cache"]["storage_mode"] = (
                str(contribution_cache.storage_mode)
            )
    if labels is not None:
        manifest["in_memory_table_hashes"]["labels"] = _sha256_frame(labels)
    destination = _write_artifacts(
        Path(artifact_dir),
        features=features,
        predictions=predictions,
        labels=labels,
        assignments=assignment if config.write_leaf_assignments else identity.assign(head_name=head, fold_id=fold),
        catalog=catalog if config.write_rule_catalog else pd.DataFrame(),
        contribution_bundle=contribution_bundle,
        manifest=manifest,
    ) if artifact_dir is not None else None
    return StrictOOFBaseReasoningResult(
        features=features,
        predictions=predictions,
        labels=labels,
        leaf_assignments=assignment,
        leaf_rule_catalog=catalog,
        contribution_bundle=contribution_bundle,
        manifest=manifest,
        artifact_dir=destination,
    )


__all__ = [
    "FEATURE_PREFIX",
    "IDENTITY_COLUMNS",
    "STRICT_OOF_BASE_REASONING_SCHEMA",
    "StrictOOFBaseReasoningConfig",
    "StrictOOFContributionCacheCapacityError",
    "StrictOOFBaseReasoningError",
    "StrictOOFBaseReasoningResult",
    "StrictOOFMulticlassContributionCache",
    "build_strict_oof_multiclass_contribution_cache",
    "materialize_strict_oof_base_reasoning",
]
