"""Fold-local, family-balanced screening for causal multi-view regime fields.

The materialized multi-view panel can contain many highly related lag, horizon
and covariance transforms.  This module reduces that panel using only the
permitted training fold.  Unsupervised selection never receives a label;
optional regime and transition supervised rankings receive their own
fold-training labels and remain separate outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


MULTIVIEW_SELECTION_SCHEMA = "fold_local_multiview_feature_selection_v1"
FORBIDDEN_FEATURE_TOKENS: tuple[str, ...] = (
    "target", "label", "outcome", "post_entry", "postentry", "future",
    "realized_pnl", "realised_pnl", "realized_ev", "realised_ev", "realized_outcome", "realised_outcome", "mfe", "mae", "pnl", "net_ev", "gross_ev",
    "ev_after", "exit", "timeout", "time_to", "barrier",
)
HORIZON_RE = re.compile(r"_(15m|\d+h)$")


@dataclass(frozen=True)
class MultiviewSelectionConfig:
    """Bounded fold-local screening and redundancy controls."""

    fold_id: str
    min_coverage: float = 0.80
    min_robust_scale: float = 1e-6
    correlation_threshold: float = 0.92
    max_correlation_rows: int = 8_000
    max_candidates_per_family_before_redundancy: int = 256
    family_caps: Mapping[str, int] = field(
        default_factory=lambda: {
            "distribution_dynamics": 24, "volatility": 16,
            "liquidity_proxy": 16, "dependence_covariance": 16, "other": 8,
        }
    )
    supervised_family_caps: Mapping[str, int] | None = None
    preserve_horizon_representation: bool = True
    require_training_identity: bool = True
    random_state: int = 20260730


@dataclass(frozen=True)
class MultiviewSelectionResult:
    unsupervised_features: list[str]
    regime_features: list[str]
    transition_features: list[str]
    lineage: pd.DataFrame
    diagnostics: dict[str, Any]


def _is_forbidden(name: str) -> bool:
    return any(token in str(name).lower() for token in FORBIDDEN_FEATURE_TOKENS)


def infer_multiview_lineage(columns: Iterable[str]) -> pd.DataFrame:
    """Infer family/horizon/source/transform lineage from materialized names."""

    rows: list[dict[str, str]] = []
    for raw in columns:
        name = str(raw)
        match = HORIZON_RE.search(name)
        horizon = match.group(1) if match else "unknown"
        stem = name[:match.start()] if match else name
        parts = stem.split("__")
        family, source, transform = "other", "unknown", "unknown"
        if len(parts) >= 3 and parts[0] == "mv":
            if parts[1] == "dependence":
                family, source, transform = "dependence_covariance", "dependence", "__".join(parts[2:])
            elif parts[1] == "liquidity":
                family = "liquidity_proxy"
                source = parts[2] if len(parts) >= 4 else "unknown"
                transform = "__".join(parts[3:]) if len(parts) >= 4 else "unknown"
            else:
                source, transform = parts[1], "__".join(parts[2:])
                family = "volatility" if transform in {"realized_vol", "vol_of_vol"} else "distribution_dynamics"
        rows.append({"feature": name, "family": family, "horizon": horizon, "source_field": source, "transform": transform})
    return pd.DataFrame(rows)


def _validate_train_frame(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        raise ValueError("fold-local train feature frame is empty")
    if not frame.index.is_unique:
        raise ValueError("fold-local feature rows require unique identities")
    columns = [str(column) for column in frame.columns]
    forbidden = [column for column in columns if _is_forbidden(column)]
    if forbidden:
        raise ValueError(f"feature frame contains forbidden outcome/post-entry fields: {forbidden[:8]}")
    non_numeric = [column for column in columns if not pd.api.types.is_numeric_dtype(frame[column])]
    if non_numeric:
        raise TypeError(f"feature frame must contain numeric candidate fields only: {non_numeric[:8]}")
    return columns


def _lineage_with_override(columns: list[str], feature_metadata: pd.DataFrame | None) -> pd.DataFrame:
    base = infer_multiview_lineage(columns)
    if feature_metadata is None:
        return base
    required = {"feature", "family", "horizon"}
    missing = required.difference(feature_metadata.columns)
    if missing:
        raise KeyError(f"feature metadata missing required columns: {sorted(missing)}")
    supplied = feature_metadata.copy()
    supplied["feature"] = supplied["feature"].astype(str)
    if supplied["feature"].duplicated().any():
        raise ValueError("feature metadata must provide one lineage row per feature")
    supplied = supplied.set_index("feature")
    for field in ("family", "horizon", "source_field", "transform"):
        if field in supplied:
            base[field] = base["feature"].map(supplied[field]).fillna(base[field]).astype(str)
    return base


def _screen_train_features(frame: pd.DataFrame, lineage: pd.DataFrame, config: MultiviewSelectionConfig) -> pd.DataFrame:
    # The production multiview panel has many thousands of columns.  Work in
    # modest column batches so an expanding-fold run does not duplicate the
    # entire training matrix in memory merely to calculate robust univariate
    # screens.  The result is identical to a whole-matrix calculation.
    batch_width = 512
    coverage_parts: list[np.ndarray] = []
    mad_parts: list[np.ndarray] = []
    for start in range(0, frame.shape[1], batch_width):
        values = frame.iloc[:, start : start + batch_width].to_numpy(dtype=float)
        coverage_parts.append(np.isfinite(values).mean(axis=0))
        median = np.nanmedian(values, axis=0)
        mad_parts.append(np.nanmedian(np.abs(values - median), axis=0) * 1.4826)
    coverage = np.concatenate(coverage_parts)
    mad = np.concatenate(mad_parts)
    result = lineage.copy()
    result["coverage_train"] = coverage.astype(float)
    result["robust_scale_train"] = mad.astype(float)
    result["unsupervised_screen_score"] = (np.log1p(np.maximum(mad, 0.0)) + coverage).astype(float)
    result["coverage_pass"] = result["coverage_train"].ge(float(config.min_coverage))
    result["variance_pass"] = result["robust_scale_train"].ge(float(config.min_robust_scale))
    result["screen_pass"] = result["coverage_pass"] & result["variance_pass"]
    return result


def _deterministic_rows(n: int, maximum: int, seed: int) -> np.ndarray:
    if n <= maximum:
        return np.arange(n, dtype=np.int64)
    return np.sort(np.random.default_rng(int(seed)).choice(n, size=int(maximum), replace=False)).astype(np.int64)


def _robust_rank_correlation(values: np.ndarray, floor: float) -> np.ndarray:
    """Rank correlation after train-only median/IQR clipping."""

    matrix = np.asarray(values, dtype=float).copy()
    if matrix.ndim != 2:
        raise ValueError("robust correlation requires a two-dimensional feature matrix")
    if matrix.shape[1] == 1:
        return np.ones((1, 1), dtype=float)
    median = np.nanmedian(matrix, axis=0)
    q25, q75 = np.nanquantile(matrix, (0.25, 0.75), axis=0)
    scale = np.maximum(q75 - q25, float(floor))
    matrix = np.where(np.isfinite(matrix), matrix, median)
    matrix = np.clip(matrix, median - 8.0 * scale, median + 8.0 * scale)
    corr = np.corrcoef(pd.DataFrame(matrix).rank(method="average").to_numpy(dtype=float), rowvar=False)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def _connected_components(correlation: np.ndarray, threshold: float) -> np.ndarray:
    n = correlation.shape[0]
    parent = np.arange(n, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for left in range(n):
        matches = np.flatnonzero(np.abs(correlation[left, left + 1 :]) >= float(threshold)) + left + 1
        for right in matches:
            union(left, int(right))
    labels = np.asarray([find(index) for index in range(n)], dtype=np.int64)
    return np.unique(labels, return_inverse=True)[1].astype(np.int64)


def _family_prune(frame: pd.DataFrame, screened: pd.DataFrame, config: MultiviewSelectionConfig) -> pd.DataFrame:
    """Attach train-only redundancy clusters and unsupervised selections."""

    out = screened.copy()
    out["redundancy_cluster"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out["unsupervised_selected"] = False
    out["horizon_representation_override"] = False
    rows = _deterministic_rows(len(frame), config.max_correlation_rows, config.random_state)
    for family, family_rows in out.loc[out["screen_pass"]].groupby("family", observed=True, sort=True):
        ranked = family_rows.sort_values(["unsupervised_screen_score", "feature"], ascending=[False, True], kind="stable").head(int(config.max_candidates_per_family_before_redundancy))
        names = ranked["feature"].tolist()
        if not names:
            continue
        clusters = _connected_components(_robust_rank_correlation(frame.iloc[rows].loc[:, names].to_numpy(dtype=float), config.min_robust_scale), config.correlation_threshold)
        cluster_names = [f"{family}:{int(cluster)}" for cluster in clusters]
        out.loc[ranked.index, "redundancy_cluster"] = cluster_names
        ranked = ranked.assign(redundancy_cluster=cluster_names)
        representatives = ranked.sort_values(["unsupervised_screen_score", "feature"], ascending=[False, True], kind="stable").drop_duplicates("redundancy_cluster", keep="first")
        cap = max(0, int(config.family_caps.get(str(family), config.family_caps.get("other", 0))))
        selected: list[int] = []
        if config.preserve_horizon_representation:
            # A representative from every observed horizon is more useful for
            # multi-timeframe state discovery than collapsing all horizons to
            # one near-identical lag.  Such a necessary duplicate is marked
            # explicitly rather than being mistaken for ordinary redundancy.
            chosen_clusters: set[str] = set()
            for _, horizon_rows in ranked.sort_values(["horizon", "unsupervised_screen_score", "feature"], ascending=[True, False, True], kind="stable").groupby("horizon", observed=True, sort=True):
                if len(selected) >= cap:
                    break
                selected_index = int(horizon_rows.index[0])
                selected.append(selected_index)
                cluster = str(horizon_rows.iloc[0]["redundancy_cluster"])
                if cluster in chosen_clusters:
                    out.loc[selected_index, "horizon_representation_override"] = True
                chosen_clusters.add(cluster)
        for index in representatives.index:
            if len(selected) >= cap:
                break
            if int(index) not in selected:
                selected.append(int(index))
        out.loc[selected, "unsupervised_selected"] = True
    return out


def _validate_labels(label: Sequence[float] | pd.Series | np.ndarray | None, index: pd.Index, *, name: str) -> np.ndarray | None:
    if label is None:
        return None
    series = pd.Series(label, index=index) if not isinstance(label, pd.Series) else label
    if not series.index.equals(index):
        raise ValueError(f"{name} labels must exactly align with fold-training row identities")
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or len(np.unique(values)) < 2:
        raise ValueError(f"{name} labels must be finite and non-constant in the training fold")
    return values


def _absolute_spearman(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    ranked_x = pd.DataFrame(values).rank(method="average").to_numpy(dtype=float)
    ranked_y = pd.Series(labels).rank(method="average").to_numpy(dtype=float)
    y_center = ranked_y - ranked_y.mean()
    x_center = ranked_x - ranked_x.mean(axis=0)
    denominator = np.sqrt(np.sum(x_center * x_center, axis=0) * np.sum(y_center * y_center))
    return np.divide(np.abs(x_center.T @ y_center), denominator, out=np.zeros(len(denominator)), where=denominator > 0)


def _supervised_select(frame: pd.DataFrame, lineage: pd.DataFrame, labels: np.ndarray | None, *, output_column: str, score_column: str, config: MultiviewSelectionConfig) -> list[str]:
    lineage[output_column] = False
    lineage[score_column] = np.nan
    if labels is None:
        return []
    pool = lineage.loc[lineage["unsupervised_selected"]].copy()
    if pool.empty:
        return []
    values = frame.loc[:, pool["feature"].tolist()].to_numpy(dtype=float)
    values = np.where(np.isfinite(values), values, np.nanmedian(values, axis=0))
    pool["_score"] = _absolute_spearman(values, labels)
    lineage.loc[pool.index, score_column] = pool["_score"]
    caps = config.supervised_family_caps or config.family_caps
    selected: list[int] = []
    for family, family_rows in pool.sort_values(["_score", "feature"], ascending=[False, True], kind="stable").groupby("family", observed=True, sort=True):
        cap = max(0, int(caps.get(str(family), caps.get("other", 0))))
        family_selected: list[int] = []
        if config.preserve_horizon_representation:
            for _, horizon_rows in family_rows.groupby("horizon", observed=True, sort=True):
                if len(family_selected) >= cap:
                    break
                family_selected.append(int(horizon_rows.index[0]))
        for index in family_rows.index:
            if len(family_selected) >= cap:
                break
            if int(index) not in family_selected:
                family_selected.append(int(index))
        selected.extend(family_selected)
    lineage.loc[selected, output_column] = True
    return lineage.loc[selected].sort_values(score_column, ascending=False, kind="stable")["feature"].tolist()


def select_fold_local_multiview_features(
    train_features: pd.DataFrame,
    *,
    config: MultiviewSelectionConfig,
    feature_metadata: pd.DataFrame | None = None,
    regime_train_labels: Sequence[float] | pd.Series | np.ndarray | None = None,
    transition_train_labels: Sequence[float] | pd.Series | np.ndarray | None = None,
    fold_training_row_ids: Sequence[Any] | pd.Index | None = None,
) -> MultiviewSelectionResult:
    """Screen one training fold; regime and transition labels stay separate."""

    if not str(config.fold_id).strip():
        raise ValueError("fold_id is required for fold-local feature selection")
    columns = _validate_train_frame(train_features)
    if config.require_training_identity:
        if fold_training_row_ids is None:
            raise ValueError("fold_training_row_ids is required to attest train-only selection")
        if not pd.Index(fold_training_row_ids).equals(train_features.index):
            raise ValueError("fold_training_row_ids must exactly equal train feature identities")
    pruned = _family_prune(train_features, _screen_train_features(train_features, _lineage_with_override(columns, feature_metadata), config), config)
    regime_labels = _validate_labels(regime_train_labels, train_features.index, name="regime")
    transition_labels = _validate_labels(transition_train_labels, train_features.index, name="transition")
    regime_features = _supervised_select(train_features, pruned, regime_labels, output_column="regime_selected", score_column="regime_supervised_score", config=config)
    transition_features = _supervised_select(train_features, pruned, transition_labels, output_column="transition_selected", score_column="transition_supervised_score", config=config)
    unsupervised_features = pruned.loc[pruned["unsupervised_selected"]].sort_values(["family", "horizon", "unsupervised_screen_score", "feature"], ascending=[True, True, False, True], kind="stable")["feature"].tolist()
    diagnostics = {
        "schema": MULTIVIEW_SELECTION_SCHEMA, "research_only": True, "fold_id": str(config.fold_id),
        "train_only_attested": bool(config.require_training_identity), "input_rows": int(len(train_features)),
        "input_features": int(len(columns)), "screen_pass_features": int(pruned["screen_pass"].sum()),
        "unsupervised_selected_features": int(len(unsupervised_features)),
        "regime_supervised_labels_used": regime_labels is not None,
        "transition_supervised_labels_used": transition_labels is not None,
        "regime_selected_features": int(len(regime_features)), "transition_selected_features": int(len(transition_features)),
        "labels_used_for_unsupervised_selection": False,
        "family_counts_unsupervised": pruned.loc[pruned["unsupervised_selected"]].groupby("family", observed=True).size().to_dict(),
        "horizons_unsupervised": sorted(pruned.loc[pruned["unsupervised_selected"], "horizon"].unique().tolist()),
    }
    return MultiviewSelectionResult(unsupervised_features, regime_features, transition_features, pruned.sort_values(["family", "horizon", "feature"], kind="stable").reset_index(drop=True), diagnostics)


__all__ = [
    "FORBIDDEN_FEATURE_TOKENS", "MULTIVIEW_SELECTION_SCHEMA", "MultiviewFeatureSelectionResult",
    "MultiviewSelectionConfig", "infer_multiview_lineage", "select_fold_local_multiview_features",
]
