"""Shallow tree leaf states for side x archetype residual failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class ResidualLeafConfig:
    max_features: int = 1200
    minimum_feature_coverage: float = 0.55
    target_quantile: float = 0.90
    n_estimators: int = 240
    feature_cluster_count: int = 6
    time_cluster_count: int = 5
    time_embedding_dim: int = 10
    random_state: int = 20260712


@dataclass
class NativeBinaryModel:
    """Small adapter around native LightGBM for sklearn-version independence."""

    booster_: lgb.Booster

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        positive = np.asarray(self.booster_.predict(x), dtype=np.float32)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, x: np.ndarray, pred_leaf: bool = False) -> np.ndarray:
        return np.asarray(self.booster_.predict(x, pred_leaf=pred_leaf))


@dataclass
class IdentitySparseProjection:
    """Serializable fallback when leaf one-hot has only one coordinate."""

    def transform(self, values: Any) -> np.ndarray:
        return np.asarray(values.toarray(), dtype=np.float32)


def observable_feature_names(frame: pd.DataFrame, config: ResidualLeafConfig) -> list[str]:
    """Return numeric pre-entry state columns, excluding all outcome signatures."""
    excluded_exact = {
        "global_state_id",
        "diagnostic_event_ids",
        "source_month",
    }
    candidates = [
        name
        for name in frame.select_dtypes(include=[np.number, "bool"]).columns
        if name not in excluded_exact
        and not name.startswith(("target_", "placebo_target_", "__"))
    ]
    if not candidates:
        return []
    coverage = frame[candidates].notna().mean()
    candidates = [
        name
        for name in candidates
        if float(coverage[name]) >= float(config.minimum_feature_coverage)
        and int(frame[name].nunique(dropna=True)) > 2
    ]
    if len(candidates) <= int(config.max_features):
        return candidates
    variance = frame[candidates].var(skipna=True).replace([np.inf, -np.inf], np.nan)
    return variance.nlargest(int(config.max_features)).index.astype(str).tolist()


def stable_binned_mi_screen(
    frame: pd.DataFrame,
    label: np.ndarray,
    features: Sequence[str],
    max_features: int = 200,
    bins: int = 10,
) -> tuple[list[str], pd.DataFrame]:
    """Screen nonlinear local event relevance across three chronological blocks."""
    y = np.asarray(label, dtype=np.int8)
    boundaries = np.linspace(0, len(frame), 4, dtype=int)
    rows: list[dict[str, float | str]] = []
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        block_scores: list[float] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            local_x = values.iloc[start:end]
            local_y = y[start:end]
            valid = np.isfinite(local_x.to_numpy(dtype=np.float64))
            if valid.sum() < 100 or np.unique(local_y[valid]).size < 2:
                block_scores.append(0.0)
                continue
            try:
                encoded = pd.qcut(
                    local_x.loc[valid], q=int(bins), labels=False, duplicates="drop"
                ).to_numpy()
            except ValueError:
                block_scores.append(0.0)
                continue
            if not np.isfinite(encoded).all() or np.unique(encoded).size < 2:
                block_scores.append(0.0)
                continue
            block_scores.append(float(mutual_info_score(local_y[valid], encoded)))
        mean_score = float(np.mean(block_scores))
        minimum_score = float(np.min(block_scores))
        rows.append(
            {
                "feature": str(feature),
                "mi_early": block_scores[0],
                "mi_middle": block_scores[1],
                "mi_late": block_scores[2],
                "mi_mean": mean_score,
                "mi_min": minimum_score,
                "stable_score": mean_score + 0.5 * minimum_score,
            }
        )
    report = pd.DataFrame(rows).sort_values(
        ["stable_score", "mi_mean", "feature"],
        ascending=[False, False, True],
        kind="stable",
    )
    selected = report.head(min(int(max_features), len(report)))["feature"].tolist()
    return selected, report.reset_index(drop=True)


def fit_matrix(
    train: pd.DataFrame,
    other: Sequence[pd.DataFrame],
    features: Sequence[str],
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Median-impute and downcast without fitting transforms outside train."""
    medians = (
        train[list(features)]
        .apply(pd.to_numeric, errors="coerce")
        .median()
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    def convert(frame: pd.DataFrame) -> np.ndarray:
        values = frame[list(features)].to_numpy(dtype=np.float32, copy=True)
        invalid = ~np.isfinite(values)
        if invalid.any():
            values[invalid] = np.take(medians, np.nonzero(invalid)[1])
        return values

    return convert(train), [convert(frame) for frame in other], medians


def failure_target(
    values: pd.Series, quantile: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build an extreme-failure class and continuous severity weights."""
    raw = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    positive = raw.loc[raw.gt(0)]
    threshold = float(positive.quantile(quantile)) if not positive.empty else np.inf
    label = raw.ge(threshold).to_numpy(dtype=np.int8)
    scale = max(float(positive.quantile(0.75)) if not positive.empty else 0.0, 1e-6)
    severity = np.clip(raw.to_numpy(dtype=np.float32) / scale, 0.0, 5.0)
    weights = (1.0 + severity).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-6)
    return label, weights, threshold


def candidate_parameter_grid(n_estimators: int) -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for depth, leaves in ((2, 4), (3, 6), (3, 8)):
        for min_child in (24, 72):
            for reg_lambda in (1.0, 5.0):
                grid.append(
                    {
                        "max_depth": depth,
                        "num_leaves": leaves,
                        "min_child_samples": min_child,
                        "reg_alpha": 0.25,
                        "reg_lambda": reg_lambda,
                        "min_split_gain": 0.01,
                        "n_estimators": n_estimators,
                    }
                )
    return grid


def fit_shallow_classifier(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    params: Mapping[str, Any],
    seed: int,
) -> NativeBinaryModel:
    native_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.025,
        "max_bin": 63,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "feature_fraction": 0.65,
        "seed": int(seed),
        "num_threads": 2,
        "verbosity": -1,
        "max_depth": int(params["max_depth"]),
        "num_leaves": int(params["num_leaves"]),
        "min_data_in_leaf": int(params["min_child_samples"]),
        "lambda_l1": float(params["reg_alpha"]),
        "lambda_l2": float(params["reg_lambda"]),
        "min_gain_to_split": float(params["min_split_gain"]),
    }
    dataset = lgb.Dataset(
        x,
        label=np.asarray(y, dtype=np.float32),
        weight=np.asarray(weights, dtype=np.float32),
        free_raw_data=False,
    )
    booster = lgb.train(
        native_params,
        dataset,
        num_boost_round=int(params["n_estimators"]),
    )
    return NativeBinaryModel(booster_=booster)


def tail_recognition_metrics(
    y: np.ndarray,
    severity: np.ndarray,
    score: np.ndarray,
    fraction: float = 0.10,
) -> dict[str, float]:
    valid = np.isfinite(score)
    y = np.asarray(y)[valid]
    severity = np.asarray(severity)[valid]
    score = np.asarray(score)[valid]
    if not len(y):
        return {}
    cutoff = float(np.quantile(score, 1.0 - fraction))
    selected = score >= cutoff
    positives = y > 0
    weighted_total = float(severity[positives].sum())
    return {
        "average_precision": float(average_precision_score(y, score))
        if np.unique(y).size > 1
        else np.nan,
        "roc_auc": float(roc_auc_score(y, score)) if np.unique(y).size > 1 else np.nan,
        "top10_precision": float(positives[selected].mean()) if selected.any() else np.nan,
        "top10_recall": float((selected & positives).sum() / max(positives.sum(), 1)),
        "top10_severity_recall": float(severity[selected & positives].sum() / weighted_total)
        if weighted_total > 0
        else np.nan,
        "top10_false_positive_rate": float((selected & ~positives).sum() / max((~positives).sum(), 1)),
        "positive_prevalence": float(positives.mean()),
        "top10_lift": float(positives[selected].mean() / max(positives.mean(), 1e-8))
        if selected.any()
        else np.nan,
    }


def _leaf_paths(model: NativeBinaryModel, feature_names: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dump = model.booster_.dump_model()

    def walk(node: Mapping[str, Any], tree_index: int, path: list[tuple[str, str]]) -> None:
        if "leaf_index" in node or "split_feature" not in node:
            rows.append(
                {
                    "tree_index": int(tree_index),
                    "leaf_index": int(node.get("leaf_index", 0)),
                    "path_features": "|".join(item[0] for item in path),
                    "path_rules": "|".join(f"{name}:{direction}" for name, direction in path),
                    "path_depth": int(len(path)),
                }
            )
            return
        feature = str(feature_names[int(node["split_feature"])])
        walk(node["left_child"], tree_index, [*path, (feature, "le")])
        walk(node["right_child"], tree_index, [*path, (feature, "gt")])

    for tree_index, tree in enumerate(dump["tree_info"]):
        walk(tree["tree_structure"], tree_index, [])
    return pd.DataFrame(rows)


def leaf_feature_clusters(
    model: NativeBinaryModel,
    feature_names: Sequence[str],
    train_leaves: np.ndarray,
    train_severity: np.ndarray,
    config: ResidualLeafConfig,
) -> tuple[pd.DataFrame, dict[tuple[int, int], tuple[int, float]]]:
    """Cluster leaves by split-feature incidence and attach train-only risk priors."""
    if train_leaves.ndim == 1:
        train_leaves = train_leaves.reshape(-1, 1)
    paths = _leaf_paths(model, feature_names)
    used = sorted(
        {feature for value in paths["path_features"] for feature in str(value).split("|") if feature}
    )
    index = {name: position for position, name in enumerate(used)}
    matrix = np.zeros((len(paths), len(used)), dtype=np.float32)
    for row_index, value in enumerate(paths["path_features"]):
        for feature in str(value).split("|"):
            if feature in index:
                matrix[row_index, index[feature]] = 1.0
    clusters = min(int(config.feature_cluster_count), max(2, len(paths) // 20), len(paths))
    if matrix.shape[1] == 0 or len(paths) == 1:
        labels = np.zeros(len(paths), dtype=np.int16)
    else:
        labels = MiniBatchKMeans(
            n_clusters=clusters,
            random_state=int(config.random_state),
            batch_size=256,
            n_init=10,
        ).fit_predict(matrix)
    paths["feature_cluster"] = labels.astype(np.int16)
    global_mean = max(float(np.mean(train_severity)), 1e-6)
    mapping: dict[tuple[int, int], tuple[int, float]] = {}
    risks: list[float] = []
    for row in paths.itertuples(index=False):
        active = train_leaves[:, int(row.tree_index)] == int(row.leaf_index)
        support = int(active.sum())
        local_sum = float(train_severity[active].sum())
        prior = (local_sum + 40.0 * global_mean) / (support + 40.0)
        risk = float(prior / global_mean)
        mapping[(int(row.tree_index), int(row.leaf_index))] = (
            int(row.feature_cluster),
            risk,
        )
        risks.append(risk)
    paths["train_failure_risk_lift"] = np.asarray(risks, dtype=np.float32)
    return paths, mapping


def feature_cluster_composites(
    leaves: np.ndarray,
    mapping: Mapping[tuple[int, int], tuple[int, float]],
    cluster_count: int,
) -> np.ndarray:
    if leaves.ndim == 1:
        leaves = leaves.reshape(-1, 1)
    output = np.zeros((len(leaves), cluster_count), dtype=np.float32)
    counts = np.zeros((len(leaves), cluster_count), dtype=np.float32)
    for tree in range(leaves.shape[1]):
        for leaf in np.unique(leaves[:, tree]):
            cluster, risk = mapping[(tree, int(leaf))]
            mask = leaves[:, tree] == leaf
            output[mask, cluster] += np.float32(risk)
            counts[mask, cluster] += 1.0
    np.divide(output, counts, out=output, where=counts > 0)
    return output


@dataclass
class TimeLeafClusters:
    encoder: OneHotEncoder
    svd: Any
    kmeans: MiniBatchKMeans
    risk_lift: np.ndarray

    def transform(self, leaves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        embedded = self.svd.transform(self.encoder.transform(leaves)).astype(np.float32)
        distance = self.kmeans.transform(embedded).astype(np.float32)
        scale = np.maximum(np.median(distance, axis=0, keepdims=True), 1e-4)
        logits = -distance / scale
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits, dtype=np.float32)
        probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-8)
        state = probability.argmax(axis=1).astype(np.int16)
        expected_risk = probability @ self.risk_lift.astype(np.float32)
        return probability, state, expected_risk.astype(np.float32)


def fit_time_leaf_clusters(
    train_leaves: np.ndarray,
    train_severity: np.ndarray,
    config: ResidualLeafConfig,
) -> TimeLeafClusters:
    if train_leaves.ndim == 1:
        train_leaves = train_leaves.reshape(-1, 1)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    encoded = encoder.fit_transform(train_leaves)
    if encoded.shape[1] < 2:
        svd: Any = IdentitySparseProjection()
        embedded = svd.transform(encoded)
        clusters = 1
    else:
        components = min(
            int(config.time_embedding_dim), max(1, encoded.shape[1] - 1)
        )
        svd = TruncatedSVD(
            n_components=components, random_state=int(config.random_state)
        )
        embedded = svd.fit_transform(encoded).astype(np.float32)
        clusters = min(int(config.time_cluster_count), max(2, len(embedded) // 200))
    kmeans = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=int(config.random_state),
        batch_size=512,
        n_init=10,
    ).fit(embedded)
    labels = kmeans.labels_
    global_mean = max(float(np.mean(train_severity)), 1e-6)
    risk = np.empty(clusters, dtype=np.float32)
    for state in range(clusters):
        mask = labels == state
        risk[state] = np.float32(
            (float(train_severity[mask].sum()) + 100.0 * global_mean)
            / (int(mask.sum()) + 100.0)
            / global_mean
        )
    return TimeLeafClusters(encoder=encoder, svd=svd, kmeans=kmeans, risk_lift=risk)

def causal_rolling_summary_features(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    window: int = 24,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Create causal persistence/extreme summaries without using future rows.

    The caller must provide rows in chronological order for one local model
    population (normally one side x archetype). The current observation is
    included, which matches inference-time availability.
    """
    minimum = int(min_periods if min_periods is not None else max(4, window // 4))
    values = frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    rolling = values.rolling(int(window), min_periods=minimum)
    parts = []
    for suffix, summary in (
        ("min", rolling.min()),
        ("max", rolling.max()),
        ("mean", rolling.mean()),
    ):
        summary.columns = [f"causal_{window}h_{suffix}__{name}" for name in features]
        parts.append(summary.astype(np.float32))
    return pd.concat(parts, axis=1, copy=False)
