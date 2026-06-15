from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans


ARCHETYPE_SVD_COMPONENTS = 16
RAW_STATE_SVD_COMPONENTS = 16
RAW_STATE_TOD_BUCKETS = 6
RAW_STATE_KNN_NEIGHBORS = 5
RAW_STATE_GMM_COMPONENTS = 8
RAW_STATE_MAX_REFERENCE_ROWS = 50_000
RAW_STATE_DISTRIBUTION_BINS = 10

CONTRIB_SUMMARY_FEATURE_NAMES = [
    "contrib_abs_sum",
    "contrib_l2_norm",
    "contrib_entropy",
    "top_1_contrib_abs",
    "top_3_contrib_abs_sum",
    "positive_contrib_sum",
    "negative_contrib_sum",
]
CONTRIB_ARCHETYPE_FEATURE_NAMES = [
    f"archetype_contrib_svd_{i:02d}" for i in range(ARCHETYPE_SVD_COMPONENTS)
]
RAW_CONTRIB_FEATURE_PREFIX = "archetype_contrib_raw_"
RAW_CONTRIB_OOF_PREFIX = f"oof_{RAW_CONTRIB_FEATURE_PREFIX}"
META_RAW_CONTRIB_SVD_FEATURE_NAMES = [
    f"meta_base_contrib_svd_{i:02d}" for i in range(ARCHETYPE_SVD_COMPONENTS)
]
RAW_STATE_SVD_FEATURE_NAMES = [
    f"raw_state_svd_{i:02d}" for i in range(RAW_STATE_SVD_COMPONENTS)
]
RAW_STATE_SVD_SUMMARY_FEATURE_NAMES = [
    "raw_state_svd_mean",
    "raw_state_svd_std",
]
RAW_STATE_DISTRIBUTION_FEATURE_NAMES = [
    "raw_state_psi_mean",
    "raw_state_psi_max",
    "raw_state_ks_mean",
    "raw_state_ks_max",
    "raw_state_svd_psi_mean",
    "raw_state_svd_psi_max",
    "raw_state_svd_ks_mean",
    "raw_state_svd_ks_max",
]
RAW_STATE_DIAGNOSTIC_FEATURE_NAMES = [
    "raw_state_mahalanobis",
    "raw_state_knn_distance",
    "raw_state_min_cluster_distance",
    "raw_state_reconstruction_error",
    "raw_state_transition_norm",
    "raw_state_transition_mahalanobis",
    "state_log_likelihood",
    "state_tod_mahalanobis",
] + RAW_STATE_DISTRIBUTION_FEATURE_NAMES + RAW_STATE_SVD_SUMMARY_FEATURE_NAMES
ARCHETYPE_FEATURE_NAMES = (
    CONTRIB_SUMMARY_FEATURE_NAMES
    + CONTRIB_ARCHETYPE_FEATURE_NAMES
    + RAW_STATE_SVD_FEATURE_NAMES
    + RAW_STATE_DIAGNOSTIC_FEATURE_NAMES
)
BASE_ERROR_ARCHETYPE_FEATURE_NAMES = [
    "base_error_archetype_id",
    "base_error_archetype_is_bad",
    "base_error_archetype_is_good",
    "base_error_archetype_is_neutral",
    "base_error_distance_to_archetype_centroid",
    "base_error_distance_to_nearest_bad_archetype",
    "base_error_archetype_oof_bad_rate_lift",
    "base_error_distance_to_bad_archetype",
    "base_error_distance_to_good_archetype",
]
ARCHETYPE_OOF_PREFIXES = tuple(f"oof_{name}" for name in ARCHETYPE_FEATURE_NAMES)


@dataclass
class ContribArchetypeState:
    feature_names: list[str]
    scaler: StandardScaler | None = None
    svd: TruncatedSVD | None = None
    component_count: int = 0


@dataclass
class RawStateArchetypeState:
    feature_names: list[str]
    scaler: StandardScaler | None = None
    svd: TruncatedSVD | None = None
    component_count: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    inv_cov: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    logdet_cov: float = 0.0
    knn: NearestNeighbors | None = None
    centroids: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    transition_mean: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    transition_inv_cov: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    tod_bucket_stats: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    raw_distribution_edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    raw_distribution_probs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    svd_distribution_edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    svd_distribution_probs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))


@dataclass
class ResidualErrorArchetypeState:
    feature_names: list[str]
    enabled: bool = False
    reason: str = ""
    center: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    scale: np.ndarray = field(default_factory=lambda: np.ones(0, dtype=np.float32))
    centroids: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    bad_cluster_ids: list[int] = field(default_factory=list)
    good_cluster_ids: list[int] = field(default_factory=list)
    clusters: list[dict[str, Any]] = field(default_factory=list)
    global_bad_rate: float = 0.5


def is_archetype_feature_name(name: str) -> bool:
    key = str(name)
    return (
        key in ARCHETYPE_FEATURE_NAMES
        or key in BASE_ERROR_ARCHETYPE_FEATURE_NAMES
        or is_raw_contrib_feature_name(key)
        or key.startswith(("pred_", "base_H"))
        and any(
            part in key
            for part in (
                "base_error_",
                "archetype_contrib_svd_",
                RAW_CONTRIB_FEATURE_PREFIX,
                "raw_state_svd_",
                "raw_state_",
                "state_log_likelihood",
                "contrib_abs_sum",
                "contrib_l2_norm",
                "contrib_entropy",
                "top_1_contrib_abs",
                "top_3_contrib_abs_sum",
                "positive_contrib_sum",
                "negative_contrib_sum",
            )
        )
    )


def is_raw_contrib_feature_name(name: str) -> bool:
    key = str(name)
    return (
        key.startswith(RAW_CONTRIB_FEATURE_PREFIX)
        or key.startswith(RAW_CONTRIB_OOF_PREFIX)
        or RAW_CONTRIB_FEATURE_PREFIX in key
        or RAW_CONTRIB_OOF_PREFIX in key
    )


def raw_contrib_feature_names(feature_names: Sequence[str]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for i, raw_name in enumerate(feature_names):
        raw_s = str(raw_name)
        digest = hashlib.sha1(raw_s.encode("utf-8", errors="ignore")).hexdigest()[:10]
        slug = _safe_slug(raw_s)
        base = f"{RAW_CONTRIB_FEATURE_PREFIX}{i:03d}_{digest}"
        candidate = f"{base}_{slug}" if slug else base
        if candidate in seen:
            candidate = f"{base}_{len(seen):03d}"
        names.append(candidate)
        seen.add(candidate)
    return names


def raw_contrib_feature_mapping(feature_names: Sequence[str]) -> dict[str, str]:
    return {
        out_name: str(in_name)
        for out_name, in_name in zip(raw_contrib_feature_names(feature_names), feature_names)
    }


def raw_contrib_frame(
    contrib: Any,
    feature_names: Sequence[str],
    *,
    index: Any = None,
) -> pd.DataFrame:
    c = _as_2d_float(contrib)
    n = int(c.shape[0])
    names = raw_contrib_feature_names(feature_names)
    data: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        if i < c.shape[1]:
            data[name] = c[:, i]
        else:
            data[name] = np.zeros(n, dtype=np.float32)
    return _frame_from_arrays(data, names, index=index)


def _safe_slug(value: str, *, max_len: int = 56) -> str:
    slug = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_").lower()
    slug = re.sub(r"_+", "_", slug)
    return slug[: int(max_len)].strip("_")


def contrib_summary_frame(contrib: Any, index: Any = None) -> pd.DataFrame:
    c = _as_2d_float(contrib)
    n = int(c.shape[0])
    if c.shape[1] == 0:
        return _empty_frame(n, CONTRIB_SUMMARY_FEATURE_NAMES, index=index)
    abs_c = np.abs(c)
    abs_sum = np.sum(abs_c, axis=1)
    sorted_abs = np.sort(abs_c, axis=1)[:, ::-1]
    top1 = sorted_abs[:, 0] if sorted_abs.shape[1] else np.zeros(n, dtype=np.float32)
    top3 = np.sum(sorted_abs[:, : min(3, sorted_abs.shape[1])], axis=1)
    share = abs_c / np.maximum(abs_sum[:, None], 1e-12)
    entropy = -np.sum(
        np.where(share > 0.0, share * np.log(share + 1e-12), 0.0),
        axis=1,
    )
    entropy = entropy / max(np.log(max(c.shape[1], 2)), 1e-12)
    data = {
        "contrib_abs_sum": abs_sum,
        "contrib_l2_norm": np.sqrt(np.sum(np.square(c), axis=1)),
        "contrib_entropy": entropy,
        "top_1_contrib_abs": top1,
        "top_3_contrib_abs_sum": top3,
        "positive_contrib_sum": np.sum(np.clip(c, 0.0, None), axis=1),
        "negative_contrib_sum": np.sum(np.clip(c, None, 0.0), axis=1),
    }
    return _frame_from_arrays(data, CONTRIB_SUMMARY_FEATURE_NAMES, index=index)


def fit_contrib_archetype_state(
    contrib_train: Any,
    feature_names: Sequence[str],
    *,
    n_components: int = ARCHETYPE_SVD_COMPONENTS,
    random_state: int = 42,
) -> ContribArchetypeState:
    c = _as_2d_float(contrib_train)
    state = ContribArchetypeState(feature_names=[str(f) for f in feature_names])
    if c.shape[0] < 2 or c.shape[1] < 1:
        return state
    scaler = StandardScaler()
    scaled = scaler.fit_transform(c).astype(np.float32, copy=False)
    comp = _component_count(scaled, n_components)
    if comp <= 0:
        return state
    svd = TruncatedSVD(n_components=comp, random_state=int(random_state))
    svd.fit(scaled)
    state.scaler = scaler
    state.svd = svd
    state.component_count = int(comp)
    return state


def transform_contrib_archetype_features(
    contrib: Any,
    state: ContribArchetypeState | None,
    *,
    index: Any = None,
) -> pd.DataFrame:
    c = _as_2d_float(contrib)
    n = int(c.shape[0])
    if state is None or state.scaler is None or state.svd is None or state.component_count <= 0:
        return _empty_frame(n, CONTRIB_ARCHETYPE_FEATURE_NAMES, index=index)
    scaled = state.scaler.transform(c).astype(np.float32, copy=False)
    z = state.svd.transform(scaled).astype(np.float32, copy=False)
    return _component_frame(z, CONTRIB_ARCHETYPE_FEATURE_NAMES, index=index)


def fit_raw_state_archetype_state(
    X_train: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    timestamps: Any = None,
    assets: Any = None,
    n_components: int = RAW_STATE_SVD_COMPONENTS,
    random_state: int = 42,
) -> RawStateArchetypeState:
    frozen_features = [str(f) for f in feature_names]
    x = _selected_matrix(X_train, frozen_features)
    state = RawStateArchetypeState(feature_names=frozen_features)
    if x.shape[0] < 2 or x.shape[1] < 1:
        return state
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x).astype(np.float32, copy=False)
    comp = _component_count(scaled, n_components)
    if comp <= 0:
        return state
    svd = TruncatedSVD(n_components=comp, random_state=int(random_state))
    z = svd.fit_transform(scaled).astype(np.float32, copy=False)
    raw_ref = _reference_rows(scaled, random_state=random_state + 17)
    ref = _reference_rows(z, random_state=random_state)
    mean, inv_cov, logdet = _fit_gaussian(ref)
    deltas = _transition_vectors(z, timestamps=timestamps, assets=assets)
    transition_mean, transition_inv_cov, _ = _fit_gaussian(deltas)
    raw_edges, raw_probs = _fit_distribution_bins(raw_ref)
    svd_edges, svd_probs = _fit_distribution_bins(ref)
    state.scaler = scaler
    state.svd = svd
    state.component_count = int(comp)
    state.mean = mean
    state.inv_cov = inv_cov
    state.logdet_cov = float(logdet)
    state.transition_mean = transition_mean
    state.transition_inv_cov = transition_inv_cov
    state.knn = _fit_knn(ref)
    state.centroids = _fit_centroids(ref, random_state=random_state)
    state.raw_distribution_edges = raw_edges
    state.raw_distribution_probs = raw_probs
    state.svd_distribution_edges = svd_edges
    state.svd_distribution_probs = svd_probs
    state.tod_bucket_stats = _fit_tod_bucket_stats(
        z,
        timestamps=timestamps,
        fallback_mean=mean,
        fallback_inv_cov=inv_cov,
    )
    return state


def transform_raw_state_archetype_features(
    X: pd.DataFrame,
    state: RawStateArchetypeState | None,
    *,
    timestamps: Any = None,
    assets: Any = None,
    index: Any = None,
) -> pd.DataFrame:
    n = int(len(X))
    if state is None or state.scaler is None or state.svd is None or state.component_count <= 0:
        return _empty_frame(n, RAW_STATE_SVD_FEATURE_NAMES + RAW_STATE_DIAGNOSTIC_FEATURE_NAMES, index=index)
    x = _selected_matrix(X, state.feature_names)
    scaled = state.scaler.transform(x).astype(np.float32, copy=False)
    z = state.svd.transform(scaled).astype(np.float32, copy=False)
    reconstructed = state.svd.inverse_transform(z).astype(np.float32, copy=False)
    reconstruction_error = np.sqrt(
        np.mean(np.square(scaled - reconstructed), axis=1)
    )
    mahal = _mahalanobis(z, state.mean, state.inv_cov)
    deltas = _transition_vectors(z, timestamps=timestamps, assets=assets)
    trans_mahal = _mahalanobis(deltas, state.transition_mean, state.transition_inv_cov)
    trans_norm = np.linalg.norm(deltas, axis=1)
    knn_distance = _knn_distance(z, state.knn)
    min_cluster_distance = _min_cluster_distance(z, state.centroids)
    log_likelihood = _gaussian_log_likelihood(
        z,
        state.mean,
        state.inv_cov,
        state.logdet_cov,
    )
    tod_mahal = _tod_mahalanobis(z, timestamps=timestamps, state=state)
    raw_psi_mean, raw_psi_max, raw_ks_mean, raw_ks_max = _distribution_scores(
        scaled,
        state.raw_distribution_edges,
        state.raw_distribution_probs,
    )
    svd_psi_mean, svd_psi_max, svd_ks_mean, svd_ks_max = _distribution_scores(
        z,
        state.svd_distribution_edges,
        state.svd_distribution_probs,
    )
    svd_mean = np.mean(z, axis=1).astype(np.float32)
    svd_std = np.std(z, axis=1).astype(np.float32)
    out = _component_frame(z, RAW_STATE_SVD_FEATURE_NAMES, index=index)
    diagnostics = _frame_from_arrays(
        {
            "raw_state_mahalanobis": mahal,
            "raw_state_knn_distance": knn_distance,
            "raw_state_min_cluster_distance": min_cluster_distance,
            "raw_state_reconstruction_error": reconstruction_error,
            "raw_state_transition_norm": trans_norm,
            "raw_state_transition_mahalanobis": trans_mahal,
            "state_log_likelihood": log_likelihood,
            "state_tod_mahalanobis": tod_mahal,
            "raw_state_psi_mean": raw_psi_mean,
            "raw_state_psi_max": raw_psi_max,
            "raw_state_ks_mean": raw_ks_mean,
            "raw_state_ks_max": raw_ks_max,
            "raw_state_svd_psi_mean": svd_psi_mean,
            "raw_state_svd_psi_max": svd_psi_max,
            "raw_state_svd_ks_mean": svd_ks_mean,
            "raw_state_svd_ks_max": svd_ks_max,
            "raw_state_svd_mean": svd_mean,
            "raw_state_svd_std": svd_std,
        },
        RAW_STATE_DIAGNOSTIC_FEATURE_NAMES,
        index=index,
    )
    return pd.concat([out, diagnostics], axis=1)


def fit_residual_error_archetype_state(
    signature_frame: pd.DataFrame,
    y_bad: Any,
    *,
    feature_names: Sequence[str] | None = None,
    min_rows: int = 100,
    min_role_support: int = 40,
    min_bad_rate_delta: float = 0.04,
    min_bad_rate_spread: float = 0.04,
    random_state: int = 42,
) -> ResidualErrorArchetypeState:
    if not isinstance(signature_frame, pd.DataFrame):
        signature_frame = pd.DataFrame(signature_frame)
    n = int(len(signature_frame))
    requested = (
        [str(c) for c in feature_names]
        if feature_names is not None
        else [str(c) for c in signature_frame.columns]
    )
    state = ResidualErrorArchetypeState(feature_names=requested)
    if n < int(min_rows) or not requested:
        state.reason = "insufficient_rows_or_features"
        return state
    frame = signature_frame.copy()
    frame.columns = [str(c) for c in frame.columns]
    numeric = pd.DataFrame(index=frame.index)
    for name in requested:
        if name in frame.columns:
            numeric[name] = pd.to_numeric(frame[name], errors="coerce")
    if numeric.empty:
        state.reason = "no_numeric_features"
        return state
    finite_share = numeric.replace([np.inf, -np.inf], np.nan).notna().mean(axis=0)
    nunique = numeric.nunique(dropna=True)
    keep = [
        str(c)
        for c in numeric.columns
        if float(finite_share.get(c, 0.0)) >= 0.70 and int(nunique.get(c, 0)) > 1
    ]
    if len(keep) > 128:
        dispersion = numeric[keep].replace([np.inf, -np.inf], np.nan).std(axis=0).fillna(0.0)
        keep = [str(c) for c in dispersion.sort_values(ascending=False).index[:128]]
    if not keep:
        state.reason = "no_reliable_numeric_features"
        return state
    y = np.asarray(y_bad, dtype=np.float32)[:n]
    finite_y = np.isfinite(y)
    if len(y) < n:
        finite_y = np.zeros(n, dtype=bool)
        finite_y[: len(y)] = np.isfinite(y)
    good_bad = np.zeros(n, dtype=np.int8)
    good_bad[: len(y)] = (y[:n] >= 0.5).astype(np.int8)
    if int(np.sum(finite_y)) < int(min_rows):
        state.feature_names = keep
        state.reason = "insufficient_label_rows"
        return state
    if len(np.unique(good_bad[finite_y])) < 2:
        state.feature_names = keep
        state.reason = "single_error_class"
        return state
    matrix_raw = _selected_matrix(numeric.loc[:, keep], keep)
    matrix_raw = matrix_raw[:n]
    center = np.nanmedian(matrix_raw[finite_y], axis=0).astype(np.float32)
    q75 = np.nanpercentile(matrix_raw[finite_y], 75.0, axis=0)
    q25 = np.nanpercentile(matrix_raw[finite_y], 25.0, axis=0)
    scale = np.asarray((q75 - q25) / 1.349, dtype=np.float32)
    fallback_scale = np.nanstd(matrix_raw[finite_y], axis=0).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, fallback_scale)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)
    x = np.nan_to_num(
        (matrix_raw - center) / scale,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)
    # KMeans is Euclidean; without clipping, a handful of extreme diagnostic
    # rows can consume most centroids and leave the normal manifold as one
    # neutral mega-cluster.
    x = np.clip(x, -8.0, 8.0).astype(np.float32, copy=False)
    x_fit = x[finite_y]
    y_fit = good_bad[finite_y]
    cluster_count = int(min(8, max(2, round(np.sqrt(max(len(x_fit), 1) / 40.0)))))
    cluster_count = int(min(cluster_count, max(2, len(x_fit) // max(int(min_role_support), 1))))
    if cluster_count < 2:
        state.feature_names = keep
        state.center = center
        state.scale = scale
        state.reason = "insufficient_cluster_support"
        return state
    try:
        clusterer = KMeans(n_clusters=cluster_count, n_init=10, random_state=int(random_state))
        labels = clusterer.fit_predict(x_fit)
        centroids = np.asarray(clusterer.cluster_centers_, dtype=np.float32)
    except Exception as exc:
        state.feature_names = keep
        state.center = center
        state.scale = scale
        state.reason = f"kmeans_failed:{exc}"
        return state
    global_bad = float(np.mean(y_fit)) if len(y_fit) else 0.5
    clusters: list[dict[str, Any]] = []
    bad_ids: list[int] = []
    good_ids: list[int] = []
    bad_rates: list[float] = []
    for cid in range(cluster_count):
        mask = labels == cid
        count = int(np.sum(mask))
        bad_rate = float(np.mean(y_fit[mask])) if count else global_bad
        bad_rates.append(bad_rate)
        delta = bad_rate - global_bad
        role = "neutral"
        if count >= int(min_role_support) and delta >= float(min_bad_rate_delta):
            role = "bad"
            bad_ids.append(cid)
        elif count >= int(min_role_support) and delta <= -float(min_bad_rate_delta):
            role = "good"
            good_ids.append(cid)
        clusters.append(
            {
                "cluster_id": int(cid),
                "count": count,
                "bad_rate": bad_rate,
                "bad_rate_lift": float(bad_rate / max(global_bad, 1e-6)),
                "role": role,
            }
        )
    if not bad_ids or not good_ids:
        order = np.argsort(np.asarray(bad_rates, dtype=np.float64))
        low = int(order[0])
        high = int(order[-1])
        spread = float(bad_rates[high] - bad_rates[low])
        if spread >= float(min_bad_rate_spread):
            if high not in bad_ids:
                bad_ids = [high]
                clusters[high]["role"] = "bad"
            if low not in good_ids:
                good_ids = [low]
                clusters[low]["role"] = "good"
    state.feature_names = keep
    state.enabled = bool(bad_ids and good_ids)
    state.reason = "ok" if state.enabled else "no_separated_error_archetypes"
    state.center = center.astype(np.float32, copy=False)
    state.scale = scale.astype(np.float32, copy=False)
    state.centroids = centroids.astype(np.float32, copy=False)
    state.bad_cluster_ids = [int(c) for c in bad_ids]
    state.good_cluster_ids = [int(c) for c in good_ids]
    state.clusters = clusters
    state.global_bad_rate = float(global_bad)
    return state


def transform_residual_error_archetype_features(
    signature_frame: pd.DataFrame,
    state: ResidualErrorArchetypeState | None,
    *,
    index: Any = None,
) -> pd.DataFrame:
    n = int(len(signature_frame))
    if index is None and isinstance(signature_frame, pd.DataFrame):
        index = signature_frame.index
    if (
        state is None
        or not getattr(state, "enabled", False)
        or getattr(state, "centroids", np.zeros((0, 0))).size == 0
        or not getattr(state, "feature_names", None)
    ):
        return _neutral_base_error_archetype_frame(n, index=index)
    x = _selected_matrix(signature_frame, state.feature_names)
    dim = int(min(x.shape[1], len(state.center), len(state.scale), state.centroids.shape[1]))
    if dim <= 0:
        return _neutral_base_error_archetype_frame(n, index=index)
    scaled = np.nan_to_num(
        (x[:, :dim] - state.center[:dim]) / np.maximum(state.scale[:dim], 1e-6),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)
    centroids = np.asarray(state.centroids[:, :dim], dtype=np.float32)
    distances = np.sqrt(
        np.sum(np.square(scaled[:, None, :] - centroids[None, :, :]), axis=2)
    ).astype(np.float32)
    nearest = np.argmin(distances, axis=1).astype(np.int32)
    nearest_distance = distances[np.arange(n), nearest].astype(np.float32)
    bad_ids = [int(c) for c in getattr(state, "bad_cluster_ids", []) if 0 <= int(c) < len(centroids)]
    good_ids = [int(c) for c in getattr(state, "good_cluster_ids", []) if 0 <= int(c) < len(centroids)]
    nearest_bad_distance = (
        np.min(distances[:, bad_ids], axis=1).astype(np.float32)
        if bad_ids
        else np.zeros(n, dtype=np.float32)
    )
    nearest_good_distance = (
        np.min(distances[:, good_ids], axis=1).astype(np.float32)
        if good_ids
        else np.zeros(n, dtype=np.float32)
    )
    lift_by_cluster = np.ones(len(centroids), dtype=np.float32)
    for cluster in getattr(state, "clusters", []) or []:
        cid = int(cluster.get("cluster_id", -1))
        if 0 <= cid < len(lift_by_cluster):
            lift_by_cluster[cid] = float(cluster.get("bad_rate_lift", 1.0))
    is_bad = np.asarray([1.0 if int(cid) in bad_ids else 0.0 for cid in nearest], dtype=np.float32)
    is_good = np.asarray([1.0 if int(cid) in good_ids else 0.0 for cid in nearest], dtype=np.float32)
    is_neutral = np.clip(1.0 - np.maximum(is_bad, is_good), 0.0, 1.0).astype(np.float32)
    return _frame_from_arrays(
        {
            "base_error_archetype_id": nearest.astype(np.float32),
            "base_error_archetype_is_bad": is_bad,
            "base_error_archetype_is_good": is_good,
            "base_error_archetype_is_neutral": is_neutral,
            "base_error_distance_to_archetype_centroid": nearest_distance,
            "base_error_distance_to_nearest_bad_archetype": nearest_bad_distance,
            "base_error_archetype_oof_bad_rate_lift": lift_by_cluster[nearest],
            "base_error_distance_to_bad_archetype": nearest_bad_distance,
            "base_error_distance_to_good_archetype": nearest_good_distance,
        },
        BASE_ERROR_ARCHETYPE_FEATURE_NAMES,
        index=index,
    )


def _neutral_base_error_archetype_frame(n: int, *, index: Any = None) -> pd.DataFrame:
    zeros = np.zeros(int(n), dtype=np.float32)
    ones = np.ones(int(n), dtype=np.float32)
    return _frame_from_arrays(
        {
            "base_error_archetype_id": zeros,
            "base_error_archetype_is_bad": zeros,
            "base_error_archetype_is_good": zeros,
            "base_error_archetype_is_neutral": ones,
            "base_error_distance_to_archetype_centroid": zeros,
            "base_error_distance_to_nearest_bad_archetype": zeros,
            "base_error_archetype_oof_bad_rate_lift": ones,
            "base_error_distance_to_bad_archetype": zeros,
            "base_error_distance_to_good_archetype": zeros,
        },
        BASE_ERROR_ARCHETYPE_FEATURE_NAMES,
        index=index,
    )


def _as_2d_float(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _selected_matrix(X: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    cols = [str(c) for c in X.columns]
    frame = X.copy()
    frame.columns = cols
    data = pd.DataFrame(index=frame.index)
    for name in feature_names:
        if name in frame.columns:
            data[name] = pd.to_numeric(frame[name], errors="coerce")
        else:
            data[name] = 0.0
    return (
        data.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32, copy=False)
    )


def _component_count(x: np.ndarray, requested: int) -> int:
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        return 0
    return int(max(1, min(int(requested), int(x.shape[1]), int(x.shape[0] - 1))))


def _component_frame(z: np.ndarray, names: Sequence[str], *, index: Any = None) -> pd.DataFrame:
    n = int(z.shape[0])
    data: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        if i < z.shape[1]:
            data[str(name)] = z[:, i]
        else:
            data[str(name)] = np.zeros(n, dtype=np.float32)
    return _frame_from_arrays(data, list(map(str, names)), index=index)


def _empty_frame(n: int, names: Sequence[str], *, index: Any = None) -> pd.DataFrame:
    return pd.DataFrame(
        {str(name): np.zeros(int(n), dtype=np.float32) for name in names},
        index=index,
    )


def _frame_from_arrays(data: dict[str, Any], names: Sequence[str], *, index: Any = None) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            str(name): np.nan_to_num(
                np.asarray(data.get(str(name), np.zeros(0)), dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            for name in names
        },
        index=index,
    )
    return out.astype(np.float32)


def _reference_rows(z: np.ndarray, *, random_state: int) -> np.ndarray:
    if len(z) <= RAW_STATE_MAX_REFERENCE_ROWS:
        return z.astype(np.float32, copy=False)
    rng = np.random.default_rng(int(random_state))
    idx = np.sort(rng.choice(len(z), size=RAW_STATE_MAX_REFERENCE_ROWS, replace=False))
    return z[idx].astype(np.float32, copy=False)


def _fit_distribution_bins(
    values: Any,
    *,
    bins: int = RAW_STATE_DISTRIBUTION_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    arr = _as_2d_float(values)
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
        )
    bin_count = max(2, int(bins))
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)
    edges = np.zeros((arr.shape[1], bin_count + 1), dtype=np.float32)
    probs = np.zeros((arr.shape[1], bin_count), dtype=np.float32)
    eps = 1e-6
    for j in range(arr.shape[1]):
        col = np.nan_to_num(arr[:, j], nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float64,
            copy=False,
        )
        try:
            q = np.nanquantile(col, quantiles).astype(np.float64, copy=False)
        except Exception:
            q = np.linspace(0.0, 1.0, bin_count + 1, dtype=np.float64)
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        lo = float(q[0]) if q.size else 0.0
        hi = float(q[-1]) if q.size else lo
        if not hi > lo:
            pad = max(abs(lo) * 1e-6, 1e-6)
            q = np.linspace(lo - pad, lo + pad, bin_count + 1, dtype=np.float64)
        else:
            min_step = max((hi - lo) * 1e-9, 1e-9)
            for k in range(1, len(q)):
                if q[k] <= q[k - 1]:
                    q[k] = q[k - 1] + min_step
        idx = np.searchsorted(q[1:-1], col, side="right")
        idx = np.clip(idx, 0, bin_count - 1)
        counts = np.bincount(idx, minlength=bin_count).astype(np.float64)
        p = (counts + eps) / max(float(counts.sum() + eps * bin_count), eps)
        edges[j] = q.astype(np.float32, copy=False)
        probs[j] = p.astype(np.float32, copy=False)
    return edges, probs


def _distribution_scores(
    values: Any,
    edges: np.ndarray,
    probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = _as_2d_float(values)
    n = int(arr.shape[0])
    zero = np.zeros(n, dtype=np.float32)
    if arr.shape[1] == 0 or edges.size == 0 or probs.size == 0:
        return zero, zero, zero, zero
    dim = min(arr.shape[1], int(edges.shape[0]), int(probs.shape[0]))
    if dim <= 0:
        return zero, zero, zero, zero
    psi_sum = np.zeros(n, dtype=np.float64)
    psi_max = np.zeros(n, dtype=np.float64)
    ks_sum = np.zeros(n, dtype=np.float64)
    ks_max = np.zeros(n, dtype=np.float64)
    used = 0
    eps = 1e-6
    for j in range(dim):
        edge = np.asarray(edges[j], dtype=np.float64)
        ref_prob = np.asarray(probs[j], dtype=np.float64)
        bin_count = min(max(len(edge) - 1, 0), len(ref_prob))
        if bin_count <= 0:
            continue
        edge = edge[: bin_count + 1]
        ref_prob = np.nan_to_num(ref_prob[:bin_count], nan=0.0, posinf=0.0, neginf=0.0)
        total = float(np.sum(ref_prob))
        if total <= eps:
            continue
        ref_prob = np.clip(ref_prob / total, eps, 1.0)
        ref_prob = ref_prob / max(float(np.sum(ref_prob)), eps)
        col_values = arr[:, j]
        idx = np.searchsorted(edge[1:-1], col_values, side="right")
        idx = np.clip(idx, 0, bin_count - 1)
        outside = (col_values < edge[0]) | (col_values > edge[-1])
        obs_prob = np.clip(ref_prob[idx], eps, 1.0)
        obs_prob = np.where(outside, eps, obs_prob)
        # Row-local PSI proxy: rarity of the observed train-distribution bin.
        psi = np.maximum(0.0, (1.0 - obs_prob) * np.log(1.0 / obs_prob))
        cdf_after = np.cumsum(ref_prob)
        cdf_before = np.concatenate(([0.0], cdf_after[:-1]))
        # Row-local KS proxy against the train CDF; high values identify tail bins.
        ks = np.clip(np.maximum(cdf_before[idx], 1.0 - cdf_after[idx]), 0.0, 1.0)
        ks = np.where(outside, 1.0, ks)
        psi_sum += psi
        psi_max = np.maximum(psi_max, psi)
        ks_sum += ks
        ks_max = np.maximum(ks_max, ks)
        used += 1
    if used <= 0:
        return zero, zero, zero, zero
    return (
        np.nan_to_num(psi_sum / used, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        np.nan_to_num(psi_max, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        np.nan_to_num(ks_sum / used, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        np.nan_to_num(ks_max, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
    )


def _fit_gaussian(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    arr = _as_2d_float(z)
    dim = int(arr.shape[1]) if arr.ndim == 2 and arr.shape[1] else 1
    if arr.shape[0] == 0:
        mean = np.zeros(dim, dtype=np.float32)
        inv_cov = np.eye(dim, dtype=np.float32)
        return mean, inv_cov, 0.0
    mean = np.mean(arr, axis=0).astype(np.float32)
    centered = arr - mean
    if arr.shape[0] <= 1:
        cov = np.eye(dim, dtype=np.float64)
    else:
        cov = np.cov(centered, rowvar=False)
        cov = np.asarray(cov, dtype=np.float64)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
    diag_scale = float(np.nanmean(np.diag(cov))) if cov.size else 1.0
    ridge = max(diag_scale * 1e-3, 1e-6)
    cov = cov + np.eye(dim, dtype=np.float64) * ridge
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0 or not np.isfinite(logdet):
        cov = cov + np.eye(dim, dtype=np.float64) * max(ridge * 10.0, 1e-5)
        sign, logdet = np.linalg.slogdet(cov)
    try:
        inv_cov = np.linalg.pinv(cov).astype(np.float32)
    except Exception:
        inv_cov = np.eye(dim, dtype=np.float32)
        logdet = 0.0
    return mean, inv_cov, float(logdet if np.isfinite(logdet) else 0.0)


def _mahalanobis(z: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    arr = _as_2d_float(z)
    if mean.size == 0 or inv_cov.size == 0:
        return np.zeros(arr.shape[0], dtype=np.float32)
    dim = min(arr.shape[1], len(mean), inv_cov.shape[0], inv_cov.shape[1])
    if dim <= 0:
        return np.zeros(arr.shape[0], dtype=np.float32)
    centered = arr[:, :dim] - mean[:dim]
    quad = np.einsum("ij,jk,ik->i", centered, inv_cov[:dim, :dim], centered)
    return np.sqrt(np.maximum(quad, 0.0)).astype(np.float32)


def _gaussian_log_likelihood(z: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray, logdet: float) -> np.ndarray:
    arr = _as_2d_float(z)
    dim = min(arr.shape[1], len(mean), inv_cov.shape[0], inv_cov.shape[1])
    if dim <= 0:
        return np.zeros(arr.shape[0], dtype=np.float32)
    centered = arr[:, :dim] - mean[:dim]
    quad = np.einsum("ij,jk,ik->i", centered, inv_cov[:dim, :dim], centered)
    ll = -0.5 * (dim * np.log(2.0 * np.pi) + float(logdet) + quad)
    return np.nan_to_num(ll, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _fit_knn(z: np.ndarray) -> NearestNeighbors | None:
    arr = _as_2d_float(z)
    if arr.shape[0] < 2:
        return None
    n_neighbors = max(1, min(RAW_STATE_KNN_NEIGHBORS, arr.shape[0]))
    try:
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        knn.fit(arr)
        return knn
    except Exception:
        return None


def _knn_distance(z: np.ndarray, knn: NearestNeighbors | None) -> np.ndarray:
    arr = _as_2d_float(z)
    if knn is None:
        return np.zeros(arr.shape[0], dtype=np.float32)
    try:
        distances, _ = knn.kneighbors(arr, return_distance=True)
        return np.mean(distances, axis=1).astype(np.float32)
    except Exception:
        return np.zeros(arr.shape[0], dtype=np.float32)


def _fit_centroids(z: np.ndarray, *, random_state: int) -> np.ndarray:
    arr = _as_2d_float(z)
    if arr.shape[0] == 0:
        return np.zeros((0, arr.shape[1]), dtype=np.float32)
    n_components = max(1, min(RAW_STATE_GMM_COMPONENTS, arr.shape[0]))
    if n_components == 1:
        return np.mean(arr, axis=0, keepdims=True).astype(np.float32)
    try:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            random_state=int(random_state),
            max_iter=100,
            reg_covar=1e-5,
        )
        gmm.fit(arr)
        return gmm.means_.astype(np.float32)
    except Exception:
        return np.mean(arr, axis=0, keepdims=True).astype(np.float32)


def _min_cluster_distance(z: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    arr = _as_2d_float(z)
    if centroids.size == 0:
        return np.zeros(arr.shape[0], dtype=np.float32)
    c = _as_2d_float(centroids)
    diff = arr[:, None, : c.shape[1]] - c[None, :, : c.shape[1]]
    return np.min(np.linalg.norm(diff, axis=2), axis=1).astype(np.float32)


def _transition_vectors(z: np.ndarray, *, timestamps: Any = None, assets: Any = None) -> np.ndarray:
    arr = _as_2d_float(z)
    out = np.zeros_like(arr, dtype=np.float32)
    if arr.shape[0] <= 1:
        return out
    ts = _timestamp_order_values(timestamps, len(arr))
    asset_values = _asset_values(assets, len(arr))
    if ts is None and asset_values is None:
        out[1:] = arr[1:] - arr[:-1]
        return out
    if asset_values is None:
        asset_values = np.repeat("", len(arr)).astype(object)
    if ts is not None:
        order = np.lexsort((np.arange(len(arr)), ts))
    else:
        order = np.arange(len(arr))
    last_by_asset: dict[str, int] = {}
    for pos in order:
        asset = str(asset_values[pos])
        prev = last_by_asset.get(asset)
        if prev is not None:
            out[pos] = arr[pos] - arr[prev]
        last_by_asset[asset] = int(pos)
    return out


def _timestamp_order_values(timestamps: Any, n: int) -> np.ndarray | None:
    if timestamps is None:
        return None
    try:
        ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce").astype("int64").to_numpy()
    except Exception:
        return None
    if len(ts) != n:
        return None
    return np.where(ts == pd.NaT.value, np.iinfo(np.int64).max, ts)


def _asset_values(assets: Any, n: int) -> np.ndarray | None:
    if assets is None:
        return None
    try:
        values = np.asarray(assets, dtype=object)
    except Exception:
        return None
    if len(values) != n:
        return None
    return values


def _tod_buckets(timestamps: Any, n: int) -> np.ndarray | None:
    if timestamps is None:
        return None
    try:
        ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    except Exception:
        return None
    if len(ts) != n:
        return None
    hours = ts.dt.hour.to_numpy(dtype=np.float64)
    valid = np.isfinite(hours)
    buckets = np.full(n, -1, dtype=np.int16)
    buckets[valid] = np.floor(hours[valid] / (24.0 / RAW_STATE_TOD_BUCKETS)).astype(np.int16)
    return np.clip(buckets, -1, RAW_STATE_TOD_BUCKETS - 1)


def _fit_tod_bucket_stats(
    z: np.ndarray,
    *,
    timestamps: Any,
    fallback_mean: np.ndarray,
    fallback_inv_cov: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    buckets = _tod_buckets(timestamps, len(z))
    stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if buckets is None:
        return stats
    for bucket in range(RAW_STATE_TOD_BUCKETS):
        mask = buckets == bucket
        if int(np.sum(mask)) < 10:
            stats[bucket] = (fallback_mean.astype(np.float32), fallback_inv_cov.astype(np.float32))
            continue
        mean, inv_cov, _ = _fit_gaussian(z[mask])
        stats[bucket] = (mean, inv_cov)
    return stats


def _tod_mahalanobis(z: np.ndarray, *, timestamps: Any, state: RawStateArchetypeState) -> np.ndarray:
    arr = _as_2d_float(z)
    buckets = _tod_buckets(timestamps, len(arr))
    if buckets is None or not state.tod_bucket_stats:
        return _mahalanobis(arr, state.mean, state.inv_cov)
    out = np.zeros(len(arr), dtype=np.float32)
    for bucket in np.unique(buckets):
        mask = buckets == bucket
        mean, inv_cov = state.tod_bucket_stats.get(int(bucket), (state.mean, state.inv_cov))
        out[mask] = _mahalanobis(arr[mask], mean, inv_cov)
    return out.astype(np.float32)
