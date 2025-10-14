"""
Cluster constraint utilities for enforcing distribution targets in 4D space.

This module provides post-processing helpers that operate purely in the given
feature space (e.g., 4D mapping) to:
- Split oversized clusters
- Merge tail clusters to reach top-K coverage
- Balance top-K clusters into a target percentage range

All operations are metric-aware and avoid any re-embedding. They work with
standard numpy arrays and integer label vectors (noise as -1 if present).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    # Prefer project-local sklearn shim if available
    from src.utils.sklearn_utils import KMeans
except Exception:  # pragma: no cover - fallback
    from sklearn.cluster import KMeans  # type: ignore


def _compute_centroids(
    X: np.ndarray,
    labels: np.ndarray,
    include_labels: Optional[Iterable[int]] = None,
    metric: str = "euclidean",
) -> Dict[int, np.ndarray]:
    """
    Compute per-cluster prototypes in the same space as X.

    For euclidean: arithmetic mean. For cosine: mean followed by L2-normalization.
    """
    if include_labels is None:
        include_labels = [l for l in np.unique(labels) if l >= 0]

    centroids: Dict[int, np.ndarray] = {}
    for c in include_labels:
        idx = labels == c
        if not np.any(idx):
            continue
        centroid = X[idx].mean(axis=0)
        if metric == "cosine":
            norm = np.linalg.norm(centroid) + 1e-12
            centroid = centroid / norm
        centroids[int(c)] = centroid
    return centroids


def _pairwise_distance(a: np.ndarray, b: np.ndarray, metric: str = "euclidean") -> float:
    if metric == "cosine":
        # distance = 1 - cosine_similarity
        denom = (np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)
        return float(1.0 - np.dot(a, b) / denom)
    # default euclidean
    return float(np.linalg.norm(a - b))


def _nearest_centroid(
    X: np.ndarray,
    centroids: Dict[int, np.ndarray],
    metric: str = "euclidean",
) -> np.ndarray:
    labels = np.empty(X.shape[0], dtype=int)
    centroid_items = list(centroids.items())
    keys = [k for k, _ in centroid_items]
    mats = np.stack([v for _, v in centroid_items], axis=0)
    if metric == "cosine":
        # Normalize X and mats to compute cosine distances efficiently
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        Mn = mats / (np.linalg.norm(mats, axis=1, keepdims=True) + 1e-12)
        # cosine similarity -> pick argmax sim => argmin distance
        sims = Xn @ Mn.T
        arg = np.argmax(sims, axis=1)
        return np.array([keys[i] for i in arg], dtype=int)
    # Euclidean distances
    # Compute squared distances efficiently: ||x-m||^2 = ||x||^2 + ||m||^2 - 2 x.m
    x2 = np.sum(X * X, axis=1, keepdims=True)
    m2 = np.sum(mats * mats, axis=1, keepdims=True).T
    dots = X @ mats.T
    d2 = x2 + m2 - 2.0 * dots
    arg = np.argmin(d2, axis=1)
    return np.array([keys[i] for i in arg], dtype=int)


def _split_cluster_with_kmeans(
    X_cluster: np.ndarray,
    k: int,
    random_state: int,
) -> np.ndarray:
    k = max(2, int(k))
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return km.fit_predict(X_cluster)


def _calc_props(labels: np.ndarray, total: int) -> Dict[int, float]:
    counts = {int(c): int(np.sum(labels == c)) for c in np.unique(labels) if c >= 0}
    return {c: cnt / max(1, total) for c, cnt in counts.items()}


def split_giant_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    max_prop: float = 0.12,
    target_range: Tuple[float, float] = (0.03, 0.08),
    metric: str = "euclidean",
    random_state: int = 42,
) -> np.ndarray:
    N = len(labels)
    new_labels = labels.copy()
    positive = [c for c in np.unique(new_labels) if c >= 0]
    if not positive:
        return new_labels
    next_label = (max(positive) + 1) if positive else 0

    for c in list(np.unique(new_labels)):
        if c < 0:
            continue
        idx = np.where(new_labels == c)[0]
        prop = len(idx) / max(1, N)
        if prop <= max_prop or len(idx) < 4:
            continue
        # determine splits; aim for upper target bound
        upper = max(target_range[1], 1e-6)
        k = int(np.ceil(prop / upper))
        k = max(2, k)
        sub = _split_cluster_with_kmeans(X[idx], k=k, random_state=random_state)
        # assign new labels
        for sk in np.unique(sub):
            new_labels[idx[sub == sk]] = next_label
            next_label += 1
    return new_labels


def merge_tail_into_topk(
    X: np.ndarray,
    labels: np.ndarray,
    k: int = 20,
    coverage_target: Tuple[float, float] = (0.90, 0.95),
    metric: str = "euclidean",
    max_iters: int = 5,
) -> np.ndarray:
    new_labels = labels.copy()
    N = len(new_labels)
    for _ in range(max_iters):
        # compute sizes excluding noise
        counts = {int(c): int(np.sum(new_labels == c)) for c in np.unique(new_labels) if c >= 0}
        if not counts:
            return new_labels
        order = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        top = [c for c, _ in order[:k]]
        coverage = sum(cnt for _, cnt in order[:k]) / max(1, N)
        if coverage >= coverage_target[0]:
            break
        # merge the smallest non-top cluster into nearest top centroid
        tail = [c for c, _ in order[k:]]
        if not tail:
            break
        centroids = _compute_centroids(X, new_labels, include_labels=top, metric=metric)
        # pick smallest tail cluster
        smallest = min(tail, key=lambda c: counts[c])
        idx = np.where(new_labels == smallest)[0]
        if idx.size == 0:
            # already empty due to previous merges
            continue
        # assign to nearest top centroid
        if not centroids:
            break
        assigned = _nearest_centroid(X[idx], centroids, metric=metric)
        # map centroid keys back to labels
        new_labels[idx] = assigned
    return new_labels


def balance_topk_range(
    X: np.ndarray,
    labels: np.ndarray,
    k: int = 20,
    target_range: Tuple[float, float] = (0.03, 0.08),
    max_prop: float = 0.12,
    metric: str = "euclidean",
    random_state: int = 42,
    max_iters: int = 5,
) -> np.ndarray:
    new_labels = labels.copy()
    N = len(new_labels)
    for _ in range(max_iters):
        # compute sizes and top-k
        counts = {int(c): int(np.sum(new_labels == c)) for c in np.unique(new_labels) if c >= 0}
        if not counts:
            return new_labels
        order = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        top = [c for c, _ in order[:k]]
        changed = False

        # handle oversized first
        for c in top:
            prop = counts[c] / max(1, N)
            if prop > max(target_range[1], max_prop):
                idx = np.where(new_labels == c)[0]
                if idx.size < 4:
                    continue
                k_splits = int(np.ceil(prop / max(target_range[1], 1e-6)))
                k_splits = max(2, k_splits)
                sub = _split_cluster_with_kmeans(X[idx], k=k_splits, random_state=random_state)
                next_label = (max([l for l in np.unique(new_labels) if l >= 0], default=-1) + 1)
                for sk in np.unique(sub):
                    new_labels[idx[sub == sk]] = next_label
                    next_label += 1
                changed = True

        # recompute after splits
        if changed:
            continue

        # handle undersized
        counts = {int(c): int(np.sum(new_labels == c)) for c in np.unique(new_labels) if c >= 0}
        order = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        top = [c for c, _ in order[:k]]
        centroids = _compute_centroids(X, new_labels, include_labels=top, metric=metric)
        for c in top:
            prop = counts[c] / max(1, N)
            if prop < target_range[0]:
                # merge with nearest other top centroid
                others = {t: v for t, v in centroids.items() if t != c}
                if not others:
                    continue
                c_vec = centroids[c]
                # find nearest neighbor
                nearest = min(others.items(), key=lambda kv: _pairwise_distance(c_vec, kv[1], metric=metric))[0]
                idx = np.where(new_labels == c)[0]
                new_labels[idx] = nearest
                changed = True
        if not changed:
            break
    return new_labels


def enforce_cluster_constraints(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    noise_label: int = -1,
    max_legit_prop: float = 0.12,
    target_topk: int = 20,
    coverage_target: Tuple[float, float] = (0.90, 0.95),
    target_range: Tuple[float, float] = (0.03, 0.08),
    metric: str = "euclidean",
    random_state: int = 42,
    max_iters: int = 4,
) -> np.ndarray:
    """
    Apply split/merge passes to satisfy size and coverage constraints.

    The operations avoid changing the feature space and work only on labels.
    """
    new_labels = labels.copy()
    for _ in range(max_iters):
        before = new_labels.copy()
        # 1) Split giants
        new_labels = split_giant_clusters(
            X, new_labels, max_prop=max_legit_prop, target_range=target_range, metric=metric, random_state=random_state
        )
        # 2) Merge tail into top-k until coverage target is reached
        new_labels = merge_tail_into_topk(
            X, new_labels, k=target_topk, coverage_target=coverage_target, metric=metric
        )
        # 3) Balance top-k into desired range (and respect max cap)
        new_labels = balance_topk_range(
            X, new_labels, k=target_topk, target_range=target_range, max_prop=max_legit_prop, metric=metric, random_state=random_state
        )
        if np.array_equal(before, new_labels):
            break
    return new_labels


def fit_hdbscan_with_noise_target(
    X: np.ndarray,
    base_kwargs: Dict[str, object],
    target_range: Tuple[float, float] = (0.05, 0.10),
    iters: int = 10,
) -> "hdbscan.HDBSCAN":  # type: ignore[name-defined]
    """
    Ternary-search-like tuning for HDBSCAN min_samples to reach noise in target_range.
    Returns the fitted clusterer that best matches the target.
    """
    import hdbscan  # local import

    N = X.shape[0]
    low = 1
    high = max(5, int(np.sqrt(N)))
    best = None
    best_gap = float("inf")

    def _fit(ms: int):
        params = dict(base_kwargs)
        params["min_samples"] = int(ms)
        clusterer = hdbscan.HDBSCAN(**params)
        labels = clusterer.fit_predict(X)
        noise = np.mean(labels == -1)
        return clusterer, noise

    for _ in range(iters):
        mid = (low + high) // 2
        clusterer, noise = _fit(mid)
        gap = 0.0
        if noise < target_range[0]:
            gap = target_range[0] - noise
        elif noise > target_range[1]:
            gap = noise - target_range[1]
        else:
            return clusterer
        if gap < best_gap:
            best = clusterer
            best_gap = gap
        # Adjust search bounds: increase min_samples to increase noise
        if noise < target_range[0]:
            low = mid + 1
        else:
            high = mid - 1
    return best if best is not None else _fit(max(2, int(np.sqrt(N))))[0]

