"""
4D-safe cluster constraint tools for market analysis pipeline.

This module provides non-invasive helpers to:
- Tune HDBSCAN noise into [5%, 10%]
- Split clusters exceeding 12% of total samples
- Merge tail clusters to reach 90–95% coverage in the top 20
- Balance top-20 clusters to 3–8% each

All steps operate in the existing feature space (e.g. 4D mapping) and respect
the chosen metric (euclidean or cosine). No re-embedding occurs.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:
    from src.utils.sklearn_utils import KMeans
except Exception:  # pragma: no cover
    from sklearn.cluster import KMeans  # type: ignore

def _compute_centroids(
    X: np.ndarray,
    labels: np.ndarray,
    include_labels: Optional[Iterable[int]] = None,
    metric: str = "euclidean",
) -> Dict[int, np.ndarray]:
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
        denom = (np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)
        return float(1.0 - (np.dot(a, b) / denom))
    return float(np.linalg.norm(a - b))

def _nearest_centroid(X: np.ndarray, centroids: Dict[int, np.ndarray], metric: str) -> np.ndarray:
    keys = list(centroids.keys())
    mats = np.stack([centroids[k] for k in keys], axis=0)
    if metric == "cosine":
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        Mn = mats / (np.linalg.norm(mats, axis=1, keepdims=True) + 1e-12)
        sims = Xn @ Mn.T
        arg = np.argmax(sims, axis=1)
    else:
        x2 = np.sum(X * X, axis=1, keepdims=True)
        m2 = np.sum(mats * mats, axis=1, keepdims=True).T
        dots = X @ mats.T
        d2 = x2 + m2 - 2.0 * dots
        arg = np.argmin(d2, axis=1)
    return np.array([keys[i] for i in arg], dtype=int)

def _split_with_kmeans(X: np.ndarray, k: int, random_state: int) -> np.ndarray:
    k = max(2, int(k))
    return KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X)

def fit_hdbscan_with_noise_target(
    X: np.ndarray,
    base_kwargs: Dict[str, object],
    target_range: Tuple[float, float] = (0.05, 0.10),
    iters: int = 10,
):
    """
    Tune HDBSCAN min_samples to reach noise in the desired range.
    Returns a fitted clusterer.
    """
    import hdbscan

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
        if target_range[0] <= noise <= target_range[1]:
            return clusterer
        gap = target_range[0] - noise if noise < target_range[0] else noise - target_range[1]
        if gap < best_gap:
            best = clusterer
            best_gap = gap
        if noise < target_range[0]:
            low = mid + 1
        else:
            high = mid - 1
    return best if best is not None else _fit(max(2, int(np.sqrt(N))))[0]

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
    next_label = (max(positive) + 1) if positive else 0
    for c in list(np.unique(new_labels)):
        if c < 0:
            continue
        idx = np.where(new_labels == c)[0]
        prop = len(idx) / max(1, N)
        if prop <= max_prop or len(idx) < 4:
            continue
        upper = max(target_range[1], 1e-6)
        k = int(np.ceil(prop / upper))
        k = max(2, k)
        # Make splitting extremely aggressive for large clusters
        if prop > 0.30:  # If cluster is >30%, split very aggressively
            k = max(k, int(prop * 40))  # Aim for ~2.5% per sub-cluster
        elif prop > 0.20:  # If cluster is >20%, split aggressively
            k = max(k, int(prop * 25))  # Aim for 4% per sub-cluster
        elif prop > 0.15:  # If cluster is >15%, split aggressively
            k = max(k, int(prop * 20))  # Aim for 5% per sub-cluster
        elif prop > 0.10:  # If cluster is >10%, split moderately
            k = max(k, int(prop * 15))  # Aim for ~6-7% per sub-cluster
        elif prop > 0.08:  # If cluster is >8%, split into ~3-5 sub-clusters
            k = max(k, int(prop * 12))  # Aim for ~8% per sub-cluster
        sub = _split_with_kmeans(X[idx], k=k, random_state=random_state)
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
        counts = {int(c): int(np.sum(new_labels == c)) for c in np.unique(new_labels) if c >= 0}
        if not counts:
            return new_labels
        order = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        top = [c for c, _ in order[:k]]
        coverage = sum(cnt for _, cnt in order[:k]) / max(1, N)
        if coverage >= coverage_target[0]:
            break
        tail = [c for c, _ in order[k:]]
        if not tail:
            break
        centroids = _compute_centroids(X, new_labels, include_labels=top, metric=metric)
        smallest = min(tail, key=lambda c: counts[c])
        idx = np.where(new_labels == smallest)[0]
        if idx.size == 0 or not centroids:
            break
        new_labels[idx] = _nearest_centroid(X[idx], centroids, metric=metric)
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
        counts = {int(c): int(np.sum(new_labels == c)) for c in np.unique(new_labels) if c >= 0}
        if not counts:
            return new_labels
        order = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        top = [c for c, _ in order[:k]]
        changed = False

        # split oversized first
        for c in top:
            prop = counts[c] / max(1, N)
            if prop > max(target_range[1], max_prop):
                idx = np.where(new_labels == c)[0]
                if idx.size < 4:
                    continue
                k_splits = int(np.ceil(prop / max(target_range[1], 1e-6)))
                k_splits = max(2, k_splits)
                sub = _split_with_kmeans(X[idx], k=k_splits, random_state=random_state)
                next_label = (max([l for l in np.unique(new_labels) if l >= 0], default=-1) + 1)
                for sk in np.unique(sub):
                    new_labels[idx[sub == sk]] = next_label
                    next_label += 1
                changed = True
        if changed:
            continue

        # merge undersized
        counts = {int(c): int(np.sum(new_labels == c)) for c in np.unique(new_labels) if c >= 0}
        order = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        top = [c for c, _ in order[:k]]
        centroids = _compute_centroids(X, new_labels, include_labels=top, metric=metric)
        for c in top:
            prop = counts[c] / max(1, N)
            if prop < target_range[0]:
                others = {t: v for t, v in centroids.items() if t != c}
                if not others:
                    continue
                c_vec = centroids[c]
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
    new_labels = labels.copy()
    for _ in range(max_iters):
        before = new_labels.copy()
        new_labels = split_giant_clusters(
            X, new_labels, max_prop=max_legit_prop, target_range=target_range, metric=metric, random_state=random_state
        )
        new_labels = merge_tail_into_topk(
            X, new_labels, k=target_topk, coverage_target=coverage_target, metric=metric
        )
        new_labels = balance_topk_range(
            X, new_labels, k=target_topk, target_range=target_range, max_prop=max_legit_prop, metric=metric, random_state=random_state
        )
        if np.array_equal(before, new_labels):
            break
    return new_labels

def summarize_distribution(labels: np.ndarray, topk: int = 20) -> dict:
    """Return noise fraction, largest cluster fraction, and top-k coverage."""
    total = len(labels)
    noise_fraction = float(np.mean(labels == -1)) if total > 0 else 0.0
    positive = [l for l in np.unique(labels) if l >= 0]
    counts = np.array([np.sum(labels == l) for l in positive]) if positive else np.array([])
    largest_fraction = float(np.max(counts) / total) if counts.size > 0 and total > 0 else 0.0
    top_counts = np.sort(counts)[::-1][:topk]
    topk_coverage = float(np.sum(top_counts) / total) if total > 0 else 0.0
    return {
        "noise_fraction": noise_fraction,
        "largest_cluster_fraction": largest_fraction,
        "topk_coverage": topk_coverage,
    }
