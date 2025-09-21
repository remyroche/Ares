"""
Usage helpers to apply 4D-safe clustering constraints in the training pipeline.

This module does not alter existing components. It provides a function
`apply_constraints_to_hdbscan` that you can call after producing an embedding
or a 4D mapping and initial HDBSCAN labels, to:
- tune noise into [5%, 10%] (optional)
- enforce max 12% per legitimate cluster
- reach 90–95% top-20 coverage
- balance top-20 clusters to 3–8% each
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .cluster_constraints import (
    fit_hdbscan_with_noise_target,
    enforce_cluster_constraints,
    summarize_distribution,
)


def apply_constraints_to_hdbscan(
    X_4d: np.ndarray,
    *,
    initial_labels: Optional[np.ndarray] = None,
    metric: str = "euclidean",
    hdbscan_params: Optional[Dict[str, object]] = None,
    noise_target: Tuple[float, float] = (0.05, 0.10),
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Fit HDBSCAN (with optional noise tuning) and enforce distribution constraints in-place.

    Args:
        X_4d: Feature array in 4D (or existing embedding) of shape (N, 4 or D).
        initial_labels: If provided, skip fitting and only enforce constraints on these labels.
        metric: Distance metric for both HDBSCAN and constraint steps ("euclidean" or "cosine").
        hdbscan_params: Parameters for HDBSCAN. You can omit min_samples; this function tunes it.
        noise_target: Desired noise fraction range.
        random_state: Random seed for deterministic splits.

    Returns:
        Dict with keys: labels, metrics, model (if fitted), and params used.
    """
    try:
        import hdbscan
    except Exception as e:  # pragma: no cover
        raise ImportError("hdbscan is required for this function. Please install hdbscan.") from e

    X = np.asarray(X_4d)
    params = dict(hdbscan_params or {})
    params.setdefault("metric", metric)
    params.setdefault("cluster_selection_method", "leaf")

    if initial_labels is None:
        # Fit with noise tuning
        clusterer = fit_hdbscan_with_noise_target(X, params, target_range=noise_target)
        labels = clusterer.labels_
        if labels is None or len(labels) != len(X):
            labels = clusterer.fit_predict(X)
        model = clusterer
    else:
        labels = np.asarray(initial_labels).copy()
        model = None

    # Enforce constraints
    labels = enforce_cluster_constraints(
        X,
        labels,
        noise_label=-1,
        max_legit_prop=0.12,
        target_topk=20,
        coverage_target=(0.90, 0.95),
        target_range=(0.03, 0.08),
        metric=metric,
        random_state=random_state,
    )

    # Summarize
    dist = summarize_distribution(labels, topk=20)
    result = {
        "labels": labels,
        "metrics": dist,
        "model": model,
        "params": params,
    }
    return result

