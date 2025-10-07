"""Utility functions for post-processing learned embeddings prior to stacking."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _safe_array(data: np.ndarray) -> np.ndarray:
    """Convert input to a 2D numpy float array."""
    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def filter_embedding_features(
    parent_features: np.ndarray,
    embedding_features: np.ndarray,
    target: Optional[np.ndarray] = None,
    parent_feature_names: Optional[Sequence[str]] = None,
    embedding_names: Optional[Sequence[str]] = None,
    corr_threshold: float = 0.8,
    ic_threshold: float = 0.05,
    min_embeddings: int = 6,
    max_embeddings: int = 10,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Filter embedding dimensions that are redundant or uninformative."""

    parent = _safe_array(parent_features)
    embeddings = _safe_array(embedding_features)

    if parent.shape[0] != embeddings.shape[0]:
        raise ValueError("Parent features and embeddings must have the same number of rows")

    n_samples, n_embeddings = embeddings.shape
    embedding_names = (
        list(embedding_names)
        if embedding_names is not None
        else [f"embedding_{i}" for i in range(n_embeddings)]
    )
    parent_feature_names = (
        list(parent_feature_names)
        if parent_feature_names is not None
        else [f"parent_{i}" for i in range(parent.shape[1])]
    )

    # Handle degenerate cases early
    if n_embeddings == 0:
        metadata: Dict[str, object] = {
            "dropped_due_to_corr": [],
            "dropped_due_to_ic": [],
            "max_abs_corr": np.array([]),
            "ic_scores": np.array([]),
            "selected_indices": [],
            "retained_embedding_names": [],
            "within_budget": True,
        }
        return embeddings, metadata

    combined = np.hstack([embeddings, parent])
    corr_matrix = np.corrcoef(combined, rowvar=False)
    embed_parent_corr = corr_matrix[:n_embeddings, n_embeddings:]
    embed_parent_corr = np.nan_to_num(embed_parent_corr, nan=0.0)
    max_abs_corr = np.max(np.abs(embed_parent_corr), axis=1)

    drop_corr_mask = max_abs_corr > corr_threshold
    dropped_due_to_corr = [embedding_names[i] for i in np.where(drop_corr_mask)[0]]

    target_array: Optional[np.ndarray]
    if target is None:
        target_array = None
    else:
        target_array = np.asarray(target, dtype=float).reshape(-1)
        if target_array.size != n_samples:
            target_array = target_array[:n_samples]
        if np.std(target_array) < 1e-8:
            target_array = None

    ic_scores = np.zeros(n_embeddings)
    if target_array is not None:
        for idx in range(n_embeddings):
            column = embeddings[:, idx]
            if np.std(column) < 1e-8:
                ic_scores[idx] = 0.0
                continue
            ic_scores[idx] = np.corrcoef(column, target_array)[0, 1]
    else:
        # Without a target, fall back to variance proxy (higher variance treated as more useful)
        ic_scores = np.std(embeddings, axis=0)

    ic_scores = np.nan_to_num(ic_scores, nan=0.0)
    ic_abs = np.abs(ic_scores)

    drop_ic_mask = ic_abs < ic_threshold if target_array is not None else np.zeros(n_embeddings, dtype=bool)
    dropped_due_to_ic = [embedding_names[i] for i in np.where(drop_ic_mask)[0]]

    keep_mask = ~(drop_corr_mask | drop_ic_mask)
    candidate_indices = np.where(keep_mask)[0]

    # Sort candidates by descending information content proxy
    sorted_candidates = list(candidate_indices[np.argsort(-ic_abs[candidate_indices])])

    if len(sorted_candidates) > max_embeddings:
        sorted_candidates = sorted_candidates[:max_embeddings]

    final_indices = sorted(sorted_candidates)
    within_budget = min_embeddings <= len(final_indices) <= max_embeddings

    filtered_embeddings = embeddings[:, final_indices] if final_indices else np.zeros((n_samples, 0))

    metadata = {
        "dropped_due_to_corr": dropped_due_to_corr,
        "dropped_due_to_ic": dropped_due_to_ic,
        "max_abs_corr": max_abs_corr,
        "ic_scores": ic_scores,
        "selected_indices": final_indices,
        "retained_embedding_names": [embedding_names[i] for i in final_indices],
        "within_budget": within_budget,
        "total_embeddings": n_embeddings,
        "retained_count": len(final_indices),
    }

    if not within_budget:
        logger.warning(
            "⚠️ Embedding budget violated: retained %d embeddings (allowed %d-%d)",
            len(final_indices),
            min_embeddings,
            max_embeddings,
        )

    return filtered_embeddings, metadata
