"""Clustering execution utilities extracted from Step 3.5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from src.utils.sklearn_utils import KMeans, MiniBatchKMeans
# Note: Removed silhouette_score, davies_bouldin_score as they are not relevant for HMMs
from src.utils.defaults import Step03_5Defaults
from src.utils.logger import system_logger


@dataclass
class ClusteringDependencies:
	logger: Any
	m1_cpu_optimizer: Any | None


def kmeans_standard(features_array: np.ndarray, n_clusters: int, random_state: int, logger: Any) -> Dict[str, Any]:
	"""Standard KMeans clustering with HMM-relevant metrics."""
	clustering = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=Step03_5Defaults.kmeans_n_init)
	labels = clustering.fit_predict(features_array)
	try:
		# Calculate HMM-relevant regime balance instead of clustering metrics
		unique_regimes, counts = np.unique(labels, return_counts=True)
		regime_percentages = counts / len(labels)
		balance_score = 1.0 - (np.max(regime_percentages) - np.min(regime_percentages))
		regime_entropy = -np.sum(regime_percentages * np.log(regime_percentages + 1e-10))
	except Exception:
		logger = logger or system_logger.getChild("ClusteringExecutor")
		logger.warning("Could not calculate regime balance metrics")
		balance_score, regime_entropy = 0.0, 0.0
	return {
		"model": clustering,
		"cluster_labels": labels,
		"n_clusters": n_clusters,
		"cluster_centers": clustering.cluster_centers_,
		"quality_metrics": {"regime_balance_score": balance_score, "regime_entropy": regime_entropy},
	}


def kmeans_minibatch(features_array: np.ndarray, n_clusters: int, random_state: int, logger: Any) -> Dict[str, Any]:
	"""MiniBatchKMeans clustering with HMM-relevant metrics."""
	mb = MiniBatchKMeans(
		n_clusters=n_clusters,
		batch_size=min(100, max(1, len(features_array) // 10)),
		n_init=Step03_5Defaults.minibatch_n_init,
		random_state=random_state,
		max_iter=Step03_5Defaults.kmeans_max_iter,
	)
	labels = mb.fit_predict(features_array)
	try:
		# Calculate HMM-relevant regime balance instead of clustering metrics
		unique_regimes, counts = np.unique(labels, return_counts=True)
		regime_percentages = counts / len(labels)
		balance_score = 1.0 - (np.max(regime_percentages) - np.min(regime_percentages))
		regime_entropy = -np.sum(regime_percentages * np.log(regime_percentages + 1e-10))
	except Exception:
		logger = logger or system_logger.getChild("ClusteringExecutor")
		logger.warning("Could not calculate regime balance metrics")
		balance_score, regime_entropy = 0.0, 0.0
	return {
		"model": mb,
		"cluster_labels": labels,
		"n_clusters": n_clusters,
		"cluster_centers": mb.cluster_centers_,
		"quality_metrics": {"regime_balance_score": balance_score, "regime_entropy": regime_entropy},
	}

