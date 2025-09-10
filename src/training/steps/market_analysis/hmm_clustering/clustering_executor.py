"""Clustering execution utilities extracted from Step 3.5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from src.utils.sklearn_utils import KMeans, MiniBatchKMeans, silhouette_score, davies_bouldin_score
from src.utils.defaults import Step03_5Defaults
from src.utils.logger import system_logger


@dataclass
class ClusteringDependencies:
	logger: Any
	m1_cpu_optimizer: Any | None


def kmeans_standard(features_array: np.ndarray, n_clusters: int, random_state: int, logger: Any) -> Dict[str, Any]:
	"""Standard KMeans clustering with metrics."""
	clustering = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=Step03_5Defaults.kmeans_n_init)
	labels = clustering.fit_predict(features_array)
	try:
		sil = silhouette_score(features_array, labels)
		db = davies_bouldin_score(features_array, labels)
	except Exception:
		logger = logger or system_logger.getChild("ClusteringExecutor")
		logger.warning("Could not calculate clustering metrics")
		sil, db = 0.0, 1.0
	return {
		"model": clustering,
		"cluster_labels": labels,
		"n_clusters": n_clusters,
		"cluster_centers": clustering.cluster_centers_,
		"quality_metrics": {"silhouette_score": sil, "davies_bouldin_score": db},
	}


def kmeans_minibatch(features_array: np.ndarray, n_clusters: int, random_state: int, logger: Any) -> Dict[str, Any]:
	"""MiniBatchKMeans clustering with metrics."""
	mb = MiniBatchKMeans(
		n_clusters=n_clusters,
		batch_size=min(100, max(1, len(features_array) // 10)),
		n_init=Step03_5Defaults.minibatch_n_init,
		random_state=random_state,
		max_iter=Step03_5Defaults.kmeans_max_iter,
	)
	labels = mb.fit_predict(features_array)
	try:
		sil = silhouette_score(features_array, labels)
		db = davies_bouldin_score(features_array, labels)
	except Exception:
		logger = logger or system_logger.getChild("ClusteringExecutor")
		logger.warning("Could not calculate clustering metrics")
		sil, db = 0.0, 1.0
	return {
		"model": mb,
		"cluster_labels": labels,
		"n_clusters": n_clusters,
		"cluster_centers": mb.cluster_centers_,
		"quality_metrics": {"silhouette_score": sil, "davies_bouldin_score": db},
	}

