"""Clustering execution utilities extracted from Step 3.5.
Enhanced with common utilities integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from src.utils.sklearn_utils import KMeans, MiniBatchKMeans
# Note: Removed silhouette_score, davies_bouldin_score as they are not relevant for HMMs
from src.utils.defaults import Step03_5Defaults
from src.utils.logger import system_logger

# Import common utilities
from src.utils.math_validation import safe_divide, safe_log
from src.utils.serialization_utils import JSONSerializer, PickleSerializer
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations


@dataclass
class ClusteringDependencies:
	logger: Any
	m1_cpu_optimizer: Any | None
	matrix_ops: Any | None
	json_serializer: Any | None
	pickle_serializer: Any | None


def kmeans_standard(features_array: np.ndarray, n_clusters: int, random_state: int, logger: Any, deps: ClusteringDependencies = None) -> Dict[str, Any]:
	"""Standard KMeans clustering with HMM-relevant metrics and common utilities integration."""
	logger = logger or system_logger.getChild("ClusteringExecutor")
	
	try:
		# Validate inputs
		if features_array is None or len(features_array) == 0:
			raise ValueError("Features array cannot be None or empty")
		if n_clusters < 1:
			raise ValueError("n_clusters must be >= 1")
		if len(features_array) < n_clusters:
			raise ValueError("Number of samples must be >= n_clusters")
		
		# Use matrix operations for optimization if available
		if deps and deps.matrix_ops and hasattr(deps.matrix_ops, 'optimize_for_clustering'):
			features_optimized = deps.matrix_ops.optimize_for_clustering(features_array)
		else:
			features_optimized = features_array
		
		# Use CPU optimization if available
		if deps and deps.m1_cpu_optimizer:
			optimal_threads = deps.m1_cpu_optimizer.get_optimal_thread_count()
			logger.info(f"Using {optimal_threads} CPU threads for KMeans clustering")
		
		clustering = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=Step03_5Defaults.kmeans_n_init)
		labels = clustering.fit_predict(features_optimized)
		
		# Calculate HMM-relevant regime balance using safe math operations
		unique_regimes, counts = np.unique(labels, return_counts=True)
		regime_percentages = safe_divide(counts, len(labels), 0.0)
		balance_score = 1.0 - (np.max(regime_percentages) - np.min(regime_percentages))
		# Safe entropy calculation with proper handling of zero probabilities
		log_probs = safe_log(regime_percentages + 1e-10, 0.0)
		regime_entropy = -np.sum(regime_percentages * log_probs)
		
		# Calculate additional quality metrics
		quality_metrics = {
			"regime_balance_score": balance_score, 
			"regime_entropy": regime_entropy,
			"n_samples": len(labels),
			"n_clusters_found": len(unique_regimes),
			"inertia": clustering.inertia_ if hasattr(clustering, 'inertia_') else 0.0
		}
		
		return {
			"model": clustering,
			"cluster_labels": labels,
			"n_clusters": n_clusters,
			"cluster_centers": clustering.cluster_centers_,
			"quality_metrics": quality_metrics,
			"used_optimization": deps and deps.matrix_ops is not None
		}
	except Exception as e:
		logger.exception(f"Standard KMeans clustering failed: {e}")
		raise


def kmeans_minibatch(features_array: np.ndarray, n_clusters: int, random_state: int, logger: Any, deps: ClusteringDependencies = None) -> Dict[str, Any]:
	"""MiniBatchKMeans clustering with HMM-relevant metrics and common utilities integration."""
	logger = logger or system_logger.getChild("ClusteringExecutor")
	
	try:
		# Validate inputs
		if features_array is None or len(features_array) == 0:
			raise ValueError("Features array cannot be None or empty")
		if n_clusters < 1:
			raise ValueError("n_clusters must be >= 1")
		if len(features_array) < n_clusters:
			raise ValueError("Number of samples must be >= n_clusters")
		
		# Use matrix operations for optimization if available
		if deps and deps.matrix_ops and hasattr(deps.matrix_ops, 'optimize_for_clustering'):
			features_optimized = deps.matrix_ops.optimize_for_clustering(features_array)
		else:
			features_optimized = features_array
		
		# Use CPU optimization if available
		if deps and deps.m1_cpu_optimizer:
			optimal_threads = deps.m1_cpu_optimizer.get_optimal_thread_count()
			logger.info(f"Using {optimal_threads} CPU threads for MiniBatch KMeans clustering")
		
		mb = MiniBatchKMeans(
			n_clusters=n_clusters,
			batch_size=min(100, max(1, len(features_optimized) // 10)),
			n_init=Step03_5Defaults.minibatch_n_init,
			random_state=random_state,
			max_iter=Step03_5Defaults.kmeans_max_iter,
		)
		labels = mb.fit_predict(features_optimized)
		
		# Calculate HMM-relevant regime balance using safe math operations
		unique_regimes, counts = np.unique(labels, return_counts=True)
		regime_percentages = safe_divide(counts, len(labels), 0.0)
		balance_score = 1.0 - (np.max(regime_percentages) - np.min(regime_percentages))
		# Safe entropy calculation with proper handling of zero probabilities
		log_probs = safe_log(regime_percentages + 1e-10, 0.0)
		regime_entropy = -np.sum(regime_percentages * log_probs)
		
		# Calculate additional quality metrics
		quality_metrics = {
			"regime_balance_score": balance_score, 
			"regime_entropy": regime_entropy,
			"n_samples": len(labels),
			"n_clusters_found": len(unique_regimes),
			"inertia": mb.inertia_ if hasattr(mb, 'inertia_') else 0.0
		}
		
		return {
			"model": mb,
			"cluster_labels": labels,
			"n_clusters": n_clusters,
			"cluster_centers": mb.cluster_centers_,
			"quality_metrics": quality_metrics,
			"used_optimization": deps and deps.matrix_ops is not None
		}
	except Exception as e:
		logger.exception(f"MiniBatch KMeans clustering failed: {e}")
		raise


def create_clustering_dependencies(logger: Any = None) -> ClusteringDependencies:
	"""Create clustering dependencies with all available common utilities."""
	if logger is None:
		logger = system_logger.getChild("ClusteringExecutor")
	
	# Initialize common utilities
	from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
	from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
	from src.utils.serialization_utils import JSONSerializer, PickleSerializer
	
	cpu_optimizer = get_m1_cpu_optimizer()
	matrix_ops = UnifiedMatrixOperations()
	json_serializer = JSONSerializer()
	pickle_serializer = PickleSerializer()
	
	logger.info("🔧 Initialized clustering dependencies with common utilities")
	logger.info(f"   CPU Optimizer: {'Available' if cpu_optimizer else 'Not Available'}")
	logger.info(f"   Matrix Operations: {'Available' if matrix_ops else 'Not Available'}")
	
	return ClusteringDependencies(
		logger=logger,
		m1_cpu_optimizer=cpu_optimizer,
		matrix_ops=matrix_ops,
		json_serializer=json_serializer,
		pickle_serializer=pickle_serializer
	)


def save_clustering_results(results: Dict[str, Any], filepath: str, deps: ClusteringDependencies) -> bool:
	"""Save clustering results using common serialization utilities."""
	try:
		# Prepare results for serialization
		serializable_results = {}
		for key, value in results.items():
			if key in ['model']:
				# Skip non-serializable objects
				continue
			elif isinstance(value, np.ndarray):
				serializable_results[key] = value.tolist()
			else:
				serializable_results[key] = value
		
		# Save using appropriate serializer
		if filepath.endswith('.json'):
			success = deps.json_serializer.save(serializable_results, filepath)
		else:
			success = deps.pickle_serializer.save(serializable_results, filepath)
		
		if success:
			deps.logger.info(f"✅ Clustering results saved to {filepath}")
		return success
		
	except Exception as e:
		deps.logger.error(f"❌ Failed to save clustering results: {e}")
		return False

