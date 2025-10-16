from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from .config import CoverageClusteringConfig
from .utils import (
	compute_cluster_size_bounds,
	extract_regime_summary_vectors,
	aggregate_assignments_to_regimes,
	map_regime_key_to_int,
)


@dataclass
class ClusteringOutputs:
	cluster_labels: Dict[str, int]
	selected_regime_keys: List[str]
	metrics: Dict[str, float]
	coverage_pct: float
	noise_regime_keys: List[str]
	cluster_sizes: Dict[int, int]
	cluster_size_pct: Dict[int, float]


class CoverageConstrainedClusterer:
	def __init__(self, config: CoverageClusteringConfig) -> None:
		self.config = config

	def _evaluate(self, X_all: np.ndarray, X_used: np.ndarray, labels: np.ndarray, sizes: Dict[int, int]) -> Dict[str, float]:
		metrics: Dict[str, float] = {}
		try:
			metrics["silhouette"] = silhouette_score(X_used, labels) if len(np.unique(labels)) > 1 else -1.0
		except Exception:
			metrics["silhouette"] = -1.0
		try:
			metrics["calinski_harabasz"] = calinski_harabasz_score(X_used, labels) if len(np.unique(labels)) > 1 else 0.0
		except Exception:
			metrics["calinski_harabasz"] = 0.0
		try:
			metrics["davies_bouldin"] = davies_bouldin_score(X_used, labels) if len(np.unique(labels)) > 1 else np.inf
		except Exception:
			metrics["davies_bouldin"] = np.inf
		# Balance metric: penalize oversized clusters
		size_arr = np.array(list(sizes.values()), dtype=float)
		if size_arr.size > 0:
			metrics["size_cv"] = float(np.std(size_arr) / (np.mean(size_arr) + 1e-9))
		else:
			metrics["size_cv"] = 0.0
		return metrics

	def _trim_outliers_per_cluster(self, X: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
		# Remove farthest points per cluster beyond quantile
		keep_mask = np.ones(len(X), dtype=bool)
		centroids = []
		for k in np.unique(labels):
			cluster_idx = np.where(labels == k)[0]
			if cluster_idx.size == 0:
				continue
			centroid = X[cluster_idx].mean(axis=0)
			centroids.append(centroid)
			dists = np.linalg.norm(X[cluster_idx] - centroid, axis=1)
			q = np.quantile(dists, self.config.per_cluster_trim_quantile)
			keep_local = dists <= q
			keep_mask[cluster_idx] = keep_local
		# Optional global trim for extreme tails
		if keep_mask.sum() > 0 and keep_mask.sum() < len(X):
			residual = X[keep_mask]
			d_all = np.linalg.norm(residual - residual.mean(axis=0), axis=1)
			qg = np.quantile(d_all, self.config.global_trim_quantile)
			keep_mask_indices = np.where(keep_mask)[0]
			keep_mask2 = np.ones_like(keep_mask)
			keep_mask2[:] = True
			keep_mask2[keep_mask_indices[d_all > qg]] = False
			keep_mask = keep_mask & keep_mask2
		return X[keep_mask], labels[keep_mask], np.where(~keep_mask)[0].tolist()

	def _enforce_size_bounds(self, labels: np.ndarray, total: int) -> np.ndarray:
		min_size, max_size = compute_cluster_size_bounds(total, self.config.min_cluster_fraction, self.config.max_cluster_fraction)
		# ZERO NOISE: If a cluster exceeds max_size, split it into multiple clusters instead of marking as noise
		new_labels = labels.copy()
		next_cluster_id = max(np.unique(labels)) + 1
		
		for k in np.unique(labels):
			idx = np.where(new_labels == k)[0]
			if idx.size > max_size:
				# Split oversized cluster into multiple clusters
				excess_count = idx.size - max_size
				# Create new cluster for excess points
				new_labels[idx[:excess_count]] = next_cluster_id
				next_cluster_id += 1
		return new_labels

	def cluster(self, hmm_artifact: Dict[str, Any]) -> ClusteringOutputs:
		cfg = self.config
		assignments: List[int] = hmm_artifact.get(cfg.regime_assignments_key, [])
		regime_chars: Dict[str, dict] = hmm_artifact.get(cfg.regime_characteristics_key, {})
		if not assignments or not regime_chars:
			raise ValueError("Missing regime assignments or regime characteristics in HMM artifact")

		# Build per-regime 4D vectors and keys
		X_regimes, regime_keys = extract_regime_summary_vectors(
			regime_chars, cfg.feature_weights, cfg.dimension_scales
		)
		if X_regimes.size == 0:
			raise ValueError("Empty regime summary vectors")

		# Use ALL regimes initially - don't filter based on coverage constraints
		regime_counts = aggregate_assignments_to_regimes(assignments)
		# map keys string -> regime int id
		key_to_id = {k: map_regime_key_to_int(k) for k in regime_keys}
		counts_per_key = {k: regime_counts.get(key_to_id[k], 0) for k in regime_keys}
		sorted_keys = sorted(regime_keys, key=lambda k: counts_per_key[k], reverse=True)

		# Use ALL regimes for clustering - no coverage filtering
		selected_keys = sorted_keys  # Use all keys, don't filter any

		# Build training matrix of selected regimes
		key_index = {k: i for i, k in enumerate(regime_keys)}
		sel_idx = np.array([key_index[k] for k in selected_keys], dtype=int)
		X_sel = X_regimes[sel_idx]

		# Calculate total for coverage calculations
		total = len(assignments)

		# Choose k close to target but not exceeding selected count
		k_target = min(max(cfg.min_num_clusters, cfg.target_num_clusters), len(selected_keys))

		best: Tuple[float, Tuple[np.ndarray, np.ndarray]] | None = None
		best_labels = None
		for trial in range(cfg.max_init_trials):
			kmeans = KMeans(n_clusters=k_target, init=cfg.kmeans_init, n_init=10, max_iter=cfg.max_iter, random_state=cfg.random_state + trial)
			labels_sel = kmeans.fit_predict(X_sel)
			# Distance-aware trim first
			X_trim, labels_trim, _ = self._trim_outliers_per_cluster(X_sel, labels_sel)
			# Enforce size bounds (on regime-count level)
			labels_bounded = self._enforce_size_bounds(labels_trim, total)
			# Keep only non-noise
			valid = labels_bounded >= 0
			if valid.sum() < 2:
				continue
			# Evaluate on the used subset
			X_used_trial = X_trim[valid]
			labels_used_trial = labels_bounded[valid]
			sizes = {int(k): int((labels_used_trial == k).sum()) for k in np.unique(labels_used_trial) if k >= 0}
			metrics = self._evaluate(X_sel, X_used_trial, labels_used_trial, sizes)
			score = float(metrics.get("silhouette", -1.0)) - 0.1 * float(metrics.get("size_cv", 0.0))
			if best is None or score > best[0]:
				best = (score, (labels_bounded, X_trim))
				best_labels = labels_bounded

		if best_labels is None:
			# Fallback: plain KMeans without trimming
			kmeans = KMeans(n_clusters=k_target, init=cfg.kmeans_init, n_init=10, max_iter=cfg.max_iter, random_state=cfg.random_state)
			best_labels = kmeans.fit_predict(X_sel)

		# Finalize cluster labels for selected regimes only
		labels_final = best_labels
		valid_mask = labels_final >= 0
		selected_used = [sk for sk, ok in zip(selected_keys, valid_mask.tolist()) if ok]
		labels_used = labels_final[valid_mask]

		# Map regime key -> cluster id (initial)
		cluster_labels: Dict[str, int] = {}
		for sk, lab in zip(selected_used, labels_used.tolist()):
			cluster_labels[sk] = int(lab)

		# Compute coverage actually used by chosen regimes
		used_counts = sum([counts_per_key.get(sk, 0) for sk in selected_used])
		coverage_pct = 100.0 * (used_counts / total) if total > 0 else 0.0

		# No noise regimes - we want to cluster ALL regimes
		noise_keys = []  # Don't mark any regimes as noise

		# Enforce cluster size constraints (3-8% of total) by redistributing regimes
		# Instead of marking regimes as noise, we'll redistribute them to balance cluster sizes
		min_count = int(np.floor(cfg.min_cluster_fraction * total))
		max_count = int(np.ceil(cfg.max_cluster_fraction * total))

		# Build regimes per cluster
		regimes_by_cluster: Dict[int, List[str]] = {}
		for sk, lab in cluster_labels.items():
			regimes_by_cluster.setdefault(lab, []).append(sk)

		# Redistribute regimes to enforce size constraints
		# Get all regime keys sorted by size
		regime_sizes = [(sk, counts_per_key.get(sk, 0)) for sk in cluster_labels.keys()]
		regime_sizes.sort(key=lambda x: x[1], reverse=True)  # Sort by size descending

		# Calculate how many clusters we need to fit all regimes within 3-8% constraint
		min_clusters_needed = int(np.ceil(total / max_count))  # Maximum clusters needed for 8% max

		# We can split regimes into multiple clusters to meet size constraints
		# Calculate how many clusters we need in total
		target_cluster_count = min_clusters_needed

		print(f"DEBUG: Total samples: {total}, Max count (8%): {max_count}, Min count (3%): {min_count}")
		print(f"DEBUG: Min clusters needed: {min_clusters_needed}, Target: {target_cluster_count}, Current: {len(regimes_by_cluster)}")
		print(f"DEBUG: Initial cluster sizes: {[(cid, sum(counts_per_key.get(sk, 0) for sk in regimes_by_cluster[cid])) for cid in regimes_by_cluster]}")

		# Create additional clusters by duplicating large regimes to meet size constraints
		new_cluster_id = max(regimes_by_cluster.keys()) + 1

		# Calculate how many clusters we need in total and how many to create
		clusters_to_create = target_cluster_count - len(regimes_by_cluster)

		print(f"DEBUG: Creating {clusters_to_create} new clusters by duplicating large regimes")

		# Get all regimes sorted by size (largest first)
		all_regimes = []
		for cluster_id, regimes in regimes_by_cluster.items():
			for regime_key in regimes:
				all_regimes.append((regime_key, counts_per_key.get(regime_key, 0)))

		# Sort by size descending
		all_regimes.sort(key=lambda x: x[1], reverse=True)

		# Create new clusters by duplicating the largest regimes
		# First, identify which regimes we want to keep in the original clusters vs new clusters
		for i in range(clusters_to_create):
			# Take the largest available regime
			if not all_regimes:
				break

			largest_regime, regime_size = all_regimes[0]

			# Find which original cluster contains this regime
			source_cluster = None
			for cluster_id, regimes in regimes_by_cluster.items():
				if largest_regime in regimes:
					source_cluster = cluster_id
					break

			if source_cluster is not None:
				# Remove the regime from its original cluster
				regimes_by_cluster[source_cluster].remove(largest_regime)

			# Create a new cluster with this regime
			regimes_by_cluster[new_cluster_id] = [largest_regime]
			cluster_labels[largest_regime] = new_cluster_id

			print(f"DEBUG: Created cluster {new_cluster_id} with regime {largest_regime} (size: {regime_size}) from cluster {source_cluster}")

			new_cluster_id += 1

			# Remove this regime from the available list since we've used it
			all_regimes.pop(0)

		# Now perform final balancing: move regimes from oversized to undersized clusters
		iterations = 0
		max_iterations = 50

		while iterations < max_iterations:
			iterations += 1

			# Calculate current cluster sizes
			cluster_sizes = {}
			for cid, regimes in regimes_by_cluster.items():
				cluster_sizes[cid] = sum(counts_per_key.get(sk, 0) for sk in regimes)

			# Find problematic clusters
			oversized = [cid for cid, size in cluster_sizes.items() if size > max_count]
			undersized = [cid for cid, size in cluster_sizes.items() if size < min_count]

			# If no problematic clusters, we're done
			if not oversized and not undersized:
				break

			# Move from biggest oversized to smallest undersized
			if oversized and undersized:
				biggest_over = max(oversized, key=lambda cid: cluster_sizes[cid])
				smallest_under = min(undersized, key=lambda cid: cluster_sizes[cid])

				# Move largest regime from oversized to undersized
				largest_regime = max(regimes_by_cluster[biggest_over], key=lambda sk: counts_per_key.get(sk, 0))
				regimes_by_cluster[biggest_over].remove(largest_regime)
				regimes_by_cluster[smallest_under].append(largest_regime)
				cluster_labels[largest_regime] = smallest_under

			# If only oversized, create a new cluster
			elif oversized:
				new_cluster_id = max(regimes_by_cluster.keys()) + 1
				biggest_over = max(oversized, key=lambda cid: cluster_sizes[cid])

				# Move largest regime to new cluster
				largest_regime = max(regimes_by_cluster[biggest_over], key=lambda sk: counts_per_key.get(sk, 0))
				regimes_by_cluster[biggest_over].remove(largest_regime)
				regimes_by_cluster[new_cluster_id] = [largest_regime]
				cluster_labels[largest_regime] = new_cluster_id

		# Recompute sizes after enforcement
		cluster_sizes: Dict[int, int] = {}
		for sk, lab in cluster_labels.items():
			cluster_sizes[lab] = cluster_sizes.get(lab, 0) + int(counts_per_key.get(sk, 0))

		print(f"DEBUG: Final cluster sizes: {cluster_sizes}")
		print(f"DEBUG: Final cluster_labels: {cluster_labels}")
		cluster_size_pct = {lab: (sz / total) * 100.0 for lab, sz in cluster_sizes.items() if total > 0}
		# Update selected_used and coverage (used_counts)
		selected_used = list(cluster_labels.keys())
		used_counts = sum([counts_per_key.get(sk, 0) for sk in selected_used])
		coverage_pct = 100.0 * (used_counts / total) if total > 0 else 0.0

		# Prepare evaluation metrics using all regime vectors but only for used labels
		X_used = X_sel[valid_mask]
		metrics = self._evaluate(X_sel, X_used, labels_used, cluster_sizes)
		metrics.update({
			"coverage_pct": coverage_pct,
			"clusters": len(np.unique(labels_used)),
		})

		return ClusteringOutputs(
			cluster_labels=cluster_labels,
			selected_regime_keys=selected_used,
			metrics=metrics,
			coverage_pct=coverage_pct,
			noise_regime_keys=noise_keys,
			cluster_sizes=cluster_sizes,
			cluster_size_pct=cluster_size_pct,
		)

