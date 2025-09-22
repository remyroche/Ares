from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CoverageClusteringConfig:
	# Targets
	target_num_clusters: int = 20
	min_num_clusters: int = 15
	max_num_clusters: int = 26
	target_coverage: float = 0.92  # 92% mid of 90–95%
	min_coverage: float = 0.90
	max_coverage: float = 0.95

	# Size constraints as fraction of total
	min_cluster_fraction: float = 0.03
	max_cluster_fraction: float = 0.08
	max_noise_fraction: float = 0.05

	# Clustering knobs
	max_init_trials: int = 8
	kmeans_init: str = "k-means++"
	max_iter: int = 300
	random_state: int = 42

	# Feature weights for 4D regime summary vectors
	feature_weights: Dict[str, float] = field(default_factory=lambda: {
		"volume": 1.0,
		"volatility": 1.0,
		"momentum": 1.0,
		"trend": 1.0,
	})

	# Distance scaling per dimension (to emphasize separation)
	dimension_scales: Dict[str, float] = field(default_factory=lambda: {
		"volume": 1.0,
		"volatility": 1.0,
		"momentum": 1.0,
		"trend": 1.0,
	})

	# Outlier trimming hyperparameters
	per_cluster_trim_quantile: float = 0.95  # keep closest 95% to centroid
	global_trim_quantile: float = 0.995      # drop top 0.5% farthest overall if needed

	# Evaluation
	min_silhouette: float = 0.15
	min_calinski: float = 50.0
	max_davies_bouldin: float = 2.0

	# Frontier refinement & size guidance
	# Target average cluster fraction (lean toward ~5%)
	target_cluster_fraction: float = 0.05
	# Max allowed size ratio for destination vs source cluster (B <= 1.5 * A)
	max_size_ratio_move: float = 1.5
	# Consider points within this distance ratio to the frontier (nearest/second-nearest <= threshold)
	frontier_ratio_threshold: float = 1.1
	# Number of refinement iterations over frontier candidates
	move_iterations: int = 5
	# Scoring weights for reassignment decisions
	cv_weight: float = 1.0
	silhouette_weight: float = 0.5
	size_balance_weight: float = 0.3
	similarity_weight: float = 0.2
	# Enforce size bounds [min_frac, max_frac] during refinement
	enforce_size_bounds_during_refinement: bool = True
	# Utilize matrix_operations/ vectorized utilities where available
	use_matrix_ops: bool = True

	# Input keys (from HMM discovery artifacts)
	hmm_artifact_key: str = "hmm_regime_discovery_result"
	regime_assignments_key: str = "regime_assignments"
	regime_characteristics_key: str = "regime_characteristics"

	# Operational
	verbose: bool = True
	max_samples_for_metrics: int = 100_000

