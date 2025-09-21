from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def compute_cluster_size_bounds(total_samples: int, min_frac: float, max_frac: float) -> Tuple[int, int]:
	min_size = int(np.floor(total_samples * min_frac))
	max_size = int(np.ceil(total_samples * max_frac))
	min_size = max(1, min_size)
	max_size = max(min_size, max_size)
	return min_size, max_size


def normalize_dict_values(values: Dict[str, float]) -> Dict[str, float]:
	total = sum(values.values())
	if total <= 0:
		return {k: 0.0 for k in values}
	return {k: v / total for k, v in values.items()}


def extract_regime_summary_vectors(
	regime_characteristics: Dict[str, dict],
	feature_weights: Dict[str, float],
	dimension_scales: Dict[str, float],
) -> Tuple[np.ndarray, List[str]]:
	"""Build a feature vector per regime from regime_characteristics.

	We expect `regime_characteristics[regime_key]['features']` to contain the 6 standardized
	features used in discovery. We map them to 4 dimensions and apply weights/scales.
	"""
	# Map standardized features to 4 dimensions
	feature_to_dim = {
		"volume_ratio_192m": "volume",
		"volatility_20": "volatility",
		"volatility_12": "volatility",
		"momentum_20": "momentum",
		"momentum_12": "momentum",
		"trend_score": "trend",
	}

	regime_keys = list(regime_characteristics.keys())
	X = []
	for key in regime_keys:
		char = regime_characteristics[key] or {}
		feat = char.get("features", {})
		# Aggregate to 4D by averaging within each dimension
		dim_vals: Dict[str, List[float]] = {"volume": [], "volatility": [], "momentum": [], "trend": []}
		for f_name, val in feat.items():
			dim = feature_to_dim.get(f_name)
			if dim is None:
				continue
			try:
				val_f = float(val)
			except Exception:
				continue
			dim_vals[dim].append(val_f)
		# Compute per-dimension means
		dim_vec = []
		for dim in ["volume", "volatility", "momentum", "trend"]:
			vals = dim_vals[dim]
			mean_val = float(np.mean(vals)) if len(vals) > 0 else 0.0
			weighted = mean_val * float(feature_weights.get(dim, 1.0)) * float(dimension_scales.get(dim, 1.0))
			dim_vec.append(weighted)
		X.append(dim_vec)
	return np.asarray(X, dtype=float), regime_keys


def aggregate_assignments_to_regimes(assignments: List[int]) -> Dict[int, int]:
	counts: Dict[int, int] = {}
	for a in assignments:
		counts[a] = counts.get(a, 0) + 1
	return counts


def map_regime_key_to_int(regime_key: str) -> int:
	# regime keys come as f"regime_{id}"; fallback to hash if missing
	try:
		return int(regime_key.split("_")[-1])
	except Exception:
		return abs(hash(regime_key)) % (10 ** 9)

