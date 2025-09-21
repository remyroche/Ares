from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
from pathlib import Path
import json


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


def _load_json(path: Path) -> Optional[dict]:
	try:
		with open(path, "r") as f:
			return json.load(f)
	except Exception:
		return None


def find_latest_hmm_discovery_artifact_path(
	base_dir: str = "artifacts",
	symbol: Optional[str] = None,
	exchange: Optional[str] = None,
	timeframe: Optional[str] = None,
) -> Optional[Path]:
	"""Locate the latest saved HMM discovery artifact JSON via metadata files.

	Matches by symbol/exchange/timeframe when provided; otherwise picks the latest overall.
	"""
	root = Path(base_dir)
	if not root.exists():
		return None
	candidates: List[Tuple[float, Path]] = []
	for session_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
		meta_files = list(session_dir.glob("hmmregimediscovery_metadata_*.json"))
		if not meta_files:
			continue
		# Use the latest metadata file in this session
		meta_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
		meta = _load_json(meta_files[0]) or {}
		if str(meta.get("component_name", "")).lower() != "hmmregimediscovery":
			continue
		if symbol and str(meta.get("symbol", "")).upper() != symbol.upper():
			continue
		if exchange and str(meta.get("exchange", "")).lower() != exchange.lower():
			continue
		if timeframe and str(meta.get("timeframe", "")) != timeframe:
			continue
		artifact_files = list(session_dir.glob("hmmregimediscovery_hmm_regime_discovery_result_*.json"))
		if not artifact_files:
			continue
		artifact_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
		latest = artifact_files[0]
		candidates.append((latest.stat().st_mtime, latest))
	if not candidates:
		return None
	candidates.sort(key=lambda x: x[0], reverse=True)
	return candidates[0][1]


def load_latest_hmm_discovery_artifact(
	base_dir: str = "artifacts",
	symbol: Optional[str] = None,
	exchange: Optional[str] = None,
	timeframe: Optional[str] = None,
) -> Optional[dict]:
	path = find_latest_hmm_discovery_artifact_path(base_dir, symbol, exchange, timeframe)
	if path is None:
		return None
	data = _load_json(path)
	if not isinstance(data, dict):
		return None
	# Some callers may save a wrapper dict; prefer direct artifact when present
	if "hmm_regime_discovery_result" in data and isinstance(data["hmm_regime_discovery_result"], dict):
		return data["hmm_regime_discovery_result"]
	return data

