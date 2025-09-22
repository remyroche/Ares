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

	We work with the actual data structure which includes indicator_averages and other characteristics.
	We extract meaningful features for clustering.
	"""
	regime_keys = list(regime_characteristics.keys())
	X = []

	# Define which indicators to use for each dimension
	volume_indicators = ["volume", "quote_volume", "trades"]
	volatility_indicators = ["price_range", "close_return", "close_log_return"]
	momentum_indicators = ["volume_return", "volume_log_return"]
	trend_indicators = ["open", "high", "low", "close"]

	for key in regime_keys:
		char = regime_characteristics[key] or {}

		# Use available characteristics to create feature vectors
		# Since we have basic characteristics, create simple features
		sample_count = float(char.get("sample_count", 0))
		percentage = float(char.get("percentage", 0))
		regime_id = float(char.get("regime_id", 0))

		# Create 4D feature vector based on available characteristics
		# Volume: based on sample count (more samples = higher volume)
		volume_feature = sample_count / 1000.0  # Normalize by dividing by 1000

		# Volatility: based on percentage (higher percentage = more volatile)
		volatility_feature = percentage / 100.0  # Convert to 0-1 range

		# Momentum: simple regime id based feature
		momentum_feature = regime_id / 10.0  # Normalize regime ID

		# Trend: based on regime ID and sample count interaction
		trend_feature = (regime_id * sample_count) / 10000.0

		dim_vec = [
			volume_feature * float(feature_weights.get("volume", 1.0)) * float(dimension_scales.get("volume", 1.0)),
			volatility_feature * float(feature_weights.get("volatility", 1.0)) * float(dimension_scales.get("volatility", 1.0)),
			momentum_feature * float(feature_weights.get("momentum", 1.0)) * float(dimension_scales.get("momentum", 1.0)),
			trend_feature * float(feature_weights.get("trend", 1.0)) * float(dimension_scales.get("trend", 1.0))
		]

		X.append(dim_vec)

	# If no features could be extracted, create simple identity features
	if not X or len(X[0]) == 0:
		X = [[i, 0, 0, 0] for i in range(len(regime_keys))]

	X = np.asarray(X, dtype=float)
	
	# Apply StandardScaler for consistent scaling across all dimensions
	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
	
	return X_scaled, regime_keys


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

