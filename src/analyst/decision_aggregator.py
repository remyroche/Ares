# src/analyst/decision_aggregator.py

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

from src.utils.logger import system_logger
from src.analyst.regime_runtime import get_current_regime_info


def _safe_get(d: dict, k: Any, default: float = 0.0) -> float:
	try:
		v = d.get(k, default)
		return float(v)
	except Exception:
		return float(default)


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
	vals = np.array([max(0.0, float(v)) for v in weights.values()], dtype=float)
	s = float(vals.sum())
	if s <= 0:
		return {k: 0.0 for k in weights}
	return {k: float(v) / s for k, v in zip(weights.keys(), vals)}


def aggregate_weights(
	exchange: str,
	symbol: str,
	timeframe: str,
	specialized_candidates: Dict[int, Dict[str, float]] | None,
	generalist_score: Optional[float] = None,
	config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	"""
	Compute ensemble weights for specialized models and an optional generalist baseline,
	using HMM composite intensities and calibrated probabilities as gates.

	Args:
		-specialized_candidates: mapping cluster_id -> {
			"confidence": calibrated prob of the specialized model's own prediction,
			"reliability": long-run reliability metric in [0,1] (optional),
			(optional) additional fields not used by weight calc.
		}
		-generalist_score: optional baseline confidence (0-1) for a generalist predictor.
		-config: optional dict with keys:
			- alpha_intensity (default 0.7), beta_emerge (0.3): gates for cluster weight
			- min_intensity (default 0.15): below this, model is considered weak
			- max_specialized (default 3): top-k specialized models to include by intensity

	Returns:
		Dict with keys:
			- weights: normalized weights for keys like "cluster_{k}" and optional "generalist"
			- gating: dict with gating factors per cluster and the exit_hazard
			- runtime: snapshot from get_current_regime_info
	"""
	logger = system_logger.getChild("DecisionAggregator")
	cfg = config or {}
	alpha_intensity = float(cfg.get("alpha_intensity", 0.7))
	beta_emerge = float(cfg.get("beta_emerge", 0.3))
	min_intensity = float(cfg.get("min_intensity", 0.15))
	max_specialized = int(cfg.get("max_specialized", 3))

	runtime = get_current_regime_info(exchange, symbol, timeframe, data_dir=cfg.get("data_dir", "data/training"), checkpoints_dir=cfg.get("checkpoints_dir", "checkpoints"))
	intensities: Dict[int, float] = runtime.get("intensities", {}) or {}
	p_emerge: Dict[int, float] = runtime.get("p_emerge", {}) or {}
	exit_hazard: Optional[float] = runtime.get("exit_hazard")
	current_cluster = int(runtime.get("cluster_id", -1) or -1)

	# Choose top-k clusters by intensity
	sorted_k = sorted(intensities.items(), key=lambda kv: kv[1], reverse=True)
	top_k = [k for k, v in sorted_k if v >= min_intensity][:max_specialized]

	weights: Dict[str, float] = {}
	gating: Dict[str, float] = {}
	if specialized_candidates:
		for k in top_k:
			cand = specialized_candidates.get(k, {})
			conf = float(cand.get("confidence", 0.0))
			rel = float(cand.get("reliability", 1.0))
			I = _safe_get(intensities, k, 0.0)
			Pe = _safe_get(p_emerge, k, 0.0)
			gate = max(0.0, min(1.0, alpha_intensity * I + beta_emerge * Pe))
			# If current cluster, down-weight by exit hazard risk
			if k == current_cluster and exit_hazard is not None:
				gate *= max(0.0, 1.0 - float(exit_hazard))
			score = max(0.0, float(conf)) * max(0.0, float(rel)) * gate
			weights[f"cluster_{k}"] = score
			gating[f"cluster_{k}"] = gate

	# Optional generalist
	if generalist_score is not None:
		# Generalist can be used as a safety net; scale it by (1 - max exit hazard)
		g = 1.0
		if exit_hazard is not None:
			g = max(0.0, 1.0 - float(exit_hazard))
		weights["generalist"] = max(0.0, float(generalist_score)) * g
		gating["generalist_gate"] = g

	norm_weights = _normalize(weights)
	# Monitoring log
	try:
		logger.info({
			"msg": "model_weights",
			"timeframe": timeframe,
			"weights": norm_weights,
			"gating": gating,
			"current_cluster": current_cluster,
		})
	except Exception:
		pass

	return {
		"weights": norm_weights,
		"gating": gating,
		"runtime": runtime,
	}