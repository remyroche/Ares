"""HMM execution utilities extracted from Step 3.5.

Provides reusable functions for HMM training and validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from src.utils.sklearn_utils import StandardScaler
from src.utils.logger import system_logger


@dataclass
class HMMDependencies:
	logger: Any
	m1_gpu_manager: Any | None
	m1_memory_optimizer: Any | None


def train_hmm_gpu_optimized(
	features: Any,
	n_components: int,
	covariance_type: str,
	n_iter: int,
	random_state: int,
	deps: HMMDependencies,
) -> Dict[str, Any]:
	"""Train HMM using GPU optimization when available."""
	from hmmlearn import hmm  # local import for optional dependency

	# Convert to numpy and optimize memory usage
	features_array = deps.m1_memory_optimizer.create_memory_efficient_array(  # type: ignore[union-attr]
		features.values, dtype=np.float32
	)

	# Scale features
	scaler = StandardScaler()
	features_scaled = scaler.fit_transform(features_array)

	# Use GPU acceleration
	features_scaled_gpu = deps.m1_gpu_manager.to_device(features_scaled, "matrix_mult")  # type: ignore[union-attr]

	# Train HMM with GPU context
	with deps.m1_gpu_manager.gpu_context("hmm_training"):  # type: ignore[union-attr]
		hmm_model = hmm.GaussianHMM(
			n_components=n_components,
			covariance_type=covariance_type,
			n_iter=n_iter,
			random_state=random_state,
		)
		hmm_model.fit(features_scaled)

		features_scaled_cpu = features_scaled_gpu.cpu().numpy()
		state_sequence = hmm_model.predict(features_scaled_cpu)
		state_probs = hmm_model.predict_proba(features_scaled_cpu)
		score = hmm_model.score(features_scaled_cpu)

	return {
		"model": hmm_model,
		"scaler": scaler,
		"state_sequence": state_sequence,
		"state_probs": state_probs,
		"n_components": n_components,
		"score": score,
		"used_gpu": True,
	}


def train_hmm_cpu_optimized(
	features: Any,
	n_components: int,
	covariance_type: str,
	n_iter: int,
	random_state: int,
	deps: HMMDependencies,
) -> Dict[str, Any]:
	"""Train HMM on CPU with scaling."""
	from hmmlearn import hmm  # local import for optional dependency

	scaler = StandardScaler()
	features_scaled = scaler.fit_transform(features.values)

	hmm_model = hmm.GaussianHMM(
		n_components=n_components,
		covariance_type=covariance_type,
		n_iter=n_iter,
		random_state=random_state,
	)

	hmm_model.fit(features_scaled)
	state_sequence = hmm_model.predict(features_scaled)
	state_probs = hmm_model.predict_proba(features_scaled)
	score = hmm_model.score(features_scaled)

	return {
		"model": hmm_model,
		"scaler": scaler,
		"state_sequence": state_sequence,
		"state_probs": state_probs,
		"n_components": n_components,
		"score": score,
		"used_gpu": False,
	}


def train_hmm_optimized(
	features: Any,
	n_components: int,
	covariance_type: str,
	n_iter: int,
	random_state: int,
	deps: HMMDependencies,
) -> Dict[str, Any]:
	"""Choose optimal backend and train HMM."""
	logger = getattr(deps, "logger", system_logger.getChild("HMMExecutor"))
	try:
		if features.size > 1_000_000 and deps.m1_gpu_manager is not None:
			logger.info("🎯 Using GPU for large dataset HMM training...")
			return train_hmm_gpu_optimized(features, n_components, covariance_type, n_iter, random_state, deps)
		else:
			logger.info("💻 Using CPU for HMM training...")
			return train_hmm_cpu_optimized(features, n_components, covariance_type, n_iter, random_state, deps)
	except Exception:
		logger.exception("HMM training failed")
		raise


def validate_hmm_model(hmm_model: Any, features: np.ndarray, n_components: int, logger: Any) -> Dict[str, Any]:
	"""Validate HMM convergence and quality metrics."""
	validation_result: Dict[str, Any] = {
		"converged": False,
		"issues": [],
		"recommendations": [],
	}
	try:
		logger.info("🔍 Validating HMM model convergence and quality...")
		if hasattr(hmm_model, "monitor_") and getattr(hmm_model, "monitor_").converged:  # type: ignore[attr-defined]
			validation_result["converged"] = True
		else:
			validation_result["issues"].append("HMM did not converge")
			validation_result["recommendations"].append("Increase n_iter or adjust parameters")
		return validation_result
	except Exception:
		logger.exception("HMM validation failed")
		validation_result["issues"].append("Validation error")
		return validation_result

