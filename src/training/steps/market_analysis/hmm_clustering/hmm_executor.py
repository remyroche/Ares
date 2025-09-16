"""HMM execution utilities extracted from Step 3.5.

Provides reusable functions for HMM training and validation.
Enhanced with common utilities integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from src.utils.sklearn_utils import StandardScaler
from src.utils.logger import system_logger

# Import common utilities
from src.utils.common_operations import (
    get_m1_gpu_manager,
    get_m1_memory_optimizer,
    get_m1_cpu_optimizer,
    validate_dataframe_columns,
    calculate_data_quality_metrics
)
from src.utils.common_utilities import safe_convert_dtypes
from src.utils.math_validation import safe_divide, safe_log
from src.utils.serialization_utils import JSONSerializer, PickleSerializer
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations


@dataclass
class HMMDependencies:
	logger: Any
	m1_gpu_manager: Any | None
	m1_memory_optimizer: Any | None
	m1_cpu_optimizer: Any | None
	matrix_ops: Any | None
	json_serializer: Any | None
	pickle_serializer: Any | None


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
	
	logger = getattr(deps, "logger", system_logger.getChild("HMMExecutor"))
	
	try:
		# Validate dependencies
		if deps.m1_memory_optimizer is None or deps.m1_gpu_manager is None:
			raise ValueError("GPU optimization dependencies not available")
		
		# Validate input data using common utilities
		if hasattr(features, 'columns'):
			# DataFrame input
			if not validate_dataframe_columns(features, features.columns.tolist()):
				logger.warning("DataFrame validation failed, proceeding with warnings")
			
			# Calculate data quality metrics
			quality_metrics = calculate_data_quality_metrics(features)
			logger.info(f"Data quality metrics: {quality_metrics}")
			
			# Convert dtypes for optimization
			features = safe_convert_dtypes(features, {
				col: 'float32' for col in features.select_dtypes(include=[np.number]).columns
			})
		
		# Convert to numpy and optimize memory usage
		features_array = deps.m1_memory_optimizer.create_memory_efficient_array(
			features.values if hasattr(features, 'values') else features, 
			dtype=np.float32
		)

		# Use matrix operations for efficient scaling if available
		if deps.matrix_ops and hasattr(deps.matrix_ops, 'optimized_scaling'):
			features_scaled, scaler = deps.matrix_ops.optimized_scaling(features_array)
		else:
			# Fallback to standard scaling
			scaler = StandardScaler()
			features_scaled = scaler.fit_transform(features_array)

		# Use GPU acceleration
		features_scaled_gpu = deps.m1_gpu_manager.to_device(features_scaled, "matrix_mult")

		# Train HMM with GPU context
		with deps.m1_gpu_manager.gpu_context("hmm_training"):
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

		# Calculate additional metrics using common utilities
		validation_metrics = {
			"converged": hmm_model.monitor_.converged if hasattr(hmm_model, 'monitor_') else False,
			"n_iterations": hmm_model.monitor_.iter if hasattr(hmm_model, 'monitor_') else n_iter,
			"log_likelihood": score,
			"aic": 2 * features_scaled.shape[1] * n_components - 2 * score,
			"bic": np.log(features_scaled.shape[0]) * features_scaled.shape[1] * n_components - 2 * score
		}

		return {
			"model": hmm_model,
			"scaler": scaler,
			"state_sequence": state_sequence,
			"state_probs": state_probs,
			"n_components": n_components,
			"score": score,
			"used_gpu": True,
			"validation_metrics": validation_metrics,
			"data_quality": quality_metrics if 'quality_metrics' in locals() else {}
		}
	except Exception as e:
		logger.exception(f"GPU HMM training failed: {e}")
		raise


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
	
	logger = getattr(deps, "logger", system_logger.getChild("HMMExecutor"))
	
	try:
		# Validate input features
		if features is None or (hasattr(features, 'empty') and features.empty):
			raise ValueError("Input features cannot be None or empty")
		
		# Validate input data using common utilities
		if hasattr(features, 'columns'):
			# DataFrame input
			if not validate_dataframe_columns(features, features.columns.tolist()):
				logger.warning("DataFrame validation failed, proceeding with warnings")
			
			# Calculate data quality metrics
			quality_metrics = calculate_data_quality_metrics(features)
			logger.info(f"Data quality metrics: {quality_metrics}")
			
			# Convert dtypes for optimization
			features = safe_convert_dtypes(features, {
				col: 'float32' for col in features.select_dtypes(include=[np.number]).columns
			})
			
			features_array = features.values
		else:
			features_array = np.array(features)
		
		# Use CPU optimization if available
		if deps.m1_cpu_optimizer:
			# Set optimal thread count
			optimal_threads = deps.m1_cpu_optimizer.get_optimal_thread_count()
			logger.info(f"Using {optimal_threads} CPU threads for HMM training")
		
		# Use matrix operations for efficient scaling if available
		if deps.matrix_ops and hasattr(deps.matrix_ops, 'optimized_scaling'):
			features_scaled, scaler = deps.matrix_ops.optimized_scaling(features_array)
		else:
			# Fallback to standard scaling
			scaler = StandardScaler()
			features_scaled = scaler.fit_transform(features_array)

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

		# Calculate additional metrics using common utilities
		validation_metrics = {
			"converged": hmm_model.monitor_.converged if hasattr(hmm_model, 'monitor_') else False,
			"n_iterations": hmm_model.monitor_.iter if hasattr(hmm_model, 'monitor_') else n_iter,
			"log_likelihood": score,
			"aic": 2 * features_scaled.shape[1] * n_components - 2 * score,
			"bic": np.log(features_scaled.shape[0]) * features_scaled.shape[1] * n_components - 2 * score
		}

		return {
			"model": hmm_model,
			"scaler": scaler,
			"state_sequence": state_sequence,
			"state_probs": state_probs,
			"n_components": n_components,
			"score": score,
			"used_gpu": False,
			"validation_metrics": validation_metrics,
			"data_quality": quality_metrics if 'quality_metrics' in locals() else {}
		}
	except Exception as e:
		logger.exception(f"CPU HMM training failed: {e}")
		raise


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
		# Validate inputs
		if features is None:
			raise ValueError("Features cannot be None")
		if n_components < 1:
			raise ValueError("n_components must be >= 1")
		if n_iter < 1:
			raise ValueError("n_iter must be >= 1")
		if covariance_type not in ['full', 'tied', 'diag', 'spherical']:
			raise ValueError(f"Invalid covariance_type: {covariance_type}")
		
		# Choose backend based on data size and available hardware
		if hasattr(features, 'size') and features.size > 1_000_000 and deps.m1_gpu_manager is not None:
			logger.info("🎯 Using GPU for large dataset HMM training...")
			return train_hmm_gpu_optimized(features, n_components, covariance_type, n_iter, random_state, deps)
		else:
			logger.info("💻 Using CPU for HMM training...")
			return train_hmm_cpu_optimized(features, n_components, covariance_type, n_iter, random_state, deps)
	except Exception as e:
		logger.exception(f"HMM training failed: {e}")
		raise


def create_hmm_dependencies(logger: Any = None) -> HMMDependencies:
	"""Create HMM dependencies with all available common utilities."""
	if logger is None:
		logger = system_logger.getChild("HMMExecutor")
	
	# Initialize common utilities
	gpu_manager = get_m1_gpu_manager()
	memory_optimizer = get_m1_memory_optimizer()
	cpu_optimizer = get_m1_cpu_optimizer()
	matrix_ops = UnifiedMatrixOperations()
	json_serializer = JSONSerializer()
	pickle_serializer = PickleSerializer()
	
	logger.info("🔧 Initialized HMM dependencies with common utilities")
	logger.info(f"   GPU Manager: {'Available' if gpu_manager else 'Not Available'}")
	logger.info(f"   Memory Optimizer: {'Available' if memory_optimizer else 'Not Available'}")
	logger.info(f"   CPU Optimizer: {'Available' if cpu_optimizer else 'Not Available'}")
	logger.info(f"   Matrix Operations: {'Available' if matrix_ops else 'Not Available'}")
	
	return HMMDependencies(
		logger=logger,
		m1_gpu_manager=gpu_manager,
		m1_memory_optimizer=memory_optimizer,
		m1_cpu_optimizer=cpu_optimizer,
		matrix_ops=matrix_ops,
		json_serializer=json_serializer,
		pickle_serializer=pickle_serializer
	)


def save_hmm_results(results: Dict[str, Any], filepath: str, deps: HMMDependencies) -> bool:
	"""Save HMM results using common serialization utilities."""
	try:
		# Prepare results for serialization
		serializable_results = {}
		for key, value in results.items():
			if key in ['model', 'scaler']:
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
			deps.logger.info(f"✅ HMM results saved to {filepath}")
		return success
		
	except Exception as e:
		deps.logger.error(f"❌ Failed to save HMM results: {e}")
		return False


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

