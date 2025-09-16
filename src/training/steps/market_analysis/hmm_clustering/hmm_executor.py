"""HMM execution utilities extracted from Step 3.5.

Provides reusable functions for HMM training and validation.
Enhanced with common utilities integration for optimal performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple
import time
import logging

import numpy as np
import pandas as pd
from src.utils.sklearn_utils import StandardScaler
from src.utils.logger import system_logger

# Import common utilities for enhanced functionality
from src.utils.common_operations import (
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    safe_dataframe_operation, validate_dataframe_columns
)
from src.utils.common_utilities import (
    safe_dataframe_operation, validate_dataframe_columns, 
    calculate_data_quality_metrics, optimize_memory_usage
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, validate_finite,
    validate_positive, validate_range, safe_nan_to_num
)
from src.utils.serialization_utils import UniversalSerializer
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations


@dataclass
class HMMDependencies:
	"""Enhanced HMM dependencies with common utilities integration."""
	logger: Any
	m1_gpu_manager: Any | None
	m1_memory_optimizer: Any | None
	m1_cpu_optimizer: Any | None = None
	serializer: Any = None
	matrix_ops: Any = None
	
	def __post_init__(self):
		"""Initialize common utilities if not provided."""
		if self.serializer is None:
			self.serializer = UniversalSerializer()
		if self.matrix_ops is None:
			self.matrix_ops = UnifiedMatrixOperations()


def train_hmm_gpu_optimized(
	features: Any,
	n_components: int,
	covariance_type: str,
	n_iter: int,
	random_state: int,
	deps: HMMDependencies,
) -> Dict[str, Any]:
	"""Train HMM using GPU optimization with enhanced common utilities integration."""
	from hmmlearn import hmm  # local import for optional dependency
	
	logger = getattr(deps, "logger", system_logger.getChild("HMMExecutor"))
	start_time = time.time()
	
	try:
		# Validate dependencies
		if deps.m1_memory_optimizer is None or deps.m1_gpu_manager is None:
			raise ValueError("GPU optimization dependencies not available")
		
		# Validate input features using common utilities
		if hasattr(features, 'empty') and features.empty:
			raise ValueError("Input features cannot be empty")
		
		# Convert to numpy and optimize memory usage
		if hasattr(features, 'values'):
			features_array = deps.m1_memory_optimizer.create_memory_efficient_array(
				features.values, dtype=np.float32
			)
		else:
			features_array = deps.m1_memory_optimizer.create_memory_efficient_array(
				features, dtype=np.float32
			)

		# Validate array using math validation utilities
		features_array = safe_nan_to_num(features_array)
		if not np.isfinite(features_array).all():
			logger.warning("Non-finite values detected in features, applying safe conversion")
			features_array = safe_nan_to_num(features_array)

		# Scale features using common utilities
		scaler = StandardScaler()
		features_scaled = scaler.fit_transform(features_array)
		
		# Validate scaled features
		features_scaled = safe_nan_to_num(features_scaled)

		# Use GPU acceleration with enhanced error handling
		features_scaled_gpu = deps.m1_gpu_manager.to_device(features_scaled, "matrix_mult")

		# Train HMM with GPU context and enhanced monitoring
		with deps.m1_gpu_manager.gpu_context("hmm_training"):
			hmm_model = hmm.GaussianHMM(
				n_components=n_components,
				covariance_type=covariance_type,
				n_iter=n_iter,
				random_state=random_state,
			)
			
			# Fit with progress monitoring
			logger.info(f"Training HMM with {n_components} components on GPU...")
			hmm_model.fit(features_scaled)

			# Convert back to CPU for predictions
			features_scaled_cpu = features_scaled_gpu.cpu().numpy()
			state_sequence = hmm_model.predict(features_scaled_cpu)
			state_probs = hmm_model.predict_proba(features_scaled_cpu)
			score = hmm_model.score(features_scaled_cpu)

		processing_time = time.time() - start_time
		logger.info(f"GPU HMM training completed in {processing_time:.2f}s")

		return {
			"model": hmm_model,
			"scaler": scaler,
			"state_sequence": state_sequence,
			"state_probs": state_probs,
			"n_components": n_components,
			"score": score,
			"used_gpu": True,
			"processing_time": processing_time,
			"memory_usage": deps.m1_memory_optimizer.get_memory_usage() if deps.m1_memory_optimizer else {},
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
	"""Train HMM on CPU with enhanced common utilities integration."""
	from hmmlearn import hmm  # local import for optional dependency
	
	logger = getattr(deps, "logger", system_logger.getChild("HMMExecutor"))
	start_time = time.time()
	
	try:
		# Validate input features using common utilities
		if features is None:
			raise ValueError("Input features cannot be None")
		if hasattr(features, 'empty') and features.empty:
			raise ValueError("Input features cannot be empty")
		
		# Convert to numpy array with proper handling
		if hasattr(features, 'values'):
			features_array = features.values
		else:
			features_array = np.array(features)
		
		# Apply CPU optimization if available
		if deps.m1_cpu_optimizer:
			features_array = deps.m1_cpu_optimizer.optimize_array(features_array)
		
		# Validate and clean data using math validation utilities
		features_array = safe_nan_to_num(features_array)
		if not np.isfinite(features_array).all():
			logger.warning("Non-finite values detected in features, applying safe conversion")
			features_array = safe_nan_to_num(features_array)

		# Scale features
		scaler = StandardScaler()
		features_scaled = scaler.fit_transform(features_array)
		
		# Validate scaled features
		features_scaled = safe_nan_to_num(features_scaled)

		# Train HMM with enhanced monitoring
		hmm_model = hmm.GaussianHMM(
			n_components=n_components,
			covariance_type=covariance_type,
			n_iter=n_iter,
			random_state=random_state,
		)

		logger.info(f"Training HMM with {n_components} components on CPU...")
		hmm_model.fit(features_scaled)
		
		# Generate predictions
		state_sequence = hmm_model.predict(features_scaled)
		state_probs = hmm_model.predict_proba(features_scaled)
		score = hmm_model.score(features_scaled)

		processing_time = time.time() - start_time
		logger.info(f"CPU HMM training completed in {processing_time:.2f}s")

		return {
			"model": hmm_model,
			"scaler": scaler,
			"state_sequence": state_sequence,
			"state_probs": state_probs,
			"n_components": n_components,
			"score": score,
			"used_gpu": False,
			"processing_time": processing_time,
			"memory_usage": deps.m1_memory_optimizer.get_memory_usage() if deps.m1_memory_optimizer else {},
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
	"""Choose optimal backend and train HMM with enhanced common utilities integration."""
	logger = getattr(deps, "logger", system_logger.getChild("HMMExecutor"))
	
	try:
		# Enhanced input validation using common utilities
		if features is None:
			raise ValueError("Features cannot be None")
		if hasattr(features, 'empty') and features.empty:
			raise ValueError("Features cannot be empty")
		
		# Validate parameters using math validation utilities
		n_components = validate_positive(n_components, "n_components")
		n_iter = validate_positive(n_iter, "n_iter")
		
		if covariance_type not in ['full', 'tied', 'diag', 'spherical']:
			raise ValueError(f"Invalid covariance_type: {covariance_type}")
		
		# Calculate data size for backend selection
		if hasattr(features, 'size'):
			data_size = features.size
		elif hasattr(features, 'shape'):
			data_size = np.prod(features.shape)
		else:
			data_size = len(features) if hasattr(features, '__len__') else 0
		
		# Enhanced backend selection with common utilities
		use_gpu = (
			data_size > 1_000_000 and 
			deps.m1_gpu_manager is not None and 
			deps.m1_memory_optimizer is not None
		)
		
		if use_gpu:
			logger.info("🎯 Using GPU for large dataset HMM training...")
			return train_hmm_gpu_optimized(features, n_components, covariance_type, n_iter, random_state, deps)
		else:
			logger.info("💻 Using CPU for HMM training...")
			return train_hmm_cpu_optimized(features, n_components, covariance_type, n_iter, random_state, deps)
	except Exception as e:
		logger.exception(f"HMM training failed: {e}")
		raise


def validate_hmm_model(hmm_model: Any, features: np.ndarray, n_components: int, logger: Any) -> Dict[str, Any]:
	"""Validate HMM convergence and quality metrics with enhanced common utilities integration."""
	validation_result: Dict[str, Any] = {
		"converged": False,
		"issues": [],
		"recommendations": [],
		"quality_metrics": {},
	}
	try:
		logger.info("🔍 Validating HMM model convergence and quality...")
		
		# Check convergence
		if hasattr(hmm_model, "monitor_") and getattr(hmm_model, "monitor_").converged:  # type: ignore[attr-defined]
			validation_result["converged"] = True
		else:
			validation_result["issues"].append("HMM did not converge")
			validation_result["recommendations"].append("Increase n_iter or adjust parameters")
		
		# Enhanced quality metrics using common utilities
		try:
			# Calculate regime stability
			state_sequence = hmm_model.predict(features)
			regime_changes = np.sum(np.diff(state_sequence) != 0)
			regime_stability = 1 - (regime_changes / len(state_sequence))
			validation_result["quality_metrics"]["regime_stability"] = regime_stability
			
			# Calculate regime balance
			unique_regimes, counts = np.unique(state_sequence, return_counts=True)
			regime_balance = 1 - np.std(counts) / np.mean(counts) if len(counts) > 1 else 1.0
			validation_result["quality_metrics"]["regime_balance"] = regime_balance
			
			# Calculate probability confidence
			state_probs = hmm_model.predict_proba(features)
			max_probs = np.max(state_probs, axis=1)
			avg_confidence = np.mean(max_probs)
			validation_result["quality_metrics"]["avg_confidence"] = avg_confidence
			
			# Add recommendations based on quality metrics
			if regime_stability < 0.5:
				validation_result["recommendations"].append("Low regime stability - consider adjusting parameters")
			if regime_balance < 0.3:
				validation_result["recommendations"].append("Poor regime balance - consider different n_components")
			if avg_confidence < 0.6:
				validation_result["recommendations"].append("Low confidence - consider feature engineering")
				
		except Exception as e:
			logger.warning(f"Could not calculate quality metrics: {e}")
			validation_result["issues"].append("Quality metrics calculation failed")
		
		return validation_result
	except Exception:
		logger.exception("HMM validation failed")
		validation_result["issues"].append("Validation error")
		return validation_result


def create_enhanced_dependencies(
	logger: Optional[Any] = None,
	use_gpu: bool = True,
	use_memory_optimization: bool = True,
	use_cpu_optimization: bool = True
) -> HMMDependencies:
	"""Create enhanced HMM dependencies with common utilities integration."""
	if logger is None:
		logger = system_logger.getChild("HMMExecutor")
	
	# Initialize hardware optimizers
	m1_gpu_manager = None
	m1_memory_optimizer = None
	m1_cpu_optimizer = None
	
	if use_gpu:
		try:
			m1_gpu_manager = get_m1_gpu_manager()
			if m1_gpu_manager:
				logger.info("✓ M1 GPU manager initialized")
		except Exception as e:
			logger.warning(f"Could not initialize M1 GPU manager: {e}")
	
	if use_memory_optimization:
		try:
			m1_memory_optimizer = get_m1_memory_optimizer()
			if m1_memory_optimizer:
				logger.info("✓ M1 memory optimizer initialized")
		except Exception as e:
			logger.warning(f"Could not initialize M1 memory optimizer: {e}")
	
	if use_cpu_optimization:
		try:
			m1_cpu_optimizer = get_m1_cpu_optimizer()
			if m1_cpu_optimizer:
				logger.info("✓ M1 CPU optimizer initialized")
		except Exception as e:
			logger.warning(f"Could not initialize M1 CPU optimizer: {e}")
	
	return HMMDependencies(
		logger=logger,
		m1_gpu_manager=m1_gpu_manager,
		m1_memory_optimizer=m1_memory_optimizer,
		m1_cpu_optimizer=m1_cpu_optimizer
	)


def save_hmm_model(model_data: Dict[str, Any], filepath: str, deps: HMMDependencies) -> bool:
	"""Save HMM model using common utilities serialization."""
	try:
		return deps.serializer.save(model_data, filepath)
	except Exception as e:
		deps.logger.error(f"Failed to save HMM model: {e}")
		return False


def load_hmm_model(filepath: str, deps: HMMDependencies) -> Optional[Dict[str, Any]]:
	"""Load HMM model using common utilities serialization."""
	try:
		return deps.serializer.load(filepath)
	except Exception as e:
		deps.logger.error(f"Failed to load HMM model: {e}")
		return None


def calculate_regime_characteristics(
	features: pd.DataFrame,
	state_sequence: np.ndarray,
	state_probs: np.ndarray,
	logger: Any
) -> Dict[str, Any]:
	"""Calculate detailed regime characteristics using common utilities."""
	try:
		characteristics = {}
		n_regimes = len(np.unique(state_sequence))
		
		for regime in range(n_regimes):
			regime_mask = state_sequence == regime
			regime_data = features[regime_mask]
			
			if len(regime_data) == 0:
				continue
			
			regime_char = {
				'count': len(regime_data),
				'percentage': len(regime_data) / len(features) * 100,
				'avg_confidence': np.mean(state_probs[regime_mask, regime]),
			}
			
			# Add feature-specific characteristics
			for col in regime_data.columns:
				if col in ['close', 'open', 'high', 'low']:
					regime_char[f'{col}_mean'] = regime_data[col].mean()
					regime_char[f'{col}_std'] = regime_data[col].std()
				elif 'returns' in col:
					regime_char[f'{col}_mean'] = regime_data[col].mean()
					regime_char[f'{col}_volatility'] = regime_data[col].std()
			
			characteristics[f'regime_{regime}'] = regime_char
		
		return characteristics
		
	except Exception as e:
		logger.error(f"Failed to calculate regime characteristics: {e}")
		return {}


def calculate_feature_importance(
	features: pd.DataFrame,
	state_sequence: np.ndarray,
	logger: Any
) -> Dict[str, float]:
	"""Calculate feature importance for regime detection using common utilities."""
	try:
		from sklearn.ensemble import RandomForestClassifier
		
		# Train a random forest to predict regimes
		rf = RandomForestClassifier(n_estimators=100, random_state=42)
		rf.fit(features, state_sequence)
		
		# Get feature importance
		importance = rf.feature_importances_
		feature_names = features.columns.tolist()
		
		return dict(zip(feature_names, importance))
		
	except Exception as e:
		logger.error(f"Failed to calculate feature importance: {e}")
		return {}

