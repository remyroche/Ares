"""
Core Optimization Logic for Feature Lookback Optimization.

This module contains the main optimization algorithms and core functionality.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import itertools
from pathlib import Path
from datetime import datetime
from functools import lru_cache
import hashlib
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import utility modules
from src.utils.common_operations import safe_dataframe_operation, validate_dataframe_columns
from src.utils.common_utilities import CommonUtilities
from src.utils.math_validation import safe_divide, safe_correlation

# Import matrix operations for vectorized processing
try:
    from src.utils.matrix_operations.unified_operations import get_unified_matrix_operations
    from src.utils.matrix_operations.batch_operations import BatchMatrixProcessor
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

# Import multi-horizon profit labeler for target alignment
try:
    from ..multi_horizon_profit_labeler import MultiHorizonConfig, apply_multi_horizon_labeling
    MULTI_HORIZON_AVAILABLE = True
except ImportError:
    MULTI_HORIZON_AVAILABLE = False
from src.utils.serialization_utils import UniversalSerializer
from src.utils.tprint import tprint, tprint_error, tprint_success, tprint_warning, tprint_debug
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time

from ..constants import OPTIMIZATION_CONSTANTS, PERFORMANCE_CONSTANTS, ALGORITHM_CONSTANTS
from ..dependency_manager import get_dependency
from src.training.config.data_locator import DataLocator as PipelineDataLocator

# Get dependencies with fallbacks
np, _ = get_dependency('numpy')
pd, _ = get_dependency('pandas')


class OptimizationMethod(Enum):
    """Available optimization methods."""
    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"
    MRMR = "mrmr"
    RANDOM_SEARCH = "random_search"
    MULTI_TARGET = "multi_target"
    COARSE_TO_REFINE = "coarse_to_refine"


@dataclass
class LookbackConstraints:
    """Constraints for lookback optimization."""
    min_lookback: int = 5
    max_lookback: int = 300
    search_step: int = 5
    enable_regularization: bool = True
    regularization_strength: float = 0.1
    preferred_lookback: int = 50
    min_stability_score: float = 0.7


@dataclass
class OptimizationResult:
    """Standardized optimization result."""
    best_lookback_period: int
    best_score: float
    optimization_method: str
    total_trials: int
    optimization_time: float
    convergence_achieved: bool
    metadata: Dict[str, Any]
    stability_score: float = 0.0  # Added for validation
    lookback_sensitivity: float = 0.0  # Added for validation

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        tprint_debug("🧮 Converting OptimizationResult to dictionary")
        # Ensure metadata contains serializable values
        def convert_metadata(obj):
            tprint_debug(f"   ↳ Normalizing metadata type: {type(obj).__name__}")
            if isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_metadata(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_metadata(item) for item in obj]
            else:
                return obj

        return {
            'best_lookback_period': int(self.best_lookback_period) if isinstance(self.best_lookback_period, np.int64) else self.best_lookback_period,
            'best_score': self.best_score,
            'optimization_method': self.optimization_method,
            'total_trials': int(self.total_trials) if isinstance(self.total_trials, np.int64) else self.total_trials,
            'optimization_time': self.optimization_time,
            'convergence_achieved': self.convergence_achieved,
            'stability_score': self.stability_score,
            'lookback_sensitivity': self.lookback_sensitivity,
            'metadata': convert_metadata(self.metadata)
        }


class CoreOptimizer:
    """
    Core optimization engine for feature lookback optimization.

    Provides standardized interface for different optimization algorithms.
    """

    def __init__(self, logger=None, rng: Optional['np.random.Generator'] = None):
        """Initialize the core optimizer."""
        self.logger = logger or get_logger('CoreOptimizer')
        self.common_utils = CommonUtilities()
        self.serializer = UniversalSerializer()

        self._rng: 'np.random.Generator' = rng or np.random.default_rng()
        
        tprint("🔧 Initializing Core Optimizer...")
        tprint("   → Performance tracking enabled")
        tprint("   → Feature calculation cache initialized")
        tprint("   → Shared forward returns cache ready")
        
        # Performance tracking
        self.optimization_history = []
        self.performance_metrics = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'average_optimization_time': 0.0,
            'best_scores': []
        }
        
        # Feature calculation cache
        self.feature_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Track lag metadata for generated features
        self.feature_lag_metadata: Dict[str, Dict[int, Dict[str, Any]]] = {}
        
        # Shared forward returns matrix cache (reused across all features)
        self.shared_forward_returns = {}
        self.shared_forward_returns_hash = None

        # Initialize matrix operations if available
        self.matrix_ops = None
        self.batch_processor = None
        if MATRIX_OPS_AVAILABLE:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.batch_processor = BatchMatrixProcessor(chunk_size_mb=128, enable_gpu=True)
                self.logger.info("✅ Matrix operations initialized for vectorized processing")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not initialize matrix operations: {e}")
                self.matrix_ops = None
                self.batch_processor = None

        self._cached_multi_horizon_limits: Optional[Tuple[int, int]] = None
        self._data_locator: Optional[PipelineDataLocator] = None

    def set_rng(self, rng: Optional['np.random.Generator']) -> None:
        """Update the RNG used for stochastic routines."""
        self._rng = rng or np.random.default_rng()

    def set_data_locator(self, locator: Optional[PipelineDataLocator]) -> None:
        """Attach a locator used when resolving shared configuration files."""

        self._data_locator = locator
        self._cached_multi_horizon_limits = None

    def optimize_single_feature(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        method: OptimizationMethod = OptimizationMethod.MRMR,
        lookback_range: Tuple[int, int] = (5, 300),
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize lookback period for a single feature.

        Args:
            data: Input data with features and target
            feature_name: Name of the feature to optimize
            target_column: Target column for optimization
            method: Optimization method to use
            lookback_range: Min and max lookback periods to test
            **kwargs: Additional parameters for optimization method

        Returns:
            OptimizationResult with best lookback period and score
        """
        try:
            start_time = time.time()
            self.logger.info(f'🎯 Starting optimization for feature: {feature_name} using {method.value}')
            tprint(f"🎯 Starting optimization for feature: {feature_name} using {method.value}")

            # Validate inputs
            if not self._validate_optimization_inputs(data, feature_name, target_column):
                tprint_error(f"❌ Input validation failed for feature: {feature_name}")
                return self._create_failed_result(method.value, time.time() - start_time)

            # Select optimization algorithm based on method
            if method == OptimizationMethod.MRMR:
                result = self._optimize_mrmr(data, feature_name, target_column, lookback_range, **kwargs)
            elif method == OptimizationMethod.GRID_SEARCH:
                result = self._optimize_grid_search(data, feature_name, target_column, lookback_range, **kwargs)
            elif method == OptimizationMethod.BAYESIAN:
                result = self._optimize_bayesian(data, feature_name, target_column, lookback_range, **kwargs)
            elif method == OptimizationMethod.RANDOM_SEARCH:
                result = self._optimize_random_search(data, feature_name, target_column, lookback_range, **kwargs)
            elif method == OptimizationMethod.MULTI_TARGET:
                result = self._optimize_multi_target(data, feature_name, target_column, lookback_range, **kwargs)
            elif method == OptimizationMethod.COARSE_TO_REFINE:
                result = self._optimize_coarse_to_refine(data, feature_name, target_column, lookback_range, **kwargs)
            else:
                # Fallback to MRMR
                self.logger.warning(f'⚠️ Unknown method {method.value}, falling back to MRMR')
                result = self._optimize_mrmr(data, feature_name, target_column, lookback_range, **kwargs)

            result.optimization_time = time.time() - start_time
            result.optimization_method = method.value

            # Update performance tracking
            self._update_performance_metrics(result, time.time() - start_time)

            self.logger.info(f'✅ Optimization completed: best_lookback={result.best_lookback_period}, score={result.best_score:.4f}')
            tprint_success(f"✅ Optimization completed for {feature_name}: best_lookback={result.best_lookback_period}, score={result.best_score:.4f}")
            return result

        except Exception as e:
            self.logger.error(f"❌ Optimization failed for feature {feature_name}: {e}")
            tprint_error(f"❌ Optimization failed for feature {feature_name}: {e}")
            return self._create_failed_result(method.value, time.time() - start_time)

    def _validate_optimization_inputs(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str
    ) -> bool:
        """Validate inputs for optimization."""
        try:
            tprint_debug(f"🔍 Validating optimization inputs for feature '{feature_name}' against target '{target_column}'")
            # Check if data is valid DataFrame
            if not isinstance(data, pd.DataFrame):
                self.logger.error("Input data must be a pandas DataFrame")
                tprint_error("❌ Optimization data must be a pandas DataFrame")
                return False

            # Check if feature and target columns exist
            if feature_name not in data.columns:
                self.logger.error(f"Feature column '{feature_name}' not found in data")
                tprint_error(f"❌ Feature column '{feature_name}' not found in input data")
                return False

            if target_column not in data.columns:
                self.logger.error(f"Target column '{target_column}' not found in data")
                tprint_error(f"❌ Target column '{target_column}' not found in input data")
                return False

            # Check for sufficient data
            if len(data) < OPTIMIZATION_CONSTANTS.DEFAULT_MIN_LOOKBACK:
                self.logger.error(f"Insufficient data: {len(data)} rows, minimum {OPTIMIZATION_CONSTANTS.DEFAULT_MIN_LOOKBACK} required")
                tprint_warning(
                    f"⚠️ Insufficient rows for optimization: {len(data)} < {OPTIMIZATION_CONSTANTS.DEFAULT_MIN_LOOKBACK}"
                )
                return False

            tprint_debug("✅ Optimization inputs validated successfully")
            return True

        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            tprint_error(f"❌ Exception during input validation: {e}")
            return False

    def _optimize_mrmr(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        lookback_range: Tuple[int, int],
        **kwargs
    ) -> OptimizationResult:
        """Optimize using MRMR approach with proper cross-validation."""
        try:
            min_lookback, max_lookback = lookback_range
            tprint_debug(
                f"🧠 Running MRMR optimization for '{feature_name}' in range [{min_lookback}, {max_lookback}]"
            )
            best_score = -float('inf')
            best_lookback = min_lookback
            trials = 0

            # Use time series cross-validation to avoid data leakage
            # Split data: use first 70% for training, last 30% for testing
            split_point = int(len(data) * 0.7)

            if split_point < min_lookback:
                # Not enough data for cross-validation, fall back to full data
                self.logger.warning(f"Insufficient data for cross-validation ({len(data)} < {min_lookback * 1.4:.0f}), using full data")
                tprint_warning(
                    f"⚠️ Using full dataset for MRMR due to insufficient cross-validation rows ({len(data)})"
                )
                train_data = data
                test_data = data
            else:
                train_data = data.iloc[:split_point]
                test_data = data.iloc[split_point:]
                tprint_debug(
                    f"   ↳ MRMR split at index {split_point}: train={len(train_data)}, test={len(test_data)}"
                )

            # Test different lookback periods using cross-validation
            for lookback in range(min_lookback, max_lookback + 1):
                try:
                    # Calculate feature on training data
                    train_features = self._calculate_feature_for_lookback(train_data, feature_name, lookback)
                    test_features = self._calculate_feature_for_lookback(test_data, feature_name, lookback)

                    # Align data lengths (features might be shorter due to rolling windows)
                    min_length = min(len(train_features), len(test_features), len(train_data[target_column]), len(test_data[target_column]))
                    if min_length <= 1:
                        continue

                    # Calculate correlation on test data to avoid overfitting
                    correlation = safe_correlation(test_features[:min_length], test_data[target_column].values[:min_length], default=0.0)
                    score = abs(correlation)  # Use absolute correlation as score

                    trials += 1

                    if score > best_score:
                        best_score = score
                        best_lookback = lookback
                        tprint_debug(
                            f"   ✅ New MRMR best for '{feature_name}': lookback={lookback}, score={score:.4f}"
                        )

                except Exception as e:
                    self.logger.warning(f"Failed to evaluate lookback {lookback} for {feature_name}: {e}")
                    tprint_warning(f"⚠️ MRMR evaluation failed for lookback {lookback}: {e}")
                    continue

            tprint_success(
                f"🏁 MRMR optimization finished for '{feature_name}' with best lookback {best_lookback} (score={best_score:.4f})"
            )
            return OptimizationResult(
                best_lookback_period=best_lookback,
                best_score=best_score,
                optimization_method="mrmr",
                total_trials=trials,
                optimization_time=0.0,  # Will be set by caller
                convergence_achieved=trials > 0,
                metadata={
                    'feature_name': feature_name,
                    'target_column': target_column,
                    'lookback_range': lookback_range,
                    'correlation_method': 'pearson',
                    'cross_validation': True,
                    'train_size': len(train_data),
                    'test_size': len(test_data)
                }
            )

        except Exception as e:
            self.logger.error(f"MRMR optimization failed: {e}")
            return self._create_failed_result("mrmr", 0.0)

    def _optimize_grid_search(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        lookback_range: Tuple[int, int],
        **kwargs
    ) -> OptimizationResult:
        """Optimize using comprehensive grid search approach with cross-validation."""
        try:
            min_lookback, max_lookback = lookback_range
            step_size = kwargs.get('step_size', 1)

            self.logger.info(f'🔍 Running grid search from {min_lookback} to {max_lookback} (step={step_size})')
            tprint_debug(
                f"🧭 Starting grid search for '{feature_name}' range [{min_lookback}, {max_lookback}] step={step_size}"
            )

            # Use time series cross-validation to avoid data leakage
            split_point = int(len(data) * 0.7)

            if split_point < min_lookback:
                self.logger.warning(f"Insufficient data for cross-validation ({len(data)} < {min_lookback * 1.4:.0f}), using full data")
                tprint_warning(
                    f"⚠️ Grid search using full dataset due to insufficient cross-validation rows ({len(data)})"
                )
                train_data = data
                test_data = data
            else:
                train_data = data.iloc[:split_point]
                test_data = data.iloc[split_point:]
                tprint_debug(
                    f"   ↳ Grid search split at {split_point}: train={len(train_data)}, test={len(test_data)}"
                )

            best_score = -float('inf')
            best_lookback = min_lookback
            all_scores = []
            trials = 0

            # Test all lookback periods in range using cross-validation
            for lookback in range(min_lookback, max_lookback + 1, step_size):
                try:
                    # Calculate feature on both train and test data
                    train_features = self._calculate_feature_for_lookback(train_data, feature_name, lookback)
                    test_features = self._calculate_feature_for_lookback(test_data, feature_name, lookback)

                    # Align data lengths
                    min_length = min(len(train_features), len(test_features), len(train_data[target_column]), len(test_data[target_column]))
                    if min_length <= 1:
                        continue

                    # Calculate correlations on test data to avoid overfitting
                    correlations = self._calculate_comprehensive_correlations(
                        test_features[:min_length], test_data[target_column].values[:min_length]
                    )

                    # Use weighted combination of correlation metrics
                    score = self._calculate_composite_score(correlations)
                    all_scores.append(score)
                    trials += 1

                    if score > best_score:
                        best_score = score
                        best_lookback = lookback
                        tprint_debug(
                            f"   ✅ Grid search best updated: lookback={lookback}, score={score:.4f}"
                        )

                    if trials % 10 == 0:
                        self.logger.debug(f'   → Progress: {trials} trials, best_score={best_score:.4f}')
                        tprint_debug(
                            f"   ↺ Grid search progress: {trials} trials, current best={best_score:.4f}"
                        )

                except Exception as e:
                    self.logger.warning(f'⚠️ Failed to evaluate lookback {lookback}: {e}')
                    tprint_warning(f"⚠️ Grid search evaluation failed for lookback {lookback}: {e}")
                    continue

            # Calculate convergence metrics
            convergence_achieved = self._check_convergence(all_scores)
            tprint_debug(
                f"📈 Grid search convergence {'achieved' if convergence_achieved else 'not achieved'} with {trials} trials"
            )

            return OptimizationResult(
                best_lookback_period=best_lookback,
                best_score=best_score,
                optimization_method="grid_search",
                total_trials=trials,
                optimization_time=0.0,  # Will be set by caller
                convergence_achieved=convergence_achieved,
                metadata={
                    'feature_name': feature_name,
                    'target_column': target_column,
                    'lookback_range': lookback_range,
                    'step_size': step_size,
                    'all_scores': all_scores,
                    'score_std': np.std(all_scores) if all_scores else 0.0,
                    'cross_validation': True,
                    'train_size': len(train_data),
                    'test_size': len(test_data)
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Grid search optimization failed: {e}")
            return self._create_failed_result("grid_search", 0.0)

    def _optimize_bayesian(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        lookback_range: Tuple[int, int],
        **kwargs
    ) -> OptimizationResult:
        """Optimize using Bayesian optimization approach with TPE."""
        try:
            min_lookback, max_lookback = lookback_range
            n_trials = kwargs.get('n_trials', 50)
            n_startup_trials = kwargs.get('n_startup_trials', 10)

            self.logger.info(f'🎯 Running Bayesian optimization with {n_trials} trials')
            tprint_debug(
                f"🧪 Starting Bayesian optimization for '{feature_name}' range [{min_lookback}, {max_lookback}] with {n_trials} trials"
            )
            
            # Initialize with random samples for exploration
            startup_trials = self._rng.integers(min_lookback, max_lookback + 1, n_startup_trials)
            all_scores = []
            all_lookbacks = []
            
            # Startup phase - random exploration
            for lookback in startup_trials:
                try:
                    feature_values = self._calculate_feature_for_lookback(data, feature_name, lookback)
                    correlations = self._calculate_comprehensive_correlations(
                        feature_values, data[target_column].values
                    )
                    score = self._calculate_composite_score(correlations)

                    all_scores.append(score)
                    all_lookbacks.append(lookback)
                    tprint_debug(
                        f"   🔄 Bayesian startup trial lookback={lookback}, score={score:.4f}"
                    )

                except Exception as e:
                    self.logger.warning(f'⚠️ Startup trial failed for lookback {lookback}: {e}')
                    tprint_warning(f"⚠️ Bayesian startup trial failed for lookback {lookback}: {e}")
                    continue

            # Bayesian optimization phase
            for trial in range(n_startup_trials, n_trials):
                try:
                    # Use simple acquisition function (exploration vs exploitation)
                    if len(all_scores) < 5:
                        # More exploration
                        lookback = int(self._rng.integers(min_lookback, max_lookback + 1))
                    else:
                        # Exploit best regions
                        best_idx = np.argmax(all_scores)
                        best_lookback = all_lookbacks[best_idx]
                        
                        # Add some exploration around best point
                        exploration_range = max(1, (max_lookback - min_lookback) // 10)
                        lookback = int(
                            self._rng.integers(
                                max(min_lookback, best_lookback - exploration_range),
                                min(max_lookback + 1, best_lookback + exploration_range + 1)
                            )
                        )
                    
                    feature_values = self._calculate_feature_for_lookback(data, feature_name, lookback)
                    correlations = self._calculate_comprehensive_correlations(
                        feature_values, data[target_column].values
                    )
                    score = self._calculate_composite_score(correlations)

                    all_scores.append(score)
                    all_lookbacks.append(lookback)
                    tprint_debug(
                        f"   🎯 Bayesian trial {trial}: lookback={lookback}, score={score:.4f}"
                    )

                    if trial % 10 == 0:
                        current_best = max(all_scores)
                        self.logger.debug(f'   → Trial {trial}: best_score={current_best:.4f}')
                        tprint_debug(
                            f"   📊 Bayesian progress trial {trial}: best_score={current_best:.4f}"
                        )

                except Exception as e:
                    self.logger.warning(f'⚠️ Bayesian trial failed: {e}')
                    tprint_warning(f"⚠️ Bayesian trial failure at iteration {trial}: {e}")
                    continue

            # Find best result
            if all_scores:
                best_idx = np.argmax(all_scores)
                best_score = all_scores[best_idx]
                best_lookback = all_lookbacks[best_idx]
                convergence_achieved = self._check_convergence(all_scores)
                tprint_success(
                    f"🏁 Bayesian optimization finished for '{feature_name}' with lookback {best_lookback} (score={best_score:.4f})"
                )
            else:
                best_score = 0.0
                best_lookback = min_lookback
                convergence_achieved = False
                tprint_warning(
                    f"⚠️ Bayesian optimization produced no valid scores for '{feature_name}', using fallback values"
                )
            
            return OptimizationResult(
                best_lookback_period=best_lookback,
                best_score=best_score,
                optimization_method="bayesian",
                total_trials=len(all_scores),
                optimization_time=0.0,  # Will be set by caller
                convergence_achieved=convergence_achieved,
                metadata={
                    'feature_name': feature_name,
                    'target_column': target_column,
                    'lookback_range': lookback_range,
                    'n_trials': n_trials,
                    'n_startup_trials': n_startup_trials,
                    'all_scores': all_scores,
                    'all_lookbacks': all_lookbacks,
                    'score_improvement': max(all_scores) - min(all_scores) if all_scores else 0.0
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Bayesian optimization failed: {e}")
            return self._create_failed_result("bayesian", 0.0)

    def _optimize_random_search(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        lookback_range: Tuple[int, int],
        **kwargs
    ) -> OptimizationResult:
        """Optimize using random search approach."""
        try:
            min_lookback, max_lookback = lookback_range
            n_trials = kwargs.get('n_trials', 30)

            self.logger.info(f'🎲 Running random search with {n_trials} trials')
            tprint_debug(
                f"🎲 Starting random search for '{feature_name}' between {min_lookback} and {max_lookback} ({n_trials} trials)"
            )
            
            best_score = -float('inf')
            best_lookback = min_lookback
            all_scores = []
            trials = 0
            
            # Random sampling
            for trial in range(n_trials):
                try:
                    lookback = int(self._rng.integers(min_lookback, max_lookback + 1))

                    feature_values = self._calculate_feature_for_lookback(data, feature_name, lookback)
                    correlations = self._calculate_comprehensive_correlations(
                        feature_values, data[target_column].values
                    )
                    score = self._calculate_composite_score(correlations)
                    
                    all_scores.append(score)
                    trials += 1

                    if score > best_score:
                        best_score = score
                        best_lookback = lookback
                        tprint_debug(
                            f"   ✅ Random search best updated on trial {trial}: lookback={lookback}, score={score:.4f}"
                        )

                except Exception as e:
                    self.logger.warning(f'⚠️ Random trial failed: {e}')
                    tprint_warning(f"⚠️ Random search trial {trial} failed: {e}")
                    continue

            convergence_achieved = self._check_convergence(all_scores)
            tprint_debug(
                f"📈 Random search convergence {'achieved' if convergence_achieved else 'not achieved'} after {trials} trials"
            )

            return OptimizationResult(
                best_lookback_period=best_lookback,
                best_score=best_score,
                optimization_method="random_search",
                total_trials=trials,
                optimization_time=0.0,  # Will be set by caller
                convergence_achieved=convergence_achieved,
                metadata={
                    'feature_name': feature_name,
                    'target_column': target_column,
                    'lookback_range': lookback_range,
                    'n_trials': n_trials,
                    'all_scores': all_scores,
                    'score_std': np.std(all_scores) if all_scores else 0.0
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Random search optimization failed: {e}")
            return self._create_failed_result("random_search", 0.0)

    def _optimize_multi_target(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        lookback_range: Tuple[int, int],
        **kwargs
    ) -> OptimizationResult:
        """Optimize using multi-target approach for multiple objectives."""
        try:
            min_lookback, max_lookback = lookback_range
            step_size = kwargs.get('step_size', 1)

            self.logger.info(f'🎯 Running multi-target optimization')
            tprint_debug(
                f"🎯 Starting multi-target optimization for '{feature_name}' range [{min_lookback}, {max_lookback}] step={step_size}"
            )
            
            best_score = -float('inf')
            best_lookback = min_lookback
            all_scores = []
            trials = 0
            
            # Test all lookback periods
            for lookback in range(min_lookback, max_lookback + 1, step_size):
                try:
                    feature_values = self._calculate_feature_for_lookback(data, feature_name, lookback)

                    # Calculate multiple target metrics
                    targets = self._calculate_multi_target_metrics(
                        feature_values, data[target_column].values
                    )

                    # Multi-objective optimization using weighted sum
                    score = self._calculate_multi_objective_score(targets)
                    all_scores.append(score)
                    trials += 1

                    if score > best_score:
                        best_score = score
                        best_lookback = lookback
                        tprint_debug(
                            f"   ✅ Multi-target best updated: lookback={lookback}, score={score:.4f}"
                        )

                except Exception as e:
                    self.logger.warning(f'⚠️ Multi-target trial failed for lookback {lookback}: {e}')
                    tprint_warning(f"⚠️ Multi-target evaluation failed for lookback {lookback}: {e}")
                    continue

            convergence_achieved = self._check_convergence(all_scores)
            tprint_debug(
                f"📈 Multi-target convergence {'achieved' if convergence_achieved else 'not achieved'} after {trials} trials"
            )

            return OptimizationResult(
                best_lookback_period=best_lookback,
                best_score=best_score,
                optimization_method="multi_target",
                total_trials=trials,
                optimization_time=0.0,  # Will be set by caller
                convergence_achieved=convergence_achieved,
                metadata={
                    'feature_name': feature_name,
                    'target_column': target_column,
                    'lookback_range': lookback_range,
                    'all_scores': all_scores,
                    'multi_target_weights': kwargs.get('target_weights', {})
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Multi-target optimization failed: {e}")
            return self._create_failed_result("multi_target", 0.0)

    def _calculate_feature_for_lookback(
        self,
        data: pd.DataFrame,
        feature_name: str,
        lookback: int
    ) -> np.ndarray:
        """
        Calculate sophisticated feature values for a given lookback period.

        This implementation uses actual technical indicators from the feature engineering
        pipeline including RSI, MACD, Bollinger Bands, moving averages, and other indicators.
        """
        try:
            tprint_debug(
                f"🧮 Calculating feature '{feature_name}' values for lookback {lookback}"
            )
            # Create feature generator based on feature name pattern
            feature_generator = self._create_feature_generator(feature_name, lookback)

            if feature_generator is None:
                # Fallback to rolling mean for unknown features
                if feature_name in data.columns:
                    tprint_warning(
                        f"⚠️ No generator found for '{feature_name}', using rolling mean fallback"
                    )
                    return data[feature_name].rolling(window=lookback, min_periods=1).mean().values
                else:
                    tprint_warning(
                        f"⚠️ Feature '{feature_name}' not in dataframe, returning zeros"
                    )
                    return np.zeros(len(data))

            tprint_debug(
                f"   ↳ Using generator {type(feature_generator).__name__} for '{feature_name}'"
            )
            # Generate feature using the technical indicator
            feature_result = feature_generator.generate(data)

            if feature_result.success and feature_result.data is not None:
                # Handle different return types from generators
                feature_data = feature_result.data

                # For Bollinger Bands, extract the specific band we want based on feature name
                if 'bb_' in feature_name.lower():
                    # Handle both DataFrame and Series cases
                    if isinstance(feature_data, pd.DataFrame):
                        if 'upper' in feature_name.lower() and len(feature_data.columns) > 0:
                            return feature_data.iloc[:, 0].values  # Upper band
                        elif 'lower' in feature_name.lower() and len(feature_data.columns) > 2:
                            return feature_data.iloc[:, 2].values  # Lower band
                        elif 'middle' in feature_name.lower() and len(feature_data.columns) > 1:
                            return feature_data.iloc[:, 1].values  # Middle band
                    elif isinstance(feature_data, pd.Series):
                        # If it's a Series, return it directly
                        return feature_data.values

                # For other indicators, return the single series or first column
                if isinstance(feature_data, pd.DataFrame):
                    if len(feature_data.columns) > 0:
                        tprint_debug(
                            f"   ↳ Returning first column from DataFrame for '{feature_name}'"
                        )
                        return feature_data.iloc[:, 0].values
                    else:
                        tprint_warning(
                            f"⚠️ Generated DataFrame empty for '{feature_name}', returning zeros"
                        )
                        return np.zeros(len(data))
                elif isinstance(feature_data, pd.Series):
                    tprint_debug(
                        f"   ↳ Returning Series values for '{feature_name}'"
                    )
                    return feature_data.values
                else:
                    tprint_debug(
                        f"   ↳ Converting generated data to numpy array for '{feature_name}'"
                    )
                    return np.array(feature_data)
            else:
                self.logger.warning(f"Feature generation failed for {feature_name}, using fallback")
                tprint_warning(
                    f"⚠️ Feature generation unsuccessful for '{feature_name}', returning zeros"
                )
                return np.zeros(len(data))

        except ImportError as e:
            self.logger.warning(f"Feature engineering modules not available: {e}, using fallback")
            # Fallback to rolling mean
            if feature_name in data.columns:
                tprint_warning(
                    f"⚠️ Feature modules missing ({e}), using rolling mean for '{feature_name}'"
                )
                return data[feature_name].rolling(window=lookback, min_periods=1).mean().values
            else:
                tprint_error(
                    f"❌ Feature modules missing and '{feature_name}' not in data, returning zeros"
                )
                return np.zeros(len(data))
        except Exception as e:
            self.logger.error(f"Failed to calculate feature {feature_name} for lookback {lookback}: {e}")
            tprint_error(
                f"❌ Exception during feature calculation for '{feature_name}' (lookback={lookback}): {e}"
            )
            return np.zeros(len(data))

    def _create_feature_generator(self, feature_name: str, lookback: int):
        """
        Create appropriate feature generator based on feature name pattern for ALL indicators
        from the feature engineering bank (excluding wavelets and autoencoders).

        Args:
            feature_name: Name of the feature (e.g., 'rsi_14', 'macd_12_26_9', 'bb_upper_20')
            lookback: Lookback period for optimization

        Returns:
            FeatureGenerator instance or None if not recognized
        """
        tprint_debug("🧠 Entering _create_feature_generator")
        try:
            from src.feature_generation.base_calculations.base_calculator import BaseCalculationType

            # Import ALL feature generators needed from the feature bank
            # Momentum indicators
            from src.feature_generation.categories.momentum import (
                RSIGenerator, MACDGenerator, StochasticGenerator, WilliamsRGenerator,
                MomentumOscillatorGenerator, RateOfChangeGenerator
            )

            # Volatility indicators
            from src.feature_generation.categories.volatility import (
                BollingerBandsGenerator, ATRGenerator, VolatilityBandsGenerator,
                VolatilityFeatureGenerator, GARCHFeatureGenerator
            )

            # Trend indicators
            from src.feature_generation.categories.trend import (
                SMAGenerator, EMAGenerator, WMAGenerator, DEMAGenerator,
                TEMAGenerator, TRIMAGenerator, VWMAGenerator, KeltnerChannelsGenerator
            )

            # Oscillator indicators
            from src.feature_generation.categories.oscillator import (
                CCIGenerator, ADXGenerator, AroonGenerator, UltimateOscillatorGenerator,
                KSTGenerator, APOGenerator, CMOGenerator, NATRGenerator, PFEGenerator,
                T3Generator, KAMAGenerator
            )

            # Volume indicators
            from src.feature_generation.categories.volume import (
                VolumeSMAGenerator, VolumeEMAGenerator, VolumeRatioGenerator,
                VolumeROCGenerator, VolumeStdGenerator, VolumePercentileGenerator,
                VolumeTrendStrengthGenerator, VolumeOscillatorGenerator,
                VolumeMomentumGenerator, VolumeVWAPGenerator, VolumePriceTrendGenerator,
                VolumeAccumulationDistributionGenerator
            )

            # Support/Resistance indicators
            from src.feature_generation.categories.support_resistance import (
                SupportLevelGenerator, ResistanceLevelGenerator, PivotPointGenerator,
                FibonacciLevelGenerator
            )

            # Returns indicators
            from src.feature_generation.categories.returns import (
                SimpleReturnsGenerator, LogReturnsGenerator, CumulativeReturnsGenerator
            )

            # Entropy indicators
            from src.feature_generation.categories.entropy import (
                PriceEntropyGenerator, VolumeEntropyGenerator, ReturnEntropyGenerator,
                PriceEntropyMAGenerator, VolumeEntropyMAGenerator, ReturnEntropyMAGenerator,
                HighLowEntropyGenerator, VolatilityEntropyGenerator, MomentumEntropyGenerator,
                RSIEntropyGenerator, MACDEntropyGenerator, BollingerBandsEntropyGenerator,
                CrossAssetEntropyGenerator, RegimeEntropyGenerator
            )

            # Acceleration indicators
            from src.feature_generation.categories.acceleration import (
                MomentumGenerator, PriceAccelerationGenerator, PriceJerkGenerator,
                TrendStrengthGenerator, TrendConsistencyGenerator,
                VolumeAccelerationGenerator, VolatilityAccelerationGenerator
            )

            # Interaction indicators
            from src.feature_generation.categories.interaction import (
                CrossTimeframeInteractionGenerator, FeatureRatioGenerator,
                PolynomialFeatureGenerator, CorrelationInteractionGenerator
            )

            # Cross-timeframe indicators
            from src.feature_generation.categories.cross_timeframe import (
                CrossTimeframeMomentumGenerator, CrossTimeframeVolatilityGenerator,
                CrossTimeframeVolumeGenerator, CrossTimeframeTrendGenerator,
                CrossTimeframeHighLowGenerator, CrossTimeframeRatioGenerator,
                CrossTimeframeCorrelationGenerator, CrossTimeframeDivergenceGenerator
            )

            # Microstructure indicators
            from src.feature_generation.categories.microstructure import (
                BidAskSpreadGenerator, OrderFlowImbalanceGenerator, TradeSizeImbalanceGenerator,
                PriceImpactGenerator, VolumeWeightedPriceGenerator, TradeIntensityGenerator,
                LiquidityProxyGenerator, MarketDepthGenerator
            )

            # Order flow indicators
            from src.feature_generation.categories.order_flow import (
                TakerBuyRatioGenerator, TakerSellRatioGenerator, MarketAggressionIndexGenerator,
                OrderFlowImbalanceGenerator as OrderFlowImbalanceGeneratorOF
            )

            # Candlestick pattern indicators (placeholder implementations)
            from src.feature_generation.categories.candlestick_pattern import (
                CandlestickPatternFeatureGenerator
            )

            # Advanced SR features (these are calculated from historical SR data)
            from src.feature_generation.utils.enhanced_sr_feature_extractor import (
                EnhancedSRFeatureExtractor, HistoricalSRAnalyzer, HistoricalSRConfig
            )

            # Parse feature name to determine type and parameters
            name_lower = feature_name.lower()

            # Skip wavelets and autoencoders as requested
            if 'wavelet' in name_lower or 'autoencoder' in name_lower:
                return None

            # MOMENTUM INDICATORS - Default to RETURNS_VWAP, but support all base calculations
            if name_lower.startswith('rsi'):
                period = self._extract_period_from_name(feature_name, 14)
                if 'price' in name_lower:
                    return RSIGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    if 'vwap' in name_lower:
                        return RSIGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)
                    else:
                        # Explicit PRICE_RETURNS variant
                        return RSIGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return RSIGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('macd'):
                params = self._extract_macd_params(feature_name)
                if params:
                    fast, slow, signal = params
                    if 'price' in name_lower:
                        return MACDGenerator(fast=fast, slow=slow, signal=signal, base_calculation=BaseCalculationType.PRICE_LEVELS)
                    elif 'returns' in name_lower:
                        if 'vwap' in name_lower:
                            return MACDGenerator(fast=fast, slow=slow, signal=signal, base_calculation=BaseCalculationType.RETURNS_VWAP)
                        else:
                            # Explicit PRICE_RETURNS variant
                            return MACDGenerator(fast=fast, slow=slow, signal=signal, base_calculation=BaseCalculationType.PRICE_RETURNS)
                    else:
                        # Default to RETURNS_VWAP (now standard in feature engineering)
                        return MACDGenerator(fast=fast, slow=slow, signal=signal, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('stoch'):
                period = self._extract_period_from_name(feature_name, 14)
                stoch_type = name_lower.split('_')[1] if len(name_lower.split('_')) > 1 else 'k'
                if 'price' in name_lower:
                    base_calc = BaseCalculationType.PRICE_LEVELS
                else:
                    # Default to RETURNS_VWAP for better signal quality
                    base_calc = BaseCalculationType.RETURNS_VWAP

                if stoch_type == 'k':
                    return StochasticGenerator(k_period=period, d_period=3, base_calculation=base_calc)
                elif stoch_type == 'd':
                    return StochasticGenerator(k_period=period, d_period=3, base_calculation=base_calc)

            elif name_lower.startswith('williams_r') or name_lower.startswith('williams%'):
                period = self._extract_period_from_name(feature_name, 14)
                if 'price' in name_lower:
                    return WilliamsRGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return WilliamsRGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return WilliamsRGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('momentum_osc') or name_lower.startswith('momentum_'):
                period = self._extract_period_from_name(feature_name, 10)
                if 'price' in name_lower:
                    return MomentumOscillatorGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return MomentumOscillatorGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return MomentumOscillatorGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('roc_'):
                period = self._extract_period_from_name(feature_name, 10)
                if 'price' in name_lower:
                    return RateOfChangeGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return RateOfChangeGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return RateOfChangeGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            # VOLATILITY INDICATORS - Default to RETURNS_VWAP, but support all base calculations
            elif name_lower.startswith('bb_'):
                period = self._extract_period_from_name(feature_name, 20)
                bb_type = name_lower.split('_')[1] if len(name_lower.split('_')) > 1 else 'middle'
                band_type = "middle"
                if bb_type == 'upper':
                    band_type = "upper"
                elif bb_type == 'lower':
                    band_type = "lower"

                if 'price' in name_lower:
                    return BollingerBandsGenerator(period=period, std_dev=2.0, base_calculation=BaseCalculationType.PRICE_LEVELS, band_type=band_type)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return BollingerBandsGenerator(period=period, std_dev=2.0, base_calculation=BaseCalculationType.PRICE_RETURNS, band_type=band_type)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return BollingerBandsGenerator(period=period, std_dev=2.0, base_calculation=BaseCalculationType.RETURNS_VWAP, band_type=band_type)

            elif name_lower.startswith('atr_'):
                period = self._extract_period_from_name(feature_name, 14)
                if 'price' in name_lower:
                    return ATRGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return ATRGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return ATRGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('volatility_bands'):
                period = self._extract_period_from_name(feature_name, 20)
                if 'price' in name_lower:
                    return VolatilityBandsGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return VolatilityBandsGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return VolatilityBandsGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('volatility_'):
                period = self._extract_period_from_name(feature_name, 20)
                if 'price' in name_lower:
                    return VolatilityFeatureGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return VolatilityFeatureGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return VolatilityFeatureGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('garch_'):
                # Parse GARCH parameters (p, q, h)
                params = self._extract_garch_params(feature_name)
                if params:
                    p, q, h = params
                    return GARCHFeatureGenerator(p=p, q=q, forecast_horizon=h)

            # TREND INDICATORS - Default to RETURNS_VWAP, but support all base calculations
            elif name_lower.startswith('sma_'):
                period = self._extract_period_from_name(feature_name, 20)
                if 'price' in name_lower:
                    return SMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return SMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return SMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('ema_'):
                period = self._extract_period_from_name(feature_name, 12)
                if 'price' in name_lower:
                    return EMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return EMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return EMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('wma_'):
                period = self._extract_period_from_name(feature_name, 20)
                if 'price' in name_lower:
                    return WMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return WMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return WMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('dema_'):
                period = self._extract_period_from_name(feature_name, 21)
                if 'price' in name_lower:
                    return DEMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return DEMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return DEMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('tema_'):
                period = self._extract_period_from_name(feature_name, 21)
                if 'price' in name_lower:
                    return TEMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return TEMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return TEMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('trima_'):
                period = self._extract_period_from_name(feature_name, 21)
                if 'price' in name_lower:
                    return TRIMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return TRIMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return TRIMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('vwma_'):
                period = self._extract_period_from_name(feature_name, 20)
                # VWMA is inherently volume-weighted, so RETURNS_VWAP makes sense
                if 'price' in name_lower:
                    return VWMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                else:
                    return VWMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('keltner_'):
                period = self._extract_period_from_name(feature_name, 20)
                if 'price' in name_lower:
                    return KeltnerChannelsGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return KeltnerChannelsGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return KeltnerChannelsGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('adx_'):
                period = self._extract_period_from_name(feature_name, 14)
                if 'price' in name_lower:
                    return ADXGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return ADXGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return ADXGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('trend_score'):
                period = self._extract_period_from_name(feature_name, 14)
                from src.feature_generation.categories.trend import TrendScoreGenerator
                if 'price' in name_lower:
                    return TrendScoreGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return TrendScoreGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return TrendScoreGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            # OSCILLATOR INDICATORS - Default to RETURNS_VWAP, but support all base calculations
            elif name_lower.startswith('cci_'):
                period = self._extract_period_from_name(feature_name, 20)
                if 'price' in name_lower:
                    return CCIGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return CCIGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return CCIGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('aroon_'):
                period = self._extract_period_from_name(feature_name, 25)
                if 'price' in name_lower:
                    return AroonGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return AroonGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return AroonGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('ultimate_'):
                period = self._extract_period_from_name(feature_name, 14)
                if 'price' in name_lower:
                    return UltimateOscillatorGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return UltimateOscillatorGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return UltimateOscillatorGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('kst_'):
                period = self._extract_period_from_name(feature_name, 10)
                if 'price' in name_lower:
                    return KSTGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return KSTGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return KSTGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('apo_'):
                period = self._extract_period_from_name(feature_name, 12)
                if 'price' in name_lower:
                    return APOGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return APOGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return APOGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('cmo_'):
                period = self._extract_period_from_name(feature_name, 14)
                if 'price' in name_lower:
                    return CMOGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return CMOGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return CMOGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('natr_'):
                period = self._extract_period_from_name(feature_name, 14)
                if 'price' in name_lower:
                    return NATRGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return NATRGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return NATRGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('pfe_'):
                period = self._extract_period_from_name(feature_name, 12)
                if 'price' in name_lower:
                    return PFEGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return PFEGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return PFEGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('t3_'):
                period = self._extract_period_from_name(feature_name, 14)
                if 'price' in name_lower:
                    return T3Generator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return T3Generator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return T3Generator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('kama_'):
                period = self._extract_period_from_name(feature_name, 30)
                if 'price' in name_lower:
                    return KAMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_LEVELS)
                elif 'returns' in name_lower:
                    # Explicit PRICE_RETURNS variant
                    return KAMAGenerator(period=period, base_calculation=BaseCalculationType.PRICE_RETURNS)
                else:
                    # Default to RETURNS_VWAP (now standard in feature engineering)
                    return KAMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            # VOLUME INDICATORS - Use VOLUME_RETURNS as default for volume-based indicators
            elif name_lower.startswith('volume_sma_'):
                period = self._extract_period_from_name(feature_name, 20)
                return VolumeSMAGenerator(period=period)

            elif name_lower.startswith('volume_ema_'):
                period = self._extract_period_from_name(feature_name, 20)
                return VolumeEMAGenerator(period=period)

            elif name_lower.startswith('volume_ratio_'):
                period = self._extract_period_from_name(feature_name, 10)
                return VolumeRatioGenerator(period=period)

            elif name_lower.startswith('volume_roc_'):
                period = self._extract_period_from_name(feature_name, 10)
                return VolumeROCGenerator(period=period)

            elif name_lower.startswith('volume_std_'):
                period = self._extract_period_from_name(feature_name, 20)
                return VolumeStdGenerator(period=period)

            elif name_lower.startswith('volume_percentile_'):
                period = self._extract_period_from_name(feature_name, 20)
                return VolumePercentileGenerator(period=period)

            elif name_lower.startswith('volume_trend_strength'):
                params = self._extract_dual_period_params(feature_name, 10, 30)
                if params:
                    short_period, long_period = params
                    return VolumeTrendStrengthGenerator(short_period=short_period, long_period=long_period)

            elif name_lower.startswith('volume_osc'):
                params = self._extract_dual_period_params(feature_name, 10, 20)
                if params:
                    short_period, long_period = params
                    return VolumeOscillatorGenerator(short_period=short_period, long_period=long_period)

            elif name_lower.startswith('volume_momentum_'):
                period = self._extract_period_from_name(feature_name, 10)
                return VolumeMomentumGenerator(period=period)

            elif name_lower.startswith('volume_vwap_'):
                period = self._extract_period_from_name(feature_name, 20)
                return VolumeVWAPGenerator(period=period)

            elif name_lower.startswith('volume_price_trend'):
                return VolumePriceTrendGenerator()

            elif name_lower.startswith('volume_acc_dist'):
                return VolumeAccumulationDistributionGenerator()

            # EXPLICIT VWAP-BASED VARIANTS (when 'vwap_' prefix is used)
            elif name_lower.startswith('vwap_rsi_'):
                period = self._extract_period_from_name(feature_name, 14)
                return RSIGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('vwap_macd_'):
                params = self._extract_macd_params(feature_name.replace('vwap_', ''))
                if params:
                    fast, slow, signal = params
                    return MACDGenerator(fast=fast, slow=slow, signal=signal, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('vwap_sma_'):
                period = self._extract_period_from_name(feature_name, 20)
                return SMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('vwap_ema_'):
                period = self._extract_period_from_name(feature_name, 12)
                return EMAGenerator(period=period, base_calculation=BaseCalculationType.RETURNS_VWAP)

            elif name_lower.startswith('vwap_bb_'):
                period = self._extract_period_from_name(feature_name, 20)
                bb_type = name_lower.split('_')[2] if len(name_lower.split('_')) > 2 else 'middle'
                band_type = "middle"
                if bb_type == 'upper':
                    band_type = "upper"
                elif bb_type == 'lower':
                    band_type = "lower"
                return BollingerBandsGenerator(period=period, std_dev=2.0, base_calculation=BaseCalculationType.RETURNS_VWAP, band_type=band_type)

            # SUPPORT/RESISTANCE INDICATORS
            elif name_lower.startswith('support_level_'):
                params = self._extract_dual_params(feature_name, 'level', 'window')
                if params:
                    level, window = params
                    return SupportLevelGenerator(level=level, window=window)

            elif name_lower.startswith('resistance_level_'):
                params = self._extract_dual_params(feature_name, 'level', 'window')
                if params:
                    level, window = params
                    return ResistanceLevelGenerator(level=level, window=window)

            elif name_lower.startswith('pivot_point_'):
                window = self._extract_period_from_name(feature_name, 20)
                return PivotPointGenerator(window=window)

            elif name_lower.startswith('fibonacci_'):
                params = self._extract_dual_params(feature_name, 'level', 'window')
                if params:
                    level, window = params
                    return FibonacciLevelGenerator(level=level, window=window)

            # RETURNS INDICATORS
            elif name_lower.startswith('return_'):
                period = self._extract_period_from_name(feature_name, 1)
                return_type = name_lower.split('_')[1] if len(name_lower.split('_')) > 1 else 'simple'
                if return_type == 'log':
                    return LogReturnsGenerator(period=period)
                elif return_type == 'cumulative':
                    return CumulativeReturnsGenerator(period=period)
                else:
                    return SimpleReturnsGenerator(period=period)

            # ENTROPY INDICATORS
            elif name_lower.startswith('price_entropy_'):
                window = self._extract_period_from_name(feature_name, 20)
                return PriceEntropyGenerator(window=window)

            elif name_lower.startswith('volume_entropy_'):
                window = self._extract_period_from_name(feature_name, 20)
                return VolumeEntropyGenerator(window=window)

            elif name_lower.startswith('return_entropy_'):
                window = self._extract_period_from_name(feature_name, 20)
                return ReturnEntropyGenerator(window=window)

            elif name_lower.startswith('price_entropy_ma_'):
                params = self._extract_dual_period_params(feature_name, 20, 5)
                if params:
                    window, ma_window = params
                    return PriceEntropyMAGenerator(window=window, ma_window=ma_window)

            elif name_lower.startswith('volume_entropy_ma_'):
                params = self._extract_dual_period_params(feature_name, 20, 5)
                if params:
                    window, ma_window = params
                    return VolumeEntropyMAGenerator(window=window, ma_window=ma_window)

            elif name_lower.startswith('return_entropy_ma_'):
                params = self._extract_dual_period_params(feature_name, 20, 5)
                if params:
                    window, ma_window = params
                    return ReturnEntropyMAGenerator(window=window, ma_window=ma_window)

            elif name_lower.startswith('high_low_entropy_'):
                window = self._extract_period_from_name(feature_name, 20)
                return HighLowEntropyGenerator(window=window)

            elif name_lower.startswith('volatility_entropy_'):
                params = self._extract_dual_period_params(feature_name, 20, 10)
                if params:
                    window, volatility_window = params
                    return VolatilityEntropyGenerator(window=window, volatility_window=volatility_window)

            elif name_lower.startswith('momentum_entropy_'):
                params = self._extract_dual_period_params(feature_name, 20, 5)
                if params:
                    window, momentum_period = params
                    return MomentumEntropyGenerator(window=window, momentum_period=momentum_period)

            elif name_lower.startswith('rsi_entropy_'):
                params = self._extract_dual_period_params(feature_name, 20, 14)
                if params:
                    window, rsi_period = params
                    return RSIEntropyGenerator(window=window, rsi_period=rsi_period)

            elif name_lower.startswith('macd_entropy_'):
                params = self._extract_macd_params(feature_name)
                if params:
                    window, fast, slow = 20, params[0], params[1]  # Use window as first param
                    return MACDEntropyGenerator(window=window, fast=fast, slow=slow)

            elif name_lower.startswith('bb_entropy_'):
                params = self._extract_dual_period_params(feature_name, 20, 20)
                if params:
                    window, bb_period = params
                    return BollingerBandsEntropyGenerator(window=window, bb_period=bb_period, bb_std=2.0)

            elif name_lower.startswith('cross_asset_entropy_'):
                params = self._extract_dual_period_params(feature_name, 20, 10)
                if params:
                    window, correlation_window = params
                    return CrossAssetEntropyGenerator(window=window, correlation_window=correlation_window)

            elif name_lower.startswith('regime_entropy_'):
                params = self._extract_dual_period_params(feature_name, 20, 50)
                if params:
                    window, regime_window = params
                    return RegimeEntropyGenerator(window=window, regime_window=regime_window)

            # ACCELERATION INDICATORS
            elif name_lower.startswith('momentum_acc_'):
                period = self._extract_period_from_name(feature_name, 10)
                return MomentumGenerator(period=period)

            elif name_lower.startswith('price_acceleration_'):
                period = self._extract_period_from_name(feature_name, 5)
                return PriceAccelerationGenerator(period=period)

            elif name_lower.startswith('price_jerk_'):
                period = self._extract_period_from_name(feature_name, 5)
                return PriceJerkGenerator(period=period)

            elif name_lower.startswith('trend_strength_'):
                window = self._extract_period_from_name(feature_name, 10)
                return TrendStrengthGenerator(window=window)

            elif name_lower.startswith('trend_consistency_'):
                window = self._extract_period_from_name(feature_name, 10)
                return TrendConsistencyGenerator(window=window)

            elif name_lower.startswith('volume_acceleration_'):
                period = self._extract_period_from_name(feature_name, 5)
                return VolumeAccelerationGenerator(period=period)

            elif name_lower.startswith('volatility_acceleration_'):
                params = self._extract_dual_period_params(feature_name, 5, 20)
                if params:
                    period, volatility_window = params
                    return VolatilityAccelerationGenerator(period=period, volatility_window=volatility_window)

            # INTERACTION INDICATORS
            elif name_lower.startswith('cross_timeframe_interaction_'):
                params = self._extract_dual_period_params(feature_name, 5, 20)
                if params:
                    short_period, long_period = params
                    interaction_type = name_lower.split('_')[-1] if len(name_lower.split('_')) > 3 else 'ratio'
                    return CrossTimeframeInteractionGenerator(short_period=short_period, long_period=long_period, interaction_type=interaction_type)

            elif name_lower.startswith('feature_ratio_'):
                # Extract column names from feature name
                columns = self._extract_column_params(feature_name)
                if columns:
                    numerator, denominator = columns
                    return FeatureRatioGenerator(numerator_column=numerator, denominator_column=denominator)

            elif name_lower.startswith('polynomial_'):
                # Extract column and degree
                parts = name_lower.replace('polynomial_', '').split('_deg_')
                if len(parts) == 2:
                    column, degree = parts
                    return PolynomialFeatureGenerator(column=column, degree=int(degree))

            elif name_lower.startswith('correlation_interaction_'):
                columns = self._extract_column_params(feature_name)
                if columns:
                    col1, col2 = columns
                    window = self._extract_period_from_name(feature_name, 20)
                    return CorrelationInteractionGenerator(column1=col1, column2=col2, window=window)

            # CROSS-TIMEFRAME INDICATORS
            elif name_lower.startswith('cross_tf_momentum_'):
                params = self._extract_dual_period_params(feature_name, 5, 20)
                if params:
                    short_period, long_period = params
                    return CrossTimeframeMomentumGenerator(short_period=short_period, long_period=long_period)

            elif name_lower.startswith('cross_tf_volatility_'):
                params = self._extract_dual_period_params(feature_name, 5, 20)
                if params:
                    short_period, long_period = params
                    return CrossTimeframeVolatilityGenerator(short_period=short_period, long_period=long_period)

            elif name_lower.startswith('cross_tf_volume_'):
                params = self._extract_dual_period_params(feature_name, 5, 20)
                if params:
                    short_period, long_period = params
                    return CrossTimeframeVolumeGenerator(short_period=short_period, long_period=long_period)

            elif name_lower.startswith('cross_tf_trend_'):
                params = self._extract_dual_period_params(feature_name, 5, 20)
                if params:
                    short_period, long_period = params
                    return CrossTimeframeTrendGenerator(short_period=short_period, long_period=long_period)

            elif name_lower.startswith('cross_tf_high_low_'):
                params = self._extract_dual_period_params(feature_name, 5, 20)
                if params:
                    short_period, long_period = params
                    return CrossTimeframeHighLowGenerator(short_period=short_period, long_period=long_period)

            elif name_lower.startswith('cross_tf_ratio_'):
                params = self._extract_dual_period_params(feature_name, 5, 20)
                if params:
                    short_period, long_period = params
                    return CrossTimeframeRatioGenerator(short_period=short_period, long_period=long_period)

            elif name_lower.startswith('cross_tf_correlation_'):
                params = self._extract_dual_period_params(feature_name, 5, 20)
                if params:
                    short_period, long_period = params
                    return CrossTimeframeCorrelationGenerator(short_period=short_period, long_period=long_period)

            elif name_lower.startswith('cross_tf_divergence_'):
                params = self._extract_dual_period_params(feature_name, 5, 20)
                if params:
                    short_period, long_period = params
                    return CrossTimeframeDivergenceGenerator(short_period=short_period, long_period=long_period)

            # MICROSTRUCTURE INDICATORS
            elif name_lower.startswith('bid_ask_spread_'):
                window = self._extract_period_from_name(feature_name, 10)
                return BidAskSpreadGenerator(window=window)

            elif name_lower.startswith('order_flow_imbalance_'):
                window = self._extract_period_from_name(feature_name, 10)
                return OrderFlowImbalanceGenerator(window=window)

            elif name_lower.startswith('trade_size_imbalance_'):
                window = self._extract_period_from_name(feature_name, 10)
                return TradeSizeImbalanceGenerator(window=window)

            elif name_lower.startswith('price_impact_'):
                window = self._extract_period_from_name(feature_name, 10)
                return PriceImpactGenerator(window=window)

            elif name_lower.startswith('volume_weighted_price_'):
                window = self._extract_period_from_name(feature_name, 10)
                return VolumeWeightedPriceGenerator(window=window)

            elif name_lower.startswith('trade_intensity_'):
                window = self._extract_period_from_name(feature_name, 10)
                return TradeIntensityGenerator(window=window)

            elif name_lower.startswith('liquidity_proxy_'):
                window = self._extract_period_from_name(feature_name, 10)
                return LiquidityProxyGenerator(window=window)

            elif name_lower.startswith('market_depth_'):
                window = self._extract_period_from_name(feature_name, 10)
                return MarketDepthGenerator(window=window)

            # ORDER FLOW INDICATORS
            elif name_lower.startswith('taker_buy_ratio_'):
                window = self._extract_period_from_name(feature_name, 20)
                return TakerBuyRatioGenerator(window=window)

            elif name_lower.startswith('taker_sell_ratio_'):
                window = self._extract_period_from_name(feature_name, 20)
                return TakerSellRatioGenerator(window=window)

            elif name_lower.startswith('market_aggression_index_'):
                window = self._extract_period_from_name(feature_name, 20)
                return MarketAggressionIndexGenerator(window=window)

            elif name_lower.startswith('order_flow_imbalance_of_'):
                window = self._extract_period_from_name(feature_name, 20)
                return OrderFlowImbalanceGeneratorOF(window=window)

            # CANDLESTICK PATTERN INDICATORS (placeholder implementations)
            elif name_lower.startswith('candlestick_pattern_'):
                # Extract pattern type if specified
                pattern_type = name_lower.replace('candlestick_pattern_', '')
                return CandlestickPatternFeatureGenerator()  # Uses default config

            # ADVANCED SUPPORT/RESISTANCE FEATURES (calculated from historical SR data)
            elif name_lower.startswith('sr_persistence_'):
                window = self._extract_period_from_name(feature_name, 20)
                sr_type = name_lower.replace('sr_persistence_', '').replace('_' + str(window), '') if '_' + str(window) in name_lower else 'avg'
                return self._create_sr_persistence_generator(window, sr_type)

            elif name_lower.startswith('sr_touch_freq_'):
                window = self._extract_period_from_name(feature_name, 20)
                sr_type = name_lower.replace('sr_touch_freq_', '').replace('_' + str(window), '') if '_' + str(window) in name_lower else 'avg'
                return self._create_sr_touch_freq_generator(window, sr_type)

            elif name_lower.startswith('sr_bounce_rate_'):
                window = self._extract_period_from_name(feature_name, 20)
                sr_type = name_lower.replace('sr_bounce_rate_', '').replace('_' + str(window), '') if '_' + str(window) in name_lower else 'avg'
                return self._create_sr_bounce_rate_generator(window, sr_type)

            elif name_lower.startswith('sr_strength_trend_'):
                window = self._extract_period_from_name(feature_name, 20)
                sr_type = name_lower.replace('sr_strength_trend_', '').replace('_' + str(window), '') if '_' + str(window) in name_lower else 'avg'
                return self._create_sr_strength_trend_generator(window, sr_type)

            elif name_lower.startswith('ml_reliability_'):
                window = self._extract_period_from_name(feature_name, 20)
                return self._create_ml_reliability_generator(window)

            elif name_lower.startswith('ml_bounce_prob_'):
                window = self._extract_period_from_name(feature_name, 20)
                return self._create_ml_bounce_prob_generator(window)

            elif name_lower.startswith('trading_sr_reliability_'):
                window = self._extract_period_from_name(feature_name, 20)
                sr_type = name_lower.replace('trading_sr_reliability_', '').replace('_' + str(window), '') if '_' + str(window) in name_lower else 'support'
                return self._create_trading_sr_reliability_generator(window, sr_type)

            elif name_lower.startswith('volume_profile_hvn_'):
                window = self._extract_period_from_name(feature_name, 20)
                return self._create_volume_profile_hvn_generator(window)

            elif name_lower.startswith('volume_profile_poc_'):
                window = self._extract_period_from_name(feature_name, 20)
                return self._create_volume_profile_poc_generator(window)

            elif name_lower.startswith('volume_profile_vah_'):
                window = self._extract_period_from_name(feature_name, 20)
                return self._create_volume_profile_vah_generator(window)

            elif name_lower.startswith('volume_profile_val_'):
                window = self._extract_period_from_name(feature_name, 20)
                return self._create_volume_profile_val_generator(window)

            # Unknown feature type
            self.logger.debug(f"Unknown feature type: {feature_name}")
            return None

        except Exception as e:
            self.logger.error(f"Error creating feature generator for {feature_name}: {e}")
            return None

    def _extract_period_from_name(self, feature_name: str, default: int) -> int:
        """Extract period parameter from feature name."""
        try:
            tprint_debug(
                f"🧾 Extracting period from feature '{feature_name}' with default {default}"
            )
            # Split by underscore and look for numeric values
            parts = feature_name.split('_')
            for part in reversed(parts):
                if part.isdigit():
                    tprint_debug(f"   ↳ Found period {part}")
                    return int(part)
            tprint_warning(
                f"⚠️ No explicit period found for '{feature_name}', using default {default}"
            )
            return default
        except Exception:
            tprint_error(
                f"❌ Failed to parse period from '{feature_name}', using default {default}"
            )
            return default

    def _extract_macd_params(self, feature_name: str) -> Optional[Tuple[int, int, int]]:
        """Extract MACD parameters (fast, slow, signal) from feature name."""
        try:
            tprint_debug(f"🧾 Extracting MACD params from '{feature_name}'")
            # Expected format: macd_12_26_9, macd_returns_12_26_9, etc.
            parts = feature_name.lower().replace('macd', '').replace('returns', '').replace('vwap', '').split('_')
            numbers = [int(p) for p in parts if p.isdigit()]

            if len(numbers) >= 3:
                tprint_debug(f"   ↳ Parsed MACD parameters: {numbers[0]}, {numbers[1]}, {numbers[2]}")
                return (numbers[0], numbers[1], numbers[2])
            elif len(numbers) == 2:
                tprint_debug(f"   ↳ Parsed partial MACD params {numbers}, defaulting signal to 9")
                return (numbers[0], numbers[1], 9)  # Default signal period
            elif len(numbers) == 1:
                tprint_debug(f"   ↳ Parsed single MACD value {numbers[0]}, defaulting fast/slow to 12/26")
                return (12, 26, numbers[0])  # Default fast/slow, use number as signal

            tprint_warning(f"⚠️ No MACD parameters found in '{feature_name}'")
            return None
        except Exception as e:
            tprint_error(f"❌ Exception extracting MACD parameters from '{feature_name}': {e}")
            return None

    def _extract_garch_params(self, feature_name: str) -> Optional[Tuple[int, int, int]]:
        """Extract GARCH parameters (p, q, h) from feature name."""
        try:
            tprint_debug(f"🧾 Extracting GARCH params from '{feature_name}'")
            # Expected format: garch_1_1_1, garch_1_1_5, etc.
            parts = feature_name.lower().replace('garch', '').split('_')
            numbers = [int(p) for p in parts if p.isdigit()]

            if len(numbers) >= 3:
                tprint_debug(f"   ↳ Parsed GARCH parameters: {numbers[0]}, {numbers[1]}, {numbers[2]}")
                return (numbers[0], numbers[1], numbers[2])
            elif len(numbers) == 2:
                tprint_debug(f"   ↳ Parsed partial GARCH params {numbers}, defaulting horizon to 1")
                return (numbers[0], numbers[1], 1)  # Default horizon
            elif len(numbers) == 1:
                tprint_debug(f"   ↳ Parsed single GARCH value {numbers[0]}, defaulting p/q to 1")
                return (1, 1, numbers[0])  # Default p,q, use number as horizon

            tprint_warning(f"⚠️ No GARCH parameters found in '{feature_name}'")
            return None
        except Exception as e:
            tprint_error(f"❌ Exception extracting GARCH parameters from '{feature_name}': {e}")
            return None

    def _extract_dual_period_params(self, feature_name: str, default_short: int, default_long: int) -> Optional[Tuple[int, int]]:
        """Extract two period parameters from feature name."""
        tprint_debug("🧠 Entering _extract_dual_period_params")
        try:
            # Expected formats: volume_trend_strength_10_30, volume_osc_10_20, etc.
            parts = feature_name.lower().split('_')
            numbers = [int(p) for p in parts if p.isdigit()]

            if len(numbers) >= 2:
                return (numbers[0], numbers[1])
            elif len(numbers) == 1:
                return (numbers[0], default_long)

            return None
        except Exception:
            return None

    def _extract_dual_params(self, feature_name: str, param1_name: str, param2_name: str) -> Optional[Tuple[int, int]]:
        """Extract two parameters from feature name based on parameter names."""
        tprint_debug("🧠 Entering _extract_dual_params")
        try:
            # Expected formats:
            # - support_level_1_20, resistance_level_2_10, fibonacci_0.382_20
            # - correlation_interaction_col1_col2_window
            parts = feature_name.lower().split('_')

            # Find parameter positions
            param1_pos = -1
            param2_pos = -1

            for i, part in enumerate(parts):
                if param1_name in part:
                    param1_pos = i
                elif param2_name in part:
                    param2_pos = i

            if param1_pos >= 0 and param2_pos >= 0:
                # Extract numeric values after parameter names
                param1_parts = parts[param1_pos].split(param1_name)
                param2_parts = parts[param2_pos].split(param2_name)

                if len(param1_parts) > 1 and param1_parts[1].isdigit():
                    param1_val = int(param1_parts[1])
                elif param1_pos + 1 < len(parts) and parts[param1_pos + 1].isdigit():
                    param1_val = int(parts[param1_pos + 1])
                else:
                    return None

                if len(param2_parts) > 1 and param2_parts[1].isdigit():
                    param2_val = int(param2_parts[1])
                elif param2_pos + 1 < len(parts) and parts[param2_pos + 1].isdigit():
                    param2_val = int(parts[param2_pos + 1])
                else:
                    return None

                return (param1_val, param2_val)

            return None
        except Exception:
            return None

    def _extract_column_params(self, feature_name: str) -> Optional[Tuple[str, str]]:
        """Extract column names from feature name for interaction indicators."""
        tprint_debug("🧠 Entering _extract_column_params")
        try:
            # Expected format: feature_ratio_close_to_volume, correlation_interaction_close_volume_20
            parts = feature_name.lower().split('_')

            # For feature_ratio_close_to_volume format
            if 'feature_ratio' in feature_name.lower():
                ratio_part = feature_name.lower().replace('feature_ratio_', '')
                if '_to_' in ratio_part:
                    col1, col2 = ratio_part.split('_to_')
                    return (col1, col2)

            # For correlation_interaction_close_volume_20 format
            elif 'correlation_interaction' in feature_name.lower():
                interaction_part = feature_name.lower().replace('correlation_interaction_', '')
                if len(parts) >= 3:
                    col1 = parts[-3]  # Third to last
                    col2 = parts[-2]  # Second to last
                    return (col1, col2)

            return None
        except Exception:
            return None

    def _create_sr_persistence_generator(self, window: int, sr_type: str):
        """Create SR persistence feature generator."""
        tprint_debug("🧠 Entering _create_sr_persistence_generator")
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class SRPersistenceGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int, sr_type: str):
                    config = FeatureConfig(
                        name=f"sr_persistence_{sr_type}_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"SR level persistence analysis for {sr_type} levels over {window} periods",
                        required_columns=["close"],
                        default_lookback=window,
                        parameters={"window": window, "sr_type": sr_type}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    close_prices = data['close'].values
                    # This would typically load historical SR data and calculate persistence
                    # For now, return a placeholder based on price volatility
                    volatility = np.std(close_prices) / np.mean(close_prices) if len(close_prices) > 1 else 0.0
                    persistence_score = 1.0 / (1.0 + volatility)  # Higher volatility = lower persistence
                    return pd.Series([persistence_score] * len(data), index=data.index, name=self.config.name)

            return SRPersistenceGenerator(window, sr_type)

        except Exception as e:
            self.logger.error(f"Error creating SR persistence generator: {e}")
            return None

    def _create_sr_touch_freq_generator(self, window: int, sr_type: str):
        """Create SR touch frequency feature generator."""
        tprint_debug("🧠 Entering _create_sr_touch_freq_generator")
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class SRTouchFreqGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int, sr_type: str):
                    config = FeatureConfig(
                        name=f"sr_touch_freq_{sr_type}_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"SR level touch frequency for {sr_type} levels over {window} periods",
                        required_columns=["close", "high", "low"],
                        default_lookback=window,
                        parameters={"window": window, "sr_type": sr_type}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    # Calculate price movement frequency as a proxy for touch frequency
                    price_changes = np.abs(np.diff(data['close'].values))
                    avg_change = np.mean(price_changes) if len(price_changes) > 0 else 0.0
                    # Normalize to 0-1 range (higher frequency = more touches)
                    touch_freq = min(1.0, avg_change * 100)  # Scale factor
                    return pd.Series([touch_freq] * len(data), index=data.index, name=self.config.name)

            return SRTouchFreqGenerator(window, sr_type)

        except Exception as e:
            self.logger.error(f"Error creating SR touch frequency generator: {e}")
            return None

    def _create_sr_bounce_rate_generator(self, window: int, sr_type: str):
        """Create SR bounce rate feature generator."""
        tprint_debug("🧠 Entering _create_sr_bounce_rate_generator")
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class SRBounceRateGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int, sr_type: str):
                    config = FeatureConfig(
                        name=f"sr_bounce_rate_{sr_type}_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"SR level bounce success rate for {sr_type} levels over {window} periods",
                        required_columns=["close", "high", "low"],
                        default_lookback=window,
                        parameters={"window": window, "sr_type": sr_type}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    # Calculate bounce rate based on price reversals
                    close_prices = data['close'].values
                    if len(close_prices) < 3:
                        return pd.Series([0.5] * len(data), index=data.index, name=self.config.name)

                    # Simple bounce detection: when price changes direction after touching a level
                    price_changes = np.diff(close_prices)
                    reversals = np.sum(np.diff(np.sign(price_changes)) != 0) if len(price_changes) > 1 else 0
                    total_changes = len(price_changes)

                    bounce_rate = reversals / total_changes if total_changes > 0 else 0.5
                    return pd.Series([bounce_rate] * len(data), index=data.index, name=self.config.name)

            return SRBounceRateGenerator(window, sr_type)

        except Exception as e:
            self.logger.error(f"Error creating SR bounce rate generator: {e}")
            return None

    def _create_sr_strength_trend_generator(self, window: int, sr_type: str):
        """Create SR strength trend feature generator."""
        tprint_debug("🧠 Entering _create_sr_strength_trend_generator")
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class SRStrengthTrendGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int, sr_type: str):
                    config = FeatureConfig(
                        name=f"sr_strength_trend_{sr_type}_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"SR level strength trend for {sr_type} levels over {window} periods",
                        required_columns=["close"],
                        default_lookback=window,
                        parameters={"window": window, "sr_type": sr_type}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    # Calculate trend in price levels as a proxy for SR strength trend
                    close_prices = data['close'].values
                    if len(close_prices) < window:
                        return pd.Series([0.0] * len(data), index=data.index, name=self.config.name)

                    # Calculate rolling trend
                    trends = []
                    for i in range(window, len(close_prices)):
                        window_prices = close_prices[i-window:i]
                        # Simple linear trend slope
                        x = np.arange(len(window_prices))
                        slope = np.polyfit(x, window_prices, 1)[0]
                        trends.append(slope / np.mean(window_prices))  # Normalized slope

                    # Pad with zeros for the beginning
                    trends_padded = [0.0] * (window - 1) + trends
                    return pd.Series(trends_padded, index=data.index, name=self.config.name)

            return SRStrengthTrendGenerator(window, sr_type)

        except Exception as e:
            self.logger.error(f"Error creating SR strength trend generator: {e}")
            return None

    def _create_ml_reliability_generator(self, window: int):
        """Create ML reliability feature generator."""
        tprint_debug("🧠 Entering _create_ml_reliability_generator")
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class MLReliabilityGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int):
                    config = FeatureConfig(
                        name=f"ml_reliability_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"ML-ready reliability score over {window} periods",
                        required_columns=["close"],
                        default_lookback=window,
                        parameters={"window": window}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    close_prices = data['close'].values
                    # Reliability based on price stability (lower volatility = higher reliability)
                    if len(close_prices) < window:
                        return pd.Series([0.5] * len(data), index=data.index, name=self.config.name)

                    reliabilities = []
                    for i in range(window - 1, len(close_prices)):
                        window_prices = close_prices[i-window+1:i+1]
                        volatility = np.std(window_prices) / np.mean(window_prices)
                        reliability = 1.0 / (1.0 + volatility)
                        reliabilities.append(reliability)

                    # Pad with default reliability for the beginning
                    reliabilities_padded = [0.5] * (window - 1) + reliabilities
                    return pd.Series(reliabilities_padded, index=data.index, name=self.config.name)

            return MLReliabilityGenerator(window)

        except Exception as e:
            self.logger.error(f"Error creating ML reliability generator: {e}")
            return None

    def _create_ml_bounce_prob_generator(self, window: int):
        """Create ML bounce probability feature generator."""
        tprint_debug("🧠 Entering _create_ml_bounce_prob_generator")
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class MLBounceProbGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int):
                    config = FeatureConfig(
                        name=f"ml_bounce_prob_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"ML-ready bounce probability over {window} periods",
                        required_columns=["close", "high", "low"],
                        default_lookback=window,
                        parameters={"window": window}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    close_prices = data['close'].values
                    high_prices = data['high'].values
                    low_prices = data['low'].values

                    if len(close_prices) < window:
                        return pd.Series([0.5] * len(data), index=data.index, name=self.config.name)

                    bounce_probs = []
                    for i in range(window - 1, len(close_prices)):
                        # Calculate bounce probability based on price action patterns
                        window_close = close_prices[i-window+1:i+1]
                        window_high = high_prices[i-window+1:i+1]
                        window_low = low_prices[i-window+1:i+1]

                        # Simple bounce detection: when price reverses after hitting high/low
                        bounces = 0
                        for j in range(1, len(window_close)):
                            if (window_close[j] > window_high[j-1] and window_close[j] < window_close[j-1]) or \
                               (window_close[j] < window_low[j-1] and window_close[j] > window_close[j-1]):
                                bounces += 1

                        bounce_prob = bounces / (len(window_close) - 1) if len(window_close) > 1 else 0.5
                        bounce_probs.append(bounce_prob)

                    # Pad with default probability for the beginning
                    bounce_probs_padded = [0.5] * (window - 1) + bounce_probs
                    return pd.Series(bounce_probs_padded, index=data.index, name=self.config.name)

            return MLBounceProbGenerator(window)

        except Exception as e:
            self.logger.error(f"Error creating ML bounce probability generator: {e}")
            return None

    def _create_trading_sr_reliability_generator(self, window: int, sr_type: str):
        """Create trading SR reliability feature generator."""
        tprint_debug("🧠 Entering _create_trading_sr_reliability_generator")
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class TradingSRReliabilityGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int, sr_type: str):
                    config = FeatureConfig(
                        name=f"trading_sr_reliability_{sr_type}_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"Trading reliability for {sr_type} levels over {window} periods",
                        required_columns=["close"],
                        default_lookback=window,
                        parameters={"window": window, "sr_type": sr_type}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    close_prices = data['close'].values
                    # Trading reliability based on price stability and trend
                    if len(close_prices) < window:
                        return pd.Series([0.5] * len(data), index=data.index, name=self.config.name)

                    reliabilities = []
                    for i in range(window - 1, len(close_prices)):
                        window_prices = close_prices[i-window+1:i+1]
                        # Calculate reliability based on consistency and trend strength
                        volatility = np.std(window_prices) / np.mean(window_prices)
                        trend_strength = abs(np.polyfit(np.arange(len(window_prices)), window_prices, 1)[0]) / np.mean(window_prices)

                        # Higher trend strength and lower volatility = higher reliability
                        reliability = (1.0 - volatility) * (1.0 + trend_strength) / 2.0
                        reliabilities.append(reliability)

                    # Pad with default reliability for the beginning
                    reliabilities_padded = [0.5] * (window - 1) + reliabilities
                    return pd.Series(reliabilities_padded, index=data.index, name=self.config.name)

            return TradingSRReliabilityGenerator(window, sr_type)

        except Exception as e:
            self.logger.error(f"Error creating trading SR reliability generator: {e}")
            return None

    def _create_volume_profile_hvn_generator(self, window: int):
        """Create volume profile HVN (High Volume Node) feature generator."""
        tprint_debug("🧠 Entering _create_volume_profile_hvn_generator")
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class VolumeProfileHVNGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int):
                    config = FeatureConfig(
                        name=f"volume_profile_hvn_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"Volume profile High Volume Nodes over {window} periods",
                        required_columns=["close", "volume"],
                        default_lookback=window,
                        parameters={"window": window}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    close_prices = data['close'].values
                    volumes = data['volume'].values

                    if len(close_prices) < window:
                        return pd.Series([0.0] * len(data), index=data.index, name=self.config.name)

                    hvn_levels = []
                    for i in range(window - 1, len(close_prices)):
                        # Calculate volume-weighted price levels for HVN detection
                        window_close = close_prices[i-window+1:i+1]
                        window_volume = volumes[i-window+1:i+1]

                        # Find price levels with highest volume concentration
                        price_volume = {}
                        for j, (price, vol) in enumerate(zip(window_close, window_volume)):
                            price_level = round(price, 2)  # Round to 2 decimal places
                            if price_level not in price_volume:
                                price_volume[price_level] = 0
                            price_volume[price_level] += vol

                        if price_volume:
                            # HVN is the price level with highest volume
                            hvn_price = max(price_volume, key=price_volume.get)
                            hvn_levels.append(hvn_price)
                        else:
                            hvn_levels.append(close_prices[i])

                    # Pad with current price for the beginning
                    hvn_levels_padded = [close_prices[window-1]] * (window - 1) + hvn_levels
                    return pd.Series(hvn_levels_padded, index=data.index, name=self.config.name)

            return VolumeProfileHVNGenerator(window)

        except Exception as e:
            self.logger.error(f"Error creating volume profile HVN generator: {e}")
            return None

    def _create_volume_profile_poc_generator(self, window: int):
        """Create volume profile POC (Point of Control) feature generator."""
        tprint_debug("🧠 Entering _create_volume_profile_poc_generator")
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class VolumeProfilePOCGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int):
                    config = FeatureConfig(
                        name=f"volume_profile_poc_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"Volume profile Point of Control over {window} periods",
                        required_columns=["close", "volume"],
                        default_lookback=window,
                        parameters={"window": window}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    # POC is essentially the same as HVN for this simple implementation
                    return self._create_volume_profile_hvn_generator(window)._generate_feature(data, **kwargs)

            return VolumeProfilePOCGenerator(window)

        except Exception as e:
            self.logger.error(f"Error creating volume profile POC generator: {e}")
            return None

    def _create_volume_profile_vah_generator(self, window: int):
        """Create volume profile VAH (Value Area High) feature generator."""
        tprint_debug("🧠 Entering _create_volume_profile_vah_generator")
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class VolumeProfileVAHGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int):
                    config = FeatureConfig(
                        name=f"volume_profile_vah_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"Volume profile Value Area High over {window} periods",
                        required_columns=["close", "volume"],
                        default_lookback=window,
                        parameters={"window": window}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    close_prices = data['close'].values
                    volumes = data['volume'].values

                    if len(close_prices) < window:
                        return pd.Series([0.0] * len(data), index=data.index, name=self.config.name)

                    vah_levels = []
                    for i in range(window - 1, len(close_prices)):
                        window_close = close_prices[i-window+1:i+1]
                        window_volume = volumes[i-window+1:i+1]

                        # Calculate volume-weighted price distribution
                        price_volume = {}
                        for price, vol in zip(window_close, window_volume):
                            price_level = round(price, 2)
                            if price_level not in price_volume:
                                price_volume[price_level] = 0
                            price_volume[price_level] += vol

                        if price_volume:
                            # Sort by volume and find the upper boundary of high volume area
                            sorted_prices = sorted(price_volume.items(), key=lambda x: x[1], reverse=True)
                            total_volume = sum(price_volume.values())
                            cumulative_volume = 0
                            vah_price = sorted_prices[0][0]  # Start with highest volume price

                            for price_level, volume in sorted_prices:
                                cumulative_volume += volume
                                if cumulative_volume / total_volume >= 0.7:  # 70% of volume
                                    vah_price = price_level
                                    break

                            vah_levels.append(vah_price)
                        else:
                            vah_levels.append(close_prices[i])

                    # Pad with current price for the beginning
                    vah_levels_padded = [close_prices[window-1]] * (window - 1) + vah_levels
                    return pd.Series(vah_levels_padded, index=data.index, name=self.config.name)

            return VolumeProfileVAHGenerator(window)

        except Exception as e:
            self.logger.error(f"Error creating volume profile VAH generator: {e}")
            return None

    def _create_volume_profile_val_generator(self, window: int):
        """Create volume profile VAL (Value Area Low) feature generator."""
        tprint_debug("🧠 Entering _create_volume_profile_val_generator")
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class VolumeProfileVALGenerator(VectorizedFeatureGenerator):
                def __init__(self, window: int):
                    config = FeatureConfig(
                        name=f"volume_profile_val_{window}",
                        category=FeatureCategory.SUPPORT_RESISTANCE,
                        description=f"Volume profile Value Area Low over {window} periods",
                        required_columns=["close", "volume"],
                        default_lookback=window,
                        parameters={"window": window}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    close_prices = data['close'].values
                    volumes = data['volume'].values

                    if len(close_prices) < window:
                        return pd.Series([0.0] * len(data), index=data.index, name=self.config.name)

                    val_levels = []
                    for i in range(window - 1, len(close_prices)):
                        window_close = close_prices[i-window+1:i+1]
                        window_volume = volumes[i-window+1:i+1]

                        # Calculate volume-weighted price distribution
                        price_volume = {}
                        for price, vol in zip(window_close, window_volume):
                            price_level = round(price, 2)
                            if price_level not in price_volume:
                                price_volume[price_level] = 0
                            price_volume[price_level] += vol

                        if price_volume:
                            # Sort by volume and find the lower boundary of high volume area
                            sorted_prices = sorted(price_volume.items(), key=lambda x: x[1], reverse=True)
                            total_volume = sum(price_volume.values())
                            cumulative_volume = 0
                            val_price = sorted_prices[-1][0]  # Start with lowest volume price

                            for price_level, volume in sorted_prices:
                                cumulative_volume += volume
                                if cumulative_volume / total_volume >= 0.7:  # 70% of volume
                                    val_price = price_level
                                    break

                            val_levels.append(val_price)
                        else:
                            val_levels.append(close_prices[i])

                    # Pad with current price for the beginning
                    val_levels_padded = [close_prices[window-1]] * (window - 1) + val_levels
                    return pd.Series(val_levels_padded, index=data.index, name=self.config.name)

            return VolumeProfileVALGenerator(window)

        except Exception as e:
            self.logger.error(f"Error creating volume profile VAL generator: {e}")
            return None

    def _create_volatility_generator(self, period: int):
        """Create a custom volatility generator for volatility features."""
        tprint_debug("🧠 Entering _create_volatility_generator")
        try:
            from src.feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

            class VolatilityGenerator(VectorizedFeatureGenerator):
                def __init__(self, period: int):
                    config = FeatureConfig(
                        name=f"volatility_{period}",
                        category=FeatureCategory.VOLATILITY,
                        description=f"Volatility indicator with {period} period lookback",
                        required_columns=["close"],
                        default_lookback=period,
                        parameters={"period": period}
                    )
                    super().__init__(config)

                def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                    close_prices = data['close'].values
                    if len(close_prices) < self.config.parameters["period"]:
                        return pd.Series(np.full(len(close_prices), np.nan), index=data.index, name=self.config.name)

                    # Calculate rolling standard deviation of returns
                    returns = np.diff(close_prices) / close_prices[:-1]
                    volatility = pd.Series(returns).rolling(window=self.config.parameters["period"]).std().values

                    # Pad the first value to match length
                    volatility = np.concatenate([[np.nan], volatility])

                    return pd.Series(volatility, index=data.index, name=self.config.name)

            return VolatilityGenerator(period)

        except Exception as e:
            self.logger.error(f"Error creating volatility generator: {e}")
            return None

    def _create_failed_result(self, method: str, optimization_time: float) -> OptimizationResult:
        """Create a failed optimization result."""
        tprint_debug("🧠 Entering _create_failed_result")
        return OptimizationResult(
            best_lookback_period=OPTIMIZATION_CONSTANTS.DEFAULT_MIN_LOOKBACK,  # Already an int
            best_score=0.0,
            optimization_method=method,
            total_trials=0,
            optimization_time=optimization_time,
            convergence_achieved=False,
            metadata={'error': 'Optimization failed'}
        )

    def _calculate_comprehensive_correlations(self, feature_values: np.ndarray, target_values: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive correlation metrics."""
        tprint_debug("🧠 Entering _calculate_comprehensive_correlations")
        try:
            correlations = {}
            
            # Pearson correlation
            correlations['pearson'] = safe_correlation(feature_values, target_values, default=0.0)
            
            # Spearman correlation (rank-based)
            try:
                from scipy.stats import spearmanr
                spearman_corr, _ = spearmanr(feature_values, target_values)
                correlations['spearman'] = spearman_corr if not np.isnan(spearman_corr) else 0.0
            except ImportError:
                # Fallback to simple rank correlation
                correlations['spearman'] = safe_correlation(
                    np.argsort(feature_values), np.argsort(target_values), default=0.0
                )
            
            # Mutual information (simplified)
            correlations['mutual_info'] = self._calculate_mutual_information(feature_values, target_values)
            
            # R-squared
            correlations['r_squared'] = correlations['pearson'] ** 2
            
            return correlations
            
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating correlations: {e}')
            return {'pearson': 0.0, 'spearman': 0.0, 'mutual_info': 0.0, 'r_squared': 0.0}

    def _calculate_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate simplified mutual information."""
        tprint_debug("🧠 Entering _calculate_mutual_information")
        try:
            # Simple binning approach
            n_bins = min(10, len(x) // 10)
            if n_bins < 2:
                return 0.0
                
            # Create bins
            x_bins = np.digitize(x, np.linspace(x.min(), x.max(), n_bins))
            y_bins = np.digitize(y, np.linspace(y.min(), y.max(), n_bins))
            
            # Calculate joint and marginal probabilities
            joint_counts = np.zeros((n_bins, n_bins))
            for i in range(len(x_bins)):
                if x_bins[i] < n_bins and y_bins[i] < n_bins:
                    joint_counts[x_bins[i]-1, y_bins[i]-1] += 1
            
            joint_probs = joint_counts / joint_counts.sum()
            x_probs = joint_probs.sum(axis=1)
            y_probs = joint_probs.sum(axis=0)
            
            # Calculate mutual information
            mi = 0.0
            for i in range(n_bins):
                for j in range(n_bins):
                    if joint_probs[i, j] > 0 and x_probs[i] > 0 and y_probs[j] > 0:
                        mi += joint_probs[i, j] * np.log2(joint_probs[i, j] / (x_probs[i] * y_probs[j]))
            
            return max(0.0, mi)
            
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating mutual information: {e}')
            return 0.0

    def _calculate_composite_score(self, correlations: Dict[str, float]) -> float:
        """Calculate composite score from multiple correlation metrics."""
        tprint_debug("🧠 Entering _calculate_composite_score")
        try:
            # Weighted combination of different correlation measures
            weights = {
                'pearson': 0.4,
                'spearman': 0.3,
                'mutual_info': 0.2,
                'r_squared': 0.1
            }
            
            composite_score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in correlations:
                    value = abs(correlations[metric])  # Use absolute value
                    composite_score += value * weight
                    total_weight += weight
            
            return composite_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating composite score: {e}')
            return 0.0

    def _calculate_multi_target_metrics(self, feature_values: np.ndarray, target_values: np.ndarray) -> Dict[str, float]:
        """Calculate multiple target metrics for multi-objective optimization."""
        tprint_debug("🧠 Entering _calculate_multi_target_metrics")
        try:
            metrics = {}
            
            # Correlation metrics
            metrics['correlation'] = safe_correlation(feature_values, target_values, default=0.0)
            metrics['r_squared'] = metrics['correlation'] ** 2
            
            # Stability metrics
            metrics['stability'] = self._calculate_stability_metric(feature_values)
            
            # Information content
            metrics['information_content'] = self._calculate_information_content(feature_values)
            
            # Predictive power
            metrics['predictive_power'] = self._calculate_predictive_power(feature_values, target_values)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating multi-target metrics: {e}')
            return {'correlation': 0.0, 'r_squared': 0.0, 'stability': 0.0, 'information_content': 0.0, 'predictive_power': 0.0}

    def _calculate_stability_metric(self, values: np.ndarray) -> float:
        """Calculate stability metric (lower variance = higher stability)."""
        tprint_debug("🧠 Entering _calculate_stability_metric")
        try:
            if len(values) < 2:
                return 0.0
            return 1.0 / (1.0 + np.var(values))  # Higher stability = lower variance
        except Exception:
            return 0.0

    def _calculate_information_content(self, values: np.ndarray) -> float:
        """Calculate information content using entropy."""
        tprint_debug("🧠 Entering _calculate_information_content")
        try:
            if len(values) < 2:
                return 0.0
            
            # Simple entropy calculation
            hist, _ = np.histogram(values, bins=min(10, len(values)//2))
            probs = hist / hist.sum()
            probs = probs[probs > 0]  # Remove zero probabilities
            
            entropy = -np.sum(probs * np.log2(probs))
            return entropy / np.log2(len(probs)) if len(probs) > 1 else 0.0
            
        except Exception:
            return 0.0

    def _calculate_predictive_power(self, feature_values: np.ndarray, target_values: np.ndarray) -> float:
        """Calculate predictive power using cross-validation-like approach."""
        tprint_debug("🧠 Entering _calculate_predictive_power")
        try:
            if len(feature_values) < 10:
                return 0.0
            
            # Simple predictive power: correlation with lagged target
            lag = min(5, len(feature_values) // 4)
            if lag > 0:
                lagged_target = target_values[lag:]
                lagged_feature = feature_values[:-lag]
                return abs(safe_correlation(lagged_feature, lagged_target, default=0.0))
            else:
                return abs(safe_correlation(feature_values, target_values, default=0.0))
                
        except Exception:
            return 0.0

    def _calculate_multi_objective_score(self, targets: Dict[str, float]) -> float:
        """Calculate multi-objective score using weighted combination."""
        tprint_debug("🧠 Entering _calculate_multi_objective_score")
        try:
            # Default weights for different objectives
            weights = {
                'correlation': 0.3,
                'r_squared': 0.2,
                'stability': 0.2,
                'information_content': 0.15,
                'predictive_power': 0.15
            }
            
            score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                if metric in targets:
                    value = abs(targets[metric])  # Use absolute value
                    score += value * weight
                    total_weight += weight
            
            return score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating multi-objective score: {e}')
            return 0.0

    def _check_convergence(self, scores: List[float]) -> bool:
        """Check if optimization has converged."""
        tprint_debug("🧠 Entering _check_convergence")
        try:
            if len(scores) < 5:
                return False
            
            # Check if the last few scores are stable
            recent_scores = scores[-5:]
            score_std = np.std(recent_scores)
            score_mean = np.mean(recent_scores)
            
            # Converged if coefficient of variation is small
            cv = score_std / (score_mean + 1e-8)
            return cv < 0.05  # 5% coefficient of variation threshold
            
        except Exception:
            return False

    def _update_performance_metrics(self, result: OptimizationResult, optimization_time: float) -> None:
        """Update performance tracking metrics."""
        tprint_debug("🧠 Entering _update_performance_metrics")
        try:
            self.performance_metrics['total_optimizations'] += 1
            
            if result.best_score > 0:
                self.performance_metrics['successful_optimizations'] += 1
                self.performance_metrics['best_scores'].append(result.best_score)
                
                # Keep only recent scores for memory efficiency
                if len(self.performance_metrics['best_scores']) > 100:
                    self.performance_metrics['best_scores'] = self.performance_metrics['best_scores'][-100:]
            
            # Update average optimization time
            total_time = self.performance_metrics['average_optimization_time'] * (self.performance_metrics['total_optimizations'] - 1)
            self.performance_metrics['average_optimization_time'] = (total_time + optimization_time) / self.performance_metrics['total_optimizations']
            
            # Store in history
            self.optimization_history.append({
                'timestamp': time.time(),
                'method': result.optimization_method,
                'best_score': result.best_score,
                'optimization_time': optimization_time,
                'convergence_achieved': result.convergence_achieved
            })
            
            # Keep only recent history
            if len(self.optimization_history) > 1000:
                self.optimization_history = self.optimization_history[-1000:]
                
        except Exception as e:
            self.logger.warning(f'⚠️ Error updating performance metrics: {e}')

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the optimizer."""
        tprint_debug("🧠 Entering get_performance_summary")
        try:
            if not self.performance_metrics['best_scores']:
                return {
                    'total_optimizations': self.performance_metrics['total_optimizations'],
                    'successful_optimizations': self.performance_metrics['successful_optimizations'],
                    'success_rate': 0.0,
                    'average_optimization_time': self.performance_metrics['average_optimization_time'],
                    'best_score_ever': 0.0,
                    'average_best_score': 0.0
                }
            
            return {
                'total_optimizations': self.performance_metrics['total_optimizations'],
                'successful_optimizations': self.performance_metrics['successful_optimizations'],
                'success_rate': self.performance_metrics['successful_optimizations'] / self.performance_metrics['total_optimizations'],
                'average_optimization_time': self.performance_metrics['average_optimization_time'],
                'best_score_ever': max(self.performance_metrics['best_scores']),
                'average_best_score': np.mean(self.performance_metrics['best_scores']),
                'score_std': np.std(self.performance_metrics['best_scores'])
            }
            
        except Exception as e:
            self.logger.warning(f'⚠️ Error getting performance summary: {e}')
            return {}

    def save_optimization_results(self, filepath: str) -> bool:
        """Save optimization results to file."""
        tprint_debug("🧠 Entering save_optimization_results")
        try:
            results = {
                'performance_metrics': self.performance_metrics,
                'optimization_history': self.optimization_history[-100:],  # Last 100 optimizations
                'timestamp': datetime.now().isoformat()
            }
            
            self.serializer.save_json(results, filepath)
            self.logger.info(f'💾 Optimization results saved to {filepath}')
            return True
            
        except Exception as e:
            self.logger.error(f'❌ Error saving optimization results: {e}')
            return False

    def test_feature_engineering(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Test method to verify comprehensive feature engineering implementation.

        Args:
            data: Test data with OHLCV columns

        Returns:
            Dictionary of feature names to their calculated values
        """
        tprint_debug("🧠 Entering test_feature_engineering")
        test_features = [
            # RETURNS_VWAP-BASED indicators (NEW STANDARD - better signal quality)
            'rsi_14', 'macd_12_26_9', 'sma_20', 'ema_12', 'bb_upper_20',
            'stoch_k_14', 'williams_r_14', 'cci_20', 'adx_14',

            # PRICE_RETURNS variants (traditional price returns - for comparison)
            'returns_rsi_14', 'returns_macd_12_26_9', 'returns_sma_20', 'returns_ema_12', 'returns_bb_upper_20',
            'returns_stoch_k_14', 'returns_williams_r_14', 'returns_cci_20',

            # PRICE_LEVELS variants (raw price levels - for comparison)
            'price_rsi_14', 'price_sma_20', 'price_bb_upper_20',

            # VWAP-BASED variants (explicit VWAP calculation)
            'vwap_rsi_14', 'vwap_macd_12_26_9', 'vwap_sma_20', 'vwap_ema_12', 'vwap_bb_upper_20',

            # Advanced SR features (using actual price levels for S/R detection)
            'support_level_1_20', 'resistance_level_2_20',
            'pivot_point_20', 'fibonacci_0.382_20',
            'sr_persistence_avg_20', 'sr_persistence_avg_200', 'volume_profile_hvn_20',

            # Volume indicators (volume returns by default)
            'volume_sma_20', 'volume_ratio_10',

            # Order flow indicators
            'taker_buy_ratio_20', 'market_aggression_index_20',

            # Returns indicators (core price returns)
            'return_1', 'return_log_1', 'return_cumulative_1',

            # Entropy indicators (advanced analysis)
            'price_entropy_20', 'rsi_entropy_20_14',

            # Acceleration indicators
            'trend_strength_10', 'momentum_acc_10'
        ]

        results = {}

        for feature_name in test_features:
            try:
                feature_values = self._calculate_feature_for_lookback(data, feature_name, 20)
                results[feature_name] = feature_values
                if feature_values is not None:
                    self.logger.info(f"✅ Successfully calculated {feature_name}: shape={feature_values.shape}")
                else:
                    self.logger.warning(f"⚠️ {feature_name} returned None")
            except Exception as e:
                self.logger.error(f"❌ Failed to calculate {feature_name}: {e}")
                results[feature_name] = None

        return results

    def _get_data_hash(self, data: pd.DataFrame, feature_name: str, horizon: int) -> str:
        """Generate hash for data caching."""
        tprint_debug("🧠 Entering _get_data_hash")
        try:
            # Create a hash based on data shape, feature name, and horizon
            data_info = f"{data.shape}_{feature_name}_{horizon}_{data.index[-1] if len(data) > 0 else 0}"
            return hashlib.md5(data_info.encode()).hexdigest()[:16]
        except Exception:
            return f"{feature_name}_{horizon}"

    def _cached_feature_calculation(self, data: pd.DataFrame, feature_name: str, horizon: int) -> Optional[np.ndarray]:
        """Calculate feature with caching to avoid recomputation."""
        tprint_debug("🧠 Entering _cached_feature_calculation")
        cache_key = self._get_data_hash(data, feature_name, horizon)
        
        if cache_key in self.feature_cache:
            self.cache_hits += 1
            return self.feature_cache[cache_key]
        
        # Calculate feature
        feature_values = self._calculate_feature_for_lookback(data, feature_name, horizon)
        
        # Cache the result (limit cache size to prevent memory issues)
        if len(self.feature_cache) < 1000:  # Limit cache size
            self.feature_cache[cache_key] = feature_values
        
        self.cache_misses += 1
        return feature_values

    def _vectorized_mi_calculation(self, features_list: List[np.ndarray], returns_list: List[np.ndarray]) -> List[float]:
        """Calculate mutual information for multiple feature-return pairs using vectorized operations."""
        tprint_debug("🧠 Entering _vectorized_mi_calculation")
        try:
            # Use batch processing for multiple MI calculations with safe_correlation
            mi_scores = []
            for features, returns in zip(features_list, returns_list):
                # Align arrays
                min_length = min(len(features), len(returns))
                if min_length < 10:
                    mi_scores.append(0.0)
                    continue
                
                # Use safe_correlation from math_validation for robust calculation
                aligned_features = features[:min_length]
                aligned_returns = returns[:min_length]
                
                # Calculate correlation using safe_correlation
                correlation = safe_correlation(aligned_features, aligned_returns)
                
                # Convert correlation to MI approximation: MI ≈ 0.5 * log(1 - corr²)
                if abs(correlation) < 0.999:  # Avoid log(0)
                    try:
                        mi_approx = 0.5 * np.log(1 - correlation**2) if correlation**2 < 1 else 0.0
                        mi_scores.append(max(0.0, -mi_approx))  # Ensure positive
                    except (ValueError, OverflowError):
                        mi_scores.append(0.0)
                else:
                    mi_scores.append(0.0)
            
            return mi_scores
            
        except Exception as e:
            self.logger.warning(f"⚠️ Vectorized MI calculation failed, using fallback: {e}")
            # Fallback to individual calculations
            return [self._calculate_mutual_information_robust(f, r) for f, r in zip(features_list, returns_list)]

    def _extract_numeric_array(self, series: Union[pd.Series, np.ndarray, None]) -> Optional[np.ndarray]:
        """Convert a Series or array-like into a sanitized numpy array."""
        tprint_debug("🧠 Entering _extract_numeric_array")
        if series is None:
            return None

        try:
            if isinstance(series, pd.Series):
                numeric = pd.to_numeric(series, errors='coerce')
                values = numeric.to_numpy(dtype=float, copy=True)
            else:
                values = np.asarray(series, dtype=float)

            if values.size == 0:
                return None

            return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            return None

    def _combine_arrays(self, arrays: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
        """Combine multiple arrays by averaging while ignoring missing inputs."""
        tprint_debug("🧠 Entering _combine_arrays")
        valid_arrays = [arr for arr in arrays if arr is not None]
        if not valid_arrays:
            return None

        try:
            stacked = np.vstack(valid_arrays)
            combined = np.nanmean(stacked, axis=0)
            return np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            return valid_arrays[0]

    def _aggregate_probability_stream(self, data: pd.DataFrame, direction: str, horizon_keyword: str) -> Optional[np.ndarray]:
        """Aggregate probability columns for a given direction and horizon keyword."""
        tprint_debug("🧠 Entering _aggregate_probability_stream")
        pattern = f"_{horizon_keyword}_{direction}_prob"
        matching_cols = [col for col in data.columns if pattern in col]
        if not matching_cols:
            return None

        aggregated = [self._extract_numeric_array(data[col]) for col in matching_cols]
        aggregated = [arr for arr in aggregated if arr is not None]
        if not aggregated:
            return None

        return self._combine_arrays(aggregated)

    def _get_multi_horizon_boundaries(self) -> Tuple[int, int]:
        """Return cached immediate and short horizon boundaries derived from configuration."""
        tprint_debug("🧠 Entering _get_multi_horizon_boundaries")
        if self._cached_multi_horizon_limits is not None:
            return self._cached_multi_horizon_limits

        immediate_default, short_default = 2, 4
        locator = self._data_locator
        if locator is not None:
            config_path = locator.config_path('multi_horizon_labeling')
        else:
            default_locator = PipelineDataLocator()
            config_path = default_locator.config_path('multi_horizon_labeling')

        if config_path.exists():
            try:
                import yaml  # type: ignore

                with open(config_path, 'r') as cfg_file:
                    config_data = yaml.safe_load(cfg_file)

                mh_config = config_data.get('multi_horizon_labeling', {}) if isinstance(config_data, dict) else {}
                horizons_cfg = mh_config.get('time_horizons', {}) if isinstance(mh_config, dict) else {}

                immediate_default = int(horizons_cfg.get('immediate', immediate_default))
                short_default = int(horizons_cfg.get('short', short_default))
            except Exception:
                # Use defaults if configuration parsing fails
                pass

        immediate_limit = max(1, immediate_default)
        short_limit = max(immediate_limit, short_default)

        self._cached_multi_horizon_limits = (immediate_limit, short_limit)
        return self._cached_multi_horizon_limits

    def _build_horizon_weighted_matrix(
        self,
        immediate_arr: Optional[np.ndarray],
        short_arr: Optional[np.ndarray],
        overall_arr: Optional[np.ndarray],
        immediate_limit: int,
        short_limit: int,
        max_horizon: int
    ) -> Dict[int, np.ndarray]:
        """Create a dictionary that maps horizons to the most appropriate opportunity stream."""
        tprint_debug("🧠 Entering _build_horizon_weighted_matrix")
        horizon_map: Dict[int, np.ndarray] = {}

        for horizon in range(1, max_horizon + 1):
            selected: Optional[np.ndarray] = None

            if immediate_arr is not None and horizon <= immediate_limit:
                selected = immediate_arr
            elif short_arr is not None and horizon <= short_limit:
                selected = short_arr
            elif overall_arr is not None:
                selected = overall_arr
            elif immediate_arr is not None:
                selected = immediate_arr
            elif short_arr is not None:
                selected = short_arr

            if selected is not None:
                horizon_map[horizon] = selected

        return horizon_map

    def _get_shared_forward_returns_matrix(self, data: pd.DataFrame, target_column: str, max_horizon: int = 300) -> Dict[int, np.ndarray]:
        """
        Get or create shared forward returns matrix that can be reused across all features.

        Args:
            data: Input data with target column
            target_column: Target column for forward returns calculation
            max_horizon: Maximum horizon to compute forward returns

        Returns:
            Dictionary mapping horizon to forward returns array
        """
        tprint_debug("🧠 Entering _get_shared_forward_returns_matrix")
        data_hash = self._get_data_hash(data, f"shared_returns_{target_column}", max_horizon)

        if (
            self.shared_forward_returns_hash == data_hash and
            isinstance(self.shared_forward_returns, dict) and
            target_column in self.shared_forward_returns and
            max(self.shared_forward_returns[target_column].keys(), default=0) >= max_horizon
        ):
            self.logger.info(f"♻️ Reusing cached multi-horizon opportunity matrix for '{target_column}'")
            return self.shared_forward_returns[target_column]

        self.logger.info(
            f"🔄 Building multi-horizon opportunity matrices up to horizon {max_horizon} for target '{target_column}'"
        )

        matrices = self._precompute_forward_returns_matrix(data, target_column, max_horizon)
        self.shared_forward_returns = matrices
        self.shared_forward_returns_hash = data_hash

        if target_column in matrices:
            return matrices[target_column]

        # Fallback: create a direct matrix from the target column if available
        if target_column in data.columns:
            direct_array = self._extract_numeric_array(data[target_column])
            if direct_array is not None:
                fallback_matrix = {h: direct_array for h in range(1, max_horizon + 1)}
                self.shared_forward_returns[target_column] = fallback_matrix
                return fallback_matrix

        return {}

    def _create_multi_horizon_aligned_targets(
        self,
        data: pd.DataFrame,
        max_horizon: int = 300,
        allow_labeler: bool = True
    ) -> Dict[str, Any]:
        """Create aligned multi-horizon opportunity streams using available labeled data."""
        tprint_debug("🧠 Entering _create_multi_horizon_aligned_targets")
        def column_array(column_name: str) -> Optional[np.ndarray]:
            return self._extract_numeric_array(data[column_name]) if column_name in data.columns else None

        long_immediate = self._combine_arrays([
            self._aggregate_probability_stream(data, 'long', 'immediate'),
            column_array('long_immediate_opportunity')
        ])
        long_short_term = column_array('long_short_term_opportunity')
        long_short = self._combine_arrays([
            self._aggregate_probability_stream(data, 'long', 'short'),
            long_short_term
        ])
        long_overall = column_array('long_overall_opportunity')
        long_leverage = column_array('long_leverage_adjusted_score')

        short_immediate = self._combine_arrays([
            self._aggregate_probability_stream(data, 'short', 'immediate'),
            column_array('short_immediate_opportunity')
        ])
        short_short_term = column_array('short_short_term_opportunity')
        short_short = self._combine_arrays([
            self._aggregate_probability_stream(data, 'short', 'short'),
            short_short_term
        ])
        short_overall = column_array('short_overall_opportunity')
        short_leverage = column_array('short_leverage_adjusted_score')

        composite_immediate = self._combine_arrays([
            column_array('immediate_opportunity'),
            long_immediate,
            short_immediate
        ])
        composite_short = self._combine_arrays([
            column_array('short_term_opportunity'),
            long_short,
            short_short
        ])
        composite_overall = self._combine_arrays([
            column_array('overall_opportunity'),
            long_overall,
            short_overall
        ])
        composite_leverage = self._combine_arrays([
            column_array('leverage_adjusted_score'),
            long_leverage,
            short_leverage,
            composite_overall
        ])

        directional_confidence = column_array('directional_confidence')
        if directional_confidence is None and long_overall is not None and short_overall is not None:
            directional_confidence = np.nan_to_num(
                np.abs(long_overall - short_overall),
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

        opportunity_asymmetry = column_array('opportunity_asymmetry')
        if opportunity_asymmetry is None and long_overall is not None and short_overall is not None:
            opportunity_asymmetry = np.nan_to_num(
                long_overall - short_overall,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

        aligned_targets: Dict[str, Any] = {
            'long': {
                'immediate': long_immediate,
                'short': long_short,
                'overall': self._combine_arrays([long_overall, long_leverage]),
                'leverage': long_leverage
            },
            'short': {
                'immediate': short_immediate,
                'short': short_short,
                'overall': self._combine_arrays([short_overall, short_leverage]),
                'leverage': short_leverage
            },
            'composite': {
                'immediate': composite_immediate,
                'short': composite_short,
                'overall': composite_overall,
                'leverage': composite_leverage
            },
            'directional': {
                'confidence': directional_confidence,
                'asymmetry': opportunity_asymmetry
            }
        }

        has_any = any(
            isinstance(bucket, dict) and any(value is not None for value in bucket.values())
            for bucket in aligned_targets.values()
        )

        if has_any:
            return aligned_targets

        if allow_labeler and MULTI_HORIZON_AVAILABLE:
            try:
                config = MultiHorizonConfig()
                labeled_data = apply_multi_horizon_labeling(data, config)
                self.logger.info("🔁 Generated multi-horizon labels on the fly for forward-return alignment")
                return self._create_multi_horizon_aligned_targets(labeled_data, max_horizon, allow_labeler=False)
            except Exception as exc:
                self.logger.warning(f"⚠️ Dynamic multi-horizon labeling failed: {exc}")

        self.logger.warning("⚠️ Falling back to simple forward returns for target alignment")
        return {'fallback': self._create_simple_forward_returns(data, max_horizon).get('simple_returns', {})}

    def _create_simple_forward_returns(self, data: pd.DataFrame, max_horizon: int) -> Dict[str, Dict[int, np.ndarray]]:
        """Fallback method to create simple forward returns."""
        tprint_debug("🧠 Entering _create_simple_forward_returns")
        targets = {'simple_returns': {}}

        if 'close' not in data.columns:
            self.logger.warning("⚠️ Close prices unavailable for simple forward returns fallback")
            return targets

        close_prices = data['close'].values
        
        for horizon in range(1, max_horizon + 1):
            if horizon >= len(close_prices):
                break
                
            future_prices = close_prices[horizon:]
            current_prices = close_prices[:-horizon]
            
            forward_returns = np.where(
                current_prices != 0,
                (future_prices - current_prices) / current_prices,
                0.0
            )
            
            targets['simple_returns'][horizon] = forward_returns
        
        return targets

    def _precompute_forward_returns_matrix(
        self,
        data: pd.DataFrame,
        target_column: str,
        max_horizon: int = 200
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """Create horizon-weighted opportunity matrices derived from multi-horizon labeling signals."""
        tprint_debug("🧠 Entering _precompute_forward_returns_matrix")
        try:
            immediate_limit, short_limit = self._get_multi_horizon_boundaries()
            immediate_limit = min(max_horizon, immediate_limit)
            short_limit = min(max_horizon, short_limit)

            aligned_targets = self._create_multi_horizon_aligned_targets(data, max_horizon)

            if 'fallback' in aligned_targets:
                fallback_matrix = aligned_targets['fallback']
                return {target_column: fallback_matrix}

            result: Dict[str, Dict[int, np.ndarray]] = {}

            long_bucket = aligned_targets.get('long', {})
            short_bucket = aligned_targets.get('short', {})
            composite_bucket = aligned_targets.get('composite', {})
            directional_bucket = aligned_targets.get('directional', {})

            def register_target(
                name: str,
                bucket: Dict[str, Optional[np.ndarray]],
                immediate_override: Optional[np.ndarray] = None,
                short_override: Optional[np.ndarray] = None,
                overall_override: Optional[np.ndarray] = None,
                leverage_override: Optional[np.ndarray] = None,
                fallback_overall: Optional[np.ndarray] = None
            ) -> None:
                immediate_arr = immediate_override if immediate_override is not None else bucket.get('immediate')
                short_arr = short_override if short_override is not None else bucket.get('short')
                overall_arr = overall_override if overall_override is not None else bucket.get('overall')

                if overall_arr is None and leverage_override is not None:
                    overall_arr = leverage_override
                if overall_arr is None and bucket.get('leverage') is not None:
                    overall_arr = bucket.get('leverage')
                if overall_arr is None and fallback_overall is not None:
                    overall_arr = fallback_overall
                if short_arr is None and overall_arr is not None:
                    short_arr = overall_arr
                if short_arr is None and immediate_arr is not None:
                    short_arr = immediate_arr

                horizon_map = self._build_horizon_weighted_matrix(
                    immediate_arr,
                    short_arr,
                    overall_arr,
                    immediate_limit,
                    short_limit,
                    max_horizon
                )

                if horizon_map:
                    result[name] = horizon_map

            composite_overall = composite_bucket.get('overall') or composite_bucket.get('leverage')
            composite_immediate = composite_bucket.get('immediate') or composite_overall
            composite_short = composite_bucket.get('short') or composite_overall

            register_target('long_overall_opportunity', long_bucket, fallback_overall=composite_overall)
            register_target('long_immediate_opportunity', long_bucket, overall_override=long_bucket.get('immediate'), fallback_overall=composite_immediate)
            register_target('long_short_term_opportunity', long_bucket, immediate_override=long_bucket.get('short'), short_override=long_bucket.get('short'), fallback_overall=composite_short)
            register_target('long_leverage_adjusted_score', long_bucket, leverage_override=long_bucket.get('leverage'), fallback_overall=composite_overall)

            register_target('short_overall_opportunity', short_bucket, fallback_overall=composite_overall)
            register_target('short_immediate_opportunity', short_bucket, overall_override=short_bucket.get('immediate'), fallback_overall=composite_immediate)
            register_target('short_short_term_opportunity', short_bucket, immediate_override=short_bucket.get('short'), short_override=short_bucket.get('short'), fallback_overall=composite_short)
            register_target('short_leverage_adjusted_score', short_bucket, leverage_override=short_bucket.get('leverage'), fallback_overall=composite_overall)

            register_target('leverage_adjusted_score', composite_bucket, leverage_override=composite_bucket.get('leverage'))
            register_target('overall_opportunity', composite_bucket)
            register_target('immediate_opportunity', composite_bucket, overall_override=composite_bucket.get('immediate'), fallback_overall=composite_immediate)
            register_target('short_term_opportunity', composite_bucket, immediate_override=composite_bucket.get('short'), short_override=composite_bucket.get('short'), fallback_overall=composite_short)

            directional_confidence = directional_bucket.get('confidence')
            if directional_confidence is not None:
                result['directional_confidence'] = self._build_horizon_weighted_matrix(
                    directional_confidence,
                    directional_confidence,
                    directional_confidence,
                    immediate_limit,
                    short_limit,
                    max_horizon
                )

            opportunity_asymmetry = directional_bucket.get('asymmetry')
            if opportunity_asymmetry is not None:
                result['opportunity_asymmetry'] = self._build_horizon_weighted_matrix(
                    opportunity_asymmetry,
                    opportunity_asymmetry,
                    opportunity_asymmetry,
                    immediate_limit,
                    short_limit,
                    max_horizon
                )

            if not result:
                fallback_matrix = self._create_simple_forward_returns(data, max_horizon).get('simple_returns', {})
                return {target_column: fallback_matrix}

            self.logger.info(f"✅ Prepared multi-horizon matrices for {len(result)} target columns")
            return result

        except Exception as exc:
            self.logger.error(f"❌ Failed to build multi-horizon opportunity matrices: {exc}")
            fallback_matrix = self._create_simple_forward_returns(data, max_horizon).get('simple_returns', {})
            return {target_column: fallback_matrix}

    def _create_time_split(self, data_length: int, train_ratio: float = 0.7) -> Tuple[int, int]:
        """
        Create time-based train/validation split indices.
        
        Args:
            data_length: Total length of data
            train_ratio: Ratio of data to use for training
            
        Returns:
            Tuple of (train_end_idx, validation_start_idx)
        """
        tprint_debug("🧠 Entering _create_time_split")
        train_end_idx = int(data_length * train_ratio)
        return train_end_idx, train_end_idx

    def _generate_coarse_horizons(self, min_horizon: int = 1, max_horizon: int = 200) -> List[int]:
        """
        Generate coarse set of horizons: 1-10 dense, then log-spaced up to max_horizon.
        
        Args:
            min_horizon: Minimum horizon
            max_horizon: Maximum horizon
            
        Returns:
            List of horizon values for coarse search
        """
        tprint_debug("🧠 Entering _generate_coarse_horizons")
        # Dense sampling for short horizons (1-10)
        dense_horizons = list(range(min_horizon, min(11, max_horizon + 1)))
        
        # Log-spaced sampling for longer horizons
        if max_horizon > 10:
            # Generate ~15 log-spaced points from 10 to max_horizon
            log_horizons = np.logspace(np.log10(10), np.log10(max_horizon), 15, dtype=int)
            # Remove duplicates and ensure we don't exceed max_horizon
            log_horizons = sorted(list(set(log_horizons)))
            log_horizons = [h for h in log_horizons if h <= max_horizon]
        else:
            log_horizons = []
        
        # Combine and sort
        all_horizons = sorted(list(set(dense_horizons + log_horizons)))
        return all_horizons

    def _calculate_mutual_information_robust(self, x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
        """
        Calculate robust mutual information using sklearn-style binning.
        
        Args:
            x: First variable
            y: Second variable
            n_bins: Number of bins for discretization
            
        Returns:
            Mutual information value
        """
        tprint_debug("🧠 Entering _calculate_mutual_information_robust")
        try:
            # Remove NaN values
            valid_mask = ~(np.isnan(x) | np.isnan(y))
            if not np.any(valid_mask):
                return 0.0
                
            x_clean = x[valid_mask]
            y_clean = y[valid_mask]
            
            if len(x_clean) < 10:  # Need minimum data points
                return 0.0
            
            # Use adaptive binning based on data size
            n_bins = min(n_bins, len(x_clean) // 5)
            if n_bins < 2:
                return 0.0
            
            # Create bins using quantiles for better distribution
            x_bins = pd.cut(x_clean, bins=n_bins, labels=False, duplicates='drop')
            y_bins = pd.cut(y_clean, bins=n_bins, labels=False, duplicates='drop')
            
            # Remove any remaining NaN bins
            valid_bins = ~(pd.isna(x_bins) | pd.isna(y_bins))
            if not np.any(valid_bins):
                return 0.0
                
            x_bins = x_bins[valid_bins]
            y_bins = y_bins[valid_bins]
            
            # Calculate mutual information using sklearn
            try:
                from sklearn.feature_selection import mutual_info_regression
                
                # Ensure we have valid data after filtering
                x_final = x_clean[valid_bins]
                y_final = y_clean[valid_bins]
                
                if len(x_final) < 10:  # Need minimum data points for sklearn
                    return self._calculate_mutual_information(x_final, y_final)
                
                # Reshape for sklearn
                x_reshaped = x_final.reshape(-1, 1)
                y_reshaped = y_final
                
                mi = mutual_info_regression(x_reshaped, y_reshaped, discrete_features=False)[0]
                return max(0.0, mi)
                
            except ImportError:
                # Fallback to manual calculation
                return self._calculate_mutual_information(x_clean[valid_bins], y_clean[valid_bins])
            except Exception as e:
                # Fallback to manual calculation if sklearn fails
                self.logger.warning(f"⚠️ Sklearn MI calculation failed, using fallback: {e}")
                return self._calculate_mutual_information(x_clean[valid_bins], y_clean[valid_bins])
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating robust mutual information: {e}")
            return 0.0

    def _bootstrap_mi_validation(self, feature_values: np.ndarray, forward_returns: np.ndarray, n_resamples: int = 10) -> Dict[str, float]:
        """
        Perform bootstrap sampling for variance estimation of mutual information.
        
        Args:
            feature_values: Feature values
            forward_returns: Forward returns for the horizon
            n_resamples: Number of bootstrap resamples
            
        Returns:
            Dictionary with mean_mi, std_mi, and objective score
        """
        tprint_debug("🧠 Entering _bootstrap_mi_validation")
        try:
            # Align arrays
            min_length = min(len(feature_values), len(forward_returns))
            if min_length < 20:  # Need sufficient data for bootstrap
                return {'mean_mi': 0.0, 'std_mi': 0.0, 'objective': 0.0}
                
            feature_aligned = feature_values[:min_length]
            returns_aligned = forward_returns[:min_length]
            
            mi_samples = []
            for _ in range(n_resamples):
                # Bootstrap sampling with replacement
                bootstrap_idx = self._rng.choice(min_length, min_length, replace=True)
                bootstrap_features = feature_aligned[bootstrap_idx]
                bootstrap_returns = returns_aligned[bootstrap_idx]
                
                # Calculate MI for this bootstrap sample
                mi = self._calculate_mutual_information_robust(bootstrap_features, bootstrap_returns)
                mi_samples.append(mi)
            
            mean_mi = np.mean(mi_samples)
            std_mi = np.std(mi_samples)
            
            # Objective function: mean_MI - 0.5 × std_MI (variance penalty)
            objective = mean_mi - 0.5 * std_mi
            
            return {
                'mean_mi': mean_mi,
                'std_mi': std_mi,
                'objective': objective,
                'samples': mi_samples
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Bootstrap validation failed: {e}")
            return {'mean_mi': 0.0, 'std_mi': 0.0, 'objective': 0.0}

    def _parallel_refinement(self, top_horizons: List[Tuple[int, float]], data: pd.DataFrame,
                           feature_name: str, forward_returns: Dict[int, np.ndarray],
                           train_end_idx: int, min_lookback: int, max_lookback: int) -> List[Tuple[int, float]]:
        """
        Parallel refinement of top horizons using ThreadPoolExecutor.
        
        Args:
            top_horizons: List of (horizon, mi_score) tuples
            data: Input data
            feature_name: Feature to optimize
            forward_returns: Precomputed forward returns
            train_end_idx: Training split end index
            min_lookback: Minimum lookback period
            max_lookback: Maximum lookback period
            
        Returns:
            List of refined (horizon, mi_score) tuples
        """
        tprint_debug("🧠 Entering _parallel_refinement")
        def refine_single_horizon(horizon_mi_tuple):
            horizon, coarse_mi = horizon_mi_tuple
            refinement_horizons = range(
                max(min_lookback, horizon - 10), 
                min(max_lookback, horizon + 11), 
                2  # Check every 2 periods
            )
            
            best_mi = coarse_mi
            best_refined_horizon = horizon
            
            for refined_horizon in refinement_horizons:
                if refined_horizon == horizon:
                    continue  # Already computed
                
                try:
                    # Use vectorized feature generation for refinement
                    vectorized_features = self._vectorized_feature_generation(data, feature_name, [refined_horizon])
                    feature_values = vectorized_features.get(refined_horizon)
                    
                    if feature_values is None or len(feature_values) == 0:
                        # Fallback to cached calculation
                        feature_values = self._cached_feature_calculation(data, feature_name, refined_horizon)
                        if feature_values is None or len(feature_values) == 0:
                            continue
                    
                    # Use train split for refinement evaluation
                    train_feature = feature_values[:train_end_idx] if len(feature_values) >= train_end_idx else feature_values
                    train_returns = forward_returns.get(refined_horizon, np.array([]))
                    
                    if len(train_returns) == 0:
                        continue
                    
                    # Align arrays
                    min_length = min(len(train_feature), len(train_returns))
                    if min_length < 10:
                        continue
                    
                    aligned_features = train_feature[:min_length]
                    aligned_returns = train_returns[:min_length]
                    
                    # Use vectorized MI calculation
                    mi_score = self._vectorized_mi_calculation([aligned_features], [aligned_returns])[0]
                    
                    if mi_score > best_mi:
                        best_mi = mi_score
                        best_refined_horizon = refined_horizon
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to refine horizon {refined_horizon}: {e}")
                    continue
            
            return (best_refined_horizon, best_mi)
        
        # Use ThreadPoolExecutor for parallel processing
        final_results = []
        with ThreadPoolExecutor(max_workers=min(4, len(top_horizons))) as executor:
            future_to_horizon = {executor.submit(refine_single_horizon, horizon_mi): horizon_mi
                               for horizon_mi in top_horizons}

            for future in as_completed(future_to_horizon):
                try:
                    result = future.result()
                    final_results.append(result)
                except Exception as e:
                    horizon_mi = future_to_horizon[future]
                    self.logger.warning(f"⚠️ Failed to refine horizon {horizon_mi[0]}: {e}")
                    final_results.append(horizon_mi)  # Fallback to original

        return final_results

    @staticmethod
    def _apply_minimum_lag(values: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Shift feature values by one period to enforce a minimum lag of 1."""
        if values is None:
            return np.array([], dtype=float)

        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return arr.astype(float)

        lagged = np.empty_like(arr, dtype=float)
        lagged[:] = np.nan
        lagged[1:] = arr[:-1]
        return lagged

    def _assert_lag_requirements(self, feature_name: str, horizon: int, values: np.ndarray) -> None:
        """Validate that feature arrays satisfy minimum lag requirements and record metadata."""
        if values is None:
            raise ValueError(f"Feature '{feature_name}' produced no values for horizon {horizon}")

        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            raise ValueError(f"Feature '{feature_name}' produced empty array for horizon {horizon}")

        required_lag = 1
        leading_window = arr[:required_lag]
        if not np.isnan(leading_window).all():
            raise ValueError(
                f"Feature '{feature_name}' (horizon={horizon}) exposes contemporaneous values; "
                "expected leading NaNs after enforcing lag."
            )

        feature_meta = self.feature_lag_metadata.setdefault(feature_name, {})
        feature_meta[horizon] = {
            'max_lag': max(required_lag, int(horizon)),
            'required_lag': required_lag,
            'has_leading_nulls': True,
        }

    def _vectorized_feature_generation(self, data: pd.DataFrame, feature_name: str,
                                     horizons: List[int]) -> Dict[int, np.ndarray]:
        """
        Vectorized feature generation for multiple horizons using numpy operations.

        Args:
            data: Input data
            feature_name: Feature to generate
            horizons: List of lookback periods
            
        Returns:
            Dictionary mapping horizon to feature values
        """
        tprint_debug("🧠 Entering _vectorized_feature_generation")
        try:
            # Extract price data for vectorized operations
            if 'close' not in data.columns:
                return {}
            
            close_prices = data['close'].values
            
            # Generate features for all horizons at once using numpy broadcasting
            results = {}
            
            for horizon in horizons:
                if horizon <= 0 or horizon >= len(close_prices):
                    continue

                # Vectorized feature generation based on feature type
                if 'returns' in feature_name.lower():
                    # Simple returns: (price[t] - price[t-horizon]) / price[t-horizon]
                    shifted_prices = np.roll(close_prices, horizon)
                    returns = (close_prices - shifted_prices) / np.maximum(shifted_prices, 1e-8)
                    returns[:horizon] = np.nan  # Remove invalid values
                    lagged = self._apply_minimum_lag(returns)
                    self._assert_lag_requirements(feature_name, horizon, lagged)
                    results[horizon] = lagged

                elif 'momentum' in feature_name.lower():
                    # Momentum: price[t] - price[t-horizon]
                    shifted_prices = np.roll(close_prices, horizon)
                    momentum = close_prices - shifted_prices
                    momentum[:horizon] = np.nan
                    lagged = self._apply_minimum_lag(momentum)
                    self._assert_lag_requirements(feature_name, horizon, lagged)
                    results[horizon] = lagged

                elif 'sma' in feature_name.lower() or 'moving_average' in feature_name.lower():
                    # Simple Moving Average
                    if horizon <= len(close_prices):
                        sma = np.full_like(close_prices, np.nan)
                        for i in range(horizon, len(close_prices)):
                            sma[i] = np.mean(close_prices[i-horizon:i])
                        lagged = self._apply_minimum_lag(sma)
                        self._assert_lag_requirements(feature_name, horizon, lagged)
                        results[horizon] = lagged

                elif 'ema' in feature_name.lower() or 'exponential' in feature_name.lower():
                    # Exponential Moving Average
                    alpha = 2.0 / (horizon + 1)
                    ema = np.full_like(close_prices, np.nan)
                    if len(close_prices) > 0:
                        ema[0] = close_prices[0]
                        for i in range(1, len(close_prices)):
                            ema[i] = alpha * close_prices[i] + (1 - alpha) * ema[i-1]
                    lagged = self._apply_minimum_lag(ema)
                    self._assert_lag_requirements(feature_name, horizon, lagged)
                    results[horizon] = lagged

                elif 'volatility' in feature_name.lower():
                    # Rolling volatility (standard deviation of returns)
                    if horizon < len(close_prices):
                        returns = np.diff(close_prices) / close_prices[:-1]
                        volatility = np.full_like(close_prices, np.nan)
                        for i in range(horizon, len(returns)):
                            volatility[i] = np.std(returns[i-horizon:i])
                        lagged = self._apply_minimum_lag(volatility)
                        self._assert_lag_requirements(feature_name, horizon, lagged)
                        results[horizon] = lagged

                else:
                    # Fallback to individual calculation
                    try:
                        feature_values = self._cached_feature_calculation(data, feature_name, horizon)
                        if feature_values is not None:
                            lagged = self._apply_minimum_lag(feature_values)
                            self._assert_lag_requirements(feature_name, horizon, lagged)
                            results[horizon] = lagged
                    except Exception:
                        continue
            
            return results
            
        except Exception as e:
            self.logger.warning(f"⚠️ Vectorized feature generation failed: {e}")
            return {}

    def _optimize_coarse_to_refine(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        lookback_range: Tuple[int, int],
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize using coarse-to-refine approach with bootstrap validation.
        
        Args:
            data: Input data with features and target
            feature_name: Name of the feature to optimize
            target_column: Target column for optimization
            lookback_range: Min and max lookback periods to test
            **kwargs: Additional parameters
            
        Returns:
            OptimizationResult with best lookback period and score
        """
        tprint_debug("🧠 Entering _optimize_coarse_to_refine")
        try:
            min_lookback, max_lookback = lookback_range
            # Step 1: Get shared forward returns matrix (reused across all features)
            forward_returns = self._get_shared_forward_returns_matrix(data, target_column, max_horizon=max_lookback)
            if not forward_returns:
                return self._create_failed_result("coarse_to_refine", 0.0)
            
            # Step 2: Create time-based train/validation split
            train_end_idx, val_start_idx = self._create_time_split(len(data), train_ratio=0.7)
            
            # Step 3: Generate coarse horizons
            coarse_horizons = self._generate_coarse_horizons(min_lookback, max_lookback)
            
            # Step 4: Vectorized coarse search with early termination
            coarse_results = []
            
            # Use vectorized feature generation for all coarse horizons
            vectorized_features = self._vectorized_feature_generation(data, feature_name, coarse_horizons)
            
            # Prepare data for vectorized MI calculation
            valid_horizons = []
            features_list = []
            returns_list = []
            
            for horizon in coarse_horizons:
                if horizon not in vectorized_features:
                    continue
                    
                feature_values = vectorized_features[horizon]
                
                if feature_values is None or len(feature_values) == 0:
                    continue
                
                # Use train split for coarse evaluation
                train_feature = feature_values[:train_end_idx] if len(feature_values) >= train_end_idx else feature_values
                train_returns = forward_returns.get(horizon, np.array([]))
                
                if len(train_returns) == 0:
                    continue
                
                # Align arrays
                min_length = min(len(train_feature), len(train_returns))
                if min_length < 10:
                    continue
                
                valid_horizons.append(horizon)
                features_list.append(train_feature[:min_length])
                returns_list.append(train_returns[:min_length])
            
            # Vectorized MI calculation for all valid horizons
            if features_list and returns_list:
                mi_scores = self._vectorized_mi_calculation(features_list, returns_list)
                coarse_results = [(horizon, mi) for horizon, mi in zip(valid_horizons, mi_scores) if mi > 0]
                
                # Smart early termination: check if MI improvements are minimal
                if len(coarse_results) >= 3:
                    coarse_results.sort(key=lambda x: x[1], reverse=True)
                    best_mi = coarse_results[0][1]
                    second_best_mi = coarse_results[1][1]
                    third_best_mi = coarse_results[2][1]
                    
                    # If top 3 results are very close, skip refinement
                    mi_range = best_mi - third_best_mi
                    if mi_range < 0.001:  # Less than 0.1% difference
                        self.logger.info(f'✅ {feature_name}: best_lookback={coarse_results[0][0]}, score={best_mi:.6f} (early termination: minimal improvement)')
                        return OptimizationResult(
                            best_lookback_period=coarse_results[0][0],
                            best_score=best_mi,
                            optimization_method="coarse_to_refine",
                            total_trials=len(coarse_results),
                            optimization_time=0.0,
                            convergence_achieved=True,
                            metadata={
                                'feature_name': feature_name,
                                'target_column': target_column,
                                'coarse_horizons': len(coarse_results),
                                'early_termination': True,
                                'reason': 'minimal_improvement',
                                'mi_range': mi_range
                            }
                        )
            
            if not coarse_results:
                return self._create_failed_result("coarse_to_refine", 0.0)
            
            # Step 5: Pick top 3 horizons
            coarse_results.sort(key=lambda x: x[1], reverse=True)
            top_3_horizons = [(horizon, mi) for horizon, mi in coarse_results[:3]]
            
            # Early stopping check
            best_coarse_mi = coarse_results[0][1]
            if best_coarse_mi < 1e-3:
                # Early stopping - return best result
                best_horizon = coarse_results[0][0]
                self.logger.info(f'✅ {feature_name}: best_lookback={best_horizon}, score={best_coarse_mi:.6f}')
                return OptimizationResult(
                    best_lookback_period=top_3_horizons[0],
                    best_score=best_coarse_mi,
                    optimization_method="coarse_to_refine",
                    total_trials=len(coarse_results),
                    optimization_time=0.0,
                    convergence_achieved=True,
                    metadata={
                        'feature_name': feature_name,
                        'target_column': target_column,
                        'coarse_horizons': len(coarse_results),
                        'early_stopping': True,
                        'reason': 'low_mi'
                    }
                )
            
            # Step 6: Parallel refinement around top horizons
            refined_results = self._parallel_refinement(
                top_3_horizons, data, feature_name, forward_returns, 
                train_end_idx, min_lookback, max_lookback
            )
            
            # Combine coarse and refined results
            all_candidates = coarse_results + refined_results
            
            # Step 7: Memory-efficient batch bootstrap validation
            bootstrap_results = []
            batch_size = 3  # Process in small batches to manage memory
            
            for i in range(0, min(10, len(all_candidates)), batch_size):
                batch = all_candidates[i:i+batch_size]
                
                for horizon, mi in batch:
                    try:
                        # Use cached feature calculation
                        feature_values = self._cached_feature_calculation(data, feature_name, horizon)
                        
                        if feature_values is None or len(feature_values) == 0:
                            continue
                        
                        train_feature = feature_values[:train_end_idx] if len(feature_values) >= train_end_idx else feature_values
                        train_returns = forward_returns.get(horizon, np.array([]))
                        
                        if len(train_returns) == 0:
                            continue
                        
                        min_length = min(len(train_feature), len(train_returns))
                        if min_length < 20:  # Need sufficient data for bootstrap
                            bootstrap_results.append((horizon, mi, 0.0, 0.0, mi))  # Use raw MI
                            continue
                        
                        # Bootstrap validation (with reduced samples: 10 instead of 20)
                        bootstrap_stats = self._bootstrap_mi_validation(
                            train_feature[:min_length], 
                            train_returns[:min_length],
                            n_resamples=10  # 50% reduction from 20 to 10
                        )
                        
                        bootstrap_results.append((
                            horizon, 
                            bootstrap_stats['mean_mi'], 
                            bootstrap_stats['std_mi'], 
                            bootstrap_stats['objective'],
                            mi  # Keep original MI for comparison
                        ))
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️ Bootstrap failed for horizon {horizon}: {e}")
                        continue
                
                # Memory cleanup after each batch
                gc.collect()
            
            # Step 8: Time stability check (validation split)
            if bootstrap_results:
                best_candidates = sorted(bootstrap_results, key=lambda x: x[3], reverse=True)[:3]  # Top 3 by objective
                
                stability_results = []
                for horizon, mean_mi, std_mi, objective, original_mi in best_candidates:
                    try:
                        # Use cached feature calculation
                        feature_values = self._cached_feature_calculation(data, feature_name, horizon)
                        
                        if feature_values is None or len(feature_values) == 0:
                            continue
                        
                        # Test on validation split
                        val_feature = feature_values[val_start_idx:] if len(feature_values) > val_start_idx else np.array([])
                        val_returns = forward_returns.get(horizon, np.array([]))
                        
                        if len(val_feature) == 0 or len(val_returns) == 0:
                            stability_results.append((horizon, mean_mi, std_mi, objective, original_mi, 0.0))
                            continue
                        
                        min_length = min(len(val_feature), len(val_returns))
                        if min_length < 10:
                            stability_results.append((horizon, mean_mi, std_mi, objective, original_mi, 0.0))
                            continue
                        
                        # Calculate MI on validation split
                        val_mi = self._calculate_mutual_information_robust(
                            val_feature[:min_length], 
                            val_returns[:min_length]
                        )
                        
                        stability_results.append((horizon, mean_mi, std_mi, objective, original_mi, val_mi))
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️ Stability check failed for horizon {horizon}: {e}")
                        continue
                
                # Prefer horizons that don't collapse OOS (validation MI > 0.5 * train MI)
                final_results = []
                for horizon, mean_mi, std_mi, objective, original_mi, val_mi in stability_results:
                    stability_penalty = 0.0
                    if mean_mi > 0 and val_mi < 0.5 * mean_mi:
                        stability_penalty = 0.1  # Penalize poor OOS performance
                    
                    final_score = objective - stability_penalty
                    final_results.append((horizon, final_score, mean_mi, std_mi, val_mi, original_mi))
                
                # Select best horizon
                if final_results:
                    final_results.sort(key=lambda x: x[1], reverse=True)
                    best_horizon, best_score, best_mean_mi, best_std_mi, best_val_mi, best_original_mi = final_results[0]
                    
                    # Log cache performance
                    cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
                    self.logger.info(f'✅ {feature_name}: best_lookback={best_horizon}, score={best_score:.6f} (cache_hit_rate={cache_hit_rate:.2%})')
                    
                    return OptimizationResult(
                        best_lookback_period=best_horizon,
                        best_score=best_score,
                        optimization_method="coarse_to_refine",
                        total_trials=len(all_candidates),
                        optimization_time=0.0,  # Will be set by caller
                        convergence_achieved=True,
                        metadata={
                            'feature_name': feature_name,
                            'target_column': target_column,
                            'coarse_horizons': len(coarse_results),
                            'refined_horizons': len(refined_results),
                            'bootstrap_samples': 10,  # Updated to reflect 50% reduction
                            'mean_mi': best_mean_mi,
                            'std_mi': best_std_mi,
                            'val_mi': best_val_mi,
                            'original_mi': best_original_mi,
                            'top_3_coarse': top_3_horizons,
                            'stability_check': True,
                            'cache_hit_rate': cache_hit_rate,
                            'vectorized_ops': MATRIX_OPS_AVAILABLE
                        }
                    )
            
            # Fallback to best coarse result
            best_horizon, best_mi = coarse_results[0]
            self.logger.info(f'✅ {feature_name}: best_lookback={best_horizon}, score={best_mi:.6f}')
            
            return OptimizationResult(
                best_lookback_period=best_horizon,
                best_score=best_mi,
                optimization_method="coarse_to_refine",
                total_trials=len(coarse_results),
                optimization_time=0.0,
                convergence_achieved=True,
                metadata={
                    'feature_name': feature_name,
                    'target_column': target_column,
                    'coarse_horizons': len(coarse_results),
                    'refined_horizons': 0,
                    'fallback': True
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Coarse-to-refine optimization failed: {e}")
            return self._create_failed_result("coarse_to_refine", 0.0)
