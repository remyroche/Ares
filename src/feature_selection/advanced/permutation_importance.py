"""
Permutation Importance Calculator

This module provides comprehensive permutation importance calculation
for feature selection with various scoring metrics and validation methods.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# Import project utilities
from src.utils.tprint import (
    tprint,
    tprint_success,
    tprint_warning,
    tprint_performance,
    tprint_debug,
    tprint_info,
)
from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, HardwareConfig

logger = logging.getLogger(__name__)

@dataclass
class PermutationConfig:
    """Configuration for permutation importance calculation."""
    # Permutation settings
    n_repeats: int = 10
    random_state: int = 42
    n_jobs: int = -1

    # Scoring settings
    scoring: str = 'neg_mean_squared_error'
    cv_folds: int = 5
    enable_cross_validation: bool = True

    # Performance settings
    enable_parallel: bool = True
    chunk_size: int = 1000
    max_workers: Optional[int] = None

    # Hardware optimization
    enable_hardware_optimization: bool = True

    # Validation settings
    enable_stability_check: bool = True
    stability_threshold: float = 0.1
    min_permutations: int = 5

class PermutationImportanceCalculator:
    """Calculator for permutation importance with comprehensive validation."""

    def __init__(self, config: Optional[PermutationConfig] = None):
        """Initialize permutation importance calculator."""
        self.config = config or PermutationConfig()
        self.logger = logger.getChild('PermutationImportanceCalculator')

        # Initialize hardware optimization
        if self.config.enable_hardware_optimization:
            self.cpu_optimizer = M1CPUOptimizer()
            if self.config.max_workers is None:
                self.config.max_workers = self.cpu_optimizer.get_optimal_worker_count()

            hw_config = HardwareConfig(
                cpu_optimization_level='balanced',
                enable_adaptive_optimization=True
            )
            self.hardware_manager = UnifiedHardwareManager(hw_config)
        else:
            self.cpu_optimizer = None
            self.hardware_manager = None
            if self.config.max_workers is None:
                self.config.max_workers = 4

        # Performance tracking
        self.performance_stats = {
            'total_calculations': 0,
            'total_time': 0.0,
            'parallel_calculations': 0,
            'stability_checks': 0
        }

        tprint_success("🔧 PermutationImportanceCalculator initialized")

    def calculate_importance(self, model: Any, X: np.ndarray, y: np.ndarray,
                           feature_names: Optional[List[str]] = None,
                           custom_scoring: Optional[Callable] = None) -> Dict[str, Any]:
        """Calculate permutation importance for a fitted model."""
        tprint_info(f"🔧 Calculating permutation importance: {X.shape}")

        start_time = time.time()

        try:
            # Prepare feature names
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            # Determine scoring function
            if custom_scoring is not None:
                scoring_func = custom_scoring
            else:
                scoring_func = self._get_scoring_function()

            # Calculate importance
            if self.config.enable_parallel:
                importance_results = self._calculate_parallel_importance(
                    model, X, y, scoring_func
                )
            else:
                importance_results = self._calculate_sequential_importance(
                    model, X, y, scoring_func
                )

            # Calculate statistics
            importance_mean = np.mean(importance_results, axis=0)
            importance_std = np.std(importance_results, axis=0)
            importance_scores = importance_results

            # Stability check
            stability_info = {}
            if self.config.enable_stability_check:
                stability_info = self._check_stability(importance_scores)
                self.performance_stats['stability_checks'] += 1

            # Create feature importance dictionary
            feature_importance = {
                name: {
                    'importance_mean': float(importance_mean[i]),
                    'importance_std': float(importance_std[i]),
                    'importance_scores': importance_scores[:, i].tolist(),
                    'stability': stability_info.get(name, {}).get('stable', True)
                }
                for i, name in enumerate(feature_names)
            }

            # Update performance stats
            end_time = time.time()
            execution_time = end_time - start_time
            self.performance_stats['total_calculations'] += 1
            self.performance_stats['total_time'] += execution_time

            if self.config.enable_parallel:
                self.performance_stats['parallel_calculations'] += 1

            result = {
                'success': True,
                'feature_importance': feature_importance,
                'importance_mean': importance_mean.tolist(),
                'importance_std': importance_std.tolist(),
                'importance_scores': importance_scores.tolist(),
                'feature_names': feature_names,
                'n_repeats': self.config.n_repeats,
                'execution_time': execution_time,
                'stability_info': stability_info,
                'performance_stats': self.performance_stats.copy()
            }

            tprint_success(f"✅ Permutation importance calculated in {execution_time:.3f}s")

            return result

        except Exception as e:
            self.logger.error(f"Permutation importance calculation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }

    def _get_scoring_function(self) -> Callable:
        """Get scoring function based on configuration."""
        if self.config.scoring == 'neg_mean_squared_error':
            from sklearn.metrics import mean_squared_error
            return lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
        elif self.config.scoring == 'r2':
            from sklearn.metrics import r2_score
            return r2_score
        elif self.config.scoring == 'accuracy':
            from sklearn.metrics import accuracy_score
            return accuracy_score
        else:
            # Default to MSE
            from sklearn.metrics import mean_squared_error
            return lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)

    def _calculate_sequential_importance(self, model: Any, X: np.ndarray, y: np.ndarray,
                                       scoring_func: Callable) -> np.ndarray:
        """Calculate permutation importance sequentially."""
        n_features = X.shape[1]
        importance_scores = np.zeros((self.config.n_repeats, n_features))

        # Get baseline score
        baseline_score = self._get_baseline_score(model, X, y, scoring_func)

        for repeat in range(self.config.n_repeats):
            tprint_debug(f"🔧 Permutation repeat {repeat + 1}/{self.config.n_repeats}")

            for feature_idx in range(n_features):
                # Create permuted data
                X_permuted = X.copy()
                np.random.seed(self.config.random_state + repeat)
                X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])

                # Calculate score with permuted feature
                permuted_score = self._get_baseline_score(model, X_permuted, y, scoring_func)

                # Importance is the difference from baseline
                importance_scores[repeat, feature_idx] = baseline_score - permuted_score

        return importance_scores

    def _calculate_parallel_importance(self, model: Any, X: np.ndarray, y: np.ndarray,
                                     scoring_func: Callable) -> np.ndarray:
        """Calculate permutation importance in parallel."""
        n_features = X.shape[1]
        importance_scores = np.zeros((self.config.n_repeats, n_features))

        # Get baseline score
        baseline_score = self._get_baseline_score(model, X, y, scoring_func)

        # Create tasks for parallel execution
        tasks = []
        for repeat in range(self.config.n_repeats):
            for feature_idx in range(n_features):
                tasks.append((repeat, feature_idx, baseline_score))

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._calculate_single_permutation, model, X, y, scoring_func, task): task
                for task in tasks
            }

            for future in futures:
                repeat, feature_idx, importance = future.result()
                importance_scores[repeat, feature_idx] = importance

        return importance_scores

    def _calculate_single_permutation(self, model: Any, X: np.ndarray, y: np.ndarray,
                                    scoring_func: Callable, task: Tuple[int, int, float]) -> Tuple[int, int, float]:
        """Calculate importance for a single permutation."""
        repeat, feature_idx, baseline_score = task

        # Create permuted data
        X_permuted = X.copy()
        np.random.seed(self.config.random_state + repeat)
        X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])

        # Calculate score with permuted feature
        permuted_score = self._get_baseline_score(model, X_permuted, y, scoring_func)

        # Importance is the difference from baseline
        importance = baseline_score - permuted_score

        return repeat, feature_idx, importance

    def _get_baseline_score(self, model: Any, X: np.ndarray, y: np.ndarray,
                           scoring_func: Callable) -> float:
        """Get baseline score for the model."""
        if self.config.enable_cross_validation:
            # Use cross-validation
            scores = cross_val_score(
                model, X, y,
                cv=self.config.cv_folds,
                scoring=make_scorer(scoring_func),
                n_jobs=1  # Avoid nested parallelism
            )
            return np.mean(scores)
        else:
            # Use single prediction
            y_pred = model.predict(X)
            return scoring_func(y, y_pred)

    def _check_stability(self, importance_scores: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Check stability of importance scores across repeats."""
        n_features = importance_scores.shape[1]
        stability_info = {}

        for feature_idx in range(n_features):
            scores = importance_scores[:, feature_idx]

            # Calculate coefficient of variation
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            cv = std_score / abs(mean_score) if mean_score != 0 else float('inf')

            # Determine stability
            stable = cv < self.config.stability_threshold

            stability_info[f"feature_{feature_idx}"] = {
                'stable': stable,
                'coefficient_of_variation': float(cv),
                'mean_score': float(mean_score),
                'std_score': float(std_score)
            }

        return stability_info

    def calculate_feature_interactions(self, model: Any, X: np.ndarray, y: np.ndarray,
                                     feature_names: Optional[List[str]] = None,
                                     top_features: int = 10) -> Dict[str, Any]:
        """Calculate feature interaction importance."""
        tprint_info("🔧 Calculating feature interactions")

        try:
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            # Get top features by individual importance
            individual_importance = self.calculate_importance(model, X, y, feature_names)
            if not individual_importance['success']:
                return individual_importance

            # Sort features by importance
            importance_scores = np.array(individual_importance['importance_mean'])
            top_indices = np.argsort(importance_scores)[-top_features:]

            # Calculate pairwise interactions
            interaction_scores = {}

            for i, idx1 in enumerate(top_indices):
                for j, idx2 in enumerate(top_indices[i+1:], i+1):
                    # Calculate interaction importance
                    interaction_importance = self._calculate_pairwise_interaction(
                        model, X, y, idx1, idx2
                    )

                    feature_pair = (feature_names[idx1], feature_names[idx2])
                    interaction_scores[feature_pair] = interaction_importance

            return {
                'success': True,
                'interaction_scores': interaction_scores,
                'top_features': [feature_names[i] for i in top_indices],
                'n_interactions': len(interaction_scores)
            }

        except Exception as e:
            self.logger.error(f"Feature interaction calculation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _calculate_pairwise_interaction(self, model: Any, X: np.ndarray, y: np.ndarray,
                                      idx1: int, idx2: int) -> float:
        """Calculate interaction importance between two features."""
        # Get baseline score
        baseline_score = self._get_baseline_score(model, X, y, self._get_scoring_function())

        # Permute both features
        X_permuted = X.copy()
        np.random.seed(self.config.random_state)
        X_permuted[:, idx1] = np.random.permutation(X_permuted[:, idx1])
        X_permuted[:, idx2] = np.random.permutation(X_permuted[:, idx2])

        # Calculate score with both features permuted
        permuted_score = self._get_baseline_score(model, X_permuted, y, self._get_scoring_function())

        # Interaction importance
        interaction_importance = baseline_score - permuted_score

        return float(interaction_importance)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()

        if stats['total_calculations'] > 0:
            stats['avg_time_per_calculation'] = stats['total_time'] / stats['total_calculations']
            stats['parallel_ratio'] = stats['parallel_calculations'] / stats['total_calculations']
        else:
            stats['avg_time_per_calculation'] = 0.0
            stats['parallel_ratio'] = 0.0

        tprint_performance(f"📊 Permutation Stats: {stats['total_calculations']} calculations, "
                         f"{stats['avg_time_per_calculation']:.3f}s avg, "
                         f"{stats['parallel_ratio']:.1%} parallel")

        return stats

def create_permutation_calculator(config: Optional[PermutationConfig] = None) -> PermutationImportanceCalculator:
    """Create a permutation importance calculator."""
    return PermutationImportanceCalculator(config)
