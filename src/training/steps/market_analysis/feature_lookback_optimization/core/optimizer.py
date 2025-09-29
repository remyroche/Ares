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

# Import utility modules
from src.utils.common_operations import safe_dataframe_operation, validate_dataframe_columns
from src.utils.common_utilities import CommonUtilities
from src.utils.math_validation import safe_divide, safe_correlation
from src.utils.serialization_utils import UniversalSerializer
from src.utils.tprint import tprint
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time

from ..constants import OPTIMIZATION_CONSTANTS, PERFORMANCE_CONSTANTS, ALGORITHM_CONSTANTS
from ..dependency_manager import get_dependency

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'best_lookback_period': self.best_lookback_period,
            'best_score': self.best_score,
            'optimization_method': self.optimization_method,
            'total_trials': self.total_trials,
            'optimization_time': self.optimization_time,
            'convergence_achieved': self.convergence_achieved,
            'metadata': self.metadata
        }


class CoreOptimizer:
    """
    Core optimization engine for feature lookback optimization.

    Provides standardized interface for different optimization algorithms.
    """

    def __init__(self, logger=None):
        """Initialize the core optimizer."""
        self.logger = logger or get_logger('CoreOptimizer')
        self.common_utils = CommonUtilities()
        self.serializer = UniversalSerializer()
        
        # Performance tracking
        self.optimization_history = []
        self.performance_metrics = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'average_optimization_time': 0.0,
            'best_scores': []
        }

    def optimize_single_feature(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        method: OptimizationMethod = OptimizationMethod.MRMR,
        lookback_range: Tuple[int, int] = (5, 100),
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

            # Validate inputs
            if not self._validate_optimization_inputs(data, feature_name, target_column):
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
            else:
                # Fallback to MRMR
                self.logger.warning(f'⚠️ Unknown method {method.value}, falling back to MRMR')
                result = self._optimize_mrmr(data, feature_name, target_column, lookback_range, **kwargs)

            result.optimization_time = time.time() - start_time
            result.optimization_method = method.value

            # Update performance tracking
            self._update_performance_metrics(result, time.time() - start_time)

            self.logger.info(f'✅ Optimization completed: best_lookback={result.best_lookback_period}, score={result.best_score:.4f}')
            return result

        except Exception as e:
            self.logger.error(f"❌ Optimization failed for feature {feature_name}: {e}")
            return self._create_failed_result(method.value, time.time() - start_time)

    def _validate_optimization_inputs(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str
    ) -> bool:
        """Validate inputs for optimization."""
        try:
            # Check if data is valid DataFrame
            if not isinstance(data, pd.DataFrame):
                self.logger.error("Input data must be a pandas DataFrame")
                return False

            # Check if feature and target columns exist
            if feature_name not in data.columns:
                self.logger.error(f"Feature column '{feature_name}' not found in data")
                return False

            if target_column not in data.columns:
                self.logger.error(f"Target column '{target_column}' not found in data")
                return False

            # Check for sufficient data
            if len(data) < OPTIMIZATION_CONSTANTS.DEFAULT_MIN_LOOKBACK:
                self.logger.error(f"Insufficient data: {len(data)} rows, minimum {OPTIMIZATION_CONSTANTS.DEFAULT_MIN_LOOKBACK} required")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False

    def _optimize_mrmr(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        lookback_range: Tuple[int, int],
        **kwargs
    ) -> OptimizationResult:
        """Optimize using MRMR approach."""
        try:
            min_lookback, max_lookback = lookback_range
            best_score = -float('inf')
            best_lookback = min_lookback
            trials = 0

            # Test different lookback periods
            for lookback in range(min_lookback, max_lookback + 1):
                try:
                    # Calculate feature value for this lookback
                    feature_values = self._calculate_feature_for_lookback(data, feature_name, lookback)

                    # Calculate correlation with target
                    if len(feature_values) > 1 and len(data[target_column]) > 1:
                        correlation = safe_correlation(feature_values, data[target_column].values, default=0.0)
                        score = abs(correlation)  # Use absolute correlation as score

                        trials += 1

                        if score > best_score:
                            best_score = score
                            best_lookback = lookback

                except Exception as e:
                    self.logger.warning(f"Failed to evaluate lookback {lookback} for {feature_name}: {e}")
                    continue

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
                    'correlation_method': 'pearson'
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
        """Optimize using comprehensive grid search approach."""
        try:
            min_lookback, max_lookback = lookback_range
            step_size = kwargs.get('step_size', 1)
            
            self.logger.info(f'🔍 Running grid search from {min_lookback} to {max_lookback} (step={step_size})')
            
            best_score = -float('inf')
            best_lookback = min_lookback
            all_scores = []
            trials = 0
            
            # Test all lookback periods in range
            for lookback in range(min_lookback, max_lookback + 1, step_size):
                try:
                    # Calculate feature value for this lookback
                    feature_values = self._calculate_feature_for_lookback(data, feature_name, lookback)
                    
                    # Calculate multiple correlation metrics
                    correlations = self._calculate_comprehensive_correlations(
                        feature_values, data[target_column].values
                    )
                    
                    # Use weighted combination of correlation metrics
                    score = self._calculate_composite_score(correlations)
                    all_scores.append(score)
                    trials += 1
                    
                    if score > best_score:
                        best_score = score
                        best_lookback = lookback
                        
                    if trials % 10 == 0:
                        self.logger.debug(f'   → Progress: {trials} trials, best_score={best_score:.4f}')
                        
                except Exception as e:
                    self.logger.warning(f'⚠️ Failed to evaluate lookback {lookback}: {e}')
                    continue
            
            # Calculate convergence metrics
            convergence_achieved = self._check_convergence(all_scores)
            
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
                    'score_std': np.std(all_scores) if all_scores else 0.0
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
            
            # Initialize with random samples for exploration
            startup_trials = np.random.randint(min_lookback, max_lookback + 1, n_startup_trials)
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
                    
                except Exception as e:
                    self.logger.warning(f'⚠️ Startup trial failed for lookback {lookback}: {e}')
                    continue
            
            # Bayesian optimization phase
            for trial in range(n_startup_trials, n_trials):
                try:
                    # Use simple acquisition function (exploration vs exploitation)
                    if len(all_scores) < 5:
                        # More exploration
                        lookback = np.random.randint(min_lookback, max_lookback + 1)
                    else:
                        # Exploit best regions
                        best_idx = np.argmax(all_scores)
                        best_lookback = all_lookbacks[best_idx]
                        
                        # Add some exploration around best point
                        exploration_range = max(1, (max_lookback - min_lookback) // 10)
                        lookback = np.random.randint(
                            max(min_lookback, best_lookback - exploration_range),
                            min(max_lookback + 1, best_lookback + exploration_range + 1)
                        )
                    
                    feature_values = self._calculate_feature_for_lookback(data, feature_name, lookback)
                    correlations = self._calculate_comprehensive_correlations(
                        feature_values, data[target_column].values
                    )
                    score = self._calculate_composite_score(correlations)
                    
                    all_scores.append(score)
                    all_lookbacks.append(lookback)
                    
                    if trial % 10 == 0:
                        current_best = max(all_scores)
                        self.logger.debug(f'   → Trial {trial}: best_score={current_best:.4f}')
                        
                except Exception as e:
                    self.logger.warning(f'⚠️ Bayesian trial failed: {e}')
                    continue
            
            # Find best result
            if all_scores:
                best_idx = np.argmax(all_scores)
                best_score = all_scores[best_idx]
                best_lookback = all_lookbacks[best_idx]
                convergence_achieved = self._check_convergence(all_scores)
            else:
                best_score = 0.0
                best_lookback = min_lookback
                convergence_achieved = False
            
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
            
            best_score = -float('inf')
            best_lookback = min_lookback
            all_scores = []
            trials = 0
            
            # Random sampling
            for trial in range(n_trials):
                try:
                    lookback = np.random.randint(min_lookback, max_lookback + 1)
                    
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
                        
                except Exception as e:
                    self.logger.warning(f'⚠️ Random trial failed: {e}')
                    continue
            
            convergence_achieved = self._check_convergence(all_scores)
            
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
                        
                except Exception as e:
                    self.logger.warning(f'⚠️ Multi-target trial failed for lookback {lookback}: {e}')
                    continue
            
            convergence_achieved = self._check_convergence(all_scores)
            
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
        Calculate feature values for a given lookback period.

        This is a placeholder - in a real implementation, this would
        calculate actual technical indicators or features based on the lookback.
        """
        try:
            # Simple example: rolling mean of the feature
            if feature_name in data.columns:
                return data[feature_name].rolling(window=lookback, min_periods=1).mean().values
            else:
                # Return zeros if feature not found
                return np.zeros(len(data))
        except Exception as e:
            self.logger.error(f"Failed to calculate feature {feature_name} for lookback {lookback}: {e}")
            return np.zeros(len(data))

    def _create_failed_result(self, method: str, optimization_time: float) -> OptimizationResult:
        """Create a failed optimization result."""
        return OptimizationResult(
            best_lookback_period=OPTIMIZATION_CONSTANTS.DEFAULT_MIN_LOOKBACK,
            best_score=0.0,
            optimization_method=method,
            total_trials=0,
            optimization_time=optimization_time,
            convergence_achieved=False,
            metadata={'error': 'Optimization failed'}
        )

    def _calculate_comprehensive_correlations(self, feature_values: np.ndarray, target_values: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive correlation metrics."""
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
        try:
            if len(values) < 2:
                return 0.0
            return 1.0 / (1.0 + np.var(values))  # Higher stability = lower variance
        except Exception:
            return 0.0

    def _calculate_information_content(self, values: np.ndarray) -> float:
        """Calculate information content using entropy."""
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
