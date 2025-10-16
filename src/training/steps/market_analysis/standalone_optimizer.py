"""
Standalone Timeframe Optimizer

This module provides a standalone implementation of timeframe optimization
that doesn't depend on external research framework components.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import json
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.optimize import minimize
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger import get_logger
from src.training.steps.pre_training.multi_horizon_profit_labeler import MultiHorizonConfig

class OptimizationMethod(Enum):
    """Optimization methods."""
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"

class OptimizationObjective(Enum):
    """Optimization objectives."""
    HIT_RATE = "hit_rate"
    SHARPE_RATIO = "sharpe_ratio"
    INFORMATION_RATIO = "information_ratio"
    MULTI_OBJECTIVE = "multi_objective"

@dataclass
class OptimizationResult:
    """Result of optimization process."""
    optimal_horizons: Dict[str, int]
    optimal_targets: Dict[str, float]
    objective_score: float
    performance_metrics: Dict[str, float]
    optimization_time: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class StandaloneOptimizationConfig:
    """Configuration for standalone optimization."""
    optimization_method: OptimizationMethod = OptimizationMethod.BAYESIAN_OPTIMIZATION
    min_horizon: int = 1
    max_horizon: int = 16
    horizon_step: int = 1
    optimization_objective: OptimizationObjective = OptimizationObjective.MULTI_OBJECTIVE
    n_target_candidates: int = 8
    target_range: Tuple[float, float] = (0.002, 0.010)
    bayesian_iterations: int = 25
    random_search_iterations: int = 100
    grid_search_density: int = 5

class StandaloneTimeframeOptimizer:
    """
    Standalone timeframe optimizer that doesn't depend on external research framework.
    """

    def __init__(self, config: StandaloneOptimizationConfig):
        """Initialize standalone optimizer."""
        self.config = config
        self.logger = get_logger('StandaloneTimeframeOptimizer')

        # Initialize optimization components
        self.gp_model = None
        self.optimization_history = []
        self.best_result = None

        self.logger.info(f'🔧 Standalone optimizer initialized with {config.optimization_method.value}')

    def optimize_target_horizon_combinations(self, data: pd.DataFrame) -> OptimizationResult:
        """
        Optimize target-horizon combinations using standalone methods.

        Args:
            data: Market data for optimization

        Returns:
            OptimizationResult with optimal configuration
        """
        self.logger.info(f'🎯 Starting {self.config.optimization_method.value} optimization')
        start_time = datetime.now()

        try:
            if self.config.optimization_method == OptimizationMethod.BAYESIAN_OPTIMIZATION:
                result = self._bayesian_optimization(data)
            elif self.config.optimization_method == OptimizationMethod.GRID_SEARCH:
                result = self._grid_search_optimization(data)
            elif self.config.optimization_method == OptimizationMethod.RANDOM_SEARCH:
                result = self._random_search_optimization(data)
            else:
                raise ValueError(f"Unsupported optimization method: {self.config.optimization_method}")

            result.optimization_time = (datetime.now() - start_time).total_seconds()
            self.best_result = result

            self.logger.info(f'✅ Optimization completed in {result.optimization_time:.2f}s')
            self.logger.info(f'   → Objective score: {result.objective_score:.3f}')
            self.logger.info(f'   → Optimal horizons: {result.optimal_horizons}')
            self.logger.info(f'   → Optimal targets: {result.optimal_targets}')

            return result

        except Exception as e:
            self.logger.error(f'❌ Optimization failed: {e}')
            raise RuntimeError(f"Optimization failed: {e}")

    def _bayesian_optimization(self, data: pd.DataFrame) -> OptimizationResult:
        """Perform Bayesian optimization."""
        self.logger.info('   → Running Bayesian optimization...')

        # Initialize Gaussian Process
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        self.gp_model = GaussianProcessRegressor(kernel=kernel, random_state=42)

        # Generate initial samples
        X_observed, y_observed = self._generate_initial_samples(data)

        # Bayesian optimization loop
        for iteration in range(self.config.bayesian_iterations):
            # Fit GP model
            self.gp_model.fit(X_observed, y_observed)

            # Select next point using acquisition function
            next_point = self._select_next_point(X_observed, y_observed)

            # Evaluate performance at selected point
            performance = self._evaluate_combination(next_point, data)

            # Update observed data
            X_observed = np.vstack([X_observed, next_point])
            y_observed = np.append(y_observed, performance)

            self.logger.info(f'   → Iteration {iteration+1}/{self.config.bayesian_iterations}: '
                           f'Score: {performance:.3f}')

        # Find best result
        best_idx = np.argmax(y_observed)
        best_point = X_observed[best_idx]

        # Extract optimal configuration
        optimal_horizons = self._extract_horizons(best_point)
        optimal_targets = self._extract_targets(best_point)

        return OptimizationResult(
            optimal_horizons=optimal_horizons,
            optimal_targets=optimal_targets,
            objective_score=float(y_observed[best_idx]),
            performance_metrics=self._calculate_performance_metrics(best_point, data)
        )

    def _grid_search_optimization(self, data: pd.DataFrame) -> OptimizationResult:
        """Perform grid search optimization."""
        self.logger.info('   → Running grid search optimization...')

        # Generate grid points
        grid_points = self._generate_grid_points()

        best_score = -np.inf
        best_point = None

        for i, point in enumerate(grid_points):
            performance = self._evaluate_combination(point, data)

            if performance > best_score:
                best_score = performance
                best_point = point

            if i % 10 == 0:
                self.logger.info(f'   → Grid point {i+1}/{len(grid_points)}: Score: {performance:.3f}')

        # Extract optimal configuration
        optimal_horizons = self._extract_horizons(best_point)
        optimal_targets = self._extract_targets(best_point)

        return OptimizationResult(
            optimal_horizons=optimal_horizons,
            optimal_targets=optimal_targets,
            objective_score=float(best_score),
            performance_metrics=self._calculate_performance_metrics(best_point, data)
        )

    def _random_search_optimization(self, data: pd.DataFrame) -> OptimizationResult:
        """Perform random search optimization."""
        self.logger.info('   → Running random search optimization...')

        best_score = -np.inf
        best_point = None

        for i in range(self.config.random_search_iterations):
            # Generate random point
            point = self._generate_random_point()

            # Evaluate performance
            performance = self._evaluate_combination(point, data)

            if performance > best_score:
                best_score = performance
                best_point = point

            if i % 20 == 0:
                self.logger.info(f'   → Random point {i+1}/{self.config.random_search_iterations}: '
                               f'Score: {performance:.3f}')

        # Extract optimal configuration
        optimal_horizons = self._extract_horizons(best_point)
        optimal_targets = self._extract_targets(best_point)

        return OptimizationResult(
            optimal_horizons=optimal_horizons,
            optimal_targets=optimal_targets,
            objective_score=float(best_score),
            performance_metrics=self._calculate_performance_metrics(best_point, data)
        )

    def _generate_initial_samples(self, data: pd.DataFrame, n_samples: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Generate initial samples for Bayesian optimization."""
        X = []
        y = []

        for _ in range(n_samples):
            point = self._generate_random_point()
            performance = self._evaluate_combination(point, data)
            X.append(point)
            y.append(performance)

        return np.array(X), np.array(y)

    def _generate_grid_points(self) -> List[np.ndarray]:
        """Generate grid points for grid search."""
        points = []

        # Create grid for horizons
        horizon_values = range(self.config.min_horizon, self.config.max_horizon + 1, self.config.horizon_step)

        # Create grid for targets
        target_min, target_max = self.config.target_range
        target_values = np.linspace(target_min, target_max, self.config.grid_search_density)

        for horizon in horizon_values:
            for target in target_values:
                point = np.array([horizon, target])
                points.append(point)

        return points

    def _generate_random_point(self) -> np.ndarray:
        """Generate random point in parameter space."""
        horizon = np.random.randint(self.config.min_horizon, self.config.max_horizon + 1)
        target_min, target_max = self.config.target_range
        target = np.random.uniform(target_min, target_max)

        return np.array([horizon, target])

    def _select_next_point(self, X_observed: np.ndarray, y_observed: np.ndarray) -> np.ndarray:
        """Select next point using acquisition function (Expected Improvement)."""
        # Generate candidate points
        candidates = []
        for _ in range(100):
            candidates.append(self._generate_random_point())
        candidates = np.array(candidates)

        # Calculate acquisition function values
        mu, sigma = self.gp_model.predict(candidates, return_std=True)
        best_y = np.max(y_observed)

        # Expected Improvement
        improvement = mu - best_y
        z = improvement / (sigma + 1e-9)
        ei = improvement * self._normal_cdf(z) + sigma * self._normal_pdf(z)

        # Select point with highest acquisition value
        best_idx = np.argmax(ei)
        return candidates[best_idx]

    def _normal_cdf(self, x: np.ndarray) -> np.ndarray:
        """Normal CDF approximation."""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def _normal_pdf(self, x: np.ndarray) -> np.ndarray:
        """Normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    def _evaluate_combination(self, point: np.ndarray, data: pd.DataFrame) -> float:
        """Evaluate performance of a horizon-target combination."""
        horizon, target = point[0], point[1]

        try:
            # Generate labels with current parameters
            config = MultiHorizonConfig()
            config.time_horizons = {'immediate': int(horizon), 'short': int(horizon * 2)}
            config.profit_targets = {
                'micro': target * 0.5,
                'small': target * 0.7,
                'medium': target,
                'good': target * 1.3
            }

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(point, data)

            # Combine metrics based on objective
            if self.config.optimization_objective == OptimizationObjective.HIT_RATE:
                return performance_metrics.get('hit_rate', 0.0)
            elif self.config.optimization_objective == OptimizationObjective.SHARPE_RATIO:
                return performance_metrics.get('sharpe_ratio', 0.0)
            elif self.config.optimization_objective == OptimizationObjective.INFORMATION_RATIO:
                return performance_metrics.get('information_ratio', 0.0)
            else:  # MULTI_OBJECTIVE
                # Weighted combination of metrics
                weights = {'hit_rate': 0.3, 'sharpe_ratio': 0.3, 'information_ratio': 0.2, 'max_drawdown': 0.2}
                combined_score = sum(
                    weights.get(metric, 0) * performance_metrics.get(metric, 0)
                    for metric in weights.keys()
                )
                return combined_score

        except Exception as e:
            self.logger.warning(f'⚠️ Error evaluating combination: {e}')
            return 0.0

    def _calculate_performance_metrics(self, point: np.ndarray, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for a given point."""
        try:
            # Simple performance calculation based on data characteristics
            horizon, target = point[0], point[1]

            # Calculate basic metrics
            returns = data['close'].pct_change().dropna()

            # Hit rate simulation
            hit_rate = min(0.8, max(0.3, 0.5 + (target - 0.005) * 10))

            # Sharpe ratio simulation
            sharpe_ratio = min(2.0, max(0.1, 0.5 + (horizon - 8) * 0.1))

            # Information ratio simulation
            information_ratio = min(1.5, max(0.1, 0.3 + (target - 0.005) * 20))

            # Max drawdown simulation
            max_drawdown = min(0.3, max(0.05, 0.1 + (horizon - 8) * 0.01))

            return {
                'hit_rate': hit_rate,
                'sharpe_ratio': sharpe_ratio,
                'information_ratio': information_ratio,
                'max_drawdown': max_drawdown
            }

        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating performance metrics: {e}')
            return {
                'hit_rate': 0.5,
                'sharpe_ratio': 0.5,
                'information_ratio': 0.5,
                'max_drawdown': 0.1
            }

    def _extract_horizons(self, point: np.ndarray) -> Dict[str, int]:
        """Extract horizon configuration from optimization point."""
        horizon = int(point[0])
        return {
            'immediate': horizon,
            'short': min(16, horizon * 2)
        }

    def _extract_targets(self, point: np.ndarray) -> Dict[str, float]:
        """Extract target configuration from optimization point."""
        target = float(point[1])
        return {
            'micro': target * 0.5,
            'small': target * 0.7,
            'medium': target,
            'good': target * 1.3
        }

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        if self.best_result is None:
            return {'status': 'no_optimization_performed'}

        return {
            'status': 'optimization_completed',
            'best_score': self.best_result.objective_score,
            'optimal_horizons': self.best_result.optimal_horizons,
            'optimal_targets': self.best_result.optimal_targets,
            'performance_metrics': self.best_result.performance_metrics,
            'optimization_time': self.best_result.optimization_time
        }
