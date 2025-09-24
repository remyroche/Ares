"""
Grid Optimizer for Regime Detection Systems.

This module provides grid search optimization that can be used by both
NAS and TAS regime detection systems for exhaustive parameter search.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from itertools import product
from src.utils.logger import system_logger


@dataclass
class GridConfig:
    """Configuration for grid search optimization."""
    n_points_per_dimension: int = 10
    enable_parallel: bool = True
    max_workers: int = 4
    random_sample: bool = False
    random_sample_size: Optional[int] = None
    verbose: bool = True


class GridOptimizer:
    """
    Grid search optimizer for exhaustive parameter search.

    This class provides systematic grid search optimization that can be used
    by both NAS and TAS systems for finding optimal parameters in discrete
    or continuous search spaces.
    """

    def __init__(self, config: GridConfig):
        """
        Initialize the grid optimizer.

        Args:
            config: Grid search configuration
        """
        self.logger = system_logger.getChild('GridOptimizer')
        self.config = config

        self.logger.info("✅ Grid Optimizer initialized"
        self.logger.info(f"   Points per dimension: {config.n_points_per_dimension}")
        self.logger.info(f"   Random sample: {config.random_sample}")

    def optimize(self,
                objective_function: Callable,
                parameter_bounds: Dict[str, Tuple[float, float]],
                n_grid_points: Optional[int] = None) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """
        Perform grid search optimization.

        Args:
            objective_function: Function to optimize (higher is better)
            parameter_bounds: Dictionary of parameter bounds {param_name: (min, max)}
            n_grid_points: Number of grid points per dimension

        Returns:
            Tuple of (best_params, best_value, search_history)
        """
        try:
            self.logger.info("🔍 Starting grid search optimization")
            self.logger.info(f"   Parameters: {list(parameter_bounds.keys())}")

            import time
            start_time = time.time()

            # Generate grid points
            grid_points = self._generate_grid_points(parameter_bounds, n_grid_points)
            self.logger.info(f"📊 Generated {len(grid_points)} grid points")

            # Evaluate grid points
            results = self._evaluate_grid_points(objective_function, grid_points)

            # Find best solution
            best_result = max(results, key=lambda x: x['value'])
            best_params = best_result['params']
            best_value = best_result['value']

            execution_time = time.time() - start_time

            # Create search history
            search_history = [
                {
                    'iteration': i,
                    'parameters': result['params'],
                    'objective_value': result['value'],
                    'grid_position': i
                }
                for i, result in enumerate(results)
            ]

            self.logger.info(f"✅ Grid search completed in {execution_time:.2f}s")
            self.logger.info(f"🏆 Best value: {best_value".4f"}")
            self.logger.info(f"📊 Evaluated {len(results)} points")

            return best_params, best_value, search_history

        except Exception as e:
            self.logger.error(f"❌ Grid search failed: {e}")
            return {}, float('-inf'), []

    def _generate_grid_points(self,
                            parameter_bounds: Dict[str, Tuple[float, float]],
                            n_grid_points: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate grid points for evaluation.

        Args:
            parameter_bounds: Dictionary of parameter bounds
            n_grid_points: Number of grid points per dimension

        Returns:
            List of parameter dictionaries
        """
        try:
            n_points = n_grid_points or self.config.n_points_per_dimension

            # Create parameter ranges
            param_ranges = {}
            for param, (min_val, max_val) in parameter_bounds.items():
                if self.config.random_sample and self.config.random_sample_size:
                    # Random sampling
                    values = np.random.uniform(min_val, max_val, self.config.random_sample_size)
                else:
                    # Regular grid
                    values = np.linspace(min_val, max_val, n_points)
                param_ranges[param] = values

            # Generate all combinations
            param_names = list(param_ranges.keys())
            param_values = [param_ranges[name] for name in param_names]

            # Create grid points
            grid_points = []
            for combination in product(*param_values):
                point = {param_names[i]: combination[i] for i in range(len(param_names))}
                grid_points.append(point)

            # Limit to reasonable size if random sampling is not used
            if not self.config.random_sample and len(grid_points) > 10000:
                self.logger.warning(f"⚠️ Grid size {len(grid_points)} is very large, consider reducing n_points_per_dimension")
                # Take a random subset
                indices = np.random.choice(len(grid_points), 10000, replace=False)
                grid_points = [grid_points[i] for i in indices]

            return grid_points

        except Exception as e:
            self.logger.error(f"❌ Grid point generation failed: {e}")
            return []

    def _evaluate_grid_points(self,
                            objective_function: Callable,
                            grid_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate objective function on grid points.

        Args:
            objective_function: Function to evaluate
            grid_points: List of parameter dictionaries

        Returns:
            List of evaluation results
        """
        try:
            results = []

            if self.config.enable_parallel:
                # Parallel evaluation
                results = self._parallel_evaluation(objective_function, grid_points)
            else:
                # Sequential evaluation
                for i, point in enumerate(grid_points):
                    try:
                        value = objective_function(point)

                        result = {
                            'params': point,
                            'value': value,
                            'index': i
                        }

                        results.append(result)

                        if self.config.verbose and i % 100 == 0:
                            self.logger.info(f"   Evaluated {i+1}/{len(grid_points)} points")

                    except Exception as e:
                        self.logger.warning(f"⚠️ Point {i} evaluation failed: {e}")
                        results.append({
                            'params': point,
                            'value': float('-inf'),
                            'index': i
                        })

            return results

        except Exception as e:
            self.logger.error(f"❌ Grid point evaluation failed: {e}")
            return []

    def _parallel_evaluation(self,
                           objective_function: Callable,
                           grid_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate grid points in parallel.

        Args:
            objective_function: Function to evaluate
            grid_points: List of parameter dictionaries

        Returns:
            List of evaluation results
        """
        try:
            import concurrent.futures

            results = []
            max_workers = min(self.config.max_workers, len(grid_points))

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_point = {
                    executor.submit(self._safe_evaluate, objective_function, point, i): i
                    for i, point in enumerate(grid_points)
                }

                # Collect results
                for future in concurrent.futures.as_completed(future_to_point):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Parallel evaluation failed: {e}")
                        # Add failed result
                        point_idx = future_to_point[future]
                        results.append({
                            'params': grid_points[point_idx],
                            'value': float('-inf'),
                            'index': point_idx
                        })

            # Sort by original order
            results.sort(key=lambda x: x['index'])
            return results

        except Exception as e:
            self.logger.warning(f"⚠️ Parallel evaluation failed, falling back to sequential: {e}")
            return self._sequential_evaluation(objective_function, grid_points)

    def _sequential_evaluation(self,
                             objective_function: Callable,
                             grid_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate grid points sequentially.

        Args:
            objective_function: Function to evaluate
            grid_points: List of parameter dictionaries

        Returns:
            List of evaluation results
        """
        try:
            results = []

            for i, point in enumerate(grid_points):
                try:
                    value = objective_function(point)

                    result = {
                        'params': point,
                        'value': value,
                        'index': i
                    }

                    results.append(result)

                    if self.config.verbose and i % 100 == 0:
                        self.logger.info(f"   Evaluated {i+1}/{len(grid_points)} points")

                except Exception as e:
                    self.logger.warning(f"⚠️ Point {i} evaluation failed: {e}")
                    results.append({
                        'params': point,
                        'value': float('-inf'),
                        'index': i
                    })

            return results

        except Exception as e:
            self.logger.error(f"❌ Sequential evaluation failed: {e}")
            return []

    def _safe_evaluate(self, objective_function: Callable, point: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Safely evaluate objective function.

        Args:
            objective_function: Function to evaluate
            point: Parameter dictionary
            index: Point index

        Returns:
            Evaluation result
        """
        try:
            value = objective_function(point)

            return {
                'params': point,
                'value': value,
                'index': index
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Point {index} evaluation failed: {e}")
            return {
                'params': point,
                'value': float('-inf'),
                'index': index
            }

    def get_grid_summary(self, parameter_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Get summary of grid search setup.

        Args:
            parameter_bounds: Dictionary of parameter bounds

        Returns:
            Dictionary with grid summary
        """
        try:
            n_points = self.config.n_points_per_dimension
            n_parameters = len(parameter_bounds)

            total_points = n_points ** n_parameters

            summary = {
                'n_parameters': n_parameters,
                'n_points_per_dimension': n_points,
                'total_grid_points': total_points,
                'parameter_bounds': parameter_bounds,
                'estimated_evaluation_time': self._estimate_evaluation_time(total_points),
                'memory_estimate_mb': self._estimate_memory_usage(total_points, n_parameters)
            }

            return summary

        except Exception as e:
            self.logger.warning(f"⚠️ Grid summary generation failed: {e}")
            return {'error': str(e)}

    def _estimate_evaluation_time(self, n_points: int) -> float:
        """
        Estimate evaluation time for grid search.

        Args:
            n_points: Number of grid points

        Returns:
            Estimated time in seconds
        """
        try:
            # Rough estimate: 0.1 seconds per evaluation
            base_time = n_points * 0.1

            # Adjust for parallel execution
            if self.config.enable_parallel:
                parallel_factor = min(self.config.max_workers, n_points)
                base_time /= parallel_factor

            return base_time

        except Exception as e:
            self.logger.warning(f"⚠️ Evaluation time estimation failed: {e}")
            return 0.0

    def _estimate_memory_usage(self, n_points: int, n_parameters: int) -> float:
        """
        Estimate memory usage for grid search.

        Args:
            n_points: Number of grid points
            n_parameters: Number of parameters

        Returns:
            Estimated memory usage in MB
        """
        try:
            # Rough estimate: each parameter value is 8 bytes (float64)
            memory_per_point = n_parameters * 8
            total_memory = n_points * memory_per_point

            # Convert to MB
            memory_mb = total_memory / (1024 * 1024)

            return memory_mb

        except Exception as e:
            self.logger.warning(f"⚠️ Memory usage estimation failed: {e}")
            return 0.0