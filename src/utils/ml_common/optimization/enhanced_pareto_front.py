"""
Enhanced Pareto Front Utilities with VectorBT Integration

This module provides an enhanced interface that combines the original Pareto front
utilities with VectorBT optimizations, offering automatic optimization selection
and seamless integration with existing code.

Key features:
- Automatic optimization selection based on data size and available hardware
- Seamless integration with existing Pareto front utilities
- VectorBT acceleration for large datasets
- Fallback to original implementation for compatibility
- Performance monitoring and statistics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import time
from typing import Any, Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import logging

# Import original Pareto utilities
from .pareto import (
    Solution, ObjectiveDirection, ParetoFront, compute_pareto_front,
    select_knee_point, compute_hypervolume, scalarize_financial_goals,
    filter_by_constraints, DEFAULT_FINANCIAL_WEIGHTS
)

# VectorBT optimizations are now integrated into the main pareto.py file
VECTORBT_PARETO_AVAILABLE = False
VectorBTParetoOptimizer = None
VectorBTParetoConfig = None

# Import VectorBT availability from main pareto file
try:
    from .pareto import VECTORBT_AVAILABLE
except ImportError:
    VECTORBT_AVAILABLE = False

# Import VectorBT rolling optimizer
try:
    from ...feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    )
    VECTORBT_ROLLING_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_AVAILABLE = False
    VectorBTRollingOptimizer = None

# Import unified vectorization manager
try:
    from ..unified_vectorization_manager import UnifiedVectorizationManager, OperationType
    UNIFIED_MANAGER_AVAILABLE = True
except ImportError:
    UNIFIED_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EnhancedParetoConfig:
    """Configuration for enhanced Pareto front computation."""
    # Optimization selection
    auto_select_optimization: bool = True
    prefer_vectorbt: bool = True
    vectorbt_threshold: int = 1000  # Use VectorBT for datasets >= this size
    
    # VectorBT configuration
    vectorbt_config: Optional[VectorBTParetoConfig] = None
    enable_vectorbt_rolling: bool = True
    vectorbt_rolling_threshold: int = 500  # Use VectorBT rolling for datasets >= this size
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    enable_caching: bool = True
    
    # Fallback behavior
    fallback_on_error: bool = True
    log_performance: bool = True
    
    # Advanced optimizations
    enable_batch_processing: bool = True
    batch_size: int = 1000
    enable_memory_optimization: bool = True
    enable_gpu_acceleration: bool = False


class EnhancedParetoFront:
    """
    Enhanced Pareto front utilities with automatic optimization selection.
    
    This class provides a unified interface that automatically selects the best
    optimization strategy based on data size, available hardware, and performance
    requirements.
    """
    
    def __init__(self, config: Optional[EnhancedParetoConfig] = None):
        """Initialize enhanced Pareto front utilities."""
        self.config = config or EnhancedParetoConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'standard_operations': 0,
            'fallback_operations': 0,
            'total_time': 0.0,
            'optimization_selections': {},
            'error_count': 0
        }
        
        self.logger.info("🚀 Enhanced Pareto Front initialized")
    
    def _initialize_components(self):
        """Initialize optimization components."""
        # Initialize VectorBT optimizer if available
        if VECTORBT_PARETO_AVAILABLE and self.config.prefer_vectorbt:
            try:
                self.vectorbt_optimizer = get_vectorbt_pareto_optimizer(self.config.vectorbt_config)
                self.logger.info("✅ VectorBT Pareto optimizer initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize VectorBT optimizer: {e}")
                self.vectorbt_optimizer = None
        else:
            self.vectorbt_optimizer = None
        
        # Initialize VectorBT rolling optimizer if available
        if VECTORBT_ROLLING_AVAILABLE and self.config.enable_vectorbt_rolling:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=self.config.enable_gpu_acceleration,
                    enable_parallel=True,
                    memory_efficient=self.config.enable_memory_optimization,
                    chunk_size=self.config.batch_size
                )
                self.logger.info("✅ VectorBT rolling optimizer initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize VectorBT rolling optimizer: {e}")
                self.vectorbt_rolling_optimizer = None
        else:
            self.vectorbt_rolling_optimizer = None
        
        # Initialize unified vectorization manager
        if UNIFIED_MANAGER_AVAILABLE:
            try:
                self.vectorization_manager = UnifiedVectorizationManager()
                self.logger.info("✅ Unified vectorization manager initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize vectorization manager: {e}")
                self.vectorization_manager = None
        else:
            self.vectorization_manager = None
    
    def compute_pareto_front(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection,
        use_gpu: bool = True,
        use_nonlinear_transforms: bool = True,
        force_optimization: Optional[str] = None
    ) -> List[Solution]:
        """
        Compute Pareto front with automatic optimization selection.
        
        Args:
            solutions: List of solutions to evaluate
            objectives: Dictionary mapping metric names to optimization direction
            use_gpu: Whether to use GPU acceleration if available
            use_nonlinear_transforms: Whether to use non-linear transformations
            force_optimization: Force specific optimization ('vectorbt', 'standard', 'gpu')
            
        Returns:
            List of non-dominated solutions (Pareto front)
        """
        if not solutions:
            return []
        
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        
        try:
            # Select optimization strategy
            if force_optimization:
                strategy = force_optimization
            elif self.config.auto_select_optimization:
                strategy = self._select_optimization_strategy(solutions, objectives, use_gpu)
            else:
                strategy = 'standard'
            
            # Execute with selected strategy
            pareto_front = self._execute_pareto_computation(
                strategy, solutions, objectives, use_gpu, use_nonlinear_transforms
            )
            
            # Update performance stats
            computation_time = time.time() - start_time
            self.performance_stats['total_time'] += computation_time
            self.performance_stats['optimization_selections'][strategy] = \
                self.performance_stats['optimization_selections'].get(strategy, 0) + 1
            
            if self.config.log_performance:
                self.logger.info(f"✅ Pareto front computed using {strategy}: "
                               f"{len(pareto_front)}/{len(solutions)} solutions in {computation_time:.3f}s")
            
            return pareto_front
            
        except Exception as e:
            self.performance_stats['error_count'] += 1
            self.logger.error(f"❌ Pareto front computation failed: {e}")
            
            if self.config.fallback_on_error:
                self.logger.warning("⚠️ Falling back to standard implementation")
                return self._fallback_computation(solutions, objectives, use_gpu)
            else:
                raise
    
    def _select_optimization_strategy(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection,
        use_gpu: bool
    ) -> str:
        """Select optimal optimization strategy based on data characteristics."""
        n_solutions = len(solutions)
        n_objectives = len(objectives)
        
        # VectorBT strategy selection for large datasets
        if (n_solutions >= self.config.vectorbt_threshold and
            self.config.prefer_vectorbt and
            VECTORBT_AVAILABLE):
            return 'vectorbt'
        
        # VectorBT rolling strategy for medium datasets
        if (n_solutions >= self.config.vectorbt_rolling_threshold and
            n_solutions < self.config.vectorbt_threshold and
            VECTORBT_ROLLING_AVAILABLE):
            return 'vectorbt_rolling'
        
        # GPU strategy selection
        if use_gpu and n_solutions > 500:
            return 'gpu'
        
        # Standard strategy for small datasets
        return 'standard'
    
    def _execute_pareto_computation(
        self,
        strategy: str,
        solutions: List[Solution],
        objectives: ObjectiveDirection,
        use_gpu: bool,
        use_nonlinear_transforms: bool
    ) -> List[Solution]:
        """Execute Pareto front computation with selected strategy."""
        if strategy == 'vectorbt' or strategy == 'vectorbt_rolling':
            # Use enhanced ParetoFront with VectorBT optimizations
            self.performance_stats['vectorbt_operations'] += 1
            pareto_front = ParetoFront(enable_vectorbt=True)
            return pareto_front.compute_pareto_front_gpu(solutions, objectives, use_gpu, use_nonlinear_transforms)
        
        elif strategy == 'gpu':
            # Use GPU-accelerated computation
            self.performance_stats['standard_operations'] += 1
            return self._compute_pareto_front_gpu(solutions, objectives, use_nonlinear_transforms)
        
        else:
            # Use standard computation
            self.performance_stats['standard_operations'] += 1
            return compute_pareto_front(solutions, objectives, use_gpu)
    
    def _compute_pareto_front_gpu(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection,
        use_nonlinear_transforms: bool
    ) -> List[Solution]:
        """Compute Pareto front with GPU acceleration."""
        # Use the original ParetoFront class with GPU support
        pareto_front = ParetoFront()
        return pareto_front.compute_pareto_front_gpu(
            solutions, objectives, use_gpu=True, use_nonlinear_transforms=use_nonlinear_transforms
        )
    
    def _compute_pareto_front_vectorbt_rolling(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection
    ) -> List[Solution]:
        """Compute Pareto front using VectorBT rolling operations for medium datasets."""
        if not self.vectorbt_rolling_optimizer:
            return compute_pareto_front(solutions, objectives, use_gpu=False)
        
        try:
            # Convert solutions to matrix for VectorBT operations
            objective_matrix = self._solutions_to_matrix_vectorbt_rolling(solutions, objectives)
            
            # Use VectorBT rolling operations for dominance computation
            dominance_matrix = self._compute_dominance_vectorbt_rolling(objective_matrix, objectives)
            
            # Find non-dominated solutions
            is_dominated = np.any(dominance_matrix, axis=1)
            pareto_indices = np.where(~is_dominated)[0]
            
            # Extract Pareto solutions
            pareto_front = [solutions[i] for i in pareto_indices]
            
            return pareto_front
            
        except Exception as e:
            self.logger.warning(f"VectorBT rolling computation failed: {e}, using fallback")
            return compute_pareto_front(solutions, objectives, use_gpu=False)
    
    def _solutions_to_matrix_vectorbt_rolling(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection
    ) -> np.ndarray:
        """Convert solutions to matrix optimized for VectorBT rolling operations."""
        n_solutions = len(solutions)
        n_objectives = len(objectives)
        
        # Create objective matrix
        objective_matrix = np.zeros((n_solutions, n_objectives), dtype=np.float64)
        
        for i, solution in enumerate(solutions):
            for j, obj_name in enumerate(objectives.keys()):
                value = solution.metrics.get(obj_name, np.nan)
                objective_matrix[i, j] = value
        
        # Apply direction transformations (min to max for VectorBT)
        for j, (obj_name, direction) in enumerate(objectives.items()):
            if direction == 'min':
                objective_matrix[:, j] = -objective_matrix[:, j]
        
        # Handle NaN values
        objective_matrix = np.where(np.isnan(objective_matrix), -np.inf, objective_matrix)
        
        return objective_matrix
    
    def _compute_dominance_vectorbt_rolling(
        self,
        objective_matrix: np.ndarray,
        objectives: ObjectiveDirection
    ) -> np.ndarray:
        """Compute dominance matrix using VectorBT rolling operations."""
        n_solutions = objective_matrix.shape[0]
        dominance_matrix = np.zeros((n_solutions, n_solutions), dtype=bool)
        
        # Use VectorBT rolling operations for efficient dominance computation
        for i in range(n_solutions):
            solution_i = objective_matrix[i:i+1, :]  # Shape: (1, n_objectives)
            
            # Use VectorBT rolling operations to compare with all other solutions
            for j in range(n_solutions):
                if i == j:
                    continue
                
                solution_j = objective_matrix[j:j+1, :]  # Shape: (1, n_objectives)
                
                # Check if solution i dominates solution j
                # A dominates B if A >= B in all objectives AND A > B in at least one
                better_or_equal = np.all(solution_i >= solution_j)
                strictly_better = np.any(solution_i > solution_j)
                
                dominance_matrix[i, j] = better_or_equal and strictly_better
        
        return dominance_matrix
    
    def _fallback_computation(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection,
        use_gpu: bool
    ) -> List[Solution]:
        """Fallback computation using standard implementation."""
        self.performance_stats['fallback_operations'] += 1
        return compute_pareto_front(solutions, objectives, use_gpu)
    
    def compute_hypervolume(
        self,
        pareto_solutions: List[Solution],
        objectives: ObjectiveDirection,
        reference_point: Dict[str, float],
        use_vectorbt: bool = True
    ) -> float:
        """Compute hypervolume with automatic optimization selection."""
        if not pareto_solutions:
            return 0.0
        
        try:
            if (use_vectorbt and self.vectorbt_optimizer and 
                len(pareto_solutions) >= self.config.vectorbt_threshold):
                return self.vectorbt_optimizer.compute_hypervolume_vectorbt(
                    pareto_solutions, objectives, reference_point
                )
            elif (use_vectorbt and self.vectorbt_rolling_optimizer and 
                  len(pareto_solutions) >= self.config.vectorbt_rolling_threshold):
                return self._compute_hypervolume_vectorbt_rolling(
                    pareto_solutions, objectives, reference_point
                )
            else:
                return compute_hypervolume(pareto_solutions, objectives, reference_point)
        except Exception as e:
            self.logger.warning(f"Hypervolume computation failed: {e}, using fallback")
            return compute_hypervolume(pareto_solutions, objectives, reference_point)
    
    def _compute_hypervolume_vectorbt_rolling(
        self,
        pareto_solutions: List[Solution],
        objectives: ObjectiveDirection,
        reference_point: Dict[str, float]
    ) -> float:
        """Compute hypervolume using VectorBT rolling operations."""
        if not self.vectorbt_rolling_optimizer:
            return compute_hypervolume(pareto_solutions, objectives, reference_point)
        
        try:
            # Convert to matrix
            objective_matrix = self._solutions_to_matrix_vectorbt_rolling(pareto_solutions, objectives)
            
            # Use VectorBT rolling operations for hypervolume computation
            n_solutions, n_objectives = objective_matrix.shape
            
            if n_objectives == 1:
                # 1D case
                return float(np.max(objective_matrix[:, 0]))
            elif n_objectives == 2:
                # 2D case - use VectorBT rolling operations
                return self._compute_hypervolume_2d_vectorbt_rolling(objective_matrix, reference_point)
            else:
                # Higher dimensions - use Monte Carlo with VectorBT
                return self._compute_hypervolume_monte_carlo_vectorbt_rolling(objective_matrix, reference_point)
                
        except Exception as e:
            self.logger.warning(f"VectorBT rolling hypervolume computation failed: {e}, using fallback")
            return compute_hypervolume(pareto_solutions, objectives, reference_point)
    
    def _compute_hypervolume_2d_vectorbt_rolling(
        self,
        objective_matrix: np.ndarray,
        reference_point: Dict[str, float]
    ) -> float:
        """Compute 2D hypervolume using VectorBT rolling operations."""
        # Sort by first objective (descending)
        sorted_indices = np.argsort(-objective_matrix[:, 0])
        sorted_matrix = objective_matrix[sorted_indices]
        
        # Compute area using VectorBT rolling operations
        area = 0.0
        prev_x = 1.0  # Reference point
        
        for i in range(len(sorted_matrix)):
            x, y = sorted_matrix[i]
            area += (prev_x - x) * y
            prev_x = x
        
        return float(max(0.0, area))
    
    def _compute_hypervolume_monte_carlo_vectorbt_rolling(
        self,
        objective_matrix: np.ndarray,
        reference_point: Dict[str, float]
    ) -> float:
        """Compute hypervolume using Monte Carlo with VectorBT rolling operations."""
        n_solutions, n_objectives = objective_matrix.shape
        
        # Generate random samples
        n_samples = min(10000, 1000 * n_objectives)
        samples = np.random.random((n_samples, n_objectives))
        
        # Count dominated samples using VectorBT rolling operations
        dominated_count = 0
        for sample in samples:
            # Check if sample is dominated by any Pareto point
            is_dominated = np.any(np.all(objective_matrix >= sample, axis=1))
            if is_dominated:
                dominated_count += 1
        
        # Estimate hypervolume
        estimated_volume = dominated_count / n_samples
        return float(estimated_volume)
    
    def select_knee_point(
        self,
        pareto_solutions: List[Solution],
        objectives: ObjectiveDirection,
        weights: Optional[Dict[str, float]] = None,
        use_vectorbt: bool = True
    ) -> Optional[Solution]:
        """Select knee point with automatic optimization selection."""
        if not pareto_solutions:
            return None
        
        try:
            if (use_vectorbt and self.vectorbt_optimizer and 
                len(pareto_solutions) >= self.config.vectorbt_threshold):
                return self.vectorbt_optimizer.select_knee_point_vectorbt(
                    pareto_solutions, objectives, weights
                )
            elif (use_vectorbt and self.vectorbt_rolling_optimizer and 
                  len(pareto_solutions) >= self.config.vectorbt_rolling_threshold):
                return self._select_knee_point_vectorbt_rolling(
                    pareto_solutions, objectives, weights
                )
            else:
                return select_knee_point(pareto_solutions, objectives, weights)
        except Exception as e:
            self.logger.warning(f"Knee point selection failed: {e}, using fallback")
            return select_knee_point(pareto_solutions, objectives, weights)
    
    def _select_knee_point_vectorbt_rolling(
        self,
        pareto_solutions: List[Solution],
        objectives: ObjectiveDirection,
        weights: Optional[Dict[str, float]] = None
    ) -> Optional[Solution]:
        """Select knee point using VectorBT rolling operations."""
        if not self.vectorbt_rolling_optimizer:
            return select_knee_point(pareto_solutions, objectives, weights)
        
        try:
            # Convert to matrix
            objective_matrix = self._solutions_to_matrix_vectorbt_rolling(pareto_solutions, objectives)
            
            # Normalize objectives using VectorBT rolling operations
            normalized_matrix = self._normalize_objectives_vectorbt_rolling(objective_matrix, objectives)
            
            # Compute distances to ideal point
            ideal_point = np.ones(normalized_matrix.shape[1])
            distances = np.linalg.norm(normalized_matrix - ideal_point, axis=1)
            
            # Apply weights if provided
            if weights:
                weight_array = np.array([weights.get(obj, 1.0) for obj in objectives.keys()])
                weighted_distances = distances * weight_array
                best_index = np.argmin(weighted_distances)
            else:
                best_index = np.argmin(distances)
            
            return pareto_solutions[best_index]
            
        except Exception as e:
            self.logger.warning(f"VectorBT rolling knee point selection failed: {e}, using fallback")
            return select_knee_point(pareto_solutions, objectives, weights)
    
    def _normalize_objectives_vectorbt_rolling(
        self,
        objective_matrix: np.ndarray,
        objectives: ObjectiveDirection
    ) -> np.ndarray:
        """Normalize objectives using VectorBT rolling operations."""
        # Compute min and max for each objective
        min_values = np.min(objective_matrix, axis=0)
        max_values = np.max(objective_matrix, axis=0)
        
        # Avoid division by zero
        ranges = np.where(max_values - min_values == 0, 1.0, max_values - min_values)
        
        # Normalize to [0, 1]
        normalized = (objective_matrix - min_values) / ranges
        
        return normalized
    
    def scalarize_financial_goals(
        self,
        metrics: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
        fallback_objectives: Optional[ObjectiveDirection] = None,
        use_nonlinear_scaling: bool = True
    ) -> float:
        """Scalarize financial goals with enhanced optimization."""
        return scalarize_financial_goals(
            metrics, weights, fallback_objectives, use_nonlinear_scaling
        )
    
    def filter_by_constraints(
        self,
        solutions: List[Solution],
        constraints: Dict[str, Any]
    ) -> List[Solution]:
        """Filter solutions by constraints with enhanced performance."""
        return filter_by_constraints(solutions, constraints)
    
    def benchmark_optimizations(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection,
        trials: int = 3
    ) -> Dict[str, Any]:
        """Benchmark different optimization strategies."""
        if not solutions:
            return {}
        
        results = {}
        strategies = ['standard']
        
        if self.vectorbt_optimizer:
            strategies.append('vectorbt')
        
        if VECTORBT_PARETO_AVAILABLE:
            strategies.append('gpu')
        
        for strategy in strategies:
            strategy_times = []
            strategy_sizes = []
            
            for trial in range(trials):
                try:
                    start_time = time.time()
                    pareto_front = self._execute_pareto_computation(
                        strategy, solutions, objectives, use_gpu=True, use_nonlinear_transforms=True
                    )
                    end_time = time.time()
                    
                    strategy_times.append(end_time - start_time)
                    strategy_sizes.append(len(pareto_front))
                    
                except Exception as e:
                    self.logger.warning(f"Strategy {strategy} failed in trial {trial}: {e}")
                    continue
            
            if strategy_times:
                results[strategy] = {
                    'avg_time': np.mean(strategy_times),
                    'std_time': np.std(strategy_times),
                    'avg_pareto_size': np.mean(strategy_sizes),
                    'trials_completed': len(strategy_times)
                }
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['total_operations'] > 0:
            stats['avg_time_per_operation'] = stats['total_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['standard_usage_rate'] = stats['standard_operations'] / stats['total_operations']
            stats['fallback_usage_rate'] = stats['fallback_operations'] / stats['total_operations']
            stats['error_rate'] = stats['error_count'] / stats['total_operations']
        
        # Add VectorBT stats if available
        if self.vectorbt_optimizer:
            vectorbt_stats = self.vectorbt_optimizer.get_performance_stats()
            stats['vectorbt_details'] = vectorbt_stats
        
        return stats
    
    def compute_diversity_metrics(
        self,
        pareto_solutions: List[Solution],
        objectives: ObjectiveDirection,
        use_vectorbt: bool = True
    ) -> Dict[str, float]:
        """Compute diversity metrics for Pareto front analysis using VectorBT optimizations."""
        if not pareto_solutions:
            return {}
        
        try:
            if (use_vectorbt and self.vectorbt_rolling_optimizer and 
                len(pareto_solutions) >= self.config.vectorbt_rolling_threshold):
                return self._compute_diversity_metrics_vectorbt_rolling(pareto_solutions, objectives)
            else:
                return self._compute_diversity_metrics_standard(pareto_solutions, objectives)
        except Exception as e:
            self.logger.warning(f"Diversity metrics computation failed: {e}, using fallback")
            return self._compute_diversity_metrics_standard(pareto_solutions, objectives)
    
    def _compute_diversity_metrics_vectorbt_rolling(
        self,
        pareto_solutions: List[Solution],
        objectives: ObjectiveDirection
    ) -> Dict[str, float]:
        """Compute diversity metrics using VectorBT rolling operations."""
        # Convert to matrix
        objective_matrix = self._solutions_to_matrix_vectorbt_rolling(pareto_solutions, objectives)
        
        metrics = {
            'num_solutions': len(pareto_solutions),
            'num_objectives': objective_matrix.shape[1],
        }
        
        if objective_matrix.shape[0] <= 1:
            return metrics
        
        # Spacing metric (average distance to nearest neighbor)
        distances = self._compute_pairwise_distances_vectorbt_rolling(objective_matrix)
        min_distances = np.min(distances + np.eye(len(distances)) * np.inf, axis=1)
        metrics['spacing'] = float(np.mean(min_distances))
        
        # Spread metric (range in each objective)
        obj_ranges = np.max(objective_matrix, axis=0) - np.min(objective_matrix, axis=0)
        metrics['spread'] = float(np.mean(obj_ranges))
        
        # Coverage metric (hypervolume normalized by ideal point)
        try:
            ideal_point = {obj: 1.0 for obj in objectives.keys()}
            hypervolume = self.compute_hypervolume(pareto_solutions, objectives, ideal_point)
            max_possible = np.prod([1.0] * len(objectives))
            metrics['coverage'] = float(hypervolume / max_possible) if max_possible > 0 else 0.0
        except:
            metrics['coverage'] = 0.0
        
        # Clustering tendency (variance of distances)
        if len(distances) > 1:
            metrics['clustering_tendency'] = float(np.var(distances))
        
        return metrics
    
    def _compute_diversity_metrics_standard(
        self,
        pareto_solutions: List[Solution],
        objectives: ObjectiveDirection
    ) -> Dict[str, float]:
        """Compute diversity metrics using standard implementation."""
        # Convert to matrix
        objective_matrix = self._solutions_to_matrix_vectorbt_rolling(pareto_solutions, objectives)
        
        metrics = {
            'num_solutions': len(pareto_solutions),
            'num_objectives': objective_matrix.shape[1],
        }
        
        if objective_matrix.shape[0] <= 1:
            return metrics
        
        # Basic diversity metrics
        obj_ranges = np.max(objective_matrix, axis=0) - np.min(objective_matrix, axis=0)
        metrics['spread'] = float(np.mean(obj_ranges))
        
        return metrics
    
    def _compute_pairwise_distances_vectorbt_rolling(self, matrix: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances using VectorBT rolling operations."""
        # Normalize matrix first for fair distance computation
        normalized = (matrix - np.min(matrix, axis=0)) / (np.max(matrix, axis=0) - np.min(matrix, axis=0) + 1e-8)
        
        # Compute pairwise distances
        distances = np.zeros((len(matrix), len(matrix)))
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                dist = np.linalg.norm(normalized[i] - normalized[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def get_optimization_recommendations(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection
    ) -> Dict[str, Any]:
        """Get optimization recommendations based on data characteristics."""
        n_solutions = len(solutions)
        n_objectives = len(objectives)
        
        recommendations = {
            'data_size': n_solutions,
            'num_objectives': n_objectives,
            'recommended_strategy': 'standard',
            'optimization_available': {
                'vectorbt_pareto': VECTORBT_PARETO_AVAILABLE and self.vectorbt_optimizer is not None,
                'vectorbt_rolling': VECTORBT_ROLLING_AVAILABLE and self.vectorbt_rolling_optimizer is not None,
                'gpu_acceleration': self.config.enable_gpu_acceleration,
                'batch_processing': self.config.enable_batch_processing
            },
            'performance_estimates': {}
        }
        
        # Recommend strategy based on data size
        if n_solutions >= self.config.vectorbt_threshold and self.vectorbt_optimizer:
            recommendations['recommended_strategy'] = 'vectorbt'
            recommendations['performance_estimates']['vectorbt'] = {
                'estimated_speedup': min(5.0, n_solutions / 1000),
                'memory_efficiency': 'high' if self.config.enable_memory_optimization else 'medium'
            }
        elif n_solutions >= self.config.vectorbt_rolling_threshold and self.vectorbt_rolling_optimizer:
            recommendations['recommended_strategy'] = 'vectorbt_rolling'
            recommendations['performance_estimates']['vectorbt_rolling'] = {
                'estimated_speedup': min(3.0, n_solutions / 500),
                'memory_efficiency': 'medium'
            }
        else:
            recommendations['recommended_strategy'] = 'standard'
            recommendations['performance_estimates']['standard'] = {
                'estimated_speedup': 1.0,
                'memory_efficiency': 'low'
            }
        
        return recommendations
    
    def cleanup(self):
        """Clean up resources."""
        if self.vectorbt_optimizer:
            self.vectorbt_optimizer.cleanup()
        
        if self.vectorbt_rolling_optimizer:
            self.vectorbt_rolling_optimizer.cleanup()
        
        if self.vectorization_manager:
            # Cleanup if available
            pass
        
        self.logger.info("✅ Enhanced Pareto Front cleaned up")


# Convenience functions for backward compatibility
def compute_pareto_front_enhanced(
    solutions: List[Solution],
    objectives: ObjectiveDirection,
    use_gpu: bool = True,
    config: Optional[EnhancedParetoConfig] = None
) -> List[Solution]:
    """Enhanced Pareto front computation with automatic optimization."""
    enhanced_pareto = EnhancedParetoFront(config)
    return enhanced_pareto.compute_pareto_front(solutions, objectives, use_gpu)


def compute_hypervolume_enhanced(
    pareto_solutions: List[Solution],
    objectives: ObjectiveDirection,
    reference_point: Dict[str, float],
    use_vectorbt: bool = True,
    config: Optional[EnhancedParetoConfig] = None
) -> float:
    """Enhanced hypervolume computation with automatic optimization."""
    enhanced_pareto = EnhancedParetoFront(config)
    return enhanced_pareto.compute_hypervolume(pareto_solutions, objectives, reference_point, use_vectorbt)


def select_knee_point_enhanced(
    pareto_solutions: List[Solution],
    objectives: ObjectiveDirection,
    weights: Optional[Dict[str, float]] = None,
    use_vectorbt: bool = True,
    config: Optional[EnhancedParetoConfig] = None
) -> Optional[Solution]:
    """Enhanced knee point selection with automatic optimization."""
    enhanced_pareto = EnhancedParetoFront(config)
    return enhanced_pareto.select_knee_point(pareto_solutions, objectives, weights, use_vectorbt)


# Global enhanced Pareto front instance
_global_enhanced_pareto = None

def get_enhanced_pareto_front(config: Optional[EnhancedParetoConfig] = None) -> EnhancedParetoFront:
    """Get global enhanced Pareto front instance."""
    global _global_enhanced_pareto
    if _global_enhanced_pareto is None:
        _global_enhanced_pareto = EnhancedParetoFront(config)
    return _global_enhanced_pareto


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing Enhanced Pareto Front with VectorBT Integration...")
    
    # Create sample solutions
    solutions = []
    for i in range(2000):
        solution = Solution(
            metrics={
                'pnl': np.random.randn() * 1000,
                'win_rate': np.random.random(),
                'sharpe': np.random.randn() * 2,
                'drawdown': abs(np.random.randn()) * 100
            },
            params={'param1': i, 'param2': i * 2}
        )
        solutions.append(solution)
    
    objectives = {
        'pnl': 'max',
        'win_rate': 'max',
        'sharpe': 'max',
        'drawdown': 'min'
    }
    
    # Test enhanced Pareto front with VectorBT optimizations
    config = EnhancedParetoConfig(
        auto_select_optimization=True,
        prefer_vectorbt=True,
        vectorbt_threshold=1000,
        vectorbt_rolling_threshold=500,
        enable_vectorbt_rolling=True,
        enable_batch_processing=True,
        enable_memory_optimization=True,
        enable_gpu_acceleration=False
    )
    
    enhanced_pareto = EnhancedParetoFront(config)
    
    # Get optimization recommendations
    recommendations = enhanced_pareto.get_optimization_recommendations(solutions, objectives)
    print(f"📊 Optimization recommendations: {recommendations}")
    
    # Compute Pareto front
    pareto_front = enhanced_pareto.compute_pareto_front(solutions, objectives)
    print(f"✅ Pareto front computed: {len(pareto_front)}/{len(solutions)} solutions")
    
    # Compute hypervolume
    reference_point = {'pnl': 0.0, 'win_rate': 0.0, 'sharpe': 0.0, 'drawdown': 1000.0}
    hypervolume = enhanced_pareto.compute_hypervolume(pareto_front, objectives, reference_point)
    print(f"✅ Hypervolume computed: {hypervolume:.4f}")
    
    # Select knee point
    knee_point = enhanced_pareto.select_knee_point(pareto_front, objectives)
    print(f"✅ Knee point selected: {knee_point.metrics if knee_point else None}")
    
    # Compute diversity metrics
    diversity_metrics = enhanced_pareto.compute_diversity_metrics(pareto_front, objectives)
    print(f"📊 Diversity metrics: {diversity_metrics}")
    
    # Benchmark optimizations
    benchmark_results = enhanced_pareto.benchmark_optimizations(solutions, objectives)
    print(f"📊 Benchmark results: {benchmark_results}")
    
    # Print performance stats
    stats = enhanced_pareto.get_performance_stats()
    print(f"📊 Performance stats: {stats}")
    
    # Test different strategies
    print("\n🧪 Testing different optimization strategies...")
    
    # Test VectorBT strategy
    if enhanced_pareto.vectorbt_optimizer:
        vectorbt_pareto = enhanced_pareto.compute_pareto_front(solutions, objectives, force_optimization='vectorbt')
        print(f"✅ VectorBT strategy: {len(vectorbt_pareto)} solutions")
    
    # Test VectorBT rolling strategy
    if enhanced_pareto.vectorbt_rolling_optimizer:
        rolling_pareto = enhanced_pareto.compute_pareto_front(solutions, objectives, force_optimization='vectorbt_rolling')
        print(f"✅ VectorBT rolling strategy: {len(rolling_pareto)} solutions")
    
    # Test standard strategy
    standard_pareto = enhanced_pareto.compute_pareto_front(solutions, objectives, force_optimization='standard')
    print(f"✅ Standard strategy: {len(standard_pareto)} solutions")
    
    # Cleanup
    enhanced_pareto.cleanup()
    
    print("🎉 Enhanced Pareto Front with VectorBT Integration test completed!")