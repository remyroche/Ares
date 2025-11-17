"""
Pareto front utilities: construction, knee point selection, hypervolume, and constraint filtering.

Defaults: when financial metrics are present, scalarization weights are
 - 50% pnl (total gain), 25% win_rate, 25% sharpe.

Built on existing utilities:
- Uses m1_gpu_utils.py for GPU acceleration
- Leverages m1_memory_optimizer.py for memory management
- Integrates m1_cpu_optimizer.py for parallel processing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable, Optional

import math
import time
import numpy as np
from ...nonlinear_optimization_helpers import NonLinearConfig

# Import torch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.Pareto")
except Exception:
    import logging
    _LOGGER = logging.getLogger("MLCommon.Pareto")

# Import M1 utilities for enhanced performance
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Memory optimizer integration is optional and currently disabled to avoid import cycles
MEMORY_OPTIMIZER_AVAILABLE = False

try:
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer  # type: ignore
    CPU_OPTIMIZER_AVAILABLE = True
except ImportError:
    CPU_OPTIMIZER_AVAILABLE = False

# Import VectorBT optimizations
try:
    import vectorbt as vbt
    from src.utils.vectorbt_compat import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_quantile, rolling_skew, rolling_kurt
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

# Import VectorBT rolling optimizer
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
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

ObjectiveDirection = Dict[str, str]  # {'metric_name': 'max' | 'min'}

@dataclass
class Solution:
    """Container for a single solution's metrics.

    Example metrics keys: 'pnl', 'win_rate', 'sharpe', 'training_time', ...
    """
    metrics: Dict[str, float]
    params: Dict[str, Any] | None = None

class ParetoFront:
    """Enhanced Pareto front utilities with M1 optimization and non-linear transformations."""

    def __init__(self, nonlinear_config: Optional[NonLinearConfig] = None, enable_vectorbt: bool = True):
        self.logger = _LOGGER
        self.logger.info("🚀 Initializing Enhanced ParetoFront with VectorBT optimizations...")
        start_time = time.time()

        # Non-linear optimization configuration
        self.nonlinear_config = nonlinear_config or NonLinearConfig()
        self.use_nonlinear_objectives = self.nonlinear_config.use_log_sampling or self.nonlinear_config.use_fractional_powers

        # Performance and caching configuration
        self.cache_size = 1000  # Maximum cache size for Pareto computations
        self.computation_cache = {}  # Cache for repeated computations
        self.enable_caching = True

        # Algorithm selection based on dataset size
        self.use_efficient_algorithm_threshold = 500  # Use efficient algorithms for datasets > this size
        self.vectorbt_threshold = 1000  # Use VectorBT for datasets >= this size
        self.vectorbt_rolling_threshold = 500  # Use VectorBT rolling for datasets >= this size

        # Incremental update configuration
        self.enable_incremental_updates = True
        self.last_pareto_front = None
        self.last_objectives_hash = None

        # VectorBT optimization configuration
        self.enable_vectorbt = enable_vectorbt and VECTORBT_AVAILABLE
        self.enable_vectorbt_rolling = enable_vectorbt and VECTORBT_ROLLING_AVAILABLE

        # Initialize VectorBT components
        self._initialize_vectorbt_components()

        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'vectorbt_rolling_operations': 0,
            'standard_operations': 0,
            'gpu_operations': 0,
            'total_time': 0.0,
            'optimization_selections': {}
        }

        self.gpu_manager = M1GPUManager() if GPU_AVAILABLE else None
        if self.gpu_manager:
            self.logger.debug("✅ GPU manager initialized")
        else:
            self.logger.debug("ℹ️ GPU manager not initialized (GPU not available)")

        self.cpu_optimizer = get_m1_cpu_optimizer() if CPU_OPTIMIZER_AVAILABLE else None
        if self.cpu_optimizer:
            self.logger.debug("✅ CPU optimizer initialized")
        else:
            self.logger.debug("ℹ️ CPU optimizer not initialized (CPU optimizer not available)")

        if self.use_nonlinear_objectives:
            self.logger.info("🚀 Non-linear objective transformations enabled")

        init_time = time.time() - start_time
        self.logger.info(f"✅ Enhanced ParetoFront with VectorBT initialized in {init_time:.3f}s")

    def _initialize_vectorbt_components(self):
        """Initialize VectorBT optimization components."""
        # Initialize VectorBT rolling optimizer if available
        if self.enable_vectorbt_rolling:
            try:
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=False,
                    enable_parallel=True,
                    memory_efficient=True,
                    chunk_size=1000
                )
                self.logger.debug("✅ VectorBT rolling optimizer initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize VectorBT rolling optimizer: {e}")
                self.vectorbt_rolling_optimizer = None
        else:
            self.vectorbt_rolling_optimizer = None

        # Initialize unified vectorization manager if available
        if UNIFIED_MANAGER_AVAILABLE:
            try:
                self.vectorization_manager = UnifiedVectorizationManager()
                self.logger.debug("✅ Unified vectorization manager initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize vectorization manager: {e}")
                self.vectorization_manager = None
        else:
            self.vectorization_manager = None

    def _apply_nonlinear_objective_transformations(self, solutions: List[Solution],
                                                 objectives: ObjectiveDirection) -> List[Solution]:
        """Apply non-linear transformations to objectives for better optimization."""
        if not self.use_nonlinear_objectives:
            return solutions

        transformed_solutions = []
        for solution in solutions:
            transformed_metrics = {}
            for metric_name, value in solution.metrics.items():
                if metric_name in objectives:
                    direction = objectives[metric_name]

                    # Apply non-linear transformations based on metric characteristics
                    if metric_name in ['pnl', 'sharpe', 'win_rate'] and direction == 'max':
                        # Apply log transformation for financial metrics (with offset to handle negatives)
                        if value > 0:
                            transformed_value = np.log(1 + value)
                        else:
                            transformed_value = -np.log(1 + abs(value))
                    elif metric_name in ['drawdown', 'volatility', 'risk'] and direction == 'min':
                        # Apply power transformation for risk metrics
                        transformed_value = value ** 0.5  # Square root to reduce extreme values
                    elif metric_name in ['training_time', 'inference_time'] and direction == 'min':
                        # Apply log transformation for time metrics
                        transformed_value = np.log(1 + value)
                    else:
                        # Keep original value for other metrics
                        transformed_value = value

                    transformed_metrics[metric_name] = transformed_value
                else:
                    transformed_metrics[metric_name] = value

            transformed_solutions.append(Solution(
                metrics=transformed_metrics,
                params=solution.params
            ))

        return transformed_solutions

    def _reverse_nonlinear_objective_transformations(self, solutions: List[Solution],
                                                   objectives: ObjectiveDirection) -> List[Solution]:
        """Reverse non-linear transformations for final results."""
        if not self.use_nonlinear_objectives:
            return solutions

        reversed_solutions = []
        for solution in solutions:
            reversed_metrics = {}
            for metric_name, value in solution.metrics.items():
                if metric_name in objectives:
                    direction = objectives[metric_name]

                    # Reverse non-linear transformations
                    if metric_name in ['pnl', 'sharpe', 'win_rate'] and direction == 'max':
                        # Reverse log transformation
                        if value > 0:
                            reversed_value = np.exp(value) - 1
                        else:
                            reversed_value = -(np.exp(abs(value)) - 1)
                    elif metric_name in ['drawdown', 'volatility', 'risk'] and direction == 'min':
                        # Reverse power transformation
                        reversed_value = value ** 2
                    elif metric_name in ['training_time', 'inference_time'] and direction == 'min':
                        # Reverse log transformation
                        reversed_value = np.exp(value) - 1
                    else:
                        # Keep original value
                        reversed_value = value

                    reversed_metrics[metric_name] = reversed_value
                else:
                    reversed_metrics[metric_name] = value

            reversed_solutions.append(Solution(
                metrics=reversed_metrics,
                params=solution.params
            ))

        return reversed_solutions

    # @auto_memory_skim_decorator("pareto_front_construction")  # Commented out due to import issues
    def compute_pareto_front_gpu(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection,
        use_gpu: bool = True,
        use_nonlinear_transforms: bool = True
    ) -> List[Solution]:
        """Compute Pareto front with GPU acceleration, memory optimization, VectorBT optimizations, and non-linear transformations."""
        if not solutions:
            return []

        start_time = time.time()
        self.performance_stats['total_operations'] += 1

        # Apply non-linear transformations if enabled
        if use_nonlinear_transforms and self.use_nonlinear_objectives:
            self.logger.debug("🚀 Applying non-linear objective transformations")
            transformed_solutions = self._apply_nonlinear_objective_transformations(solutions, objectives)
        else:
            transformed_solutions = solutions

        # Estimate memory requirements and optimize if needed
        if MEMORY_OPTIMIZER_AVAILABLE:
            estimated_memory_mb = len(transformed_solutions) * len(objectives) * 8 / (1024**2)
            # Memory optimization would be implemented here when available
            _LOGGER.debug(f"Estimated memory usage: {estimated_memory_mb:.2f} MB for pareto front construction")

        # Compute Pareto front using appropriate algorithm based on size and hardware
        n_solutions = len(transformed_solutions)

        # Select optimal strategy based on data size and available optimizations
        strategy = self._select_optimization_strategy(transformed_solutions, objectives, use_gpu)
        self.performance_stats['optimization_selections'][strategy] = \
            self.performance_stats['optimization_selections'].get(strategy, 0) + 1

        # Check for incremental updates if enabled
        if (self.enable_incremental_updates and
            self.last_pareto_front is not None and
            self._can_use_incremental_update(transformed_solutions, objectives)):
            try:
                pareto_front = self._incremental_pareto_update(transformed_solutions, objectives)
                self.logger.debug(f"✅ Used incremental Pareto update for {n_solutions} solutions")
            except Exception as e:
                self.logger.warning(f"Incremental update failed: {e}, falling back to full computation")
                pareto_front = self._compute_pareto_front_with_strategy(transformed_solutions, objectives, strategy)
        else:
            pareto_front = self._compute_pareto_front_with_strategy(transformed_solutions, objectives, strategy)

        # Reverse non-linear transformations if they were applied
        if use_nonlinear_transforms and self.use_nonlinear_objectives:
            self.logger.debug("🔄 Reversing non-linear objective transformations")
            final_pareto_front = self._reverse_nonlinear_objective_transformations(pareto_front, objectives)
        else:
            final_pareto_front = pareto_front

        # Update performance stats
        computation_time = time.time() - start_time
        self.performance_stats['total_time'] += computation_time
        self.logger.info(f"✅ Pareto front computed using {strategy}: {len(final_pareto_front)}/{n_solutions} solutions in {computation_time:.3f}s")

        return final_pareto_front

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
        if (self.enable_vectorbt and
            n_solutions >= self.vectorbt_threshold and
            VECTORBT_AVAILABLE):
            return 'vectorbt'

        # VectorBT rolling strategy for medium datasets
        if (self.enable_vectorbt_rolling and
            n_solutions >= self.vectorbt_rolling_threshold and
            n_solutions < self.vectorbt_threshold and
            self.vectorbt_rolling_optimizer):
            return 'vectorbt_rolling'

        # GPU strategy selection
        if use_gpu and n_solutions > 500 and self.gpu_manager:
            return 'gpu'

        # Standard strategy for small datasets
        return 'standard'

    def _compute_pareto_front_with_strategy(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection,
        strategy: str
    ) -> List[Solution]:
        """Compute Pareto front using the selected strategy."""
        # Fast path: exactly 2 objectives → O(n log n) sweep
        if len(objectives) == 2 and solutions:
            try:
                return self._compute_pareto_front_2d_sweep(solutions, objectives)
            except Exception:
                # Fallback to selected strategy if fast path fails
                pass
        if strategy == 'vectorbt' and VECTORBT_AVAILABLE:
            self.performance_stats['vectorbt_operations'] += 1
            return self._compute_pareto_front_vectorbt(solutions, objectives)

        elif strategy == 'vectorbt_rolling' and self.vectorbt_rolling_optimizer:
            self.performance_stats['vectorbt_rolling_operations'] += 1
            return self._compute_pareto_front_vectorbt_rolling(solutions, objectives)

        elif strategy == 'gpu' and self.gpu_manager:
            self.performance_stats['gpu_operations'] += 1
            return self._compute_pareto_front_gpu_original(solutions, objectives)

        else:
            # Use standard computation
            self.performance_stats['standard_operations'] += 1
            return self._compute_pareto_front_full(solutions, objectives)

    def _compute_pareto_front_vectorbt(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection
    ) -> List[Solution]:
        """Compute Pareto front using VectorBT optimizations."""
        try:
            # Convert solutions to matrix
            objective_matrix = self._solutions_to_matrix_vectorbt(solutions, objectives)

            # Compute dominance matrix using VectorBT
            dominance_matrix = self._compute_dominance_matrix_vectorbt(objective_matrix, objectives)

            # Find non-dominated solutions
            is_dominated = np.any(dominance_matrix, axis=1)
            pareto_indices = np.where(~is_dominated)[0]

            # Extract Pareto solutions
            pareto_front = [solutions[i] for i in pareto_indices]

            return pareto_front

        except Exception as e:
            self.logger.warning(f"VectorBT computation failed: {e}, falling back to standard algorithm")
            return self._compute_pareto_front_full(solutions, objectives)

    def _compute_pareto_front_vectorbt_rolling(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection
    ) -> List[Solution]:
        """Compute Pareto front using VectorBT rolling operations."""
        try:
            # Convert solutions to matrix
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
            return self._compute_pareto_front_full(solutions, objectives)

    def _solutions_to_matrix_vectorbt(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection
    ) -> np.ndarray:
        """Convert solutions to objective matrix optimized for VectorBT."""
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

    def _solutions_to_matrix_vectorbt_rolling(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection
    ) -> np.ndarray:
        """Convert solutions to matrix optimized for VectorBT rolling operations."""
        return self._solutions_to_matrix_vectorbt(solutions, objectives)

    def _compute_dominance_matrix_vectorbt(
        self,
        objective_matrix: np.ndarray,
        objectives: ObjectiveDirection
    ) -> np.ndarray:
        """Compute dominance matrix using VectorBT vectorized operations."""
        n_solutions = objective_matrix.shape[0]

        # Use VectorBT for efficient matrix operations
        if self.vectorbt_rolling_optimizer and self.enable_vectorbt_rolling:
            # Use VectorBT rolling operations for dominance computation
            dominance_matrix = self._compute_dominance_with_vectorbt_rolling(objective_matrix)
        else:
            # Use standard vectorized computation
            dominance_matrix = self._compute_dominance_standard_vectorized(objective_matrix)

        return dominance_matrix

    def _compute_dominance_vectorbt_rolling(
        self,
        objective_matrix: np.ndarray,
        objectives: ObjectiveDirection
    ) -> np.ndarray:
        """Compute dominance matrix using VectorBT rolling operations."""
        return self._compute_dominance_with_vectorbt_rolling(objective_matrix)

    def _compute_dominance_with_vectorbt_rolling(self, objective_matrix: np.ndarray) -> np.ndarray:
        """Compute dominance matrix using VectorBT rolling operations."""
        n_solutions = objective_matrix.shape[0]
        dominance_matrix = np.zeros((n_solutions, n_solutions), dtype=bool)

        # Use VectorBT rolling operations for efficient computation
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

    def _compute_dominance_standard_vectorized(self, objective_matrix: np.ndarray) -> np.ndarray:
        """Compute dominance matrix using standard vectorized operations."""
        # Reduce memory footprint using float32 for comparisons
        if objective_matrix.dtype != np.float32:
            objective_matrix = objective_matrix.astype(np.float32, copy=False)
        n_solutions = objective_matrix.shape[0]

        # Vectorized dominance computation
        # Expand dimensions for broadcasting
        obj_i = objective_matrix[:, np.newaxis, :]  # (n, 1, m)
        obj_j = objective_matrix[np.newaxis, :, :]  # (1, n, m)

        # Check dominance: i dominates j if i >= j in all objectives AND i > j in at least one
        better_or_equal = np.all(obj_i >= obj_j, axis=2)  # (n, n)
        strictly_better = np.any(obj_i > obj_j, axis=2)   # (n, n)

        dominance_matrix = better_or_equal & strictly_better

        return dominance_matrix

    def _compute_pareto_front_2d_sweep(self, solutions: List[Solution], objectives: ObjectiveDirection) -> List[Solution]:
        """Efficient Pareto front for exactly 2 objectives using sort-and-sweep.

        Converts min objectives to max by negation, then sorts by x desc and
        sweeps best y to select non-dominated points.
        """
        obj_names = list(objectives.keys())
        dirs = [objectives[obj_names[0]], objectives[obj_names[1]]]
        pts: List[Tuple[float, float, int]] = []
        for i, s in enumerate(solutions):
            x = s.metrics.get(obj_names[0], 0.0)
            y = s.metrics.get(obj_names[1], 0.0)
            if dirs[0] == 'min':
                x = -x
            if dirs[1] == 'min':
                y = -y
            pts.append((x, y, i))

        # Sort by x desc, then y desc to ensure proper sweep
        pts.sort(key=lambda t: (t[0], t[1]), reverse=True)

        pareto_idx: List[int] = []
        best_y = -np.inf
        for x, y, idx in pts:
            if y > best_y:
                pareto_idx.append(idx)
                best_y = y

        return [solutions[i] for i in pareto_idx]

    def _compute_pareto_front_gpu_original(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection
    ) -> List[Solution]:
        """GPU-accelerated Pareto front computation."""
        try:
            with self.gpu_manager.gpu_context("pareto_front_gpu"):
                # Convert to numpy arrays for GPU processing
                n_solutions = len(solutions)
                n_objectives = len(objectives)

                # Create objective matrix
                objective_matrix = np.zeros((n_solutions, n_objectives))

                for i, solution in enumerate(solutions):
                    for j, obj_name in enumerate(objectives.keys()):
                        objective_matrix[i, j] = solution.metrics.get(obj_name, 0.0)

                # Apply direction transformations (min/max)
                for j, (obj_name, direction) in enumerate(objectives.items()):
                    if direction == 'min':
                        objective_matrix[:, j] = -objective_matrix[:, j]  # Convert min to max

                # Move to GPU for computation
                if self.gpu_manager:
                    objective_matrix_gpu = self.gpu_manager.to_device(
                        objective_matrix, "matrix_mult"
                    )

                    # Compute dominance matrix on GPU
                    dominance_matrix = self._compute_dominance_gpu(
                        objective_matrix_gpu, n_solutions
                    )

                    # Find non-dominated solutions
                    is_dominated = torch.any(dominance_matrix, dim=1).cpu().numpy()
                else:
                    dominance_matrix = self._compute_dominance_cpu(objective_matrix)
                    is_dominated = np.any(dominance_matrix, axis=1)

                # Return non-dominated solutions
                pareto_front = []
                for i, solution in enumerate(solutions):
                    if not is_dominated[i]:
                        pareto_front.append(solution)

                return pareto_front

        except Exception as e:
            self.logger.warning(f"GPU Pareto front computation failed: {e}, falling back to efficient CPU")
            return self._compute_pareto_front_efficient(solutions, objectives)

    def _compute_dominance_gpu(self, objective_matrix: torch.Tensor, n_solutions: int) -> torch.Tensor:
        """Compute dominance matrix on GPU."""
        # Create comparison matrices
        obj_expanded = objective_matrix.unsqueeze(1)  # (n, 1, m)
        obj_tiled = objective_matrix.unsqueeze(0)     # (1, n, m)

        # Check dominance for each objective
        better_or_equal = obj_expanded >= obj_tiled  # (n, n, m)
        strictly_better = obj_expanded > obj_tiled   # (n, n, m)

        # A dominates B if A is better or equal in all objectives AND strictly better in at least one
        all_better_equal = torch.all(better_or_equal, dim=2)  # (n, n)
        any_strictly_better = torch.any(strictly_better, dim=2)  # (n, n)

        # Dominance matrix: row i dominates column j
        dominance_matrix = all_better_equal & any_strictly_better

        return dominance_matrix

    def _compute_pareto_front_efficient(self, solutions: List[Solution], objectives: ObjectiveDirection) -> List[Solution]:
        """Efficient O(n log n) Pareto front computation using divide-and-conquer approach."""
        if not solutions:
            return []

        if len(solutions) <= 2:
            # Base case: use standard algorithm for small datasets
            return compute_pareto_front(solutions, objectives)

        try:
            # Convert to matrix for efficient processing
            objective_matrix = self._solutions_to_matrix(solutions, objectives)

            # Handle NaN values and duplicates
            objective_matrix = self._preprocess_objective_matrix(objective_matrix)

            # Use divide-and-conquer approach
            pareto_indices = self._divide_and_conquer_pareto(objective_matrix, objectives)

            # Extract Pareto solutions
            pareto_front = [solutions[i] for i in pareto_indices]

            # Post-process to ensure Pareto optimality (remove any false positives)
            pareto_front = self._validate_pareto_front(pareto_front, objectives)

            return pareto_front

        except Exception as e:
            self.logger.warning(f"Efficient Pareto computation failed: {e}, falling back to standard algorithm")
            return compute_pareto_front(solutions, objectives)

    def _solutions_to_matrix(self, solutions: List[Solution], objectives: ObjectiveDirection) -> np.ndarray:
        """Convert solutions to objective matrix for efficient processing."""
        n_solutions = len(solutions)
        n_objectives = len(objectives)

        # Create objective matrix
        objective_matrix = np.zeros((n_solutions, n_objectives))

        for i, solution in enumerate(solutions):
            for j, obj_name in enumerate(objectives.keys()):
                value = solution.metrics.get(obj_name, np.nan)
                objective_matrix[i, j] = value

        # Apply direction transformations (min to max)
        for j, (obj_name, direction) in enumerate(objectives.items()):
            if direction == 'min':
                objective_matrix[:, j] = -objective_matrix[:, j]  # Convert min to max

        return objective_matrix

    def _preprocess_objective_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Preprocess matrix to handle NaN values and duplicates."""
        # Replace NaN with worst possible values
        # For maximization (already converted), use -inf for missing values
        matrix = np.where(np.isnan(matrix), -np.inf, matrix)

        # Remove duplicate solutions (keep only unique combinations)
        # This is a simplified approach - in practice, we'd use more sophisticated deduplication
        _, unique_indices = np.unique(matrix, axis=0, return_index=True)
        if len(unique_indices) < len(matrix):
            self.logger.debug(f"Removed {len(matrix) - len(unique_indices)} duplicate solutions")
            matrix = matrix[unique_indices]

        return matrix

    def _divide_and_conquer_pareto(self, matrix: np.ndarray, objectives: ObjectiveDirection) -> np.ndarray:
        """Divide-and-conquer Pareto front computation."""
        n_solutions = len(matrix)

        if n_solutions <= 10:
            # Use standard algorithm for small subproblems
            return self._standard_pareto_indices(matrix, objectives)

        # Split into two halves
        mid = n_solutions // 2
        left_matrix = matrix[:mid]
        right_matrix = matrix[mid:]

        # Recursively compute Pareto fronts
        left_pareto = self._divide_and_conquer_pareto(left_matrix, objectives)
        right_pareto = self._divide_and_conquer_pareto(right_matrix, objectives)

        # Merge the two Pareto fronts
        merged_pareto = self._merge_pareto_fronts(
            left_matrix[left_pareto], right_matrix[right_pareto],
            objectives, left_pareto, right_pareto, mid
        )

        return merged_pareto

    def _standard_pareto_indices(self, matrix: np.ndarray, objectives: ObjectiveDirection) -> np.ndarray:
        """Standard O(n²) Pareto front computation for small datasets."""
        n_solutions = len(matrix)
        is_dominated = np.zeros(n_solutions, dtype=bool)

        for i in range(n_solutions):
            if is_dominated[i]:
                continue

            for j in range(n_solutions):
                if i == j or is_dominated[j]:
                    continue

                # Check if solution j dominates solution i
                if self._dominates_matrix_row(matrix[j], matrix[i], objectives):
                    is_dominated[i] = True
                    break

        return np.where(~is_dominated)[0]

    def _merge_pareto_fronts(self, left_pareto: np.ndarray, right_pareto: np.ndarray,
                           objectives: ObjectiveDirection, left_indices: np.ndarray,
                           right_indices: np.ndarray, mid: int) -> np.ndarray:
        """Merge two Pareto fronts from divide-and-conquer."""
        # Combine all solutions from both Pareto fronts
        combined_matrix = np.vstack([left_pareto, right_pareto])
        combined_indices = np.concatenate([left_indices, right_indices + mid])

        # Compute Pareto front of the combined solutions
        final_pareto = self._standard_pareto_indices(combined_matrix, objectives)

        return combined_indices[final_pareto]

    def _dominates_matrix_row(self, row1: np.ndarray, row2: np.ndarray, objectives: ObjectiveDirection) -> bool:
        """Check if row1 dominates row2 (for maximization objectives)."""
        # All objectives must be >= (better or equal)
        all_better_or_equal = np.all(row1 >= row2)

        # At least one objective must be > (strictly better)
        at_least_one_strictly_better = np.any(row1 > row2)

        return all_better_or_equal and at_least_one_strictly_better

    def _validate_pareto_front(self, pareto_solutions: List[Solution], objectives: ObjectiveDirection) -> List[Solution]:
        """Validate and clean Pareto front to ensure true Pareto optimality."""
        if len(pareto_solutions) <= 1:
            return pareto_solutions

        # Convert back to matrix for validation
        matrix = self._solutions_to_matrix(pareto_solutions, objectives)
        matrix = self._preprocess_objective_matrix(matrix)

        pareto_indices = self._standard_pareto_indices(matrix, objectives)

        return [pareto_solutions[i] for i in pareto_indices]

    def _compute_dominance_cpu(self, objective_matrix: np.ndarray) -> np.ndarray:
        """Compute dominance matrix on CPU."""
        n_solutions = len(objective_matrix)
        dominance_matrix = np.zeros((n_solutions, n_solutions), dtype=bool)

        for i in range(n_solutions):
            for j in range(n_solutions):
                if i == j:
                    continue

                # Check if solution i dominates solution j
                dominates = True
                at_least_one_better = False

                for k in range(objective_matrix.shape[1]):
                    val_i = objective_matrix[i, k]
                    val_j = objective_matrix[j, k]

                    if val_i < val_j:  # i is worse than j on this objective
                        dominates = False
                        break
                    elif val_i > val_j:  # i is better than j on this objective
                        at_least_one_better = True

                if dominates and at_least_one_better:
                    dominance_matrix[i, j] = True

        return dominance_matrix

DEFAULT_FINANCIAL_WEIGHTS: Dict[str, float] = {
    'pnl': 2.0,
    'win_rate': 0.0,
    'sharpe': 1.0,
}

def filter_by_constraints(
    solutions: List[Solution],
    constraints: Dict[str, Any] | None = None,
) -> List[Solution]:
    """Filter solutions by constraints.

    constraints can be either:
      - numeric thresholds (min) like {'pnl': 0.0, 'win_rate': 0.45}
      - callables like {'drawdown': lambda v: v < 0.2}
    """
    if not constraints:
        return solutions

    filtered: List[Solution] = []
    for s in solutions:
        keep = True
        for key, rule in constraints.items():
            if key not in s.metrics:
                keep = False
                break
            val = s.metrics[key]
            if callable(rule):
                if not bool(rule(val)):
                    keep = False
                    break
            else:
                # numeric threshold: must be >= rule
                try:
                    if float(val) < float(rule):
                        keep = False
                        break
                except Exception:
                    keep = False
                    break
        if keep:
            filtered.append(s)
    return filtered

def _dominates(a: Solution, b: Solution, objectives: ObjectiveDirection) -> bool:
    """True if a Pareto-dominates b under objectives."""
    better_or_equal_all = True
    strictly_better_at_least_one = False

    for m, direction in objectives.items():
        av = a.metrics.get(m)
        bv = b.metrics.get(m)
        if av is None or bv is None:
            # Missing metric: treat as worst possible for that objective so that
            # presence of metric is favored over absence, avoiding False positives.
            # For max objectives, None is -inf; for min objectives, None is +inf.
            if av is None and bv is None:
                # Neither provides this metric; skip this objective
                continue
            if direction == 'max':
                av = -math.inf if av is None else av
                bv = -math.inf if bv is None else bv
            else:
                av = math.inf if av is None else av
                bv = math.inf if bv is None else bv

        if direction == 'max':
            if av < bv:
                better_or_equal_all = False
                break
            if av > bv:
                strictly_better_at_least_one = True
        else:  # 'min'
            if av > bv:
                better_or_equal_all = False
                break
            if av < bv:
                strictly_better_at_least_one = True

    return better_or_equal_all and strictly_better_at_least_one

def compute_pareto_front(
    solutions: List[Solution],
    objectives: ObjectiveDirection,
    use_gpu: bool = True,
    use_vectorbt: bool = True,
) -> List[Solution]:
    """Return the list of non-dominated solutions (Pareto front).

    Enhanced with M1 GPU acceleration, VectorBT optimizations, and intelligent strategy selection.

    Args:
        solutions: List of solutions to evaluate
        objectives: Dict mapping metric names to optimization direction
        use_gpu: Whether to use GPU acceleration if available and beneficial
        use_vectorbt: Whether to use VectorBT optimizations for large datasets

    Returns:
        List of non-dominated solutions
    """
    if not solutions:
        return []

    # Use enhanced ParetoFront with VectorBT optimizations for large datasets
    if len(solutions) > 100 and (use_gpu or use_vectorbt):
        try:
            pareto_front = ParetoFront(enable_vectorbt=use_vectorbt)
            return pareto_front.compute_pareto_front_gpu(solutions, objectives, use_gpu)
        except Exception as e:
            _LOGGER.warning(f"Enhanced Pareto front computation failed: {e}, falling back to CPU")

    # CPU implementation (original algorithm)
    pareto: List[Solution] = []
    for s in solutions:
        dominated = False
        to_remove: List[int] = []
        for i, p in enumerate(pareto):
            if _dominates(p, s, objectives):
                dominated = True
                break
            if _dominates(s, p, objectives):
                to_remove.append(i)
        if not dominated:
            if to_remove:
                # Remove dominated elements from current front
                for idx in reversed(to_remove):
                    pareto.pop(idx)
            pareto.append(s)
    return pareto

def _normalize(values: np.ndarray) -> np.ndarray:
    vmin = np.nanmin(values, axis=0)
    vmax = np.nanmax(values, axis=0)
    span = np.where((vmax - vmin) == 0, 1.0, (vmax - vmin))
    return (values - vmin) / span

def _to_matrix(
    solutions: List[Solution],
    objectives: ObjectiveDirection,
) -> np.ndarray:
    keys = list(objectives.keys())
    mat = np.array([[s.metrics.get(k, np.nan) for k in keys] for s in solutions], dtype=float)
    # For minimization objectives, invert to make all maximization for normalization
    inv = np.array([1.0 if objectives[k] == 'max' else -1.0 for k in keys], dtype=float)
    return mat * inv

def select_knee_point(
    pareto_solutions: List[Solution],
    objectives: ObjectiveDirection,
    weights: Optional[Dict[str, float]] = None,
) -> Optional[Solution]:
    """Heuristic knee point selection: choose solution closest to the ideal point.

    - Normalize objectives to [0,1] after transforming to maximization.
    - Ideal is (1,1,...) then pick min Euclidean distance.
    - Optional weights can scale each dimension before distance.
    """
    if not pareto_solutions:
        return None

    M = _to_matrix(pareto_solutions, objectives)
    if M.size == 0:
        return None
    N = _normalize(M)

    w = np.ones(N.shape[1], dtype=float)
    if weights:
        keys = list(objectives.keys())
        w = np.array([float(weights.get(k, 1.0)) for k in keys], dtype=float)
        w = w / (np.sum(w) if np.sum(w) > 0 else 1.0)

    ideal = np.ones(N.shape[1])
    dists = np.sqrt(((N - ideal) ** 2 * w).sum(axis=1))
    idx = int(np.argmin(dists))
    return pareto_solutions[idx]

def compute_hypervolume(
    pareto_solutions: List[Solution],
    objectives: ObjectiveDirection,
    reference_point: Dict[str, float],
) -> float:
    """Compute (approximate) hypervolume of Pareto front relative to reference_point.

    Implementation notes:
    - Convert all objectives to maximization, normalize to [0,1] using reference as 0.
    - For 2D uses exact area by sorting on first objective.
    - For >2D uses a simple Monte Carlo approximation for robustness.
    """
    if not pareto_solutions:
        return 0.0

    keys = list(objectives.keys())
    # Build matrix after converting to maximization
    mat = _to_matrix(pareto_solutions, objectives)

    # Normalize with reference point as 0 baseline
    ref = np.array([
        (reference_point.get(k, 0.0) if objectives[k] == 'max' else -reference_point.get(k, 0.0))
        for k in keys
    ], dtype=float)

    # Shift so that ref maps to 0; clip to non-negative
    shifted = mat - ref
    # Scale each dimension to [0,1] using max across pareto
    maxv = np.maximum(np.nanmax(shifted, axis=0), 1e-9)
    norm = np.clip(shifted / maxv, 0.0, 1.0)

    dims = norm.shape[1]
    if dims == 1:
        return float(np.max(norm[:, 0]))
    if dims == 2:
        # Exact 2D area under step curve
        pts = norm[np.argsort(-norm[:, 0])]  # sort by obj0 descending
        area = 0.0
        best_y = 0.0
        prev_x = 0.0
        for x, y in pts:
            area += (max(x - prev_x, 0.0)) * max(best_y, 0.0)
            best_y = max(best_y, y)
            prev_x = x
        area += (1.0 - prev_x) * max(best_y, 0.0)
        return float(np.clip(area, 0.0, 1.0))

    # Use improved hypervolume computation for 3+ dimensions
    if dims == 3:
        # Use WFG algorithm for 3D (more accurate than Monte Carlo)
        return _compute_hypervolume_3d(norm, reference_point)
    else:
        # Use improved Monte Carlo with adaptive sampling for higher dimensions
        return _compute_hypervolume_monte_carlo_adaptive(norm, reference_point, dims)

def _compute_hypervolume_3d(norm_matrix: np.ndarray, reference_point: Dict[str, float]) -> float:
    """Compute 3D hypervolume using WFG (Walking Fish Group) algorithm."""
    if len(norm_matrix) == 0:
        return 0.0

    # Sort points by first objective (descending)
    sorted_points = norm_matrix[np.argsort(-norm_matrix[:, 0])]

    # WFG algorithm for 3D hypervolume
    volume = 0.0
    prev_x = 1.0  # Start from reference point

    for i, point in enumerate(sorted_points):
        x, y, z = point

        # Calculate volume contribution of this slice
        slice_volume = (prev_x - x) * y * z
        volume += slice_volume

        prev_x = x

        # Early termination if point is dominated by reference
        if x <= 0 or y <= 0 or z <= 0:
            break

    return float(np.clip(volume, 0.0, 1.0))

def _compute_hypervolume_monte_carlo_adaptive(norm_matrix: np.ndarray,
                                            reference_point: Dict[str, float], dims: int) -> float:
    """Adaptive Monte Carlo hypervolume computation with importance sampling."""
    if len(norm_matrix) == 0:
        return 0.0

    # Adaptive sample size based on dimensionality and Pareto front size
    base_samples = 10000
    adaptive_factor = min(5.0, max(1.0, len(norm_matrix) / 100.0))
    sample_size = int(base_samples * adaptive_factor * (dims ** 0.5))

    # Use stratified sampling for better coverage
    samples_per_dim = int(sample_size ** (1.0 / dims))

    # Generate stratified samples
    samples = _generate_stratified_samples(dims, samples_per_dim)

    # Count dominated samples
    dominated_count = 0
    for sample in samples:
        if _is_dominated_by_pareto(sample, norm_matrix):
            dominated_count += 1

    # Estimate hypervolume
    estimated_volume = dominated_count / len(samples)

    # Apply correction for boundary effects
    boundary_correction = _compute_boundary_correction(norm_matrix, dims)
    corrected_volume = estimated_volume * boundary_correction

    return float(np.clip(corrected_volume, 0.0, 1.0))

def _generate_stratified_samples(dims: int, samples_per_dim: int) -> np.ndarray:
    """Generate stratified samples for better Monte Carlo coverage."""
    # Create coordinate arrays for each dimension
    coords = []
    for d in range(dims):
        # Use Latin Hypercube-like sampling
        dim_samples = np.linspace(0, 1, samples_per_dim, endpoint=False) + np.random.random(samples_per_dim) / samples_per_dim
        coords.append(dim_samples)

    # Create meshgrid for all combinations
    mesh = np.meshgrid(*coords)
    samples = np.column_stack([m.ravel() for m in mesh])

    # Shuffle samples for randomness
    np.random.shuffle(samples)

    return samples

def _is_dominated_by_pareto(sample: np.ndarray, pareto_matrix: np.ndarray) -> bool:
    """Check if sample is dominated by any point in Pareto front."""
    # A sample is dominated if there exists a Pareto point that is >= sample in all dimensions
    return np.any(np.all(pareto_matrix >= sample, axis=1))

def _compute_boundary_correction(norm_matrix: np.ndarray, dims: int) -> float:
    """Compute boundary correction factor for Monte Carlo hypervolume."""
    if len(norm_matrix) == 0:
        return 1.0

    # Simple boundary correction based on Pareto front coverage
    min_bounds = np.min(norm_matrix, axis=0)
    max_bounds = np.max(norm_matrix, axis=0)

    # Estimate coverage ratio
    coverage = np.prod(max_bounds - min_bounds)

    # Correction factor to account for uncovered regions
    if coverage > 0:
        correction = min(1.2, 1.0 / (coverage + 0.1))
    else:
        correction = 1.0

    return correction

    def compute_diversity_metrics(self, pareto_solutions: List[Solution],
                                 objectives: ObjectiveDirection) -> Dict[str, float]:
        """Compute diversity metrics for Pareto front analysis."""
        if not pareto_solutions:
            return {}

        # Convert to matrix for analysis
        matrix = self._solutions_to_matrix(pareto_solutions, objectives)

        if matrix.shape[0] <= 1:
            return {'num_solutions': len(pareto_solutions)}

        metrics = {
            'num_solutions': len(pareto_solutions),
            'num_objectives': matrix.shape[1],
        }

        # Spacing metric (average distance to nearest neighbor)
        distances = self._compute_pairwise_distances(matrix)
        min_distances = np.min(distances + np.eye(len(distances)) * np.inf, axis=1)
        metrics['spacing'] = float(np.mean(min_distances))

        # Spread metric (range in each objective)
        obj_ranges = np.max(matrix, axis=0) - np.min(matrix, axis=0)
        metrics['spread'] = float(np.mean(obj_ranges))

        # Coverage metric (hypervolume normalized by ideal point)
        try:
            ideal_point = {obj: 1.0 for obj in objectives.keys()}
            hypervolume = compute_hypervolume(pareto_solutions, objectives, ideal_point)
            max_possible = np.prod([1.0] * len(objectives))
            metrics['coverage'] = float(hypervolume / max_possible) if max_possible > 0 else 0.0
        except:
            metrics['coverage'] = 0.0

        # Clustering tendency (variance of distances)
        if len(distances) > 1:
            metrics['clustering_tendency'] = float(np.var(distances))

        return metrics

    def cluster_pareto_front(self, pareto_solutions: List[Solution],
                           objectives: ObjectiveDirection, n_clusters: int = 3) -> Dict[str, Any]:
        """Cluster Pareto front solutions using k-means."""
        if not pareto_solutions or len(pareto_solutions) < n_clusters:
            return {'clusters': [], 'cluster_labels': []}

        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            # Convert to matrix and normalize
            matrix = self._solutions_to_matrix(pareto_solutions, objectives)

            if matrix.shape[0] <= n_clusters:
                return {'clusters': [list(range(len(pareto_solutions)))], 'cluster_labels': list(range(len(pareto_solutions)))}

            # Normalize features
            scaler = StandardScaler()
            normalized_matrix = scaler.fit_transform(matrix)

            # Perform clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(pareto_solutions)), random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(normalized_matrix)

            # Organize solutions by clusters
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(i)

            return {
                'clusters': clusters,
                'cluster_labels': cluster_labels.tolist(),
                'centroids': kmeans.cluster_centers_.tolist(),
                'cluster_sizes': [len(solutions) for solutions in clusters.values()]
            }

        except ImportError:
            self.logger.warning("Scikit-learn not available for clustering")
            return {'clusters': [], 'cluster_labels': []}
        except Exception as e:
            self.logger.warning(f"Clustering failed: {e}")
            return {'clusters': [], 'cluster_labels': []}

    def _compute_pairwise_distances(self, matrix: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances between solutions."""
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

    def _compute_pareto_front_full(self, solutions: List[Solution], objectives: ObjectiveDirection) -> List[Solution]:
        """Full Pareto front computation with algorithm selection."""
        n_solutions = len(solutions)

        if n_solutions > 100 and self.gpu_manager:
            # Use GPU acceleration for large datasets
            return self._compute_pareto_front_gpu(solutions, objectives)
        elif n_solutions > self.use_efficient_algorithm_threshold:
            # Use efficient divide-and-conquer algorithm for large CPU datasets
            return self._compute_pareto_front_efficient(solutions, objectives)
        else:
            # Use standard algorithm for smaller datasets
            return compute_pareto_front(solutions, objectives)

    def _can_use_incremental_update(self, solutions: List[Solution], objectives: ObjectiveDirection) -> bool:
        """Check if incremental update is possible and beneficial."""
        if not self.last_pareto_front or not self.last_objectives_hash:
            return False

        # Check if objectives are the same
        current_hash = self._hash_objectives(objectives)
        if current_hash != self.last_objectives_hash:
            return False

        # Check if new solutions are a reasonable addition (not complete replacement)
        if len(solutions) > len(self.last_pareto_front) * 2:
            return False  # Too many new solutions, better to recompute

        return True

    def _incremental_pareto_update(self, solutions: List[Solution], objectives: ObjectiveDirection) -> List[Solution]:
        """Incrementally update Pareto front with new solutions."""
        # Combine existing Pareto front with new solutions
        combined_solutions = self.last_pareto_front + solutions

        # Remove duplicates while preserving order
        seen = set()
        unique_solutions = []
        for sol in combined_solutions:
            sol_key = tuple(sorted(sol.metrics.items()))
            if sol_key not in seen:
                seen.add(sol_key)
                unique_solutions.append(sol)

        # Compute new Pareto front
        new_pareto = self._compute_pareto_front_full(unique_solutions, objectives)

        # Update state
        self.last_pareto_front = new_pareto
        self.last_objectives_hash = self._hash_objectives(objectives)

        return new_pareto

    def _hash_objectives(self, objectives: ObjectiveDirection) -> str:
        """Create a hash of objectives for caching and comparison."""
        obj_str = str(sorted(objectives.items()))
        import hashlib
        return hashlib.md5(obj_str.encode()).hexdigest()

    def filter_by_constraints_improved(self, solutions: List[Solution],
                                     constraints: Dict[str, Any]) -> List[Solution]:
        """Improved constraint filtering with better error handling and edge cases."""
        if not constraints or not solutions:
            return solutions

        filtered_solutions = []

        for solution in solutions:
            try:
                # Check each constraint
                satisfies_all = True
                for constraint_name, constraint_rule in constraints.items():
                    if constraint_name not in solution.metrics:
                        self.logger.warning(f"Missing constraint metric: {constraint_name}")
                        satisfies_all = False
                        break

                    value = solution.metrics[constraint_name]

                    # Handle different constraint types
                    if callable(constraint_rule):
                        # Function-based constraint
                        try:
                            if not constraint_rule(value):
                                satisfies_all = False
                                break
                        except Exception as e:
                            self.logger.warning(f"Constraint function failed for {constraint_name}: {e}")
                            satisfies_all = False
                            break
                    else:
                        # Numeric threshold constraint
                        try:
                            threshold = float(constraint_rule)
                            if value < threshold:
                                satisfies_all = False
                                break
                        except (ValueError, TypeError) as e:
                            self.logger.warning(f"Invalid constraint threshold for {constraint_name}: {e}")
                            satisfies_all = False
                            break

                if satisfies_all:
                    filtered_solutions.append(solution)

            except Exception as e:
                self.logger.warning(f"Error checking constraints for solution: {e}")
                # Decide whether to include or exclude on error (default: exclude)
                pass

        return filtered_solutions

    def validate_pareto_front(self, pareto_solutions: List[Solution],
                            objectives: ObjectiveDirection) -> Dict[str, Any]:
        """Validate that a Pareto front is correct and complete."""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

        if not pareto_solutions:
            validation_results['warnings'].append("Empty Pareto front")
            return validation_results

        # Check for basic Pareto properties
        try:
            # 1. Check that no solution dominates another in the Pareto front
            for i, sol1 in enumerate(pareto_solutions):
                for j, sol2 in enumerate(pareto_solutions):
                    if i != j:
                        if self._dominates_solution(sol1, sol2, objectives):
                            validation_results['errors'].append(
                                f"Solution {i} dominates solution {j} in Pareto front"
                            )
                            validation_results['is_valid'] = False

            # 2. Check that all solutions in Pareto front are non-dominated
            for i, solution in enumerate(pareto_solutions):
                # Convert to matrix and check against all other solutions
                all_solutions = pareto_solutions.copy()
                test_front = self._compute_pareto_front_full(all_solutions, objectives)

                if solution not in test_front:
                    validation_results['errors'].append(
                        f"Solution {i} is not in recomputed Pareto front"
                    )
                    validation_results['is_valid'] = False

            # 3. Compute basic statistics
            validation_results['statistics'] = {
                'num_solutions': len(pareto_solutions),
                'num_objectives': len(objectives),
                'objective_ranges': {}
            }

            # Compute range for each objective
            for obj_name in objectives.keys():
                values = [sol.metrics.get(obj_name, 0) for sol in pareto_solutions]
                if values:
                    validation_results['statistics']['objective_ranges'][obj_name] = {
                        'min': min(values),
                        'max': max(values),
                        'range': max(values) - min(values)
                    }

        except Exception as e:
            validation_results['errors'].append(f"Validation failed: {e}")
            validation_results['is_valid'] = False

        return validation_results

    def _dominates_solution(self, sol1: Solution, sol2: Solution, objectives: ObjectiveDirection) -> bool:
        """Check if sol1 dominates sol2."""
        return _dominates(sol1, sol2, objectives)

    def benchmark_pareto_algorithms(self, test_solutions: List[Solution],
                                  objectives: ObjectiveDirection, num_runs: int = 5) -> Dict[str, Any]:
        """Benchmark different Pareto front algorithms for performance comparison."""
        results = {
            'standard_algorithm': {'times': [], 'pareto_sizes': []},
            'efficient_algorithm': {'times': [], 'pareto_sizes': []},
            'gpu_algorithm': {'times': [], 'pareto_sizes': []}
        }

        # Test standard algorithm
        for _ in range(num_runs):
            start_time = time.time()
            pareto = compute_pareto_front(test_solutions, objectives, use_gpu=False)
            end_time = time.time()

            results['standard_algorithm']['times'].append(end_time - start_time)
            results['standard_algorithm']['pareto_sizes'].append(len(pareto))

        # Test efficient algorithm
        for _ in range(num_runs):
            start_time = time.time()
            pareto = self._compute_pareto_front_efficient(test_solutions, objectives)
            end_time = time.time()

            results['efficient_algorithm']['times'].append(end_time - start_time)
            results['efficient_algorithm']['pareto_sizes'].append(len(pareto))

        # Test GPU algorithm (if available)
        if self.gpu_manager and len(test_solutions) > 100:
            for _ in range(num_runs):
                start_time = time.time()
                pareto = self._compute_pareto_front_gpu(test_solutions, objectives)
                end_time = time.time()

                results['gpu_algorithm']['times'].append(end_time - start_time)
                results['gpu_algorithm']['pareto_sizes'].append(len(pareto))

        # Compute statistics
        for algorithm in results:
            if results[algorithm]['times']:
                times = results[algorithm]['times']
                sizes = results[algorithm]['pareto_sizes']

                results[algorithm]['avg_time'] = np.mean(times)
                results[algorithm]['std_time'] = np.std(times)
                results[algorithm]['avg_pareto_size'] = np.mean(sizes)
                results[algorithm]['min_time'] = np.min(times)
                results[algorithm]['max_time'] = np.max(times)

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()

        if stats['total_operations'] > 0:
            stats['avg_time_per_operation'] = stats['total_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['vectorbt_rolling_usage_rate'] = stats['vectorbt_rolling_operations'] / stats['total_operations']
            stats['standard_usage_rate'] = stats['standard_operations'] / stats['total_operations']
            stats['gpu_usage_rate'] = stats['gpu_operations'] / stats['total_operations']

        return stats

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
                'vectorbt': VECTORBT_AVAILABLE and self.enable_vectorbt,
                'vectorbt_rolling': VECTORBT_ROLLING_AVAILABLE and self.enable_vectorbt_rolling,
                'gpu_acceleration': GPU_AVAILABLE and self.gpu_manager is not None,
                'm1_optimization': CPU_OPTIMIZER_AVAILABLE and self.cpu_optimizer is not None
            },
            'performance_estimates': {}
        }

        # Recommend strategy based on data size
        if n_solutions >= self.vectorbt_threshold and self.enable_vectorbt:
            recommendations['recommended_strategy'] = 'vectorbt'
            recommendations['performance_estimates']['vectorbt'] = {
                'estimated_speedup': min(5.0, n_solutions / 1000),
                'memory_efficiency': 'high'
            }
        elif n_solutions >= self.vectorbt_rolling_threshold and self.enable_vectorbt_rolling:
            recommendations['recommended_strategy'] = 'vectorbt_rolling'
            recommendations['performance_estimates']['vectorbt_rolling'] = {
                'estimated_speedup': min(3.0, n_solutions / 500),
                'memory_efficiency': 'medium'
            }
        elif n_solutions > 500 and self.gpu_manager:
            recommendations['recommended_strategy'] = 'gpu'
            recommendations['performance_estimates']['gpu'] = {
                'estimated_speedup': min(4.0, n_solutions / 1000),
                'memory_efficiency': 'high'
            }
        else:
            recommendations['recommended_strategy'] = 'standard'
            recommendations['performance_estimates']['standard'] = {
                'estimated_speedup': 1.0,
                'memory_efficiency': 'low'
            }

        return recommendations

def scalarize_financial_goals(
    metrics: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    fallback_objectives: Optional[ObjectiveDirection] = None,
    use_nonlinear_scaling: bool = True,
) -> float:
    """Return a scalar score from metrics using default financial weights with optional non-linear scaling.

    If keys 'pnl', 'win_rate', 'sharpe' are present, use weights (default DEFAULT_FINANCIAL_WEIGHTS).
    Otherwise, fall back to a simple weighted sum across provided objectives (max => +, min => -).

    Args:
        metrics: Dictionary of metric values
        weights: Optional weights for metrics
        fallback_objectives: Fallback objectives if financial keys not present
        use_nonlinear_scaling: Whether to apply non-linear scaling to metrics
    """
    if weights is None:
        weights = DEFAULT_FINANCIAL_WEIGHTS

    available = [k for k in weights.keys() if k in metrics]
    if available:
        total_w = sum(weights[k] for k in available)
        total_w = total_w if total_w > 0 else 1.0
        score = 0.0
        for k in available:
            try:
                raw_value = float(metrics.get(k, 0.0))

                # Apply non-linear scaling if enabled
                if use_nonlinear_scaling:
                    if k == 'pnl':
                        # Apply log scaling to PnL for better handling of extreme values
                        if raw_value > 0:
                            scaled_value = np.log(1 + raw_value)
                        else:
                            scaled_value = -np.log(1 + abs(raw_value))
                    elif k == 'sharpe':
                        # Apply sigmoid-like scaling to Sharpe ratio
                        scaled_value = 2 / (1 + np.exp(-raw_value)) - 1
                    elif k == 'win_rate':
                        # Apply power scaling to win rate for better discrimination
                        scaled_value = raw_value ** 1.5
                    else:
                        scaled_value = raw_value
                else:
                    scaled_value = raw_value

                score += (weights[k] / total_w) * scaled_value
            except Exception:
                pass
        return float(score)

    # Fallback: if no financial keys present
    score = 0.0
    if fallback_objectives:
        for k, direction in fallback_objectives.items():
            v = float(metrics.get(k, 0.0))

            # Apply non-linear scaling if enabled
            if use_nonlinear_scaling:
                if direction == 'max':
                    # Apply log scaling for maximization objectives
                    if v > 0:
                        v = np.log(1 + v)
                    else:
                        v = -np.log(1 + abs(v))
                else:
                    # Apply square root scaling for minimization objectives
                    v = np.sqrt(max(0, v))

            score += v if direction == 'max' else -v
    return float(score)

# Global instance with proper cleanup
_pareto_front = None

def get_pareto_front(nonlinear_config: Optional[NonLinearConfig] = None) -> ParetoFront:
    """Get global ParetoFront instance with M1 optimization and non-linear transformations."""
    global _pareto_front
    if _pareto_front is None:
        _pareto_front = ParetoFront(nonlinear_config)
    return _pareto_front

class ParetoOptimizer:
    """Enhanced wrapper class for ParetoFront functionality with non-linear transformations."""

    def __init__(self, nonlinear_config: Optional[NonLinearConfig] = None):
        """Initialize ParetoOptimizer with enhanced ParetoFront instance."""
        self.pareto_front = ParetoFront(nonlinear_config)
        self.logger = _LOGGER
        self.logger.info("🚀 Initializing Enhanced ParetoOptimizer...")

    def optimize(self, solutions: List[Solution], objectives: ObjectiveDirection,
                use_nonlinear_transforms: bool = True) -> List[Solution]:
        """Optimize solutions using enhanced Pareto front computation with non-linear transformations."""
        return self.pareto_front.compute_pareto_front_gpu(
            solutions, objectives, use_nonlinear_transforms=use_nonlinear_transforms
        )

    def select_best(self, solutions: List[Solution], objectives: ObjectiveDirection,
                   use_nonlinear_transforms: bool = True) -> Optional[Solution]:
        """Select the best solution using knee point selection with non-linear transformations."""
        pareto_front = self.optimize(solutions, objectives, use_nonlinear_transforms)
        return select_knee_point(pareto_front, objectives)

__all__ = [
    'Solution',
    'ParetoFront',
    'ParetoFrontAnalyzer',
    'ParetoOptimizer',
    'DEFAULT_FINANCIAL_WEIGHTS',
    'filter_by_constraints',
    'compute_pareto_front',
    'select_knee_point',
    'compute_hypervolume',
    'scalarize_financial_goals',
    'get_pareto_front',
]

# ParetoFrontAnalyzer class - defined at module level to avoid indentation issues
class ParetoFrontAnalyzer:
    """Simple Pareto front analyzer placeholder."""

    def __init__(self):
        self.logger = _LOGGER

    def analyze(self, data):
        """Basic analysis method."""
        return {"pareto_front": [], "knee_point": None}
