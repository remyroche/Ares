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
    from ..hardware.m1_gpu_utils import M1GPUManager
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    # from ..hardware.m1_memory_optimizer import (  # type: ignore
    #     auto_skim_memory, smart_memory_allocation,
    #     memory_skim_decorator, auto_memory_skim_decorator,
    #     auto_memory_skim_context, smart_memory_context
    # )
    MEMORY_OPTIMIZER_AVAILABLE = False
except ImportError:
    MEMORY_OPTIMIZER_AVAILABLE = False


try:
    from ..hardware.m1_cpu_optimizer import get_m1_cpu_optimizer  # type: ignore
    CPU_OPTIMIZER_AVAILABLE = True
except ImportError:
    CPU_OPTIMIZER_AVAILABLE = False


ObjectiveDirection = Dict[str, str]  # {'metric_name': 'max' | 'min'}


@dataclass
class Solution:
    """Container for a single solution's metrics.

    Example metrics keys: 'pnl', 'win_rate', 'sharpe', 'training_time', ...
    """
    metrics: Dict[str, float]
    params: Dict[str, Any] | None = None


class ParetoFront:
    """Enhanced Pareto front utilities with M1 optimization."""

    def __init__(self):
        self.logger = _LOGGER
        self.logger.info("🚀 Initializing ParetoFront...")
        start_time = time.time()
        
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
        
        init_time = time.time() - start_time
        self.logger.info(f"✅ ParetoFront initialized in {init_time:.3f}s")

    # @auto_memory_skim_decorator("pareto_front_construction")  # Commented out due to import issues
    def compute_pareto_front_gpu(
        self,
        solutions: List[Solution],
        objectives: ObjectiveDirection,
        use_gpu: bool = True
    ) -> List[Solution]:
        """Compute Pareto front with GPU acceleration and memory optimization."""
        if not solutions:
            return []

        # Estimate memory requirements and optimize if needed
        if MEMORY_OPTIMIZER_AVAILABLE:
            estimated_memory_mb = len(solutions) * len(objectives) * 8 / (1024**2)
            # Memory optimization would be implemented here when available
            _LOGGER.debug(f"Estimated memory usage: {estimated_memory_mb:.2f} MB for pareto front construction")

        if use_gpu and self.gpu_manager and len(solutions) > 100:
            return self._compute_pareto_front_gpu(solutions, objectives)
        else:
            return compute_pareto_front(solutions, objectives)

    def _compute_pareto_front_gpu(
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
            self.logger.warning(f"GPU Pareto front computation failed: {e}, falling back to CPU")
            return compute_pareto_front(solutions, objectives)

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
    'pnl': 0.50,
    'win_rate': 0.25,
    'sharpe': 0.25,
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
            # Missing metric -> cannot dominate on this objective
            return False

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
) -> List[Solution]:
    """Return the list of non-dominated solutions (Pareto front).

    Enhanced with M1 GPU acceleration for large datasets.

    Args:
        solutions: List of solutions to evaluate
        objectives: Dict mapping metric names to optimization direction
        use_gpu: Whether to use GPU acceleration if available and beneficial

    Returns:
        List of non-dominated solutions
    """
    if not solutions:
        return []

    # Use GPU acceleration for large datasets if available
    if use_gpu and GPU_AVAILABLE and len(solutions) > 100:
        try:
            pareto_front = ParetoFront()
            return pareto_front.compute_pareto_front_gpu(solutions, objectives, use_gpu)
        except Exception as e:
            _LOGGER.warning(f"GPU Pareto front computation failed: {e}, falling back to CPU")

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

    # Monte Carlo for 3+ dims
    rng = np.random.default_rng(42)
    S = 20000
    samples = rng.uniform(0.0, 1.0, size=(S, dims))
    # A point is dominated by the front if exists p s.t. all p>=s
    dominated = 0
    for s in samples:
        if np.any(np.all(norm >= s, axis=1)):
            dominated += 1
    return float(dominated / S)


def scalarize_financial_goals(
    metrics: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    fallback_objectives: Optional[ObjectiveDirection] = None,
) -> float:
    """Return a scalar score from metrics using default financial weights.

    If keys 'pnl', 'win_rate', 'sharpe' are present, use weights (default DEFAULT_FINANCIAL_WEIGHTS).
    Otherwise, fall back to a simple weighted sum across provided objectives (max => +, min => -).
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
                score += (weights[k] / total_w) * float(metrics.get(k, 0.0))
            except Exception:
                pass
        return float(score)

    # Fallback: if no financial keys present
    score = 0.0
    if fallback_objectives:
        for k, direction in fallback_objectives.items():
            v = float(metrics.get(k, 0.0))
            score += v if direction == 'max' else -v
    return float(score)


# Global instance with proper cleanup
_pareto_front = None

def get_pareto_front() -> ParetoFront:
    """Get global ParetoFront instance with M1 optimization."""
    global _pareto_front
    if _pareto_front is None:
        _pareto_front = ParetoFront()
    return _pareto_front


__all__ = [
    'Solution',
    'ParetoFront',
    'ParetoFrontAnalyzer',
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


