"""
Unified Clustering Optimization Goals Configuration - Predictive & Economic Focus

This module defines optimization goals for clustering with focus on:
1. Rolling/blocked predictive log-likelihood (33%)
2. One-step-ahead log-likelihood/predictive density (33%)
3. Out-of-sample economic utility (Sharpe, risk-adjusted) (33%)

Key Features:
- Time series cross-validation with rolling folds
- Metric normalization (z-score and rank-scaling)
- Pareto front optimization for multi-objective tradeoffs
- Soft constraints and penalties to avoid pathological fits
- Robustness checks and statistical validation
- Integration with VectorBT, hardware optimization, and ML utilities

Enhanced Features:
- Gradual Duration Penalties: Prevents noise flips by penalizing short episodes
  - Very high penalty for 1-2 bar episodes (50.0 per episode)
  - High penalty for 3-4 bar episodes (15.0 per episode)
  - No penalty for 5-6 bar episodes (0.0 per episode)
  - No penalty for 7+ bar episodes
- Smooth Transitions: Optimizes for gradual regime changes instead of abrupt jumps
  - Evaluates transition probabilities (if soft labels available)
  - Penalizes abrupt transitions (low transition probability)
  - Encourages smooth regime transitions
- Noise Handling: Optimizes noise point assignment
  - Penalizes high noise ratios (>10% default)
  - Penalizes unassigned noise points
  - Encourages proper noise point assignment to nearest cluster or neutral class

Used by:
- iterative_optimization.py
- iterative_optimization_tuner.py
- hdbscan_clustering optimization
- regime_clustering_step.py
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Callable, Any, Union
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
import logging

# sklearn metrics import (used in robustness validation)
try:
    from sklearn.metrics import adjusted_rand_score
    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    SKLEARN_METRICS_AVAILABLE = False
    adjusted_rand_score = None

# Numba imports for JIT compilation
try:
    from numba import njit, prange
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range
    numba = None

# VectorBT imports for efficient computations
try:
    from src.utils.vectorbt_compat import (
        vbt, rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, VECTORBT_AVAILABLE
    )
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None

# VectorBT optimization tools for enhanced performance
try:
    from src.feature_generation.utils.statistical_calculations_optimizer import (
        StatisticalCalculationsOptimizer,
        StatisticalOperationType
    )
    STAT_OPTIMIZER_AVAILABLE = True
except ImportError:
    STAT_OPTIMIZER_AVAILABLE = False
    StatisticalCalculationsOptimizer = None
    StatisticalOperationType = None

try:
    from src.feature_generation.utils.consolidated_rolling_optimizer import (
        ConsolidatedRollingOptimizer,
        RollingOperationType
    )
    ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    ROLLING_OPTIMIZER_AVAILABLE = False
    ConsolidatedRollingOptimizer = None
    RollingOperationType = None

try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer
    )
    VECTORBT_ROLLING_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_AVAILABLE = False
    VectorBTRollingOptimizer = None

# Import Pareto front utilities
try:
    from src.utils.ml_common.optimization.pareto import ParetoFront, Solution
    PARETO_AVAILABLE = True
except ImportError:
    PARETO_AVAILABLE = False
    ParetoFront = None
    Solution = None

# Import matrix cross-validation
try:
    from src.utils.ml_common.matrix_cross_validation import MatrixCrossValidator
    MATRIX_CV_AVAILABLE = True
except ImportError:
    MATRIX_CV_AVAILABLE = False
    MatrixCrossValidator = None

# Import feature normalization/scaling
try:
    from src.features_common.transforms.vectorbt_scaler import VectorBTScaler
    from src.features_common.transforms.scaling_normalization import (
        zscore_normalize, rank_normalize, robust_normalize
    )
    SCALING_AVAILABLE = True
except ImportError:
    try:
        from sklearn.preprocessing import StandardScaler, RobustScaler
        SCALING_AVAILABLE = True
    except ImportError:
        SCALING_AVAILABLE = False

# Import hardware optimization
try:
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    from src.utils.hardware.m1_gpu_utils import M1GPUManager
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False

from src.utils.tprint import tprint_debug

logger = logging.getLogger(__name__)


# ===== VECTORBT OPTIMIZATION WRAPPERS =====

# Global optimizer instances (initialized lazily)
_stat_optimizer = None
_rolling_optimizer = None
_vectorbt_rolling_optimizer = None


def _get_stat_optimizer():
    """Get or create StatisticalCalculationsOptimizer instance."""
    global _stat_optimizer
    if _stat_optimizer is None and STAT_OPTIMIZER_AVAILABLE:
        try:
            _stat_optimizer = StatisticalCalculationsOptimizer()
            logger.debug("StatisticalCalculationsOptimizer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize StatisticalCalculationsOptimizer: {e}")
            return None
    return _stat_optimizer


def _get_rolling_optimizer():
    """Get or create ConsolidatedRollingOptimizer instance."""
    global _rolling_optimizer
    if _rolling_optimizer is None and ROLLING_OPTIMIZER_AVAILABLE:
        try:
            _rolling_optimizer = ConsolidatedRollingOptimizer()
            logger.debug("ConsolidatedRollingOptimizer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize ConsolidatedRollingOptimizer: {e}")
            return None
    return _rolling_optimizer


def _get_vectorbt_rolling_optimizer():
    """Get or create VectorBTRollingOptimizer instance."""
    global _vectorbt_rolling_optimizer
    if _vectorbt_rolling_optimizer is None and VECTORBT_ROLLING_AVAILABLE:
        try:
            _vectorbt_rolling_optimizer = VectorBTRollingOptimizer()
            logger.debug("VectorBTRollingOptimizer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize VectorBTRollingOptimizer: {e}")
            return None
    return _vectorbt_rolling_optimizer


def calculate_variance_hybrid(data: np.ndarray, use_vectorbt: bool = True) -> float:
    """
    Calculate variance using VectorBT if available, fallback to Numba.

    Args:
        data: Data array (N, D) or (N,)
        use_vectorbt: Try VectorBT first if True

    Returns:
        Variance value
    """
    if use_vectorbt:
        stat_opt = _get_stat_optimizer()
        if stat_opt is not None:
            try:
                # VectorBT path
                if data.ndim == 1:
                    return float(stat_opt.calculate_variance(data, batch_mode=False))
                else:
                    # Calculate variance per feature and sum
                    total_var = 0.0
                    for d in range(data.shape[1]):
                        total_var += stat_opt.calculate_variance(data[:, d], batch_mode=False)
                    return float(total_var)
            except Exception as e:
                logger.debug(f"VectorBT variance calculation failed, using JIT fallback: {e}")

    # Fallback to standard numpy
    return float(np.var(data))


def calculate_rolling_mean_hybrid(data: np.ndarray, window: int, use_vectorbt: bool = True) -> np.ndarray:
    """
    Calculate rolling mean using VectorBT if available, fallback to pandas.

    Args:
        data: Data array (N,)
        window: Rolling window size
        use_vectorbt: Try VectorBT first if True

    Returns:
        Rolling mean array
    """
    if use_vectorbt:
        rolling_opt = _get_rolling_optimizer()
        if rolling_opt is not None:
            try:
                # VectorBT path
                result = rolling_opt.calculate_single(
                    data=pd.Series(data),
                    operation=RollingOperationType.MEAN,
                    window=window
                )
                if result is not None:
                    return result.values
            except Exception as e:
                logger.debug(f"VectorBT rolling mean failed, using pandas fallback: {e}")

    # Fallback to pandas
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values


# ===== NUMBA-OPTIMIZED HELPER FUNCTIONS =====

@njit(cache=True)
def _calculate_within_cluster_variance_jit(data: np.ndarray, labels: np.ndarray, n_clusters: int) -> float:
    """
    JIT-compiled within-cluster variance calculation.

    Args:
        data: Feature matrix (N, D)
        labels: Cluster labels (N,)
        n_clusters: Number of clusters

    Returns:
        Within-cluster variance
    """
    n_samples, n_features = data.shape
    within_var = 0.0

    for k in range(n_clusters):
        # Get cluster mask
        cluster_mask = labels == k
        cluster_size = np.sum(cluster_mask)

        if cluster_size == 0:
            continue

        # Calculate cluster centroid
        centroid = np.zeros(n_features)
        for i in range(n_samples):
            if cluster_mask[i]:
                for j in range(n_features):
                    centroid[j] += data[i, j]
        centroid /= cluster_size

        # Calculate variance
        for i in range(n_samples):
            if cluster_mask[i]:
                for j in range(n_features):
                    diff = data[i, j] - centroid[j]
                    within_var += diff * diff

    return within_var / n_samples


@njit(cache=True)
def _calculate_between_cluster_variance_jit(data: np.ndarray, labels: np.ndarray, n_clusters: int) -> float:
    """
    JIT-compiled between-cluster variance calculation.

    Args:
        data: Feature matrix (N, D)
        labels: Cluster labels (N,)
        n_clusters: Number of clusters

    Returns:
        Between-cluster variance
    """
    n_samples, n_features = data.shape

    # Global centroid
    global_centroid = np.zeros(n_features)
    for i in range(n_samples):
        for j in range(n_features):
            global_centroid[j] += data[i, j]
    global_centroid /= n_samples

    # Between-cluster variance
    between_var = 0.0

    for k in range(n_clusters):
        # Get cluster mask and size
        cluster_mask = labels == k
        cluster_size = np.sum(cluster_mask)

        if cluster_size == 0:
            continue

        # Calculate cluster centroid
        centroid = np.zeros(n_features)
        for i in range(n_samples):
            if cluster_mask[i]:
                for j in range(n_features):
                    centroid[j] += data[i, j]
        centroid /= cluster_size

        # Add contribution
        for j in range(n_features):
            diff = centroid[j] - global_centroid[j]
            between_var += cluster_size * diff * diff

    return between_var / n_samples


@njit(cache=True)
def _calculate_temporal_smoothness_jit(labels: np.ndarray) -> float:
    """
    JIT-compiled temporal smoothness calculation.

    Measures stability of regime assignments over time.
    High smoothness = few transitions.

    Args:
        labels: Regime labels (T,)

    Returns:
        Smoothness score [0, 1], higher is better
    """
    n_samples = len(labels)
    if n_samples <= 1:
        return 1.0

    # Count transitions
    n_transitions = 0
    for i in range(1, n_samples):
        if labels[i] != labels[i-1]:
            n_transitions += 1

    # Normalize: 0 transitions = 1.0, all transitions = 0.0
    max_transitions = n_samples - 1
    smoothness = 1.0 - (n_transitions / max_transitions)

    return smoothness


@njit(cache=True)
def _calculate_episode_durations_jit(labels: np.ndarray) -> np.ndarray:
    """
    JIT-compiled episode duration calculation.

    Args:
        labels: Regime labels (T,)

    Returns:
        Array of episode durations
    """
    n_samples = len(labels)
    if n_samples == 0:
        return np.zeros(0, dtype=np.int64)

    # Pre-allocate (worst case: all different)
    durations_temp = np.zeros(n_samples, dtype=np.int64)
    n_episodes = 0

    current_label = labels[0]
    current_duration = 1

    for i in range(1, n_samples):
        if labels[i] == current_label:
            current_duration += 1
        else:
            durations_temp[n_episodes] = current_duration
            n_episodes += 1
            current_label = labels[i]
            current_duration = 1

    # Add final episode
    durations_temp[n_episodes] = current_duration
    n_episodes += 1

    # Return only filled portion
    return durations_temp[:n_episodes]


@njit(cache=True, parallel=True)
def _calculate_sharpe_ratio_jit(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    JIT-compiled Sharpe ratio calculation.

    Args:
        returns: Return series (T,)
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sharpe ratio
    """
    n = len(returns)
    if n == 0:
        return 0.0

    # Calculate mean
    mean_return = 0.0
    for i in prange(n):
        mean_return += returns[i]
    mean_return /= n

    # Calculate std
    variance = 0.0
    for i in prange(n):
        diff = returns[i] - mean_return
        variance += diff * diff
    variance /= n
    std_return = np.sqrt(variance)

    if std_return == 0.0:
        return 0.0

    sharpe = (mean_return / std_return) * np.sqrt(float(periods_per_year))

    return sharpe


# ===== CONSTANTS =====

class Constants:
    """Constants used throughout the optimization module."""
    LOG_LIKELIHOOD_MIN = -50.0
    LOG_LIKELIHOOD_MAX = 50.0
    TRADING_DAYS_PER_YEAR = 252
    MIN_BLOCK_SIZE = 1  # Minimum block size for bootstrap


class OptimizationGoal(Enum):
    """Core clustering optimization goals - Predictive & Economic Focus."""
    
    # PRIMARY GOALS (33% each)
    ROLLING_LOG_LIKELIHOOD = "rolling_log_likelihood"  # Rolling/blocked predictive LL
    ONE_STEP_LOG_LIKELIHOOD = "one_step_log_likelihood"  # One-step-ahead predictive density
    ECONOMIC_UTILITY = "economic_utility"  # OOS Sharpe / risk-adjusted metric
    
    # LEGACY GOALS (for backward compatibility, low weight)
    CV_SCORE = "cv_score"  # Between/Within Variance Ratio
    SILHOUETTE = "silhouette_score"  # Cluster cohesion
    DBI = "dbi_score"  # Davies-Bouldin Index
    
    # CONSTRAINT METRICS
    BALANCE = "balance_score"  # Cluster size balance
    TEMPORAL_SMOOTHNESS = "temporal_smoothness"  # Temporal stability


class OptimizationObjective(Enum):
    """Optimization direction for each goal."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    MAINTAIN = "maintain"  # Soft constraint


class NormalizationMethod(Enum):
    """Methods for normalizing metrics across trials."""
    ZSCORE = "zscore"  # z = (value - mean) / std
    RANK = "rank"  # Rank-based normalization [0, 1]
    ROBUST_ZSCORE = "robust_zscore"  # (value - median) / MAD
    MIN_MAX = "minmax"  # (value - min) / (max - min)


class CVStrategy(Enum):
    """Cross-validation strategies for time series."""
    ROLLING = "rolling"  # Rolling window (train=18mo, val=3mo)
    EXPANDING = "expanding"  # Expanding window
    BLOCKED = "blocked"  # Non-overlapping blocks
    PURGED = "purged"  # Purged CV (for financial data)


@dataclass
class GoalConfig:
    """Configuration for a single optimization goal."""
    name: str
    objective: OptimizationObjective
    weight: float  # Importance weight in composite score
    target_range: Tuple[float, float]  # (min, max) acceptable values
    constraint_threshold: Optional[float] = None  # Hard constraint if specified
    description: str = ""
    enable_normalization: bool = True  # Whether to normalize this metric
    normalization_method: NormalizationMethod = NormalizationMethod.RANK
    stability_threshold: float = 0.4  # std/mean < threshold for stable metric
    sub_weights: Optional[Dict[str, float]] = None  # Optional sub-component weights


@dataclass
class CVConfig:
    """Configuration for time series cross-validation."""
    n_splits: int = 5  # Number of CV folds (K=5-10 recommended)
    strategy: CVStrategy = CVStrategy.ROLLING
    train_months: int = 18  # Training window (for rolling CV)
    val_months: int = 3  # Validation window
    min_train_samples: int = 200  # Minimum samples for training
    min_val_samples: int = 50  # Minimum samples for validation
    overlap: bool = False  # Allow temporal overlap (use purged CV if True)
    shuffle: bool = False  # Never shuffle time series
    random_state: int = 42
    
    # Robustness settings
    n_bootstrap_samples: int = 100  # For statistical significance tests
    bootstrap_alpha: float = 0.10  # Significance threshold for Sharpe improvement
    stability_check: bool = True  # Check metric stability across folds


@dataclass
class PenaltyConfig:
    """Configuration for penalties to avoid pathological fits."""
    
    # Minimum occupancy penalty
    min_occupancy_pct: float = 0.01  # 1% minimum (penalize if below)
    min_occupancy_penalty: float = 10.0  # Large penalty for violation
    
    # Minimum expected duration penalty (gradual penalty structure)
    min_duration_bars: int = 7  # For daily data (tunable) - legacy threshold
    min_duration_penalty: float = 5.0  # Legacy penalty (kept for backward compatibility)
    
    # Gradual duration penalties (prevents noise flips)
    # Very high penalty for 1-2 bars, high for 3-4, low for 5-6, none above 6
    duration_penalty_1_2_bars: float = 50.0  # Very high penalty for 1-2 bar episodes
    duration_penalty_3_4_bars: float = 15.0  # High penalty for 3-4 bar episodes
    duration_penalty_5_6_bars: float = 0.0   # No penalty for 5-6 bar episodes
    duration_penalty_threshold: int = 6     # No penalty above this threshold
    
    # Turnover penalty
    max_monthly_turnover: float = 4.0  # Max regime switches per month
    turnover_penalty: float = 3.0
    
    # Stability penalty (ARI across restarts)
    min_ari_stability: float = 0.5  # Median ARI across restarts
    stability_penalty: float = 5.0
    
    # Calibration penalty (CRPS / PIT)
    max_calibration_error: float = 0.2  # Acceptable calibration error
    calibration_penalty: float = 4.0
    
    # Metric stability penalty
    max_cv_variation: float = 0.4  # std/mean < threshold
    cv_variation_penalty: float = 3.0
    
    # Smooth transitions penalty
    smooth_transitions_enabled: bool = True  # Enable smooth transition optimization
    min_transition_probability: float = 0.3  # Minimum transition probability for smoothness
    abrupt_transition_penalty: float = 2.0   # Penalty for abrupt transitions (low transition prob)
    
    # Noise handling penalty
    noise_handling_enabled: bool = True  # Enable noise handling optimization
    max_noise_ratio: float = 0.10  # Maximum acceptable noise ratio (10%)
    noise_ratio_penalty: float = 15.0  # Penalty for high noise ratio
    unassigned_noise_penalty: float = 10.0  # Penalty for unassigned noise points


@dataclass
class ClusteringOptimizationGoals:
    """
    Unified clustering optimization goals - Temporal, Economic & Statistical Focus.

    Primary optimization goals (balanced weighting):
    1. Temporal Smoothness (33%): Regime persistence and stability
    2. Economic Quality (33%): Rolling log-likelihood + Economic utility (Sharpe)
    3. Statistical Quality (34%): CV ratio (between/within cluster variance)

    Structural constraints:
    - Cluster count: 4-8 (preferred)
    - Cluster size: 2%-20% each
    """

    # ===== PRIMARY OPTIMIZATION GOALS =====

    # Goal 1: Temporal Smoothness (33%)
    # Higher is better - measures regime persistence and temporal stability
    # Critical for financial markets: regimes should be persistent and economically meaningful
    temporal_smoothness: GoalConfig = field(default_factory=lambda: GoalConfig(
        name="Temporal Smoothness",
        objective=OptimizationObjective.MAXIMIZE,
        weight=0.33,  # 33% of composite score
        target_range=(0.7, 1.0),  # Aim for high persistence (few transitions)
        constraint_threshold=0.5,  # Soft constraint: smoothness >= 0.5
        description="Regime persistence and temporal stability (penalizes rapid switching)",
        enable_normalization=True,
        normalization_method=NormalizationMethod.RANK,
        stability_threshold=0.3
    ))

    # Goal 2: Economic Quality (33%)
    # Composite of rolling log-likelihood and economic utility
    # Sub-weights: 50% rolling LL + 50% Sharpe
    economic_quality: GoalConfig = field(default_factory=lambda: GoalConfig(
        name="Economic Quality",
        objective=OptimizationObjective.MAXIMIZE,
        weight=0.33,  # 33% of composite score
        target_range=(0.0, 10.0),  # Normalized composite
        constraint_threshold=None,
        description="Economic quality: predictive likelihood + trading Sharpe",
        enable_normalization=True,
        normalization_method=NormalizationMethod.RANK,
        stability_threshold=0.4,
        # Sub-components for economic quality
        sub_weights={
            'rolling_ll': 0.5,  # 50% rolling log-likelihood
            'sharpe': 0.5       # 50% economic utility (Sharpe)
        }
    ))

    # Sub-component configs for economic quality
    rolling_log_likelihood: GoalConfig = field(default_factory=lambda: GoalConfig(
        name="Rolling Log-Likelihood",
        objective=OptimizationObjective.MAXIMIZE,
        weight=0.165,  # 16.5% of total (50% of economic 33%)
        target_range=(-10.0, 0.0),  # Closer to 0 is better
        constraint_threshold=None,
        description="Rolling/blocked predictive log-likelihood on held-out time blocks",
        enable_normalization=True,
        normalization_method=NormalizationMethod.RANK,
        stability_threshold=0.4
    ))

    economic_utility: GoalConfig = field(default_factory=lambda: GoalConfig(
        name="Economic Utility (Sharpe)",
        objective=OptimizationObjective.MAXIMIZE,
        weight=0.165,  # 16.5% of total (50% of economic 33%)
        target_range=(0.5, 3.0),  # Aim for Sharpe > 1.0
        constraint_threshold=0.5,  # Soft constraint: Sharpe >= 0.5
        description="Out-of-sample annualized Sharpe of regime-aware strategy",
        enable_normalization=True,
        normalization_method=NormalizationMethod.RANK,
        stability_threshold=0.4
    ))

    # Goal 3: Statistical Quality (34%) - CV Ratio
    # Higher is better - measures cluster separation quality
    # CV ratio = between-cluster variance / within-cluster variance
    statistical_quality: GoalConfig = field(default_factory=lambda: GoalConfig(
        name="Statistical Quality (CV Ratio)",
        objective=OptimizationObjective.MAXIMIZE,
        weight=0.34,  # 34% of composite score (total=100%)
        target_range=(10.0, 1000.0),  # Higher is better
        constraint_threshold=5.0,  # Soft constraint: CV ratio >= 5
        description="Coefficient of variation ratio (between/within cluster variance)",
        enable_normalization=True,
        normalization_method=NormalizationMethod.RANK,
        stability_threshold=0.3
    ))
    
    # ===== STRUCTURAL CONSTRAINTS =====
    
    # Constraint 1: Number of Clusters
    cluster_count_range: Tuple[int, int] = (4, 8)  # Preferred: 4-8 clusters
    cluster_count_min: int = 3  # Absolute minimum
    cluster_count_max: int = 10  # Absolute maximum
    
    # Constraint 2: Cluster Size Bounds
    min_cluster_size_pct: float = 0.02  # 2% minimum
    max_cluster_size_pct: float = 0.20  # 20% maximum
    
    # ===== CROSS-VALIDATION CONFIGURATION =====
    
    cv_config: CVConfig = field(default_factory=lambda: CVConfig())
    
    # ===== PENALTY CONFIGURATION =====
    
    penalty_config: PenaltyConfig = field(default_factory=lambda: PenaltyConfig())
    
    # ===== PARETO OPTIMIZATION =====
    
    use_pareto_optimization: bool = True  # Use Pareto front for multi-objective
    pareto_knee_selection: bool = True  # Auto-select knee point
    
    # ===== ROBUSTNESS CHECKS =====
    
    n_robustness_seeds: int = 30  # Number of seeds for robustness check
    top_n_candidates: int = 10  # Validate top N candidates
    require_significance: bool = True  # Require statistical significance
    
    def get_all_goals(self) -> Dict[str, GoalConfig]:
        """Get all goal configurations as a dictionary."""
        return {
            'temporal_smoothness': self.temporal_smoothness,
            'economic_quality': self.economic_quality,
            'statistical_quality': self.statistical_quality,
            # Sub-components for reference
            'rolling_ll': self.rolling_log_likelihood,
            'economic_utility': self.economic_utility,
        }

    def get_primary_goals(self) -> Dict[str, GoalConfig]:
        """Get primary optimization goals (Temporal + Economic + Statistical)."""
        return {
            'temporal_smoothness': self.temporal_smoothness,
            'economic_quality': self.economic_quality,
            'statistical_quality': self.statistical_quality,
        }
    
    def get_weights_dict(self) -> Dict[str, float]:
        """Get weights for composite score calculation."""
        return {
            'temporal_smoothness': self.temporal_smoothness.weight,
            'economic_quality': self.economic_quality.weight,
            'statistical_quality': self.statistical_quality.weight,
        }

    def validate_weights(self) -> bool:
        """Validate that weights sum to 1.0 (or close to it)."""
        total_weight = sum(self.get_weights_dict().values())
        return abs(total_weight - 1.0) < 1e-6

    def normalize_weights(self):
        """Normalize weights to sum to 1.0."""
        weights = self.get_weights_dict()
        total = sum(weights.values())
        if total > 0:
            self.temporal_smoothness.weight = weights['temporal_smoothness'] / total
            self.economic_quality.weight = weights['economic_quality'] / total
            self.statistical_quality.weight = weights['statistical_quality'] / total


@dataclass
class OptimizationTargets:
    """
    Specific target values for optimization (Predictive & Economic).
    """
    
    # Primary targets (predictive + economic)
    min_rolling_ll: float = -8.0  # Minimum acceptable rolling LL
    min_one_step_ll: float = -4.0  # Minimum acceptable one-step LL
    min_sharpe: float = 0.5  # Minimum acceptable Sharpe ratio
    
    # Aspirational targets (excellent performance)
    target_rolling_ll: float = -3.0  # Target rolling LL
    target_one_step_ll: float = -1.5  # Target one-step LL
    target_sharpe: float = 1.5  # Target Sharpe ratio
    
    # Economic metrics
    max_drawdown_threshold: float = 0.30  # Max acceptable drawdown (30%)
    min_win_rate: float = 0.45  # Minimum win rate
    
    # ===== STRUCTURAL CONSTRAINTS =====
    
    # Cluster count constraints - reduced for better temporal smoothness
    min_clusters: int = 4  # Absolute minimum
    max_clusters: int = 5  # Absolute maximum (reduced for better temporal smoothness)
    target_clusters: Tuple[int, int] = (4, 5)  # Preferred range
    
    # Cluster size constraints (as percentage of total samples)
    min_cluster_size_pct: float = 0.02  # 2% minimum
    max_cluster_size_pct: float = 0.20  # 20% maximum
    
    # ===== CLUSTERING QUALITY TARGETS =====
    
    # CV Score (Calinski-Harabasz) - higher is better
    min_cv_score: float = 1.2  # Minimum acceptable CV score
    target_cv_score: float = 2.0  # Target CV score
    
    # Silhouette Score - higher is better (-1 to 1)
    min_silhouette_score: float = 0.1  # Minimum acceptable silhouette
    target_silhouette_score: float = 0.3  # Target silhouette score
    
    # Davies-Bouldin Index - lower is better
    max_dbi_score: float = 2.0  # Maximum acceptable DBI
    target_dbi_score: float = 1.0  # Target DBI
    
    # Balance Score - higher is better (0 to 1)
    min_balance_score: float = 0.3  # Minimum acceptable balance
    target_balance_score: float = 0.7  # Target balance score
    
    # Temporal Smoothness - higher is better (0 to 1) - RELAXED for data reality
    min_temporal_smoothness: float = 0.20  # Minimum acceptable temporal smoothness (RELAXED from 0.6 to 0.20)
    target_temporal_smoothness: float = 0.40  # Target temporal smoothness (RELAXED from 0.9 to 0.40)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            'min_rolling_ll': self.min_rolling_ll,
            'min_one_step_ll': self.min_one_step_ll,
            'min_sharpe': self.min_sharpe,
            'target_rolling_ll': self.target_rolling_ll,
            'target_one_step_ll': self.target_one_step_ll,
            'target_sharpe': self.target_sharpe,
            'max_drawdown_threshold': self.max_drawdown_threshold,
            'min_win_rate': self.min_win_rate,
            'min_clusters': self.min_clusters,
            'max_clusters': self.max_clusters,
            'min_cv_score': self.min_cv_score,
            'target_cv_score': self.target_cv_score,
            'min_silhouette_score': self.min_silhouette_score,
            'target_silhouette_score': self.target_silhouette_score,
            'max_dbi_score': self.max_dbi_score,
            'target_dbi_score': self.target_dbi_score,
            'min_balance_score': self.min_balance_score,
            'target_balance_score': self.target_balance_score,
            'min_temporal_smoothness': self.min_temporal_smoothness,
            'target_temporal_smoothness': self.target_temporal_smoothness,
        }


# ===== GLOBAL INSTANCES =====

# Default goals configuration used across all clustering components
DEFAULT_CLUSTERING_GOALS = ClusteringOptimizationGoals()

# Default optimization targets
DEFAULT_OPTIMIZATION_TARGETS = OptimizationTargets()


# ===== TIME SERIES CROSS-VALIDATION =====

class TimeSeriesCrossValidator:
    """
    Time series cross-validation with rolling/blocked folds.
    
    Implements reliable measurement across K=5-10 folds with robust statistics.
    For daily crypto: train=18 months, val=3 months, slide forward.
    """
    
    def __init__(self, cv_config: Optional[CVConfig] = None):
        """Initialize time series cross-validator."""
        self.cv_config = cv_config or CVConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def split(self, data: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/validation splits for time series data.
        
        Args:
            data: Time series DataFrame (must have datetime index)
            
        Returns:
            List of (train_idx, val_idx) tuples
        """
        n_samples = len(data)
        
        if self.cv_config.strategy == CVStrategy.ROLLING:
            return self._rolling_split(data)
        elif self.cv_config.strategy == CVStrategy.EXPANDING:
            return self._expanding_split(data)
        elif self.cv_config.strategy == CVStrategy.BLOCKED:
            return self._blocked_split(data)
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_config.strategy}")
    
    def _rolling_split(self, data: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate rolling window splits."""
        if len(data) == 0:
            self.logger.warning("⚠️ Empty data provided for rolling split")
            return []
        
        # Calculate samples per window
        freq = pd.infer_freq(data.index)
        if freq is None:
            # Estimate from median time diff
            freq = pd.to_timedelta(data.index.to_series().diff().median())
        
        # Convert months to samples
        if isinstance(freq, str):
            freq_td = pd.to_timedelta(freq)
        else:
            freq_td = freq
        
        # Handle zero or invalid frequency
        if freq_td.total_seconds() <= 0:
            self.logger.warning("⚠️ Invalid frequency detected, using default daily frequency")
            freq_td = pd.Timedelta(days=1)
        
        try:
            samples_per_month = int(pd.Timedelta(days=30) / freq_td)
        except (ZeroDivisionError, ValueError) as e:
            self.logger.warning(f"⚠️ Error calculating samples per month: {e}, using default")
            samples_per_month = 30  # Default to daily data
        
        samples_per_month = max(1, samples_per_month)  # Ensure at least 1
        
        train_size = self.cv_config.train_months * samples_per_month
        val_size = self.cv_config.val_months * samples_per_month
        
        splits = []
        n_samples = len(data)
        
        # Generate rolling splits
        for i in range(self.cv_config.n_splits):
            start_idx = i * val_size
            train_end_idx = start_idx + train_size
            val_end_idx = train_end_idx + val_size
            
            if val_end_idx > n_samples:
                break
            
            train_idx = np.arange(start_idx, train_end_idx)
            val_idx = np.arange(train_end_idx, val_end_idx)
            
            if len(train_idx) >= self.cv_config.min_train_samples and \
               len(val_idx) >= self.cv_config.min_val_samples:
                splits.append((train_idx, val_idx))
        
        if len(splits) == 0:
            self.logger.warning("⚠️ No valid splits generated. Check data size and CV configuration.")
        
        self.logger.info(f"✅ Generated {len(splits)} rolling CV splits")
        return splits
    
    def _expanding_split(self, data: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate expanding window splits."""
        if len(data) == 0:
            self.logger.warning("⚠️ Empty data provided for expanding split")
            return []
        
        n_samples = len(data)
        
        # Ensure we have enough data for splits
        if n_samples < self.cv_config.min_train_samples + self.cv_config.min_val_samples:
            self.logger.warning(
                f"⚠️ Insufficient data for expanding split: {n_samples} samples "
                f"(need at least {self.cv_config.min_train_samples + self.cv_config.min_val_samples})"
            )
            return []
        
        val_size = n_samples // (self.cv_config.n_splits + 1)
        val_size = max(val_size, self.cv_config.min_val_samples)
        
        splits = []
        for i in range(1, self.cv_config.n_splits + 1):
            train_idx = np.arange(0, i * val_size)
            val_idx = np.arange(i * val_size, min((i + 1) * val_size, n_samples))
            
            if len(train_idx) >= self.cv_config.min_train_samples and \
               len(val_idx) >= self.cv_config.min_val_samples:
                splits.append((train_idx, val_idx))
        
        if len(splits) == 0:
            self.logger.warning("⚠️ No valid splits generated. Check data size and CV configuration.")
        
        self.logger.info(f"✅ Generated {len(splits)} expanding CV splits")
        return splits
    
    def _blocked_split(self, data: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate non-overlapping blocked splits."""
        if len(data) == 0:
            self.logger.warning("⚠️ Empty data provided for blocked split")
            return []
        
        n_samples = len(data)
        
        # Ensure we have enough data for splits
        if n_samples < self.cv_config.min_train_samples + self.cv_config.min_val_samples:
            self.logger.warning(
                f"⚠️ Insufficient data for blocked split: {n_samples} samples "
                f"(need at least {self.cv_config.min_train_samples + self.cv_config.min_val_samples})"
            )
            return []
        
        block_size = n_samples // self.cv_config.n_splits
        block_size = max(block_size, self.cv_config.min_val_samples)
        
        splits = []
        for i in range(self.cv_config.n_splits - 1):
            # Use all other blocks for training
            val_start = i * block_size
            val_end = min((i + 1) * block_size, n_samples)
            
            train_idx = np.concatenate([
                np.arange(0, val_start),
                np.arange(val_end, n_samples)
            ])
            val_idx = np.arange(val_start, val_end)
            
            if len(train_idx) >= self.cv_config.min_train_samples and \
               len(val_idx) >= self.cv_config.min_val_samples:
                splits.append((train_idx, val_idx))
        
        if len(splits) == 0:
            self.logger.warning("⚠️ No valid splits generated. Check data size and CV configuration.")
        
        self.logger.info(f"✅ Generated {len(splits)} blocked CV splits")
        return splits


# ===== METRIC CALCULATORS =====

class MetricCalculator:
    """Calculator for predictive and economic metrics."""
    
    def __init__(self, use_vectorbt: bool = True):
        """
        Initialize metric calculator.
        
        Args:
            use_vectorbt: Whether to use VectorBT for calculations (currently unused)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE
        if self.use_vectorbt:
            self.logger.debug("VectorBT available but not currently used in calculations")
    
    def calculate_rolling_log_likelihood(
        self,
        data: np.ndarray,
        regime_probs: np.ndarray,
        regime_params: Dict[int, Dict[str, float]]
    ) -> Tuple[float, float]:
        """
        Calculate rolling predictive log-likelihood.
        
        Args:
            data: Held-out data (T, features)
            regime_probs: Regime probabilities (T, n_regimes)
            regime_params: Parameters for each regime (mean, cov)
            
        Returns:
            (mean_ll, std_ll) across time
        """
        n_samples, n_features = data.shape
        n_regimes = regime_probs.shape[1]
        
        log_likelihoods = []
        
        for t in range(n_samples):
            # Log-likelihood contribution from each regime
            ll_per_regime = []
            
            for regime_id in range(n_regimes):
                # Safe access to regime parameters
                regime_param = regime_params.get(regime_id, {})
                mean = regime_param.get('mean', np.zeros(n_features))
                cov = regime_param.get('cov', np.eye(n_features))
                
                # Multivariate normal log-likelihood
                try:
                    diff = data[t] - mean
                    # Use solve() instead of inv() for better numerical stability
                    stabilized_cov = cov + np.eye(n_features) * 1e-6
                    quadratic_form = diff.T @ np.linalg.solve(stabilized_cov, diff)
                    ll = -0.5 * (
                        n_features * np.log(2 * np.pi) +
                        np.log(np.linalg.det(stabilized_cov) + 1e-8) +
                        quadratic_form
                    )
                    ll_per_regime.append(ll)
                except (ValueError, np.linalg.LinAlgError, OverflowError, RuntimeError) as e:
                    self.logger.debug(
                        f"Numerical issue in log-likelihood calculation for regime {regime_id}: {e}"
                    )
                    ll_per_regime.append(-1e6)  # Numerical issues
            
            # Weighted log-likelihood (mixture)
            ll_per_regime = np.array(ll_per_regime)
            log_weights = np.log(regime_probs[t] + 1e-10)
            total_ll = logsumexp(ll_per_regime + log_weights)
            
            log_likelihoods.append(total_ll)
        
        log_likelihoods = np.array(log_likelihoods)
        
        # Filter out extreme outliers
        log_likelihoods = np.clip(
            log_likelihoods,
            Constants.LOG_LIKELIHOOD_MIN,
            Constants.LOG_LIKELIHOOD_MAX
        )
        
        return float(np.mean(log_likelihoods)), float(np.std(log_likelihoods))
    
    def calculate_one_step_log_likelihood(
        self,
        data: np.ndarray,
        regime_labels: np.ndarray,
        regime_params: Dict[int, Dict[str, float]]
    ) -> Tuple[float, float]:
        """
        Calculate one-step-ahead predictive log-likelihood.
        
        Args:
            data: Data (T, features)
            regime_labels: Regime assignments (T,)
            regime_params: Parameters for each regime
            
        Returns:
            (mean_ll, std_ll) one-step-ahead
        """
        n_samples, n_features = data.shape
        log_likelihoods = []
        
        for t in range(1, n_samples):
            # Predict from previous regime
            prev_regime = int(regime_labels[t-1])
            
            # Safe access to regime parameters
            regime_param = regime_params.get(prev_regime, {})
            mean = regime_param.get('mean', np.zeros(n_features))
            cov = regime_param.get('cov', np.eye(n_features))
            
            # Log-likelihood of current observation given previous regime
            try:
                diff = data[t] - mean
                # Use solve() instead of inv() for better numerical stability
                stabilized_cov = cov + np.eye(n_features) * 1e-6
                quadratic_form = diff.T @ np.linalg.solve(stabilized_cov, diff)
                ll = -0.5 * (
                    n_features * np.log(2 * np.pi) +
                    np.log(np.linalg.det(stabilized_cov) + 1e-8) +
                    quadratic_form
                )
                log_likelihoods.append(ll)
            except (ValueError, np.linalg.LinAlgError, OverflowError, RuntimeError) as e:
                self.logger.debug(
                    f"Numerical issue in one-step log-likelihood calculation for regime {prev_regime}: {e}"
                )
                log_likelihoods.append(-1e6)
        
        log_likelihoods = np.array(log_likelihoods)
        log_likelihoods = np.clip(
            log_likelihoods,
            Constants.LOG_LIKELIHOOD_MIN,
            Constants.LOG_LIKELIHOOD_MAX
        )
        
        return float(np.mean(log_likelihoods)), float(np.std(log_likelihoods))
    
    def calculate_economic_utility(
        self,
        returns: np.ndarray,
        regime_labels: np.ndarray,
        regime_signals: Optional[Dict[int, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate out-of-sample economic utility (Sharpe ratio, etc.).
        
        Args:
            returns: Asset returns (T,)
            regime_labels: Regime assignments (T,)
            regime_signals: Optional regime-specific signals/positions
            
        Returns:
            Dictionary with Sharpe, turnover, max_dd, etc.
        """
        # Default regime signals (simple: long volatile regimes, short calm)
        if regime_signals is None:
            # Calculate regime volatilities
            unique_regimes = np.unique(regime_labels)
            regime_vols = {}
            for regime in unique_regimes:
                mask = regime_labels == regime
                regime_vols[regime] = np.std(returns[mask])
            
            # Normalize signals: high vol = +1, low vol = -1
            vol_values = list(regime_vols.values())
            vol_median = np.median(vol_values)
            regime_signals = {
                r: 1.0 if v > vol_median else -1.0
                for r, v in regime_vols.items()
            }
        
        # Generate strategy returns
        positions = np.array([regime_signals.get(int(r), 0.0) for r in regime_labels])
        strategy_returns = positions * returns
        
        # Calculate metrics
        sharpe = self._calculate_sharpe(strategy_returns)
        max_dd = self._calculate_max_drawdown(strategy_returns)
        turnover = self._calculate_monthly_turnover(positions)
        win_rate = self._calculate_win_rate(strategy_returns)
        
        return {
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'monthly_turnover': turnover,
            'win_rate': win_rate,
            'total_return': np.sum(strategy_returns),
            'volatility': np.std(strategy_returns) * np.sqrt(Constants.TRADING_DAYS_PER_YEAR)  # Annualized
        }
    
    def _calculate_sharpe(
        self,
        returns: np.ndarray,
        periods_per_year: int = Constants.TRADING_DAYS_PER_YEAR
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        
        std_return = np.std(returns)
        if std_return == 0 or np.isnan(std_return) or np.isinf(std_return):
            return 0.0
        
        mean_return = np.mean(returns)
        if np.isnan(mean_return) or np.isinf(mean_return):
            return 0.0
        
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        
        # Handle edge cases
        if np.isnan(sharpe) or np.isinf(sharpe):
            return 0.0
        
        return float(sharpe)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_dd = np.min(drawdown)
        
        return float(abs(max_dd))
    
    def _calculate_monthly_turnover(self, positions: np.ndarray, bars_per_month: int = 21) -> float:
        """Calculate average monthly turnover (regime switches)."""
        if len(positions) == 0:
            return 0.0
        
        # Count position changes
        changes = np.diff(positions) != 0
        total_changes = np.sum(changes)
        
        # Convert to monthly
        n_months = len(positions) / bars_per_month
        monthly_turnover = total_changes / max(n_months, 1)
        
        return float(monthly_turnover)
    
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calculate win rate (fraction of positive returns)."""
        if len(returns) == 0:
            return 0.0
        
        wins = np.sum(returns > 0)
        win_rate = wins / len(returns)
        
        return float(win_rate)


# ===== NORMALIZATION UTILITIES =====

class MetricNormalizer:
    """Normalize metrics across trials for meaningful composite scores."""
    
    def __init__(self, method: NormalizationMethod = NormalizationMethod.RANK):
        """Initialize normalizer."""
        self.method = method
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def normalize_metrics(
        self,
        metrics: Dict[str, List[float]],
        objectives: Dict[str, OptimizationObjective]
    ) -> Dict[str, List[float]]:
        """
        Normalize metrics across trials.
        
        Args:
            metrics: Dict of {metric_name: [values across trials]}
            objectives: Dict of {metric_name: objective (max/min)}
            
        Returns:
            Normalized metrics (higher is always better)
        """
        normalized = {}
        
        for metric_name, values in metrics.items():
            values_array = np.array(values)
            objective = objectives.get(metric_name, OptimizationObjective.MAXIMIZE)
            
            # Invert if minimization objective
            if objective == OptimizationObjective.MINIMIZE:
                values_array = -values_array
            
            # Apply normalization method
            if self.method == NormalizationMethod.ZSCORE:
                norm_values = self._zscore_normalize(values_array)
            elif self.method == NormalizationMethod.RANK:
                norm_values = self._rank_normalize(values_array)
            elif self.method == NormalizationMethod.ROBUST_ZSCORE:
                norm_values = self._robust_zscore_normalize(values_array)
            else:  # MIN_MAX
                norm_values = self._minmax_normalize(values_array)
            
            normalized[metric_name] = norm_values.tolist()
        
        return normalized
    
    def _zscore_normalize(self, values: np.ndarray) -> np.ndarray:
        """Z-score normalization."""
        # Handle NaN values
        if np.any(np.isnan(values)):
            self.logger.warning("⚠️ NaN values detected in z-score normalization, replacing with 0")
            values = np.nan_to_num(values, nan=0.0)
        
        mean = np.mean(values)
        std = np.std(values)
        if std == 0 or np.isnan(std) or np.isinf(std):
            return np.zeros_like(values)
        return (values - mean) / std
    
    def _rank_normalize(self, values: np.ndarray) -> np.ndarray:
        """Rank-based normalization to [0, 1]."""
        if len(values) <= 1:
            return np.ones_like(values) * 0.5
        
        # Handle NaN values
        if np.any(np.isnan(values)):
            self.logger.warning("⚠️ NaN values detected in rank normalization, replacing with 0")
            values = np.nan_to_num(values, nan=0.0)
        
        # Rank (higher is better after objective adjustment)
        ranks = stats.rankdata(values)
        # Normalize to [0, 1]
        normalized = (ranks - 1) / (len(ranks) - 1)
        return normalized
    
    def _robust_zscore_normalize(self, values: np.ndarray) -> np.ndarray:
        """Robust z-score using median and MAD."""
        # Handle NaN values
        if np.any(np.isnan(values)):
            self.logger.warning("⚠️ NaN values detected in robust z-score normalization, replacing with 0")
            values = np.nan_to_num(values, nan=0.0)
        
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad == 0 or np.isnan(mad) or np.isinf(mad):
            return np.zeros_like(values)
        return (values - median) / (1.4826 * mad)  # 1.4826 for normal consistency
    
    def _minmax_normalize(self, values: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]."""
        # Handle NaN values
        if np.any(np.isnan(values)):
            self.logger.warning("⚠️ NaN values detected in min-max normalization, replacing with 0")
            values = np.nan_to_num(values, nan=0.0)
        
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val == min_val or np.isnan(min_val) or np.isnan(max_val) or \
           np.isinf(min_val) or np.isinf(max_val):
            return np.ones_like(values) * 0.5
        return (values - min_val) / (max_val - min_val)


# ===== CV RATIO CALCULATION =====

def calculate_cv_ratio(
    data: np.ndarray,
    labels: np.ndarray,
    use_jit: bool = True,
    use_vectorbt: bool = True
) -> float:
    """
    Calculate CV ratio (Calinski-Harabasz Index) with VectorBT optimization.

    CV Ratio = Between-cluster variance / Within-cluster variance
    Higher is better - indicates well-separated clusters.

    Hybrid approach:
    1. Try VectorBT StatisticalCalculationsOptimizer (fastest for large datasets)
    2. Fall back to Numba JIT (fast for medium datasets)
    3. Fall back to numpy (always works)

    Args:
        data: Feature matrix (N, D)
        labels: Cluster labels (N,)
        use_jit: Use JIT-compiled version if VectorBT fails
        use_vectorbt: Try VectorBT optimization first

    Returns:
        CV ratio (higher is better)
    """
    if len(data) == 0 or len(labels) == 0:
        return 0.0

    n_clusters = len(np.unique(labels))
    if n_clusters <= 1:
        return 0.0

    within_var = 0.0
    between_var = 0.0

    # Try VectorBT first for large datasets
    if use_vectorbt and len(data) > 100:  # Use VectorBT for datasets > 100 samples
        stat_opt = _get_stat_optimizer()
        if stat_opt is not None:
            try:
                # Calculate global centroid
                global_centroid = np.mean(data, axis=0)

                # Calculate variances using VectorBT
                for k in range(n_clusters):
                    mask = labels == k
                    cluster_size = np.sum(mask)

                    if cluster_size == 0:
                        continue

                    cluster_data = data[mask]
                    cluster_centroid = np.mean(cluster_data, axis=0)

                    # Within-cluster variance using VectorBT
                    for d in range(data.shape[1]):
                        within_var += stat_opt.calculate_variance(cluster_data[:, d], batch_mode=False) * cluster_size

                    # Between-cluster variance
                    between_var += cluster_size * np.sum((cluster_centroid - global_centroid) ** 2)

                # Normalize
                n_samples = len(data)
                within_var /= n_samples
                between_var /= n_samples

                # Success with VectorBT
                logger.debug("CV ratio calculated using VectorBT")
            except Exception as e:
                logger.debug(f"VectorBT CV ratio calculation failed: {e}, falling back to JIT")
                within_var = 0.0
                between_var = 0.0

    # Use JIT-compiled versions if VectorBT not used or failed
    if (within_var == 0.0 and between_var == 0.0) and use_jit and NUMBA_AVAILABLE:
        within_var = _calculate_within_cluster_variance_jit(data, labels, n_clusters)
        between_var = _calculate_between_cluster_variance_jit(data, labels, n_clusters)
        logger.debug("CV ratio calculated using Numba JIT")

    # Final fallback to numpy
    if within_var == 0.0 and between_var == 0.0:
        # Global centroid
        global_centroid = np.mean(data, axis=0)

        for k in range(n_clusters):
            mask = labels == k
            cluster_size = np.sum(mask)

            if cluster_size == 0:
                continue

            cluster_data = data[mask]
            cluster_centroid = np.mean(cluster_data, axis=0)

            # Within-cluster variance
            within_var += np.sum((cluster_data - cluster_centroid) ** 2)

            # Between-cluster variance
            between_var += cluster_size * np.sum((cluster_centroid - global_centroid) ** 2)

        # Normalize
        n_samples = len(data)
        within_var /= n_samples
        between_var /= n_samples
        logger.debug("CV ratio calculated using numpy fallback")

    # Calculate ratio
    if within_var == 0 or np.isnan(within_var) or np.isinf(within_var):
        return 0.0

    cv_ratio = between_var / within_var

    # Handle edge cases
    if np.isnan(cv_ratio) or np.isinf(cv_ratio):
        return 0.0

    return float(cv_ratio)


def _compute_episode_duration_stats(labels: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Helper to derive per-regime episode durations and summary stats."""

    episode_lengths = []
    if len(labels) == 0:
        return np.array([]), {
            'mean': 0.0,
            'median': 0.0,
            'pct_short': 0.0,
            'pct_actionable': 0.0,
            'pct_in_target': 0.0
        }

    current_label = labels[0]
    current_length = 1

    for label in labels[1:]:
        if label == current_label:
            current_length += 1
        else:
            episode_lengths.append(current_length)
            current_label = label
            current_length = 1

    episode_lengths.append(current_length)

    durations = np.array(episode_lengths, dtype=np.float64)

    if NUMBA_AVAILABLE:
        stats_tuple = _calculate_episode_duration_stats_jit(durations)
        mean_duration, median_duration, _, pct_short, pct_actionable, pct_in_target = stats_tuple
    else:
        if len(durations) == 0:
            mean_duration = median_duration = pct_short = pct_actionable = pct_in_target = 0.0
        else:
            mean_duration = float(np.mean(durations))
            median_duration = float(np.median(durations))
            pct_short = float(np.mean(durations < 7))
            pct_actionable = float(np.mean(durations >= 20))
            pct_in_target = float(np.mean((durations >= 5) & (durations <= 20)))

    stats = {
        'mean': float(mean_duration),
        'median': float(median_duration),
        'pct_short': float(pct_short),
        'pct_actionable': float(pct_actionable),
        'pct_in_target': float(pct_in_target)
    }

    tprint_debug(
        "📊 Episode duration stats",
        extra={
            'mean': stats['mean'],
            'median': stats['median'],
            'pct_short': stats['pct_short'],
            'pct_actionable': stats['pct_actionable'],
            'pct_in_target': stats['pct_in_target'],
            'episodes': len(durations)
        }
    )

    return durations, stats


def calculate_temporal_smoothness(labels: np.ndarray, use_jit: bool = True) -> float:
    """
    Calculate temporal smoothness of regime assignments.

    Smoothness = 1 - (n_transitions / max_transitions)
    Higher values indicate more stable regimes with fewer transitions.

    Args:
        labels: Regime labels (T,)
        use_jit: Use JIT-compiled version if available

    Returns:
        Smoothness score [0, 1], higher is better
    """
    if len(labels) <= 1:
        return 1.0

    # Use JIT-compiled version if available
    if use_jit and NUMBA_AVAILABLE:
        base_smoothness = _calculate_temporal_smoothness_jit(labels)
    else:
        n_transitions = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                n_transitions += 1

        max_transitions = len(labels) - 1
        base_smoothness = 1.0 - (n_transitions / max_transitions)

    base_smoothness = float(np.clip(base_smoothness, 0.0, 1.0))

    # Episode-duration aware adjustment
    _, duration_stats = _compute_episode_duration_stats(labels)
    short_penalty = duration_stats['pct_short']
    target_bonus = duration_stats['pct_in_target'] * 0.2  # modest reward for desirable durations

    # Flip-flop detection: count rapid back-and-forth transitions (A->B->A within 3 steps)
    flip_flops = 0
    for i in range(2, len(labels)):
        if labels[i] == labels[i-2] and labels[i] != labels[i-1]:
            flip_flops += 1

    max_flip_flops = max(1, len(labels) - 2)
    flip_flop_ratio = flip_flops / max_flip_flops

    flip_flop_penalty = flip_flop_ratio * 0.5  # weight flip-flop effect

    adjusted = base_smoothness
    adjusted -= short_penalty * 0.3  # penalize high short-episode share
    adjusted -= flip_flop_penalty
    adjusted += target_bonus

    final_score = float(np.clip(adjusted, 0.0, 1.0))

    tprint_debug(
        "🧭 Temporal smoothness calculation",
        extra={
            'base_smoothness': base_smoothness,
            'short_penalty': short_penalty,
            'target_bonus': target_bonus,
            'flip_flop_ratio': flip_flop_ratio,
            'flip_flop_penalty': flip_flop_penalty,
            'final_score': final_score
        }
    )

    return final_score


# ===== ENHANCED TEMPORAL METRICS =====

@njit(cache=True)
def _calculate_episode_duration_stats_jit(durations: np.ndarray, target_min: int = 5, target_max: int = 20) -> Tuple:
    """
    JIT-compiled episode duration statistics with target range.

    Args:
        durations: Array of episode durations
        target_min: Minimum target duration (default 5 bars)
        target_max: Maximum target duration (default 20 bars)

    Returns:
        Tuple of (mean, median, cv, pct_short, pct_actionable, pct_in_target)
    """
    if len(durations) == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Calculate basic statistics
    mean_duration = np.mean(durations)
    median_duration = np.median(durations)
    std_duration = np.std(durations)
    cv = std_duration / mean_duration if mean_duration > 0 else 0.0

    # Calculate percentages
    n_episodes = len(durations)
    pct_short_episodes = np.sum(durations < 7) / n_episodes  # < 7 bars is short
    pct_actionable = np.sum(durations >= 20) / n_episodes  # >= 20 bars is actionable
    pct_in_target = np.sum((durations >= target_min) & (durations <= target_max)) / n_episodes

    return (mean_duration, median_duration, cv, pct_short_episodes, pct_actionable, pct_in_target)


@njit(cache=True)
def _calculate_transition_predictability_jit(labels: np.ndarray, features: np.ndarray, lookback: int = 10) -> float:
    """
    JIT-compiled transition predictability calculation.

    Measures if regime transitions can be predicted from recent features.
    High similarity in features before transitions = predictable transitions.

    Args:
        labels: Regime labels (T,)
        features: Feature matrix (T, D)
        lookback: Number of bars to look back before transition

    Returns:
        Predictability score [0, 1], higher is better
    """
    n_samples = len(labels)
    if n_samples < 2:
        return 0.0

    # Find transition points
    transition_indices = []
    for i in range(1, n_samples):
        if labels[i] != labels[i-1]:
            transition_indices.append(i)

    if len(transition_indices) < 2:
        return 0.0

    # Extract feature vectors before transitions
    n_transitions = len(transition_indices)
    n_features = features.shape[1]

    # Calculate pairwise correlations between pre-transition features
    similarities = []
    for i in range(n_transitions):
        for j in range(i+1, n_transitions):
            idx_i = transition_indices[i]
            idx_j = transition_indices[j]

            # Check if we have enough lookback
            if idx_i >= lookback and idx_j >= lookback:
                # Get feature vectors
                feat_i = features[idx_i-lookback:idx_i].flatten()
                feat_j = features[idx_j-lookback:idx_j].flatten()

                # Calculate correlation
                mean_i = np.mean(feat_i)
                mean_j = np.mean(feat_j)

                num = np.sum((feat_i - mean_i) * (feat_j - mean_j))
                denom = np.sqrt(np.sum((feat_i - mean_i)**2) * np.sum((feat_j - mean_j)**2))

                if denom > 1e-10:
                    corr = num / denom
                    similarities.append(abs(corr))

    if len(similarities) == 0:
        return 0.0

    # High average similarity = predictable transitions
    predictability = np.mean(np.array(similarities))
    return predictability


@njit(cache=True)
def _calculate_regime_autocorrelation_jit(features: np.ndarray, labels: np.ndarray, max_lag: int = 20) -> Tuple:
    """
    JIT-compiled regime persistence autocorrelation.

    Calculates autocorrelation of features within each regime to measure persistence.

    Args:
        features: Feature matrix (T, D)
        labels: Regime labels (T,)
        max_lag: Maximum lag for autocorrelation

    Returns:
        Tuple of (mean_ac_lag1, mean_ac_lag5, half_life)
    """
    n_regimes = len(np.unique(labels))

    ac_lag1_list = []
    ac_lag5_list = []
    half_lives = []

    for regime_id in range(n_regimes):
        # Get data for this regime
        mask = labels == regime_id
        regime_size = np.sum(mask)

        if regime_size < max_lag + 1:
            continue

        # Get regime features (use first feature for autocorrelation)
        regime_feat = features[mask, 0]

        # Calculate AC(1)
        if len(regime_feat) > 1:
            mean_val = np.mean(regime_feat)
            var_val = np.var(regime_feat)

            if var_val > 1e-10:
                ac1 = np.sum((regime_feat[:-1] - mean_val) * (regime_feat[1:] - mean_val)) / ((len(regime_feat) - 1) * var_val)
                ac_lag1_list.append(max(0.0, min(1.0, ac1)))  # Clip to [0, 1]

        # Calculate AC(5)
        if len(regime_feat) > 5:
            mean_val = np.mean(regime_feat)
            var_val = np.var(regime_feat)

            if var_val > 1e-10:
                ac5 = np.sum((regime_feat[:-5] - mean_val) * (regime_feat[5:] - mean_val)) / ((len(regime_feat) - 5) * var_val)
                ac_lag5_list.append(max(0.0, min(1.0, ac5)))  # Clip to [0, 1]

        # Estimate half-life (simplified)
        if len(ac_lag1_list) > 0 and ac_lag1_list[-1] > 0:
            half_life = -1.0 / np.log(max(ac_lag1_list[-1], 0.01))
            half_lives.append(min(half_life, 100.0))  # Cap at 100

    # Calculate averages
    mean_ac_lag1 = np.mean(np.array(ac_lag1_list)) if len(ac_lag1_list) > 0 else 0.0
    mean_ac_lag5 = np.mean(np.array(ac_lag5_list)) if len(ac_lag5_list) > 0 else 0.0
    mean_half_life = np.mean(np.array(half_lives)) if len(half_lives) > 0 else 0.0

    return (mean_ac_lag1, mean_ac_lag5, mean_half_life)


@njit(cache=True)
def _calculate_economic_transition_cost_jit(
    labels: np.ndarray,
    returns: np.ndarray,
    transaction_cost_bps: float = 10.0,
    lookforward: int = 20
) -> Tuple:
    """
    JIT-compiled economic transition cost calculation.

    Evaluates if regime transitions are economically justified.

    Args:
        labels: Regime labels (T,)
        returns: Return series (T,)
        transaction_cost_bps: Transaction cost in basis points
        lookforward: Bars to look forward after transition

    Returns:
        Tuple of (total_cost_pct, avg_benefit_vs_cost, profitable_transitions_pct)
    """
    n_samples = len(labels)
    if n_samples < 2:
        return (0.0, 0.0, 0.0)

    # Find transitions
    n_transitions = 0
    for i in range(1, n_samples):
        if labels[i] != labels[i-1]:
            n_transitions += 1

    if n_transitions == 0:
        return (0.0, 0.0, 0.0)

    # Calculate total cost
    total_cost = n_transitions * (transaction_cost_bps / 10000.0)
    total_returns = np.sum(returns)
    cost_pct = total_cost / abs(total_returns) if abs(total_returns) > 1e-10 else 0.0

    # Analyze each transition
    benefits = []
    for i in range(1, n_samples):
        if labels[i] != labels[i-1]:
            # Look forward
            end_idx = min(i + lookforward, n_samples)

            # Benefit = cumulative returns in new regime
            benefit = np.sum(returns[i:end_idx])

            # Cost
            cost = transaction_cost_bps / 10000.0

            # Benefit/cost ratio
            if cost > 1e-10:
                benefits.append(benefit / cost)
            else:
                benefits.append(0.0)

    # Calculate statistics
    avg_benefit_vs_cost = np.mean(np.array(benefits)) if len(benefits) > 0 else 0.0
    profitable_pct = np.sum(np.array(benefits) > 1.0) / len(benefits) if len(benefits) > 0 else 0.0

    return (cost_pct, avg_benefit_vs_cost, profitable_pct)


def calculate_episode_duration_stats(
    labels: np.ndarray,
    target_mean_duration: Tuple[int, int] = (5, 20),
    use_jit: bool = True
) -> Dict[str, float]:
    """
    Calculate comprehensive episode duration statistics.

    Args:
        labels: Regime labels (T,)
        target_mean_duration: Target range for mean duration (min, max) in bars
        use_jit: Use JIT-compiled version if available

    Returns:
        Dictionary with duration statistics
    """
    # Get episode durations
    if use_jit and NUMBA_AVAILABLE:
        durations_array = _calculate_episode_durations_jit(labels)
    else:
        durations = calculate_episode_durations(labels)
        durations_array = np.array(durations, dtype=np.int64)

    if len(durations_array) == 0:
        return {
            'mean_duration': 0.0,
            'median_duration': 0.0,
            'duration_cv': 0.0,
            'pct_short_episodes': 0.0,
            'pct_actionable': 0.0,
            'pct_in_target_range': 0.0,
            'target_quality_score': 0.0
        }

    # Calculate statistics with JIT if available
    if use_jit and NUMBA_AVAILABLE:
        mean_dur, median_dur, cv, pct_short, pct_action, pct_target = _calculate_episode_duration_stats_jit(
            durations_array.astype(np.float64),
            target_mean_duration[0],
            target_mean_duration[1]
        )
    else:
        # Fallback implementation
        mean_dur = float(np.mean(durations_array))
        median_dur = float(np.median(durations_array))
        std_dur = float(np.std(durations_array))
        cv = std_dur / mean_dur if mean_dur > 0 else 0.0

        n_episodes = len(durations_array)
        pct_short = float(np.sum(durations_array < 7) / n_episodes)
        pct_action = float(np.sum(durations_array >= 20) / n_episodes)
        pct_target = float(np.sum((durations_array >= target_mean_duration[0]) &
                                   (durations_array <= target_mean_duration[1])) / n_episodes)

    # Calculate target quality score
    # Reward durations within target range
    target_quality_score = pct_target + 0.5 * (1.0 - pct_short)

    return {
        'mean_duration': mean_dur,
        'median_duration': median_dur,
        'duration_cv': cv,
        'pct_short_episodes': pct_short,
        'pct_actionable': pct_action,
        'pct_in_target_range': pct_target,
        'target_quality_score': target_quality_score
    }


def calculate_transition_predictability(
    labels: np.ndarray,
    features: np.ndarray,
    lookback: int = 10,
    use_jit: bool = True
) -> float:
    """
    Calculate transition predictability score.

    Measures if regime transitions occur in similar market conditions.

    Args:
        labels: Regime labels (T,)
        features: Feature matrix (T, D)
        lookback: Number of bars to look back before transition
        use_jit: Use JIT-compiled version if available

    Returns:
        Predictability score [0, 1], higher is better
    """
    if len(labels) < 2 or len(features) == 0:
        return 0.0

    if use_jit and NUMBA_AVAILABLE:
        return _calculate_transition_predictability_jit(labels, features, lookback)
    else:
        # Fallback: simplified version
        transitions = []
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                transitions.append(i)

        if len(transitions) < 2:
            return 0.0

        # Simple heuristic: fewer transitions = more predictable
        max_transitions = len(labels) - 1
        predictability = 1.0 - (len(transitions) / max_transitions)
        return float(predictability)


def calculate_regime_autocorrelation(
    labels: np.ndarray,
    features: np.ndarray,
    max_lag: int = 20,
    use_jit: bool = True
) -> Dict[str, float]:
    """
    Calculate regime persistence via autocorrelation.

    Args:
        labels: Regime labels (T,)
        features: Feature matrix (T, D)
        max_lag: Maximum lag for autocorrelation
        use_jit: Use JIT-compiled version if available

    Returns:
        Dictionary with autocorrelation statistics
    """
    if len(labels) < max_lag or len(features) == 0:
        return {
            'mean_ac_lag1': 0.0,
            'mean_ac_lag5': 0.0,
            'half_life': 0.0,
            'persistence_score': 0.0
        }

    if use_jit and NUMBA_AVAILABLE:
        ac1, ac5, half_life = _calculate_regime_autocorrelation_jit(features, labels, max_lag)
    else:
        # Fallback: simplified version
        ac1, ac5, half_life = 0.0, 0.0, 0.0

    # Calculate composite persistence score
    persistence_score = 0.5 * ac1 + 0.3 * ac5 + 0.2 * min(1.0, half_life / 20.0)

    return {
        'mean_ac_lag1': ac1,
        'mean_ac_lag5': ac5,
        'half_life': half_life,
        'persistence_score': persistence_score
    }


def calculate_economic_transition_cost(
    labels: np.ndarray,
    returns: np.ndarray,
    transaction_cost_bps: float = 10.0,
    lookforward: int = 20,
    use_jit: bool = True
) -> Dict[str, float]:
    """
    Calculate economic cost of regime transitions.

    Args:
        labels: Regime labels (T,)
        returns: Return series (T,)
        transaction_cost_bps: Transaction cost in basis points
        lookforward: Bars to look forward after transition
        use_jit: Use JIT-compiled version if available

    Returns:
        Dictionary with economic cost statistics
    """
    if len(labels) < 2 or len(returns) == 0:
        return {
            'total_cost_pct': 0.0,
            'avg_benefit_vs_cost': 0.0,
            'profitable_transitions_pct': 0.0,
            'economic_efficiency': 0.0
        }

    if use_jit and NUMBA_AVAILABLE:
        cost_pct, benefit_cost, profitable_pct = _calculate_economic_transition_cost_jit(
            labels, returns, transaction_cost_bps, lookforward
        )
    else:
        # Fallback: simplified version
        cost_pct, benefit_cost, profitable_pct = 0.0, 0.0, 0.0

    # Calculate composite economic efficiency score
    economic_efficiency = (
        0.4 * (1.0 - min(1.0, cost_pct)) +
        0.3 * min(1.0, benefit_cost / 3.0) +
        0.3 * profitable_pct
    )

    return {
        'total_cost_pct': cost_pct,
        'avg_benefit_vs_cost': benefit_cost,
        'profitable_transitions_pct': profitable_pct,
        'economic_efficiency': economic_efficiency
    }


def calculate_comprehensive_temporal_score(
    labels: np.ndarray,
    features: np.ndarray,
    returns: Optional[np.ndarray] = None,
    target_mean_duration: Tuple[int, int] = (5, 20),
    use_jit: bool = True
) -> Dict[str, float]:
    """
    Calculate comprehensive temporal quality score with 5 enhanced metrics.

    Components (with weights):
    - Basic smoothness (30%): Penalizes rapid switching
    - Duration quality (25%): Encourages tradeable episode lengths (5-20 bars target)
    - Transition predictability (15%): Rewards predictable transitions
    - Regime persistence (15%): Rewards autocorrelation
    - Economic efficiency (15%): Rewards profitable transitions (if returns available)

    Args:
        labels: Regime labels (T,)
        features: Feature matrix (T, D)
        returns: Optional return series (T,)
        target_mean_duration: Target range for mean duration (min, max) in bars
        use_jit: Use JIT-compiled version if available

    Returns:
        Dictionary with comprehensive temporal score and components
    """
    scores = {}
    weights = {}

    # 1. Basic smoothness (30%)
    scores['smoothness'] = calculate_temporal_smoothness(labels, use_jit=use_jit)
    weights['smoothness'] = 0.30

    # 2. Duration quality (25%)
    duration_stats = calculate_episode_duration_stats(
        labels,
        target_mean_duration=target_mean_duration,
        use_jit=use_jit
    )
    scores['duration'] = (
        0.4 * (1.0 - duration_stats['pct_short_episodes']) +
        0.3 * duration_stats['pct_actionable'] +
        0.3 * duration_stats['target_quality_score']
    )
    weights['duration'] = 0.25

    # 3. Transition predictability (15%)
    scores['predictability'] = calculate_transition_predictability(
        labels,
        features,
        use_jit=use_jit
    )
    weights['predictability'] = 0.15

    # 4. Regime persistence (15%)
    ac_stats = calculate_regime_autocorrelation(
        labels,
        features,
        use_jit=use_jit
    )
    scores['persistence'] = ac_stats['persistence_score']
    weights['persistence'] = 0.15

    # 5. Economic efficiency (15%) - only if returns available
    if returns is not None and len(returns) > 0:
        econ_stats = calculate_economic_transition_cost(
            labels,
            returns,
            use_jit=use_jit
        )
        scores['economic'] = econ_stats['economic_efficiency']
        weights['economic'] = 0.15
    else:
        # Redistribute weight if no returns
        weights['smoothness'] += 0.075
        weights['duration'] += 0.075

    # Calculate weighted composite score
    total_score = sum(scores[k] * weights[k] for k in scores.keys())

    # Return comprehensive results
    return {
        'composite_temporal_score': total_score,
        'smoothness_score': scores['smoothness'],
        'duration_score': scores['duration'],
        'predictability_score': scores.get('predictability', 0.0),
        'persistence_score': scores.get('persistence', 0.0),
        'economic_score': scores.get('economic', 0.0),
        'duration_stats': duration_stats,
        'weights': weights
    }


# ===== COMPOSITE SCORE CALCULATION =====

def calculate_composite_score(
    temporal_smoothness: float,
    rolling_ll: float,
    economic_utility: float,
    cv_ratio: float,
    goals: Optional[ClusteringOptimizationGoals] = None,
    penalties: Optional[Dict[str, float]] = None,
    normalize: bool = True,
    labels: Optional[np.ndarray] = None,
    features: Optional[np.ndarray] = None,
    returns: Optional[np.ndarray] = None,
    use_comprehensive_temporal: bool = False,
    target_mean_duration: Tuple[int, int] = (5, 20)
) -> Union[float, Dict[str, Any]]:
    """
    Calculate weighted composite score from individual metrics.

    New Structure:
    - 33% Temporal Smoothness: Regime persistence (can use comprehensive temporal score)
    - 33% Economic Quality: 50% rolling LL + 50% Sharpe
    - 34% Statistical Quality: CV ratio (between/within variance)

    Args:
        temporal_smoothness: Temporal smoothness score [0, 1] (higher is better)
        rolling_ll: Rolling log-likelihood (higher is better)
        economic_utility: Economic Sharpe ratio (higher is better)
        cv_ratio: CV ratio = between/within cluster variance (higher is better)
        goals: Optional custom goals configuration
        penalties: Optional penalties dict
        normalize: Whether to normalize sub-components
        labels: Optional regime labels for comprehensive temporal scoring
        features: Optional feature matrix for comprehensive temporal scoring
        returns: Optional returns for comprehensive temporal scoring
        use_comprehensive_temporal: Use comprehensive temporal score (5 metrics)
        target_mean_duration: Target mean duration range (min, max) in bars

    Returns:
        Composite score (higher is better) or dict with detailed breakdown if comprehensive
    """
    if goals is None:
        goals = DEFAULT_CLUSTERING_GOALS

    weights = goals.get_weights_dict()

    # Use comprehensive temporal score if requested and data is available
    if use_comprehensive_temporal and labels is not None and features is not None:
        comprehensive_temporal_result = calculate_comprehensive_temporal_score(
            labels=labels,
            features=features,
            returns=returns,
            target_mean_duration=target_mean_duration,
            use_jit=True
        )
        temporal_normalized = comprehensive_temporal_result['composite_temporal_score']

        # Store comprehensive breakdown for return
        comprehensive_breakdown = comprehensive_temporal_result
    else:
        # Use simple temporal smoothness
        if normalize:
            temporal_normalized = temporal_smoothness  # Already in [0, 1]
        else:
            temporal_normalized = temporal_smoothness
        comprehensive_breakdown = None

    # Normalize other components if requested
    if normalize:
        # Rolling LL: typical range [-10, 0], normalize to [0, 1]
        rolling_ll_normalized = np.clip((rolling_ll + 10.0) / 10.0, 0, 1)

        # Economic utility (Sharpe): typical range [0, 3], normalize to [0, 1]
        sharpe_normalized = np.clip(economic_utility / 3.0, 0, 1)

        # CV ratio: typical range [5, 1000+], use log scale and normalize
        cv_ratio_normalized = np.clip(np.log10(max(cv_ratio, 1.0)) / 3.0, 0, 1)  # log10(1000) = 3
    else:
        rolling_ll_normalized = rolling_ll
        sharpe_normalized = economic_utility
        cv_ratio_normalized = cv_ratio

    # Calculate economic quality as composite of rolling LL and Sharpe
    # Sub-weights: 50% each
    economic_quality = 0.5 * rolling_ll_normalized + 0.5 * sharpe_normalized

    # Calculate weighted composite
    composite = (
        weights['temporal_smoothness'] * temporal_normalized +
        weights['economic_quality'] * economic_quality +
        weights['statistical_quality'] * cv_ratio_normalized
    )

    # Apply penalties
    if penalties is not None:
        total_penalty = sum(penalties.values())
        composite -= total_penalty

    # Return detailed breakdown if using comprehensive temporal
    if use_comprehensive_temporal and comprehensive_breakdown is not None:
        return {
            'composite_score': composite,
            'temporal_component': temporal_normalized,
            'economic_component': economic_quality,
            'statistical_component': cv_ratio_normalized,
            'comprehensive_temporal_breakdown': comprehensive_breakdown,
            'weights': weights,
            'penalties': penalties
        }

    return composite


# ===== PENALTY CALCULATOR =====

def calculate_episode_durations(regime_labels: np.ndarray) -> List[int]:
    """
    Calculate duration of each episode (consecutive same label).

    Uses JIT-compiled implementation for performance.

    Args:
        regime_labels: Regime assignments (T,)

    Returns:
        List of episode durations in bars
    """
    if len(regime_labels) == 0:
        return []

    # Use JIT-compiled version if available
    if NUMBA_AVAILABLE:
        durations_array = _calculate_episode_durations_jit(regime_labels)
        return durations_array.tolist()
    else:
        # Fallback to original implementation
        durations = []
        current_label = regime_labels[0]
        current_duration = 1

        for i in range(1, len(regime_labels)):
            if regime_labels[i] == current_label:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_label = regime_labels[i]
                current_duration = 1

        # Add final episode
        durations.append(current_duration)

        return durations


def calculate_gradual_duration_penalty(
    episode_durations: List[int],
    penalty_config: PenaltyConfig
) -> float:
    """
    Calculate gradual duration penalty based on episode lengths.
    
    Penalty structure:
    - 1-2 bars: Very high penalty (50.0 per episode)
    - 3-4 bars: High penalty (15.0 per episode)
    - 5-6 bars: No penalty (0.0 per episode)
    - 7+ bars: No penalty
    
    Args:
        episode_durations: List of episode durations
        penalty_config: Penalty configuration
        
    Returns:
        Total duration penalty
    """
    total_penalty = 0.0
    
    for duration in episode_durations:
        if duration <= 2:
            # Very high penalty for 1-2 bar episodes
            total_penalty += penalty_config.duration_penalty_1_2_bars
        elif duration <= 4:
            # High penalty for 3-4 bar episodes
            total_penalty += penalty_config.duration_penalty_3_4_bars
        elif duration <= 6:
            # Low penalty for 5-6 bar episodes
            total_penalty += penalty_config.duration_penalty_5_6_bars
        # No penalty for 7+ bars
    
    return total_penalty


def calculate_smooth_transition_metrics(
    regime_labels: np.ndarray,
    transition_probs: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate smooth transition metrics from regime labels.
    
    Args:
        regime_labels: Regime assignments (T,)
        transition_probs: Optional transition probabilities (T-1, n_regimes) if soft labels available
        
    Returns:
        Dictionary with transition metrics
    """
    metrics = {
        'n_transitions': 0,
        'mean_transition_probability': 0.0,
        'abrupt_transitions': 0,
        'smooth_transitions': 0
    }
    
    if len(regime_labels) < 2:
        return metrics
    
    # Count transitions
    transitions = np.diff(regime_labels) != 0
    n_transitions = np.sum(transitions)
    metrics['n_transitions'] = int(n_transitions)
    
    if transition_probs is not None and transition_probs.shape[0] == len(regime_labels) - 1:
        # Calculate transition probabilities at transition points
        transition_probs_at_transitions = []
        for i, is_transition in enumerate(transitions):
            if is_transition:
                # Get transition probability from previous to current regime
                prev_regime = int(regime_labels[i])
                curr_regime = int(regime_labels[i + 1])
                
                if (prev_regime >= 0 and curr_regime >= 0 and 
                    prev_regime < transition_probs.shape[1] and 
                    curr_regime < transition_probs.shape[1]):
                    prob = transition_probs[i, curr_regime]
                    transition_probs_at_transitions.append(prob)
        
        if len(transition_probs_at_transitions) > 0:
            metrics['mean_transition_probability'] = float(np.mean(transition_probs_at_transitions))
            
            # Count abrupt vs smooth transitions
            threshold = 0.3  # Threshold for smooth transition
            metrics['abrupt_transitions'] = int(np.sum(
                np.array(transition_probs_at_transitions) < threshold
            ))
            metrics['smooth_transitions'] = int(np.sum(
                np.array(transition_probs_at_transitions) >= threshold
            ))
    
    return metrics


def calculate_noise_handling_metrics(
    regime_labels: np.ndarray,
    noise_label: int = -1
) -> Dict[str, float]:
    """
    Calculate noise handling metrics.
    
    Args:
        regime_labels: Regime assignments (T,)
        noise_label: Label value for noise points (typically -1)
        
    Returns:
        Dictionary with noise metrics
    """
    metrics = {
        'n_noise_points': 0,
        'noise_ratio': 0.0,
        'n_assigned_noise': 0,
        'n_unassigned_noise': 0
    }
    
    if len(regime_labels) == 0:
        return metrics
    
    # Count noise points
    noise_mask = regime_labels == noise_label
    n_noise = np.sum(noise_mask)
    noise_ratio = float(n_noise / len(regime_labels))
    
    metrics['n_noise_points'] = int(n_noise)
    metrics['noise_ratio'] = noise_ratio
    
    # Check if noise points are assigned (all points with noise_label are considered unassigned)
    metrics['n_unassigned_noise'] = int(n_noise)
    metrics['n_assigned_noise'] = int(len(regime_labels) - n_noise)
    
    return metrics


def calculate_penalties(
    regime_labels: np.ndarray,
    n_total_samples: int,
    regime_durations: Optional[np.ndarray] = None,
    monthly_turnover: Optional[float] = None,
    ari_scores: Optional[List[float]] = None,
    calibration_error: Optional[float] = None,
    metric_cv_variation: Optional[float] = None,
    transition_probs: Optional[np.ndarray] = None,
    noise_label: int = -1,
    penalty_config: Optional[PenaltyConfig] = None
) -> Dict[str, float]:
    """
    Calculate penalties for pathological fits.
    
    Args:
        regime_labels: Regime assignments
        n_total_samples: Total number of samples
        regime_durations: Expected durations per regime (legacy parameter)
        monthly_turnover: Monthly turnover rate
        ari_scores: ARI scores across restarts
        calibration_error: CRPS or PIT calibration error
        metric_cv_variation: CV variation of metrics (std/mean)
        transition_probs: Optional transition probabilities for smooth transition evaluation
        noise_label: Label value for noise points (typically -1)
        penalty_config: Penalty configuration
        
    Returns:
        Dictionary of penalties
    """
    if penalty_config is None:
        penalty_config = PenaltyConfig()
    
    penalties = {}
    
    # Minimum occupancy penalty
    unique, counts = np.unique(regime_labels, return_counts=True)
    # Filter out noise label for occupancy calculation
    valid_mask = unique != noise_label
    valid_unique = unique[valid_mask]
    valid_counts = counts[valid_mask]
    
    if len(valid_unique) > 0:
        occupancies = valid_counts / n_total_samples
        min_occupancy = np.min(occupancies)
        if min_occupancy < penalty_config.min_occupancy_pct:
            penalties['min_occupancy'] = penalty_config.min_occupancy_penalty * \
                (penalty_config.min_occupancy_pct - min_occupancy)
    
    # Gradual duration penalty (enhanced with episode-based calculation)
    episode_durations = calculate_episode_durations(regime_labels)
    if len(episode_durations) > 0:
        # Filter out noise episodes for duration calculation
        # Calculate durations only for non-noise episodes
        non_noise_durations = []
        current_label = regime_labels[0]
        current_duration = 1
        
        for i in range(1, len(regime_labels)):
            if regime_labels[i] == current_label:
                current_duration += 1
            else:
                if current_label != noise_label:
                    non_noise_durations.append(current_duration)
                current_label = regime_labels[i]
                current_duration = 1
        
        # Add final episode if not noise
        if current_label != noise_label:
            non_noise_durations.append(current_duration)
        
        if len(non_noise_durations) > 0:
            gradual_penalty = calculate_gradual_duration_penalty(
                non_noise_durations, penalty_config
            )
            if gradual_penalty > 0:
                penalties['duration_episodes'] = gradual_penalty
    
    # Legacy minimum duration penalty (backward compatibility)
    if regime_durations is not None and len(regime_durations) > 0:
        min_duration = np.min(regime_durations)
        if min_duration < penalty_config.min_duration_bars:
            penalties['min_duration_legacy'] = penalty_config.min_duration_penalty * \
                (penalty_config.min_duration_bars - min_duration)
    
    # Turnover penalty
    if monthly_turnover is not None and monthly_turnover > penalty_config.max_monthly_turnover:
        penalties['turnover'] = penalty_config.turnover_penalty * \
            (monthly_turnover - penalty_config.max_monthly_turnover)
    
    # Stability penalty (ARI)
    if ari_scores is not None and len(ari_scores) > 0:
        median_ari = np.median(ari_scores)
        if median_ari < penalty_config.min_ari_stability:
            penalties['stability'] = penalty_config.stability_penalty * \
                (penalty_config.min_ari_stability - median_ari)
    
    # Calibration penalty
    if calibration_error is not None and calibration_error > penalty_config.max_calibration_error:
        penalties['calibration'] = penalty_config.calibration_penalty * \
            (calibration_error - penalty_config.max_calibration_error)
    
    # Metric stability penalty
    if metric_cv_variation is not None and metric_cv_variation > penalty_config.max_cv_variation:
        penalties['metric_stability'] = penalty_config.cv_variation_penalty * \
            (metric_cv_variation - penalty_config.max_cv_variation)
    
    # Smooth transitions penalty
    if penalty_config.smooth_transitions_enabled:
        transition_metrics = calculate_smooth_transition_metrics(
            regime_labels, transition_probs
        )
        
        # Penalize if mean transition probability is too low (abrupt transitions)
        if transition_metrics['mean_transition_probability'] > 0:
            if transition_metrics['mean_transition_probability'] < penalty_config.min_transition_probability:
                abrupt_factor = (penalty_config.min_transition_probability - 
                               transition_metrics['mean_transition_probability']) / \
                               penalty_config.min_transition_probability
                penalties['abrupt_transitions'] = penalty_config.abrupt_transition_penalty * abrupt_factor
        
        # Also penalize based on number of abrupt transitions
        if transition_metrics['abrupt_transitions'] > 0:
            # Penalize proportionally to number of abrupt transitions
            total_transitions = transition_metrics['n_transitions']
            if total_transitions > 0:
                abrupt_ratio = transition_metrics['abrupt_transitions'] / total_transitions
                penalties['abrupt_transition_ratio'] = penalty_config.abrupt_transition_penalty * abrupt_ratio
    
    # Noise handling penalty
    if penalty_config.noise_handling_enabled:
        noise_metrics = calculate_noise_handling_metrics(regime_labels, noise_label)
        
        # Penalize high noise ratio
        if noise_metrics['noise_ratio'] > penalty_config.max_noise_ratio:
            excess_noise = noise_metrics['noise_ratio'] - penalty_config.max_noise_ratio
            penalties['noise_ratio'] = penalty_config.noise_ratio_penalty * excess_noise
        
        # Penalize unassigned noise points
        if noise_metrics['n_unassigned_noise'] > 0:
            # Penalize per unassigned noise point (normalized)
            unassigned_ratio = noise_metrics['n_unassigned_noise'] / len(regime_labels)
            penalties['unassigned_noise'] = penalty_config.unassigned_noise_penalty * unassigned_ratio
    
    return penalties


# ===== VALIDATION UTILITIES =====

def validate_cluster_sizes(
    cluster_sizes: List[int],
    n_total_samples: int,
    targets: Optional[OptimizationTargets] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate cluster sizes meet constraints.
    
    Args:
        cluster_sizes: List of cluster sizes
        n_total_samples: Total number of samples
        targets: Optional custom targets
    
    Returns:
        Tuple of (all_valid, details)
    """
    if targets is None:
        targets = DEFAULT_OPTIMIZATION_TARGETS
    
    min_size = int(n_total_samples * targets.min_cluster_size_pct)
    max_size = int(n_total_samples * targets.max_cluster_size_pct)
    
    violations = []
    for i, size in enumerate(cluster_sizes):
        size_pct = size / n_total_samples
        if size < min_size:
            violations.append({
                'cluster': i,
                'size': size,
                'size_pct': size_pct,
                'violation': 'too_small',
                'threshold': targets.min_cluster_size_pct
            })
        elif size > max_size:
            violations.append({
                'cluster': i,
                'size': size,
                'size_pct': size_pct,
                'violation': 'too_large',
                'threshold': targets.max_cluster_size_pct
            })
    
    details = {
        'all_valid': len(violations) == 0,
        'min_size': min_size,
        'max_size': max_size,
        'min_size_pct': targets.min_cluster_size_pct,
        'max_size_pct': targets.max_cluster_size_pct,
        'violations': violations,
        'n_violations': len(violations)
    }
    
    return len(violations) == 0, details


def meets_optimization_constraints(
    rolling_ll: float,
    one_step_ll: float,
    sharpe: float,
    n_clusters: int,
    cluster_sizes: Optional[List[int]] = None,
    n_total_samples: Optional[int] = None,
    targets: Optional[OptimizationTargets] = None
) -> Tuple[bool, Dict[str, bool]]:
    """
    Check if metrics meet minimum constraints.
    
    Args:
        rolling_ll: Rolling log-likelihood
        one_step_ll: One-step log-likelihood
        sharpe: Sharpe ratio
        n_clusters: Number of clusters
        cluster_sizes: Optional list of cluster sizes
        n_total_samples: Optional total samples
        targets: Optional custom targets
    
    Returns:
        Tuple of (all_met, individual_checks)
    """
    if targets is None:
        targets = DEFAULT_OPTIMIZATION_TARGETS
    
    checks = {
        'rolling_ll': rolling_ll >= targets.min_rolling_ll,
        'one_step_ll': one_step_ll >= targets.min_one_step_ll,
        'sharpe': sharpe >= targets.min_sharpe,
        'cluster_count': targets.min_clusters <= n_clusters <= targets.max_clusters,
        'cluster_count_preferred': targets.target_clusters[0] <= n_clusters <= targets.target_clusters[1],
    }
    
    # Validate cluster sizes if provided
    if cluster_sizes is not None and n_total_samples is not None:
        sizes_valid, size_details = validate_cluster_sizes(cluster_sizes, n_total_samples, targets)
        checks['cluster_sizes_valid'] = sizes_valid
        checks['cluster_sizes_details'] = size_details
    
    all_met = all(v if isinstance(v, bool) else True for v in checks.values())
    
    return all_met, checks


def format_metrics_report(
    rolling_ll: float,
    one_step_ll: float,
    sharpe: float,
    n_clusters: int,
    composite_score: float,
    penalties: Optional[Dict[str, float]] = None,
    targets: Optional[OptimizationTargets] = None
) -> str:
    """
    Format metrics into a human-readable report.
    
    Args:
        rolling_ll: Rolling log-likelihood
        one_step_ll: One-step log-likelihood
        sharpe: Sharpe ratio
        n_clusters: Number of clusters
        composite_score: Composite score
        penalties: Optional penalties applied
        targets: Optional custom targets
    
    Returns:
        Formatted report string
    """
    if targets is None:
        targets = DEFAULT_OPTIMIZATION_TARGETS
    
    all_met, checks = meets_optimization_constraints(
        rolling_ll, one_step_ll, sharpe, n_clusters, targets=targets
    )
    
    report = []
    report.append("=" * 70)
    report.append("CLUSTERING OPTIMIZATION METRICS REPORT - PREDICTIVE & ECONOMIC")
    report.append("=" * 70)
    report.append(f"\nComposite Score: {composite_score:.4f}")
    report.append(f"Number of Clusters: {n_clusters}\n")
    report.append("Primary Metrics (33% each):")
    report.append(f"  Rolling Log-Likelihood:     {rolling_ll:.4f} (target: ≥{targets.min_rolling_ll:.2f}) {'✅' if checks['rolling_ll'] else '❌'}")
    report.append(f"  One-Step Log-Likelihood:    {one_step_ll:.4f} (target: ≥{targets.min_one_step_ll:.2f}) {'✅' if checks['one_step_ll'] else '❌'}")
    report.append(f"  Economic Utility (Sharpe):  {sharpe:.4f} (target: ≥{targets.min_sharpe:.2f}) {'✅' if checks['sharpe'] else '❌'}")
    
    if penalties:
        report.append("\nPenalties Applied:")
        for penalty_name, penalty_value in penalties.items():
            report.append(f"  {penalty_name}: -{penalty_value:.4f}")
    
    report.append(f"\nOverall Status: {'✅ ALL CONSTRAINTS MET' if all_met else '❌ SOME CONSTRAINTS NOT MET'}")
    report.append("=" * 70)
    
    return "\n".join(report)


# ===== PARETO FRONT UTILITIES =====

def create_pareto_front(
    trial_results: List[Dict[str, float]],
    trial_params: List[Dict[str, Any]],
    goals: Optional[ClusteringOptimizationGoals] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Create Pareto front from trial results.
    
    Args:
        trial_results: List of {metric_name: value} for each trial
        trial_params: List of parameters for each trial
        goals: Optional goals configuration
        
    Returns:
        List of non-dominated solutions on Pareto front, or None if unavailable
    """
    if not PARETO_AVAILABLE:
        logger.warning("⚠️ Pareto optimization not available. Install required dependencies.")
        return None
    
    if goals is None:
        goals = DEFAULT_CLUSTERING_GOALS
    
    # Create Solution objects
    solutions = []
    for result, params in zip(trial_results, trial_params):
        metrics = {
            'rolling_ll': result.get('rolling_ll', float('-inf')),
            'one_step_ll': result.get('one_step_ll', float('-inf')),
            'economic': result.get('sharpe', 0.0)
        }
        solutions.append(Solution(metrics=metrics, params=params))
    
    # Define objectives (all maximize after normalization)
    objectives = {
        'rolling_ll': 'max',
        'one_step_ll': 'max',
        'economic': 'max'
    }
    
    # Compute Pareto front
    pareto_front = ParetoFront()
    front_indices = pareto_front.pareto_front(solutions, objectives)
    
    # Extract Pareto-optimal solutions
    pareto_solutions = []
    for idx in front_indices:
        pareto_solutions.append({
            'metrics': trial_results[idx],
            'params': trial_params[idx],
            'trial_idx': idx
        })
    
    logger.info(f"✅ Pareto front: {len(pareto_solutions)} non-dominated solutions")
    
    return pareto_solutions


def select_knee_point(
    pareto_solutions: List[Dict[str, Any]],
    goals: Optional[ClusteringOptimizationGoals] = None
) -> Dict[str, Any]:
    """
    Select knee point from Pareto front (balanced tradeoff).
    
    Args:
        pareto_solutions: List of Pareto-optimal solutions
        goals: Optional goals configuration
        
    Returns:
        Selected solution at knee point
    """
    if len(pareto_solutions) == 0:
        raise ValueError("No Pareto solutions provided")
    
    if len(pareto_solutions) == 1:
        return pareto_solutions[0]
    
    if goals is None:
        goals = DEFAULT_CLUSTERING_GOALS
    
    # Extract metrics for normalization
    rolling_lls = [s['metrics']['rolling_ll'] for s in pareto_solutions]
    one_step_lls = [s['metrics']['one_step_ll'] for s in pareto_solutions]
    sharpes = [s['metrics']['sharpe'] for s in pareto_solutions]
    
    # Normalize to [0, 1]
    def normalize(values: np.ndarray) -> np.ndarray:
        """Normalize values to [0, 1] range."""
        min_val, max_val = np.min(values), np.max(values)
        if max_val == min_val:
            return np.ones_like(values) * 0.5
        return (np.array(values) - min_val) / (max_val - min_val)
    
    norm_rolling = normalize(rolling_lls)
    norm_one_step = normalize(one_step_lls)
    norm_sharpe = normalize(sharpes)
    
    # Calculate distance from ideal point (1, 1, 1)
    distances = []
    for i in range(len(pareto_solutions)):
        dist = np.sqrt(
            (1 - norm_rolling[i])**2 +
            (1 - norm_one_step[i])**2 +
            (1 - norm_sharpe[i])**2
        )
        distances.append(dist)
    
    # Knee point is closest to ideal
    knee_idx = int(np.argmin(distances))
    
    logger.info(f"✅ Selected knee point: trial {pareto_solutions[knee_idx]['trial_idx']}")
    logger.info(f"   Rolling LL: {rolling_lls[knee_idx]:.4f}")
    logger.info(f"   One-Step LL: {one_step_lls[knee_idx]:.4f}")
    logger.info(f"   Sharpe: {sharpes[knee_idx]:.4f}")
    
    return pareto_solutions[knee_idx]


def rank_pareto_solutions(
    pareto_solutions: List[Dict[str, Any]],
    goals: Optional[ClusteringOptimizationGoals] = None
) -> List[Tuple[int, float, Dict[str, Any]]]:
    """
    Rank Pareto solutions by composite score.
    
    Args:
        pareto_solutions: List of Pareto-optimal solutions
        goals: Optional goals configuration
        
    Returns:
        List of (rank, composite_score, solution) sorted by score
    """
    if goals is None:
        goals = DEFAULT_CLUSTERING_GOALS
    
    # Calculate composite scores
    scored_solutions = []
    for solution in pareto_solutions:
        composite = calculate_composite_score(
            rolling_ll=solution['metrics']['rolling_ll'],
            one_step_ll=solution['metrics']['one_step_ll'],
            economic_utility=solution['metrics']['sharpe'],
            goals=goals
        )
        scored_solutions.append((composite, solution))
    
    # Sort by score (descending)
    scored_solutions.sort(key=lambda x: x[0], reverse=True)
    
    # Add ranks
    ranked = [
        (rank + 1, score, sol)
        for rank, (score, sol) in enumerate(scored_solutions)
    ]
    
    return ranked


# ===== ROBUSTNESS VALIDATION =====

def validate_robustness(
    candidate_params: Dict[str, Any],
    data: pd.DataFrame,
    clustering_fn: Callable,
    n_seeds: int = 30,
    min_ari: float = 0.5
) -> Tuple[bool, Dict[str, float]]:
    """
    Validate robustness of clustering solution across multiple seeds.
    
    Args:
        candidate_params: Parameters to validate
        data: Data for clustering
        clustering_fn: Function that performs clustering
        n_seeds: Number of random seeds to test
        min_ari: Minimum median ARI required
        
    Returns:
        (is_robust, robustness_metrics)
    """
    if not SKLEARN_METRICS_AVAILABLE or adjusted_rand_score is None:
        logger.error("❌ sklearn.metrics not available. Cannot perform robustness validation.")
        return False, {'error': 'sklearn_not_available', 'median_ari': 0.0}
    
    logger.info(f"🔍 Validating robustness with {n_seeds} seeds...")
    
    # Perform clustering with different seeds
    all_labels = []
    for seed in range(n_seeds):
        params = candidate_params.copy()
        params['random_state'] = seed
        
        try:
            result = clustering_fn(data, **params)
            if hasattr(result, 'labels_'):
                labels = result.labels_
            elif isinstance(result, dict) and 'labels' in result:
                labels = result['labels']
            else:
                labels = result
            
            all_labels.append(labels)
        except Exception as e:
            logger.warning(f"⚠️ Clustering failed for seed {seed}: {e}")
            continue
    
    if len(all_labels) < 2:
        logger.error("❌ Not enough successful clustering runs for robustness check")
        return False, {'median_ari': 0.0, 'n_successful': len(all_labels)}
    
    # Calculate pairwise ARI
    ari_scores = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            ari = adjusted_rand_score(all_labels[i], all_labels[j])
            ari_scores.append(ari)
    
    # Robustness metrics
    median_ari = np.median(ari_scores)
    mean_ari = np.mean(ari_scores)
    std_ari = np.std(ari_scores)
    
    is_robust = median_ari >= min_ari
    
    metrics = {
        'median_ari': float(median_ari),
        'mean_ari': float(mean_ari),
        'std_ari': float(std_ari),
        'n_successful': len(all_labels),
        'n_comparisons': len(ari_scores)
    }
    
    if is_robust:
        logger.info(f"✅ Robust solution: median ARI = {median_ari:.4f}")
    else:
        logger.warning(f"⚠️ Unstable solution: median ARI = {median_ari:.4f} < {min_ari:.4f}")
    
    return is_robust, metrics


def statistical_significance_test(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
    n_bootstrap: int = 100,
    alpha: float = 0.10,
    random_state: Optional[int] = None
) -> Tuple[bool, float, Dict[str, float]]:
    """
    Test statistical significance of Sharpe improvement using block bootstrap.
    
    Args:
        strategy_returns: Returns from regime-aware strategy
        baseline_returns: Returns from baseline (e.g., buy-and-hold)
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        random_state: Optional random seed for reproducibility
        
    Returns:
        (is_significant, p_value, metrics)
    """
    logger.info(f"🔍 Testing statistical significance with {n_bootstrap} bootstrap samples...")
    
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # Calculate observed Sharpe difference
    def sharpe(returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        
        std_return = np.std(returns)
        if std_return == 0 or np.isnan(std_return) or np.isinf(std_return):
            return 0.0
        
        mean_return = np.mean(returns)
        if np.isnan(mean_return) or np.isinf(mean_return):
            return 0.0
        
        sharpe_val = (mean_return / std_return) * np.sqrt(Constants.TRADING_DAYS_PER_YEAR)
        
        if np.isnan(sharpe_val) or np.isinf(sharpe_val):
            return 0.0
        
        return float(sharpe_val)
    
    observed_strategy_sharpe = sharpe(strategy_returns)
    observed_baseline_sharpe = sharpe(baseline_returns)
    observed_diff = observed_strategy_sharpe - observed_baseline_sharpe
    
    # Block bootstrap (preserve autocorrelation)
    # Ensure minimum block size
    block_size = max(Constants.MIN_BLOCK_SIZE, int(np.sqrt(len(strategy_returns))))
    n_blocks = len(strategy_returns) // block_size
    
    if n_blocks < 2:
        logger.warning(
            f"⚠️ Insufficient data for block bootstrap: n_blocks={n_blocks}, "
            f"block_size={block_size}, len={len(strategy_returns)}"
        )
        # Fall back to simple bootstrap if blocks don't make sense
        block_size = max(1, len(strategy_returns) // 10)
        n_blocks = max(1, len(strategy_returns) // block_size)
        if n_blocks < 2:
            logger.warning(
                f"⚠️ Insufficient data for block bootstrap: n_blocks={n_blocks}, "
                f"block_size={block_size}, len={len(strategy_returns)}"
            )
            return False, 1.0, {
                'error': 'insufficient_data',
                'strategy_sharpe': observed_strategy_sharpe,
                'baseline_sharpe': observed_baseline_sharpe,
                'sharpe_diff': observed_diff
            }
    
    # Handle remainder samples
    remainder = len(strategy_returns) % block_size
    effective_length = len(strategy_returns) - remainder
    
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        # Sample blocks with replacement
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
        
        boot_strategy = []
        boot_baseline = []
        
        for block_idx in block_indices:
            start = block_idx * block_size
            end = min(start + block_size, effective_length)  # Ensure we don't go out of bounds
            if start < effective_length:
                boot_strategy.extend(strategy_returns[start:end])
                boot_baseline.extend(baseline_returns[start:end])
        
        # If we have remainder, randomly sample additional points
        if remainder > 0:
            additional_indices = np.random.choice(
                effective_length,
                size=min(remainder, effective_length),
                replace=True
            )
            boot_strategy.extend(strategy_returns[additional_indices])
            boot_baseline.extend(baseline_returns[additional_indices])
        
        boot_strategy = np.array(boot_strategy)
        boot_baseline = np.array(boot_baseline)
        
        boot_diff = sharpe(boot_strategy) - sharpe(boot_baseline)
        bootstrap_diffs.append(boot_diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # Calculate p-value (one-sided: strategy > baseline)
    p_value = np.mean(bootstrap_diffs <= 0)
    
    # Validate p-value
    if np.isnan(p_value) or np.isinf(p_value):
        logger.warning("⚠️ Invalid p-value calculated")
        p_value = 1.0  # Conservative default
    
    is_significant = p_value < alpha
    
    metrics = {
        'strategy_sharpe': observed_strategy_sharpe,
        'baseline_sharpe': observed_baseline_sharpe,
        'sharpe_diff': observed_diff,
        'p_value': float(p_value),
        'alpha': alpha,
        'bootstrap_mean': float(np.mean(bootstrap_diffs)),
        'bootstrap_std': float(np.std(bootstrap_diffs))
    }
    
    if is_significant:
        logger.info(f"✅ Significant improvement: p-value = {p_value:.4f} < {alpha:.4f}")
    else:
        logger.warning(f"⚠️ Not significant: p-value = {p_value:.4f} >= {alpha:.4f}")
    
    return is_significant, float(p_value), metrics


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Example: Using new predictive & economic goals
    goals = DEFAULT_CLUSTERING_GOALS
    targets = DEFAULT_OPTIMIZATION_TARGETS
    
    print("New Clustering Optimization Goals - Predictive & Economic:")
    print("=" * 70)
    for goal_name, goal_config in goals.get_all_goals().items():
        print(f"\n{goal_config.name}:")
        print(f"  Objective: {goal_config.objective.value}")
        print(f"  Weight: {goal_config.weight:.2%}")
        print(f"  Target Range: {goal_config.target_range}")
        print(f"  Normalization: {goal_config.normalization_method.value}")
        print(f"  Description: {goal_config.description}")
    
    print("\n\nStructural Constraints:")
    print("=" * 70)
    print(f"  Cluster Count Range: {goals.cluster_count_range}")
    print(f"  Min Cluster Size: {goals.min_cluster_size_pct:.1%}")
    print(f"  Max Cluster Size: {goals.max_cluster_size_pct:.1%}")
    
    print("\n\nCross-Validation Configuration:")
    print("=" * 70)
    print(f"  Strategy: {goals.cv_config.strategy.value}")
    print(f"  N Splits: {goals.cv_config.n_splits}")
    print(f"  Train Window: {goals.cv_config.train_months} months")
    print(f"  Val Window: {goals.cv_config.val_months} months")
    
    print("\n\nPenalty Configuration:")
    print("=" * 70)
    print(f"  Min Occupancy: {goals.penalty_config.min_occupancy_pct:.1%} (penalty: {goals.penalty_config.min_occupancy_penalty})")
    print(f"  Min Duration: {goals.penalty_config.min_duration_bars} bars (penalty: {goals.penalty_config.min_duration_penalty})")
    print(f"  Max Turnover: {goals.penalty_config.max_monthly_turnover}/month (penalty: {goals.penalty_config.turnover_penalty})")
    
    print("\n\nEnhanced Duration Penalties (Gradual):")
    print("=" * 70)
    print(f"  1-2 bars: Very high penalty ({goals.penalty_config.duration_penalty_1_2_bars} per episode)")
    print(f"  3-4 bars: High penalty ({goals.penalty_config.duration_penalty_3_4_bars} per episode)")
    print(f"  5-6 bars: No penalty ({goals.penalty_config.duration_penalty_5_6_bars} per episode)")
    print(f"  7+ bars: No penalty")
    
    print("\n\nSmooth Transitions Configuration:")
    print("=" * 70)
    print(f"  Enabled: {goals.penalty_config.smooth_transitions_enabled}")
    print(f"  Min Transition Probability: {goals.penalty_config.min_transition_probability}")
    print(f"  Abrupt Transition Penalty: {goals.penalty_config.abrupt_transition_penalty}")
    
    print("\n\nNoise Handling Configuration:")
    print("=" * 70)
    print(f"  Enabled: {goals.penalty_config.noise_handling_enabled}")
    print(f"  Max Noise Ratio: {goals.penalty_config.max_noise_ratio:.1%}")
    print(f"  Noise Ratio Penalty: {goals.penalty_config.noise_ratio_penalty}")
    print(f"  Unassigned Noise Penalty: {goals.penalty_config.unassigned_noise_penalty}")
    
    # Example: Calculate composite score
    print("\n\nExample Metrics Evaluation:")
    print("=" * 70)
    rolling_ll = -4.2
    one_step_ll = -2.1
    sharpe = 1.3
    n_clust = 6
    
    composite = calculate_composite_score(rolling_ll, one_step_ll, sharpe)
    
    # Example: Calculate penalties with new features
    print("\n\nExample Penalty Calculation with Enhanced Features:")
    print("=" * 70)
    
    # Create sample regime labels with short episodes and noise
    sample_labels = np.array([0, 0, 0, 1, 1, 0, 0, 2, 2, 2, 2, 2, -1, -1, 3, 3, 3, 3, 3, 3, 3])
    sample_penalties = calculate_penalties(
        regime_labels=sample_labels,
        n_total_samples=len(sample_labels),
        penalty_config=goals.penalty_config
    )
    
    print(f"Sample labels: {sample_labels}")
    print(f"\nPenalties calculated:")
    for penalty_name, penalty_value in sample_penalties.items():
        print(f"  {penalty_name}: {penalty_value:.4f}")
    
    # Example: Episode durations
    episode_durations = calculate_episode_durations(sample_labels)
    print(f"\nEpisode durations: {episode_durations}")
    
    # Example: Smooth transition metrics
    transition_metrics = calculate_smooth_transition_metrics(sample_labels)
    print(f"\nSmooth transition metrics:")
    print(f"  N transitions: {transition_metrics['n_transitions']}")
    print(f"  Mean transition probability: {transition_metrics['mean_transition_probability']:.4f}")
    print(f"  Abrupt transitions: {transition_metrics['abrupt_transitions']}")
    print(f"  Smooth transitions: {transition_metrics['smooth_transitions']}")
    
    # Example: Noise handling metrics
    noise_metrics = calculate_noise_handling_metrics(sample_labels)
    print(f"\nNoise handling metrics:")
    print(f"  N noise points: {noise_metrics['n_noise_points']}")
    print(f"  Noise ratio: {noise_metrics['noise_ratio']:.2%}")
    print(f"  Assigned noise: {noise_metrics['n_assigned_noise']}")
    print(f"  Unassigned noise: {noise_metrics['n_unassigned_noise']}")
    
    report = format_metrics_report(
        rolling_ll, one_step_ll, sharpe, n_clust, composite
    )
    print(report)
