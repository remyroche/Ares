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

# VectorBT imports for efficient computations
try:
    from src.vectorbt import (
        vbt, rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, VECTORBT_AVAILABLE
    )
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

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

logger = logging.getLogger(__name__)


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
    
    # Minimum expected duration penalty
    min_duration_bars: int = 7  # For daily data (tunable)
    min_duration_penalty: float = 5.0
    
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


@dataclass
class ClusteringOptimizationGoals:
    """
    Unified clustering optimization goals - Predictive & Economic Focus.
    
    Primary optimization goals (33% each):
    1. Rolling/blocked predictive log-likelihood on held-out time blocks
    2. One-step-ahead log-likelihood or predictive density  
    3. Out-of-sample economic utility (Sharpe, risk-adjusted)
    
    Structural constraints:
    - Cluster count: 4-8 (preferred)
    - Cluster size: 2%-20% each
    """
    
    # ===== PRIMARY OPTIMIZATION GOALS (33% each) =====
    
    # Goal 1: Rolling Predictive Log-Likelihood
    # Higher is better - measures predictive quality on held-out blocks
    rolling_log_likelihood: GoalConfig = field(default_factory=lambda: GoalConfig(
        name="Rolling Log-Likelihood",
        objective=OptimizationObjective.MAXIMIZE,
        weight=0.33,  # 33% of composite score
        target_range=(-10.0, 0.0),  # Closer to 0 is better
        constraint_threshold=None,
        description="Rolling/blocked predictive log-likelihood on held-out time blocks",
        enable_normalization=True,
        normalization_method=NormalizationMethod.RANK,
        stability_threshold=0.4
    ))
    
    # Goal 2: One-Step-Ahead Log-Likelihood
    # Higher is better - measures one-step predictive density
    one_step_log_likelihood: GoalConfig = field(default_factory=lambda: GoalConfig(
        name="One-Step Log-Likelihood",
        objective=OptimizationObjective.MAXIMIZE,
        weight=0.33,  # 33% of composite score
        target_range=(-5.0, 0.0),  # Closer to 0 is better
        constraint_threshold=None,
        description="One-step-ahead log-likelihood averaged over held-out times",
        enable_normalization=True,
        normalization_method=NormalizationMethod.RANK,
        stability_threshold=0.4
    ))
    
    # Goal 3: Economic Utility (Out-of-Sample Sharpe)
    # Higher is better - measures economic value of regime-aware strategy
    economic_utility: GoalConfig = field(default_factory=lambda: GoalConfig(
        name="Economic Utility (Sharpe)",
        objective=OptimizationObjective.MAXIMIZE,
        weight=0.34,  # 34% of composite score (total=100%)
        target_range=(0.5, 3.0),  # Aim for Sharpe > 1.0
        constraint_threshold=0.5,  # Soft constraint: Sharpe >= 0.5
        description="Out-of-sample annualized Sharpe of regime-aware strategy",
        enable_normalization=True,
        normalization_method=NormalizationMethod.RANK,
        stability_threshold=0.4
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
            OptimizationGoal.ROLLING_LOG_LIKELIHOOD.value: self.rolling_log_likelihood,
            OptimizationGoal.ONE_STEP_LOG_LIKELIHOOD.value: self.one_step_log_likelihood,
            OptimizationGoal.ECONOMIC_UTILITY.value: self.economic_utility,
        }
    
    def get_primary_goals(self) -> Dict[str, GoalConfig]:
        """Get primary optimization goals (Predictive + Economic)."""
        return self.get_all_goals()
    
    def get_weights_dict(self) -> Dict[str, float]:
        """Get weights for composite score calculation."""
        return {
            'rolling_ll': self.rolling_log_likelihood.weight,
            'one_step_ll': self.one_step_log_likelihood.weight,
            'economic': self.economic_utility.weight,
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
            self.rolling_log_likelihood.weight = weights['rolling_ll'] / total
            self.one_step_log_likelihood.weight = weights['one_step_ll'] / total
            self.economic_utility.weight = weights['economic'] / total


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
    
    # Cluster count constraints
    min_clusters: int = 3  # Absolute minimum
    max_clusters: int = 10  # Absolute maximum
    target_clusters: Tuple[int, int] = (4, 8)  # Preferred range
    
    # Cluster size constraints (as percentage of total samples)
    min_cluster_size_pct: float = 0.02  # 2% minimum
    max_cluster_size_pct: float = 0.20  # 20% maximum
    
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
        
        samples_per_month = int(pd.Timedelta(days=30) / freq_td)
        
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
        
        self.logger.info(f"✅ Generated {len(splits)} rolling CV splits")
        return splits
    
    def _expanding_split(self, data: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate expanding window splits."""
        n_samples = len(data)
        val_size = n_samples // (self.cv_config.n_splits + 1)
        
        splits = []
        for i in range(1, self.cv_config.n_splits + 1):
            train_idx = np.arange(0, i * val_size)
            val_idx = np.arange(i * val_size, (i + 1) * val_size)
            
            if len(train_idx) >= self.cv_config.min_train_samples and \
               len(val_idx) >= self.cv_config.min_val_samples:
                splits.append((train_idx, val_idx))
        
        self.logger.info(f"✅ Generated {len(splits)} expanding CV splits")
        return splits
    
    def _blocked_split(self, data: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate non-overlapping blocked splits."""
        n_samples = len(data)
        block_size = n_samples // self.cv_config.n_splits
        
        splits = []
        for i in range(self.cv_config.n_splits - 1):
            # Use all other blocks for training
            val_start = i * block_size
            val_end = (i + 1) * block_size
            
            train_idx = np.concatenate([
                np.arange(0, val_start),
                np.arange(val_end, n_samples)
            ])
            val_idx = np.arange(val_start, val_end)
            
            if len(train_idx) >= self.cv_config.min_train_samples and \
               len(val_idx) >= self.cv_config.min_val_samples:
                splits.append((train_idx, val_idx))
        
        self.logger.info(f"✅ Generated {len(splits)} blocked CV splits")
        return splits


# ===== METRIC CALCULATORS =====

class MetricCalculator:
    """Calculator for predictive and economic metrics."""
    
    def __init__(self, use_vectorbt: bool = True):
        """Initialize metric calculator."""
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE
        self.logger = logging.getLogger(self.__class__.__name__)
    
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
                mean = regime_params[regime_id].get('mean', np.zeros(n_features))
                cov = regime_params[regime_id].get('cov', np.eye(n_features))
                
                # Multivariate normal log-likelihood
                try:
                    diff = data[t] - mean
                    ll = -0.5 * (
                        n_features * np.log(2 * np.pi) +
                        np.log(np.linalg.det(cov) + 1e-8) +
                        diff.T @ np.linalg.inv(cov + np.eye(n_features) * 1e-6) @ diff
                    )
                    ll_per_regime.append(ll)
                except:
                    ll_per_regime.append(-1e6)  # Numerical issues
            
            # Weighted log-likelihood (mixture)
            ll_per_regime = np.array(ll_per_regime)
            log_weights = np.log(regime_probs[t] + 1e-10)
            total_ll = logsumexp(ll_per_regime + log_weights)
            
            log_likelihoods.append(total_ll)
        
        log_likelihoods = np.array(log_likelihoods)
        
        # Filter out extreme outliers
        log_likelihoods = np.clip(log_likelihoods, -50, 50)
        
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
            
            if prev_regime in regime_params:
                mean = regime_params[prev_regime].get('mean', np.zeros(n_features))
                cov = regime_params[prev_regime].get('cov', np.eye(n_features))
                
                # Log-likelihood of current observation given previous regime
                try:
                    diff = data[t] - mean
                    ll = -0.5 * (
                        n_features * np.log(2 * np.pi) +
                        np.log(np.linalg.det(cov) + 1e-8) +
                        diff.T @ np.linalg.inv(cov + np.eye(n_features) * 1e-6) @ diff
                    )
                    log_likelihoods.append(ll)
                except:
                    log_likelihoods.append(-1e6)
        
        log_likelihoods = np.array(log_likelihoods)
        log_likelihoods = np.clip(log_likelihoods, -50, 50)
        
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
            'volatility': np.std(strategy_returns) * np.sqrt(252)  # Annualized
        }
    
    def _calculate_sharpe(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
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
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return np.zeros_like(values)
        return (values - mean) / std
    
    def _rank_normalize(self, values: np.ndarray) -> np.ndarray:
        """Rank-based normalization to [0, 1]."""
        if len(values) <= 1:
            return np.ones_like(values) * 0.5
        
        # Rank (higher is better after objective adjustment)
        ranks = stats.rankdata(values)
        # Normalize to [0, 1]
        normalized = (ranks - 1) / (len(ranks) - 1)
        return normalized
    
    def _robust_zscore_normalize(self, values: np.ndarray) -> np.ndarray:
        """Robust z-score using median and MAD."""
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad == 0:
            return np.zeros_like(values)
        return (values - median) / (1.4826 * mad)  # 1.4826 for normal consistency
    
    def _minmax_normalize(self, values: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]."""
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val == min_val:
            return np.ones_like(values) * 0.5
        return (values - min_val) / (max_val - min_val)


# ===== COMPOSITE SCORE CALCULATION =====

def calculate_composite_score(
    rolling_ll: float,
    one_step_ll: float,
    economic_utility: float,
    goals: Optional[ClusteringOptimizationGoals] = None,
    penalties: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate weighted composite score from individual metrics.
    
    Args:
        rolling_ll: Rolling log-likelihood (higher is better)
        one_step_ll: One-step log-likelihood (higher is better)
        economic_utility: Economic Sharpe ratio (higher is better)
        goals: Optional custom goals configuration
        penalties: Optional penalties dict
        
    Returns:
        Composite score (higher is better)
    """
    if goals is None:
        goals = DEFAULT_CLUSTERING_GOALS
    
    weights = goals.get_weights_dict()
    
    # Calculate weighted sum
    composite = (
        weights['rolling_ll'] * rolling_ll +
        weights['one_step_ll'] * one_step_ll +
        weights['economic'] * economic_utility
    )
    
    # Apply penalties
    if penalties is not None:
        total_penalty = sum(penalties.values())
        composite -= total_penalty
    
    return composite


# ===== PENALTY CALCULATOR =====

def calculate_penalties(
    regime_labels: np.ndarray,
    n_total_samples: int,
    regime_durations: Optional[np.ndarray] = None,
    monthly_turnover: Optional[float] = None,
    ari_scores: Optional[List[float]] = None,
    calibration_error: Optional[float] = None,
    metric_cv_variation: Optional[float] = None,
    penalty_config: Optional[PenaltyConfig] = None
) -> Dict[str, float]:
    """
    Calculate penalties for pathological fits.
    
    Args:
        regime_labels: Regime assignments
        n_total_samples: Total number of samples
        regime_durations: Expected durations per regime
        monthly_turnover: Monthly turnover rate
        ari_scores: ARI scores across restarts
        calibration_error: CRPS or PIT calibration error
        metric_cv_variation: CV variation of metrics (std/mean)
        penalty_config: Penalty configuration
        
    Returns:
        Dictionary of penalties
    """
    if penalty_config is None:
        penalty_config = PenaltyConfig()
    
    penalties = {}
    
    # Minimum occupancy penalty
    unique, counts = np.unique(regime_labels, return_counts=True)
    occupancies = counts / n_total_samples
    
    min_occupancy = np.min(occupancies)
    if min_occupancy < penalty_config.min_occupancy_pct:
        penalties['min_occupancy'] = penalty_config.min_occupancy_penalty * \
            (penalty_config.min_occupancy_pct - min_occupancy)
    
    # Minimum duration penalty
    if regime_durations is not None:
        min_duration = np.min(regime_durations)
        if min_duration < penalty_config.min_duration_bars:
            penalties['min_duration'] = penalty_config.min_duration_penalty * \
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
    def normalize(values):
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
    from sklearn.metrics import adjusted_rand_score
    
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
    alpha: float = 0.10
) -> Tuple[bool, float, Dict[str, float]]:
    """
    Test statistical significance of Sharpe improvement using block bootstrap.
    
    Args:
        strategy_returns: Returns from regime-aware strategy
        baseline_returns: Returns from baseline (e.g., buy-and-hold)
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        
    Returns:
        (is_significant, p_value, metrics)
    """
    logger.info(f"🔍 Testing statistical significance with {n_bootstrap} bootstrap samples...")
    
    # Calculate observed Sharpe difference
    def sharpe(returns):
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    observed_strategy_sharpe = sharpe(strategy_returns)
    observed_baseline_sharpe = sharpe(baseline_returns)
    observed_diff = observed_strategy_sharpe - observed_baseline_sharpe
    
    # Block bootstrap (preserve autocorrelation)
    block_size = int(np.sqrt(len(strategy_returns)))
    n_blocks = len(strategy_returns) // block_size
    
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        # Sample blocks with replacement
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
        
        boot_strategy = []
        boot_baseline = []
        
        for block_idx in block_indices:
            start = block_idx * block_size
            end = start + block_size
            boot_strategy.extend(strategy_returns[start:end])
            boot_baseline.extend(baseline_returns[start:end])
        
        boot_strategy = np.array(boot_strategy)
        boot_baseline = np.array(boot_baseline)
        
        boot_diff = sharpe(boot_strategy) - sharpe(boot_baseline)
        bootstrap_diffs.append(boot_diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # Calculate p-value (one-sided: strategy > baseline)
    p_value = np.mean(bootstrap_diffs <= 0)
    
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
    
    # Example: Calculate composite score
    print("\n\nExample Metrics Evaluation:")
    print("=" * 70)
    rolling_ll = -4.2
    one_step_ll = -2.1
    sharpe = 1.3
    n_clust = 6
    
    composite = calculate_composite_score(rolling_ll, one_step_ll, sharpe)
    
    report = format_metrics_report(
        rolling_ll, one_step_ll, sharpe, n_clust, composite
    )
    print(report)
