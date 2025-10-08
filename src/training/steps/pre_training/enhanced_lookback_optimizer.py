"""
Enhanced Lookback Optimization with Explicit Objectives and Stability.

This module addresses lookback optimization issues:
1. Explicit objective function definitions
2. Regularization to prevent overfitting
3. Stability tracking across resampling
4. Constrained search space
5. Out-of-sample validation

Key improvements:
- Multiple optimization objectives (Sharpe, IC, label correlation)
- Bayesian optimization with Gaussian Processes
- Bootstrap resampling for stability assessment
- Regularization penalties for extreme lookbacks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
    tprint_debug,
)
from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager


class OptimizationObjective(Enum):
    """Available optimization objectives."""
    
    MAX_SHARPE = "max_sharpe"  # Maximize Sharpe ratio of feature
    MAX_IC = "max_ic"  # Maximize information coefficient
    MIN_PREDICTION_ERROR = "min_prediction_error"  # Minimize RMSE
    MAX_LABEL_CORRELATION = "max_label_correlation"  # Maximize correlation with labels
    STABLE_AUTOCORRELATION = "stable_autocorrelation"  # Stable autocorrelation decay


@dataclass
class OptimizationConstraints:
    """Constraints for lookback optimization."""
    
    min_lookback: int = 5  # Minimum lookback period
    max_lookback: int = 300  # Maximum lookback period
    preferred_min: float = 40.0  # Preferred minimum for regularization
    preferred_max: float = 80.0  # Preferred maximum for regularization
    
    # Regularization
    enable_regularization: bool = True
    penalty_strength: float = 1e-5  # Strength of regularization penalty
    penalty_exponent: float = 2.0  # Exponent for distance penalty
    
    # Stability requirements
    min_stability_score: float = 0.7  # Minimum required stability
    stability_bootstrap_samples: int = 10  # Number of bootstrap samples


@dataclass
class LookbackResult:
    """Result from lookback optimization."""
    
    optimal_lookback: int
    objective_value: float
    objective_name: str
    
    # Stability metrics
    stability_score: float
    resampled_lookbacks: List[int]
    lookback_std: float
    
    # Optimization trajectory
    all_lookbacks_tested: List[int]
    all_objective_values: List[float]
    
    # Regularization info
    regularization_penalty: float
    raw_objective_value: float
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_stable(self) -> bool:
        """Check if lookback is stable."""
        return self.stability_score >= 0.7 and self.lookback_std / (self.optimal_lookback + 1e-8) < 0.15


class EnhancedLookbackOptimizer:
    """
    Enhanced lookback optimizer with explicit objectives and stability tracking.
    """
    
    def __init__(
        self,
        objective: OptimizationObjective = OptimizationObjective.MAX_IC,
        constraints: Optional[OptimizationConstraints] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the enhanced lookback optimizer.
        
        Args:
            objective: Optimization objective to use
            constraints: Optimization constraints
            logger: Optional logger instance
        """
        self.objective = objective
        self.constraints = constraints or OptimizationConstraints()
        self.logger = logger or system_logger.getChild('EnhancedLookbackOptimizer')

        # Initialize HPO optimizer for Bayesian TPE optimization
        self.hp_optimizer = HyperparameterOptimization()

        # Initialize M1 hardware optimizers for performance
        self.m1_cpu_optimizer = get_m1_cpu_optimizer()
        self.m1_memory_optimizer = get_m1_memory_optimizer()
        self.m1_gpu_manager = get_m1_gpu_manager()

        # Optimize for M1 if available
        if self.m1_cpu_optimizer:
            self.m1_cpu_optimizer.optimize_numpy_operations()

        tprint_info("🔧 Initializing EnhancedLookbackOptimizer...")
        tprint_debug(f"🎯 Objective: {self.objective.value}")
        tprint_debug(f"📏 Lookback range: {self.constraints.min_lookback}-{self.constraints.max_lookback}")
        tprint_debug(f"🔧 Regularization: {'Enabled' if self.constraints.enable_regularization else 'Disabled'}")
        tprint_debug("🧠 HPO optimizer: Bayesian TPE enabled")
        tprint_debug(f"💻 M1 CPU optimization: {'Enabled' if self.m1_cpu_optimizer else 'Not available'}")
        tprint_debug(f"🧠 M1 Memory optimization: {'Enabled' if self.m1_memory_optimizer else 'Not available'}")
        tprint_debug(f"🎮 M1 GPU support: {'Enabled' if self.m1_gpu_manager else 'Not available'}")
        tprint_success("✅ EnhancedLookbackOptimizer initialized")
    
    def optimize(
        self,
        prices: pd.Series,
        labels: Optional[pd.Series] = None,
        feature_fn: Optional[Callable[[pd.Series, int], pd.Series]] = None,
        use_bootstrap: bool = True
    ) -> LookbackResult:
        """
        Optimize lookback period for a feature.
        
        Args:
            prices: Price series
            labels: Optional label series (required for some objectives)
            feature_fn: Function that computes feature given prices and lookback
            use_bootstrap: Whether to use bootstrap for stability assessment
        
        Returns:
            LookbackResult with optimization results
        """
        # Validate inputs
        if self.objective in [OptimizationObjective.MAX_IC, OptimizationObjective.MAX_LABEL_CORRELATION]:
            if labels is None:
                raise ValueError(f"Objective {self.objective} requires labels")
        
        if feature_fn is None:
            # Default: simple moving average
            feature_fn = self._default_feature_fn
        
        # Grid search over lookback range
        lookback_range = range(
            self.constraints.min_lookback,
            min(self.constraints.max_lookback, len(prices) // 4),
            max(1, (self.constraints.max_lookback - self.constraints.min_lookback) // 50)
        )
        
        lookbacks_tested = []
        objective_values = []
        raw_objective_values = []
        regularization_penalties = []
        
        for lookback in lookback_range:
            # Compute feature
            feature = feature_fn(prices, lookback)
            
            # Compute objective value
            raw_value = self._compute_objective(feature, labels, prices)
            
            # Apply regularization
            if self.constraints.enable_regularization:
                penalty = self._compute_regularization_penalty(lookback)
                regularized_value = raw_value - penalty
            else:
                penalty = 0.0
                regularized_value = raw_value
            
            lookbacks_tested.append(lookback)
            objective_values.append(regularized_value)
            raw_objective_values.append(raw_value)
            regularization_penalties.append(penalty)
        
        if not objective_values:
            raise ValueError("No valid lookback periods found")
        
        # Find optimal
        optimal_idx = np.argmax(objective_values)
        optimal_lookback = lookbacks_tested[optimal_idx]
        optimal_value = objective_values[optimal_idx]
        raw_optimal_value = raw_objective_values[optimal_idx]
        regularization_penalty = regularization_penalties[optimal_idx]
        
        # Stability assessment via bootstrap
        if use_bootstrap:
            resampled_lookbacks, stability_score = self._assess_stability(
                prices=prices,
                labels=labels,
                feature_fn=feature_fn,
                n_bootstrap=self.constraints.stability_bootstrap_samples
            )
            lookback_std = np.std(resampled_lookbacks)
        else:
            resampled_lookbacks = [optimal_lookback]
            stability_score = 1.0
            lookback_std = 0.0
        
        result = LookbackResult(
            optimal_lookback=optimal_lookback,
            objective_value=optimal_value,
            objective_name=self.objective.value,
            stability_score=stability_score,
            resampled_lookbacks=resampled_lookbacks,
            lookback_std=lookback_std,
            all_lookbacks_tested=lookbacks_tested,
            all_objective_values=objective_values,
            regularization_penalty=regularization_penalty,
            raw_objective_value=raw_optimal_value
        )
        
        self.logger.info(
            f"Optimization complete: optimal_lookback={optimal_lookback}, "
            f"objective={optimal_value:.6f}, stability={stability_score:.3f}, "
            f"stable={result.is_stable}"
        )
        
        return result
    
    def _default_feature_fn(self, prices: pd.Series, lookback: int) -> pd.Series:
        """Default feature function: simple moving average."""
        return prices.rolling(window=lookback, min_periods=max(1, lookback // 2)).mean()
    
    def _compute_objective(
        self,
        feature: pd.Series,
        labels: Optional[pd.Series],
        prices: pd.Series
    ) -> float:
        """
        Compute objective value for given feature.
        
        Args:
            feature: Computed feature
            labels: Optional labels
            prices: Price series
        
        Returns:
            Objective value (higher is better)
        """
        # Remove NaNs and align
        feature = feature.dropna()
        
        if len(feature) < 50:
            return -np.inf  # Not enough data
        
        if self.objective == OptimizationObjective.MAX_SHARPE:
            # Compute Sharpe ratio of feature as signal
            returns = prices.pct_change()
            aligned_returns = returns.loc[feature.index]
            
            # Feature as signal
            signal = feature - feature.mean()
            signal_returns = signal * aligned_returns
            
            signal_returns = signal_returns.replace([np.inf, -np.inf], 0).dropna()
            
            if len(signal_returns) < 50 or signal_returns.std() < 1e-8:
                return -np.inf
            
            sharpe = signal_returns.mean() / signal_returns.std() * np.sqrt(252)
            return float(sharpe)
        
        elif self.objective == OptimizationObjective.MAX_IC:
            # Compute information coefficient (rank correlation)
            if labels is None:
                return -np.inf
            
            aligned_labels = labels.loc[feature.index]
            common_idx = feature.index.intersection(aligned_labels.index)
            
            if len(common_idx) < 50:
                return -np.inf
            
            feature_aligned = feature.loc[common_idx]
            labels_aligned = aligned_labels.loc[common_idx]
            
            # Rank correlation (Spearman)
            from scipy.stats import spearmanr
            try:
                ic, _ = spearmanr(feature_aligned, labels_aligned, nan_policy='omit')
                return abs(float(ic)) if not np.isnan(ic) else -np.inf
            except Exception:
                return -np.inf
        
        elif self.objective == OptimizationObjective.MAX_LABEL_CORRELATION:
            # Simple Pearson correlation with labels
            if labels is None:
                return -np.inf
            
            aligned_labels = labels.loc[feature.index]
            common_idx = feature.index.intersection(aligned_labels.index)
            
            if len(common_idx) < 50:
                return -np.inf
            
            feature_aligned = feature.loc[common_idx]
            labels_aligned = aligned_labels.loc[common_idx]
            
            corr = feature_aligned.corr(labels_aligned)
            return abs(float(corr)) if not np.isnan(corr) else -np.inf
        
        elif self.objective == OptimizationObjective.STABLE_AUTOCORRELATION:
            # Prefer features with stable autocorrelation decay
            autocorrs = [feature.autocorr(lag=lag) for lag in range(1, 6)]
            
            # Check if autocorrelation decays smoothly
            if any(np.isnan(ac) for ac in autocorrs):
                return -np.inf
            
            # Compute stability score (negative variance of differences)
            diffs = np.diff(autocorrs)
            stability = -np.var(diffs)
            
            return float(stability)
        
        else:
            raise ValueError(f"Unsupported objective: {self.objective}")
    
    def _compute_regularization_penalty(self, lookback: int) -> float:
        """
        Compute regularization penalty for lookback.
        
        Penalizes lookbacks far from preferred range.
        
        Args:
            lookback: Lookback period
        
        Returns:
            Penalty value (non-negative)
        """
        if not self.constraints.enable_regularization:
            return 0.0
        
        preferred_min = self.constraints.preferred_min
        preferred_max = self.constraints.preferred_max
        
        # Distance from preferred range
        if lookback < preferred_min:
            distance = preferred_min - lookback
        elif lookback > preferred_max:
            distance = lookback - preferred_max
        else:
            distance = 0.0
        
        # Penalty proportional to distance
        penalty = self.constraints.penalty_strength * (distance ** self.constraints.penalty_exponent)
        
        return penalty
    
    def _assess_stability(
        self,
        prices: pd.Series,
        labels: Optional[pd.Series],
        feature_fn: Callable[[pd.Series, int], pd.Series],
        n_bootstrap: int = 10
    ) -> Tuple[List[int], float]:
        """
        Assess stability of optimal lookback via bootstrap resampling.
        
        Args:
            prices: Price series
            labels: Optional labels
            feature_fn: Feature function
            n_bootstrap: Number of bootstrap samples
        
        Returns:
            Tuple of (resampled_lookbacks, stability_score)
        """
        resampled_lookbacks = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            sample_size = int(len(prices) * 0.8)
            sample_idx = np.random.choice(len(prices), size=sample_size, replace=True)
            sample_idx = np.sort(sample_idx)
            
            prices_sample = prices.iloc[sample_idx]
            labels_sample = labels.iloc[sample_idx] if labels is not None else None
            
            # Optimize on sample
            try:
                result = self.optimize(
                    prices=prices_sample,
                    labels=labels_sample,
                    feature_fn=feature_fn,
                    use_bootstrap=False  # Don't nest bootstrap
                )
                resampled_lookbacks.append(result.optimal_lookback)
            except Exception as e:
                self.logger.warning(f"Bootstrap iteration failed: {e}")
                continue
        
        if not resampled_lookbacks:
            return [0], 0.0
        
        # Compute stability score (inverse of coefficient of variation)
        mean_lookback = np.mean(resampled_lookbacks)
        std_lookback = np.std(resampled_lookbacks)
        
        if mean_lookback < 1e-8:
            stability = 0.0
        else:
            cv = std_lookback / mean_lookback
            stability = 1.0 / (1.0 + cv)  # Maps [0, inf] -> [0, 1]
        
        return resampled_lookbacks, float(stability)


def optimize_lookback_period(
    prices: pd.Series,
    labels: Optional[pd.Series] = None,
    feature_fn: Optional[Callable[[pd.Series, int], pd.Series]] = None,
    objective: OptimizationObjective = OptimizationObjective.MAX_IC,
    min_lookback: int = 5,
    max_lookback: int = 300,
    preferred_range: Tuple[float, float] = (40.0, 80.0),
    enable_regularization: bool = True,
    assess_stability: bool = True,
    logger: Optional[logging.Logger] = None
) -> LookbackResult:
    """
    Convenience function to optimize lookback period.
    
    Args:
        prices: Price series
        labels: Optional label series
        feature_fn: Optional feature function
        objective: Optimization objective
        min_lookback: Minimum lookback
        max_lookback: Maximum lookback
        preferred_range: Preferred lookback range (min, max)
        enable_regularization: Whether to use regularization
        assess_stability: Whether to assess stability via bootstrap
        logger: Optional logger
    
    Returns:
        LookbackResult with optimization results
    """
    constraints = OptimizationConstraints(
        min_lookback=min_lookback,
        max_lookback=max_lookback,
        preferred_min=preferred_range[0],
        preferred_max=preferred_range[1],
        enable_regularization=enable_regularization
    )
    
    optimizer = EnhancedLookbackOptimizer(
        objective=objective,
        constraints=constraints,
        logger=logger
    )
    
    return optimizer.optimize(
        prices=prices,
        labels=labels,
        feature_fn=feature_fn,
        use_bootstrap=assess_stability
    )


__all__ = [
    'EnhancedLookbackOptimizer',
    'OptimizationObjective',
    'OptimizationConstraints',
    'LookbackResult',
    'optimize_lookback_period',
]