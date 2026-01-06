"""
Diversity Defense Objectives for Bagged LGBM Ensemble

This module implements a "Diversity Defense" approach for bagged LGBM models that trains 
a committee of 10 models with specialized roles (Direction, Magnitude, Risk).

Key Philosophy:
- Specialization: Some models only care about direction (Trend), some about size (Magnitude), 
  and some about risk (Sharpe).
- Consensus: We only trade when the committee agrees.
- Filters: We use their disagreement (MAD) as a measure of market chaos. High chaos = No trade.

The 3 "Specialist" Layers:
1. "Smart Money" (3x models): Sharpe Proxy (Vol-Normalized) - Maximizes PnL consistency
2. "Trend Follower" (3x models): Robust Tanh Loss - Maximizes directional accuracy
3. "Reality Check" (4x models): Huber / Asymmetric Huber - Minimizes regression error

Usage:
    from src.utils.ml_common.optimization.diversity_defense_objectives import (
        DiversityDefenseObjectives,
        DiversityDefenseAggregator,
        DiversityDefenseHPO
    )
    
    # Create objectives
    objectives = DiversityDefenseObjectives()
    
    # Get objective for model type
    sharpe_obj = objectives.get_sharpe_objective(lmbda=1.0, volatility=vol_series)
    tanh_obj = objectives.get_tanh_objective()
    huber_obj = objectives.get_huber_objective(delta=0.01)
    
    # Aggregate predictions
    aggregator = DiversityDefenseAggregator()
    final_signal = aggregator.aggregate(predictions_matrix, y_val)

References:
- Vol-normalized Sharpe optimization for risk-adjusted returns
- Tanh robust loss for directional accuracy (hit rate)
- Huber loss for outlier robustness
- Asymmetric penalties for wrong-direction errors

Author: Ares Trading System
Date: 2025-12-08
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.special import expit  # Numerically stable sigmoid

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

logger = logging.getLogger(__name__)

# Epsilon for numerical stability
EPS = 1e-8


# =============================================================================
# Custom Loss Functions (Focal Loss)
# =============================================================================

def get_focal_loss(alpha=0.25, gamma=2.0):
    """
    Returns a custom Focal Loss objective function for LightGBM.

    Parameters:
    -----------
    alpha : float
        Balancing factor. In the paper, alpha=0.25 works best for imbalanced data
        (even though class 1 is minority). It balances the scale of the loss.
    gamma : float
        Focusing parameter.
        gamma=0.0 -> Equivalent to standard LogLoss.
        gamma>0.0 -> Down-weights 'easy' examples (where p is close to y).

    Returns:
    --------
    focal_loss : function
        The objective function required by LightGBM (returns grad, hess).
    """

    def focal_loss(y_pred, dtrain):
        # 1. Retrieve true labels
        if hasattr(dtrain, 'get_label'):
            y_true = dtrain.get_label()
        else:
            y_true = dtrain

        # 2. Compute probabilities using robust sigmoid
        # y_pred comes as raw logits (margin scores) from LGBM
        p = expit(y_pred)

        # 3. Compute gradients and hessians term by term
        # We need the derivatives of the Loss with respect to the LOGITS (y_pred), not probability.

        # --- Precompute common terms ---
        # The 'focus' terms
        term1 = (1 - p) ** gamma
        term2 = p ** gamma

        # Log terms (add epsilon to avoid log(0))
        epsilon = 1e-15
        log_p = np.log(p + epsilon)
        log_1_p = np.log(1 - p + epsilon)

        # --- Gradient (First Derivative dL/dx) ---
        # For y=1: grad = alpha * term1 * (gamma * p * log_p + p - 1)
        # For y=0: grad = (1-alpha) * term2 * (gamma * (1-p) * log_1_p + p)

        # Vectorized implementation:
        grad = np.zeros_like(y_pred)

        # Case y=1
        pos_mask = (y_true == 1)
        grad[pos_mask] = alpha * term1[pos_mask] * (
            gamma * p[pos_mask] * log_p[pos_mask] + p[pos_mask] - 1
        )

        # Case y=0
        neg_mask = (y_true == 0)
        grad[neg_mask] = (1 - alpha) * term2[neg_mask] * (
            gamma * (1 - p[neg_mask]) * log_1_p[neg_mask] + p[neg_mask]
        )

        # --- Hessian (Second Derivative d2L/dx2) ---
        # Robust/Simplified Hessian (Recommended for Stability):
        # This keeps the 'shape' of the curvature but ensures it is positive.
        hess = np.zeros_like(y_pred)

        hess[pos_mask] = alpha * term1[pos_mask] * p[pos_mask] * (1 - p[pos_mask]) * (1 + p[pos_mask]*gamma)
        hess[neg_mask] = (1 - alpha) * term2[neg_mask] * p[neg_mask] * (1 - p[neg_mask]) * (1 + (1-p[neg_mask])*gamma)

        return grad, hess

    return focal_loss


def get_sharpe_weights(returns_series, window_hours, data_freq='15m'):
    """
    Calculates weights based on Rolling Sharpe logic efficiently.
    returns_series: pd.Series of raw returns
    window_hours: integer (e.g., 12 or 96 for 4 days)
    """
    # 1. Convert hours to periods based on data frequency
    freq_map = {'15m': 4, '1h': 1, '5m': 12, '1m': 60}
    multiplier = freq_map.get(data_freq, 4) # Default to 15m
    window = int(window_hours * multiplier)

    # 2. Efficient Rolling Volatility (Standard Deviation)
    # min_periods=window//2 allows weights to generate earlier without dropping too much data
    rolling_vol = returns_series.rolling(window=window, min_periods=1).std()

    # 3. Handle Zero Volatility (prevent division by zero)
    # Replace 0 with mean vol or small epsilon
    rolling_vol = rolling_vol.replace(0, rolling_vol.mean())

    # 4. Calculate Weight: Future Return / Past Volatility
    # (We align 'future return' with 'past vol' to ensure no leakage,
    # assuming returns_series is ALREADY the target return for that row)
    weights = np.abs(returns_series) / rolling_vol

    # 5. Clip weights to prevent explosions (e.g., huge return on tiny vol)
    weights = weights.clip(upper=3.0)

    # 6. Fill NaNs with mean or 1.0
    weights = weights.fillna(1.0)

    return weights.values


# =============================================================================
# Ensemble Sharpe Ratio (ESR) Scoring
# =============================================================================

def calculate_esr_score(
    avg_sharpe: float, 
    avg_corr: float, 
    n_models: int = 10
) -> float:
    """
    Calculate the Theoretical Ensemble Sharpe Ratio (ESR).
    
    This is the exact metric to maximize for risk-adjusted PnL.
    Based on portfolio theory: diversification provides a "free lunch"
    when assets (models) are uncorrelated.
    
    Formula: ESR = avg_sharpe * sqrt(n / (1 + (n-1) * corr))
    
    The diversity_gain factor represents how much variance is cancelled out
    by combining uncorrelated predictions.
    
    Args:
        avg_sharpe: Average Sharpe ratio across models
        avg_corr: Average pairwise correlation of model predictions
        n_models: Number of models in ensemble (default 10)
        
    Returns:
        Theoretical Ensemble Sharpe Ratio
        
    Example:
        >>> calculate_esr_score(avg_sharpe=0.5, avg_corr=0.3, n_models=10)
        1.02  # Diversity multiplier ~2x
        >>> calculate_esr_score(avg_sharpe=0.5, avg_corr=0.9, n_models=10)
        0.53  # Almost no diversity benefit
    """
    # Protect against div/0 or negative inputs
    if avg_corr < 0:
        avg_corr = 0.001  # Assume at least some minimal positive correlation
    if avg_corr > 0.999:
        avg_corr = 0.999  # Prevent division issues
    
    # The Diversity Multiplier (The "Free Lunch" from uncorrelated assets)
    # This factor represents how much variance is cancelled out.
    diversity_gain = np.sqrt(n_models / (1 + (n_models - 1) * avg_corr))
    
    # Projected Ensemble Sharpe
    esr = avg_sharpe * diversity_gain
    
    return esr


# Backward compatibility alias
def calculate_das_score(
    sharpe: float, 
    avg_corr: float, 
    target_floor: float = 0.5, 
    penalty_weight: float = 4.0
) -> float:
    """Deprecated: Use calculate_esr_score instead."""
    return calculate_esr_score(sharpe, avg_corr, n_models=10)


def calculate_simple_sharpe(
    returns: np.ndarray, 
    annualization_factor: float = np.sqrt(24 * 365)
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Strategy returns
        annualization_factor: sqrt(periods_per_year), default for hourly
        
    Returns:
        Annualized Sharpe ratio
    """
    if returns is None:
        return 0.0
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return 0.0
    std = np.std(returns)
    if not np.isfinite(std) or std < EPS:
        return 0.0
    mu = np.mean(returns)
    if not np.isfinite(mu):
        return 0.0
    return (mu / std) * annualization_factor


# =============================================================================
# Regime-Conditional Meta-Features
# =============================================================================

def generate_regime_meta_features(
    df: pd.DataFrame,
    close_col: str = 'close',
    high_col: str = 'high',
    low_col: str = 'low',
    volume_col: str = 'volume',
    atr_window: int = 14,
    ma_window: int = 50,
    vol_window: int = 24,
) -> pd.DataFrame:
    """
    Generate 3 regime-conditional meta-features that should bypass colsample_bytree.
    
    These features allow trees to create root splits based on market regime,
    making the ensemble "context-aware".
    
    Features:
    1. Volatility Regime: Current_ATR / 30d_Avg_ATR
       - Is market waking up (>1) or sleeping (<1)?
    2. Trendiness: ADX-like measure using |Price - MA50| / Price
       - Are we trending (high) or ranging (low)?
    3. Volume Shock: Current_Vol / 24h_Avg_Vol
       - Is liquidity rushing in (>1)?
    
    Args:
        df: DataFrame with OHLCV data
        close_col: Column name for close price
        high_col: Column name for high price
        low_col: Column name for low price
        volume_col: Column name for volume
        atr_window: Window for ATR calculation
        ma_window: Window for trend MA
        vol_window: Window for volume average
        
    Returns:
        DataFrame with 3 meta-feature columns:
        - meta_volatility_regime
        - meta_trendiness
        - meta_volume_shock
    """
    result = pd.DataFrame(index=df.index)
    
    # 1. Volatility Regime: Current ATR / 30d Avg ATR
    # ATR = Average True Range
    high = df[high_col] if high_col in df.columns else df[close_col]
    low = df[low_col] if low_col in df.columns else df[close_col]
    close = df[close_col]
    
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    current_atr = true_range.rolling(window=atr_window, min_periods=1).mean()
    avg_atr_30d = current_atr.rolling(window=30 * 24, min_periods=atr_window).mean()  # 30 days for hourly
    
    result['meta_volatility_regime'] = (current_atr / (avg_atr_30d + EPS)).clip(0.1, 10.0)
    
    # 2. Trendiness: |Price - MA50| / Price
    # High value = strong trend, Low value = ranging
    ma = close.rolling(window=ma_window, min_periods=1).mean()
    result['meta_trendiness'] = (np.abs(close - ma) / (close + EPS)).clip(0, 1.0)
    
    # 3. Volume Shock: Current Volume / 24h Avg Volume
    if volume_col in df.columns:
        volume = df[volume_col]
        avg_vol = volume.rolling(window=vol_window, min_periods=1).mean()
        result['meta_volume_shock'] = (volume / (avg_vol + EPS)).clip(0.1, 10.0)
    else:
        result['meta_volume_shock'] = 1.0  # Neutral if no volume data

    # 4. Market Efficiency (Trending vs Random Walk)
    # Kaufman Efficiency Ratio: Directional Move / Total Path Length
    # High (-> 1) = Efficient Trend, Low (-> 0) = Choppy/Noise
    change = (close - close.shift(ma_window)).abs()
    path_len = close.diff().abs().rolling(window=ma_window, min_periods=ma_window//2).sum()
    result['meta_efficiency'] = (change / (path_len + EPS)).clip(0.0, 1.0)

    # 5. Market Memory (Mean Reversion vs Momentum)
    # Rolling Autocorrelation of returns
    # Positive = Momentum/Trend, Negative = Mean Reversion
    ret = df[close_col].pct_change()
    result['meta_autocorr'] = ret.rolling(window=ma_window, min_periods=ma_window//2).corr(ret.shift(1)).fillna(0.0).clip(-0.99, 0.99)

    # Fill any NaN values
    result = result.fillna(0.0)
    
    return result


META_FEATURE_COLUMNS = ['meta_volatility_regime', 'meta_trendiness', 'meta_volume_shock', 'meta_efficiency', 'meta_autocorr']


# =============================================================================
# Enums and Configuration
# =============================================================================

class SpecialistType(Enum):
    """Types of specialist models in the ensemble."""
    SHARPE = "sharpe"           # Risk-adjusted PnL maximization (Smart Money)
    TANH = "tanh"               # Directional accuracy (Trend Follower)
    HUBER = "huber"             # Standard Huber regression (Reality Check)
    ASYMMETRIC_HUBER = "asymmetric_huber"  # Asymmetric Huber (Reality Check)


@dataclass
class SpecialistConfig:
    """Configuration for a specialist model."""
    specialist_type: SpecialistType
    count: int
    learning_rate_key: str
    role_description: str
    
    # Type-specific parameters
    lmbda: Optional[float] = None          # For Sharpe models (risk aversion)
    delta: Optional[float] = None          # For Huber models (outlier threshold)
    penalty_factor: Optional[float] = None  # For Asymmetric Huber (wrong-direction penalty)

    # Focal Loss parameters
    gamma: Optional[float] = None
    alpha: Optional[float] = None


@dataclass 
class DiversityDefenseConfig:
    """Full configuration for the Diversity Defense ensemble."""
    
    # Specialist layer configuration (10 models total)
    sharpe_count: int = 3          # "Smart Money" layer
    tanh_count: int = 3            # "Trend Follower" layer
    huber_standard_count: int = 2  # "Reality Check" - Standard (now Focal Loss)
    huber_asymmetric_count: int = 2  # "Reality Check" - Asymmetric (now Focal Loss)
    
    # Sharpe lambdas (risk aversion) - increasing values
    sharpe_lambdas: List[float] = field(default_factory=lambda: [0.5, 2.0, 8.0])
    
    # Huber/Focal parameters
    huber_delta: float = 0.01        # Outlier threshold for Huber (kept for legacy ref)
    asymmetric_penalty: float = 3.0   # Penalty factor for wrong-direction (2-5x typical)
    
    # Focal Loss params
    focal_gamma_standard: float = 2.0
    focal_alpha_standard: float = 0.25
    focal_gamma_asymmetric: float = 2.0
    focal_alpha_asymmetric: float = 0.75

    # Learning rates per layer
    lr_sharpe: float = 0.03          # Conservative default
    lr_tanh: float = 0.03
    lr_huber: float = 0.03
    
    # Aggregation parameters
    z_score_window: int = 1000       # Rolling window for Z-score normalization
    mad_floor: float = 0.25          # Soft floor for MAD (prevents div by zero)
    noise_threshold: float = 0.3     # Below this, signal = 0
    cap_threshold: float = 0.7       # Above this, signal capped
    
    # Kill Switch parameters
    kill_switch_high_corr: float = 0.90   # Correlation > threshold: diversity collapsed
    kill_switch_low_corr: float = 0.0     # Correlation < threshold: models confused
    kill_switch_window: int = 24          # Rolling window for correlation monitoring
    
    # Disagreement scaling parameters
    mad_decay_factor: float = 2.0         # Exponential decay factor for MAD
    mad_consensus_threshold: float = 0.1  # MAD below this = consensus lock
    mad_chaos_threshold: float = 0.5      # MAD above this = chaos
    mad_consensus_bonus: float = 1.5      # Multiplier when in consensus lock
    mad_chaos_penalty: float = 0.5        # Multiplier when in chaos regime
    
    # Orthogonality penalty parameters (for DAS scoring)
    correlation_threshold: float = 0.5   # Safe floor - correlations below this are free
    das_penalty_weight: float = 4.0      # Quadratic penalty weight for DAS formula
    orthogonality_penalty_strength: float = 0.1  # Linear penalty weight for HPO
    
    # Sample bagging fraction (row sampling via external loop)
    sample_fraction: float = 0.7     # 70% of samples per bag
    
    # LGBM base parameters
    # NOTE: Feature diversity is controlled via colsample_bytree, NOT external loop
    # Optimal range is 0.5-0.6 based on Diversity-Adjusted Sharpe analysis
    n_estimators: int = 300
    num_leaves: int = 40
    max_depth: int = 6
    subsample: float = 0.7           # Row sampling within LightGBM
    colsample_bytree: float = 0.5    # CRITICAL: Controls feature diversity (0.3-0.8 range)
    
    def get_specialist_configs(self) -> List[SpecialistConfig]:
        """Generate specialist configurations for all 10 models."""
        configs = []
        
        # Sharpe models (3x)
        # Using binary objective (needs weights calculated externally)
        for i, lmbda in enumerate(self.sharpe_lambdas[:self.sharpe_count]):
            configs.append(SpecialistConfig(
                specialist_type=SpecialistType.SHARPE,
                count=1,
                learning_rate_key='lr_sharpe',
                role_description=f"Smart Money #{i+1} (λ={lmbda})",
                lmbda=lmbda
            ))
        
        # Tanh models (3x)
        # Using binary objective (needs custom labels 0.3%, 0.6%, 0.9%)
        for i in range(self.tanh_count):
            configs.append(SpecialistConfig(
                specialist_type=SpecialistType.TANH,
                count=1,
                learning_rate_key='lr_tanh',
                role_description=f"Trend Follower #{i+1}"
            ))
        
        # Huber Standard models (2x) (Now Focal Loss - Standard)
        # gamma=[0.0, 2.0], alpha=0.25
        huber_gammas = [0.0, 2.0]
        for i in range(self.huber_standard_count):
            gamma = huber_gammas[i % len(huber_gammas)]
            configs.append(SpecialistConfig(
                specialist_type=SpecialistType.HUBER,
                count=1,
                learning_rate_key='lr_huber',
                role_description=f"Reality Check (Huber) #{i+1}",
                gamma=gamma,
                alpha=0.25
            ))
        
        # Huber Asymmetric models (2x) (Now Focal Loss - Asymmetric/Modified)
        # gamma=5.0, alpha=0.25 (model 3)
        # gamma=2.0, alpha=0.75 (model 4)
        huber_asym_configs = [
            {'gamma': 5.0, 'alpha': 0.25},
            {'gamma': 2.0, 'alpha': 0.75}
        ]
        for i in range(self.huber_asymmetric_count):
            conf = huber_asym_configs[i % len(huber_asym_configs)]
            configs.append(SpecialistConfig(
                specialist_type=SpecialistType.ASYMMETRIC_HUBER,
                count=1,
                learning_rate_key='lr_huber',
                role_description=f"Reality Check (Asymmetric) #{i+1}",
                gamma=conf['gamma'],
                alpha=conf['alpha']
            ))
        
        return configs


# =============================================================================
# Custom Objective Functions
# =============================================================================

class DiversityDefenseObjectives:
    """
    Factory for creating custom LightGBM objective functions for diversity defense.
    
    Each objective function returns (gradient, hessian) for LightGBM optimization.
    """
    
    def __init__(self, config: Optional[DiversityDefenseConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or DiversityDefenseConfig()
    
    @staticmethod
    def _get_labels_from_dataset(dataset) -> np.ndarray:
        """
        Extract labels from LightGBM dataset, handling both Dataset objects and arrays.
        
        Args:
            dataset: LightGBM Dataset object or numpy array
            
        Returns:
            Labels as numpy array
        """
        if hasattr(dataset, 'get_label'):
            return dataset.get_label().astype(np.float64)
        elif isinstance(dataset, np.ndarray):
            return dataset.astype(np.float64)
        else:
            return np.asarray(dataset, dtype=np.float64)
    
    # NOTE: Old regression factories kept for reference but unused in new Classifier flow
    # New Classifier flow uses 'binary' objective mostly, with weights/labels handling the logic.
    # Except for Huber/Focal which needs custom objective.

    @staticmethod
    def focal_loss_factory(alpha: float, gamma: float) -> Callable:
        """
        Create a Focal Loss objective function.
        
        Args:
            alpha: Balancing factor
            gamma: Focusing parameter
            
        Returns:
            Objective function (grad, hess) callable
        """
        # We reuse the global function defined above
        # Create a closure to capture alpha/gamma
        loss_fn = get_focal_loss(alpha, gamma)
        return loss_fn
    
    def get_objective_for_specialist(
        self, 
        specialist_config: SpecialistConfig,
        volatility: Optional[np.ndarray] = None
    ) -> Optional[Callable]:
        """
        Get the appropriate objective function for a specialist configuration.
        
        Args:
            specialist_config: Configuration for the specialist model
            volatility: Pre-calculated volatility (required for Sharpe objectives)
            
        Returns:
            Objective function callable or None for standard objective
        """
        # Sharpe and Tanh now use standard binary objective with custom labels/weights
        # So we return None here to let XGBoost/LGBM use 'binary'
        if specialist_config.specialist_type == SpecialistType.SHARPE:
            return None # Use 'binary'
        
        elif specialist_config.specialist_type == SpecialistType.TANH:
            return None # Use 'binary'
        
        elif specialist_config.specialist_type == SpecialistType.HUBER:
            # Use Focal Loss
            return self.focal_loss_factory(
                alpha=specialist_config.alpha or 0.25,
                gamma=specialist_config.gamma or 2.0
            )
        
        elif specialist_config.specialist_type == SpecialistType.ASYMMETRIC_HUBER:
            # Use Focal Loss with asymmetric params
            return self.focal_loss_factory(
                alpha=specialist_config.alpha or 0.25,
                gamma=specialist_config.gamma or 2.0
            )
        
        return None


# =============================================================================
# Aggregation Pipeline
# =============================================================================

class DiversityDefenseAggregator:
    """
    Aggregation pipeline for combining specialist model predictions.
    
    Implements:
    1. Z-Score standardization across models
    2. Median-based robust aggregation
    3. Smart Disagreement Scaling with exponential decay
    4. Bucketed execution sizing
    5. Kill Switch monitoring for live trading safety
    """
    
    # Kill Switch thresholds
    KILL_SWITCH_HIGH_CORR = 0.90   # Diversity collapsed - halve position
    KILL_SWITCH_LOW_CORR = 0.0    # Models confused - stop trading
    
    def __init__(self, config: Optional[DiversityDefenseConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or DiversityDefenseConfig()
        
        # Disagreement scaling parameters (from config, with sane defaults)
        self.mad_consensus_threshold = getattr(self.config, "mad_consensus_threshold", 0.1)
        self.mad_chaos_threshold = getattr(self.config, "mad_chaos_threshold", 0.5)
        self.mad_decay_factor = getattr(self.config, "mad_decay_factor", 2.0)
        self.mad_consensus_bonus = getattr(self.config, "mad_consensus_bonus", 1.5)
        self.mad_chaos_penalty = getattr(self.config, "mad_chaos_penalty", 0.5)
    
    def compute_z_matrix(
        self, 
        preds_matrix: np.ndarray,
        window: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute Z-score normalized prediction matrix.
        
        Args:
            preds_matrix: Shape (n_models, n_samples)
            window: Rolling window for standardization
            
        Returns:
            Z-score normalized matrix of same shape
        """
        # Transform probability [0, 1] to Direction [-1, 1]
        # Since we are now using Classifiers
        # 2*p - 1
        signal_matrix = 2.0 * preds_matrix - 1.0

        window = window or self.config.z_score_window
        z_out = []
        
        for row in signal_matrix:
            s = pd.Series(row)
            r = s.rolling(window=window, min_periods=100)
            mu = r.mean()
            sigma = r.std() + EPS
            z = (s - mu) / sigma
            z_out.append(z.fillna(0).values)
        
        return np.array(z_out)
    
    def compute_rolling_correlation(
        self,
        z_matrix: np.ndarray,
        window: Optional[int] = None
    ) -> Tuple[float, str]:
        """
        Compute rolling pairwise correlation for Kill Switch monitoring.
        
        Args:
            z_matrix: Z-score normalized predictions (n_models, n_samples)
            window: Rolling window (default 24 = 24 hours for hourly data)
            
        Returns:
            Tuple of (avg_correlation, kill_switch_status)
            Status: "normal", "danger_high_corr", "danger_low_corr"
        """
        if z_matrix.ndim != 2 or z_matrix.shape[0] < 2:
            return 0.5, "normal"
        
        # Determine effective window from argument or config
        window_eff = window or getattr(self.config, "kill_switch_window", 24)
        
        # Use last `window_eff` samples for recent correlation
        recent_z = z_matrix[:, -window_eff:] if z_matrix.shape[1] >= window_eff else z_matrix
        
        try:
            corr_matrix = np.corrcoef(recent_z)
            if corr_matrix.ndim != 2:
                return 0.5, "normal"
            
            upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            avg_corr = np.nanmean(upper_tri)
            
            if np.isnan(avg_corr):
                return 0.5, "normal"
            
            # Kill Switch logic from config thresholds
            high_thr = getattr(self.config, "kill_switch_high_corr", self.KILL_SWITCH_HIGH_CORR)
            low_thr = getattr(self.config, "kill_switch_low_corr", self.KILL_SWITCH_LOW_CORR)
            if avg_corr > high_thr:
                return avg_corr, "danger_high_corr"
            elif avg_corr < low_thr:
                return avg_corr, "danger_low_corr"
            else:
                return avg_corr, "normal"
                
        except Exception:
            return 0.5, "normal"
    
    def compute_correlation_penalty(
        self, 
        z_matrix: np.ndarray
    ) -> float:
        """
        Compute orthogonality penalty based on correlation of Z-scores.
        
        Args:
            z_matrix: Z-score normalized predictions (n_models, n_samples)
            
        Returns:
            Penalty value (0 if correlation below threshold or insufficient models)
        """
        # Need at least 2 models for correlation
        if z_matrix.ndim != 2 or z_matrix.shape[0] < 2:
            return 0.0
        
        # Compute correlation matrix
        try:
            corr_matrix = np.corrcoef(z_matrix)
            
            # Handle degenerate cases
            if corr_matrix.ndim != 2:
                return 0.0
            
            # Get upper triangular (excluding diagonal)
            upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            avg_corr = np.nanmean(upper_tri)
            
            # Handle NaN (can happen with constant predictions)
            if np.isnan(avg_corr):
                return 0.0
            
            # Thresholded penalty: only punish if avg_corr > threshold
            penalty = np.maximum(0.0, avg_corr - self.config.correlation_threshold)
            penalty *= self.config.orthogonality_penalty_strength
            
            return float(penalty)
        except Exception:
            return 0.0
    
    def smart_disagreement_multiplier(
        self,
        mad: np.ndarray,
        median_abs: np.ndarray
    ) -> np.ndarray:
        """
        Compute smart disagreement scaling multiplier.
        
        Instead of simple 1/(1+MAD), we use regime-aware scaling:
        - Low MAD + High Median: "Consensus Lock" (Trend) -> Scale UP
        - High MAD: "Chaos/Noise" -> Scale DOWN aggressively
        
        Uses exponential decay: multiplier = exp(-MAD * k)
        This punishes disagreement more aggressively than linear division.
        
        Args:
            mad: MAD values per sample
            median_abs: Absolute median Z-score per sample
            
        Returns:
            Multiplier array (0-1.5 range)
        """
        # Base multiplier: exponential decay based on MAD
        mad_floor = getattr(self.config, "mad_floor", 0.0)
        if mad_floor > 0.0:
            mad_effective = np.maximum(mad, mad_floor)
        else:
            mad_effective = mad
        base_multiplier = np.exp(-mad_effective * self.mad_decay_factor)
        
        # Consensus Lock bonus: Low MAD AND High Median = trend confidence
        # Scale up to configured bonus when in consensus lock
        consensus_mask = (mad < self.mad_consensus_threshold) & (median_abs > 0.3)
        consensus_bonus = np.where(consensus_mask, self.mad_consensus_bonus, 1.0)
        
        # Chaos penalty: Very high MAD = aggressive scale down
        # Already handled by exponential decay, but add extra penalty
        chaos_mask = mad > self.mad_chaos_threshold
        chaos_penalty = np.where(chaos_mask, self.mad_chaos_penalty, 1.0)
        
        # Final multiplier
        multiplier = base_multiplier * consensus_bonus * chaos_penalty
        
        # Clip to reasonable range
        return np.clip(multiplier, 0.0, 1.5)
    
    def robust_aggregate(
        self, 
        z_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform robust aggregation using median and smart MAD scaling.
        
        Uses exponential decay for disagreement penalty instead of 1/(1+MAD).
        
        Args:
            z_matrix: Z-score normalized predictions (n_models, n_samples)
            
        Returns:
            Tuple of (median_z, mad_z, raw_signal)
        """
        # Median Z-score (ignores outliers)
        med_z = np.median(z_matrix, axis=0)
        
        # MAD (Median Absolute Deviation) - measures disagreement
        mad_z = np.median(np.abs(z_matrix - med_z), axis=0)
        
        # Smart disagreement multiplier (replaces simple 1/(1+MAD))
        multiplier = self.smart_disagreement_multiplier(mad_z, np.abs(med_z))
        
        # Raw signal: Median * multiplier
        raw_signal = med_z * multiplier
        
        return med_z, mad_z, raw_signal
    
    def bucketed_sizing(
        self, 
        signal: np.ndarray
    ) -> np.ndarray:
        """
        Apply bucketed position sizing.
        
        - Signal < noise_threshold: Position = 0 (noise)
        - Signal between thresholds: Linear scaling
        - Signal > cap_threshold: Capped at cap_threshold
        
        Args:
            signal: Raw aggregated signal
            
        Returns:
            Position sizes with same shape
        """
        abs_sig = np.abs(signal)
        sign_sig = np.sign(signal)
        
        noise_floor = self.config.noise_threshold
        cap = self.config.cap_threshold
        
        # Vectorized bucketing
        magnitude = np.where(
            abs_sig < noise_floor, 
            0.0,
            np.where(abs_sig > cap, cap, abs_sig)
        )
        
        return sign_sig * magnitude
    
    def aggregate(
        self, 
        preds_matrix: np.ndarray,
        compute_penalty: bool = True
    ) -> Dict[str, Any]:
        """
        Full aggregation pipeline.
        
        Args:
            preds_matrix: Raw predictions (n_models, n_samples)
            compute_penalty: Whether to compute orthogonality penalty
            
        Returns:
            Dict with:
                - 'final_signal': Bucketed position sizes
                - 'raw_signal': Pre-bucket signal
                - 'median_z': Median Z-scores
                - 'mad_z': MAD values
                - 'z_matrix': Normalized prediction matrix
                - 'orthogonality_penalty': Correlation-based penalty
        """
        # Z-score normalization
        z_matrix = self.compute_z_matrix(preds_matrix)
        
        # Robust aggregation
        med_z, mad_z, raw_signal = self.robust_aggregate(z_matrix)
        
        # Bucketed sizing
        final_signal = self.bucketed_sizing(raw_signal)
        
        # Orthogonality penalty
        ortho_penalty = 0.0
        if compute_penalty:
            ortho_penalty = self.compute_correlation_penalty(z_matrix)
        
        return {
            'final_signal': final_signal,
            'raw_signal': raw_signal,
            'median_z': med_z,
            'mad_z': mad_z,
            'z_matrix': z_matrix,
            'orthogonality_penalty': ortho_penalty
        }


# =============================================================================
# HPO Objective for Meta-Model Optimization
# =============================================================================

class DiversityDefenseHPO:
    """
    Optuna-based HPO for tuning the Diversity Defense ensemble.
    
    Optimizes the ensemble as a single organism rather than individual models.
    Tuning Goal: Maximize Sharpe Ratio of final aggregated signal after fees.
    
    Key Parameters Tuned:
    - λ (Risk Aversion): How much should Sharpe models hate variance?
    - δ (Outlier Threshold): When should Huber models ignore extreme spikes?
    - Learning Rates: Specific speeds for each layer
    - Asymmetric Penalty: Wrong-direction penalty multiplier
    """
    
    def __init__(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray,
        config: Optional[DiversityDefenseConfig] = None,
        fee_rate: float = 0.0005,
        annualization_factor: float = np.sqrt(24 * 365)
    ):
        """
        Initialize HPO.
        
        Args:
            X: Feature matrix
            y: Target returns
            config: Base configuration
            fee_rate: Transaction fee rate (default: 5 bps)
            annualization_factor: For Sharpe calculation
        """
        self.X = X
        self.y = np.asarray(y)
        self.config = config or DiversityDefenseConfig()
        self.fee_rate = fee_rate
        self.annualization_factor = annualization_factor
        
        # Pre-calculate volatility for Sharpe normalization
        y_series = pd.Series(y)
        rolling_vol = y_series.rolling(window=1000, min_periods=1).std()
        global_std = float(y_series.std()) if len(y_series) > 1 else 0.0
        fallback_std = global_std if global_std > 0.0 else 1.0
        self.y_vol = rolling_vol.fillna(fallback_std).replace(0.0, fallback_std).values
        
        self.objectives = DiversityDefenseObjectives(self.config)
        self.aggregator = DiversityDefenseAggregator(self.config)
    
    def create_objective(self) -> Callable:
        """
        Create Optuna objective function.
        
        Returns:
            Callable that takes an Optuna trial and returns score
        """
        try:
            import optuna
            import lightgbm as lgb
            from sklearn.model_selection import TimeSeriesSplit
        except ImportError as e:
            raise ImportError(f"Required packages not available: {e}")
        
        def objective(trial: 'optuna.Trial') -> float:
            """Optuna objective function."""
            
            # A. Base LGBM parameters
            param_grid = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 600),
                'num_leaves': trial.suggest_int('num_leaves', 30, 80),
                'max_depth': trial.suggest_int('max_depth', 4, 8),
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'verbosity': -1,
                'n_jobs': 1
            }
            
            # B. Learning Rates (Critical Balance)
            lr_sharpe = trial.suggest_float('lr_sharpe', 1e-3, 0.05, log=True)
            lr_tanh = trial.suggest_float('lr_tanh', 1e-3, 0.05, log=True)
            lr_huber = trial.suggest_float('lr_huber', 1e-3, 0.05, log=True)
            
            # C. Strategy Parameters
            # Sharpe Lambdas (Risk Aversion)
            lmbda_1 = trial.suggest_float('lmbda_1', 0.01, 1.0)
            lmbda_2 = trial.suggest_float('lmbda_2', 1.0, 5.0)
            lmbda_3 = trial.suggest_float('lmbda_3', 5.0, 15.0)
            sharpe_lambdas = [lmbda_1, lmbda_2, lmbda_3]
            
            # Huber & Asymmetric Parameters
            delta = trial.suggest_float('huber_delta', 0.001, 0.02)
            asym_penalty = trial.suggest_float('asym_penalty', 2.0, 5.0)
            
            # Aggregation parameters
            mad_floor = trial.suggest_float('mad_floor', 0.1, 0.5)
            noise_threshold = trial.suggest_float('noise_threshold', 0.2, 0.4)
            
            # D. Cross-Validation
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(self.X):
                X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_train, y_val = self.y[train_idx], self.y[val_idx]
                vol_train = self.y_vol[train_idx]
                
                preds_store = []
                
                # 1. Sharpe Models (Vol-Normalized) - 3 models
                for lam in sharpe_lambdas:
                    # Note: Sharpe now uses binary objective in production flow, but for HPO
                    # we keep using the regression proxy or switch to binary if needed.
                    # For simplicity, we use binary objective here assuming y_train is returns
                    # and we convert to binary targets for HPO training.
                    # However, DiversityDefenseObjectives HPO currently uses regression logic.
                    # TODO: Update HPO to use binary classification logic if needed.

                    # For now, sticking to the existing pattern but acknowledging the shift
                    # The get_objective_for_specialist returns None for Sharpe (meaning binary)
                    # so we should use binary:logistic here too.

                    y_train_bin = (y_train > 0).astype(int)

                    model = lgb.LGBMClassifier(
                        learning_rate=lr_sharpe, 
                        objective='binary',
                        **param_grid
                    )
                    try:
                        # Sharpe weighting logic needs to be applied manually here
                        # weights = abs(return) / vol
                        w_train = np.abs(y_train) / (vol_train + EPS)
                        model.fit(X_train, y_train_bin, sample_weight=w_train)
                        preds_store.append(model.predict_proba(X_val)[:, 1])
                    except Exception:
                        preds_store.append(np.zeros(len(X_val)))
                
                # 2. Tanh Models - 3 models
                for i in range(3):
                    # Tanh logic: Thresholds 0.3%, 0.6%, 0.9%
                    threshold = [0.003, 0.006, 0.009][i]
                    y_train_bin = (y_train > threshold).astype(int)

                    model = lgb.LGBMClassifier(
                        learning_rate=lr_tanh, 
                        objective='binary',
                        random_state=42 + i,
                        **param_grid
                    )
                    try:
                        model.fit(X_train, y_train_bin)
                        preds_store.append(model.predict_proba(X_val)[:, 1])
                    except Exception:
                        preds_store.append(np.zeros(len(X_val)))
                
                # 3. Huber Standard Models - 2 models
                # Focal Loss
                fobj_huber = self.objectives.focal_loss_factory(alpha=0.25, gamma=2.0)
                y_train_bin = (y_train > 0).astype(int)
                for i in range(2):
                    # For custom objective, we use train method usually, or pass fobj to LGBMClassifier
                    # LGBMClassifier accepts `objective` as callable
                    model = lgb.LGBMClassifier(
                        learning_rate=lr_huber, 
                        objective=fobj_huber,
                        random_state=100 + i,
                        **param_grid
                    )
                    try:
                        model.fit(X_train, y_train_bin)
                        # Predict returns raw scores for custom objective, need sigmoid
                        raw_preds = model.predict(X_val, raw_score=True)
                        preds_store.append(expit(raw_preds))
                    except Exception:
                        preds_store.append(np.zeros(len(X_val)))
                
                # 4. Asymmetric Huber Models - 2 models
                # Focal Loss (Modified)
                fobj_asym = self.objectives.focal_loss_factory(alpha=0.75, gamma=2.0)
                y_train_bin = (y_train > 0).astype(int)
                for i in range(2):
                    model = lgb.LGBMClassifier(
                        learning_rate=lr_huber, 
                        objective=fobj_asym,
                        random_state=200 + i,
                        **param_grid
                    )
                    try:
                        model.fit(X_train, y_train_bin)
                        raw_preds = model.predict(X_val, raw_score=True)
                        preds_store.append(expit(raw_preds))
                    except Exception:
                        preds_store.append(np.zeros(len(X_val)))
                
                # E. Aggregation
                preds_matrix = np.array(preds_store)
                
                # Update aggregator config with trial params
                agg_config = DiversityDefenseConfig(
                    mad_floor=mad_floor,
                    noise_threshold=noise_threshold
                )
                aggregator = DiversityDefenseAggregator(agg_config)
                
                # Aggregate predictions to get final signal and Z-matrix
                agg_result = aggregator.aggregate(preds_matrix, compute_penalty=False)
                pos_size = agg_result['final_signal']
                z_matrix = agg_result['z_matrix']
                
                # F. Scoring: Sharpe after fees
                strategy_ret = pos_size * y_val
                turnover = np.abs(np.diff(pos_size, prepend=0))
                net_ret = strategy_ret - (turnover * self.fee_rate)
                
                if np.std(net_ret) < EPS:
                    sharpe = 0.0
                else:
                    sharpe = (np.mean(net_ret) / np.std(net_ret)) * self.annualization_factor
                
                # G. Diversity: compute average correlation across models
                if z_matrix.ndim == 2 and z_matrix.shape[0] >= 2:
                    try:
                        corr_matrix = np.corrcoef(z_matrix)
                        if corr_matrix.ndim == 2:
                            upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                            avg_corr = np.nanmean(upper_tri)
                        else:
                            avg_corr = 0.0
                    except Exception:
                        avg_corr = 0.0
                else:
                    avg_corr = 0.0
                if np.isnan(avg_corr):
                    avg_corr = 0.0
                
                # H. ESR (Ensemble Sharpe Ratio) score
                n_models_eff = z_matrix.shape[0] if z_matrix.ndim == 2 else preds_matrix.shape[0]
                esr = calculate_esr_score(sharpe, avg_corr, n_models=n_models_eff)
                
                cv_scores.append(esr)
            
            return np.mean(cv_scores) if cv_scores else 0.0
        
        return objective
    
    def optimize(
        self, 
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1
    ) -> Dict[str, Any]:
        """
        Run HPO optimization.
        
        Args:
            n_trials: Number of trials
            timeout: Maximum time in seconds
            n_jobs: Parallel jobs
            
        Returns:
            Dict with best parameters and study results
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for HPO")
        
        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(
            self.create_objective(),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'study': study
        }


# =============================================================================
# Diversity Sweep - Feature Fraction Optimization
# =============================================================================

class DiversitySweep:
    """
    Diversity vs. Performance Sweep for finding optimal colsample_bytree.
    
    Instead of a full hyperparameter search, we fix all "brain" parameters
    (learning rates, objectives) and only vary the "vision" parameter
    (colsample_bytree). We measure:
    - Sharpe: Did the performance hold up?
    - Correlation: Did the models actually become distinct?
    
    Test regimes:
    - Regime A (0.8): "The Clones" - High overlap
    - Regime B (0.5): "The Specialists" - Balanced (recommended)
    - Regime C (0.3): "The Blindfolded" - Aggressive diversity
    """
    
    # Fixed "reasonable" defaults - lock these to isolate feature sampling effect
    FIXED_PARAMS = {
        'n_estimators': 300,
        'num_leaves': 40,
        'max_depth': 6,
        'learning_rate': 0.03,
        'subsample': 0.7,
        'verbosity': -1,
        'n_jobs': -1,
    }
    
    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        colsample_settings: Optional[List[float]] = None,
        n_splits: int = 2,
        n_models: int = 5,
    ):
        """
        Initialize the diversity sweep.
        
        Uses ESR (Ensemble Sharpe Ratio) to find optimal colsample_bytree.
        ESR = avg_sharpe * sqrt(n / (1 + (n-1) * corr))
        
        Args:
            X: Feature matrix
            y: Target returns
            colsample_settings: List of colsample_bytree values to test
            n_splits: Number of time series CV splits
            n_models: Number of models in mini-ensemble (5 recommended for speed)
        """
        self.X = X
        self.y = np.asarray(y)
        self.colsample_settings = colsample_settings or [0.8, 0.6, 0.5, 0.4, 0.3]
        self.n_splits = n_splits
        self.n_models = n_models
        
        self.objectives = DiversityDefenseObjectives()
    
    @staticmethod
    def _sharpe_proxy(preds: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Simplified Sharpe proxy objective with fixed lambda=1.0."""
        # Note: In binary classifier mode, this proxy is less relevant,
        # but kept for legacy regression-based sweep if needed.
        # Ideally sweep should use the new classifier logic.
        r = DiversityDefenseObjectives._get_labels_from_dataset(dataset)
        p = preds.astype(np.float64)
        # Simple fixed lambda=1.0
        grad = -r + 2.0 * 1.0 * (r ** 2) * p
        hess = np.maximum(2.0 * 1.0 * (r ** 2), EPS)
        return grad, hess
    
    def _train_mini_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        col_fraction: float,
    ) -> np.ndarray:
        """
        Train a mini-ensemble and return predictions matrix.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            col_fraction: colsample_bytree value
            
        Returns:
            Predictions matrix (n_models, n_val_samples)
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM is required")
        
        preds_store = []
        
        # Common params for this run
        run_params = dict(self.FIXED_PARAMS)
        run_params['colsample_bytree'] = col_fraction
        
        # Distribution: 2 Sharpe, 1 Tanh, 2 Huber (Focal) for 5 models
        n_sharpe = max(1, self.n_models * 2 // 5)
        n_tanh = max(1, self.n_models * 1 // 5)
        n_huber = self.n_models - n_sharpe - n_tanh
        
        # 1. Sharpe Models (Binary + Weights)
        y_train_bin = (y_train > 0).astype(int)
        for i in range(n_sharpe):
            try:
                # Approximate Sharpe weighting
                w_train = np.abs(y_train) / (np.std(y_train) + EPS)
                model = lgb.LGBMClassifier(
                    objective='binary',
                    random_state=i,
                    **run_params
                )
                model.fit(X_train, y_train_bin, sample_weight=w_train)
                preds_store.append(model.predict_proba(X_val)[:, 1])
            except Exception:
                preds_store.append(np.zeros(len(X_val)))
        
        # 2. Tanh Models (Binary with thresholds)
        for i in range(n_tanh):
            try:
                thresh = 0.003
                y_train_bin_t = (y_train > thresh).astype(int)
                model = lgb.LGBMClassifier(
                    objective='binary',
                    random_state=42 + i,
                    **run_params
                )
                model.fit(X_train, y_train_bin_t)
                preds_store.append(model.predict_proba(X_val)[:, 1])
            except Exception:
                preds_store.append(np.zeros(len(X_val)))
        
        # 3. Huber/Focal Models
        focal_obj = self.objectives.focal_loss_factory(alpha=0.25, gamma=2.0)
        y_train_bin_f = (y_train > 0).astype(int)
        for i in range(n_huber):
            try:
                model = lgb.LGBMClassifier(
                    objective=focal_obj,
                    random_state=100 + i,
                    **run_params
                )
                model.fit(X_train, y_train_bin_f)
                raw = model.predict(X_val, raw_score=True)
                preds_store.append(expit(raw))
            except Exception:
                preds_store.append(np.zeros(len(X_val)))
        
        return np.array(preds_store)
    
    def run(self, verbose: bool = True) -> pd.DataFrame:
        """
        Run the diversity sweep.
        
        Args:
            verbose: Print progress
            
        Returns:
            DataFrame with columns: Fraction, Sharpe, Avg_Corr, ESR_Score
        """
        try:
            from sklearn.model_selection import TimeSeriesSplit
        except ImportError:
            raise ImportError("scikit-learn is required")
        
        results = []
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        if verbose:
            print(f"{'Fraction':<10} | {'Sharpe':<10} | {'Avg Corr':<10} | {'ESR Score':<12}")
            print("-" * 55)
        
        for col_fraction in self.colsample_settings:
            fold_sharpes = []
            fold_corrs = []
            
            for train_idx, val_idx in tscv.split(self.X):
                X_train = self.X.iloc[train_idx]
                X_val = self.X.iloc[val_idx]
                y_train = self.y[train_idx]
                y_val = self.y[val_idx]
                
                # Train mini-ensemble
                preds_matrix = self._train_mini_ensemble(X_train, y_train, X_val, col_fraction)
                
                if len(preds_matrix) == 0:
                    continue
                
                # A. Z-Score (Simple batch normalization)
                # Now preds are Probabilities [0,1]
                # Convert to Signal [-1, 1] before correlation/sharpe
                sig_matrix = 2.0 * preds_matrix - 1.0

                z_matrix = []
                for row in sig_matrix:
                    z = (row - np.mean(row)) / (np.std(row) + EPS)
                    z_matrix.append(z)
                z_matrix = np.array(z_matrix)
                
                # B. Calculate Correlation (Diversity Metric)
                if z_matrix.shape[0] >= 2:
                    corr_mat = np.corrcoef(z_matrix)
                    n_models = z_matrix.shape[0]
                    avg_corr = np.mean(corr_mat[np.triu_indices(n_models, 1)])
                    if np.isnan(avg_corr):
                        avg_corr = 0.0
                else:
                    avg_corr = 0.0
                fold_corrs.append(avg_corr)
                
                # C. Calculate Sharpe (Performance Metric)
                med_z = np.median(z_matrix, axis=0)
                mad_z = np.median(np.abs(z_matrix - med_z), axis=0)
                # Use exponential decay for disagreement
                multiplier = np.exp(-mad_z * 2.0)
                final_sig = med_z * multiplier
                
                ret = final_sig * y_val
                sharpe = calculate_simple_sharpe(ret)
                fold_sharpes.append(sharpe)
            
            # Aggregate using ESR (Ensemble Sharpe Ratio)
            avg_sharpe = np.mean(fold_sharpes) if fold_sharpes else 0.0
            avg_corr = np.mean(fold_corrs) if fold_corrs else 0.0
            esr_score = calculate_esr_score(
                avg_sharpe, avg_corr, 
                n_models=self.n_models
            )
            
            results.append({
                'Fraction': col_fraction,
                'Sharpe': avg_sharpe,
                'Avg_Corr': avg_corr,
                'ESR_Score': esr_score
            })
            
            if verbose:
                print(f"{col_fraction:<10.2f} | {avg_sharpe:<10.4f} | {avg_corr:<10.4f} | {esr_score:<12.4f}")
        
        df_results = pd.DataFrame(results)
        
        # Find optimal based on ESR (Ensemble Sharpe Ratio)
        if len(df_results) > 0:
            esr_scores = pd.to_numeric(df_results['ESR_Score'], errors='coerce')
            if esr_scores.notna().any():
                best_idx = esr_scores.idxmax()
                best_fraction = float(df_results.loc[best_idx, 'Fraction'])
                if verbose:
                    print("-" * 55)
                    print(
                        f"✅ Optimal colsample_bytree: {best_fraction:.2f} "
                        f"(ESR Score: {float(esr_scores.loc[best_idx]):.4f})"
                    )
            elif verbose:
                print("-" * 55)
                print("⚠️ No valid ESR_Score values; using default colsample_bytree=0.50")
        
        return df_results
    
    def get_optimal_fraction(self) -> float:
        """
        Run sweep and return the optimal colsample_bytree value.
        
        Returns:
            Optimal colsample_bytree value
        """
        df = self.run(verbose=False)
        if len(df) == 0:
            return 0.5  # Default
        esr_scores = pd.to_numeric(df['ESR_Score'], errors='coerce')
        if not esr_scores.notna().any():
            return 0.5
        best_idx = esr_scores.idxmax()
        return float(df.loc[best_idx, 'Fraction'])


# =============================================================================
# Integration Helper: Create Full Ensemble
# =============================================================================

def create_diversity_defense_ensemble(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    config: Optional[DiversityDefenseConfig] = None,
    sample_weights: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Create a full Diversity Defense ensemble.
    
    Feature diversity is controlled via colsample_bytree (NOT external feature loops).
    This uses LightGBM's internal column sampling which is more efficient and 
    ensures different trees within each model also see different feature subsets.
    
    Row sampling is done externally via sample_fraction for bootstrap diversity.
    
    Args:
        X_train: Training features
        y_train: Training targets (returns)
        config: Ensemble configuration
        sample_weights: Optional sample weights
        verbose: Print progress
        
    Returns:
        Dict with:
            - 'models': List of trained models
            - 'specialist_types': List of specialist types
            - 'config': Configuration used
            - 'n_features': Number of features (all models use full feature set)
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM is required")
    
    config = config or DiversityDefenseConfig()
    objectives = DiversityDefenseObjectives(config)
    
    # Pre-calculate volatility for Sharpe objectives if needed
    # Note: In classifier mode we calculate weights per model iteration often
    # but we can reuse this logic
    
    n_samples, n_features = X_train.shape
    rng = np.random.RandomState(42)
    
    if isinstance(X_train, pd.DataFrame):
        meta_feature_columns = [c for c in META_FEATURE_COLUMNS if c in X_train.columns]
        base_feature_columns = [c for c in X_train.columns if c not in meta_feature_columns]
    else:
        meta_feature_columns = []
        base_feature_columns = list(range(n_features))
    
    models = []
    specialist_types = []
    
    specialist_configs = config.get_specialist_configs()
    
    # Base params
    base_params = {
        'n_estimators': config.n_estimators,
        'num_leaves': config.num_leaves,
        'max_depth': config.max_depth,
        'subsample': config.subsample,
        'colsample_bytree': config.colsample_bytree,
        'verbosity': -1,
        'n_jobs': -1
    }
    
    if verbose:
        logger.info(f"Training {len(specialist_configs)} specialists with colsample_bytree={config.colsample_bytree}")
    
    for i, spec_config in enumerate(specialist_configs):
        if verbose:
            logger.info(f"Training specialist {i+1}/{len(specialist_configs)}: {spec_config.role_description}")
        
        # Get learning rate
        lr = getattr(config, spec_config.learning_rate_key)
        
        # Get objective
        fobj = objectives.get_objective_for_specialist(spec_config)
        
        # Row sampling
        n_rows_sub = max(10, int(round(config.sample_fraction * n_samples)))
        row_idx = np.sort(rng.choice(n_samples, size=n_rows_sub, replace=False))
        
        if isinstance(X_train, pd.DataFrame):
            X_bag = X_train.iloc[row_idx]
        else:
            X_bag = X_train[row_idx]
        
        # Original Targets (Returns)
        y_bag_raw = y_train[row_idx]
        
        # Logic to prepare targets/weights for Classifier based on Specialist Type
        y_bag_target = None
        sample_weight_bag = None

        if spec_config.specialist_type == SpecialistType.SHARPE:
            # Target: Binary (>0)
            y_bag_target = (y_bag_raw > 0).astype(int)
            # Weights: Future_Return / Volatility
            # Need Volatility of y_bag_raw
            # We use the helper function, assuming y_bag_raw is a time series (might not be if shuffled)
            # But here we are bootstrapping. Best effort: use pre-calc global weights if possible or simple vol.
            # Assuming y_train was chronological before this function, we can slice pre-calc weights.

            # Better approach: Pass full weights in sample_weights if pre-calculated,
            # Or assume y_bag_raw has enough structure.
            # Since we shuffle in bagging, we lose time structure for rolling vol calculation.
            # The caller should ideally provide volatility-based weights if possible.
            # Fallback: calculate weights on the bag (might be noisy) or use simple abs return.

            # Using simple Sharpe proxy: |Return| / Global_Std
            vol = np.std(y_bag_raw) + EPS
            sample_weight_bag = np.abs(y_bag_raw) / vol
            sample_weight_bag = np.clip(sample_weight_bag, 0, 3.0)

        elif spec_config.specialist_type == SpecialistType.TANH:
            # Target: Binary (> Threshold)
            # Thresholds: 0.3%, 0.6%, 0.9% for the 3 models
            # We map the i-th Tanh model to a threshold
            # Count how many Tanh models we have processed so far or check index
            # This relies on order.
            # Let's use simple logic: i % 3 maps to 0.003, 0.006, 0.009
            threshold = [0.003, 0.006, 0.009][i % 3]
            y_bag_target = (y_bag_raw > threshold).astype(int)

        elif spec_config.specialist_type in (SpecialistType.HUBER, SpecialistType.ASYMMETRIC_HUBER):
            # Target: Binary (>0)
            y_bag_target = (y_bag_raw > 0).astype(int)
            # Objective: Custom Focal Loss

        if y_bag_target is None:
             # Fallback
             y_bag_target = (y_bag_raw > 0).astype(int)

        try:
            params = dict(base_params)
            params['learning_rate'] = lr
            params['random_state'] = 42 + i
            
            if fobj is not None:
                # Custom objective (Focal)
                model = lgb.LGBMClassifier(objective=fobj, **params)
            else:
                # Standard binary
                model = lgb.LGBMClassifier(objective='binary', **params)
            
            if sample_weight_bag is not None:
                model.fit(X_bag, y_bag_target, sample_weight=sample_weight_bag)
            else:
                model.fit(X_bag, y_bag_target)
            
            models.append(model)
            specialist_types.append(spec_config.specialist_type)
            
        except Exception as e:
            logger.warning(f"Failed to train specialist {i+1}: {e}")
            continue
    
    return {
        'models': models,
        'specialist_types': specialist_types,
        'config': config,
        'n_features': n_features,
        'meta_feature_columns': meta_feature_columns,
        'base_feature_columns': base_feature_columns,
    }


# =============================================================================
# Label Diagnostic Dashboard
# =============================================================================

@dataclass
class LabelDiagnosticResult:
    """Results from the Label Diagnostic Dashboard."""
    
    # Learnability (Overfitting Check)
    train_error: float
    val_error: float
    learnability_gap: float
    learnability_status: str  # "good", "warning", "poor"
    
    # Signal-to-Noise Ratio (Consensus Check)
    avg_mad: float
    snr_status: str  # "high_snr", "medium_snr", "low_snr"
    
    # Directional Causality (Alpha Check)
    tanh_correlation: float
    alpha_status: str  # "strong", "weak", "none"
    
    # Stationarity (Regime Check)
    rolling_sharpe_std: float
    stationarity_status: str  # "stable", "moderate", "unstable"
    
    # Overall recommendation
    label_quality_score: float  # 0-100
    recommendation: str


class LabelDiagnosticDashboard:
    """
    Diagnostic dashboard for auditing the quality of target labels.
    
    Uses ensemble internal metrics to measure:
    1. Learnability: Training vs Validation error gap
    2. Signal-to-Noise Ratio: Ensemble MAD (model agreement)
    3. Directional Causality: Tanh model correlation with target
    4. Stationarity: Rolling Sharpe stability
    
    Recommended baseline label: Volatility-Normalized Log-Returns
    Formula: log(Future_Price / Current_Price) / Rolling_Volatility
    """
    
    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        config: Optional[DiversityDefenseConfig] = None,
        n_splits: int = 3,
        rolling_sharpe_window: int = 100,
    ):
        """
        Initialize the Label Diagnostic Dashboard.
        
        Args:
            X: Feature matrix
            y: Target variable to diagnose
            config: Diversity Defense configuration
            n_splits: Number of CV splits for learnability check
            rolling_sharpe_window: Window for rolling Sharpe stability check
        """
        self.X = X
        self.y = np.asarray(y)
        self.config = config or DiversityDefenseConfig()
        self.n_splits = n_splits
        self.rolling_sharpe_window = rolling_sharpe_window
        
        self.objectives = DiversityDefenseObjectives(self.config)
        self.aggregator = DiversityDefenseAggregator(self.config)
    
    def _compute_learnability(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> Tuple[float, float]:
        """Train Huber (Focal) models and compute train/val error (Accuracy/LogLoss)."""
        try:
            import lightgbm as lgb
            from sklearn.metrics import log_loss
        except ImportError:
            return 0.0, 0.0
        
        # Use Focal Loss (Reality Check layer)
        fobj = self.objectives.focal_loss_factory(alpha=0.25, gamma=2.0)
        
        params = {
            'n_estimators': 200,
            'num_leaves': 40,
            'max_depth': 6,
            'learning_rate': 0.03,
            'colsample_bytree': 0.5,
            'verbosity': -1,
        }
        
        y_train_bin = (y_train > 0).astype(int)
        y_val_bin = (y_val > 0).astype(int)

        model = lgb.LGBMClassifier(objective=fobj, **params)
        model.fit(X_train, y_train_bin)
        
        train_probs = expit(model.predict(X_train, raw_score=True))
        val_probs = expit(model.predict(X_val, raw_score=True))
        
        train_error = log_loss(y_train_bin, train_probs)
        val_error = log_loss(y_val_bin, val_probs)
        
        return train_error, val_error
    
    def _compute_snr(self, preds_matrix: np.ndarray) -> float:
        """Compute average MAD across predictions (lower = higher SNR)."""
        z_matrix = self.aggregator.compute_z_matrix(preds_matrix)
        _, mad_z, _ = self.aggregator.robust_aggregate(z_matrix)
        return float(np.mean(mad_z))
    
    def _compute_alpha(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> float:
        """Compute correlation between Tanh model predictions and target."""
        try:
            import lightgbm as lgb
        except ImportError:
            return 0.0
        
        # Train Tanh models (directional accuracy)
        # Using binary objective
        
        params = {
            'n_estimators': 200,
            'num_leaves': 40,
            'max_depth': 6,
            'learning_rate': 0.03,
            'colsample_bytree': 0.5,
            'verbosity': -1,
        }
        
        y_train_bin = (y_train > 0.003).astype(int) # Using first Tanh threshold

        model = lgb.LGBMClassifier(objective='binary', **params)
        model.fit(X_train, y_train_bin)
        
        val_probs = model.predict_proba(X_val)[:, 1]
        
        # Compute Spearman correlation
        if spearmanr is not None:
            corr, _ = spearmanr(val_probs, y_val) # Correlate probs with raw returns
        else:
            if len(val_probs) < 2:
                return 0.0
            corr_matrix = np.corrcoef(val_probs, y_val)
            if corr_matrix.shape == (2, 2):
                corr = corr_matrix[0, 1]
            else:
                corr = 0.0
        
        return float(corr) if not np.isnan(corr) else 0.0
    
    def _compute_stationarity(self, returns: np.ndarray) -> float:
        """Compute rolling Sharpe standard deviation (stability metric)."""
        if len(returns) < self.rolling_sharpe_window * 2:
            return 0.0
        
        # Compute rolling Sharpe
        rolling_sharpes = []
        for i in range(self.rolling_sharpe_window, len(returns)):
            window_returns = returns[i - self.rolling_sharpe_window:i]
            sharpe = calculate_simple_sharpe(window_returns, annualization_factor=1.0)
            rolling_sharpes.append(sharpe)
        
        # Standard deviation of rolling Sharpe (lower = more stable)
        return float(np.std(rolling_sharpes)) if rolling_sharpes else 0.0
    
    def run(self, verbose: bool = True) -> LabelDiagnosticResult:
        """
        Run the full label diagnostic analysis.
        
        Args:
            verbose: Print progress and results
            
        Returns:
            LabelDiagnosticResult with all metrics
        """
        try:
            import lightgbm as lgb
            from sklearn.model_selection import TimeSeriesSplit
        except ImportError:
            raise ImportError("LightGBM and scikit-learn are required")
        
        if verbose:
            print("=" * 60)
            print("LABEL DIAGNOSTIC DASHBOARD")
            print("=" * 60)
        
        # 1. LEARNABILITY CHECK
        if verbose:
            print("\n[1] Learnability (Overfitting Check)...")
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        train_errors = []
        val_errors = []
        
        for train_idx, val_idx in tscv.split(self.X):
            X_train = self.X.iloc[train_idx]
            X_val = self.X.iloc[val_idx]
            y_train = self.y[train_idx]
            y_val = self.y[val_idx]
            
            te, ve = self._compute_learnability(X_train, y_train, X_val, y_val)
            train_errors.append(te)
            val_errors.append(ve)
        
        avg_train_error = np.mean(train_errors)
        avg_val_error = np.mean(val_errors)
        learnability_gap = avg_val_error - avg_train_error
        
        if learnability_gap < 0.1:
            learnability_status = "good"
        elif learnability_gap < 0.3:
            learnability_status = "warning"
        else:
            learnability_status = "poor"
        
        if verbose:
            print(f"   Train Error: {avg_train_error:.4f}")
            print(f"   Val Error: {avg_val_error:.4f}")
            print(f"   Gap: {learnability_gap:.4f} -> {learnability_status.upper()}")
        
        # 2. SIGNAL-TO-NOISE RATIO CHECK
        if verbose:
            print("\n[2] Signal-to-Noise Ratio (Consensus Check)...")
        
        # Train mini-ensemble
        sweep = DiversitySweep(self.X, self.y, n_splits=2, n_models=5)
        
        # Get predictions from a sweep run
        all_mads = []
        tscv = TimeSeriesSplit(n_splits=2)
        for train_idx, val_idx in tscv.split(self.X):
            X_train = self.X.iloc[train_idx]
            X_val = self.X.iloc[val_idx]
            y_train = self.y[train_idx]
            
            preds_matrix = sweep._train_mini_ensemble(X_train, y_train, X_val, 0.5)
            if len(preds_matrix) > 0:
                avg_mad = self._compute_snr(preds_matrix)
                all_mads.append(avg_mad)
        
        avg_mad = np.mean(all_mads) if all_mads else 0.5
        
        if avg_mad < 0.2:
            snr_status = "high_snr"
        elif avg_mad < 0.5:
            snr_status = "medium_snr"
        else:
            snr_status = "low_snr"
        
        if verbose:
            print(f"   Average MAD: {avg_mad:.4f}")
            print(f"   SNR Status: {snr_status.upper()}")
        
        # 3. ALPHA CHECK (Directional Causality)
        if verbose:
            print("\n[3] Directional Causality (Alpha Check)...")
        
        alphas = []
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        for train_idx, val_idx in tscv.split(self.X):
            X_train = self.X.iloc[train_idx]
            X_val = self.X.iloc[val_idx]
            y_train = self.y[train_idx]
            y_val = self.y[val_idx]
            
            alpha = self._compute_alpha(X_train, y_train, X_val, y_val)
            alphas.append(alpha)
        
        tanh_correlation = np.mean(alphas)
        
        if abs(tanh_correlation) > 0.3:
            alpha_status = "strong"
        elif abs(tanh_correlation) > 0.1:
            alpha_status = "weak"
        else:
            alpha_status = "none"
        
        if verbose:
            print(f"   Tanh-Target Correlation: {tanh_correlation:.4f}")
            print(f"   Alpha Status: {alpha_status.upper()}")
        
        # 4. STATIONARITY CHECK
        if verbose:
            print("\n[4] Stationarity (Regime Check)...")
        
        rolling_sharpe_std = self._compute_stationarity(self.y)
        
        if rolling_sharpe_std < 0.5:
            stationarity_status = "stable"
        elif rolling_sharpe_std < 1.0:
            stationarity_status = "moderate"
        else:
            stationarity_status = "unstable"
        
        if verbose:
            print(f"   Rolling Sharpe Std: {rolling_sharpe_std:.4f}")
            print(f"   Stationarity Status: {stationarity_status.upper()}")
        
        # 5. OVERALL SCORE & RECOMMENDATION
        # Score components (0-25 each)
        learn_score = 25 if learnability_status == "good" else (15 if learnability_status == "warning" else 5)
        snr_score = 25 if snr_status == "high_snr" else (15 if snr_status == "medium_snr" else 5)
        alpha_score = 25 if alpha_status == "strong" else (15 if alpha_status == "weak" else 5)
        stat_score = 25 if stationarity_status == "stable" else (15 if stationarity_status == "moderate" else 5)
        
        label_quality_score = learn_score + snr_score + alpha_score + stat_score
        
        # Generate recommendation
        if label_quality_score >= 80:
            recommendation = "EXCELLENT: Label is high quality. Proceed with training."
        elif label_quality_score >= 60:
            recommendation = "GOOD: Label is usable. Consider improving weakest area."
        elif label_quality_score >= 40:
            recommendation = "FAIR: Label has issues. Try volatility-normalized returns or different horizon."
        else:
            recommendation = "POOR: Label is problematic. Recommend: log(P_t+h/P_t) / rolling_vol"
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"OVERALL LABEL QUALITY SCORE: {label_quality_score}/100")
            print(f"RECOMMENDATION: {recommendation}")
            print("=" * 60)
        
        return LabelDiagnosticResult(
            train_error=avg_train_error,
            val_error=avg_val_error,
            learnability_gap=learnability_gap,
            learnability_status=learnability_status,
            avg_mad=avg_mad,
            snr_status=snr_status,
            tanh_correlation=tanh_correlation,
            alpha_status=alpha_status,
            rolling_sharpe_std=rolling_sharpe_std,
            stationarity_status=stationarity_status,
            label_quality_score=label_quality_score,
            recommendation=recommendation,
        )


def create_volatility_normalized_label(
    close_prices: pd.Series,
    horizon: int = 4,
    vol_window: int = 24,
) -> pd.Series:
    """
    Create the recommended volatility-normalized log-return label.
    
    Formula: log(Price_t+h / Price_t) / rolling_volatility
    
    This normalizes returns by volatility, making the label more stationary
    and easier to learn across different market regimes.
    
    Args:
        close_prices: Close price series
        horizon: Forward horizon in periods (e.g., 4 for 4-hour returns)
        vol_window: Rolling window for volatility calculation
        
    Returns:
        Volatility-normalized log-returns
    """
    # Log returns at horizon h
    log_returns = np.log(close_prices.shift(-horizon) / close_prices)
    
    # Rolling volatility
    rolling_vol = close_prices.pct_change().rolling(window=vol_window, min_periods=1).std()
    
    # Volatility-normalized returns
    vol_norm_returns = log_returns / (rolling_vol + EPS)
    
    # Clip extreme values
    vol_norm_returns = vol_norm_returns.clip(-5, 5)
    
    return vol_norm_returns


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core classes
    'SpecialistType',
    'SpecialistConfig',
    'DiversityDefenseConfig',
    'DiversityDefenseObjectives',
    'DiversityDefenseAggregator',
    'DiversityDefenseHPO',
    'DiversitySweep',
    'LabelDiagnosticDashboard',
    'LabelDiagnosticResult',
    
    # Functions
    'create_diversity_defense_ensemble',
    'calculate_esr_score',
    'calculate_das_score',  # Backward compatibility
    'calculate_simple_sharpe',
    'generate_regime_meta_features',
    'create_volatility_normalized_label',
    'get_focal_loss',
    'get_sharpe_weights',
    
    # Constants
    'EPS',
    'META_FEATURE_COLUMNS',
]
