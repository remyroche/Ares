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

logger = logging.getLogger(__name__)

# Epsilon for numerical stability
EPS = 1e-8


# =============================================================================
# Diversity-Adjusted Sharpe (DAS) Scoring
# =============================================================================

def calculate_das_score(
    sharpe: float, 
    avg_corr: float, 
    target_floor: float = 0.5, 
    penalty_weight: float = 4.0
) -> float:
    """
    Calculate the Diversity-Adjusted Sharpe (DAS) score.
    
    Rewards raw performance (Sharpe) but applies a progressively heavier 
    tax as correlation creeps above the target floor.
    
    Formula: Score = Sharpe - λ * max(0, Corr - 0.5)²
    
    Args:
        sharpe: The annualized Sharpe ratio of the ensemble
        avg_corr: The average pairwise correlation of the Z-scores
        target_floor: Safe floor - correlations below this are free (default 0.5)
        penalty_weight: Controls how hard to punish "herding" (default 4.0)
        
    Returns:
        Diversity-Adjusted Sharpe score
        
    Example:
        >>> calculate_das_score(sharpe=1.5, avg_corr=0.4)  # Below floor
        1.5  # No penalty
        >>> calculate_das_score(sharpe=1.5, avg_corr=0.7)  # Above floor
        1.34  # Penalized: 1.5 - 4.0 * (0.2)² = 1.5 - 0.16
    """
    # Calculate how much we are 'over' the limit
    excess_corr = max(0.0, avg_corr - target_floor)
    
    # Apply Quadratic Penalty (punishes extreme excesses harder)
    penalty = penalty_weight * (excess_corr ** 2)
    
    # Final Score
    return sharpe - penalty


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
    if len(returns) == 0:
        return 0.0
    std = np.std(returns)
    if std < EPS:
        return 0.0
    return (np.mean(returns) / std) * annualization_factor


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


@dataclass 
class DiversityDefenseConfig:
    """Full configuration for the Diversity Defense ensemble."""
    
    # Specialist layer configuration (10 models total)
    sharpe_count: int = 3          # "Smart Money" layer
    tanh_count: int = 3            # "Trend Follower" layer
    huber_standard_count: int = 2  # "Reality Check" - Standard
    huber_asymmetric_count: int = 2  # "Reality Check" - Asymmetric
    
    # Sharpe lambdas (risk aversion) - increasing values
    sharpe_lambdas: List[float] = field(default_factory=lambda: [0.5, 2.0, 8.0])
    
    # Huber parameters
    huber_delta: float = 0.01        # Outlier threshold for Huber
    asymmetric_penalty: float = 3.0   # Penalty factor for wrong-direction (2-5x typical)
    
    # Learning rates per layer
    lr_sharpe: float = 0.03          # Conservative default
    lr_tanh: float = 0.03
    lr_huber: float = 0.03
    
    # Aggregation parameters
    z_score_window: int = 1000       # Rolling window for Z-score normalization
    mad_floor: float = 0.25          # Soft floor for MAD (prevents div by zero)
    noise_threshold: float = 0.3     # Below this, signal = 0
    cap_threshold: float = 0.7       # Above this, signal capped
    
    # Orthogonality penalty parameters (for DAS scoring)
    correlation_threshold: float = 0.5   # Safe floor - correlations below this are free
    das_penalty_weight: float = 4.0      # Quadratic penalty weight for DAS formula
    
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
        for i, lmbda in enumerate(self.sharpe_lambdas[:self.sharpe_count]):
            configs.append(SpecialistConfig(
                specialist_type=SpecialistType.SHARPE,
                count=1,
                learning_rate_key='lr_sharpe',
                role_description=f"Smart Money #{i+1} (λ={lmbda})",
                lmbda=lmbda
            ))
        
        # Tanh models (3x)
        for i in range(self.tanh_count):
            configs.append(SpecialistConfig(
                specialist_type=SpecialistType.TANH,
                count=1,
                learning_rate_key='lr_tanh',
                role_description=f"Trend Follower #{i+1}"
            ))
        
        # Huber Standard models (2x)
        for i in range(self.huber_standard_count):
            configs.append(SpecialistConfig(
                specialist_type=SpecialistType.HUBER,
                count=1,
                learning_rate_key='lr_huber',
                role_description=f"Reality Check (Huber) #{i+1}",
                delta=self.huber_delta
            ))
        
        # Huber Asymmetric models (2x)
        for i in range(self.huber_asymmetric_count):
            configs.append(SpecialistConfig(
                specialist_type=SpecialistType.ASYMMETRIC_HUBER,
                count=1,
                learning_rate_key='lr_huber',
                role_description=f"Reality Check (Asymmetric) #{i+1}",
                delta=self.huber_delta,
                penalty_factor=self.asymmetric_penalty
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
    
    @staticmethod
    def vol_normalized_sharpe_factory(
        lmbda: float, 
        volatility: np.ndarray
    ) -> Callable:
        """
        Create a Vol-Normalized Sharpe objective function.
        
        Maximizes: E[p * r_norm] - lambda * E[(p * r_norm)^2]
        Where r_norm = r / vol.
        
        This optimizes Sharpe on Strategy PnL, independent of market regimes.
        
        Args:
            lmbda: Risk aversion parameter (higher = more conservative)
            volatility: Pre-calculated volatility vector matching the dataset
            
        Returns:
            Objective function (grad, hess) callable
        """
        # Convert volatility to numpy array and ensure it's float
        vol = np.asarray(volatility, dtype=np.float64) + EPS
        
        def fobj(preds: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
            """LightGBM objective function."""
            r = DiversityDefenseObjectives._get_labels_from_dataset(dataset)
            p = preds.astype(np.float64)
            
            # Handle length mismatches (volatility may be from different fold)
            n = len(r)
            if len(vol) != n:
                # Use mean volatility or slice to match
                if len(vol) > n:
                    v = vol[:n]
                else:
                    v = np.full(n, np.mean(vol), dtype=np.float64) + EPS
            else:
                v = vol
            
            # Normalize returns by volatility
            r_norm = r / v
            
            # Gradient: -r_norm + 2 * lambda * p * (r_norm^2)
            # This penalizes variance of the PnL (p*r), not just variance of r
            grad = -r_norm + 2.0 * lmbda * p * (r_norm ** 2)
            
            # Hessian: 2 * lambda * (r_norm^2)
            hess = 2.0 * lmbda * (r_norm ** 2)
            hess = np.maximum(hess, EPS)  # Ensure positive curvature
            
            return grad, hess
        
        return fobj
    
    @staticmethod
    def robust_tanh_factory() -> Callable:
        """
        Create a Robust Tanh objective function.
        
        Optimizes directional accuracy (hit rate) while being robust to magnitude.
        Uses absolute value of hessian to ensure positive curvature.
        
        Returns:
            Objective function (grad, hess) callable
        """
        def fobj(preds: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
            """LightGBM objective function."""
            r = DiversityDefenseObjectives._get_labels_from_dataset(dataset)
            p = preds.astype(np.float64)
            
            # Tanh transformation for bounded output
            t = np.tanh(p)
            
            # Gradient: -r * (1 - tanh(p)^2)
            grad = -r * (1.0 - t * t)
            
            # Hessian: abs(2 * r * tanh(p) * (1 - tanh(p)^2)) + eps
            # Absolute value ensures positive curvature for Newton step stability
            hess = np.abs(2.0 * r * t * (1.0 - t * t)) + EPS
            
            return grad, hess
        
        return fobj
    
    @staticmethod
    def huber_factory(delta: float) -> Callable:
        """
        Create a Standard Huber objective function.
        
        Combines MSE for small errors and MAE for large errors.
        Robust to outliers beyond the delta threshold.
        
        Args:
            delta: Threshold for switching between MSE and MAE
            
        Returns:
            Objective function (grad, hess) callable
        """
        return DiversityDefenseObjectives.asymmetric_huber_factory(delta, penalty_factor=1.0)
    
    @staticmethod
    def asymmetric_huber_factory(
        delta: float, 
        penalty_factor: float = 3.0
    ) -> Callable:
        """
        Create an Asymmetric Huber objective function.
        
        Scales GRADIENT ONLY for wrong-direction errors.
        Maintains curvature stability (Hessian=1) but takes larger steps
        for wrong-direction predictions.
        
        Args:
            delta: Threshold for Huber transition
            penalty_factor: Multiplier for gradient when direction is wrong (1.0-5.0)
            
        Returns:
            Objective function (grad, hess) callable
        """
        def fobj(preds: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
            """LightGBM objective function."""
            r = DiversityDefenseObjectives._get_labels_from_dataset(dataset)
            p = preds.astype(np.float64)
            z = p - r  # Residual
            
            # Direction Mask: True where direction is WRONG (signs differ)
            # When p and r have different signs, the product is negative
            wrong_dir_mask = (p * r) < 0
            
            # Base Huber loss gradient and hessian
            absz = np.abs(z)
            
            # Gradient: z for |z| <= delta, else delta * sign(z)
            base_grad = np.where(absz <= delta, z, delta * np.sign(z))
            
            # Hessian: 1 for |z| <= delta, else eps (flat region)
            base_hess = np.where(absz <= delta, 1.0, EPS)
            
            # Apply asymmetric scaling to gradient only
            scale = np.where(wrong_dir_mask, penalty_factor, 1.0)
            grad = base_grad * scale
            
            # Return base_hess to keep Newton step stable
            return grad, base_hess
        
        return fobj
    
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
        if specialist_config.specialist_type == SpecialistType.SHARPE:
            if volatility is None:
                logger.warning("Sharpe objective requires volatility; using default")
                volatility = np.ones(1)  # Will be handled in factory
            return self.vol_normalized_sharpe_factory(
                lmbda=specialist_config.lmbda or 1.0,
                volatility=volatility
            )
        
        elif specialist_config.specialist_type == SpecialistType.TANH:
            return self.robust_tanh_factory()
        
        elif specialist_config.specialist_type == SpecialistType.HUBER:
            return self.huber_factory(
                delta=specialist_config.delta or self.config.huber_delta
            )
        
        elif specialist_config.specialist_type == SpecialistType.ASYMMETRIC_HUBER:
            return self.asymmetric_huber_factory(
                delta=specialist_config.delta or self.config.huber_delta,
                penalty_factor=specialist_config.penalty_factor or self.config.asymmetric_penalty
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
    3. MAD-based confidence weighting (veto mechanism)
    4. Bucketed execution sizing
    """
    
    def __init__(self, config: Optional[DiversityDefenseConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or DiversityDefenseConfig()
    
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
        window = window or self.config.z_score_window
        z_out = []
        
        for row in preds_matrix:
            s = pd.Series(row)
            r = s.rolling(window=window, min_periods=100)
            mu = r.mean()
            sigma = r.std() + EPS
            z = (s - mu) / sigma
            z_out.append(z.fillna(0).values)
        
        return np.array(z_out)
    
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
    
    def robust_aggregate(
        self, 
        z_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform robust aggregation using median and MAD.
        
        Args:
            z_matrix: Z-score normalized predictions (n_models, n_samples)
            
        Returns:
            Tuple of (median_z, mad_z, raw_signal)
        """
        # Median Z-score (ignores outliers)
        med_z = np.median(z_matrix, axis=0)
        
        # MAD (Median Absolute Deviation) - measures disagreement
        mad_z = np.median(np.abs(z_matrix - med_z), axis=0)
        
        # Soft floor on MAD to prevent division issues
        mad_eff = np.maximum(mad_z, self.config.mad_floor)
        
        # Raw signal: Median / (1 + MAD)
        # Effect: High disagreement (large MAD) shrinks signal toward zero
        raw_signal = med_z / (1.0 + mad_eff)
        
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
        self.y_vol = pd.Series(y).rolling(window=1000).std().bfill().values
        
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
                    fobj = self.objectives.vol_normalized_sharpe_factory(lam, vol_train)
                    model = lgb.LGBMRegressor(
                        learning_rate=lr_sharpe, 
                        objective=fobj, 
                        **param_grid
                    )
                    try:
                        model.fit(X_train, y_train)
                        preds_store.append(model.predict(X_val))
                    except Exception:
                        preds_store.append(np.zeros(len(X_val)))
                
                # 2. Tanh Models - 3 models
                fobj_tanh = self.objectives.robust_tanh_factory()
                for i in range(3):
                    model = lgb.LGBMRegressor(
                        learning_rate=lr_tanh, 
                        objective=fobj_tanh,
                        random_state=42 + i,
                        **param_grid
                    )
                    try:
                        model.fit(X_train, y_train)
                        preds_store.append(model.predict(X_val))
                    except Exception:
                        preds_store.append(np.zeros(len(X_val)))
                
                # 3. Huber Standard Models - 2 models
                fobj_huber = self.objectives.huber_factory(delta)
                for i in range(2):
                    model = lgb.LGBMRegressor(
                        learning_rate=lr_huber, 
                        objective=fobj_huber,
                        random_state=100 + i,
                        **param_grid
                    )
                    try:
                        model.fit(X_train, y_train)
                        preds_store.append(model.predict(X_val))
                    except Exception:
                        preds_store.append(np.zeros(len(X_val)))
                
                # 4. Asymmetric Huber Models - 2 models
                fobj_asym = self.objectives.asymmetric_huber_factory(delta, asym_penalty)
                for i in range(2):
                    model = lgb.LGBMRegressor(
                        learning_rate=lr_huber, 
                        objective=fobj_asym,
                        random_state=200 + i,
                        **param_grid
                    )
                    try:
                        model.fit(X_train, y_train)
                        preds_store.append(model.predict(X_val))
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
                
                agg_result = aggregator.aggregate(preds_matrix)
                ortho_penalty = agg_result['orthogonality_penalty']
                pos_size = agg_result['final_signal']
                
                # F. Scoring: Sharpe after fees
                strategy_ret = pos_size * y_val
                turnover = np.abs(np.diff(pos_size, prepend=0))
                net_ret = strategy_ret - (turnover * self.fee_rate)
                
                if np.std(net_ret) < EPS:
                    sharpe = 0.0
                else:
                    sharpe = (np.mean(net_ret) / np.std(net_ret)) * self.annualization_factor
                
                # Optimize for Sharpe minus Correlation Penalty
                cv_scores.append(sharpe - ortho_penalty)
            
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
        das_floor: float = 0.5,
        das_penalty: float = 4.0,
    ):
        """
        Initialize the diversity sweep.
        
        Args:
            X: Feature matrix
            y: Target returns
            colsample_settings: List of colsample_bytree values to test
            n_splits: Number of time series CV splits
            n_models: Number of models in mini-ensemble (5 recommended for speed)
            das_floor: Target floor for DAS scoring
            das_penalty: Penalty weight for DAS scoring
        """
        self.X = X
        self.y = np.asarray(y)
        self.colsample_settings = colsample_settings or [0.8, 0.6, 0.5, 0.4, 0.3]
        self.n_splits = n_splits
        self.n_models = n_models
        self.das_floor = das_floor
        self.das_penalty = das_penalty
        
        self.objectives = DiversityDefenseObjectives()
    
    @staticmethod
    def _sharpe_proxy(preds: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Simplified Sharpe proxy objective with fixed lambda=1.0."""
        r = DiversityDefenseObjectives._get_labels_from_dataset(dataset)
        p = preds.astype(np.float64)
        # Simple fixed lambda=1.0
        grad = -r + 2.0 * 1.0 * (r ** 2) * p
        hess = np.maximum(2.0 * 1.0 * (r ** 2), EPS)
        return grad, hess
    
    @staticmethod
    def _tanh_proxy(preds: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Simplified Tanh proxy objective."""
        r = DiversityDefenseObjectives._get_labels_from_dataset(dataset)
        p = preds.astype(np.float64)
        t = np.tanh(p)
        grad = -r * (1.0 - t * t)
        hess = np.maximum(np.abs(2.0 * r * t * (1.0 - t * t)), EPS)
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
        
        # Distribution: 2 Sharpe, 1 Tanh, 2 Huber (MSE) for 5 models
        # Or scale proportionally for different n_models
        n_sharpe = max(1, self.n_models * 2 // 5)
        n_tanh = max(1, self.n_models * 1 // 5)
        n_huber = self.n_models - n_sharpe - n_tanh
        
        # 1. Sharpe Models
        for i in range(n_sharpe):
            try:
                model = lgb.LGBMRegressor(
                    objective=self._sharpe_proxy,
                    random_state=i,
                    **run_params
                )
                model.fit(X_train, y_train)
                preds_store.append(model.predict(X_val))
            except Exception:
                preds_store.append(np.zeros(len(X_val)))
        
        # 2. Tanh Models
        for i in range(n_tanh):
            try:
                model = lgb.LGBMRegressor(
                    objective=self._tanh_proxy,
                    random_state=42 + i,
                    **run_params
                )
                model.fit(X_train, y_train)
                preds_store.append(model.predict(X_val))
            except Exception:
                preds_store.append(np.zeros(len(X_val)))
        
        # 3. Huber/MSE Models (standard regression)
        for i in range(n_huber):
            try:
                model = lgb.LGBMRegressor(
                    objective='regression',
                    random_state=100 + i,
                    **run_params
                )
                model.fit(X_train, y_train)
                preds_store.append(model.predict(X_val))
            except Exception:
                preds_store.append(np.zeros(len(X_val)))
        
        return np.array(preds_store)
    
    def run(self, verbose: bool = True) -> pd.DataFrame:
        """
        Run the diversity sweep.
        
        Args:
            verbose: Print progress
            
        Returns:
            DataFrame with columns: Fraction, Sharpe, Avg_Corr, DAS_Score
        """
        try:
            from sklearn.model_selection import TimeSeriesSplit
        except ImportError:
            raise ImportError("scikit-learn is required")
        
        results = []
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        if verbose:
            print(f"{'Fraction':<10} | {'Sharpe':<10} | {'Avg Corr':<10} | {'DAS Score':<12}")
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
                z_matrix = []
                for row in preds_matrix:
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
                final_sig = med_z / (1.0 + np.maximum(mad_z, 0.25))
                
                ret = final_sig * y_val
                sharpe = calculate_simple_sharpe(ret)
                fold_sharpes.append(sharpe)
            
            # Aggregate
            avg_sharpe = np.mean(fold_sharpes) if fold_sharpes else 0.0
            avg_corr = np.mean(fold_corrs) if fold_corrs else 0.0
            das_score = calculate_das_score(
                avg_sharpe, avg_corr, 
                target_floor=self.das_floor,
                penalty_weight=self.das_penalty
            )
            
            results.append({
                'Fraction': col_fraction,
                'Sharpe': avg_sharpe,
                'Avg_Corr': avg_corr,
                'DAS_Score': das_score
            })
            
            if verbose:
                print(f"{col_fraction:<10.2f} | {avg_sharpe:<10.4f} | {avg_corr:<10.4f} | {das_score:<12.4f}")
        
        df_results = pd.DataFrame(results)
        
        # Find optimal
        if len(df_results) > 0:
            best_idx = df_results['DAS_Score'].idxmax()
            best_fraction = df_results.loc[best_idx, 'Fraction']
            if verbose:
                print("-" * 55)
                print(f"✅ Optimal colsample_bytree: {best_fraction:.2f} (DAS Score: {df_results.loc[best_idx, 'DAS_Score']:.4f})")
        
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
        best_idx = df['DAS_Score'].idxmax()
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
    
    # Pre-calculate volatility for Sharpe objectives
    y_vol = pd.Series(y_train).rolling(window=config.z_score_window).std().bfill().values
    
    n_samples, n_features = X_train.shape
    rng = np.random.RandomState(42)
    
    models = []
    specialist_types = []
    
    specialist_configs = config.get_specialist_configs()
    
    # Base params - Feature diversity via colsample_bytree (NOT external loop)
    base_params = {
        'n_estimators': config.n_estimators,
        'num_leaves': config.num_leaves,
        'max_depth': config.max_depth,
        'subsample': config.subsample,
        'colsample_bytree': config.colsample_bytree,  # CRITICAL: Controls feature diversity
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
        fobj = objectives.get_objective_for_specialist(spec_config, y_vol)
        
        # Row sampling (external bootstrap for sample diversity)
        n_rows_sub = max(10, int(round(config.sample_fraction * n_samples)))
        row_idx = np.sort(rng.choice(n_samples, size=n_rows_sub, replace=False))
        
        # Use all features - colsample_bytree handles feature diversity internally
        if isinstance(X_train, pd.DataFrame):
            X_bag = X_train.iloc[row_idx]
        else:
            X_bag = X_train[row_idx]
        y_bag = y_train[row_idx]
        vol_bag = y_vol[row_idx]
        
        if sample_weights is not None:
            sw_bag = sample_weights[row_idx]
        else:
            sw_bag = None
        
        try:
            params = dict(base_params)
            params['learning_rate'] = lr
            params['random_state'] = 42 + i
            
            # Update objective with current volatility for Sharpe models
            if fobj is not None and spec_config.specialist_type == SpecialistType.SHARPE:
                fobj = objectives.vol_normalized_sharpe_factory(
                    spec_config.lmbda or 1.0, vol_bag
                )
            
            if fobj is not None:
                model = lgb.LGBMRegressor(objective=fobj, **params)
            else:
                model = lgb.LGBMRegressor(**params)
            
            if sw_bag is not None:
                model.fit(X_bag, y_bag, sample_weight=sw_bag)
            else:
                model.fit(X_bag, y_bag)
            
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
    }


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
    
    # Functions
    'create_diversity_defense_ensemble',
    'calculate_das_score',
    'calculate_simple_sharpe',
    
    # Constants
    'EPS',
]
