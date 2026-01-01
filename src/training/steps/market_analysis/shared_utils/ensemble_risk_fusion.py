"""
Ensemble Risk Fusion Utilities

Phase 2 implementation for combining multiple risk models into optimized ensemble.
Implements adaptive weight optimization and regime-specific fusion strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
import warnings

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning


@dataclass
class EnsembleRiskConfig:
    """Configuration for ensemble risk fusion."""
    # Risk model weights (adaptive)
    risk_score_weight: float = 0.35
    path_risk_weight: float = 0.35
    market_risk_weight: float = 0.30
    
    # Adaptive optimization
    enable_adaptive_weights: bool = True
    weight_lookback_window: int = 100  # bars for weight optimization
    weight_update_frequency: int = 20   # bars between weight updates
    
    # Regime-specific weights
    enable_regime_weights: bool = True
    regime_weight_smoothing: float = 0.8  # EMA smoothing for regime weights
    
    # Fusion method
    fusion_method: str = "weighted_average"  # "weighted_average", "ridge_ensemble", "quantile_fusion"
    
    # Risk calibration
    calibrate_output: bool = True
    target_risk_level: float = 0.5  # Target mean risk level
    risk_tolerance: float = 0.1     # Acceptable deviation from target


class EnsembleRiskFusion:
    """
    Ensemble risk fusion system combining multiple risk models.
    
    Features:
    - Adaptive weight optimization based on recent performance
    - Regime-specific weight matrices
    - Multiple fusion methods (weighted average, ridge ensemble, quantile fusion)
    - Risk calibration and stabilization
    """
    
    def __init__(self, config: Optional[EnsembleRiskConfig] = None):
        self.config = config or EnsembleRiskConfig()
        self.risk_scalers = {}
        self.adaptive_weights = None
        self.regime_weights = {}
        self.weight_history = []
        self.performance_history = []
        
        tprint("✅ Initialized EnsembleRiskFusion with adaptive optimization")
    
    def fuse_risk_scores(
        self,
        risk_scores: np.ndarray,
        path_risk_scores: np.ndarray,
        market_risk_scores: Optional[np.ndarray] = None,
        regime_labels: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fuse multiple risk scores into optimized ensemble risk score.
        
        Args:
            risk_scores: Enhanced risk_score from ML risk regime step
            path_risk_scores: Directional path_risk_score from ML path regime step  
            market_risk_scores: Optional market risk scores
            regime_labels: Current regime labels for regime-specific weighting
            returns: Recent returns for performance-based weight optimization
            config: Runtime configuration overrides
            
        Returns:
            Tuple of (ensemble_risk_scores, fusion_metadata)
        """
        if config:
            # Update config with runtime overrides
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        tprint_info("🚀 Fusing risk scores with ensemble optimization...")
        
        # Prepare risk score matrix
        risk_matrix = self._prepare_risk_matrix(risk_scores, path_risk_scores, market_risk_scores)
        
        # Normalize risk scores to [0, 1]
        risk_matrix_normalized = self._normalize_risk_scores(risk_matrix)
        
        # Calculate adaptive weights if enabled
        if self.config.enable_adaptive_weights and returns is not None:
            adaptive_weights = self._calculate_adaptive_weights(risk_matrix_normalized, returns)
        else:
            adaptive_weights = self._get_default_weights()
        
        # Apply regime-specific weights if enabled
        if self.config.enable_regime_weights and regime_labels is not None:
            regime_weights = self._calculate_regime_weights(risk_matrix_normalized, regime_labels)
            final_weights = self._combine_adaptive_and_regime_weights(adaptive_weights, regime_weights)
        else:
            final_weights = adaptive_weights
        
        # Fuse risk scores using selected method
        if self.config.fusion_method == "weighted_average":
            ensemble_scores = self._weighted_average_fusion(risk_matrix_normalized, final_weights)
        elif self.config.fusion_method == "ridge_ensemble":
            ensemble_scores = self._ridge_ensemble_fusion(risk_matrix_normalized, returns)
        elif self.config.fusion_method == "quantile_fusion":
            ensemble_scores = self._quantile_fusion(risk_matrix_normalized, final_weights)
        else:
            raise ValueError(f"Unknown fusion method: {self.config.fusion_method}")
        
        # Calibrate output if enabled
        if self.config.calibrate_output:
            ensemble_scores = self._calibrate_risk_scores(ensemble_scores)
        
        # Apply smoothing to reduce noise
        ensemble_scores = self._apply_output_smoothing(ensemble_scores)
        
        # Prepare metadata
        metadata = {
            'fusion_method': self.config.fusion_method,
            'final_weights': final_weights.tolist(),
            'adaptive_weights': adaptive_weights.tolist(),
            'regime_weights': self.regime_weights if hasattr(self, 'regime_weights') else {},
            'risk_matrix_shape': risk_matrix_normalized.shape,
            'ensemble_mean': np.nanmean(ensemble_scores),
            'ensemble_std': np.nanstd(ensemble_scores),
            'weight_entropy': self._calculate_weight_entropy(final_weights),
        }
        
        tprint_success(f"✅ Risk fusion complete: mean={metadata['ensemble_mean']:.4f}, std={metadata['ensemble_std']:.4f}")
        
        return ensemble_scores, metadata
    
    def _prepare_risk_matrix(
        self, 
        risk_scores: np.ndarray, 
        path_risk_scores: np.ndarray, 
        market_risk_scores: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Prepare risk score matrix from individual risk models."""
        n_samples = len(risk_scores)
        risk_matrix = np.column_stack([risk_scores, path_risk_scores])
        
        if market_risk_scores is not None:
            risk_matrix = np.column_stack([risk_matrix, market_risk_scores])
        
        # Handle NaN values
        risk_matrix = np.nan_to_num(risk_matrix, nan=0.5)
        
        return risk_matrix
    
    def _normalize_risk_scores(self, risk_matrix: np.ndarray) -> np.ndarray:
        """Normalize risk scores to [0, 1] range."""
        normalized_matrix = np.zeros_like(risk_matrix)
        
        for i in range(risk_matrix.shape[1]):
            scores = risk_matrix[:, i]
            
            # Remove outliers for robust normalization
            q1, q99 = np.percentile(scores, [1, 99])
            scores_clipped = np.clip(scores, q1, q99)
            
            # Min-max normalization
            min_val, max_val = np.min(scores_clipped), np.max(scores_clipped)
            if max_val > min_val:
                normalized_matrix[:, i] = (scores_clipped - min_val) / (max_val - min_val)
            else:
                normalized_matrix[:, i] = 0.5  # Default if constant
        
        return np.clip(normalized_matrix, 0, 1)
    
    def _get_default_weights(self, n_models: int = 2) -> np.ndarray:
        """Get default risk model weights."""
        if n_models == 3:
            # All three models available
            total = self.config.risk_score_weight + self.config.path_risk_weight + self.config.market_risk_weight
            return np.array([
                self.config.risk_score_weight / total,
                self.config.path_risk_weight / total,
                self.config.market_risk_weight / total
            ])
        else:
            # Only risk_score and path_risk_score available
            total = self.config.risk_score_weight + self.config.path_risk_weight
            return np.array([
                self.config.risk_score_weight / total,
                self.config.path_risk_weight / total
            ])
    
    def _calculate_adaptive_weights(
        self, 
        risk_matrix: np.ndarray, 
        returns: np.ndarray
    ) -> np.ndarray:
        """Calculate adaptive weights based on recent performance."""
        lookback = min(self.config.weight_lookback_window, len(returns))
        
        if lookback < 20:  # Need sufficient history
            return self._get_default_weights(n_models=risk_matrix.shape[1])
        
        # Use recent returns for performance evaluation
        recent_returns = returns[-lookback:]
        recent_risk_matrix = risk_matrix[-lookback:]
        
        # Calculate performance metrics for each risk model
        n_models = risk_matrix.shape[1]
        performance_scores = np.zeros(n_models)
        
        for i in range(n_models):
            risk_scores = recent_risk_matrix[:, i]
            
            # Calculate risk-adjusted performance (lower risk should correlate with better returns)
            risk_return_correlation = np.corrcoef(risk_scores, recent_returns)[0, 1]
            
            # Inverse correlation: lower risk should predict better returns
            performance_scores[i] = -risk_return_correlation if not np.isnan(risk_return_correlation) else 0
        
        # Convert performance scores to weights
        if np.sum(np.abs(performance_scores)) > 0:
            # Use softmax for weight normalization
            exp_scores = np.exp(performance_scores * 2)  # Temperature scaling
            weights = exp_scores / np.sum(exp_scores)
        else:
            weights = self._get_default_weights(n_models=risk_matrix.shape[1])
        
        # Store weight history
        self.weight_history.append(weights.tolist())
        if len(self.weight_history) > 100:  # Keep recent history
            self.weight_history.pop(0)
        
        return weights
    
    def _calculate_regime_weights(
        self, 
        risk_matrix: np.ndarray, 
        regime_labels: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """Calculate regime-specific weights."""
        regime_weights = {}
        n_models = risk_matrix.shape[1]
        
        for regime_id in np.unique(regime_labels):
            if regime_id < 0:  # Skip invalid labels
                continue
            
            regime_mask = regime_labels == regime_id
            if np.sum(regime_mask) < 10:  # Need sufficient samples
                continue
            
            regime_risk_matrix = risk_matrix[regime_mask]
            
            # Calculate regime-specific performance based on risk variance
            # Lower variance risk scores are more stable => higher weight
            regime_variances = np.var(regime_risk_matrix, axis=0)
            stability_scores = 1.0 / (1.0 + regime_variances)  # Inverse variance
            
            # Normalize to weights
            if np.sum(stability_scores) > 0:
                weights = stability_scores / np.sum(stability_scores)
            else:
                weights = np.ones(n_models) / n_models
            
            regime_weights[int(regime_id)] = weights
        
        return regime_weights
    
    def _combine_adaptive_and_regime_weights(
        self, 
        adaptive_weights: np.ndarray, 
        regime_weights: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """Combine adaptive and regime-specific weights."""
        # For now, use adaptive weights as primary
        # In future, could implement more sophisticated combination
        return adaptive_weights
    
    def _weighted_average_fusion(
        self, 
        risk_matrix: np.ndarray, 
        weights: np.ndarray
    ) -> np.ndarray:
        """Weighted average fusion of risk scores."""
        return np.dot(risk_matrix, weights)
    
    def _ridge_ensemble_fusion(
        self, 
        risk_matrix: np.ndarray, 
        returns: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Ridge regression ensemble fusion."""
        if returns is None or len(returns) < 50:
            # Fallback to weighted average
            default_weights = self._get_default_weights()
            return self._weighted_average_fusion(risk_matrix, default_weights)
        
        try:
            # Use Ridge regression to learn optimal weights
            ridge = Ridge(alpha=0.1, fit_intercept=True)
            
            # Prepare training data
            X = risk_matrix[:-1]  # Risk scores as features
            y = returns[1:]       # Next period returns as target
            
            # Remove NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            if np.sum(valid_mask) < 20:
                default_weights = self._get_default_weights()
                return self._weighted_average_fusion(risk_matrix, default_weights)
            
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            # Fit Ridge regression
            ridge.fit(X_valid, y_valid)
            
            # Use learned coefficients as weights (normalize to positive)
            raw_weights = ridge.coef_
            weights = np.abs(raw_weights)  # Use absolute values
            weights = weights / np.sum(weights)  # Normalize
            
            return self._weighted_average_fusion(risk_matrix, weights)
            
        except Exception as e:
            tprint_warning(f"Ridge ensemble failed, using weighted average: {e}")
            default_weights = self._get_default_weights()
            return self._weighted_average_fusion(risk_matrix, default_weights)
    
    def _quantile_fusion(
        self, 
        risk_matrix: np.ndarray, 
        weights: np.ndarray
    ) -> np.ndarray:
        """Quantile-based fusion (robust to outliers)."""
        # Calculate weighted quantiles
        n_samples = risk_matrix.shape[0]
        fused_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            sample_risks = risk_matrix[i]
            
            # Weighted median as robust fusion
            sorted_indices = np.argsort(sample_risks)
            sorted_risks = sample_risks[sorted_indices]
            sorted_weights = weights[sorted_indices]
            
            # Calculate weighted median
            cumsum_weights = np.cumsum(sorted_weights)
            median_idx = np.where(cumsum_weights >= 0.5)[0]
            
            if len(median_idx) > 0:
                fused_scores[i] = sorted_risks[median_idx[0]]
            else:
                fused_scores[i] = np.average(sample_risks, weights=weights)
        
        return fused_scores
    
    def _calibrate_risk_scores(self, scores: np.ndarray) -> np.ndarray:
        """Calibrate risk scores to target level."""
        current_mean = np.nanmean(scores)
        
        if np.isnan(current_mean):
            return scores
        
        # Scale to target risk level
        if current_mean > 0:
            scale_factor = self.config.target_risk_level / current_mean
            calibrated_scores = scores * scale_factor
        else:
            calibrated_scores = scores
        
        # Clip to [0, 1]
        return np.clip(calibrated_scores, 0, 1)
    
    def _apply_output_smoothing(self, scores: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing to reduce noise."""
        if len(scores) < 10:
            return scores
        
        smoothed_scores = pd.Series(scores).ewm(span=5, adjust=False).mean().values
        return np.clip(smoothed_scores, 0, 1)
    
    def _calculate_weight_entropy(self, weights: np.ndarray) -> float:
        """Calculate entropy of weights (diversification measure)."""
        # Avoid log(0)
        weights_safe = np.clip(weights, 1e-10, 1.0)
        entropy = -np.sum(weights_safe * np.log(weights_safe))
        return entropy
    
    def get_weight_stability(self) -> Dict[str, float]:
        """Get weight stability metrics."""
        if len(self.weight_history) < 10:
            return {'stability': 0.0, 'trend_strength': 0.0}
        
        recent_weights = np.array(self.weight_history[-10:])
        
        # Calculate weight volatility (inverse of stability)
        weight_volatility = np.std(recent_weights, axis=0)
        stability = 1.0 - np.mean(weight_volatility)
        
        # Calculate trend strength
        if len(recent_weights) >= 5:
            weight_changes = np.diff(recent_weights, axis=0)
            trend_strength = np.mean(np.abs(weight_changes))
        else:
            trend_strength = 0.0
        
        return {
            'stability': max(0.0, stability),
            'trend_strength': trend_strength,
            'recent_weights': recent_weights[-1].tolist()
        }
