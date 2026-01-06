"""
Regime-Specific Weight Matrices

Phase 2 implementation for adaptive feature weighting based on regime characteristics.
Implements dynamic weight optimization per regime with smooth transitions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import warnings

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning


@dataclass
class RegimeWeightConfig:
    """Configuration for regime-specific weight matrices."""
    # Feature categories and base weights
    feature_categories = {
        'volatility_features': 0.25,
        'momentum_features': 0.20,
        'trend_features': 0.20,
        'microstructure_features': 0.15,
        'correlation_features': 0.10,
        'stability_features': 0.10,
    }
    
    # Adaptive optimization
    enable_adaptive_optimization: bool = True
    optimization_window: int = 200  # bars for weight learning
    regularization_strength: float = 0.1
    
    # Regime transition smoothing
    enable_transition_smoothing: bool = True
    transition_smoothing_factor: float = 0.8
    
    # Weight constraints
    min_weight_per_feature: float = 0.05
    max_weight_per_feature: float = 0.40
    weight_change_limit: float = 0.10  # Maximum change per update


class RegimeSpecificWeights:
    """
    Regime-specific weight matrix system for adaptive feature importance.
    
    Features:
    - Per-regime feature weight optimization
    - Smooth weight transitions between regimes
    - Performance-based weight adaptation
    - Constraint enforcement for stability
    """
    
    def __init__(self, config: Optional[RegimeWeightConfig] = None):
        self.config = config or RegimeWeightConfig()
        self.regime_weights = {}
        self.feature_names = []
        self.feature_categories = {}
        self.weight_history = {}
        self.performance_metrics = {}
        
        tprint("✅ Initialized RegimeSpecificWeights with adaptive optimization")
    
    def initialize_regime_weights(
        self,
        feature_names: List[str],
        n_regimes: int,
        initial_weights: Optional[Dict[int, np.ndarray]] = None
    ) -> None:
        """
        Initialize weight matrices for all regimes.
        
        Args:
            feature_names: List of feature names
            n_regimes: Number of regimes
            initial_weights: Optional initial weights per regime
        """
        self.feature_names = feature_names
        self.feature_categories = self._categorize_features(feature_names)
        
        for regime_id in range(n_regimes):
            if initial_weights and regime_id in initial_weights:
                self.regime_weights[regime_id] = initial_weights[regime_id].copy()
            else:
                # Initialize with category-based weights
                self.regime_weights[regime_id] = self._get_initial_weights(regime_id)
            
            # Initialize weight history
            self.weight_history[regime_id] = [self.regime_weights[regime_id].copy()]
        
        tprint_info(f"  Initialized weights for {n_regimes} regimes with {len(feature_names)} features")
    
    def get_regime_weights(
        self,
        regime_id: int,
        regime_probabilities: Optional[np.ndarray] = None,
        smooth_transition: bool = True
    ) -> np.ndarray:
        """
        Get weights for specific regime with optional smoothing.
        
        Args:
            regime_id: Target regime ID
            regime_probabilities: Current regime probabilities for smooth transitions
            smooth_transition: Whether to apply transition smoothing
            
        Returns:
            Weight vector for the regime
        """
        if regime_id not in self.regime_weights:
            # Return default weights if regime not found
            return self._get_default_weights()
        
        base_weights = self.regime_weights[regime_id].copy()
        
        if smooth_transition and regime_probabilities is not None and self.config.enable_transition_smoothing:
            # Apply smooth transition based on regime probabilities
            smoothed_weights = self._apply_transition_smoothing(
                base_weights, regime_probabilities
            )
            return smoothed_weights
        
        return base_weights
    
    def update_regime_weights(
        self,
        regime_id: int,
        features: pd.DataFrame,
        targets: np.ndarray,
        regime_mask: np.ndarray,
        performance_metric: str = 'correlation'
    ) -> Dict[str, float]:
        """
        Update weights for specific regime based on performance.
        
        Args:
            regime_id: Regime to update
            features: Feature matrix
            targets: Target values (returns, labels, etc.)
            regime_mask: Boolean mask for regime samples
            performance_metric: Metric for weight optimization
            
        Returns:
            Performance metrics for the update
        """
        if regime_id not in self.regime_weights:
            return {'error': f'Regime {regime_id} not initialized'}
        
        regime_features = features[regime_mask]
        regime_targets = targets[regime_mask]
        
        if len(regime_features) < 20:  # Need sufficient samples
            return {'error': 'Insufficient samples for weight update'}
        
        try:
            # Calculate optimal weights using Ridge regression
            optimal_weights = self._optimize_weights(
                regime_features, regime_targets, performance_metric
            )
            
            # Apply constraints and smoothing
            smoothed_weights = self._apply_weight_constraints(
                optimal_weights, regime_id
            )
            
            # Update regime weights
            old_weights = self.regime_weights[regime_id].copy()
            self.regime_weights[regime_id] = smoothed_weights
            
            # Store in history
            self.weight_history[regime_id].append(smoothed_weights.copy())
            if len(self.weight_history[regime_id]) > 50:
                self.weight_history[regime_id].pop(0)
            
            # Calculate performance metrics
            metrics = self._calculate_update_metrics(
                old_weights, smoothed_weights, regime_features, regime_targets
            )
            
            tprint_info(f"  Updated regime {regime_id} weights: improvement={metrics.get('improvement', 0):.4f}")
            
            return metrics
            
        except Exception as e:
            tprint_warning(f"Weight update failed for regime {regime_id}: {e}")
            return {'error': str(e)}
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Categorize features by type for weight initialization."""
        categories = {
            'volatility_features': [],
            'momentum_features': [],
            'trend_features': [],
            'microstructure_features': [],
            'correlation_features': [],
            'stability_features': [],
        }
        
        for feature in feature_names:
            feature_lower = feature.lower()
            
            if any(vol in feature_lower for vol in ['volatility', 'vol', 'parkinson', 'atr']):
                categories['volatility_features'].append(feature)
            elif any(mom in feature_lower for mom in ['momentum', 'impulse', 'decay']):
                categories['momentum_features'].append(feature)
            elif any(trend in feature_lower for trend in ['trend', 'path_', 'efficiency']):
                categories['trend_features'].append(feature)
            elif any(micro in feature_lower for micro in ['spread', 'imbalance', 'volume', 'order']):
                categories['microstructure_features'].append(feature)
            elif any(corr in feature_lower for corr in ['correlation', 'dominance', 'btc']):
                categories['correlation_features'].append(feature)
            elif any(stab in feature_lower for stab in ['stability', 'kurtosis', 'skewness']):
                categories['stability_features'].append(feature)
            else:
                # Default to trend features
                categories['trend_features'].append(feature)
        
        return categories
    
    def _get_initial_weights(self, regime_id: int) -> np.ndarray:
        """Get initial weights for regime based on feature categories."""
        n_features = len(self.feature_names)
        weights = np.zeros(n_features)
        
        for category, category_weight in self.config.feature_categories.items():
            category_features = self.feature_categories.get(category, [])
            
            for feature in category_features:
                if feature in self.feature_names:
                    feature_idx = self.feature_names.index(feature)
                    # Distribute category weight evenly among features
                    weight_per_feature = category_weight / max(len(category_features), 1)
                    weights[feature_idx] = weight_per_feature
        
        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # Fallback to equal weights
            weights = np.ones(n_features) / n_features
        
        return weights
    
    def _get_default_weights(self) -> np.ndarray:
        """Get default equal weights."""
        n_features = len(self.feature_names)
        return np.ones(n_features) / n_features
    
    def _optimize_weights(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        performance_metric: str = 'correlation'
    ) -> np.ndarray:
        """Optimize weights using Ridge regression."""
        # Prepare data
        X = features.values
        y = targets
        
        # Handle NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        if np.sum(valid_mask) < 10:
            return self._get_default_weights()
        
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        try:
            # Use Ridge regression for robust weight estimation
            ridge = Ridge(
                alpha=self.config.regularization_strength,
                fit_intercept=True,
                random_state=42
            )
            
            ridge.fit(X_valid, y_valid)
            
            # Use absolute coefficients as feature importance weights
            raw_weights = np.abs(ridge.coef_)
            
            # Normalize to sum to 1
            if np.sum(raw_weights) > 0:
                weights = raw_weights / np.sum(raw_weights)
            else:
                weights = self._get_default_weights()
            
            return weights
            
        except Exception as e:
            tprint_warning(f"Weight optimization failed: {e}")
            return self._get_default_weights()
    
    def _apply_weight_constraints(
        self,
        weights: np.ndarray,
        regime_id: int
    ) -> np.ndarray:
        """Apply constraints to weights for stability."""
        constrained_weights = weights.copy()
        
        # Apply per-feature weight limits
        constrained_weights = np.clip(
            constrained_weights,
            self.config.min_weight_per_feature,
            self.config.max_weight_per_feature
        )
        
        # Re-normalize to sum to 1
        if np.sum(constrained_weights) > 0:
            constrained_weights = constrained_weights / np.sum(constrained_weights)
        
        # Apply weight change limit (if previous weights exist)
        if regime_id in self.regime_weights:
            old_weights = self.regime_weights[regime_id]
            weight_change = np.abs(constrained_weights - old_weights)
            
            # Limit large changes
            excessive_change = weight_change > self.config.weight_change_limit
            if np.any(excessive_change):
                # Smooth the change
                constrained_weights[excessive_change] = (
                    old_weights[excessive_change] + 
                    np.sign(constrained_weights[excessive_change] - old_weights[excessive_change]) * 
                    self.config.weight_change_limit
                )
                
                # Re-normalize
                constrained_weights = constrained_weights / np.sum(constrained_weights)
        
        return constrained_weights
    
    def _apply_transition_smoothing(
        self,
        base_weights: np.ndarray,
        regime_probabilities: np.ndarray
    ) -> np.ndarray:
        """Apply smooth transition based on regime probabilities."""
        smoothed_weights = base_weights.copy()
        
        # Weighted average with neighboring regimes based on probabilities
        for regime_id, prob in enumerate(regime_probabilities):
            if prob > 0.01 and regime_id in self.regime_weights:  # Significant probability
                regime_weights = self.regime_weights[regime_id]
                
                # Blend weights based on probability
                blend_factor = prob * self.config.transition_smoothing_factor
                smoothed_weights = (
                    (1 - blend_factor) * smoothed_weights + 
                    blend_factor * regime_weights
                )
        
        return smoothed_weights
    
    def _calculate_update_metrics(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        features: pd.DataFrame,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate performance metrics for weight update."""
        metrics = {}
        
        try:
            # Calculate weight change
            weight_change = np.mean(np.abs(new_weights - old_weights))
            metrics['weight_change'] = weight_change
            
            # Calculate performance improvement
            old_performance = self._calculate_weight_performance(old_weights, features, targets)
            new_performance = self._calculate_weight_performance(new_weights, features, targets)
            
            metrics['old_performance'] = old_performance
            metrics['new_performance'] = new_performance
            metrics['improvement'] = new_performance - old_performance
            
            # Calculate weight entropy (diversification)
            entropy = -np.sum(new_weights * np.log(new_weights + 1e-10))
            metrics['weight_entropy'] = entropy
            
        except Exception as e:
            tprint_warning(f"Metric calculation failed: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _calculate_weight_performance(
        self,
        weights: np.ndarray,
        features: pd.DataFrame,
        targets: np.ndarray
    ) -> float:
        """Calculate performance metric for given weights."""
        try:
            # Calculate weighted feature combination
            weighted_features = (features.values * weights).sum(axis=1)
            
            # Calculate correlation with targets
            correlation = np.corrcoef(weighted_features, targets)[0, 1]
            
            return abs(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def get_regime_weight_summary(self) -> Dict[int, Dict[str, Any]]:
        """Get summary of weights for all regimes."""
        summary = {}
        
        for regime_id, weights in self.regime_weights.items():
            # Get top features
            top_indices = np.argsort(weights)[-5:][::-1]
            top_features = [
                (self.feature_names[i], weights[i]) 
                for i in top_indices
            ]
            
            summary[regime_id] = {
                'top_features': top_features,
                'weight_entropy': -np.sum(weights * np.log(weights + 1e-10)),
                'max_weight': np.max(weights),
                'min_weight': np.min(weights),
                'weight_std': np.std(weights),
            }
        
        return summary
