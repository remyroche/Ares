"""
CLVSA Attention Wrapper for Tree-Based Models

This module provides CLVSA-style attention mechanisms for tree-based models
(XGBoost, LightGBM, CatBoost) without requiring full PyTorch implementation.

Key Features:
1. Preprocessing attention for tree models
2. Feature importance weighting
3. Temporal attention patterns
4. Regime-aware feature selection
5. Lightweight implementation using sklearn
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import logging

logger = logging.getLogger(__name__)


class CLVSAAttentionWrapper(BaseEstimator, RegressorMixin):
    """CLVSA-style attention wrapper for tree-based models."""

    def __init__(self, base_model, attention_dim: int = 64,
                 use_temporal_attention: bool = True,
                 regime_aware: bool = True,
                 attention_dropout: float = 0.1):
        """Initialize CLVSA attention wrapper.

        Args:
            base_model: Tree-based model (XGBoost, LightGBM, CatBoost)
            attention_dim: Attention dimension
            use_temporal_attention: Whether to use temporal attention
            regime_aware: Whether to use regime-aware attention
            attention_dropout: Attention dropout rate
        """
        self.base_model = base_model
        self.attention_dim = attention_dim
        self.use_temporal_attention = use_temporal_attention
        self.regime_aware = regime_aware
        self.attention_dropout = attention_dropout

        # Attention components
        self.feature_attention_weights = None
        self.temporal_attention_weights = None
        self.regime_attention_weights = None
        self.scaler = StandardScaler()

    def _compute_feature_attention(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute feature attention weights using mutual information."""
        try:
            # Use mutual information for feature importance
            mi_scores = mutual_info_regression(X, y, random_state=42)
            mi_scores = mi_scores / np.max(mi_scores)  # Normalize to [0, 1]

            # Add small noise to prevent zero weights
            mi_scores = mi_scores + np.random.normal(0, 0.01, size=mi_scores.shape)

            # Apply softmax-like transformation
            attention_weights = np.exp(mi_scores) / np.sum(np.exp(mi_scores))

            return attention_weights

        except Exception as e:
            logger.warning(f"⚠️ Feature attention computation failed: {e}")
            return np.ones(X.shape[1]) / X.shape[1]  # Uniform weights

    def _compute_temporal_attention(self, X: np.ndarray) -> np.ndarray:
        """Compute temporal attention patterns."""
        try:
            # Use rolling window analysis for temporal patterns
            window_size = min(20, len(X) // 10)

            temporal_weights = np.zeros(X.shape[1])

            for i in range(X.shape[1]):
                feature_series = X[:, i]

                # Calculate autocorrelation for different lags
                autocorrs = []
                for lag in range(1, min(10, len(feature_series) // 2)):
                    if len(feature_series) > lag:
                        autocorr = np.corrcoef(feature_series[:-lag], feature_series[lag:])[0, 1]
                        autocorrs.append(abs(autocorr))

                # Average autocorrelation as temporal importance
                temporal_importance = np.mean(autocorrs) if autocorrs else 0.0
                temporal_weights[i] = temporal_importance

            # Normalize temporal weights
            if np.sum(temporal_weights) > 0:
                temporal_weights = temporal_weights / np.sum(temporal_weights)
            else:
                temporal_weights = np.ones(X.shape[1]) / X.shape[1]

            return temporal_weights

        except Exception as e:
            logger.warning(f"⚠️ Temporal attention computation failed: {e}")
            return np.ones(X.shape[1]) / X.shape[1]  # Uniform weights

    def _apply_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """Apply attention weights to features."""
        if self.feature_attention_weights is None:
            return X

        # Apply feature attention
        X_weighted = X * self.feature_attention_weights[None, :]

        # Apply temporal attention if enabled
        if self.use_temporal_attention and self.temporal_attention_weights is not None:
            X_weighted = X_weighted * self.temporal_attention_weights[None, :]

        # Apply dropout to attention weights
        if self.attention_dropout > 0:
            attention_mask = np.random.binomial(
                1, 1 - self.attention_dropout,
                size=X.shape[1]
            )
            X_weighted = X_weighted * attention_mask[None, :]

        return X_weighted

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            regimes: Optional[np.ndarray] = None) -> 'CLVSAAttentionWrapper':
        """Fit the CLVSA attention wrapper."""

        # Compute feature attention weights
        self.feature_attention_weights = self._compute_feature_attention(X, y)

        # Compute temporal attention weights
        if self.use_temporal_attention:
            self.temporal_attention_weights = self._compute_temporal_attention(X)

        # Compute regime-aware attention if enabled
        if self.regime_aware and regimes is not None:
            self.regime_attention_weights = self._compute_regime_attention(X, y, regimes)

        # Apply attention to training data
        X_attentioned = self._apply_attention_weights(X)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_attentioned)

        # Fit base model
        if sample_weight is not None:
            self.base_model.fit(X_scaled, y, sample_weight=sample_weight)
        else:
            self.base_model.fit(X_scaled, y)

        logger.info(f"✅ CLVSA Attention Wrapper fitted with {X.shape[1]} features")
        return self

    def _compute_regime_attention(self, X: np.ndarray, y: np.ndarray,
                                regimes: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute regime-specific attention weights."""
        try:
            unique_regimes = np.unique(regimes)
            regime_weights = {}

            for regime in unique_regimes:
                regime_mask = regimes == regime
                if np.sum(regime_mask) < 10:
                    continue

                X_regime = X[regime_mask]
                y_regime = y[regime_mask]

                # Compute attention weights for this regime
                regime_attention = self._compute_feature_attention(X_regime, y_regime)
                regime_weights[regime] = regime_attention

            return regime_weights

        except Exception as e:
            logger.warning(f"⚠️ Regime attention computation failed: {e}")
            return {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using CLVSA attention wrapper."""
        # Apply attention weights
        X_attentioned = self._apply_attention_weights(X)

        # Scale features
        X_scaled = self.scaler.transform(X_attentioned)

        # Make predictions
        return self.base_model.predict(X_scaled)

    def get_attention_weights(self) -> Dict[str, Any]:
        """Get attention weights for analysis."""
        return {
            'feature_attention': self.feature_attention_weights,
            'temporal_attention': self.temporal_attention_weights,
            'regime_attention': self.regime_attention_weights
        }


# Factory function for creating CLVSA wrappers
def create_clvsa_wrapper(base_model, config: Dict[str, Any]):
    """Create CLVSA wrapper for tree-based models."""
    return CLVSAAttentionWrapper(
        base_model=base_model,
        attention_dim=config.get('attention_dim', 64),
        use_temporal_attention=config.get('use_temporal_attention', True),
        regime_aware=config.get('regime_aware', True),
        attention_dropout=config.get('attention_dropout', 0.1)
    )


# Integration with existing model factory
def integrate_clvsa_with_existing_models():
    """Integration guide for existing model factory."""
    integration_code = """
# Example integration in model_factory.py

def _create_attention_xgboost_model(self, model_config: ModelConfig) -> Any:
    \"\"\"Create CLVSA-enhanced XGBoost model.\"\"\"
    try:
        from src.training.steps.model_training.clvsa_attention_wrapper import create_clvsa_wrapper

        # Create base XGBoost model
        base_model = self._create_xgboost_model(model_config)

        # Wrap with CLVSA attention if requested
        use_clvsa = model_config.model_params.get('use_clvsa', False)

        if use_clvsa:
            clvsa_config = model_config.model_params.get('clvsa_config', {})
            return create_clvsa_wrapper(base_model, clvsa_config)

        return base_model

    except Exception as e:
        logger.warning(f"⚠️ CLVSA-XGBoost creation failed: {e}")
        return self._create_xgboost_model(model_config)
"""

    return integration_code