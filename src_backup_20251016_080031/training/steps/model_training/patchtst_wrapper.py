"""
PatchTST Wrapper for Tree-Based Models

This module provides PatchTST-style transformer-based feature enhancement for tree-based models
(XGBoost, LightGBM, CatBoost, Random Forest) to improve their performance on time series data.

Key Features:
1. Patch-based time series segmentation
2. Transformer-style attention mechanisms
3. Temporal feature enhancement
4. Regime-aware patch selection
5. Lightweight implementation using sklearn and numpy
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


class PatchTSTWrapper(BaseEstimator, RegressorMixin):
    """PatchTST-style transformer wrapper for tree-based models."""

    def __init__(self, base_model, patch_len: int = 16, stride: int = 8,
                 use_transformer_attention: bool = True,
                 regime_aware: bool = True,
                 attention_dropout: float = 0.1,
                 num_heads: int = 4,
                 sign_dropout_rate: float = 0.0,
                 sign_threshold: float = 0.2):
        """Initialize PatchTST wrapper.

        Args:
            base_model: Tree-based model (XGBoost, LightGBM, CatBoost, Random Forest)
            patch_len: Length of each patch for time series segmentation
            stride: Stride for patch extraction
            use_transformer_attention: Whether to use transformer-style attention
            regime_aware: Whether to use regime-aware patch selection
            attention_dropout: Attention dropout rate
            num_heads: Number of attention heads
            sign_dropout_rate: Dropout rate applied to dominant sign activations in a patch
            sign_threshold: Minimum dominance difference before sign-based dropout is applied
        """
        self.base_model = base_model
        self.patch_len = patch_len
        self.stride = stride
        self.use_transformer_attention = use_transformer_attention
        self.regime_aware = regime_aware
        self.attention_dropout = attention_dropout
        self.num_heads = num_heads
        self.sign_dropout_rate = sign_dropout_rate
        self.sign_threshold = sign_threshold

        # PatchTST components
        self.patch_embeddings = None
        self.attention_weights = None
        self.regime_patch_weights = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self._last_dropout_stats: Optional[Dict[str, Any]] = None

    def _create_patches(self, X: np.ndarray) -> np.ndarray:
        """Create patches from time series data."""
        try:
            n_samples, n_features = X.shape
            patches = []
            
            for i in range(0, n_samples - self.patch_len + 1, self.stride):
                patch = X[i:i + self.patch_len, :]
                patches.append(patch.flatten())
            
            if not patches:
                # If no patches can be created, use the original data
                return X
            
            return np.array(patches)
        
        except Exception as e:
            logger.warning(f"⚠️ Patch creation failed: {e}")
            return X

    def _compute_patch_attention(self, patches: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute attention weights for patches using transformer-style mechanism."""
        try:
            n_patches, patch_dim = patches.shape
            
            # Initialize attention weights
            attention_weights = np.ones(n_patches)
            
            if self.use_transformer_attention:
                # Compute patch importance using mutual information
                if len(y) >= len(patches):
                    y_aligned = y[:len(patches)]
                else:
                    # Repeat y to match patch length
                    y_aligned = np.tile(y, (len(patches) // len(y) + 1))[:len(patches)]
                
                # Use mutual information for patch importance
                mi_scores = []
                for i in range(0, patch_dim, self.patch_len):
                    patch_features = patches[:, i:i + self.patch_len]
                    if patch_features.shape[1] > 0:
                        mi_score = mutual_info_regression(
                            patch_features, y_aligned, random_state=42
                        )
                        mi_scores.extend(mi_score)
                
                if mi_scores:
                    # Average MI scores for each patch
                    patch_scores = []
                    for i in range(n_patches):
                        patch_start = i * self.patch_len
                        patch_end = min((i + 1) * self.patch_len, len(mi_scores))
                        if patch_start < len(mi_scores):
                            patch_score = np.mean(mi_scores[patch_start:patch_end])
                            patch_scores.append(patch_score)
                        else:
                            patch_scores.append(0.0)
                    
                    # Normalize and apply softmax
                    patch_scores = np.array(patch_scores)
                    if np.sum(patch_scores) > 0:
                        patch_scores = patch_scores / np.max(patch_scores)
                        attention_weights = np.exp(patch_scores) / np.sum(np.exp(patch_scores))
            
            return attention_weights

        except Exception as e:
            logger.warning(f"⚠️ Patch attention computation failed: {e}")
            return np.ones(patches.shape[0]) / patches.shape[0]

    def _apply_patch_attention(self, patches: np.ndarray) -> np.ndarray:
        """Apply attention weights to patches."""
        if self.attention_weights is None:
            self._last_dropout_stats = None
            return patches

        # Apply attention weights
        weighted_patches = patches * self.attention_weights[:, None]

        dropout_stats = {
            'attention_dropout_rate': self.attention_dropout,
            'sign_dropout_rate': self.sign_dropout_rate,
            'positive_dropped': 0,
            'negative_dropped': 0,
            'positive_total': 0,
            'negative_total': 0
        }

        # Apply dropout to attention weights
        if self.attention_dropout > 0:
            attention_mask = np.random.binomial(
                1, 1 - self.attention_dropout,
                size=patches.shape[0]
            )
            weighted_patches = weighted_patches * attention_mask[:, None]

        sign_mask = np.ones_like(weighted_patches) if self.sign_dropout_rate > 0 else None

        for idx, patch in enumerate(weighted_patches):
            positive_mask = patch > 0
            negative_mask = patch < 0
            pos_count = int(np.sum(positive_mask))
            neg_count = int(np.sum(negative_mask))

            dropout_stats['positive_total'] += pos_count
            dropout_stats['negative_total'] += neg_count

            total_count = pos_count + neg_count
            if total_count == 0 or self.sign_dropout_rate <= 0:
                continue

            pos_ratio = pos_count / total_count
            neg_ratio = neg_count / total_count

            dominant_sign = None
            if pos_ratio - neg_ratio >= self.sign_threshold:
                dominant_sign = 'positive'
            elif neg_ratio - pos_ratio >= self.sign_threshold:
                dominant_sign = 'negative'

            if dominant_sign == 'positive' and pos_count > 0:
                drop_samples = np.random.rand(pos_count) < self.sign_dropout_rate
                mask_values = np.ones(pos_count, dtype=weighted_patches.dtype)
                mask_values[drop_samples] = 0.0
                sign_mask[idx, positive_mask] = mask_values
                dropout_stats['positive_dropped'] += int(np.sum(drop_samples))
            elif dominant_sign == 'negative' and neg_count > 0:
                drop_samples = np.random.rand(neg_count) < self.sign_dropout_rate
                mask_values = np.ones(neg_count, dtype=weighted_patches.dtype)
                mask_values[drop_samples] = 0.0
                sign_mask[idx, negative_mask] = mask_values
                dropout_stats['negative_dropped'] += int(np.sum(drop_samples))

        if sign_mask is not None:
            weighted_patches = weighted_patches * sign_mask

        if dropout_stats['positive_total'] > 0:
            active = dropout_stats['positive_total'] - dropout_stats['positive_dropped']
            dropout_stats['positive_active_ratio'] = active / dropout_stats['positive_total']
        else:
            dropout_stats['positive_active_ratio'] = 1.0

        if dropout_stats['negative_total'] > 0:
            active = dropout_stats['negative_total'] - dropout_stats['negative_dropped']
            dropout_stats['negative_active_ratio'] = active / dropout_stats['negative_total']
        else:
            dropout_stats['negative_active_ratio'] = 1.0

        self._last_dropout_stats = dropout_stats

        return weighted_patches

    def _compute_regime_patch_weights(self, patches: np.ndarray, y: np.ndarray,
                                    regimes: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute regime-specific patch weights."""
        try:
            unique_regimes = np.unique(regimes)
            regime_weights = {}

            for regime in unique_regimes:
                regime_mask = regimes == regime
                if np.sum(regime_mask) < 10:
                    continue

                # Align regime mask with patches
                if len(regimes) >= len(patches):
                    regime_patch_mask = regimes[:len(patches)] == regime
                else:
                    regime_patch_mask = np.tile(regimes, (len(patches) // len(regimes) + 1))[:len(patches)] == regime

                if np.sum(regime_patch_mask) < 5:
                    continue

                patches_regime = patches[regime_patch_mask]
                y_regime = y[:len(patches_regime)] if len(y) >= len(patches_regime) else y

                # Compute attention weights for this regime
                regime_attention = self._compute_patch_attention(patches_regime, y_regime)
                regime_weights[regime] = regime_attention

            return regime_weights

        except Exception as e:
            logger.warning(f"⚠️ Regime patch weights computation failed: {e}")
            return {}

    def _enhance_features_with_patches(self, X: np.ndarray) -> np.ndarray:
        """Enhance features using patch-based transformations."""
        try:
            # Create patches
            patches = self._create_patches(X)
            
            # Apply attention if available
            if self.attention_weights is not None:
                patches = self._apply_patch_attention(patches)
            
            # Create enhanced features by combining original and patch features
            if patches.shape[0] > 0:
                # Pad or truncate patches to match original data length
                if patches.shape[0] < X.shape[0]:
                    # Pad with last patch
                    padding = np.tile(patches[-1:], (X.shape[0] - patches.shape[0], 1))
                    patches = np.vstack([patches, padding])
                elif patches.shape[0] > X.shape[0]:
                    # Truncate to match original length
                    patches = patches[:X.shape[0]]
                
                # Combine original features with patch features
                enhanced_features = np.hstack([X, patches])
            else:
                enhanced_features = X
            
            return enhanced_features

        except Exception as e:
            logger.warning(f"⚠️ Feature enhancement failed: {e}")
            return X

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            regimes: Optional[np.ndarray] = None) -> 'PatchTSTWrapper':
        """Fit the PatchTST wrapper."""

        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
            X = X.values

        # Create patches and compute attention weights
        patches = self._create_patches(X)
        self.attention_weights = self._compute_patch_attention(patches, y)

        # Compute regime-aware patch weights if enabled
        if self.regime_aware and regimes is not None:
            self.regime_patch_weights = self._compute_regime_patch_weights(patches, y, regimes)

        # Enhance features with patch-based transformations
        X_enhanced = self._enhance_features_with_patches(X)

        if self._last_dropout_stats is not None:
            stats = self._last_dropout_stats
            logger.info(
                "PatchTST dropout stats — attention: %.3f, sign: %.3f, "+
                "positive active ratio: %.3f (%d/%d), negative active ratio: %.3f (%d/%d)",
                stats.get('attention_dropout_rate', 0.0),
                stats.get('sign_dropout_rate', 0.0),
                stats.get('positive_active_ratio', 0.0),
                stats.get('positive_total', 0) - stats.get('positive_dropped', 0),
                stats.get('positive_total', 0),
                stats.get('negative_active_ratio', 0.0),
                stats.get('negative_total', 0) - stats.get('negative_dropped', 0),
                stats.get('negative_total', 0)
            )

        # Scale features
        X_scaled = self.scaler.fit_transform(X_enhanced)

        # Fit base model
        if sample_weight is not None:
            self.base_model.fit(X_scaled, y, sample_weight=sample_weight)
        else:
            self.base_model.fit(X_scaled, y)

        logger.info(f"✅ PatchTST Wrapper fitted with {X.shape[1]} features, {patches.shape[0]} patches")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using PatchTST-style enhancement."""
        # Convert to numpy if pandas DataFrame
        if hasattr(X, 'values'):
            X = X.values

        # Enhance features with patch-based transformations
        X_enhanced = self._enhance_features_with_patches(X)

        return X_enhanced

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using PatchTST wrapper."""
        # Convert to numpy if pandas DataFrame
        if hasattr(X, 'values'):
            X = X.values

        # Enhance features with patch-based transformations
        X_enhanced = self._enhance_features_with_patches(X)

        # Scale features
        X_scaled = self.scaler.transform(X_enhanced)

        # Make predictions
        return self.base_model.predict(X_scaled)

    def get_patch_attention_weights(self) -> Dict[str, Any]:
        """Get patch attention weights for analysis."""
        return {
            'patch_attention': self.attention_weights,
            'regime_patch_weights': self.regime_patch_weights,
            'patch_len': self.patch_len,
            'stride': self.stride
        }

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from the base model."""
        if hasattr(self.base_model, 'feature_importances_'):
            return self.base_model.feature_importances_
        elif hasattr(self.base_model, 'coef_'):
            return np.abs(self.base_model.coef_)
        else:
            logger.warning("⚠️ Base model does not support feature importance")
            return np.array([])


# Factory function for creating PatchTST wrappers
def create_patchtst_wrapper(base_model, patch_len=16, stride=8, use_transformer_attention=True,
                          regime_aware=True, attention_dropout=0.1, num_heads=4,
                          sign_dropout_rate: float = 0.0, sign_threshold: float = 0.2):
    """Create PatchTST wrapper for tree-based models."""
    return PatchTSTWrapper(
        base_model=base_model,
        patch_len=patch_len,
        stride=stride,
        use_transformer_attention=use_transformer_attention,
        regime_aware=regime_aware,
        attention_dropout=attention_dropout,
        num_heads=num_heads,
        sign_dropout_rate=sign_dropout_rate,
        sign_threshold=sign_threshold
    )


# Integration with existing model factory
def integrate_patchtst_with_existing_models():
    """Integration guide for existing model factory."""
    integration_code = """
# Example integration in model_factory.py

def _create_patchtst_xgboost_model(self, model_config: ModelConfig) -> Any:
    \"\"\"Create PatchTST-enhanced XGBoost model.\"\"\"
    try:
        from src.training.steps.model_training.patchtst_wrapper import create_patchtst_wrapper

        # Create base XGBoost model
        base_model = self._create_xgboost_model(model_config)

        # Wrap with PatchTST if requested
        use_patchtst = model_config.model_params.get('use_patchtst', False)

        if use_patchtst:
            patchtst_config = model_config.model_params.get('patchtst_config', {})
            return create_patchtst_wrapper(base_model, **patchtst_config)

        return base_model

    except Exception as e:
        logger.warning(f"⚠️ PatchTST-XGBoost creation failed: {e}")
        return self._create_xgboost_model(model_config)

def _create_patchtst_lightgbm_model(self, model_config: ModelConfig) -> Any:
    \"\"\"Create PatchTST-enhanced LightGBM model.\"\"\"
    try:
        from src.training.steps.model_training.patchtst_wrapper import create_patchtst_wrapper

        # Create base LightGBM model
        base_model = self._create_lightgbm_model(model_config)

        # Wrap with PatchTST if requested
        use_patchtst = model_config.model_params.get('use_patchtst', False)

        if use_patchtst:
            patchtst_config = model_config.model_params.get('patchtst_config', {})
            return create_patchtst_wrapper(base_model, **patchtst_config)

        return base_model

    except Exception as e:
        logger.warning(f"⚠️ PatchTST-LightGBM creation failed: {e}")
        return self._create_lightgbm_model(model_config)
"""

    return integration_code
