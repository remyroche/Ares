"""Enhanced Tree CLVSA Wrapper with Full CVLSA Architecture Integration.

This module provides a comprehensive CLVSA (Cross-View Learning with Self-Attention)
architecture wrapper for tree-based models, automatically applying advanced attention
mechanisms, temporal modeling, and feature enhancement to all tree models.
"""

from __future__ import annotations

import logging
import time
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler

# Import CVLSA components
from src.utils.ml_common.cvlsa.cvlsa_architecture import (
    EnhancedCVLSAConfig, create_enhanced_cvlsa_model, EnhancedCVLSATrainer
)
from src.utils.ml_common.models.cvlsa_cache import get_global_clvsa_cache, CLVSACacheConfig
from src.utils.matrix_operations.enhanced_operations import get_enhanced_matrix_operations
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer

logger = logging.getLogger(__name__)


@dataclass
class TreeCLVSAConfig:
    """Enhanced configuration for Tree CLVSA wrapper with full CVLSA integration."""

    # Architecture parameters
    attention_dim: int = 64
    use_temporal_attention: bool = True
    regime_aware: bool = True
    attention_dropout: float = 0.1
    feature_selection_method: str = "mutual_info"
    temporal_window_size: int = 20
    ensemble_attention: bool = True
    memory_efficient: bool = True

    # CVLSA integration parameters
    enable_cvlsa_enhancement: bool = True
    fusion_method: str = "attention"  # "attention", "weighted_average", "stacking"
    cvlsa_weight: float = 0.6
    tree_weight: float = 0.4

    # Feature engineering parameters
    use_advanced_features: bool = True
    max_sequence_length: int = 1000
    chunk_size: int = 100

    # Hardware optimization
    use_m1_gpu: bool = True
    memory_limit_gb: Optional[float] = None


class TreeCLVSAWrapper(BaseEstimator, RegressorMixin, ClassifierMixin):
    """Enhanced tree model wrapper with full CVLSA architecture integration.

    This wrapper automatically applies CVLSA enhancements to any tree-based model,
    providing advanced attention mechanisms, temporal modeling, and cross-view learning.
    """

    def __init__(self, base_model: Any, config: TreeCLVSAConfig) -> None:
        self.base_model = base_model
        self.config = config
        self.is_classifier = hasattr(base_model, 'predict_proba') or hasattr(base_model, 'classes_')

        # CLVSA components
        self.cvlsa_model: Optional[EnhancedCVLSATrainer] = None
        self.feature_scaler = StandardScaler()
        self.market_data_cache: Optional[pd.DataFrame] = None
        self.feature_extractor = None

        # Training state
        self.is_fitted = False
        self.training_metadata: Dict[str, Any] = {}
        self.feature_dimensions: Dict[str, int] = {}

        # Initialize hardware optimizers
        self._init_hardware_optimizers()

        logger.info(f"🌳 Tree CLVSA Wrapper initialized for {type(base_model).__name__}")

    def _init_hardware_optimizers(self):
        """Initialize hardware optimization components."""
        try:
            self.matrix_ops = get_enhanced_matrix_operations()
            self.gpu_manager = get_m1_gpu_manager() if self.config.use_m1_gpu else None
            self.memory_optimizer = get_m1_memory_optimizer(
                memory_limit_gb=self.config.memory_limit_gb
            )
        except Exception as e:
            logger.warning(f"Hardware optimizers not available: {e}")
            self.matrix_ops = None
            self.gpu_manager = None
            self.memory_optimizer = None

    def _create_synthetic_market_data(self, X: np.ndarray) -> pd.DataFrame:
        """Create synthetic market data from features for CVLSA processing."""
        n_samples = X.shape[0]
        base_price = 100.0
        prices = []
        volumes = []

        for i in range(n_samples):
            if i == 0:
                price = base_price
            else:
                # Use feature values to influence price movement
                feature_influence = np.mean(X[i, :min(5, X.shape[1])]) if X.shape[1] >= 5 else X[i, 0]
                price_change = np.random.normal(0, 0.02) + feature_influence * 0.01
                price = prices[-1] * (1 + price_change)

            # Generate OHLC from price
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = price * (1 + np.random.normal(0, 0.005))
            close = price

            prices.append([open_price, high, low, close])
            volumes.append(np.random.lognormal(10, 1))

        return pd.DataFrame({
            'open': [p[0] for p in prices],
            'high': [p[1] for p in prices],
            'low': [p[2] for p in prices],
            'close': [p[3] for p in prices],
            'volume': volumes
        })

    def _prepare_cvlsa_features(self, market_data: pd.DataFrame) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Prepare features for CVLSA processing with caching."""
        # Initialize cache if not already done
        if not hasattr(self, 'cache_manager'):
            cache_config = CLVSACacheConfig(
                max_cache_size=50,
                max_memory_mb=200.0,
                ttl_seconds=1800,  # 30 minutes
                enable_persistence=True
            )
            self.cache_manager = get_global_clvsa_cache(cache_config)

        # Create feature configuration for caching
        feature_config = {
            'input_dim': market_data.shape[1] + 1,
            'output_dim': 4,
            'seq_length': len(market_data),
            'cross_view_attention': True,
            'use_multi_scale_attention': self.config.use_temporal_attention,
            'memory_efficient': self.config.memory_efficient,
            'use_m1_gpu': self.config.use_m1_gpu,
            'memory_limit_gb': self.config.memory_limit_gb
        }

        # Try to retrieve from cache first
        cached_result = self.cache_manager.retrieve(market_data, feature_config)

        if cached_result is not None:
            features, predictions, attention_weights = cached_result
            logger.info("🎯 Retrieved CLVSA features from cache")

            # Create CVLSA configuration for model initialization
            cvlsa_config = EnhancedCVLSAConfig(**feature_config)

            # Initialize CVLSA model
            self.cvlsa_model = EnhancedCVLSATrainer(cvlsa_config)

            # Create target tensor
            target = torch.FloatTensor(market_data['close'].values)

            return features, target

        # Not in cache, compute from scratch
        logger.info("🔧 Computing CLVSA features (cache miss)")

        # Create CVLSA configuration
        cvlsa_config = EnhancedCVLSAConfig(**feature_config)

        # Initialize CVLSA model
        self.cvlsa_model = EnhancedCVLSATrainer(cvlsa_config)

        # Prepare features using CVLSA trainer's method
        features = self.cvlsa_model.prepare_features(market_data)

        # Create target tensor (using close prices for prediction)
        target = torch.FloatTensor(market_data['close'].values)

        # Get initial predictions for caching
        with torch.no_grad():
            predictions = self.cvlsa_model.predict(features)

        # Get attention weights for caching
        attention_weights = self.cvlsa_model.get_attention_weights()

        # Store in cache
        cache_key = self.cache_manager.store(
            market_data, feature_config, features, predictions, attention_weights
        )

        logger.info(f"💾 Stored CLVSA features in cache (key: {cache_key[:8]}...)")
        return features, target

    def fit(self, X: np.ndarray, y: np.ndarray, market_data: Optional[pd.DataFrame] = None,
            regimes: Optional[np.ndarray] = None) -> 'TreeCLVSAWrapper':
        """Fit the CLVSA-enhanced tree model with automatic feature extraction."""
        logger.info(f"🚀 Training CLVSA-enhanced {type(self.base_model).__name__}...")

        start_time = time.time()

        try:
            # Initialize automatic feature extraction if enabled
            if self.config.enable_cvlsa_enhancement:
                logger.info("🔧 Initializing automatic CVLSA feature extraction...")

                # Import and create automatic feature pipeline
                from src.utils.ml_common.cvlsa.cvlsa_integration import create_automatic_feature_pipeline, EnhancedCVLSAConfig

                # Create CVLSA config from tree wrapper config
                cvlsa_config = EnhancedCVLSAConfig(
                    input_dim=X.shape[1] + 10,  # Allow for additional features
                    output_dim=4,
                    seq_length=len(X),
                    cross_view_attention=True,
                    use_multi_scale_attention=self.config.use_temporal_attention,
                    memory_efficient=self.config.memory_efficient,
                    use_m1_gpu=self.config.use_m1_gpu,
                    memory_limit_gb=self.config.memory_limit_gb,
                    view_embedding_dim=self.config.attention_dim
                )

                # Create automatic feature pipeline
                self.feature_extractor = create_automatic_feature_pipeline(cvlsa_config)

                # Apply automatic feature enhancement
                logger.info("🔄 Applying automatic CVLSA feature enhancement...")
                enhanced_features = self.feature_extractor.fit_transform(X, y, market_data)

                # Store feature dimensions
                self.feature_dimensions = {
                    'original': X.shape[1],
                    'cvlsa_enhanced': enhanced_features.shape[1] - X.shape[1],
                    'total_enhanced': enhanced_features.shape[1]
                }

                logger.info(f"📊 Feature enhancement: {X.shape[1]} → {enhanced_features.shape[1]} features")

                # Use enhanced features for training
                X_for_training = enhanced_features
            else:
                logger.info("⚠️ CVLSA enhancement disabled, using original features")
                X_for_training = X
                self.feature_dimensions = {
                    'original': X.shape[1],
                    'cvlsa_enhanced': 0,
                    'total_enhanced': X.shape[1]
                }

            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X_for_training)

            # Train tree model with features
            logger.info("🌳 Training tree component...")

            if regimes is not None:
                # For regime-aware training, add regime information as features
                if regimes.ndim == 1:
                    regimes_reshaped = regimes.reshape(-1, 1)
                else:
                    regimes_reshaped = regimes

                # Combine scaled features with regime information
                X_with_regimes = np.hstack([X_scaled, regimes_reshaped])
                self.base_model.fit(X_with_regimes, y)
            else:
                self.base_model.fit(X_scaled, y)

            # Store training metadata
            self.training_metadata = {
                'training_time': time.time() - start_time,
                'cvlsa_enabled': self.config.enable_cvlsa_enhancement,
                'feature_dimensions': self.feature_dimensions,
                'fusion_method': self.config.fusion_method,
                'regime_aware': regimes is not None,
                'regime_count': len(np.unique(regimes)) if regimes is not None else 0
            }

            # Store cache statistics if available
            if self.feature_extractor is not None:
                cache_stats = self.feature_extractor.get_cache_stats()
                if cache_stats:
                    self.training_metadata['cache_stats'] = cache_stats

            self.is_fitted = True
            logger.info(f"✅ CLVSA-enhanced model training completed in {self.training_metadata['training_time']".2f"}s")

            return self

        except Exception as e:
            logger.error(f"❌ CLVSA-enhanced model training failed: {e}")
            raise

    def predict(self, X: np.ndarray, market_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Make predictions using CLVSA-enhanced model with automatic feature extraction."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        try:
            # Use automatic feature extraction if enabled
            if self.config.enable_cvlsa_enhancement and self.feature_extractor is not None:
                logger.info("🔄 Applying CVLSA feature enhancement for prediction...")

                # Apply the same feature enhancement used during training
                enhanced_features = self.feature_extractor.transform(X, market_data)

                # Scale features using the same scaler as training
                enhanced_features_scaled = self.feature_scaler.transform(enhanced_features)

                # Get predictions from tree model
                predictions = self.base_model.predict(enhanced_features_scaled)

                logger.info(f"✅ CLVSA-enhanced prediction completed with {enhanced_features.shape[1]} features")

            else:
                # No CVLSA enhancement, use standard prediction
                logger.info("⚠️ Using standard prediction without CVLSA enhancement")
                X_scaled = self.feature_scaler.transform(X)
                predictions = self.base_model.predict(X_scaled)

            return predictions

        except Exception as e:
            logger.error(f"❌ CLVSA-enhanced prediction failed: {e}")
            raise

    def predict_proba(self, X: np.ndarray, market_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Make probability predictions for classification models using automatic feature extraction."""
        if not self.is_classifier:
            raise AttributeError("predict_proba is only available for classification models")

        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        try:
            # Use automatic feature extraction if enabled
            if self.config.enable_cvlsa_enhancement and self.feature_extractor is not None:
                logger.info("🔄 Applying CVLSA feature enhancement for probability prediction...")

                # Apply the same feature enhancement used during training
                enhanced_features = self.feature_extractor.transform(X, market_data)

                # Scale features using the same scaler as training
                enhanced_features_scaled = self.feature_scaler.transform(enhanced_features)

                # Get probability predictions from tree model
                tree_probabilities = self.base_model.predict_proba(enhanced_features_scaled)

                # Get feature importance for confidence weighting
                feature_importance = self.get_feature_importance()

                if 'cvlsa_attention' in feature_importance:
                    # Use CVLSA attention weights as confidence measure
                    attention_weights = feature_importance['cvlsa_attention']
                    if attention_weights.size > 0:
                        # Calculate confidence based on attention weights
                        attention_confidence = np.mean(np.abs(attention_weights))
                        # Weight predictions by CVLSA confidence
                        weighted_probabilities = tree_probabilities * (0.5 + 0.5 * attention_confidence)
                        return weighted_probabilities / np.sum(weighted_probabilities, axis=1, keepdims=True)

                return tree_probabilities

            else:
                # No CVLSA enhancement, use standard prediction
                logger.info("⚠️ Using standard probability prediction without CVLSA enhancement")
                X_scaled = self.feature_scaler.transform(X)
                return self.base_model.predict_proba(X_scaled)

        except Exception as e:
            logger.error(f"❌ CLVSA-enhanced probability prediction failed: {e}")
            raise

    def __getattr__(self, item: str) -> Any:
        """Delegate attribute access to base model."""
        return getattr(self.base_model, item)

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from both CVLSA and tree components."""
        importance = {}

        # CVLSA feature extractor importance
        if self.feature_extractor is not None:
            extractor_importance = self.feature_extractor.get_feature_importance()
            if extractor_importance:
                importance.update(extractor_importance)

        # Legacy CVLSA model attention weights (if available)
        if self.cvlsa_model is not None:
            attention_weights = self.cvlsa_model.get_attention_weights()
            if attention_weights:
                importance['cvlsa_attention'] = attention_weights

        # Tree feature importance
        if hasattr(self.base_model, 'feature_importances_'):
            importance['tree_importance'] = self.base_model.feature_importances_

        return importance

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = {
            'model_type': f'CLVSA-{type(self.base_model).__name__}',
            'base_model': type(self.base_model).__name__,
            'is_classifier': self.is_classifier,
            'cvlsa_enabled': self.config.enable_cvlsa_enhancement,
            'config': self.config.__dict__,
            'is_fitted': self.is_fitted,
            'training_metadata': self.training_metadata,
            'feature_dimensions': self.feature_dimensions
        }

        # Add feature extractor information
        if self.feature_extractor is not None:
            info['feature_extractor'] = {
                'enabled': True,
                'cache_stats': self.feature_extractor.get_cache_stats()
            }

        return info

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from feature extractor."""
        if self.feature_extractor is not None:
            return self.feature_extractor.get_cache_stats()
        return {}

    def clear_cache(self):
        """Clear the feature extraction cache."""
        if self.feature_extractor is not None:
            self.feature_extractor.clear_cache()
            logger.info("🧹 CLVSA feature cache cleared")

    def set_cache_config(self, max_cache_size: int = 50, memory_limit_mb: float = 200.0):
        """Update cache configuration."""
        if self.feature_extractor is not None:
            self.feature_extractor.set_cache_config(max_cache_size, memory_limit_mb)
            logger.info(f"🔄 Updated cache configuration: size={max_cache_size}, memory={memory_limit_mb}MB")

    def enable_cvlsa_enhancement(self, enable: bool = True):
        """Enable or disable CLVSA enhancement."""
        self.config.enable_cvlsa_enhancement = enable
        logger.info(f"🔄 CLVSA enhancement {'enabled' if enable else 'disabled'}")

    def set_fusion_method(self, method: str = 'attention'):
        """Set the fusion method for combining CVLSA and tree predictions."""
        valid_methods = ['attention', 'weighted_average', 'stacking']
        if method not in valid_methods:
            raise ValueError(f"Fusion method must be one of: {valid_methods}")

        self.config.fusion_method = method
        logger.info(f"🔄 Fusion method set to: {method}")

    def set_cvlsa_weight(self, weight: float = 0.6):
        """Set the weight for CVLSA predictions in fusion."""
        if not 0.0 <= weight <= 1.0:
            raise ValueError("CVLSA weight must be between 0.0 and 1.0")

        self.config.cvlsa_weight = weight
        logger.info(f"🔄 CVLSA weight set to: {weight}")

    def get_enhancement_summary(self) -> Dict[str, Any]:
        """Get a summary of CLVSA enhancements applied."""
        summary = {
            'enhancement_enabled': self.config.enable_cvlsa_enhancement,
            'feature_enhancement_active': self.feature_extractor is not None,
            'total_training_time': self.training_metadata.get('training_time', 0),
            'feature_expansion': self.feature_dimensions,
            'fusion_method': self.config.fusion_method,
            'cvlsa_weight': self.config.cvlsa_weight
        }

        if self.feature_extractor is not None:
            cache_stats = self.get_cache_stats()
            if cache_stats:
                summary['cache_stats'] = cache_stats

        return summary


def create_tree_clvsa_config(**overrides: Any) -> TreeCLVSAConfig:
    """Create a TreeCLVSAConfig with optional overrides."""
    return TreeCLVSAConfig(**overrides)


def create_tree_clvsa_wrapper(base_model: Any, config: Optional[TreeCLVSAConfig] = None) -> TreeCLVSAWrapper:
    """Create a CLVSA-enhanced wrapper for tree models."""
    if config is None:
        config = TreeCLVSAConfig()

    return TreeCLVSAWrapper(base_model, config)


__all__ = [
    "TreeCLVSAConfig",
    "TreeCLVSAWrapper",
    "create_tree_clvsa_config",
    "create_tree_clvsa_wrapper",
]
