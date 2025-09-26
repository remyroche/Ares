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
        """Fit the CLVSA-enhanced tree model."""
        logger.info(f"🚀 Training CLVSA-enhanced {type(self.base_model).__name__}...")

        start_time = time.time()

        try:
            # Prepare market data for CVLSA
            if market_data is None:
                market_data = self._create_synthetic_market_data(X)

            # Cache market data for reuse
            self.market_data_cache = market_data

            # Prepare CVLSA features
            cvlsa_features, target = self._prepare_cvlsa_features(market_data)

            # Train CVLSA model
            if self.config.enable_cvlsa_enhancement:
                logger.info("🔧 Training CVLSA component...")
                cvlsa_results = self.cvlsa_model.train(cvlsa_features, cvlsa_features, target)

                # Get CVLSA predictions for tree training
                with torch.no_grad():
                    cvlsa_predictions = self.cvlsa_model.predict(cvlsa_features)
                    cvlsa_features_np = cvlsa_predictions.cpu().numpy()

                # Combine original features with CVLSA features
                enhanced_features = np.hstack([X, cvlsa_features_np])

                # Scale features
                enhanced_features_scaled = self.feature_scaler.fit_transform(enhanced_features)

                # Store feature dimensions
                self.feature_dimensions = {
                    'original': X.shape[1],
                    'cvlsa_enhanced': cvlsa_features_np.shape[1],
                    'total_enhanced': enhanced_features.shape[1]
                }
            else:
                enhanced_features_scaled = self.feature_scaler.fit_transform(X)
                self.feature_dimensions = {
                    'original': X.shape[1],
                    'cvlsa_enhanced': 0,
                    'total_enhanced': X.shape[1]
                }

            # Train tree model with enhanced features
            logger.info("🌳 Training tree component with CLVSA enhancements...")

            if regimes is not None:
                # For regime-aware training, we could pass regimes as additional features
                # For now, just fit with enhanced features
                self.base_model.fit(enhanced_features_scaled, y)
            else:
                self.base_model.fit(enhanced_features_scaled, y)

            # Store training metadata
            self.training_metadata = {
                'training_time': time.time() - start_time,
                'cvlsa_enabled': self.config.enable_cvlsa_enhancement,
                'feature_dimensions': self.feature_dimensions,
                'fusion_method': self.config.fusion_method,
                'regime_aware': regimes is not None
            }

            self.is_fitted = True
            logger.info(f"✅ CLVSA-enhanced model training completed in {self.training_metadata['training_time']".2f"}s")

            return self

        except Exception as e:
            logger.error(f"❌ CLVSA-enhanced model training failed: {e}")
            raise

    def predict(self, X: np.ndarray, market_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Make predictions using CLVSA-enhanced model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        try:
            # Use cached market data or create new if not provided
            if market_data is None and self.market_data_cache is not None:
                market_data = self.market_data_cache
            elif market_data is None:
                market_data = self._create_synthetic_market_data(X)

            # Get CVLSA predictions
            cvlsa_features, _ = self._prepare_cvlsa_features(market_data)

            if self.config.enable_cvlsa_enhancement:
                with torch.no_grad():
                    cvlsa_predictions = self.cvlsa_model.predict(cvlsa_features)
                    cvlsa_features_np = cvlsa_predictions.cpu().numpy()

                # Combine features
                enhanced_features = np.hstack([X, cvlsa_features_np])
                enhanced_features_scaled = self.feature_scaler.transform(enhanced_features)

                # Get tree predictions
                tree_predictions = self.base_model.predict(enhanced_features_scaled)

                # Fuse predictions based on method
                if self.config.fusion_method == 'weighted_average':
                    predictions = (self.config.cvlsa_weight * cvlsa_predictions.cpu().numpy() +
                                 self.config.tree_weight * tree_predictions)
                elif self.config.fusion_method == 'attention':
                    # Use CVLSA attention weights for fusion
                    attention_weights = self.cvlsa_model.get_attention_weights()
                    if 'cross_view' in attention_weights:
                        # Weight by attention importance
                        cvlsa_importance = np.mean(attention_weights['cross_view'])
                        predictions = (cvlsa_importance * cvlsa_predictions.cpu().numpy() +
                                     (1 - cvlsa_importance) * tree_predictions)
                    else:
                        predictions = (self.config.cvlsa_weight * cvlsa_predictions.cpu().numpy() +
                                     self.config.tree_weight * tree_predictions)
                else:
                    predictions = tree_predictions
            else:
                # No CVLSA enhancement, just use tree model
                enhanced_features_scaled = self.feature_scaler.transform(X)
                predictions = self.base_model.predict(enhanced_features_scaled)

            return predictions

        except Exception as e:
            logger.error(f"❌ CLVSA-enhanced prediction failed: {e}")
            raise

    def predict_proba(self, X: np.ndarray, market_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Make probability predictions for classification models."""
        if not self.is_classifier:
            raise AttributeError("predict_proba is only available for classification models")

        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        try:
            # Use same logic as predict but for probabilities
            if market_data is None and self.market_data_cache is not None:
                market_data = self.market_data_cache
            elif market_data is None:
                market_data = self._create_synthetic_market_data(X)

            cvlsa_features, _ = self._prepare_cvlsa_features(market_data)

            if self.config.enable_cvlsa_enhancement:
                with torch.no_grad():
                    cvlsa_predictions = self.cvlsa_model.predict(cvlsa_features)
                    cvlsa_features_np = cvlsa_predictions.cpu().numpy()

                enhanced_features = np.hstack([X, cvlsa_features_np])
                enhanced_features_scaled = self.feature_scaler.transform(enhanced_features)

                tree_predictions = self.base_model.predict_proba(enhanced_features_scaled)

                # For classification, use CVLSA predictions as confidence weights
                cvlsa_confidence = np.mean(np.abs(cvlsa_predictions.cpu().numpy()), axis=1, keepdims=True)
                cvlsa_confidence = cvlsa_confidence / np.max(cvlsa_confidence)  # Normalize

                # Weight predictions by CVLSA confidence
                weighted_predictions = tree_predictions * cvlsa_confidence
                return weighted_predictions / np.sum(weighted_predictions, axis=1, keepdims=True)
            else:
                enhanced_features_scaled = self.feature_scaler.transform(X)
                return self.base_model.predict_proba(enhanced_features_scaled)

        except Exception as e:
            logger.error(f"❌ CLVSA-enhanced probability prediction failed: {e}")
            raise

    def __getattr__(self, item: str) -> Any:
        """Delegate attribute access to base model."""
        return getattr(self.base_model, item)

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from both CVLSA and tree components."""
        importance = {}

        # CVLSA attention weights
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
        return {
            'model_type': f'CLVSA-{type(self.base_model).__name__}',
            'base_model': type(self.base_model).__name__,
            'is_classifier': self.is_classifier,
            'cvlsa_enabled': self.config.enable_cvlsa_enhancement,
            'config': self.config.__dict__,
            'is_fitted': self.is_fitted,
            'training_metadata': self.training_metadata,
            'feature_dimensions': self.feature_dimensions
        }


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
