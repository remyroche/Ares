"""
TAS (Tree Architecture Search) Integration Component

Integrates Tree Architecture Search functionality from ml_common TAS system
for hybrid regime detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime

from ..config.hybrid_regime_config import HybridRegimeConfig

logger = logging.getLogger(__name__)


class TASIntegrationComponent:
    """
    TAS Integration Component

    Integrates Tree Architecture Search functionality from the ml_common TAS system
    to provide tree-based feature extraction and regime detection capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize TAS integration component."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize TAS components
        self._initialize_tas_components()

        self.logger.info("✅ TAS Integration Component initialized")
        self.logger.info(f"   Tree models: {len(self.config.get('tree_models', []))}")
        self.logger.info(f"   Feature importance threshold: {self.config.get('min_feature_importance', 0.01)}")

    def _initialize_tas_components(self):
        """Initialize TAS-specific components."""
        try:
            # Import TAS components dynamically
            from src.utils.ml_common.optimization.tas.regime_analysis.clustering_regime_detection import (
                TreeBasedClusteringRegimeDetector,
                ClusteringRegimeConfig
            )

            # Create TAS regime detector
            tas_config = ClusteringRegimeConfig(
                clustering_strategy=self.config.get('clustering_strategy', 'auto'),
                n_regimes=self.config.get('n_regimes', 8),
                tree_models=self.config.get('tree_models', [
                    "random_forest", "xgboost", "lightgbm", "extra_trees"
                ]),
                max_features_per_model=self.config.get('max_features_per_model', 50),
                min_feature_importance=self.config.get('min_feature_importance', 0.01)
            )

            self.tas_detector = TreeBasedClusteringRegimeDetector(tas_config)

            self.logger.info("✅ TAS components initialized successfully")

        except ImportError as e:
            self.logger.warning(f"TAS components not available: {e}, using fallback")
            self.tas_detector = None

    def extract_features(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract features using TAS approach.

        Args:
            market_data: Market data for feature extraction

        Returns:
            Tuple of (features, metadata)
        """
        try:
            if self.tas_detector is not None:
                # Use TAS detector for feature extraction
                tas_results = self.tas_detector.detect_regimes(market_data)

                # Extract features from TAS results
                features = self._extract_features_from_tas_results(tas_results)

                # Add metadata
                metadata = {
                    'method': 'tas_detector',
                    'feature_dimensions': features.shape[1] if features.ndim > 1 else 1,
                    'confidence': tas_results.get('clustering_metrics', {}).get('silhouette_score', 0.5),
                    'strategy': tas_results.get('strategy', 'unknown'),
                    'execution_time': tas_results.get('execution_time', 0.0)
                }

                return features, metadata

            else:
                # Fallback to manual feature extraction
                return self._extract_tas_features_fallback(market_data)

        except Exception as e:
            self.logger.warning(f"TAS feature extraction failed: {e}, using fallback")
            return self._extract_tas_features_fallback(market_data)

    def _extract_features_from_tas_results(self, tas_results: Dict[str, Any]) -> np.ndarray:
        """Extract features from TAS detector results."""
        try:
            # Use ensemble predictions or features from TAS results
            if 'ensemble_predictions' in tas_results:
                features = tas_results['ensemble_predictions']
            elif 'features' in tas_results:
                features = tas_results['features']
            else:
                # Create features from regime predictions
                labels = tas_results.get('labels', np.array([]))
                if len(labels) > 0:
                    # Create one-hot encoded regime features
                    n_regimes = len(set(labels))
                    features = np.zeros((len(labels), n_regimes))
                    for i, label in enumerate(labels):
                        if 0 <= label < n_regimes:
                            features[i, label] = 1.0
                else:
                    raise ValueError("No valid features found in TAS results")

            return features

        except Exception as e:
            self.logger.error(f"Feature extraction from TAS results failed: {e}")
            raise

    def _extract_tas_features_fallback(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fallback TAS feature extraction."""
        try:
            self.logger.info("🔄 Using fallback TAS feature extraction")

            # Extract comprehensive tree-based features
            features = self._extract_tree_based_features(market_data)

            # Calculate confidence based on feature quality
            confidence = self._calculate_feature_confidence(features)

            metadata = {
                'method': 'fallback',
                'feature_dimensions': features.shape[1] if features.ndim > 1 else 1,
                'confidence': confidence,
                'strategy': 'manual',
                'execution_time': 0.0
            }

            return features, metadata

        except Exception as e:
            self.logger.error(f"Fallback TAS feature extraction failed: {e}")
            # Return minimal features
            basic_features = market_data['close'].values.reshape(-1, 1)
            return basic_features, {
                'method': 'minimal',
                'feature_dimensions': 1,
                'confidence': 0.0,
                'strategy': 'error',
                'execution_time': 0.0
            }

    def _extract_tree_based_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract comprehensive tree-based features."""
        try:
            # Price-based features
            close_prices = market_data['close'].values
            high_prices = market_data['high'].values
            low_prices = market_data['low'].values
            open_prices = market_data['open'].values

            # Volume features
            volume = market_data.get('volume', np.ones(len(market_data))).values

            # Calculate returns
            returns = np.diff(close_prices, prepend=close_prices[0])
            returns = returns.reshape(-1, 1)

            # Calculate price ratios
            high_low_ratio = (high_prices - low_prices) / close_prices
            high_low_ratio = high_low_ratio.reshape(-1, 1)

            # Calculate volume ratios
            volume_ratio = volume / np.mean(volume) if np.mean(volume) > 0 else volume
            volume_ratio = volume_ratio.reshape(-1, 1)

            # Calculate moving averages and volatility
            ma_5 = pd.Series(close_prices).rolling(window=5, min_periods=1).mean().values
            ma_10 = pd.Series(close_prices).rolling(window=10, min_periods=1).mean().values
            ma_20 = pd.Series(close_prices).rolling(window=20, min_periods=1).mean().values

            volatility_10 = pd.Series(close_prices).rolling(window=10, min_periods=1).std().values
            volatility_20 = pd.Series(close_prices).rolling(window=20, min_periods=1).std().values

            # Calculate momentum indicators
            momentum_5 = close_prices - ma_5
            momentum_10 = close_prices - ma_10

            # Calculate RSI-like indicator
            gains = np.maximum(returns.ravel(), 0)
            losses = np.abs(np.minimum(returns.ravel(), 0))
            avg_gain = pd.Series(gains).rolling(window=14, min_periods=1).mean().values
            avg_loss = pd.Series(losses).rolling(window=14, min_periods=1).mean().values
            rs = avg_gain / np.where(avg_loss == 0, 1, avg_loss)
            rsi = 100 - (100 / (1 + rs))

            # Calculate Bollinger Bands
            bb_upper = ma_20 + 2 * volatility_20
            bb_lower = ma_20 - 2 * volatility_20
            bb_width = (bb_upper - bb_lower) / ma_20

            # Combine all features
            features_list = [
                close_prices.reshape(-1, 1),
                returns,
                high_low_ratio,
                volume_ratio,
                ma_5.reshape(-1, 1),
                ma_10.reshape(-1, 1),
                ma_20.reshape(-1, 1),
                volatility_10.reshape(-1, 1),
                volatility_20.reshape(-1, 1),
                momentum_5.reshape(-1, 1),
                momentum_10.reshape(-1, 1),
                rsi.reshape(-1, 1),
                bb_width.reshape(-1, 1)
            ]

            # Filter out NaN values
            valid_features = []
            for feature in features_list:
                if not np.isnan(feature).all():
                    valid_features.append(feature)

            if not valid_features:
                # Fallback to basic features
                return close_prices.reshape(-1, 1)

            # Combine features
            features = np.hstack(valid_features)

            # Remove rows with NaN
            mask = ~np.isnan(features).any(axis=1)
            features = features[mask]

            return features

        except Exception as e:
            self.logger.error(f"Tree-based feature extraction failed: {e}")
            # Return basic features as fallback
            return market_data['close'].values.reshape(-1, 1)

    def _calculate_feature_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence score for extracted features."""
        try:
            # Basic confidence based on feature diversity and quality
            n_features = features.shape[1] if features.ndim > 1 else 1
            n_samples = features.shape[0]

            # Feature diversity score (higher is better)
            if n_features > 1:
                feature_variance = np.var(features, axis=0)
                diversity_score = np.mean(feature_variance > 0.01)  # Features with meaningful variance
            else:
                diversity_score = 0.5

            # Data quality score (higher is better)
            data_quality = 1.0 - np.mean(np.isnan(features))

            # Sample adequacy score (higher is better)
            sample_adequacy = min(n_samples / 100, 1.0)  # Need at least 100 samples

            # Combine scores
            confidence = (
                0.4 * diversity_score +
                0.4 * data_quality +
                0.2 * sample_adequacy
            )

            return min(confidence, 1.0)

        except Exception as e:
            self.logger.warning(f"Feature confidence calculation failed: {e}")
            return 0.5


def create_tas_integration(config: Dict[str, Any]) -> TASIntegrationComponent:
    """Create TAS integration component."""
    return TASIntegrationComponent(config)