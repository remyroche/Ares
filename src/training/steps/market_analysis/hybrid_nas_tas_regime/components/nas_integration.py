"""
NAS (Neural Architecture Search) Integration Component

Integrates Neural Architecture Search functionality from nas_regime/
for hybrid regime detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime

from ..config.hybrid_regime_config import HybridRegimeConfig
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)


class NASIntegrationComponent:
    """
    NAS Integration Component

    Integrates Neural Architecture Search functionality from the nas_regime system
    to provide neural network-based feature extraction and regime detection capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize NAS integration component."""
        tprint_info("🚀 Initializing NAS Integration Component")
        tprint_debug(f"Configuration: {config}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize NAS components
        tprint_info("🔧 Initializing NAS components...")
        self._initialize_nas_components()

        tprint_success("✅ NAS Integration Component initialized")
        tprint_info(f"   Architecture: {self.config.get('primary_architecture', 'hybrid')}")
        tprint_info(f"   Neural ODEs: {self.config.get('enable_neural_odes', True)}")
        tprint_info(f"   Vision Transformers: {self.config.get('enable_vision_transformers', True)}")
        self.logger.info("✅ NAS Integration Component initialized")
        self.logger.info(f"   Architecture: {self.config.get('primary_architecture', 'hybrid')}")
        self.logger.info(f"   Neural ODEs: {self.config.get('enable_neural_odes', True)}")
        self.logger.info(f"   Vision Transformers: {self.config.get('enable_vision_transformers', True)}")

    def _initialize_nas_components(self):
        """Initialize NAS-specific components."""
        tprint_debug("🔧 Initializing NAS-specific components...")
        try:
            # Import NAS components dynamically
            tprint_debug("📦 Importing NAS components...")
            from src.training.steps.market_analysis.nas_regime.core.perfect_nas_regime_detector import (
                PerfectNASRegimeDetector,
                PerfectNASConfig,
                PerfectNASResult
            )
            tprint_success("✅ NAS components imported")

            # Create NAS regime detector
            tprint_debug("🔍 Creating NAS regime detector...")
            nas_config = PerfectNASConfig(
                primary_architecture=self.config.get('primary_architecture', 'hybrid'),
                enable_neural_odes=self.config.get('enable_neural_odes', True),
                enable_vision_transformers=self.config.get('enable_vision_transformers', True),
                enable_meta_learning=self.config.get('enable_meta_learning', True),
                search_strategy=self.config.get('search_strategy', 'evolutionary'),
                n_regimes=self.config.get('n_regimes', 8)
            )

            self.nas_detector = PerfectNASRegimeDetector(nas_config)

            self.logger.info("✅ NAS components initialized successfully")

        except ImportError as e:
            self.logger.warning(f"NAS components not available: {e}, using fallback")
            self.nas_detector = None

    def extract_features(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract features using NAS approach.

        Args:
            market_data: Market data for feature extraction

        Returns:
            Tuple of (features, metadata)
        """
        try:
            tprint("🧠 [NAS_INTEGRATION] Starting NAS feature extraction", color="blue", bold=True)
            tprint_debug(f"📊 [NAS_INTEGRATION] Market data shape: {market_data.shape}")
            tprint_debug(f"📊 [NAS_INTEGRATION] Market data columns: {list(market_data.columns)}")
            tprint_debug(f"🔧 [NAS_INTEGRATION] NAS detector available: {self.nas_detector is not None}")
            
            if self.nas_detector is not None:
                tprint("🔧 [NAS_INTEGRATION] Using NAS detector for feature extraction", color="blue")
                # Use NAS detector for feature extraction
                nas_results = self.nas_detector.detect_regimes(market_data)
                tprint_success(f"✅ [NAS_INTEGRATION] NAS regime detection completed")
                tprint_debug(f"📈 [NAS_INTEGRATION] NAS results keys: {list(nas_results.keys()) if isinstance(nas_results, dict) else 'Not a dict'}")

                # Extract features from NAS results
                tprint("🔧 [NAS_INTEGRATION] Extracting features from NAS results", color="cyan")
                features = self._extract_features_from_nas_results(nas_results)
                tprint_success(f"✅ [NAS_INTEGRATION] Features extracted: {features.shape}")
                tprint_performance(f"⚡ [NAS_INTEGRATION] NAS features: {features.shape[0]} samples, {features.shape[1]} features")

                # Calculate adaptive weight based on performance
                tprint("⚖️ [NAS_INTEGRATION] Calculating adaptive weight", color="cyan")
                adaptive_weight = self._calculate_adaptive_weight(nas_results)
                tprint_debug(f"⚖️ [NAS_INTEGRATION] Adaptive weight: {adaptive_weight:.3f}")

                # Add metadata
                tprint("📊 [NAS_INTEGRATION] Building metadata", color="cyan")
                metadata = {
                    'method': 'nas_detector',
                    'feature_dimensions': features.shape[1] if features.ndim > 1 else 1,
                    'confidence': self._calculate_nas_confidence(nas_results),
                    'architecture': self.config.get('primary_architecture', 'unknown'),
                    'execution_time': nas_results.execution_time if hasattr(nas_results, 'execution_time') else 0.0,
                    'adaptive_weight': adaptive_weight,
                    'performance_metrics': self._extract_performance_metrics(nas_results),
                    'feature_quality': self._calculate_feature_quality(features)
                }
                tprint_success(f"✅ [NAS_INTEGRATION] Metadata built: {len(metadata)} fields")
                tprint_debug(f"📊 [NAS_INTEGRATION] Confidence: {metadata['confidence']:.3f}, Architecture: {metadata['architecture']}")

                tprint_success(f"🎉 [NAS_INTEGRATION] NAS feature extraction completed successfully")
                return features, metadata

            else:
                tprint_warning("⚠️ [NAS_INTEGRATION] NAS detector not available, using fallback")
                tprint_debug(f"🔍 [NAS_INTEGRATION] NAS detector status: {self.nas_detector is None}")
                # Fallback to manual feature extraction
                return self._extract_nas_features_fallback(market_data)

        except Exception as e:
            tprint_error(f"❌ [NAS_INTEGRATION] NAS feature extraction failed: {e}")
            tprint_debug(f"🔍 [NAS_INTEGRATION] Error details: {str(e)}")
            self.logger.warning(f"NAS feature extraction failed: {e}, using fallback")
            tprint("🔄 [NAS_INTEGRATION] Using fallback feature extraction", color="yellow")
            return self._extract_nas_features_fallback(market_data)

    def _extract_features_from_nas_results(self, nas_results) -> np.ndarray:
        """Extract features from NAS detector results."""
        try:
            # Use regime probabilities as features
            if hasattr(nas_results, 'regime_probabilities') and nas_results.regime_probabilities.size > 0:
                features = nas_results.regime_probabilities
            elif hasattr(nas_results, 'regime_predictions') and nas_results.regime_predictions.size > 0:
                # Create one-hot encoded features from predictions
                predictions = nas_results.regime_predictions
                n_regimes = len(set(predictions))
                features = np.zeros((len(predictions), n_regimes))
                for i, pred in enumerate(predictions):
                    if 0 <= pred < n_regimes:
                        features[i, pred] = 1.0
            else:
                raise ValueError("No valid features found in NAS results")

            return features

        except Exception as e:
            self.logger.error(f"Feature extraction from NAS results failed: {e}")
            raise

    def _calculate_nas_confidence(self, nas_results) -> float:
        """Calculate confidence score from NAS results."""
        try:
            confidence = 0.5  # Base confidence

            # Factor in success status
            if hasattr(nas_results, 'success') and nas_results.success:
                confidence += 0.2

            # Factor in economic significance
            if hasattr(nas_results, 'economic_significance_scores'):
                avg_economic = np.mean(nas_results.economic_significance_scores)
                confidence += 0.2 * min(avg_economic, 1.0)

            # Factor in financial relevance
            if hasattr(nas_results, 'trading_viability_scores'):
                avg_financial = np.mean(nas_results.trading_viability_scores)
                confidence += 0.1 * min(avg_financial, 1.0)

            return min(confidence, 1.0)

        except Exception as e:
            self.logger.warning(f"NAS confidence calculation failed: {e}")
            return 0.5

    def _extract_nas_features_fallback(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fallback NAS feature extraction."""
        try:
            self.logger.info("🔄 Using fallback NAS feature extraction")

            # Extract neural network-inspired features
            features = self._extract_neural_features(market_data)

            # Calculate confidence based on feature quality
            confidence = self._calculate_feature_confidence(features)

            metadata = {
                'method': 'fallback',
                'feature_dimensions': features.shape[1] if features.ndim > 1 else 1,
                'confidence': confidence,
                'architecture': 'manual',
                'execution_time': 0.0
            }

            return features, metadata

        except Exception as e:
            self.logger.error(f"Fallback NAS feature extraction failed: {e}")
            # Return minimal features
            basic_features = market_data['close'].values.reshape(-1, 1)
            return basic_features, {
                'method': 'minimal',
                'feature_dimensions': 1,
                'confidence': 0.0,
                'architecture': 'error',
                'execution_time': 0.0
            }

    def _extract_neural_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract neural network-inspired features."""
        try:
            # Price-based features
            close_prices = market_data['close'].values
            high_prices = market_data['high'].values
            low_prices = market_data['low'].values
            open_prices = market_data['open'].values

            # Volume features
            volume = market_data.get('volume', np.ones(len(market_data))).values

            # Calculate returns with different lookbacks
            returns_1 = np.diff(close_prices, prepend=close_prices[0])
            returns_5 = pd.Series(close_prices).pct_change(5).fillna(0).values
            returns_10 = pd.Series(close_prices).pct_change(10).fillna(0).values
            returns_20 = pd.Series(close_prices).pct_change(20).fillna(0).values

            # Calculate volatility with different windows
            vol_5 = pd.Series(close_prices).rolling(window=5, min_periods=1).std().values
            vol_10 = pd.Series(close_prices).rolling(window=10, min_periods=1).std().values
            vol_20 = pd.Series(close_prices).rolling(window=20, min_periods=1).std().values

            # Calculate moving averages
            ma_5 = pd.Series(close_prices).rolling(window=5, min_periods=1).mean().values
            ma_10 = pd.Series(close_prices).rolling(window=10, min_periods=1).mean().values
            ma_20 = pd.Series(close_prices).rolling(window=20, min_periods=1).mean().values

            # Calculate exponential moving averages
            ema_5 = pd.Series(close_prices).ewm(span=5, adjust=False).mean().values
            ema_10 = pd.Series(close_prices).ewm(span=10, adjust=False).mean().values
            ema_20 = pd.Series(close_prices).ewm(span=20, adjust=False).mean().values

            # Calculate price ratios and differences
            high_close_ratio = (high_prices - close_prices) / close_prices
            low_close_ratio = (close_prices - low_prices) / close_prices
            open_close_ratio = (open_prices - close_prices) / close_prices

            # Calculate volume-price relationships
            volume_price_corr_10 = pd.Series(volume).rolling(window=10, min_periods=1).corr(
                pd.Series(close_prices)
            ).fillna(0).values

            # Calculate momentum indicators
            roc_5 = (close_prices - pd.Series(close_prices).shift(5).fillna(close_prices[0])) / close_prices
            roc_10 = (close_prices - pd.Series(close_prices).shift(10).fillna(close_prices[0])) / close_prices

            # Calculate stochastic oscillators
            lowest_low_14 = pd.Series(low_prices).rolling(window=14, min_periods=1).min()
            highest_high_14 = pd.Series(high_prices).rolling(window=14, min_periods=1).max()
            stoch_k = 100 * (close_prices - lowest_low_14) / (highest_high_14 - lowest_low_14)
            stoch_k = stoch_k.fillna(50).values  # Fill NaN with neutral value

            # Calculate Williams %R
            williams_r = -100 * (highest_high_14 - close_prices) / (highest_high_14 - lowest_low_14)
            williams_r = williams_r.fillna(-50).values

            # Combine all features
            features_list = [
                close_prices.reshape(-1, 1),
                returns_1.reshape(-1, 1),
                returns_5.reshape(-1, 1),
                returns_10.reshape(-1, 1),
                returns_20.reshape(-1, 1),
                vol_5.reshape(-1, 1),
                vol_10.reshape(-1, 1),
                vol_20.reshape(-1, 1),
                ma_5.reshape(-1, 1),
                ma_10.reshape(-1, 1),
                ma_20.reshape(-1, 1),
                ema_5.reshape(-1, 1),
                ema_10.reshape(-1, 1),
                ema_20.reshape(-1, 1),
                high_close_ratio.reshape(-1, 1),
                low_close_ratio.reshape(-1, 1),
                open_close_ratio.reshape(-1, 1),
                volume_price_corr_10.reshape(-1, 1),
                roc_5.reshape(-1, 1),
                roc_10.reshape(-1, 1),
                stoch_k.reshape(-1, 1),
                williams_r.reshape(-1, 1),
                volume.reshape(-1, 1)
            ]

            # Filter out NaN values and empty arrays
            valid_features = []
            for feature in features_list:
                if (isinstance(feature, np.ndarray) and
                    feature.size > 0 and
                    not np.isnan(feature).all()):
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
            self.logger.error(f"Neural feature extraction failed: {e}")
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

    def _calculate_adaptive_weight(self, nas_results) -> float:
        """Calculate adaptive weight based on NAS performance."""
        try:
            base_weight = self.config.get('base_weight', 0.6)
            performance_weight = self.config.get('performance_weight', 0.3)

            # Extract performance metrics
            performance_score = 0.0

            # Success factor
            if hasattr(nas_results, 'success') and nas_results.success:
                performance_score += 0.3

            # Economic significance factor
            if hasattr(nas_results, 'economic_significance_scores'):
                avg_economic = np.mean(nas_results.economic_significance_scores)
                performance_score += 0.4 * min(avg_economic, 1.0)

            # Financial relevance factor
            if hasattr(nas_results, 'trading_viability_scores'):
                avg_financial = np.mean(nas_results.trading_viability_scores)
                performance_score += 0.3 * min(avg_financial, 1.0)

            # Calculate adaptive weight
            adaptive_weight = base_weight + performance_weight * performance_score

            # Apply bounds
            min_weight = self.config.get('min_weight', 0.1)
            max_weight = self.config.get('max_weight', 0.9)

            return max(min_weight, min(max_weight, adaptive_weight))

        except Exception as e:
            self.logger.warning(f"Adaptive weight calculation failed: {e}")
            return self.config.get('base_weight', 0.6)

    def _extract_performance_metrics(self, nas_results) -> Dict[str, float]:
        """Extract performance metrics from NAS results."""
        try:
            metrics = {}

            # Success metric
            metrics['success'] = 1.0 if hasattr(nas_results, 'success') and nas_results.success else 0.0

            # Economic significance
            if hasattr(nas_results, 'economic_significance_scores'):
                metrics['avg_economic_significance'] = np.mean(nas_results.economic_significance_scores)
                metrics['max_economic_significance'] = np.max(nas_results.economic_significance_scores)
                metrics['min_economic_significance'] = np.min(nas_results.economic_significance_scores)

            # Financial viability
            if hasattr(nas_results, 'trading_viability_scores'):
                metrics['avg_financial_viability'] = np.mean(nas_results.trading_viability_scores)
                metrics['max_financial_viability'] = np.max(nas_results.trading_viability_scores)
                metrics['min_financial_viability'] = np.min(nas_results.trading_viability_scores)

            # Execution time
            if hasattr(nas_results, 'execution_time'):
                metrics['execution_time'] = nas_results.execution_time

            # Architecture used
            metrics['architecture'] = self.config.get('primary_architecture', 'unknown')

            return metrics

        except Exception as e:
            self.logger.warning(f"Performance metrics extraction failed: {e}")
            return {}

    def _calculate_feature_quality(self, features: np.ndarray) -> Dict[str, float]:
        """Calculate feature quality metrics for NAS features."""
        try:
            if features.size == 0:
                return {'quality_score': 0.0}

            quality = {}

            # Feature variance (higher is better)
            feature_variance = np.var(features, axis=0)
            quality['avg_feature_variance'] = np.mean(feature_variance)
            quality['min_feature_variance'] = np.min(feature_variance)
            quality['max_feature_variance'] = np.max(feature_variance)

            # Feature correlation (lower is better for diversity)
            if features.shape[1] > 1:
                correlations = np.corrcoef(features.T)
                # Average absolute correlation (excluding diagonal)
                n_features = correlations.shape[0]
                avg_corr = (np.sum(np.abs(correlations)) - n_features) / (n_features * (n_features - 1))
                quality['avg_correlation'] = avg_corr

            # Information content (entropy-based)
            if features.shape[1] > 1:
                # Normalize features for entropy calculation
                normalized_features = (features - np.min(features, axis=0)) / (np.max(features, axis=0) - np.min(features, axis=0) + 1e-8)
                normalized_features = np.nan_to_num(normalized_features, nan=0.5)

                # Calculate entropy for each feature
                entropies = []
                for i in range(features.shape[1]):
                    feature_vals = normalized_features[:, i]
                    hist, _ = np.histogram(feature_vals, bins=10, range=(0, 1))
                    hist_probs = hist / np.sum(hist)
                    entropy = -np.sum(hist_probs * np.log2(hist_probs + 1e-8))
                    entropies.append(entropy)

                quality['avg_entropy'] = np.mean(entropies)
                quality['information_content'] = np.mean(entropies) / np.log2(10)  # Normalized to max possible

            # Neural network inspired metrics
            # Feature complexity (higher variance in feature distributions)
            complexity = np.mean([np.var(np.diff(np.sort(features[:, i]))) for i in range(features.shape[1])])
            quality['feature_complexity'] = complexity

            # Overall quality score
            quality_score = 0.4 * min(quality.get('avg_feature_variance', 0.1), 1.0)
            quality_score += 0.3 * quality.get('information_content', 0.5)
            if 'avg_correlation' in quality:
                quality_score += 0.3 * (1 - min(quality['avg_correlation'], 1.0))

            quality['quality_score'] = min(quality_score, 1.0)

            return quality

        except Exception as e:
            self.logger.warning(f"Feature quality calculation failed: {e}")
            return {'quality_score': 0.5}


def create_nas_integration(config: Dict[str, Any]) -> NASIntegrationComponent:
    """Create NAS integration component."""
    return NASIntegrationComponent(config)