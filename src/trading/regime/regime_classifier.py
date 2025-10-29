"""
Regime Classifier

ML-based regime classification using ensemble methods
and integration with existing HMM models.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from ..config.regime_config import RegimeConfig, RegimeType

logger = system_logger.getChild('RegimeClassifier')

class RegimeClassifier:
    """
    ML-based regime classifier using ensemble methods.

    Integrates with existing HMM models and provides
    regime classification with confidence scores.
    """

    def __init__(self, config: RegimeConfig):
        self.config = config
        self.logger = logger.getChild('RegimeClassifier')

        # Classification models
        self.models: List[Any] = []
        self.model_weights: List[float] = []

        # Feature engineering
        self.feature_extractors = []

        # Performance tracking
        self.classification_count = 0
        self.accuracy_metrics: Dict[str, float] = {}

    @handles_errors
    async def initialize(self) -> bool:
        """Initialize regime classifier."""
        try:
            tprint_info("🔄 Initializing Regime Classifier...")

            # Load pre-trained models
            await self._load_classification_models()

            # Initialize feature extractors
            await self._initialize_feature_extractors()

            # Load model weights
            await self._load_model_weights()

            tprint_success("✅ Regime Classifier initialized")
            return True

        except Exception as e:
            tprint_error(f"❌ Failed to initialize Regime Classifier: {e}")
            return False

    async def _load_classification_models(self):
        """Load pre-trained classification models."""
        try:
            # Load models using unified model loader
            from src.trading.integration.unified_model_loader import get_unified_model_loader
            
            unified_loader = get_unified_model_loader()
            
            symbol = getattr(self.config, 'symbol', 'ETHUSDT')
            regime_timeframe = getattr(self.config, 'regime_timeframe', '1h')
            
            try:
                # Load regime base models
                regime_base_models = await unified_loader.load_regime_base_models(
                    symbol=symbol,
                    timeframe=regime_timeframe
                )
                
                # Load regime ensemble model
                regime_ensemble_model = await unified_loader.load_regime_ensemble_model(
                    symbol=symbol,
                    timeframe=regime_timeframe
                )
                
                # Add base models to list
                for model_name, model in regime_base_models.items():
                    self.models.append(model)
                    self.logger.info(f"✅ Loaded regime base model: {model_name}")
                
                # Add ensemble model if available
                if regime_ensemble_model:
                    self.models.append(regime_ensemble_model)
                    self.logger.info("✅ Loaded regime ensemble model")
                
                # Also try standardized model manager for backward compatibility
                from src.utils.standardized_model_manager import standardized_model_manager
                
                model_ids = ["regime_classifier_1", "regime_classifier_2", "regime_classifier_hmm"]
                
                for model_id in model_ids:
                    try:
                        model_result = standardized_model_manager.load_model(model_id, "regime_classification")
                        if model_result:
                            model, metadata = model_result
                            if model not in self.models:  # Avoid duplicates
                                self.models.append(model)
                                self.logger.info(f"✅ Loaded regime classification model: {model_id}")
                    except Exception as e:
                        self.logger.debug(f"Could not load regime model {model_id}: {e}")

            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load models via unified loader: {e}")

            # If no models loaded, create fallback classifiers
            if not self.models:
                self.logger.warning("⚠️ No pre-trained models found, using fallback classifiers")
                await self._create_fallback_classifiers()

        except Exception as e:
            self.logger.error(f"❌ Failed to load classification models: {e}")
            await self._create_fallback_classifiers()

    async def _create_fallback_classifiers(self):
        """Create fallback classifiers when pre-trained models are not available."""
        try:
            # Create simple rule-based classifiers as fallback
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC

            # Random Forest classifier
            rf_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

            # SVM classifier
            svm_classifier = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )

            self.models = [rf_classifier, svm_classifier]
            self.logger.info("✅ Created fallback classifiers")

        except ImportError:
            self.logger.warning("⚠️ Scikit-learn not available, using rule-based classifier")
            self.models = ['rule_based']

    async def _initialize_feature_extractors(self):
        """Initialize feature extraction methods."""
        try:
            # Basic feature extractors
            self.feature_extractors = [
                self._extract_price_features,
                self._extract_volatility_features,
                self._extract_volume_features,
                self._extract_technical_features
            ]

            self.logger.info("✅ Feature extractors initialized")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize feature extractors: {e}")

    async def _load_model_weights(self):
        """Load model ensemble weights."""
        try:
            # Default equal weights
            num_models = len(self.models)
            if num_models > 0:
                self.model_weights = [1.0 / num_models] * num_models
            else:
                self.model_weights = []

            self.logger.info(f"✅ Model weights initialized: {self.model_weights}")

        except Exception as e:
            self.logger.error(f"❌ Failed to load model weights: {e}")

    @handles_errors
    @traced(span_name="regime_classification")
    @log_execution_time()
    async def classify(
        self,
        features: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> Dict[RegimeType, float]:
        """
        Classify market regime based on features and market data.

        Args:
            features: Extracted features for classification
            market_data: Market data DataFrame

        Returns:
            Dictionary mapping regime types to probabilities
        """
        try:
            if not self.models:
                tprint_warning("⚠️ No classification models available")
                return self._default_regime_probabilities()

            # Extract additional features
            all_features = await self._extract_all_features(features, market_data)

            # Get predictions from all models
            regime_predictions = []

            for i, model in enumerate(self.models):
                try:
                    prediction = await self._get_model_prediction(model, all_features)
                    if prediction:
                        regime_predictions.append(prediction)
                except Exception as e:
                    self.logger.warning(f"⚠️ Model {i} prediction failed: {e}")

            if not regime_predictions:
                return self._default_regime_probabilities()

            # Ensemble predictions
            ensemble_result = await self._ensemble_predictions(regime_predictions)

            self.classification_count += 1

            return ensemble_result

        except Exception as e:
            self.logger.error(f"❌ Regime classification failed: {e}")
            return self._default_regime_probabilities()

    async def _extract_all_features(
        self,
        base_features: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Extract all features for classification."""
        try:
            all_features = dict(base_features)

            # Run all feature extractors
            for extractor in self.feature_extractors:
                try:
                    features = extractor(market_data)
                    all_features.update(features)
                except Exception as e:
                    self.logger.warning(f"⚠️ Feature extractor failed: {e}")

            return all_features

        except Exception as e:
            self.logger.error(f"❌ Feature extraction failed: {e}")
            return base_features

    def _extract_price_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract price-based features."""
        try:
            if len(market_data) < 20:
                return {}

            close_prices = market_data['close'].values

            # Price momentum features
            returns_1d = (close_prices[-1] - close_prices[-2]) / close_prices[-2]
            returns_5d = (close_prices[-1] - close_prices[-6]) / close_prices[-6]
            returns_20d = (close_prices[-1] - close_prices[-21]) / close_prices[-21]

            # Trend features
            ma_5 = np.mean(close_prices[-5:])
            ma_20 = np.mean(close_prices[-20:])
            ma_ratio = ma_5 / ma_20 if ma_20 > 0 else 1.0

            return {
                'returns_1d': returns_1d,
                'returns_5d': returns_5d,
                'returns_20d': returns_20d,
                'ma_ratio_5_20': ma_ratio
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Price feature extraction failed: {e}")
            return {}

    def _extract_volatility_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract volatility-based features."""
        try:
            if len(market_data) < 20:
                return {}

            close_prices = market_data['close'].values
            returns = np.diff(close_prices) / close_prices[:-1]

            # Volatility features
            vol_5d = np.std(returns[-5:])
            vol_20d = np.std(returns[-20:])
            vol_ratio = vol_5d / vol_20d if vol_20d > 0 else 1.0

            # High-low volatility
            if 'high' in market_data.columns and 'low' in market_data.columns:
                hl_vol = np.mean((market_data['high'].iloc[-20:] - market_data['low'].iloc[-20:]) / market_data['close'].iloc[-20:])
            else:
                hl_vol = vol_20d

            return {
                'volatility_5d': vol_5d,
                'volatility_20d': vol_20d,
                'volatility_ratio': vol_ratio,
                'hl_volatility': hl_vol
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Volatility feature extraction failed: {e}")
            return {}

    def _extract_volume_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract volume-based features."""
        try:
            if len(market_data) < 20 or 'volume' not in market_data.columns:
                return {}

            volumes = market_data['volume'].values

            # Volume features
            vol_ma_5 = np.mean(volumes[-5:])
            vol_ma_20 = np.mean(volumes[-20:])
            vol_ratio = vol_ma_5 / vol_ma_20 if vol_ma_20 > 0 else 1.0

            # Volume volatility
            vol_std = np.std(volumes[-20:])
            vol_cv = vol_std / vol_ma_20 if vol_ma_20 > 0 else 0.0

            return {
                'volume_ratio_5_20': vol_ratio,
                'volume_cv': vol_cv,
                'volume_trend': vol_ratio - 1.0
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Volume feature extraction failed: {e}")
            return {}

    def _extract_technical_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract technical indicator features."""
        try:
            if len(market_data) < 50:
                return {}

            close_prices = market_data['close'].values

            # RSI
            delta = np.diff(close_prices)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)

            avg_gain = np.mean(gain[-14:])
            avg_loss = np.mean(loss[-14:])

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100.0

            # MACD approximation
            ema_12 = np.mean(close_prices[-12:])
            ema_26 = np.mean(close_prices[-26:])
            macd = ema_12 - ema_26

            return {
                'rsi': rsi / 100.0,  # Normalize to [0, 1]
                'macd': macd / close_prices[-1],  # Normalize by price
                'price_position': (close_prices[-1] - np.min(close_prices[-20:])) / (np.max(close_prices[-20:]) - np.min(close_prices[-20:]))
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Technical feature extraction failed: {e}")
            return {}

    async def _get_model_prediction(self, model, features: Dict[str, float]) -> Optional[Dict[RegimeType, float]]:
        """Get prediction from a single model."""
        try:
            if model == 'rule_based':
                return self._rule_based_prediction(features)

            # Prepare feature vector
            feature_vector = self._features_to_vector(features)

            if hasattr(model, 'predict_proba'):
                # Probabilistic prediction
                probabilities = model.predict_proba([feature_vector])[0]
                return self._probabilities_to_regime_dict(probabilities)
            elif hasattr(model, 'predict'):
                # Deterministic prediction
                prediction = model.predict([feature_vector])[0]
                return self._prediction_to_regime_dict(prediction)
            else:
                return None

        except Exception as e:
            self.logger.warning(f"⚠️ Model prediction failed: {e}")
            return None

    def _rule_based_prediction(self, features: Dict[str, float]) -> Dict[RegimeType, float]:
        """Rule-based regime prediction as fallback."""
        try:
            # Simple rule-based classification
            volatility = features.get('volatility_20d', 0.02)
            returns_5d = features.get('returns_5d', 0.0)
            ma_ratio = features.get('ma_ratio_5_20', 1.0)
            volume_ratio = features.get('volume_ratio_5_20', 1.0)

            # Initialize probabilities
            probabilities = {regime: 0.1 for regime in RegimeType}

            # High volatility regime
            if volatility > 0.05:
                probabilities[RegimeType.HIGH_VOLATILITY] += 0.3
                probabilities[RegimeType.BREAKOUT] += 0.2
            else:
                probabilities[RegimeType.LOW_VOLATILITY] += 0.3

            # Trending regimes
            if abs(returns_5d) > 0.02:
                if returns_5d > 0:
                    probabilities[RegimeType.TRENDING_UP] += 0.3
                    probabilities[RegimeType.MOMENTUM] += 0.2
                else:
                    probabilities[RegimeType.TRENDING_DOWN] += 0.3
                    probabilities[RegimeType.REVERSAL] += 0.1
            else:
                probabilities[RegimeType.SIDEWAYS] += 0.3
                probabilities[RegimeType.MEAN_REVERSION] += 0.2

            # Moving average regime
            if ma_ratio > 1.02:
                probabilities[RegimeType.TRENDING_UP] += 0.1
            elif ma_ratio < 0.98:
                probabilities[RegimeType.TRENDING_DOWN] += 0.1

            # Normalize probabilities
            total_prob = sum(probabilities.values())
            if total_prob > 0:
                for regime in probabilities:
                    probabilities[regime] /= total_prob

            return probabilities

        except Exception as e:
            self.logger.error(f"❌ Rule-based prediction failed: {e}")
            return self._default_regime_probabilities()

    def _features_to_vector(self, features: Dict[str, float]) -> List[float]:
        """Convert feature dictionary to vector."""
        # Define standard feature order
        feature_names = [
            'returns_1d', 'returns_5d', 'returns_20d', 'ma_ratio_5_20',
            'volatility_5d', 'volatility_20d', 'volatility_ratio', 'hl_volatility',
            'volume_ratio_5_20', 'volume_cv', 'volume_trend',
            'rsi', 'macd', 'price_position'
        ]

        return [features.get(name, 0.0) for name in feature_names]

    def _probabilities_to_regime_dict(self, probabilities: np.ndarray) -> Dict[RegimeType, float]:
        """Convert model probabilities to regime dictionary."""
        # This assumes the model was trained with regime types in a specific order
        # In practice, you would save this mapping with the trained model
        regime_order = list(RegimeType)

        result = {}
        for i, regime in enumerate(regime_order):
            if i < len(probabilities):
                result[regime] = float(probabilities[i])
            else:
                result[regime] = 0.0

        return result

    def _prediction_to_regime_dict(self, prediction) -> Dict[RegimeType, float]:
        """Convert single prediction to regime probability dictionary."""
        result = {regime: 0.1 for regime in RegimeType}

        # Try to map prediction to regime
        if isinstance(prediction, (int, np.integer)):
            regime_list = list(RegimeType)
            if 0 <= prediction < len(regime_list):
                result[regime_list[prediction]] = 0.9

        return result

    async def _ensemble_predictions(self, predictions: List[Dict[RegimeType, float]]) -> Dict[RegimeType, float]:
        """Ensemble multiple model predictions."""
        try:
            if not predictions:
                return self._default_regime_probabilities()

            # Weighted average of predictions
            ensemble_result = {regime: 0.0 for regime in RegimeType}

            for i, prediction in enumerate(predictions):
                weight = self.model_weights[i] if i < len(self.model_weights) else 1.0 / len(predictions)

                for regime, prob in prediction.items():
                    ensemble_result[regime] += prob * weight

            # Normalize
            total_prob = sum(ensemble_result.values())
            if total_prob > 0:
                for regime in ensemble_result:
                    ensemble_result[regime] /= total_prob

            return ensemble_result

        except Exception as e:
            self.logger.error(f"❌ Ensemble prediction failed: {e}")
            return self._default_regime_probabilities()

    def _default_regime_probabilities(self) -> Dict[RegimeType, float]:
        """Return default regime probabilities."""
        return {
            RegimeType.SIDEWAYS: 0.4,
            RegimeType.LOW_VOLATILITY: 0.2,
            RegimeType.TRENDING_UP: 0.1,
            RegimeType.TRENDING_DOWN: 0.1,
            RegimeType.HIGH_VOLATILITY: 0.1,
            RegimeType.BREAKOUT: 0.05,
            RegimeType.REVERSAL: 0.05
        }

    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification performance statistics."""
        return {
            'total_classifications': self.classification_count,
            'num_models': len(self.models),
            'model_weights': self.model_weights,
            'accuracy_metrics': self.accuracy_metrics
        }

    async def stop(self):
        """Stop regime classifier."""
        try:
            self.logger.info("🛑 Stopping Regime Classifier...")

            # Clear models and data
            self.models.clear()
            self.model_weights.clear()

            self.logger.info("✅ Regime Classifier stopped")

        except Exception as e:
            self.logger.error(f"❌ Error stopping Regime Classifier: {e}")
