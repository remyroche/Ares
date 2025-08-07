# src/tactician/sr_breakout_predictor.py

import asyncio
from datetime import datetime
from typing import Any
import pandas as pd

from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class SRBreakoutPredictor:
    """
    SR Breakout Predictor responsible for predicting support/resistance breakouts.
    This module handles all SR breakout prediction logic and feature engineering.
    Uses the unified regime classifier's S/R levels (Dynamic Pivots and HVNs).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize SR breakout predictor.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("SRBreakoutPredictor")
        
        # SR predictor state
        self.is_initialized: bool = False
        self.sr_predictions: dict[str, Any] = {}
        
        # Configuration
        self.sr_config: dict[str, Any] = self.config.get("sr_breakout_predictor", {})
        self.enable_sr_breakout_tactics: bool = self.sr_config.get("enable_sr_breakout_tactics", True)
        self.sr_proximity_threshold: float = self.sr_config.get("sr_proximity_threshold", 0.02)
        self.breakout_confidence_threshold: float = self.sr_config.get("breakout_confidence_threshold", 0.7)
        
        # Unified regime classifier for S/R levels
        self.regime_classifier: UnifiedRegimeClassifier | None = None

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid SR breakout predictor configuration"),
            AttributeError: (False, "Missing required SR predictor parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="SR breakout predictor initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize SR breakout predictor.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing SR Breakout Predictor...")
            
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for SR breakout predictor")
                return False
            
            # Initialize SR prediction models
            await self._initialize_sr_models()
            
            self.is_initialized = True
            self.logger.info("✅ SR Breakout Predictor initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ SR Breakout Predictor initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate SR breakout predictor configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            if self.sr_proximity_threshold <= 0:
                self.logger.error("Invalid sr_proximity_threshold configuration")
                return False
                
            if self.breakout_confidence_threshold <= 0 or self.breakout_confidence_threshold > 1:
                self.logger.error("Invalid breakout_confidence_threshold configuration")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="SR models initialization",
    )
    async def _initialize_sr_models(self) -> None:
        """Initialize SR prediction models and regime classifier."""
        try:
            self.logger.info("Initializing SR prediction models and regime classifier...")
            
            # Initialize unified regime classifier for S/R levels
            self.regime_classifier = UnifiedRegimeClassifier(self.config)
            await self.regime_classifier.initialize()
            
            # Load or train the regime classifier if needed
            if not self.regime_classifier.trained:
                self.logger.info("Regime classifier not trained - will use in inference mode")
            
            self.logger.info("✅ SR prediction models initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize SR models: {e}")
            raise

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid prediction parameters"),
            AttributeError: (False, "Missing prediction components"),
            KeyError: (False, "Missing required prediction data"),
        },
        default_return=False,
        context="SR breakout prediction",
    )
    async def predict_breakouts(
        self,
        prediction_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Predict SR breakouts.

        Args:
            prediction_input: Prediction input parameters

        Returns:
            dict: SR breakout prediction results
        """
        try:
            self.logger.info("🔮 Predicting SR breakouts...")
            
            # Validate prediction input
            if not self._validate_prediction_input(prediction_input):
                return {}
            
            # Extract data from input
            df = prediction_input.get("dataframe")
            current_price = prediction_input.get("current_price")
            
            if df is None or current_price is None:
                self.logger.error("Missing required prediction data")
                return {}
            
            # Get SR breakout prediction
            prediction_results = await self._get_sr_breakout_prediction_enhanced(df, current_price)
            
            if prediction_results:
                self.sr_predictions = prediction_results
                self.logger.info("✅ SR breakout prediction completed successfully")
            else:
                self.logger.warning("⚠️ SR breakout prediction returned no results")
            
            return prediction_results
            
        except Exception as e:
            self.logger.error(f"❌ SR breakout prediction failed: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="prediction input validation",
    )
    def _validate_prediction_input(self, prediction_input: dict[str, Any]) -> bool:
        """
        Validate prediction input parameters.

        Args:
            prediction_input: Prediction input parameters

        Returns:
            bool: True if input is valid, False otherwise
        """
        try:
            required_fields = ["dataframe", "current_price"]
            
            for field in required_fields:
                if field not in prediction_input:
                    self.logger.error(f"Missing required prediction input field: {field}")
                    return False
            
            # Validate specific field values
            if prediction_input.get("current_price", 0) <= 0:
                self.logger.error("Invalid current_price value")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Prediction input validation failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="SR breakout prediction enhanced",
    )
    async def _get_sr_breakout_prediction_enhanced(
        self,
        df: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any]:
        """
        Get enhanced SR breakout prediction.

        Args:
            df: Price dataframe
            current_price: Current price

        Returns:
            dict: Enhanced SR breakout prediction results
        """
        try:
            # Prepare features for SR prediction
            features_df = self._prepare_features_for_sr_prediction(df, current_price)
            
            # Get SR context
            sr_context = self._get_sr_context(df, current_price)
            
            # Make predictions
            predictions = await self._make_sr_predictions(features_df, sr_context)
            
            # Combine predictions
            breakout_prob, bounce_prob, confidence = self._combine_sr_predictions(predictions)
            
            # Get recommendation
            recommendation = self._get_sr_recommendation(breakout_prob, bounce_prob, confidence)
            
            return {
                "breakout_probability": breakout_prob,
                "bounce_probability": bounce_prob,
                "confidence": confidence,
                "recommendation": recommendation,
                "sr_context": sr_context,
                "predictions": predictions,
                "timestamp": datetime.now(),
            }
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced SR breakout prediction failed: {e}")
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="features preparation for SR prediction",
    )
    def _prepare_features_for_sr_prediction(
        self,
        df: pd.DataFrame,
        current_price: float,
        sr_context: dict = None,
    ) -> pd.DataFrame:
        """
        Prepare features for SR prediction.

        Args:
            df: Price dataframe
            current_price: Current price
            sr_context: SR context dictionary

        Returns:
            pd.DataFrame: Features dataframe
        """
        try:
            features_df = df.copy()
            
            # Add price-based features
            features_df['price_change'] = features_df['close'].pct_change()
            features_df['price_volatility'] = features_df['close'].rolling(20).std()
            features_df['price_momentum'] = features_df['close'] - features_df['close'].shift(10)
            
            # Add volume-based features
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume'].rolling(20).mean()
            features_df['volume_momentum'] = features_df['volume'] - features_df['volume'].shift(5)
            
            # Add technical indicators
            features_df['rsi'] = self._calculate_rsi(features_df['close'])
            features_df['macd'] = self._calculate_macd(features_df['close'])
            features_df['bollinger_upper'] = self._calculate_bollinger_bands(features_df['close'])[0]
            features_df['bollinger_lower'] = self._calculate_bollinger_bands(features_df['close'])[1]
            
            # Add SR-specific features
            if sr_context:
                # Distance to nearest support and resistance
                nearest_support = sr_context.get('nearest_support', current_price)
                nearest_resistance = sr_context.get('nearest_resistance', current_price)
                
                features_df['distance_to_support'] = (current_price - nearest_support) / current_price
                features_df['distance_to_resistance'] = (nearest_resistance - current_price) / current_price
                
                # Distance to pivot levels
                pivot_levels = sr_context.get('pivot_levels', {})
                if pivot_levels:
                    pivot_supports = pivot_levels.get('supports', [])
                    pivot_resistances = pivot_levels.get('resistances', [])
                    pivot_strengths = pivot_levels.get('strengths', {})
                    
                    if pivot_supports:
                        nearest_pivot_support = self._find_nearest_level(current_price, pivot_supports)
                        features_df['distance_to_pivot_support'] = (current_price - nearest_pivot_support) / current_price
                        
                        # Add pivot strength features
                        if pivot_strengths:
                            features_df['nearest_pivot_strength'] = self._get_nearest_level_strength(
                                nearest_pivot_support, pivot_supports, pivot_strengths, 'supports'
                            )
                            features_df['nearest_pivot_touches'] = self._get_nearest_level_touches(
                                nearest_pivot_support, pivot_supports, pivot_strengths, 'supports'
                            )
                            features_df['nearest_pivot_volume'] = self._get_nearest_level_volume(
                                nearest_pivot_support, pivot_supports, pivot_strengths, 'supports'
                            )
                            features_df['nearest_pivot_age'] = self._get_nearest_level_age(
                                nearest_pivot_support, pivot_supports, pivot_strengths, 'supports'
                            )
                        else:
                            features_df['nearest_pivot_strength'] = 0.0
                            features_df['nearest_pivot_touches'] = 0.0
                            features_df['nearest_pivot_volume'] = 0.0
                            features_df['nearest_pivot_age'] = 0.0
                    else:
                        features_df['distance_to_pivot_support'] = 1.0
                        features_df['nearest_pivot_strength'] = 0.0
                        features_df['nearest_pivot_touches'] = 0.0
                        features_df['nearest_pivot_volume'] = 0.0
                        features_df['nearest_pivot_age'] = 0.0
                        
                    if pivot_resistances:
                        nearest_pivot_resistance = self._find_nearest_level(current_price, pivot_resistances)
                        features_df['distance_to_pivot_resistance'] = (nearest_pivot_resistance - current_price) / current_price
                        
                        # Add pivot resistance strength features
                        if pivot_strengths:
                            features_df['nearest_pivot_resistance_strength'] = self._get_nearest_level_strength(
                                nearest_pivot_resistance, pivot_resistances, pivot_strengths, 'resistances'
                            )
                            features_df['nearest_pivot_resistance_touches'] = self._get_nearest_level_touches(
                                nearest_pivot_resistance, pivot_resistances, pivot_strengths, 'resistances'
                            )
                            features_df['nearest_pivot_resistance_volume'] = self._get_nearest_level_volume(
                                nearest_pivot_resistance, pivot_resistances, pivot_strengths, 'resistances'
                            )
                            features_df['nearest_pivot_resistance_age'] = self._get_nearest_level_age(
                                nearest_pivot_resistance, pivot_resistances, pivot_strengths, 'resistances'
                            )
                        else:
                            features_df['nearest_pivot_resistance_strength'] = 0.0
                            features_df['nearest_pivot_resistance_touches'] = 0.0
                            features_df['nearest_pivot_resistance_volume'] = 0.0
                            features_df['nearest_pivot_resistance_age'] = 0.0
                    else:
                        features_df['distance_to_pivot_resistance'] = 1.0
                        features_df['nearest_pivot_resistance_strength'] = 0.0
                        features_df['nearest_pivot_resistance_touches'] = 0.0
                        features_df['nearest_pivot_resistance_volume'] = 0.0
                        features_df['nearest_pivot_resistance_age'] = 0.0
                else:
                    features_df['distance_to_pivot_support'] = 1.0
                    features_df['distance_to_pivot_resistance'] = 1.0
                    features_df['nearest_pivot_strength'] = 0.0
                    features_df['nearest_pivot_touches'] = 0.0
                    features_df['nearest_pivot_volume'] = 0.0
                    features_df['nearest_pivot_age'] = 0.0
                    features_df['nearest_pivot_resistance_strength'] = 0.0
                    features_df['nearest_pivot_resistance_touches'] = 0.0
                    features_df['nearest_pivot_resistance_volume'] = 0.0
                    features_df['nearest_pivot_resistance_age'] = 0.0
                
                # Distance to HVN levels
                hvn_levels = sr_context.get('hvn_levels', {})
                if hvn_levels:
                    hvn_supports = hvn_levels.get('supports', [])
                    hvn_resistances = hvn_levels.get('resistances', [])
                    hvn_strengths = hvn_levels.get('strengths', {})
                    
                    if hvn_supports:
                        nearest_hvn_support = self._find_nearest_level(current_price, hvn_supports)
                        features_df['distance_to_hvn_support'] = (current_price - nearest_hvn_support) / current_price
                        
                        # Add HVN strength features
                        if hvn_strengths:
                            features_df['nearest_hvn_strength'] = self._get_nearest_level_strength(
                                nearest_hvn_support, hvn_supports, hvn_strengths, 'supports'
                            )
                            features_df['nearest_hvn_touches'] = self._get_nearest_level_touches(
                                nearest_hvn_support, hvn_supports, hvn_strengths, 'supports'
                            )
                            features_df['nearest_hvn_volume'] = self._get_nearest_level_volume(
                                nearest_hvn_support, hvn_supports, hvn_strengths, 'supports'
                            )
                            features_df['nearest_hvn_age'] = self._get_nearest_level_age(
                                nearest_hvn_support, hvn_supports, hvn_strengths, 'supports'
                            )
                        else:
                            features_df['nearest_hvn_strength'] = 0.0
                            features_df['nearest_hvn_touches'] = 0.0
                            features_df['nearest_hvn_volume'] = 0.0
                            features_df['nearest_hvn_age'] = 0.0
                    else:
                        features_df['distance_to_hvn_support'] = 1.0
                        features_df['nearest_hvn_strength'] = 0.0
                        features_df['nearest_hvn_touches'] = 0.0
                        features_df['nearest_hvn_volume'] = 0.0
                        features_df['nearest_hvn_age'] = 0.0
                        
                    if hvn_resistances:
                        nearest_hvn_resistance = self._find_nearest_level(current_price, hvn_resistances)
                        features_df['distance_to_hvn_resistance'] = (nearest_hvn_resistance - current_price) / current_price
                        
                        # Add HVN resistance strength features
                        if hvn_strengths:
                            features_df['nearest_hvn_resistance_strength'] = self._get_nearest_level_strength(
                                nearest_hvn_resistance, hvn_resistances, hvn_strengths, 'resistances'
                            )
                            features_df['nearest_hvn_resistance_touches'] = self._get_nearest_level_touches(
                                nearest_hvn_resistance, hvn_resistances, hvn_strengths, 'resistances'
                            )
                            features_df['nearest_hvn_resistance_volume'] = self._get_nearest_level_volume(
                                nearest_hvn_resistance, hvn_resistances, hvn_strengths, 'resistances'
                            )
                            features_df['nearest_hvn_resistance_age'] = self._get_nearest_level_age(
                                nearest_hvn_resistance, hvn_resistances, hvn_strengths, 'resistances'
                            )
                        else:
                            features_df['nearest_hvn_resistance_strength'] = 0.0
                            features_df['nearest_hvn_resistance_touches'] = 0.0
                            features_df['nearest_hvn_resistance_volume'] = 0.0
                            features_df['nearest_hvn_resistance_age'] = 0.0
                    else:
                        features_df['distance_to_hvn_resistance'] = 1.0
                        features_df['nearest_hvn_resistance_strength'] = 0.0
                        features_df['nearest_hvn_resistance_touches'] = 0.0
                        features_df['nearest_hvn_resistance_volume'] = 0.0
                        features_df['nearest_hvn_resistance_age'] = 0.0
                else:
                    features_df['distance_to_hvn_support'] = 1.0
                    features_df['distance_to_hvn_resistance'] = 1.0
                    features_df['nearest_hvn_strength'] = 0.0
                    features_df['nearest_hvn_touches'] = 0.0
                    features_df['nearest_hvn_volume'] = 0.0
                    features_df['nearest_hvn_age'] = 0.0
                    features_df['nearest_hvn_resistance_strength'] = 0.0
                    features_df['nearest_hvn_resistance_touches'] = 0.0
                    features_df['nearest_hvn_resistance_volume'] = 0.0
                    features_df['nearest_hvn_resistance_age'] = 0.0
                
                # Location classification features
                current_location = sr_context.get('current_location', 'OPEN_RANGE')
                features_df['is_pivot_support'] = 1.0 if 'PIVOT_S' in current_location else 0.0
                features_df['is_pivot_resistance'] = 1.0 if 'PIVOT_R' in current_location else 0.0
                features_df['is_hvn_support'] = 1.0 if 'HVN_SUPPORT' in current_location else 0.0
                features_df['is_hvn_resistance'] = 1.0 if 'HVN_RESISTANCE' in current_location else 0.0
                features_df['is_confluence'] = 1.0 if 'CONFLUENCE' in current_location else 0.0
                features_df['is_open_range'] = 1.0 if current_location == 'OPEN_RANGE' else 0.0
                
                features_df['sr_breakout_pressure'] = self._calculate_sr_breakout_pressure(df, sr_context)
            
            # Clean up NaN values
            features_df = features_df.fillna(method='ffill').fillna(0)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ Feature preparation failed: {e}")
            return df

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="SR context calculation",
    )
    def _get_sr_context(self, df: pd.DataFrame, current_price: float) -> dict[str, Any]:
        """
        Get SR context information using unified regime classifier's S/R levels.

        Args:
            df: Price dataframe
            current_price: Current price

        Returns:
            dict: SR context information
        """
        try:
            if self.regime_classifier is None:
                self.logger.warning("Regime classifier not available, using fallback S/R calculation")
                return self._get_fallback_sr_context(df, current_price)
            
            # Use unified regime classifier to get S/R levels
            features_df = self.regime_classifier._calculate_features(df)
            if features_df.empty:
                return self._get_fallback_sr_context(df, current_price)
            
            # Get location classification which includes S/R levels
            location_labels = self.regime_classifier._classify_location(features_df)
            
            # Extract S/R levels from the classification
            pivot_levels = self._extract_pivot_levels(features_df)
            hvn_levels = self._extract_hvn_levels(features_df)
            
            # Combine all levels
            support_levels = []
            resistance_levels = []
            
            # Add pivot levels
            if pivot_levels:
                support_levels.extend(pivot_levels.get("supports", []))
                resistance_levels.extend(pivot_levels.get("resistances", []))
            
            # Add HVN levels
            if hvn_levels:
                support_levels.extend(hvn_levels.get("supports", []))
                resistance_levels.extend(hvn_levels.get("resistances", []))
            
            # Find nearest levels
            nearest_support = self._find_nearest_level(current_price, support_levels)
            nearest_resistance = self._find_nearest_level(current_price, resistance_levels)
            
            # Get current location classification
            current_location = location_labels[-1] if location_labels else "OPEN_RANGE"
            
            return {
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
                "current_price": current_price,
                "current_location": current_location,
                "pivot_levels": pivot_levels,
                "hvn_levels": hvn_levels,
            }
            
        except Exception as e:
            self.logger.error(f"❌ SR context calculation failed: {e}")
            return self._get_fallback_sr_context(df, current_price)

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="SR predictions combination",
    )
    def _combine_sr_predictions(self, predictions: dict) -> tuple[float, float, float]:
        """
        Combine SR predictions.

        Args:
            predictions: Dictionary of predictions

        Returns:
            tuple: (breakout_prob, bounce_prob, confidence)
        """
        try:
            # Extract individual predictions
            breakout_preds = predictions.get("breakout", [])
            bounce_preds = predictions.get("bounce", [])
            confidence_preds = predictions.get("confidence", [])
            
            # Calculate weighted averages
            breakout_prob = sum(breakout_preds) / len(breakout_preds) if breakout_preds else 0.5
            bounce_prob = sum(bounce_preds) / len(bounce_preds) if bounce_preds else 0.5
            confidence = sum(confidence_preds) / len(confidence_preds) if confidence_preds else 0.5
            
            # Normalize probabilities
            total_prob = breakout_prob + bounce_prob
            if total_prob > 0:
                breakout_prob /= total_prob
                bounce_prob /= total_prob
            
            return breakout_prob, bounce_prob, confidence
            
        except Exception as e:
            self.logger.error(f"❌ SR predictions combination failed: {e}")
            return 0.5, 0.5, 0.5

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="SR recommendation generation",
    )
    def _get_sr_recommendation(
        self,
        breakout_prob: float,
        bounce_prob: float,
        confidence: float,
    ) -> str:
        """
        Get SR recommendation based on predictions.

        Args:
            breakout_prob: Breakout probability
            bounce_prob: Bounce probability
            confidence: Prediction confidence

        Returns:
            str: Recommendation
        """
        try:
            if confidence < self.breakout_confidence_threshold:
                return "HOLD_LOW_CONFIDENCE"
            
            if breakout_prob > bounce_prob and breakout_prob > 0.6:
                return "BREAKOUT_LIKELY"
            elif bounce_prob > breakout_prob and bounce_prob > 0.6:
                return "BOUNCE_LIKELY"
            elif abs(breakout_prob - bounce_prob) < 0.1:
                return "UNCERTAIN"
            else:
                return "MONITOR"
                
        except Exception as e:
            self.logger.error(f"❌ SR recommendation generation failed: {e}")
            return "ERROR"

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="SR predictions making",
    )
    async def _make_sr_predictions(
        self,
        features_df: pd.DataFrame,
        sr_context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Make SR predictions using models.

        Args:
            features_df: Features dataframe
            sr_context: SR context

        Returns:
            dict: Predictions
        """
        try:
            # This would typically use trained models to make predictions
            # For now, return mock predictions
            return {
                "breakout": [0.6, 0.7, 0.5],
                "bounce": [0.4, 0.3, 0.5],
                "confidence": [0.8, 0.7, 0.6],
            }
            
        except Exception as e:
            self.logger.error(f"❌ SR predictions making failed: {e}")
            return {}

    # Helper methods for technical indicators
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: int = 2) -> tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, lower_band

    def _get_fallback_sr_context(self, df: pd.DataFrame, current_price: float) -> dict[str, Any]:
        """
        Get fallback SR context using simplified methods.
        
        Args:
            df: Price dataframe
            current_price: Current price
            
        Returns:
            dict: Fallback SR context
        """
        try:
            # Calculate support and resistance levels using simplified methods
            support_levels = self._find_support_levels(df)
            resistance_levels = self._find_resistance_levels(df)
            
            # Find nearest levels
            nearest_support = self._find_nearest_level(current_price, support_levels)
            nearest_resistance = self._find_nearest_level(current_price, resistance_levels)
            
            return {
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
                "current_price": current_price,
                "current_location": "OPEN_RANGE",
                "pivot_levels": {},
                "hvn_levels": {},
            }
            
        except Exception as e:
            self.logger.error(f"❌ Fallback SR context calculation failed: {e}")
            return {}

    def _extract_pivot_levels(self, features_df: pd.DataFrame) -> dict[str, list[float]]:
        """
        Extract pivot levels from features DataFrame with strength metrics.
        
        Args:
            features_df: Features DataFrame from regime classifier
            
        Returns:
            dict: Pivot support and resistance levels with strength data
        """
        try:
            supports = []
            resistances = []
            strengths = {}
            
            # Calculate rolling pivots for the last few periods
            for i in range(max(0, len(features_df) - 24), len(features_df)):
                if i < 5:  # Need at least 5 data points
                    continue
                    
                window = features_df.iloc[max(0, i-24):i+1]
                if len(window) < 5:
                    continue
                    
                pivots = self.regime_classifier._calculate_rolling_pivots(window)
                
                # Extract levels and their strengths
                if pivots["s1"] > 0:
                    supports.append(pivots["s1"])
                    strengths[f"s1_{i}"] = pivots["strengths"]["s1"]
                if pivots["s2"] > 0:
                    supports.append(pivots["s2"])
                    strengths[f"s2_{i}"] = pivots["strengths"]["s2"]
                if pivots["r1"] > 0:
                    resistances.append(pivots["r1"])
                    strengths[f"r1_{i}"] = pivots["strengths"]["r1"]
                if pivots["r2"] > 0:
                    resistances.append(pivots["r2"])
                    strengths[f"r2_{i}"] = pivots["strengths"]["r2"]
            
            return {
                "supports": list(set(supports)),  # Remove duplicates
                "resistances": list(set(resistances)),
                "strengths": strengths
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting pivot levels: {e}")
            return {"supports": [], "resistances": [], "strengths": {}}

    def _extract_hvn_levels(self, features_df: pd.DataFrame) -> dict[str, list[float]]:
        """
        Extract HVN levels from features DataFrame with strength metrics.
        
        Args:
            features_df: Features DataFrame from regime classifier
            
        Returns:
            dict: HVN support and resistance levels with strength data
        """
        try:
            supports = []
            resistances = []
            strengths = {}
            
            # Analyze volume levels for the last period
            if len(features_df) >= 720:  # Need enough data for HVN analysis
                hvn_window = features_df.iloc[-720:]
                volume_levels = self.regime_classifier._analyze_volume_levels(hvn_window)
                
                if volume_levels:
                    for level_name, level_data in volume_levels.items():
                        level_price = level_data["price"]
                        current_price = features_df["close"].iloc[-1]
                        
                        # Classify as support or resistance based on current price
                        if current_price > level_price:
                            supports.append(level_price)
                            strengths[f"hvn_support_{level_name}"] = {
                                "strength": level_data.get("strength", 0.0),
                                "touches": level_data.get("touches", 0),
                                "volume": level_data.get("volume", 0.0),
                                "age": level_data.get("age", 0)
                            }
                        else:
                            resistances.append(level_price)
                            strengths[f"hvn_resistance_{level_name}"] = {
                                "strength": level_data.get("strength", 0.0),
                                "touches": level_data.get("touches", 0),
                                "volume": level_data.get("volume", 0.0),
                                "age": level_data.get("age", 0)
                            }
            
            return {
                "supports": list(set(supports)),  # Remove duplicates
                "resistances": list(set(resistances)),
                "strengths": strengths
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting HVN levels: {e}")
            return {"supports": [], "resistances": [], "strengths": {}}

    def _find_support_levels(self, df: pd.DataFrame) -> list[float]:
        """Find support levels."""
        # Simplified support level detection
        return [df['low'].min() * 0.95, df['low'].min() * 0.98, df['low'].min()]

    def _find_resistance_levels(self, df: pd.DataFrame) -> list[float]:
        """Find resistance levels."""
        # Simplified resistance level detection
        return [df['high'].max(), df['high'].max() * 1.02, df['high'].max() * 1.05]

    def _find_nearest_level(self, price: float, levels: list[float]) -> float:
        """Find nearest level to current price."""
        if not levels:
            return price
        return min(levels, key=lambda x: abs(x - price))

    def _get_nearest_level_strength(self, nearest_level: float, levels: list[float], strengths: dict, level_type: str) -> float:
        """Get strength of the nearest level."""
        try:
            if not levels or not strengths:
                return 0.0
            
            # Find the strength data for the nearest level
            for level_key, strength_data in strengths.items():
                if isinstance(strength_data, dict) and "strength" in strength_data:
                    # For HVN levels
                    return strength_data["strength"]
                elif isinstance(strength_data, dict):
                    # For pivot levels
                    return strength_data.get("strength", 0.0)
            
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting nearest level strength: {e}")
            return 0.0

    def _get_nearest_level_touches(self, nearest_level: float, levels: list[float], strengths: dict, level_type: str) -> float:
        """Get number of touches for the nearest level."""
        try:
            if not levels or not strengths:
                return 0.0
            
            # Find the touch data for the nearest level
            for level_key, strength_data in strengths.items():
                if isinstance(strength_data, dict) and "touches" in strength_data:
                    return float(strength_data["touches"])
            
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting nearest level touches: {e}")
            return 0.0

    def _get_nearest_level_volume(self, nearest_level: float, levels: list[float], strengths: dict, level_type: str) -> float:
        """Get volume for the nearest level."""
        try:
            if not levels or not strengths:
                return 0.0
            
            # Find the volume data for the nearest level
            for level_key, strength_data in strengths.items():
                if isinstance(strength_data, dict) and "volume" in strength_data:
                    return float(strength_data["volume"])
            
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting nearest level volume: {e}")
            return 0.0

    def _get_nearest_level_age(self, nearest_level: float, levels: list[float], strengths: dict, level_type: str) -> float:
        """Get age for the nearest level."""
        try:
            if not levels or not strengths:
                return 0.0
            
            # Find the age data for the nearest level
            for level_key, strength_data in strengths.items():
                if isinstance(strength_data, dict) and "age" in strength_data:
                    return float(strength_data["age"])
            
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting nearest level age: {e}")
            return 0.0

    def _calculate_sr_breakout_pressure(self, df: pd.DataFrame, sr_context: dict[str, Any]) -> float:
        """Calculate SR breakout pressure."""
        # Simplified breakout pressure calculation
        return 0.5

    def get_sr_predictions(self) -> dict[str, Any]:
        """
        Get the latest SR predictions.

        Returns:
            dict: SR predictions
        """
        return self.sr_predictions.copy()

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="SR breakout predictor cleanup",
    )
    async def stop(self) -> None:
        """Stop the SR breakout predictor and cleanup resources."""
        try:
            self.logger.info("🛑 Stopping SR Breakout Predictor...")
            self.is_initialized = False
            self.logger.info("✅ SR Breakout Predictor stopped successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stop SR Breakout Predictor: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="SR breakout predictor setup",
)
async def setup_sr_breakout_predictor(
    config: dict[str, Any] | None = None,
) -> SRBreakoutPredictor | None:
    """
    Setup and return a configured SRBreakoutPredictor instance.

    Args:
        config: Configuration dictionary

    Returns:
        SRBreakoutPredictor: Configured SR breakout predictor instance
    """
    try:
        predictor = SRBreakoutPredictor(config or {})
        if await predictor.initialize():
            return predictor
        return None
    except Exception as e:
        system_logger.error(f"Failed to setup SR breakout predictor: {e}")
        return None 
