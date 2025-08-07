# src/training/steps/vectorized_advanced_feature_engineering.py

"""
Vectorized Advanced Feature Engineering for enhanced financial performance.
Implements sophisticated market microstructure features, regime detection,
and adaptive indicators for improved prediction accuracy with vectorized operations.
"""

import numpy as np
import pandas as pd
import pywt
from typing import Any, Dict, List, Optional, Tuple
from datetime import timedelta

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class VectorizedAdvancedFeatureEngineering:
    """
    Comprehensive vectorized advanced feature engineering system.
    Integrates all feature engineering components including wavelet transforms.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedAdvancedFeatureEngineering")

        # Configuration
        self.feature_config = config.get("vectorized_advanced_features", {})
        self.enable_volatility_modeling = self.feature_config.get("enable_volatility_modeling", True)
        self.enable_correlation_analysis = self.feature_config.get("enable_correlation_analysis", True)
        self.enable_momentum_analysis = self.feature_config.get("enable_momentum_analysis", True)
        self.enable_liquidity_analysis = self.feature_config.get("enable_liquidity_analysis", True)
        self.enable_candlestick_patterns = self.feature_config.get("enable_candlestick_patterns", True)
        self.enable_sr_distance = self.feature_config.get("enable_sr_distance", True)
        self.enable_wavelet_transforms = self.feature_config.get("enable_wavelet_transforms", True)
        self.enable_multi_timeframe = self.feature_config.get("enable_multi_timeframe", True)
        self.enable_meta_labeling = self.feature_config.get("enable_meta_labeling", True)

        # Multi-timeframe configuration
        self.timeframes = ["1m", "5m", "15m", "30m"]

        # Initialize components
        self.volatility_model = None
        self.correlation_analyzer = None
        self.momentum_analyzer = None
        self.liquidity_analyzer = None
        self.candlestick_analyzer = None
        self.sr_distance_calculator = None
        self.wavelet_analyzer = None

        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="vectorized advanced feature engineering initialization",
    )
    async def initialize(self) -> bool:
        """Initialize vectorized advanced feature engineering components."""
        try:
            self.logger.info("🚀 Initializing vectorized advanced feature engineering...")

            # Initialize volatility modeling
            if self.enable_volatility_modeling:
                self.volatility_model = VectorizedVolatilityRegimeModel(self.config)
                await self.volatility_model.initialize()

            # Initialize correlation analysis
            if self.enable_correlation_analysis:
                self.correlation_analyzer = VectorizedCorrelationAnalyzer(self.config)
                await self.correlation_analyzer.initialize()

            # Initialize momentum analysis
            if self.enable_momentum_analysis:
                self.momentum_analyzer = VectorizedMomentumAnalyzer(self.config)
                await self.momentum_analyzer.initialize()

            # Initialize liquidity analysis
            if self.enable_liquidity_analysis:
                self.liquidity_analyzer = VectorizedLiquidityAnalyzer(self.config)
                await self.liquidity_analyzer.initialize()

            # Initialize candlestick pattern analyzer
            if self.enable_candlestick_patterns:
                self.candlestick_analyzer = VectorizedCandlestickPatternAnalyzer(self.config)
                await self.candlestick_analyzer.initialize()

            # Initialize S/R distance calculator
            if self.enable_sr_distance:
                self.sr_distance_calculator = VectorizedSRDistanceCalculator(self.config)
                await self.sr_distance_calculator.initialize()

            # Initialize wavelet transform analyzer
            if self.enable_wavelet_transforms:
                self.wavelet_analyzer = VectorizedWaveletTransformAnalyzer(self.config)
                await self.wavelet_analyzer.initialize()

            # Initialize meta-labeling system
            if self.enable_meta_labeling:
                from src.analyst.meta_labeling_system import MetaLabelingSystem

                self.meta_labeling_system = MetaLabelingSystem(self.config)
                await self.meta_labeling_system.initialize()

            self.is_initialized = True
            self.logger.info("✅ Vectorized advanced feature engineering initialized successfully")
            return True

        except Exception as e:
            self.logger.error(
                f"❌ Error initializing vectorized advanced feature engineering: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="vectorized advanced feature engineering",
    )
    async def engineer_features(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
        sr_levels: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Engineer advanced features for improved prediction accuracy using vectorized operations.

        Args:
            price_data: OHLCV price data
            volume_data: Volume and trade flow data
            order_flow_data: Order book and flow data (optional)
            sr_levels: Support/resistance levels (optional)

        Returns:
            Dictionary containing engineered features
        """
        try:
            if not self.is_initialized:
                self.logger.error("Vectorized advanced feature engineering not initialized")
                return {}

            features = {}

            # Market microstructure features
            microstructure_features = await self._engineer_microstructure_features_vectorized(
                price_data,
                volume_data,
                order_flow_data,
            )
            features.update(microstructure_features)

            # Volatility regime features
            if self.volatility_model:
                volatility_features = await self.volatility_model.model_volatility_vectorized(
                    price_data,
                )
                features.update(volatility_features)

            # Correlation analysis features
            if self.correlation_analyzer:
                correlation_features = (
                    await self.correlation_analyzer.analyze_correlations_vectorized(price_data)
                )
                features.update(correlation_features)

            # Momentum analysis features
            if self.momentum_analyzer:
                momentum_features = await self.momentum_analyzer.analyze_momentum_vectorized(
                    price_data,
                )
                features.update(momentum_features)

            # Liquidity analysis features
            if self.liquidity_analyzer:
                liquidity_features = await self.liquidity_analyzer.analyze_liquidity_vectorized(
                    price_data,
                    volume_data,
                    order_flow_data,
                )
                features.update(liquidity_features)

            # Candlestick pattern features
            if self.candlestick_analyzer:
                candlestick_features = await self.candlestick_analyzer.analyze_patterns(
                    price_data,
                )
                features.update(candlestick_features)

            # S/R distance features
            if self.sr_distance_calculator and sr_levels:
                sr_distance_features = await self.sr_distance_calculator.calculate_sr_distances(
                    price_data,
                    sr_levels,
                )
                features.update(sr_distance_features)

            # Wavelet transform features
            if self.wavelet_analyzer:
                wavelet_features = await self.wavelet_analyzer.analyze_wavelet_transforms(
                    price_data,
                    volume_data,
                )
                features.update(wavelet_features)

            # Adaptive indicators
            adaptive_features = self._engineer_adaptive_indicators_vectorized(price_data)
            features.update(adaptive_features)

            # Feature selection and dimensionality reduction
            selected_features = self._select_optimal_features_vectorized(features)

            # Add multi-timeframe features if enabled
            if self.enable_multi_timeframe:
                multi_timeframe_features = (
                    await self._engineer_multi_timeframe_features_vectorized(
                        price_data,
                        volume_data,
                        order_flow_data,
                        sr_levels,
                    )
                )
                selected_features.update(multi_timeframe_features)

            # Add meta-labeling if enabled
            if self.enable_meta_labeling:
                meta_labels = await self._generate_meta_labels_vectorized(
                    price_data,
                    volume_data,
                    order_flow_data,
                )
                selected_features.update(meta_labels)

            self.logger.info(
                f"✅ Engineered {len(selected_features)} vectorized advanced features including wavelet transforms",
            )
            return selected_features

        except Exception as e:
            self.logger.error(f"Error engineering vectorized advanced features: {e}")
            return {}

    async def _engineer_microstructure_features_vectorized(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Engineer market microstructure features using vectorized operations."""
        try:
            features = {}

            # Price impact features
            features["price_impact"] = self._calculate_price_impact_vectorized(price_data, volume_data)
            features["volume_price_impact"] = self._calculate_volume_price_impact_vectorized(price_data, volume_data)

            # Order flow imbalance features
            if order_flow_data is not None:
                features["order_flow_imbalance"] = self._calculate_order_flow_imbalance_vectorized(order_flow_data)
                features["bid_ask_spread"] = self._calculate_bid_ask_spread_vectorized(order_flow_data)

            # Market depth features
            features["market_depth"] = self._calculate_market_depth_vectorized(price_data, volume_data)

            return features

        except Exception as e:
            self.logger.error(f"Error engineering microstructure features: {e}")
            return {}

    def _calculate_price_impact_vectorized(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> float:
        """Calculate price impact using vectorized operations."""
        try:
            price_changes = price_data["close"].pct_change().abs()
            volume_changes = volume_data["volume"].pct_change().abs()
            
            # Calculate price impact as correlation between price and volume changes
            correlation = np.corrcoef(price_changes.dropna(), volume_changes.dropna())[0, 1]
            return correlation if not np.isnan(correlation) else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating price impact: {e}")
            return 0.0

    def _calculate_volume_price_impact_vectorized(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> float:
        """Calculate volume-price impact using vectorized operations."""
        try:
            # Calculate volume-weighted price changes
            price_changes = price_data["close"].pct_change()
            volume_weights = volume_data["volume"] / volume_data["volume"].sum()
            
            # Volume-weighted average price change
            vwap_change = np.sum(price_changes.dropna() * volume_weights.dropna())
            return vwap_change

        except Exception as e:
            self.logger.error(f"Error calculating volume-price impact: {e}")
            return 0.0

    def _calculate_order_flow_imbalance_vectorized(self, order_flow_data: pd.DataFrame) -> float:
        """Calculate order flow imbalance using vectorized operations."""
        try:
            # Simplified order flow imbalance calculation
            # In practice, this would use actual order book data
            return 0.0  # Placeholder

        except Exception as e:
            self.logger.error(f"Error calculating order flow imbalance: {e}")
            return 0.0

    def _calculate_bid_ask_spread_vectorized(self, order_flow_data: pd.DataFrame) -> float:
        """Calculate bid-ask spread using vectorized operations."""
        try:
            # Simplified bid-ask spread calculation
            # In practice, this would use actual order book data
            return 0.0  # Placeholder

        except Exception as e:
            self.logger.error(f"Error calculating bid-ask spread: {e}")
            return 0.0

    def _calculate_market_depth_vectorized(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> float:
        """Calculate market depth using vectorized operations."""
        try:
            # Market depth as average volume over time
            market_depth = volume_data["volume"].rolling(window=20).mean().iloc[-1]
            return market_depth

        except Exception as e:
            self.logger.error(f"Error calculating market depth: {e}")
            return 0.0

    def _engineer_adaptive_indicators_vectorized(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """Engineer adaptive indicators using vectorized operations."""
        try:
            features = {}

            # Adaptive moving averages
            features["adaptive_sma"] = self._calculate_adaptive_sma_vectorized(price_data)
            features["adaptive_ema"] = self._calculate_adaptive_ema_vectorized(price_data)

            # Adaptive volatility indicators
            features["adaptive_atr"] = self._calculate_adaptive_atr_vectorized(price_data)
            features["adaptive_bollinger"] = self._calculate_adaptive_bollinger_vectorized(price_data)

            return features

        except Exception as e:
            self.logger.error(f"Error engineering adaptive indicators: {e}")
            return {}

    def _calculate_adaptive_sma_vectorized(self, price_data: pd.DataFrame) -> float:
        """Calculate adaptive SMA using vectorized operations."""
        try:
            # Adaptive SMA based on volatility
            volatility = price_data["close"].pct_change().rolling(window=20).std()
            adaptive_window = np.clip(20 / (1 + volatility * 100), 5, 50).astype(int)
            
            # Calculate adaptive SMA
            adaptive_sma = price_data["close"].rolling(window=adaptive_window.iloc[-1]).mean().iloc[-1]
            return adaptive_sma

        except Exception as e:
            self.logger.error(f"Error calculating adaptive SMA: {e}")
            return 0.0

    def _calculate_adaptive_ema_vectorized(self, price_data: pd.DataFrame) -> float:
        """Calculate adaptive EMA using vectorized operations."""
        try:
            # Adaptive EMA based on volatility
            volatility = price_data["close"].pct_change().rolling(window=20).std()
            adaptive_span = np.clip(12 / (1 + volatility * 100), 2, 50)
            
            # Calculate adaptive EMA
            adaptive_ema = price_data["close"].ewm(span=adaptive_span.iloc[-1]).mean().iloc[-1]
            return adaptive_ema

        except Exception as e:
            self.logger.error(f"Error calculating adaptive EMA: {e}")
            return 0.0

    def _calculate_adaptive_atr_vectorized(self, price_data: pd.DataFrame) -> float:
        """Calculate adaptive ATR using vectorized operations."""
        try:
            # Adaptive ATR based on volatility regime
            high_low = price_data["high"] - price_data["low"]
            high_close = np.abs(price_data["high"] - price_data["close"].shift())
            low_close = np.abs(price_data["low"] - price_data["close"].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Adaptive window based on volatility
            volatility = true_range.rolling(window=20).std()
            adaptive_window = np.clip(14 / (1 + volatility * 10), 5, 30).astype(int)
            
            adaptive_atr = true_range.rolling(window=adaptive_window.iloc[-1]).mean().iloc[-1]
            return adaptive_atr

        except Exception as e:
            self.logger.error(f"Error calculating adaptive ATR: {e}")
            return 0.0

    def _calculate_adaptive_bollinger_vectorized(self, price_data: pd.DataFrame) -> float:
        """Calculate adaptive Bollinger Bands using vectorized operations."""
        try:
            # Adaptive Bollinger Bands based on volatility
            volatility = price_data["close"].pct_change().rolling(window=20).std()
            adaptive_window = np.clip(20 / (1 + volatility * 100), 10, 50).astype(int)
            
            # Calculate adaptive Bollinger Bands
            sma = price_data["close"].rolling(window=adaptive_window.iloc[-1]).mean()
            std = price_data["close"].rolling(window=adaptive_window.iloc[-1]).std()
            
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            # Return position within bands
            current_price = price_data["close"].iloc[-1]
            position = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
            return position

        except Exception as e:
            self.logger.error(f"Error calculating adaptive Bollinger Bands: {e}")
            return 0.0

    def _select_optimal_features_vectorized(self, features: dict[str, Any]) -> dict[str, Any]:
        """Select optimal features using vectorized operations."""
        try:
            # Simple feature selection based on variance
            selected_features = {}
            
            for feature_name, feature_value in features.items():
                if isinstance(feature_value, (int, float)) and not np.isnan(feature_value):
                    selected_features[feature_name] = feature_value
            
            return selected_features

        except Exception as e:
            self.logger.error(f"Error selecting optimal features: {e}")
            return features

    async def _engineer_multi_timeframe_features_vectorized(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
        sr_levels: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Engineer multi-timeframe features using vectorized operations."""
        try:
            features = {}

            # Multi-timeframe features for different timeframes
            for timeframe in self.timeframes:
                # Resample data to timeframe
                resampled_price = self._resample_data_vectorized(price_data, timeframe)
                resampled_volume = self._resample_data_vectorized(volume_data, timeframe)
                
                # Calculate features for this timeframe
                timeframe_features = await self._calculate_timeframe_features_vectorized(
                    resampled_price,
                    resampled_volume,
                    timeframe,
                )
                
                # Add timeframe prefix to features
                for feature_name, feature_value in timeframe_features.items():
                    features[f"{timeframe}_{feature_name}"] = feature_value

            return features

        except Exception as e:
            self.logger.error(f"Error engineering multi-timeframe features: {e}")
            return {}

    def _resample_data_vectorized(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to specified timeframe using vectorized operations."""
        try:
            # Convert timeframe string to pandas offset
            timeframe_map = {
                "1m": "1T",
                "5m": "5T",
                "15m": "15T",
                "30m": "30T",
            }
            
            offset = timeframe_map.get(timeframe, "1T")
            resampled = data.resample(offset).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            })
            
            return resampled.dropna()

        except Exception as e:
            self.logger.error(f"Error resampling data: {e}")
            return data

    async def _calculate_timeframe_features_vectorized(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        timeframe: str,
    ) -> dict[str, Any]:
        """Calculate features for specific timeframe using vectorized operations."""
        try:
            features = {}

            # Basic price features
            features["price_change"] = price_data["close"].pct_change().iloc[-1]
            features["price_volatility"] = price_data["close"].pct_change().rolling(window=20).std().iloc[-1]
            
            # Volume features
            features["volume_change"] = volume_data["volume"].pct_change().iloc[-1]
            features["volume_ma_ratio"] = volume_data["volume"].iloc[-1] / volume_data["volume"].rolling(window=20).mean().iloc[-1]

            return features

        except Exception as e:
            self.logger.error(f"Error calculating timeframe features: {e}")
            return {}

    async def _generate_meta_labels_vectorized(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Generate meta labels using vectorized operations."""
        try:
            features = {}

            # Meta-labeling based on volatility regime
            volatility = price_data["close"].pct_change().rolling(window=20).std()
            features["volatility_regime"] = 1 if volatility.iloc[-1] > volatility.quantile(0.75) else 0

            # Meta-labeling based on volume regime
            volume_ma = volume_data["volume"].rolling(window=20).mean()
            features["volume_regime"] = 1 if volume_data["volume"].iloc[-1] > volume_ma.iloc[-1] else 0

            # Meta-labeling based on trend regime
            sma_short = price_data["close"].rolling(window=10).mean()
            sma_long = price_data["close"].rolling(window=30).mean()
            features["trend_regime"] = 1 if sma_short.iloc[-1] > sma_long.iloc[-1] else 0

            return features

        except Exception as e:
            self.logger.error(f"Error generating meta labels: {e}")
            return {}


# Placeholder classes for other analyzers
class VectorizedVolatilityRegimeModel:
    """Placeholder for volatility regime modeling."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedVolatilityRegimeModel")
    
    async def initialize(self) -> bool:
        return True
    
    async def model_volatility_vectorized(self, price_data: pd.DataFrame) -> dict[str, Any]:
        return {"volatility_regime": 0.5}


class VectorizedCorrelationAnalyzer:
    """Placeholder for correlation analysis."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedCorrelationAnalyzer")
    
    async def initialize(self) -> bool:
        return True
    
    async def analyze_correlations_vectorized(self, price_data: pd.DataFrame) -> dict[str, Any]:
        return {"correlation_strength": 0.3}


class VectorizedMomentumAnalyzer:
    """Placeholder for momentum analysis."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedMomentumAnalyzer")
    
    async def initialize(self) -> bool:
        return True
    
    async def analyze_momentum_vectorized(self, price_data: pd.DataFrame) -> dict[str, Any]:
        return {"momentum_strength": 0.4}


class VectorizedLiquidityAnalyzer:
    """Placeholder for liquidity analysis."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedLiquidityAnalyzer")
    
    async def initialize(self) -> bool:
        return True
    
    async def analyze_liquidity_vectorized(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        return {"liquidity_score": 0.6}


class VectorizedSRDistanceCalculator:
    """Placeholder for S/R distance calculation."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedSRDistanceCalculator")
    
    async def initialize(self) -> bool:
        return True
    
    async def calculate_sr_distances(
        self,
        price_data: pd.DataFrame,
        sr_levels: dict[str, Any],
    ) -> dict[str, Any]:
        return {"nearest_support_distance": 0.02, "nearest_resistance_distance": 0.03}


class VectorizedCandlestickPatternAnalyzer:
    """
    Comprehensive candlestick pattern analyzer implementing all major patterns
    for enhanced feature engineering and ML model training with vectorized operations.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedCandlestickPatternAnalyzer")

        # Pattern detection parameters
        self.pattern_config = config.get("candlestick_patterns", {})
        self.doji_threshold = self.pattern_config.get("doji_threshold", 0.1)
        self.hammer_ratio = self.pattern_config.get("hammer_ratio", 0.3)
        self.shadow_ratio = self.pattern_config.get("shadow_ratio", 2.0)
        self.engulfing_ratio = self.pattern_config.get("engulfing_ratio", 1.1)
        self.tweezer_threshold = self.pattern_config.get("tweezer_threshold", 0.02)
        self.marubozu_threshold = self.pattern_config.get("marubozu_threshold", 0.1)

        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="candlestick pattern analyzer initialization",
    )
    async def initialize(self) -> bool:
        """Initialize candlestick pattern analyzer."""
        try:
            self.logger.info("🚀 Initializing vectorized candlestick pattern analyzer...")
            self.is_initialized = True
            self.logger.info("✅ Vectorized candlestick pattern analyzer initialized successfully")
            return True
        except Exception as e:
            self.logger.error(
                f"❌ Error initializing vectorized candlestick pattern analyzer: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return={},
        context="candlestick pattern analysis",
    )
    async def analyze_patterns(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze candlestick patterns and return features for ML training using vectorized operations.

        Args:
            price_data: OHLCV price data

        Returns:
            Dictionary containing candlestick pattern features
        """
        try:
            if not self.is_initialized:
                self.logger.error("Candlestick pattern analyzer not initialized")
                return {}

            if price_data.empty or len(price_data) < 3:
                self.logger.warning("Insufficient data for pattern analysis")
                return {}

            # Prepare data with calculated metrics using vectorized operations
            df = self._prepare_candlestick_data_vectorized(price_data)

            # Analyze all patterns using vectorized operations
            patterns = {
                "engulfing_patterns": self._detect_engulfing_patterns_vectorized(df),
                "hammer_hanging_man": self._detect_hammer_hanging_man_vectorized(df),
                "shooting_star_inverted_hammer": self._detect_shooting_star_inverted_hammer_vectorized(df),
                "tweezer_patterns": self._detect_tweezer_patterns_vectorized(df),
                "marubozu_patterns": self._detect_marubozu_patterns_vectorized(df),
                "three_methods_patterns": self._detect_three_methods_patterns_vectorized(df),
                "doji_patterns": self._detect_doji_patterns_vectorized(df),
                "spinning_top_patterns": self._detect_spinning_top_patterns_vectorized(df),
            }

            # Convert patterns to ML features using vectorized operations
            features = self._convert_patterns_to_features_vectorized(patterns, df)

            self.logger.info(f"✅ Analyzed {len(patterns)} pattern categories using vectorized operations")
            return features

        except Exception as e:
            self.logger.error(f"Error analyzing candlestick patterns: {e}")
            return {}

    def _prepare_candlestick_data_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with candlestick metrics using vectorized operations."""
        try:
            df = df.copy()

            # Calculate basic candlestick metrics using vectorized operations
            df["body_size"] = np.abs(df["close"] - df["open"])
            df["upper_shadow"] = df["high"] - np.maximum(df["open"], df["close"])
            df["lower_shadow"] = np.minimum(df["open"], df["close"]) - df["low"]
            df["total_range"] = df["high"] - df["low"]
            df["body_ratio"] = df["body_size"] / df["total_range"].replace(0, 1)
            df["is_bullish"] = df["close"] > df["open"]

            # Calculate moving averages for context using vectorized operations
            df["avg_body_size"] = df["body_size"].rolling(window=20).mean()
            df["avg_range"] = df["total_range"].rolling(window=20).mean()

            return df.dropna()

        except Exception as e:
            self.logger.error(f"Error preparing candlestick data: {e}")
            return pd.DataFrame()

    def _detect_engulfing_patterns_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect bullish and bearish engulfing patterns using vectorized operations."""
        try:
            # Vectorized calculations
            current_body_size = df["body_size"].values
            previous_body_size = df["body_size"].shift(1).values
            current_is_bullish = df["is_bullish"].values
            previous_is_bullish = df["is_bullish"].shift(1).values
            current_open = df["open"].values
            current_close = df["close"].values
            previous_open = df["open"].shift(1).values
            previous_close = df["close"].shift(1).values

            # Bullish engulfing conditions
            bullish_engulfing = (
                current_is_bullish &
                ~previous_is_bullish &
                (current_open < previous_close) &
                (current_close > previous_open) &
                (current_body_size > previous_body_size * self.engulfing_ratio)
            )

            # Bearish engulfing conditions
            bearish_engulfing = (
                ~current_is_bullish &
                previous_is_bullish &
                (current_open > previous_close) &
                (current_close < previous_open) &
                (current_body_size > previous_body_size * self.engulfing_ratio)
            )

            # Calculate confidence scores
            bullish_confidence = np.where(
                bullish_engulfing,
                np.minimum(current_body_size / (previous_body_size + 1e-8), 2.0),
                0.0
            )
            bearish_confidence = np.where(
                bearish_engulfing,
                np.minimum(current_body_size / (previous_body_size + 1e-8), 2.0),
                0.0
            )

            return {
                "bullish_engulfing": bullish_engulfing,
                "bearish_engulfing": bearish_engulfing,
                "bullish_confidence": bullish_confidence,
                "bearish_confidence": bearish_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting engulfing patterns: {e}")
            return {}

    def _detect_hammer_hanging_man_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect hammer and hanging man patterns using vectorized operations."""
        try:
            # Vectorized calculations
            body_ratio = df["body_ratio"].values
            lower_shadow = df["lower_shadow"].values
            upper_shadow = df["upper_shadow"].values
            body_size = df["body_size"].values
            is_bullish = df["is_bullish"].values
            close_prices = df["close"].values

            # Hammer pattern conditions
            hammer_conditions = (
                (body_ratio <= self.hammer_ratio) &
                (lower_shadow > body_size * self.shadow_ratio) &
                (upper_shadow < body_size * 0.5)
            )

            # Hanging man pattern conditions (need previous close for context)
            previous_close = df["close"].shift(1).values
            hanging_man_conditions = (
                (body_ratio <= self.hammer_ratio) &
                (lower_shadow > body_size * self.shadow_ratio) &
                (upper_shadow < body_size * 0.5) &
                (previous_close > close_prices)
            )

            # Calculate confidence scores
            hammer_confidence = np.where(
                hammer_conditions,
                np.minimum(lower_shadow / (body_size + 1e-8), 3.0),
                0.0
            )
            hanging_man_confidence = np.where(
                hanging_man_conditions,
                np.minimum(lower_shadow / (body_size + 1e-8), 3.0),
                0.0
            )

            return {
                "hammer": hammer_conditions,
                "hanging_man": hanging_man_conditions,
                "hammer_confidence": hammer_confidence,
                "hanging_man_confidence": hanging_man_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting hammer/hanging man patterns: {e}")
            return {}

    def _detect_shooting_star_inverted_hammer_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect shooting star and inverted hammer patterns using vectorized operations."""
        try:
            # Vectorized calculations
            body_ratio = df["body_ratio"].values
            upper_shadow = df["upper_shadow"].values
            lower_shadow = df["lower_shadow"].values
            body_size = df["body_size"].values
            close_prices = df["close"].values
            previous_close = df["close"].shift(1).values

            # Shooting star pattern conditions
            shooting_star_conditions = (
                (body_ratio <= self.hammer_ratio) &
                (upper_shadow > body_size * self.shadow_ratio) &
                (lower_shadow < body_size * 0.5)
            )

            # Inverted hammer pattern conditions
            inverted_hammer_conditions = (
                (body_ratio <= self.hammer_ratio) &
                (upper_shadow > body_size * self.shadow_ratio) &
                (lower_shadow < body_size * 0.5) &
                (previous_close < close_prices)
            )

            # Calculate confidence scores
            shooting_star_confidence = np.where(
                shooting_star_conditions,
                np.minimum(upper_shadow / (body_size + 1e-8), 3.0),
                0.0
            )
            inverted_hammer_confidence = np.where(
                inverted_hammer_conditions,
                np.minimum(upper_shadow / (body_size + 1e-8), 3.0),
                0.0
            )

            return {
                "shooting_star": shooting_star_conditions,
                "inverted_hammer": inverted_hammer_conditions,
                "shooting_star_confidence": shooting_star_confidence,
                "inverted_hammer_confidence": inverted_hammer_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting shooting star/inverted hammer patterns: {e}")
            return {}

    def _detect_tweezer_patterns_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect tweezer tops and bottoms patterns using vectorized operations."""
        try:
            # Vectorized calculations
            current_high = df["high"].values
            current_low = df["low"].values
            current_close = df["close"].values
            current_open = df["open"].values
            previous_high = df["high"].shift(1).values
            previous_low = df["low"].shift(1).values
            previous_close = df["close"].shift(1).values
            previous_open = df["open"].shift(1).values

            # Tweezer tops conditions
            tweezer_top_conditions = (
                (np.abs(current_high - previous_high) <= self.tweezer_threshold * current_high) &
                (current_high > current_close) &
                (previous_high > previous_close)
            )

            # Tweezer bottoms conditions
            tweezer_bottom_conditions = (
                (np.abs(current_low - previous_low) <= self.tweezer_threshold * current_low) &
                (current_low < current_open) &
                (previous_low < previous_open)
            )

            # Calculate confidence scores
            tweezer_top_confidence = np.where(
                tweezer_top_conditions,
                1.0 - np.abs(current_high - previous_high) / (current_high + 1e-8),
                0.0
            )
            tweezer_bottom_confidence = np.where(
                tweezer_bottom_conditions,
                1.0 - np.abs(current_low - previous_low) / (current_low + 1e-8),
                0.0
            )

            return {
                "tweezer_top": tweezer_top_conditions,
                "tweezer_bottom": tweezer_bottom_conditions,
                "tweezer_top_confidence": tweezer_top_confidence,
                "tweezer_bottom_confidence": tweezer_bottom_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting tweezer patterns: {e}")
            return {}

    def _detect_marubozu_patterns_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect bullish and bearish marubozu patterns using vectorized operations."""
        try:
            # Vectorized calculations
            upper_shadow = df["upper_shadow"].values
            lower_shadow = df["lower_shadow"].values
            total_range = df["total_range"].values
            is_bullish = df["is_bullish"].values

            # Marubozu conditions (no shadows or very small shadows)
            marubozu_conditions = (
                (upper_shadow < total_range * self.marubozu_threshold) &
                (lower_shadow < total_range * self.marubozu_threshold)
            )

            # Separate bullish and bearish marubozu
            bullish_marubozu = marubozu_conditions & is_bullish
            bearish_marubozu = marubozu_conditions & ~is_bullish

            # Calculate confidence scores
            marubozu_confidence = np.where(
                marubozu_conditions,
                1.0 - (upper_shadow + lower_shadow) / (total_range + 1e-8),
                0.0
            )

            return {
                "bullish_marubozu": bullish_marubozu,
                "bearish_marubozu": bearish_marubozu,
                "marubozu_confidence": marubozu_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting marubozu patterns: {e}")
            return {}

    def _detect_three_methods_patterns_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect rising and falling three methods patterns using vectorized operations."""
        try:
            # This is a complex pattern that requires looking at 5 candles
            # For vectorized implementation, we'll use rolling windows
            window_size = 5
            patterns = {
                "rising_three_methods": np.zeros(len(df), dtype=bool),
                "falling_three_methods": np.zeros(len(df), dtype=bool),
                "rising_three_methods_confidence": np.zeros(len(df), dtype=float),
                "falling_three_methods_confidence": np.zeros(len(df), dtype=float),
            }

            # Process each window
            for i in range(window_size - 1, len(df)):
                window = df.iloc[i - window_size + 1:i + 1]
                
                if self._is_rising_three_methods_vectorized(window):
                    patterns["rising_three_methods"][i] = True
                    patterns["rising_three_methods_confidence"][i] = 0.8
                
                if self._is_falling_three_methods_vectorized(window):
                    patterns["falling_three_methods"][i] = True
                    patterns["falling_three_methods_confidence"][i] = 0.8

            return patterns

        except Exception as e:
            self.logger.error(f"Error detecting three methods patterns: {e}")
            return {}

    def _is_rising_three_methods_vectorized(self, window: pd.DataFrame) -> bool:
        """Check if the 5-candle pattern is a rising three methods using vectorized operations."""
        try:
            if len(window) != 5:
                return False

            candles = window.values
            body_sizes = np.abs(candles[:, 3] - candles[:, 0])  # close - open
            highs = candles[:, 1]
            lows = candles[:, 2]
            closes = candles[:, 3]
            opens = candles[:, 0]
            is_bullish = closes > opens

            # First candle should be a long bullish candle
            if not (is_bullish[0] and body_sizes[0] > np.mean(body_sizes)):
                return False

            # Next three candles should be small bearish candles within the range of the first
            for i in range(1, 4):
                if (is_bullish[i] or highs[i] > highs[0] or lows[i] < lows[0]):
                    return False

            # Last candle should be a long bullish candle closing above the first
            if not (is_bullish[4] and closes[4] > closes[0] and body_sizes[4] > np.mean(body_sizes)):
                return False

            return True

        except Exception:
            return False

    def _is_falling_three_methods_vectorized(self, window: pd.DataFrame) -> bool:
        """Check if the 5-candle pattern is a falling three methods using vectorized operations."""
        try:
            if len(window) != 5:
                return False

            candles = window.values
            body_sizes = np.abs(candles[:, 3] - candles[:, 0])  # close - open
            highs = candles[:, 1]
            lows = candles[:, 2]
            closes = candles[:, 3]
            opens = candles[:, 0]
            is_bullish = closes > opens

            # First candle should be a long bearish candle
            if not (~is_bullish[0] and body_sizes[0] > np.mean(body_sizes)):
                return False

            # Next three candles should be small bullish candles within the range of the first
            for i in range(1, 4):
                if (~is_bullish[i] or highs[i] > highs[0] or lows[i] < lows[0]):
                    return False

            # Last candle should be a long bearish candle closing below the first
            if not (~is_bullish[4] and closes[4] < closes[0] and body_sizes[4] > np.mean(body_sizes)):
                return False

            return True

        except Exception:
            return False

    def _detect_doji_patterns_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect doji patterns using vectorized operations."""
        try:
            # Vectorized calculations
            body_ratio = df["body_ratio"].values

            # Doji pattern conditions (very small body)
            doji_conditions = body_ratio <= self.doji_threshold

            # Calculate confidence scores
            doji_confidence = np.where(doji_conditions, 1.0 - body_ratio, 0.0)

            return {
                "doji": doji_conditions,
                "doji_confidence": doji_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting doji patterns: {e}")
            return {}

    def _detect_spinning_top_patterns_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect spinning top patterns using vectorized operations."""
        try:
            # Vectorized calculations
            body_ratio = df["body_ratio"].values
            upper_shadow = df["upper_shadow"].values
            lower_shadow = df["lower_shadow"].values
            total_range = df["total_range"].values

            # Spinning top conditions (small body, equal shadows)
            spinning_top_conditions = (
                (body_ratio <= 0.3) &
                (np.abs(upper_shadow - lower_shadow) < 0.2 * total_range) &
                (upper_shadow > 0.1 * total_range) &
                (lower_shadow > 0.1 * total_range)
            )

            # Calculate confidence scores
            spinning_top_confidence = np.where(spinning_top_conditions, 0.7, 0.0)

            return {
                "spinning_top": spinning_top_conditions,
                "spinning_top_confidence": spinning_top_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting spinning top patterns: {e}")
            return {}

    def _convert_patterns_to_features_vectorized(
        self,
        patterns: dict[str, dict[str, np.ndarray]],
        df: pd.DataFrame,
    ) -> dict[str, float]:
        """Convert pattern analysis to ML features using vectorized operations."""
        try:
            features = {}

            # Calculate pattern type features (count and presence)
            pattern_types = [
                "engulfing_patterns",
                "hammer_hanging_man",
                "shooting_star_inverted_hammer",
                "tweezer_patterns",
                "marubozu_patterns",
                "three_methods_patterns",
                "doji_patterns",
                "spinning_top_patterns",
            ]

            for pattern_type in pattern_types:
                if pattern_type in patterns:
                    pattern_data = patterns[pattern_type]
                    # Count patterns
                    pattern_count = sum(
                        np.sum(pattern_data.get(key, np.zeros(len(df), dtype=bool)))
                        for key in pattern_data.keys()
                        if isinstance(pattern_data[key], np.ndarray) and pattern_data[key].dtype == bool
                    )
                    features[f"{pattern_type}_count"] = pattern_count
                    features[f"{pattern_type}_present"] = 1.0 if pattern_count > 0 else 0.0

            # Calculate specific pattern features
            specific_patterns = [
                "bullish_engulfing",
                "bearish_engulfing",
                "hammer",
                "hanging_man",
                "shooting_star",
                "inverted_hammer",
                "tweezer_top",
                "tweezer_bottom",
                "bullish_marubozu",
                "bearish_marubozu",
                "rising_three_methods",
                "falling_three_methods",
                "doji",
                "spinning_top",
            ]

            for pattern in specific_patterns:
                pattern_count = 0
                for pattern_data in patterns.values():
                    if pattern in pattern_data:
                        pattern_count += np.sum(pattern_data[pattern])
                
                features[f"{pattern}_count"] = pattern_count
                features[f"{pattern}_present"] = 1.0 if pattern_count > 0 else 0.0

            # Calculate pattern density features
            total_patterns = sum(
                features.get(f"{pt}_count", 0) for pt in pattern_types
            )
            features["total_patterns"] = total_patterns
            features["pattern_density"] = total_patterns / len(df) if len(df) > 0 else 0.0

            # Calculate bullish vs bearish pattern features
            bullish_patterns = sum(
                features.get(f"{pattern}_count", 0)
                for pattern in ["bullish_engulfing", "hammer", "inverted_hammer", "tweezer_bottom", "bullish_marubozu", "rising_three_methods"]
            )
            bearish_patterns = sum(
                features.get(f"{pattern}_count", 0)
                for pattern in ["bearish_engulfing", "hanging_man", "shooting_star", "tweezer_top", "bearish_marubozu", "falling_three_methods"]
            )

            features["bullish_patterns"] = bullish_patterns
            features["bearish_patterns"] = bearish_patterns
            features["bullish_bearish_ratio"] = bullish_patterns / (bearish_patterns + 1e-8)

            # Calculate recent pattern features (last 5 candles)
            recent_patterns = 0
            recent_bullish_patterns = 0
            recent_bearish_patterns = 0

            for pattern_data in patterns.values():
                for key, pattern_array in pattern_data.items():
                    if isinstance(pattern_array, np.ndarray) and pattern_array.dtype == bool:
                        if len(pattern_array) >= 5:
                            recent_count = np.sum(pattern_array[-5:])
                            recent_patterns += recent_count
                            
                            if any(bullish in key for bullish in ["bullish", "hammer", "inverted_hammer", "tweezer_bottom", "rising"]):
                                recent_bullish_patterns += recent_count
                            elif any(bearish in key for bearish in ["bearish", "hanging_man", "shooting_star", "tweezer_top", "falling"]):
                                recent_bearish_patterns += recent_count

            features["recent_patterns_count"] = recent_patterns
            features["recent_bullish_patterns"] = recent_bullish_patterns
            features["recent_bearish_patterns"] = recent_bearish_patterns

            # Calculate pattern confidence features
            all_confidences = []
            for pattern_data in patterns.values():
                for key, confidence_array in pattern_data.items():
                    if "confidence" in key and isinstance(confidence_array, np.ndarray):
                        all_confidences.extend(confidence_array[confidence_array > 0])

            if all_confidences:
                features["avg_pattern_confidence"] = np.mean(all_confidences)
                features["max_pattern_confidence"] = np.max(all_confidences)
                features["pattern_confidence_std"] = np.std(all_confidences)
            else:
                features["avg_pattern_confidence"] = 0.0
                features["max_pattern_confidence"] = 0.0
                features["pattern_confidence_std"] = 0.0

            return features

        except Exception as e:
            self.logger.error(f"Error converting patterns to features: {e}")
            return {}


class VectorizedWaveletTransformAnalyzer:
    """
    Comprehensive wavelet transform analyzer for signal processing and feature extraction.
    Implements various wavelet transforms for financial time series analysis.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedWaveletTransformAnalyzer")

        # Wavelet configuration
        self.wavelet_config = config.get("wavelet_transforms", {})
        self.wavelet_type = self.wavelet_config.get("wavelet_type", "db4")
        self.decomposition_level = self.wavelet_config.get("decomposition_level", 4)
        self.enable_continuous_wavelet = self.wavelet_config.get("enable_continuous_wavelet", True)
        self.enable_discrete_wavelet = self.wavelet_config.get("enable_discrete_wavelet", True)
        self.enable_wavelet_packet = self.wavelet_config.get("enable_wavelet_packet", True)
        self.enable_denoising = self.wavelet_config.get("enable_denoising", True)

        # Wavelet types for different analyses
        self.wavelet_types = ["db1", "db2", "db4", "db8", "haar", "sym2", "sym4", "coif1", "coif2"]
        
        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="wavelet transform analyzer initialization",
    )
    async def initialize(self) -> bool:
        """Initialize wavelet transform analyzer."""
        try:
            self.logger.info("🚀 Initializing vectorized wavelet transform analyzer...")
            self.is_initialized = True
            self.logger.info("✅ Vectorized wavelet transform analyzer initialized successfully")
            return True
        except Exception as e:
            self.logger.error(
                f"❌ Error initializing vectorized wavelet transform analyzer: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return={},
        context="wavelet transform analysis",
    )
    async def analyze_wavelet_transforms(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Analyze wavelet transforms for signal processing and feature extraction.

        Args:
            price_data: OHLCV price data
            volume_data: Volume data (optional)

        Returns:
            Dictionary containing wavelet transform features
        """
        try:
            if not self.is_initialized:
                self.logger.error("Wavelet transform analyzer not initialized")
                return {}

            if price_data.empty:
                self.logger.warning("Empty price data provided for wavelet analysis")
                return {}

            self.logger.info("🔍 Performing wavelet transform analysis...")

            features = {}

            # 1. Discrete Wavelet Transform (DWT) analysis
            if self.enable_discrete_wavelet:
                dwt_features = self._analyze_discrete_wavelet_transforms(price_data)
                features.update(dwt_features)

            # 2. Continuous Wavelet Transform (CWT) analysis
            if self.enable_continuous_wavelet:
                cwt_features = self._analyze_continuous_wavelet_transforms(price_data)
                features.update(cwt_features)

            # 3. Wavelet Packet analysis
            if self.enable_wavelet_packet:
                packet_features = self._analyze_wavelet_packets(price_data)
                features.update(packet_features)

            # 4. Wavelet denoising
            if self.enable_denoising:
                denoising_features = self._analyze_wavelet_denoising(price_data)
                features.update(denoising_features)

            # 5. Multi-wavelet analysis
            multi_wavelet_features = self._analyze_multi_wavelet_transforms(price_data)
            features.update(multi_wavelet_features)

            # 6. Volume wavelet analysis (if available)
            if volume_data is not None and not volume_data.empty:
                volume_wavelet_features = self._analyze_volume_wavelet_transforms(volume_data)
                features.update(volume_wavelet_features)

            self.logger.info(f"✅ Wavelet transform analysis completed. Generated {len(features)} features")
            return features

        except Exception as e:
            self.logger.error(f"Error in wavelet transform analysis: {e}")
            return {}

    def _analyze_discrete_wavelet_transforms(self, price_data: pd.DataFrame) -> dict[str, float]:
        """Analyze discrete wavelet transforms using vectorized operations."""
        try:
            features = {}
            
            # Use close prices for wavelet analysis
            close_prices = price_data["close"].values
            
            # Perform DWT for different wavelet types
            for wavelet_type in self.wavelet_types[:3]:  # Use first 3 types for efficiency
                try:
                    # Perform wavelet decomposition
                    coeffs = pywt.wavedec(close_prices, wavelet_type, level=self.decomposition_level)
                    
                    # Extract features from coefficients
                    dwt_features = self._extract_dwt_features(coeffs, wavelet_type)
                    features.update(dwt_features)
                    
                except Exception as e:
                    self.logger.warning(f"Error with wavelet type {wavelet_type}: {e}")
                    continue

            return features

        except Exception as e:
            self.logger.error(f"Error in discrete wavelet transform analysis: {e}")
            return {}

    def _extract_dwt_features(self, coeffs: list, wavelet_type: str) -> dict[str, float]:
        """Extract features from DWT coefficients using vectorized operations."""
        try:
            features = {}
            
            # Energy features for each level
            for i, coeff in enumerate(coeffs):
                if len(coeff) > 0:
                    energy = np.sum(coeff ** 2)
                    features[f"{wavelet_type}_level_{i}_energy"] = energy
                    features[f"{wavelet_type}_level_{i}_energy_normalized"] = energy / len(coeff)
                    
                    # Statistical features
                    features[f"{wavelet_type}_level_{i}_mean"] = np.mean(coeff)
                    features[f"{wavelet_type}_level_{i}_std"] = np.std(coeff)
                    features[f"{wavelet_type}_level_{i}_max"] = np.max(coeff)
                    features[f"{wavelet_type}_level_{i}_min"] = np.min(coeff)
                    
                    # Entropy features
                    if energy > 0:
                        entropy = -np.sum((coeff ** 2) / energy * np.log((coeff ** 2) / energy + 1e-10))
                        features[f"{wavelet_type}_level_{i}_entropy"] = entropy

            # Cross-level features
            if len(coeffs) > 1:
                # Energy ratio between levels
                for i in range(len(coeffs) - 1):
                    if np.sum(coeffs[i] ** 2) > 0:
                        energy_ratio = np.sum(coeffs[i + 1] ** 2) / np.sum(coeffs[i] ** 2)
                        features[f"{wavelet_type}_energy_ratio_{i}_{i+1}"] = energy_ratio

            return features

        except Exception as e:
            self.logger.error(f"Error extracting DWT features: {e}")
            return {}

    def _analyze_continuous_wavelet_transforms(self, price_data: pd.DataFrame) -> dict[str, float]:
        """Analyze continuous wavelet transforms using vectorized operations."""
        try:
            features = {}
            
            # Use close prices for CWT analysis
            close_prices = price_data["close"].values
            
            # Perform CWT for different scales
            scales = np.arange(1, 32, 2)  # Use fewer scales for efficiency
            
            for wavelet_type in ["morl", "cmor1.5-1.0"]:  # Morlet and complex Morlet wavelets
                try:
                    # Perform continuous wavelet transform
                    coeffs, freqs = pywt.cwt(close_prices, scales, wavelet_type)
                    
                    # Extract CWT features
                    cwt_features = self._extract_cwt_features(coeffs, freqs, wavelet_type)
                    features.update(cwt_features)
                    
                except Exception as e:
                    self.logger.warning(f"Error with CWT wavelet type {wavelet_type}: {e}")
                    continue

            return features

        except Exception as e:
            self.logger.error(f"Error in continuous wavelet transform analysis: {e}")
            return {}

    def _extract_cwt_features(self, coeffs: np.ndarray, freqs: np.ndarray, wavelet_type: str) -> dict[str, float]:
        """Extract features from CWT coefficients using vectorized operations."""
        try:
            features = {}
            
            # Energy features
            energy = np.sum(np.abs(coeffs) ** 2, axis=1)
            features[f"{wavelet_type}_total_energy"] = np.sum(energy)
            features[f"{wavelet_type}_max_energy"] = np.max(energy)
            features[f"{wavelet_type}_min_energy"] = np.min(energy)
            features[f"{wavelet_type}_energy_std"] = np.std(energy)
            
            # Frequency features
            features[f"{wavelet_type}_dominant_freq"] = freqs[np.argmax(energy)]
            features[f"{wavelet_type}_freq_range"] = np.max(freqs) - np.min(freqs)
            
            # Statistical features
            features[f"{wavelet_type}_coeff_mean"] = np.mean(np.abs(coeffs))
            features[f"{wavelet_type}_coeff_std"] = np.std(np.abs(coeffs))
            features[f"{wavelet_type}_coeff_max"] = np.max(np.abs(coeffs))
            features[f"{wavelet_type}_coeff_min"] = np.min(np.abs(coeffs))
            
            # Entropy features
            total_energy = np.sum(np.abs(coeffs) ** 2)
            if total_energy > 0:
                entropy = -np.sum((np.abs(coeffs) ** 2) / total_energy * 
                                np.log((np.abs(coeffs) ** 2) / total_energy + 1e-10))
                features[f"{wavelet_type}_entropy"] = entropy

            return features

        except Exception as e:
            self.logger.error(f"Error extracting CWT features: {e}")
            return {}

    def _analyze_wavelet_packets(self, price_data: pd.DataFrame) -> dict[str, float]:
        """Analyze wavelet packets using vectorized operations."""
        try:
            features = {}
            
            # Use close prices for wavelet packet analysis
            close_prices = price_data["close"].values
            
            # Perform wavelet packet decomposition
            for wavelet_type in ["db4", "sym4"]:  # Use common wavelet types
                try:
                    # Create wavelet packet tree
                    wp = pywt.WaveletPacket(close_prices, wavelet_type, mode='symmetric')
                    
                    # Extract packet features
                    packet_features = self._extract_wavelet_packet_features(wp, wavelet_type)
                    features.update(packet_features)
                    
                except Exception as e:
                    self.logger.warning(f"Error with wavelet packet type {wavelet_type}: {e}")
                    continue

            return features

        except Exception as e:
            self.logger.error(f"Error in wavelet packet analysis: {e}")
            return {}

    def _extract_wavelet_packet_features(self, wp: pywt.WaveletPacket, wavelet_type: str) -> dict[str, float]:
        """Extract features from wavelet packets using vectorized operations."""
        try:
            features = {}
            
            # Get packet coefficients
            packets = []
            for node in wp.get_level(3):  # Level 3 decomposition
                packets.append(node.data)
            
            if packets:
                # Energy features
                energies = [np.sum(packet ** 2) for packet in packets]
                features[f"{wavelet_type}_packet_total_energy"] = np.sum(energies)
                features[f"{wavelet_type}_packet_max_energy"] = np.max(energies)
                features[f"{wavelet_type}_packet_min_energy"] = np.min(energies)
                features[f"{wavelet_type}_packet_energy_std"] = np.std(energies)
                
                # Statistical features
                all_coeffs = np.concatenate(packets)
                features[f"{wavelet_type}_packet_coeff_mean"] = np.mean(all_coeffs)
                features[f"{wavelet_type}_packet_coeff_std"] = np.std(all_coeffs)
                features[f"{wavelet_type}_packet_coeff_max"] = np.max(all_coeffs)
                features[f"{wavelet_type}_packet_coeff_min"] = np.min(all_coeffs)
                
                # Entropy features
                total_energy = np.sum(energies)
                if total_energy > 0:
                    entropy = -np.sum(np.array(energies) / total_energy * 
                                    np.log(np.array(energies) / total_energy + 1e-10))
                    features[f"{wavelet_type}_packet_entropy"] = entropy

            return features

        except Exception as e:
            self.logger.error(f"Error extracting wavelet packet features: {e}")
            return {}

    def _analyze_wavelet_denoising(self, price_data: pd.DataFrame) -> dict[str, float]:
        """Analyze wavelet denoising using vectorized operations."""
        try:
            features = {}
            
            # Use close prices for denoising analysis
            close_prices = price_data["close"].values
            
            # Perform wavelet denoising
            for wavelet_type in ["db4", "sym4"]:
                try:
                    # Denoise signal
                    denoised = pywt.threshold(close_prices, np.std(close_prices) * 0.1, mode='soft')
                    
                    # Extract denoising features
                    denoising_features = self._extract_denoising_features(close_prices, denoised, wavelet_type)
                    features.update(denoising_features)
                    
                except Exception as e:
                    self.logger.warning(f"Error with denoising wavelet type {wavelet_type}: {e}")
                    continue

            return features

        except Exception as e:
            self.logger.error(f"Error in wavelet denoising analysis: {e}")
            return {}

    def _extract_denoising_features(self, original: np.ndarray, denoised: np.ndarray, wavelet_type: str) -> dict[str, float]:
        """Extract features from denoising analysis using vectorized operations."""
        try:
            features = {}
            
            # Noise estimation
            noise = original - denoised
            features[f"{wavelet_type}_noise_std"] = np.std(noise)
            features[f"{wavelet_type}_noise_mean"] = np.mean(noise)
            features[f"{wavelet_type}_noise_energy"] = np.sum(noise ** 2)
            
            # Signal quality metrics
            signal_energy = np.sum(denoised ** 2)
            total_energy = np.sum(original ** 2)
            if total_energy > 0:
                features[f"{wavelet_type}_signal_to_noise_ratio"] = signal_energy / np.sum(noise ** 2)
                features[f"{wavelet_type}_signal_energy_ratio"] = signal_energy / total_energy
            
            # Correlation between original and denoised
            correlation = np.corrcoef(original, denoised)[0, 1]
            features[f"{wavelet_type}_denoising_correlation"] = correlation if not np.isnan(correlation) else 0.0

            return features

        except Exception as e:
            self.logger.error(f"Error extracting denoising features: {e}")
            return {}

    def _analyze_multi_wavelet_transforms(self, price_data: pd.DataFrame) -> dict[str, float]:
        """Analyze multiple wavelet transforms for comprehensive feature extraction."""
        try:
            features = {}
            
            # Use different price series for wavelet analysis
            price_series = {
                "close": price_data["close"].values,
                "returns": price_data["close"].pct_change().dropna().values,
                "log_returns": np.log(price_data["close"] / price_data["close"].shift(1)).dropna().values,
            }
            
            for series_name, series_data in price_series.items():
                if len(series_data) > 0:
                    # Perform DWT for each series
                    for wavelet_type in ["db4", "sym4"]:
                        try:
                            coeffs = pywt.wavedec(series_data, wavelet_type, level=3)
                            
                            # Extract multi-series features
                            multi_features = self._extract_multi_series_features(coeffs, wavelet_type, series_name)
                            features.update(multi_features)
                            
                        except Exception as e:
                            self.logger.warning(f"Error with multi-wavelet {series_name} {wavelet_type}: {e}")
                            continue

            return features

        except Exception as e:
            self.logger.error(f"Error in multi-wavelet transform analysis: {e}")
            return {}

    def _extract_multi_series_features(self, coeffs: list, wavelet_type: str, series_name: str) -> dict[str, float]:
        """Extract features from multiple series wavelet analysis using vectorized operations."""
        try:
            features = {}
            
            # Energy distribution features
            energies = [np.sum(coeff ** 2) for coeff in coeffs]
            total_energy = np.sum(energies)
            
            if total_energy > 0:
                # Energy distribution
                for i, energy in enumerate(energies):
                    features[f"{wavelet_type}_{series_name}_level_{i}_energy_ratio"] = energy / total_energy
                
                # Energy concentration
                features[f"{wavelet_type}_{series_name}_energy_concentration"] = np.max(energies) / total_energy
                
                # Energy spread
                features[f"{wavelet_type}_{series_name}_energy_spread"] = np.std(energies) / np.mean(energies) if np.mean(energies) > 0 else 0.0
            
            # Statistical features across levels
            all_coeffs = np.concatenate(coeffs)
            features[f"{wavelet_type}_{series_name}_total_coeff_mean"] = np.mean(all_coeffs)
            features[f"{wavelet_type}_{series_name}_total_coeff_std"] = np.std(all_coeffs)
            features[f"{wavelet_type}_{series_name}_total_coeff_kurtosis"] = self._calculate_kurtosis(all_coeffs)
            features[f"{wavelet_type}_{series_name}_total_coeff_skewness"] = self._calculate_skewness(all_coeffs)

            return features

        except Exception as e:
            self.logger.error(f"Error extracting multi-series features: {e}")
            return {}

    def _analyze_volume_wavelet_transforms(self, volume_data: pd.DataFrame) -> dict[str, float]:
        """Analyze wavelet transforms for volume data using vectorized operations."""
        try:
            features = {}
            
            if "volume" in volume_data.columns:
                volume_series = volume_data["volume"].values
                
                # Perform DWT on volume data
                for wavelet_type in ["db4", "sym4"]:
                    try:
                        coeffs = pywt.wavedec(volume_series, wavelet_type, level=3)
                        
                        # Extract volume-specific features
                        volume_features = self._extract_volume_wavelet_features(coeffs, wavelet_type)
                        features.update(volume_features)
                        
                    except Exception as e:
                        self.logger.warning(f"Error with volume wavelet {wavelet_type}: {e}")
                        continue

            return features

        except Exception as e:
            self.logger.error(f"Error in volume wavelet transform analysis: {e}")
            return {}

    def _extract_volume_wavelet_features(self, coeffs: list, wavelet_type: str) -> dict[str, float]:
        """Extract volume-specific wavelet features using vectorized operations."""
        try:
            features = {}
            
            # Volume energy features
            energies = [np.sum(coeff ** 2) for coeff in coeffs]
            total_energy = np.sum(energies)
            
            if total_energy > 0:
                features[f"{wavelet_type}_volume_total_energy"] = total_energy
                features[f"{wavelet_type}_volume_max_energy"] = np.max(energies)
                features[f"{wavelet_type}_volume_min_energy"] = np.min(energies)
                features[f"{wavelet_type}_volume_energy_std"] = np.std(energies)
                
                # Volume energy distribution
                for i, energy in enumerate(energies):
                    features[f"{wavelet_type}_volume_level_{i}_energy_ratio"] = energy / total_energy
            
            # Volume statistical features
            all_coeffs = np.concatenate(coeffs)
            features[f"{wavelet_type}_volume_coeff_mean"] = np.mean(all_coeffs)
            features[f"{wavelet_type}_volume_coeff_std"] = np.std(all_coeffs)
            features[f"{wavelet_type}_volume_coeff_max"] = np.max(all_coeffs)
            features[f"{wavelet_type}_volume_coeff_min"] = np.min(all_coeffs)

            return features

        except Exception as e:
            self.logger.error(f"Error extracting volume wavelet features: {e}")
            return {}

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis using vectorized operations."""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std > 0:
                kurtosis = np.mean(((data - mean) / std) ** 4) - 3
                return kurtosis
            return 0.0
        except Exception:
            return 0.0

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness using vectorized operations."""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std > 0:
                skewness = np.mean(((data - mean) / std) ** 3)
                return skewness
            return 0.0
        except Exception:
            return 0.0
