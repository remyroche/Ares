#!/usr/bin/env python3
"""Exit Strategy Feature Engineering System.

This module provides comprehensive exit strategy features that calculate the likelihood
of price action reversal once a position is open. This enables the Tactician to make
informed decisions about when to close positions based on confidence levels.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

# Import essential decorators
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger

# Import Numba for performance optimization
try:
    import numba
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    jit = lambda *args, **kwargs: lambda func: func


class ExitStrategyFeatureEngineering:
    """Exit strategy feature engineering system for position management.
    
    This class calculates the likelihood of price action reversal once a position
    is open, enabling the Tactician to make informed exit decisions based on
    confidence levels.
    """

    def __init__(
        self,
        price_column: str = "close",
        volume_column: str = "volume",
        position_column: str = "position",  # 1 for LONG, -1 for SHORT, 0 for no position
        profit_column: str = "potential_profit_pct",
        confidence_threshold: float = 0.6,
        use_numba: bool = True,
        memory_efficient: bool = True,
    ) -> None:
        """Initialize the exit strategy feature engineering system.
        
        Args:
            price_column: Name of the price column
            volume_column: Name of the volume column
            position_column: Name of the position column (1=LONG, -1=SHORT, 0=none)
            profit_column: Name of the profit percentage column
            confidence_threshold: Default confidence threshold for exit decisions
            use_numba: Whether to use Numba acceleration
            memory_efficient: Whether to use memory-efficient operations
        """
        self.price_column = price_column
        self.volume_column = volume_column
        self.position_column = position_column
        self.profit_column = profit_column
        self.confidence_threshold = confidence_threshold
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.memory_efficient = memory_efficient
        
        # Initialize logger
        self.logger = system_logger.getChild("System.ExitStrategyFeatureEngineering")
        
        # Feature configuration
        self.feature_config = {
            "momentum_reversal": True,
            "volatility_reversal": True,
            "volume_reversal": True,
            "support_resistance": True,
            "trend_strength": True,
            "profit_decay": True,
            "time_decay": True,
            "market_regime": True,
        }
        
        # Performance tracking
        self.performance_metrics = {
            "total_features_generated": 0,
            "processing_time": 0.0,
            "memory_usage": 0.0,
        }
        
        self.logger.info("🔧 Exit strategy feature engineering system initialized")
        if self.use_numba:
            self.logger.info("🚀 Using Numba acceleration")
        else:
            self.logger.info("🐍 Using Python vectorized operations")

    @handle_errors(
        exceptions=(ValueError, TypeError, MemoryError),
        default_return=pd.DataFrame(),
        context="exit_strategy_feature_engineering.apply_all_features"
    )
    def apply_all_features(
        self,
        data: pd.DataFrame,
        feature_categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Apply all exit strategy feature engineering categories.
        
        Args:
            data: Input DataFrame with price, volume, position, and profit data
            feature_categories: Specific feature categories to apply
            
        Returns:
            DataFrame with all exit strategy features added
        """
        start_time = time.time()
        
        # Generate unique correlation ID for tracking
        import uuid
        correlation_id = str(uuid.uuid4())[:8]
        
        self.logger.info(f"➡️ ExitStrategy.apply_all start {correlation_id}")
        self.logger.info(f"🚀 Applying exit strategy feature engineering {correlation_id}")
        self.logger.info(f"   - Input shape: {data.shape} {correlation_id}")
        
        # Determine which feature categories to apply
        if feature_categories is None:
            feature_categories = list(self.feature_config.keys())
        
        self.logger.info(f"   - Feature categories: {feature_categories} {correlation_id}")
        
        # Validate input data
        if data.empty:
            self.logger.error(f"❌ Input data is empty {correlation_id}")
            return data
        
        required_columns = [self.price_column, self.position_column]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f"❌ Missing required columns: {missing_columns} {correlation_id}")
            return data
        
        # Apply each feature category
        result_data = data.copy()
        
        if "momentum_reversal" in feature_categories:
            result_data = self._apply_momentum_reversal_features(result_data)
            self.logger.info(f"   ✅ Applied momentum_reversal features {correlation_id}")
        
        if "volatility_reversal" in feature_categories:
            result_data = self._apply_volatility_reversal_features(result_data)
            self.logger.info(f"   ✅ Applied volatility_reversal features {correlation_id}")
        
        if "volume_reversal" in feature_categories:
            result_data = self._apply_volume_reversal_features(result_data)
            self.logger.info(f"   ✅ Applied volume_reversal features {correlation_id}")
        
        if "support_resistance" in feature_categories:
            result_data = self._apply_support_resistance_features(result_data)
            self.logger.info(f"   ✅ Applied support_resistance features {correlation_id}")
        
        if "trend_strength" in feature_categories:
            result_data = self._apply_trend_strength_features(result_data)
            self.logger.info(f"   ✅ Applied trend_strength features {correlation_id}")
        
        if "profit_decay" in feature_categories:
            result_data = self._apply_profit_decay_features(result_data)
            self.logger.info(f"   ✅ Applied profit_decay features {correlation_id}")
        
        if "time_decay" in feature_categories:
            result_data = self._apply_time_decay_features(result_data)
            self.logger.info(f"   ✅ Applied time_decay features {correlation_id}")
        
        if "market_regime" in feature_categories:
            result_data = self._apply_market_regime_features(result_data)
            self.logger.info(f"   ✅ Applied market_regime features {correlation_id}")
        
        # Calculate performance metrics
        processing_time = time.time() - start_time
        features_added = len(result_data.columns) - len(data.columns)
        
        self.logger.info(f"✅ Exit strategy feature engineering completed {correlation_id}")
        self.logger.info(f"   - Output shape: {result_data.shape} {correlation_id}")
        self.logger.info(f"   - Features added: {features_added} {correlation_id}")
        self.logger.info(f"   - Total features: {len(result_data.columns)} {correlation_id}")
        
        # Update performance metrics
        self.performance_metrics.update({
            "total_features_generated": features_added,
            "processing_time": processing_time,
            "memory_usage": result_data.memory_usage(deep=True).sum() / 1024**3  # GB
        })
        
        self.logger.info(f"✅ ExitStrategy.apply_all done {correlation_id}")
        
        return result_data

    @handle_errors(
        exceptions=(ValueError, TypeError),
        default_return=pd.DataFrame(),
        context="momentum_reversal_features"
    )
    def _apply_momentum_reversal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply momentum reversal features.
        
        Features: momentum_strength, momentum_decay, reversal_probability
        """
        price = data[self.price_column].values
        position = data[self.position_column].values
        
        # Calculate momentum indicators
        windows = [5, 10, 20, 50]
        for window in windows:
            # Price momentum
            momentum = pd.Series(price).pct_change(window).fillna(0)
            data[f"momentum_{window}"] = momentum
            
            # Momentum strength (how strong the current momentum is)
            momentum_ma = momentum.rolling(window=window, min_periods=1).mean()
            momentum_std = momentum.rolling(window=window, min_periods=1).std()
            momentum_strength = np.where(
                momentum_std > 0,
                (momentum - momentum_ma) / momentum_std,
                0.0
            )
            data[f"momentum_strength_{window}"] = momentum_strength
            
            # Momentum decay (how momentum is weakening)
            momentum_decay = momentum.rolling(window=window, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
            data[f"momentum_decay_{window}"] = momentum_decay
        
        # Reversal probability based on momentum
        for window in [10, 20]:
            momentum = data[f"momentum_{window}"]
            momentum_strength = data[f"momentum_strength_{window}"]
            
            # Calculate reversal probability based on momentum weakening
            reversal_prob = np.where(
                position != 0,  # Only for open positions
                np.maximum(0, -momentum_strength * np.sign(position)),
                0.0
            )
            data[f"momentum_reversal_prob_{window}"] = reversal_prob
        
        return data

    @handle_errors(
        exceptions=(ValueError, TypeError),
        default_return=pd.DataFrame(),
        context="volatility_reversal_features"
    )
    def _apply_volatility_reversal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply volatility reversal features.
        
        Features: volatility_spike, volatility_regime, reversal_volatility
        """
        price = data[self.price_column].values
        position = data[self.position_column].values
        
        # Calculate volatility indicators
        windows = [10, 20, 50]
        for window in windows:
            # Price volatility
            returns = pd.Series(price).pct_change().fillna(0)
            volatility = returns.rolling(window=window, min_periods=1).std()
            data[f"volatility_{window}"] = volatility
            
            # Volatility spike detection
            vol_ma = volatility.rolling(window=window*2, min_periods=1).mean()
            vol_std = volatility.rolling(window=window*2, min_periods=1).std()
            volatility_spike = np.where(
                vol_std > 0,
                (volatility - vol_ma) / vol_std,
                0.0
            )
            data[f"volatility_spike_{window}"] = volatility_spike
        
        # Volatility regime classification
        short_vol = data["volatility_10"]
        long_vol = data["volatility_50"]
        
        # High volatility regime (potential reversal)
        high_vol_regime = np.where(
            short_vol > long_vol * 1.5,
            1.0,
            0.0
        )
        data["high_volatility_regime"] = high_vol_regime
        
        # Reversal probability based on volatility
        for window in [10, 20]:
            volatility = data[f"volatility_{window}"]
            vol_spike = data[f"volatility_spike_{window}"]
            
            # High volatility increases reversal probability
            reversal_prob = np.where(
                position != 0,
                np.minimum(1.0, vol_spike * 0.1 + high_vol_regime * 0.2),
                0.0
            )
            data[f"volatility_reversal_prob_{window}"] = reversal_prob
        
        return data

    @handle_errors(
        exceptions=(ValueError, TypeError),
        default_return=pd.DataFrame(),
        context="volume_reversal_features"
    )
    def _apply_volume_reversal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply volume reversal features.
        
        Features: volume_surge, volume_divergence, reversal_volume
        """
        if self.volume_column not in data.columns:
            self.logger.warning(f"Volume column '{self.volume_column}' not found, skipping volume features")
            return data
        
        volume = data[self.volume_column].values
        price = data[self.price_column].values
        position = data[self.position_column].values
        
        # Calculate volume indicators
        windows = [10, 20, 50]
        for window in windows:
            # Volume moving average
            volume_ma = pd.Series(volume).rolling(window=window, min_periods=1).mean()
            data[f"volume_ma_{window}"] = volume_ma
            
            # Volume surge detection
            volume_std = pd.Series(volume).rolling(window=window*2, min_periods=1).std()
            volume_surge = np.where(
                volume_std > 0,
                (volume - volume_ma) / volume_std,
                0.0
            )
            data[f"volume_surge_{window}"] = volume_surge
        
        # Volume-price divergence
        for window in [10, 20]:
            price_momentum = pd.Series(price).pct_change(window).fillna(0)
            volume_momentum = pd.Series(volume).pct_change(window).fillna(0)
            
            # Divergence: price up but volume down (bearish)
            price_volume_divergence = np.where(
                (price_momentum > 0) & (volume_momentum < 0),
                -price_momentum * abs(volume_momentum),
                0.0
            )
            data[f"price_volume_divergence_{window}"] = price_volume_divergence
        
        # Reversal probability based on volume
        for window in [10, 20]:
            volume_surge = data[f"volume_surge_{window}"]
            divergence = data[f"price_volume_divergence_{window}"]
            
            # High volume surge and divergence increase reversal probability
            reversal_prob = np.where(
                position != 0,
                np.minimum(1.0, 
                    np.maximum(0, volume_surge * 0.1) + 
                    np.maximum(0, divergence * 0.2)
                ),
                0.0
            )
            data[f"volume_reversal_prob_{window}"] = reversal_prob
        
        return data

    @handle_errors(
        exceptions=(ValueError, TypeError),
        default_return=pd.DataFrame(),
        context="support_resistance_features"
    )
    def _apply_support_resistance_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply support/resistance features.
        
        Features: distance_to_support, distance_to_resistance, sr_reversal_prob
        """
        price = data[self.price_column].values
        position = data[self.position_column].values
        
        # Calculate support and resistance levels
        windows = [20, 50, 100]
        for window in windows:
            # Rolling high and low
            rolling_high = pd.Series(price).rolling(window=window, min_periods=1).max()
            rolling_low = pd.Series(price).rolling(window=window, min_periods=1).min()
            
            # Distance to support and resistance
            distance_to_resistance = (rolling_high - price) / price
            distance_to_support = (price - rolling_low) / price
            
            data[f"distance_to_resistance_{window}"] = distance_to_resistance
            data[f"distance_to_support_{window}"] = distance_to_support
        
        # Support/resistance reversal probability
        for window in [20, 50]:
            distance_to_resistance = data[f"distance_to_resistance_{window}"]
            distance_to_support = data[f"distance_to_support_{window}"]
            
            # Close to resistance (LONG positions at risk)
            resistance_reversal = np.where(
                (position == 1) & (distance_to_resistance < 0.01),  # Within 1% of resistance
                0.8,  # High reversal probability
                0.0
            )
            
            # Close to support (SHORT positions at risk)
            support_reversal = np.where(
                (position == -1) & (distance_to_support < 0.01),  # Within 1% of support
                0.8,  # High reversal probability
                0.0
            )
            
            reversal_prob = np.maximum(resistance_reversal, support_reversal)
            data[f"sr_reversal_prob_{window}"] = reversal_prob
        
        return data

    @handle_errors(
        exceptions=(ValueError, TypeError),
        default_return=pd.DataFrame(),
        context="trend_strength_features"
    )
    def _apply_trend_strength_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply trend strength features.
        
        Features: trend_strength, trend_consistency, trend_reversal_prob
        """
        price = data[self.price_column].values
        position = data[self.position_column].values
        
        # Calculate trend indicators
        windows = [10, 20, 50]
        for window in windows:
            # Linear regression slope
            def calc_trend_slope(x):
                if len(x) < 2:
                    return 0.0
                return np.polyfit(range(len(x)), x, 1)[0]
            
            trend_slope = pd.Series(price).rolling(window=window, min_periods=1).apply(calc_trend_slope)
            data[f"trend_slope_{window}"] = trend_slope
            
            # Trend strength (R-squared of linear fit)
            def calc_trend_strength(x):
                if len(x) < 2:
                    return 0.0
                y = np.array(x)
                x_vals = np.arange(len(x))
                correlation_matrix = np.corrcoef(x_vals, y)
                correlation = correlation_matrix[0, 1]
                return correlation ** 2 if not np.isnan(correlation) else 0.0
            
            trend_strength = pd.Series(price).rolling(window=window, min_periods=1).apply(calc_trend_strength)
            data[f"trend_strength_{window}"] = trend_strength
        
        # Trend reversal probability
        for window in [20, 50]:
            trend_slope = data[f"trend_slope_{window}"]
            trend_strength = data[f"trend_strength_{window}"]
            
            # Weak trend or trend reversal increases exit probability
            trend_reversal_prob = np.where(
                position != 0,
                np.where(
                    (trend_slope * position < 0) | (trend_strength < 0.3),  # Trend against position or weak trend
                    0.6,  # Medium reversal probability
                    0.1   # Low reversal probability
                ),
                0.0
            )
            data[f"trend_reversal_prob_{window}"] = trend_reversal_prob
        
        return data

    @handle_errors(
        exceptions=(ValueError, TypeError),
        default_return=pd.DataFrame(),
        context="profit_decay_features"
    )
    def _apply_profit_decay_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply profit decay features.
        
        Features: profit_decay_rate, profit_peak, profit_reversal_prob
        """
        if self.profit_column not in data.columns:
            self.logger.warning(f"Profit column '{self.profit_column}' not found, skipping profit decay features")
            return data
        
        profit = data[self.profit_column].values
        position = data[self.position_column].values
        
        # Calculate profit decay indicators
        windows = [5, 10, 20]
        for window in windows:
            # Profit decay rate (how quickly profit is decreasing)
            profit_ma = pd.Series(profit).rolling(window=window, min_periods=1).mean()
            profit_decay = profit - profit_ma
            data[f"profit_decay_{window}"] = profit_decay
            
            # Profit peak detection
            profit_peak = pd.Series(profit).rolling(window=window, min_periods=1).max()
            distance_from_peak = (profit_peak - profit) / (profit_peak + 1e-8)
            data[f"distance_from_peak_{window}"] = distance_from_peak
        
        # Profit reversal probability
        for window in [10, 20]:
            profit_decay = data[f"profit_decay_{window}"]
            distance_from_peak = data[f"distance_from_peak_{window}"]
            
            # Significant profit decay increases exit probability
            profit_reversal_prob = np.where(
                position != 0,
                np.minimum(1.0, 
                    np.maximum(0, -profit_decay * 10) +  # Decay factor
                    np.maximum(0, distance_from_peak * 0.5)  # Distance from peak
                ),
                0.0
            )
            data[f"profit_reversal_prob_{window}"] = profit_reversal_prob
        
        return data

    @handle_errors(
        exceptions=(ValueError, TypeError),
        default_return=pd.DataFrame(),
        context="time_decay_features"
    )
    def _apply_time_decay_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply time decay features.
        
        Features: position_duration, time_decay_factor, time_reversal_prob
        """
        position = data[self.position_column].values
        
        # Calculate position duration
        position_duration = np.zeros(len(position))
        current_duration = 0
        
        for i in range(len(position)):
            if position[i] != 0:  # Position is open
                current_duration += 1
            else:  # No position
                current_duration = 0
            position_duration[i] = current_duration
        
        data["position_duration"] = position_duration
        
        # Time decay factor (exponential decay)
        time_decay_factor = np.where(
            position_duration > 0,
            1.0 - np.exp(-position_duration / 50.0),  # 50 periods half-life
            0.0
        )
        data["time_decay_factor"] = time_decay_factor
        
        # Time-based reversal probability
        time_reversal_prob = np.where(
            position != 0,
            np.minimum(1.0, time_decay_factor * 0.3),  # Max 30% probability from time decay
            0.0
        )
        data["time_reversal_prob"] = time_reversal_prob
        
        return data

    @handle_errors(
        exceptions=(ValueError, TypeError),
        default_return=pd.DataFrame(),
        context="market_regime_features"
    )
    def _apply_market_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply market regime features.
        
        Features: market_regime, regime_stability, regime_reversal_prob
        """
        price = data[self.price_column].values
        position = data[self.position_column].values
        
        # Calculate market regime indicators
        windows = [20, 50]
        for window in windows:
            # Volatility regime
            returns = pd.Series(price).pct_change().fillna(0)
            volatility = returns.rolling(window=window, min_periods=1).std()
            
            # Trend regime
            trend_slope = data.get(f"trend_slope_{window}", pd.Series(0, index=data.index))
            
            # Market regime classification
            regime = np.where(
                volatility > volatility.rolling(window=window*2, min_periods=1).mean(),
                np.where(trend_slope > 0, "trending_high_vol", "reversing_high_vol"),
                np.where(trend_slope > 0, "trending_low_vol", "reversing_low_vol")
            )
            
            # Convert to numeric for easier processing
            regime_numeric = np.where(
                regime == "trending_low_vol", 0,
                np.where(regime == "trending_high_vol", 1,
                np.where(regime == "reversing_low_vol", 2, 3))
            )
            
            data[f"market_regime_{window}"] = regime_numeric
        
        # Regime stability (how long regime has been stable)
        for window in [20, 50]:
            regime = data[f"market_regime_{window}"]
            
            # Calculate regime stability
            regime_stability = np.zeros(len(regime))
            current_stability = 0
            
            for i in range(1, len(regime)):
                if regime[i] == regime[i-1]:
                    current_stability += 1
                else:
                    current_stability = 0
                regime_stability[i] = current_stability
            
            data[f"regime_stability_{window}"] = regime_stability
        
        # Regime-based reversal probability
        for window in [20, 50]:
            regime = data[f"market_regime_{window}"]
            stability = data[f"regime_stability_{window}"]
            
            # High volatility regimes and regime changes increase reversal probability
            regime_reversal_prob = np.where(
                position != 0,
                np.where(
                    (regime == 2) | (regime == 3) | (stability < 5),  # Reversing regimes or unstable
                    0.4,  # Medium reversal probability
                    0.1   # Low reversal probability
                ),
                0.0
            )
            data[f"regime_reversal_prob_{window}"] = regime_reversal_prob
        
        return data

    def calculate_exit_confidence(
        self,
        data: pd.DataFrame,
        position_type: str = "auto"  # "auto", "long", "short"
    ) -> pd.Series:
        """Calculate overall exit confidence for positions.
        
        Args:
            data: DataFrame with exit strategy features
            position_type: Type of position to calculate confidence for
            
        Returns:
            Series with exit confidence scores (0-1, higher = more likely to exit)
        """
        try:
            # Get all reversal probability features
            reversal_features = [col for col in data.columns if "reversal_prob" in col]
            
            if not reversal_features:
                self.logger.warning("No reversal probability features found")
                return pd.Series(0.0, index=data.index)
            
            # Calculate weighted average of reversal probabilities
            weights = {
                "momentum_reversal_prob": 0.25,
                "volatility_reversal_prob": 0.20,
                "volume_reversal_prob": 0.15,
                "sr_reversal_prob": 0.15,
                "trend_reversal_prob": 0.10,
                "profit_reversal_prob": 0.10,
                "time_reversal_prob": 0.03,
                "regime_reversal_prob": 0.02,
            }
            
            # Calculate weighted confidence
            confidence = pd.Series(0.0, index=data.index)
            total_weight = 0.0
            
            for feature in reversal_features:
                for weight_key, weight in weights.items():
                    if weight_key in feature:
                        if feature in data.columns:
                            confidence += data[feature] * weight
                            total_weight += weight
                        break
            
            # Normalize by total weight
            if total_weight > 0:
                confidence = confidence / total_weight
            
            # Ensure confidence is between 0 and 1
            confidence = np.clip(confidence, 0.0, 1.0)
            
            # Filter by position type if specified
            if position_type != "auto":
                position = data[self.position_column]
                if position_type == "long":
                    confidence = np.where(position == 1, confidence, 0.0)
                elif position_type == "short":
                    confidence = np.where(position == -1, confidence, 0.0)
            
            return confidence
            
        except Exception as e:
            self.logger.exception(f"Error calculating exit confidence: {e}")
            return pd.Series(0.0, index=data.index)

    def get_exit_recommendations(
        self,
        data: pd.DataFrame,
        confidence_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """Get exit recommendations based on confidence threshold.
        
        Args:
            data: DataFrame with exit strategy features
            confidence_threshold: Confidence threshold for exit (default: self.confidence_threshold)
            
        Returns:
            DataFrame with exit recommendations
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        # Calculate exit confidence
        exit_confidence = self.calculate_exit_confidence(data)
        
        # Create recommendations
        recommendations = pd.DataFrame({
            "exit_confidence": exit_confidence,
            "should_exit": exit_confidence >= confidence_threshold,
            "position": data[self.position_column],
            "exit_reason": self._get_exit_reasons(data, exit_confidence)
        })
        
        return recommendations

    def _get_exit_reasons(self, data: pd.DataFrame, exit_confidence: pd.Series) -> pd.Series:
        """Get primary reasons for exit recommendations.
        
        Args:
            data: DataFrame with exit strategy features
            exit_confidence: Exit confidence scores
            
        Returns:
            Series with exit reasons
        """
        reasons = []
        
        for i in range(len(data)):
            if exit_confidence.iloc[i] >= self.confidence_threshold:
                # Find the highest contributing factor
                reversal_features = [col for col in data.columns if "reversal_prob" in col]
                max_prob = 0.0
                max_reason = "general_reversal"
                
                for feature in reversal_features:
                    if feature in data.columns:
                        prob = data[feature].iloc[i]
                        if prob > max_prob:
                            max_prob = prob
                            max_reason = feature.replace("_reversal_prob", "_reversal")
                
                reasons.append(max_reason)
            else:
                reasons.append("hold")
        
        return pd.Series(reasons, index=data.index)

    def get_feature_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of exit strategy features.
        
        Args:
            data: DataFrame with exit strategy features
            
        Returns:
            Dictionary with feature summary information
        """
        exit_features = [col for col in data.columns if any(x in col for x in [
            "reversal_prob", "momentum", "volatility", "volume", "support", 
            "resistance", "trend", "profit_decay", "time_decay", "market_regime"
        ])]
        
        # Categorize features
        feature_categories = {
            "momentum_reversal": [],
            "volatility_reversal": [],
            "volume_reversal": [],
            "support_resistance": [],
            "trend_strength": [],
            "profit_decay": [],
            "time_decay": [],
            "market_regime": []
        }
        
        for feature in exit_features:
            if "momentum" in feature:
                feature_categories["momentum_reversal"].append(feature)
            elif "volatility" in feature:
                feature_categories["volatility_reversal"].append(feature)
            elif "volume" in feature:
                feature_categories["volume_reversal"].append(feature)
            elif "support" in feature or "resistance" in feature or "sr_" in feature:
                feature_categories["support_resistance"].append(feature)
            elif "trend" in feature:
                feature_categories["trend_strength"].append(feature)
            elif "profit" in feature:
                feature_categories["profit_decay"].append(feature)
            elif "time" in feature or "duration" in feature:
                feature_categories["time_decay"].append(feature)
            elif "regime" in feature:
                feature_categories["market_regime"].append(feature)
        
        return {
            "total_features": len(exit_features),
            "feature_categories": feature_categories,
            "performance_metrics": self.performance_metrics
        }


if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pandas as pd
    
    # Create sample data
    dates = pd.date_range("2024-01-01", periods=1000, freq="1min")
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 110, 1000),
        'high': np.random.uniform(105, 115, 1000),
        'low': np.random.uniform(95, 105, 1000),
        'close': np.random.uniform(100, 110, 1000),
        'volume': np.random.uniform(1000, 10000, 1000),
        'position': np.random.choice([1, -1, 0], 1000, p=[0.3, 0.3, 0.4]),
        'potential_profit_pct': np.random.uniform(-0.01, 0.01, 1000),
    }, index=dates)
    
    # Initialize exit strategy feature engineering
    exit_strategy = ExitStrategyFeatureEngineering()
    
    # Apply all features
    result = exit_strategy.apply_all_features(data)
    
    # Get exit recommendations
    recommendations = exit_strategy.get_exit_recommendations(result, confidence_threshold=0.6)
    
    # Get feature summary
    summary = exit_strategy.get_feature_summary(result)
    print(f"Generated {summary['total_features']} exit strategy features")
    
    # Show recommendations
    exit_signals = recommendations[recommendations['should_exit']]
    print(f"Exit signals: {len(exit_signals)} out of {len(recommendations)}")
    if len(exit_signals) > 0:
        print(f"Sample exit reasons: {exit_signals['exit_reason'].value_counts().head()}")