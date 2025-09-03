"""Support/Resistance Feature Extractor Module."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.decorators import handles_errors
from src.utils.logger import system_logger


class SRFeatureExtractor:
    """Extracts ML features from S/R analysis."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize feature extractor."""
        self.config = config
        self.logger = system_logger.getChild("SRFeatureExtractor")
        
        # Configuration
        self.sr_config = config.get("sr_breakout_predictor", {})
        self.lookback_periods = self.sr_config.get("sr_lookback_periods", 100)
        
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return={},
        context="extract SR features"
    )
    def extract_ml_features(
        self, 
        market_data: pd.DataFrame, 
        current_price: float,
        sr_context: dict[str, Any]
    ) -> dict[str, float]:
        """
        Extract ML features for S/R analysis.
        
        Args:
            market_data: Market data DataFrame
            current_price: Current price
            sr_context: S/R context with levels
            
        Returns:
            Dictionary of ML features
        """
        try:
            features = {}
            
            # Price features
            price_features = self._extract_price_features(market_data, current_price)
            features.update(price_features)
            
            # S/R proximity features
            sr_features = self._extract_sr_proximity_features(
                current_price, sr_context
            )
            features.update(sr_features)
            
            # Strength features
            strength_features = self._extract_strength_features(sr_context)
            features.update(strength_features)
            
            # Technical indicator features
            technical_features = self._extract_technical_features(market_data)
            features.update(technical_features)
            
            # Volume features
            volume_features = self._extract_volume_features(market_data)
            features.update(volume_features)
            
            # Momentum features
            momentum_features = self._extract_momentum_features(market_data)
            features.update(momentum_features)
            
            # Pattern features
            pattern_features = self._extract_pattern_features(
                market_data, sr_context
            )
            features.update(pattern_features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting ML features: {e}")
            return self._get_default_features()
    
    def _extract_price_features(
        self, 
        market_data: pd.DataFrame, 
        current_price: float
    ) -> dict[str, float]:
        """Extract price-based features."""
        try:
            features = {}
            
            # Price position in range
            high_20 = market_data["high"].rolling(20).max().iloc[-1]
            low_20 = market_data["low"].rolling(20).min().iloc[-1]
            if high_20 > low_20:
                price_position = (current_price - low_20) / (high_20 - low_20)
            else:
                price_position = 0.5
            features["price_position_20"] = float(price_position)
            
            # Price relative to moving averages
            sma_10 = market_data["close"].rolling(10).mean().iloc[-1]
            sma_20 = market_data["close"].rolling(20).mean().iloc[-1]
            sma_50 = market_data["close"].rolling(50).mean().iloc[-1]
            
            features["price_vs_sma10"] = float((current_price - sma_10) / sma_10)
            features["price_vs_sma20"] = float((current_price - sma_20) / sma_20)
            features["price_vs_sma50"] = float((current_price - sma_50) / sma_50)
            
            # Volatility
            returns = market_data["close"].pct_change()
            features["volatility_20"] = float(returns.rolling(20).std().iloc[-1])
            features["volatility_50"] = float(returns.rolling(50).std().iloc[-1])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting price features: {e}")
            return {}
    
    def _extract_sr_proximity_features(
        self, 
        current_price: float, 
        sr_context: dict[str, Any]
    ) -> dict[str, float]:
        """Extract S/R proximity features."""
        try:
            features = {}
            
            support_levels = sr_context.get("support", [])
            resistance_levels = sr_context.get("resistance", [])
            
            # Find nearest levels
            nearest_support = min(
                support_levels,
                key=lambda x: abs(x["price"] - current_price),
                default=None
            )
            nearest_resistance = min(
                resistance_levels,
                key=lambda x: abs(x["price"] - current_price),
                default=None
            )
            
            # Support proximity
            if nearest_support:
                support_distance = (current_price - nearest_support["price"]) / current_price
                features["support_proximity"] = float(support_distance)
                features["nearest_support_strength"] = float(nearest_support["strength"])
            else:
                features["support_proximity"] = 1.0
                features["nearest_support_strength"] = 0.0
            
            # Resistance proximity
            if nearest_resistance:
                resistance_distance = (nearest_resistance["price"] - current_price) / current_price
                features["resistance_proximity"] = float(resistance_distance)
                features["nearest_resistance_strength"] = float(nearest_resistance["strength"])
            else:
                features["resistance_proximity"] = 1.0
                features["nearest_resistance_strength"] = 0.0
            
            # S/R balance
            features["sr_balance"] = float(
                len(support_levels) / (len(support_levels) + len(resistance_levels))
                if (len(support_levels) + len(resistance_levels)) > 0 else 0.5
            )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting S/R proximity features: {e}")
            return {}
    
    def _extract_strength_features(self, sr_context: dict[str, Any]) -> dict[str, float]:
        """Extract S/R strength features."""
        try:
            features = {}
            
            support_levels = sr_context.get("support", [])
            resistance_levels = sr_context.get("resistance", [])
            
            # Average strengths
            if support_levels:
                avg_support_strength = np.mean([l["strength"] for l in support_levels])
                max_support_strength = max([l["strength"] for l in support_levels])
            else:
                avg_support_strength = 0.0
                max_support_strength = 0.0
                
            if resistance_levels:
                avg_resistance_strength = np.mean([l["strength"] for l in resistance_levels])
                max_resistance_strength = max([l["strength"] for l in resistance_levels])
            else:
                avg_resistance_strength = 0.0
                max_resistance_strength = 0.0
            
            features["avg_support_strength"] = float(avg_support_strength)
            features["max_support_strength"] = float(max_support_strength)
            features["avg_resistance_strength"] = float(avg_resistance_strength)
            features["max_resistance_strength"] = float(max_resistance_strength)
            
            # Overall strength
            all_strengths = (
                [l["strength"] for l in support_levels] +
                [l["strength"] for l in resistance_levels]
            )
            features["overall_sr_strength"] = float(
                np.mean(all_strengths) if all_strengths else 0.0
            )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting strength features: {e}")
            return {}
    
    def _extract_technical_features(self, market_data: pd.DataFrame) -> dict[str, float]:
        """Extract technical indicator features."""
        try:
            features = {}
            
            # RSI
            rsi = self._calculate_rsi(market_data["close"])
            features["rsi"] = float(rsi.iloc[-1])
            features["rsi_overbought"] = float(rsi.iloc[-1] > 70)
            features["rsi_oversold"] = float(rsi.iloc[-1] < 30)
            
            # MACD
            macd_line, signal_line, histogram = self._calculate_macd(market_data["close"])
            features["macd"] = float(macd_line.iloc[-1])
            features["macd_signal"] = float(signal_line.iloc[-1])
            features["macd_histogram"] = float(histogram.iloc[-1])
            features["macd_bullish"] = float(macd_line.iloc[-1] > signal_line.iloc[-1])
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
                market_data["close"]
            )
            current_price = market_data["close"].iloc[-1]
            bb_position = (current_price - bb_lower.iloc[-1]) / (
                bb_upper.iloc[-1] - bb_lower.iloc[-1]
            ) if bb_upper.iloc[-1] > bb_lower.iloc[-1] else 0.5
            features["bb_position"] = float(bb_position)
            features["bb_width"] = float(
                (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
            )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting technical features: {e}")
            return {}
    
    def _extract_volume_features(self, market_data: pd.DataFrame) -> dict[str, float]:
        """Extract volume-based features."""
        try:
            features = {}
            
            # Volume metrics
            volume = market_data["volume"]
            features["volume_ratio_20"] = float(
                volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]
            )
            features["volume_ratio_50"] = float(
                volume.iloc[-1] / volume.rolling(50).mean().iloc[-1]
            )
            
            # Volume trend
            volume_sma_10 = volume.rolling(10).mean()
            volume_sma_20 = volume.rolling(20).mean()
            features["volume_trend"] = float(
                (volume_sma_10.iloc[-1] - volume_sma_20.iloc[-1]) / volume_sma_20.iloc[-1]
            )
            
            # On-Balance Volume (OBV) trend
            obv = self._calculate_obv(market_data)
            obv_sma = obv.rolling(20).mean()
            features["obv_trend"] = float(
                (obv.iloc[-1] - obv_sma.iloc[-1]) / abs(obv_sma.iloc[-1])
                if obv_sma.iloc[-1] != 0 else 0
            )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting volume features: {e}")
            return {}
    
    def _extract_momentum_features(self, market_data: pd.DataFrame) -> dict[str, float]:
        """Extract momentum features."""
        try:
            features = {}
            
            close_prices = market_data["close"]
            
            # Rate of change
            features["roc_5"] = float(close_prices.pct_change(5).iloc[-1])
            features["roc_10"] = float(close_prices.pct_change(10).iloc[-1])
            features["roc_20"] = float(close_prices.pct_change(20).iloc[-1])
            
            # Momentum
            features["momentum_10"] = float(
                (close_prices.iloc[-1] - close_prices.iloc[-10]) / close_prices.iloc[-10]
            )
            features["momentum_20"] = float(
                (close_prices.iloc[-1] - close_prices.iloc[-20]) / close_prices.iloc[-20]
            )
            
            # Price acceleration
            roc_5 = close_prices.pct_change(5)
            features["acceleration"] = float(roc_5.diff().iloc[-1])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting momentum features: {e}")
            return {}
    
    def _extract_pattern_features(
        self, 
        market_data: pd.DataFrame, 
        sr_context: dict[str, Any]
    ) -> dict[str, float]:
        """Extract pattern-based features."""
        try:
            features = {}
            
            # Recent S/R tests
            support_tests = self._count_recent_tests(
                market_data, sr_context.get("support", [])
            )
            resistance_tests = self._count_recent_tests(
                market_data, sr_context.get("resistance", [])
            )
            
            features["recent_support_tests"] = float(support_tests)
            features["recent_resistance_tests"] = float(resistance_tests)
            
            # Breakout potential
            features["breakout_potential"] = self._calculate_breakout_potential(
                market_data, sr_context
            )
            
            # Consolidation score
            features["consolidation_score"] = self._calculate_consolidation_score(
                market_data
            )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting pattern features: {e}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(
        self, 
        prices: pd.Series, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> tuple:
        """Calculate MACD."""
        exp1 = prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = prices.ewm(span=slow_period, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(
        self, 
        prices: pd.Series, 
        period: int = 20, 
        num_std: float = 2
    ) -> tuple:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower
    
    def _calculate_obv(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=market_data.index, dtype=float)
        obv.iloc[0] = 0
        
        for i in range(1, len(market_data)):
            if market_data["close"].iloc[i] > market_data["close"].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + market_data["volume"].iloc[i]
            elif market_data["close"].iloc[i] < market_data["close"].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - market_data["volume"].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
                
        return obv
    
    def _count_recent_tests(
        self, 
        market_data: pd.DataFrame, 
        levels: List[Dict[str, Any]], 
        lookback: int = 20
    ) -> int:
        """Count recent tests of S/R levels."""
        if not levels:
            return 0
            
        count = 0
        recent_prices = market_data["close"].iloc[-lookback:]
        
        for level in levels:
            level_price = level["price"]
            # Check if price came within 0.5% of level
            touches = abs(recent_prices - level_price) / level_price < 0.005
            count += touches.sum()
            
        return count
    
    def _calculate_breakout_potential(
        self, 
        market_data: pd.DataFrame, 
        sr_context: dict[str, Any]
    ) -> float:
        """Calculate breakout potential score."""
        try:
            current_price = market_data["close"].iloc[-1]
            resistance_levels = sr_context.get("resistance", [])
            
            if not resistance_levels:
                return 0.5
            
            # Find nearest resistance
            nearest_resistance = min(
                resistance_levels,
                key=lambda x: abs(x["price"] - current_price)
            )
            
            # Check momentum toward resistance
            momentum = market_data["close"].pct_change(5).iloc[-1]
            
            # Check volume increase
            volume_ratio = (
                market_data["volume"].iloc[-5:].mean() /
                market_data["volume"].iloc[-20:].mean()
            )
            
            # Calculate potential
            distance_to_resistance = (
                (nearest_resistance["price"] - current_price) / current_price
            )
            
            if distance_to_resistance < 0.01 and momentum > 0 and volume_ratio > 1:
                potential = 0.8
            elif distance_to_resistance < 0.02 and momentum > 0:
                potential = 0.6
            else:
                potential = 0.3
                
            return float(potential)
            
        except Exception as e:
            self.logger.error(f"Error calculating breakout potential: {e}")
            return 0.5
    
    def _calculate_consolidation_score(self, market_data: pd.DataFrame) -> float:
        """Calculate consolidation score."""
        try:
            # Check recent price range
            recent_high = market_data["high"].iloc[-20:].max()
            recent_low = market_data["low"].iloc[-20:].min()
            price_range = (recent_high - recent_low) / recent_low
            
            # Lower range means higher consolidation
            if price_range < 0.02:
                score = 0.9
            elif price_range < 0.05:
                score = 0.7
            elif price_range < 0.10:
                score = 0.5
            else:
                score = 0.2
                
            return float(score)
            
        except Exception as e:
            self.logger.error(f"Error calculating consolidation score: {e}")
            return 0.5
    
    def _get_default_features(self) -> dict[str, float]:
        """Get default feature values."""
        return {
            # Price features
            "price_position_20": 0.5,
            "price_vs_sma10": 0.0,
            "price_vs_sma20": 0.0,
            "price_vs_sma50": 0.0,
            "volatility_20": 0.02,
            "volatility_50": 0.02,
            
            # S/R features
            "support_proximity": 1.0,
            "resistance_proximity": 1.0,
            "nearest_support_strength": 0.5,
            "nearest_resistance_strength": 0.5,
            "sr_balance": 0.5,
            
            # Strength features
            "avg_support_strength": 0.5,
            "max_support_strength": 0.5,
            "avg_resistance_strength": 0.5,
            "max_resistance_strength": 0.5,
            "overall_sr_strength": 0.5,
            
            # Technical features
            "rsi": 50.0,
            "rsi_overbought": 0.0,
            "rsi_oversold": 0.0,
            "macd": 0.0,
            "macd_signal": 0.0,
            "macd_histogram": 0.0,
            "macd_bullish": 0.5,
            "bb_position": 0.5,
            "bb_width": 0.02,
            
            # Volume features
            "volume_ratio_20": 1.0,
            "volume_ratio_50": 1.0,
            "volume_trend": 0.0,
            "obv_trend": 0.0,
            
            # Momentum features
            "roc_5": 0.0,
            "roc_10": 0.0,
            "roc_20": 0.0,
            "momentum_10": 0.0,
            "momentum_20": 0.0,
            "acceleration": 0.0,
            
            # Pattern features
            "recent_support_tests": 0.0,
            "recent_resistance_tests": 0.0,
            "breakout_potential": 0.5,
            "consolidation_score": 0.5
        }