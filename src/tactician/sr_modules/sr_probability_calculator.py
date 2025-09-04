"""S/R Probability Calculator using optimized parameters.

This module calculates S/R breakout, rebounce, and consolidation probabilities
using parameters optimized in step 2.5.
"""

import json
import os
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from src.core.decorators import handles_errors
from src.utils.logger import system_logger


class SRProbabilityCalculator:
    """Calculates S/R probabilities using optimized parameters."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the S/R probability calculator."""
        self.config = config
        self.logger = system_logger.getChild("SRProbabilityCalculator")
        
        # Load optimized parameters
        self.parameters = self._load_optimized_parameters()
        
    def _load_optimized_parameters(self) -> Dict[str, float]:
        """Load optimized parameters from step 2.5."""
        try:
            # Check if optimized parameters are in config
            if "sr_probability_calculation" in self.config:
                self.logger.info("✅ Using optimized S/R parameters from config")
                return self.config["sr_probability_calculation"]
            
            # Try to load from file
            param_file = os.path.join(
                self.config.get("model_save_path", "models"),
                "optimized_sr_parameters.json"
            )
            
            if os.path.exists(param_file):
                with open(param_file, 'r') as f:
                    data = json.load(f)
                    self.logger.info(f"✅ Loaded optimized S/R parameters from {param_file}")
                    return data["parameters"]
            
            # Fallback to default parameters
            self.logger.warning("⚠️ No optimized parameters found, using defaults")
            return self._get_default_parameters()
            
        except Exception as e:
            self.logger.error(f"Error loading optimized parameters: {e}")
            return self._get_default_parameters()
    
    def _get_default_parameters(self) -> Dict[str, float]:
        """Get default parameters if optimization hasn't run."""
        return {
            "price_action_weight": 0.3,
            "momentum_weight": 0.2,
            "trend_strength_weight": 0.2,
            "volume_weight": 0.2,
            "volatility_weight": 0.1,
            "volume_surge_multiplier": 2.0,
            "volume_confirmation_threshold": 1.5,
            "high_volatility_breakout_boost": 0.15,
            "low_volatility_consolidation_boost": 0.1,
            "level_strength_weight": 0.2,
            "touch_count_weight": 0.3,
            "age_decay_factor": 0.95,
            "proximity_threshold": 0.002,
            "proximity_decay_rate": 2.0,
            "min_breakout_probability": 0.2,
            "max_breakout_probability": 0.8,
            "default_probability": 0.33
        }
    
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return={},
        context="calculate SR probabilities"
    )
    def calculate_probabilities(
        self,
        market_data: pd.DataFrame,
        current_price: float,
        sr_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate S/R interaction probabilities.
        
        Args:
            market_data: Recent market data
            current_price: Current price
            sr_context: S/R context with detected levels
            
        Returns:
            Dictionary with breakout, rebounce, and consolidation probabilities
        """
        try:
            # Get nearest S/R levels
            nearest_support = self._get_nearest_level(
                current_price, sr_context.get("support", []), "below"
            )
            nearest_resistance = self._get_nearest_level(
                current_price, sr_context.get("resistance", []), "above"
            )
            
            # Calculate component scores
            components = {}
            
            # Price action component
            price_action_score = self._calculate_price_action_score(
                market_data, current_price
            )
            components["price_action"] = price_action_score * self.parameters["price_action_weight"]
            
            # Momentum component
            momentum_score = self._calculate_momentum_score(market_data)
            components["momentum"] = momentum_score * self.parameters["momentum_weight"]
            
            # Trend strength component
            trend_score = self._calculate_trend_strength_score(market_data)
            components["trend"] = trend_score * self.parameters["trend_strength_weight"]
            
            # Volume component
            volume_score = self._calculate_volume_score(market_data)
            components["volume"] = volume_score * self.parameters["volume_weight"]
            
            # Volatility component
            volatility_score = self._calculate_volatility_score(market_data)
            components["volatility"] = volatility_score * self.parameters["volatility_weight"]
            
            # S/R proximity component
            proximity_score = self._calculate_proximity_score(
                current_price, nearest_support, nearest_resistance
            )
            components["proximity"] = proximity_score
            
            # Combine components
            combined_score = sum(components.values())
            
            # Calculate final probabilities
            probabilities = self._calculate_final_probabilities(
                combined_score, proximity_score, volatility_score
            )
            
            # Add debug information
            probabilities["components"] = components
            probabilities["combined_score"] = combined_score
            
            return probabilities
            
        except Exception as e:
            self.logger.error(f"Error calculating S/R probabilities: {e}")
            return {
                "breakout": self.parameters["default_probability"],
                "rebounce": self.parameters["default_probability"],
                "consolidation": self.parameters["default_probability"]
            }
    
    def _get_nearest_level(
        self,
        current_price: float,
        levels: list,
        direction: str
    ) -> Optional[Dict[str, Any]]:
        """Get nearest S/R level in specified direction."""
        if not levels:
            return None
        
        if direction == "below":
            valid_levels = [l for l in levels if l["price"] < current_price]
            if valid_levels:
                return max(valid_levels, key=lambda x: x["price"])
        else:  # above
            valid_levels = [l for l in levels if l["price"] > current_price]
            if valid_levels:
                return min(valid_levels, key=lambda x: x["price"])
        
        return None
    
    def _calculate_price_action_score(
        self,
        market_data: pd.DataFrame,
        current_price: float
    ) -> float:
        """Calculate price action component score."""
        if len(market_data) < 20:
            return 0.5
        
        # Recent price momentum
        recent_returns = market_data["close"].pct_change().iloc[-10:]
        momentum = recent_returns.mean()
        
        # Candle patterns
        bullish_candles = (
            market_data["close"].iloc[-10:] > market_data["open"].iloc[-10:]
        ).sum() / 10
        
        # Price position relative to recent range
        recent_high = market_data["high"].iloc[-20:].max()
        recent_low = market_data["low"].iloc[-20:].min()
        price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
        
        # Combine factors
        score = 0.3 * price_position + 0.4 * bullish_candles + 0.3 * np.tanh(momentum * 100)
        
        return np.clip(score, 0, 1)
    
    def _calculate_momentum_score(self, market_data: pd.DataFrame) -> float:
        """Calculate momentum component score."""
        if len(market_data) < 14:
            return 0.5
        
        # Simple RSI calculation
        returns = market_data["close"].pct_change().iloc[-14:]
        gains = returns[returns > 0].mean() if (returns > 0).any() else 0
        losses = -returns[returns < 0].mean() if (returns < 0).any() else 0
        
        if losses == 0:
            rsi = 100
        else:
            rs = gains / losses
            rsi = 100 - (100 / (1 + rs))
        
        # Convert to score
        if rsi > 70:
            return 0.7 + (rsi - 70) / 100
        elif rsi < 30:
            return 0.3 - (30 - rsi) / 100
        else:
            return 0.5
    
    def _calculate_trend_strength_score(self, market_data: pd.DataFrame) -> float:
        """Calculate trend strength component score."""
        if len(market_data) < 50:
            return 0.5
        
        # Moving average alignment
        sma_10 = market_data["close"].iloc[-10:].mean()
        sma_20 = market_data["close"].iloc[-20:].mean()
        sma_50 = market_data["close"].iloc[-50:].mean()
        current_price = market_data["close"].iloc[-1]
        
        # Trend alignment score
        if current_price > sma_10 > sma_20 > sma_50:
            return 0.8  # Strong uptrend
        elif current_price < sma_10 < sma_20 < sma_50:
            return 0.2  # Strong downtrend
        else:
            # Calculate partial alignment
            uptrend_score = 0
            if current_price > sma_10: uptrend_score += 0.25
            if sma_10 > sma_20: uptrend_score += 0.25
            if sma_20 > sma_50: uptrend_score += 0.25
            if current_price > sma_50: uptrend_score += 0.25
            return uptrend_score
    
    def _calculate_volume_score(self, market_data: pd.DataFrame) -> float:
        """Calculate volume component score."""
        if len(market_data) < 20:
            return 0.5
        
        current_volume = market_data["volume"].iloc[-1]
        avg_volume = market_data["volume"].iloc[-20:].mean()
        
        if avg_volume == 0:
            return 0.5
        
        volume_ratio = current_volume / avg_volume
        
        # Apply thresholds
        if volume_ratio > self.parameters["volume_surge_multiplier"]:
            return 0.8
        elif volume_ratio > self.parameters["volume_confirmation_threshold"]:
            return 0.6 + 0.2 * (volume_ratio - self.parameters["volume_confirmation_threshold"])
        else:
            return 0.5 * volume_ratio
    
    def _calculate_volatility_score(self, market_data: pd.DataFrame) -> float:
        """Calculate volatility component score."""
        if len(market_data) < 14:
            return 0.5
        
        # ATR-based volatility
        high_low = market_data["high"].iloc[-14:] - market_data["low"].iloc[-14:]
        atr = high_low.mean()
        
        # Normalize by price
        current_price = market_data["close"].iloc[-1]
        volatility = atr / current_price if current_price > 0 else 0
        
        # High volatility favors breakouts
        if volatility > 0.02:
            return 0.5 + self.parameters["high_volatility_breakout_boost"]
        elif volatility < 0.005:
            return 0.5 - self.parameters["low_volatility_consolidation_boost"]
        else:
            # Linear interpolation
            return 0.5 + (volatility - 0.0125) * 10
    
    def _calculate_proximity_score(
        self,
        current_price: float,
        nearest_support: Optional[Dict[str, Any]],
        nearest_resistance: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate S/R proximity component score."""
        score = 0.5
        
        # Support proximity
        if nearest_support:
            support_distance = (current_price - nearest_support["price"]) / current_price
            if support_distance < self.parameters["proximity_threshold"]:
                proximity_factor = 1 - (support_distance / self.parameters["proximity_threshold"])
                # Near support decreases breakout probability
                score -= 0.3 * (proximity_factor ** self.parameters["proximity_decay_rate"])
        
        # Resistance proximity
        if nearest_resistance:
            resistance_distance = (nearest_resistance["price"] - current_price) / current_price
            if resistance_distance < self.parameters["proximity_threshold"]:
                proximity_factor = 1 - (resistance_distance / self.parameters["proximity_threshold"])
                # Near resistance decreases breakout probability
                score -= 0.3 * (proximity_factor ** self.parameters["proximity_decay_rate"])
        
        return np.clip(score, 0, 1)
    
    def _calculate_final_probabilities(
        self,
        combined_score: float,
        proximity_score: float,
        volatility_score: float
    ) -> Dict[str, float]:
        """Calculate final probabilities based on component scores."""
        
        # Base probabilities
        if combined_score > 0.6:
            # Bullish bias - higher breakout probability
            breakout_prob = self.parameters["default_probability"] + 0.3
            rebounce_prob = self.parameters["default_probability"] - 0.1
            consolidation_prob = self.parameters["default_probability"] - 0.2
        elif combined_score < 0.4:
            # Bearish bias - higher rebounce probability
            breakout_prob = self.parameters["default_probability"] - 0.1
            rebounce_prob = self.parameters["default_probability"] + 0.3
            consolidation_prob = self.parameters["default_probability"] - 0.2
        else:
            # Neutral - higher consolidation probability
            breakout_prob = self.parameters["default_probability"] - 0.08
            rebounce_prob = self.parameters["default_probability"] - 0.08
            consolidation_prob = self.parameters["default_probability"] + 0.16
        
        # Adjust for proximity
        if proximity_score < 0.3:
            # Very close to S/R - increase rebounce probability
            rebounce_prob += 0.2
            breakout_prob -= 0.1
            consolidation_prob -= 0.1
        
        # Adjust for volatility
        if volatility_score > 0.6:
            # High volatility - increase breakout/rebounce, decrease consolidation
            breakout_prob += 0.1
            rebounce_prob += 0.1
            consolidation_prob -= 0.2
        elif volatility_score < 0.4:
            # Low volatility - increase consolidation
            consolidation_prob += 0.2
            breakout_prob -= 0.1
            rebounce_prob -= 0.1
        
        # Apply bounds
        breakout_prob = np.clip(breakout_prob, 
                               self.parameters["min_breakout_probability"],
                               self.parameters["max_breakout_probability"])
        rebounce_prob = np.clip(rebounce_prob,
                               self.parameters["min_breakout_probability"],
                               self.parameters["max_breakout_probability"])
        consolidation_prob = max(0.1, consolidation_prob)
        
        # Normalize to sum to 1
        total = breakout_prob + rebounce_prob + consolidation_prob
        
        return {
            "breakout": breakout_prob / total,
            "rebounce": rebounce_prob / total,
            "consolidation": consolidation_prob / total
        }