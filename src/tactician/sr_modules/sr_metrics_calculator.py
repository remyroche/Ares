"""Support/Resistance Metrics Calculator Module."""

from typing import Any

import numpy as np
import pandas as pd

from src.core.decorators import handles_errors
from src.utils.logger import system_logger


class SRMetricsCalculator:
    """Calculates various metrics for S/R analysis."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize metrics calculator."""
        self.config = config
        self.logger = system_logger.getChild("SRMetricsCalculator")
        
        # Configuration
        self.sr_config = config.get("sr_breakout_predictor", {})
        self.atr_multiplier = self.sr_config.get("atr_multiplier", 1.5)
        self.volume_weight = self.sr_config.get("volume_weight", 0.7)
        self.price_weight = self.sr_config.get("price_weight", 0.3)
        
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return={},
        context="calculate SR metrics"
    )
    def calculate_comprehensive_metrics(
        self, 
        market_data: pd.DataFrame, 
        sr_context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Calculate comprehensive metrics for S/R analysis.
        
        Args:
            market_data: Market data DataFrame
            sr_context: S/R context with detected levels
            
        Returns:
            Dictionary containing various metrics
        """
        try:
            # Market metrics
            market_metrics = self._calculate_market_metrics(market_data)
            
            # S/R specific metrics
            sr_metrics = self._calculate_sr_metrics(sr_context, market_data)
            
            # Quality metrics
            quality_metrics = self._calculate_quality_metrics(market_data, sr_context)
            
            # Advanced metrics
            advanced_metrics = self._calculate_advanced_metrics(market_data, sr_context)
            
            # Combine all metrics
            comprehensive_metrics = {
                "market_metrics": market_metrics,
                "sr_metrics": sr_metrics,
                "quality_metrics": quality_metrics,
                "advanced_metrics": advanced_metrics,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            return comprehensive_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating comprehensive metrics: {e}")
            return {}
    
    def _calculate_market_metrics(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """Calculate general market metrics."""
        try:
            # Price metrics
            current_price = float(market_data["close"].iloc[-1])
            price_change = float(market_data["close"].pct_change().iloc[-1])
            volatility = float(market_data["close"].pct_change().std())
            
            # Volume metrics
            volume = float(market_data["volume"].iloc[-1])
            avg_volume = float(market_data["volume"].rolling(20).mean().iloc[-1])
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            
            # ATR
            high_low = market_data["high"] - market_data["low"]
            high_close = np.abs(market_data["high"] - market_data["close"].shift())
            low_close = np.abs(market_data["low"] - market_data["close"].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = float(true_range.rolling(14).mean().iloc[-1])
            
            # Trend metrics
            sma_20 = float(market_data["close"].rolling(20).mean().iloc[-1])
            sma_50 = float(market_data["close"].rolling(50).mean().iloc[-1])
            trend_strength = (current_price - sma_50) / sma_50 if sma_50 > 0 else 0
            
            return {
                "current_price": current_price,
                "price_change": price_change,
                "volatility": volatility,
                "volume": volume,
                "volume_ratio": volume_ratio,
                "atr": atr,
                "sma_20": sma_20,
                "sma_50": sma_50,
                "trend_strength": trend_strength
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating market metrics: {e}")
            return {}
    
    def _calculate_sr_metrics(
        self, 
        sr_context: dict[str, Any], 
        market_data: pd.DataFrame
    ) -> dict[str, Any]:
        """Calculate S/R specific metrics."""
        try:
            support_levels = sr_context.get("support", [])
            resistance_levels = sr_context.get("resistance", [])
            current_price = float(market_data["close"].iloc[-1])
            
            # Distance to nearest levels
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
            
            # Calculate distances
            support_distance = (
                abs(nearest_support["price"] - current_price) / current_price
                if nearest_support else float('inf')
            )
            resistance_distance = (
                abs(nearest_resistance["price"] - current_price) / current_price
                if nearest_resistance else float('inf')
            )
            
            # Level density
            price_range = market_data["high"].max() - market_data["low"].min()
            support_density = len(support_levels) / price_range if price_range > 0 else 0
            resistance_density = len(resistance_levels) / price_range if price_range > 0 else 0
            
            # Average strength
            avg_support_strength = (
                np.mean([level["strength"] for level in support_levels])
                if support_levels else 0
            )
            avg_resistance_strength = (
                np.mean([level["strength"] for level in resistance_levels])
                if resistance_levels else 0
            )
            
            return {
                "support_count": len(support_levels),
                "resistance_count": len(resistance_levels),
                "nearest_support_distance": support_distance,
                "nearest_resistance_distance": resistance_distance,
                "support_density": support_density,
                "resistance_density": resistance_density,
                "avg_support_strength": avg_support_strength,
                "avg_resistance_strength": avg_resistance_strength,
                "nearest_support_price": nearest_support["price"] if nearest_support else None,
                "nearest_resistance_price": nearest_resistance["price"] if nearest_resistance else None
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating S/R metrics: {e}")
            return {}
    
    def _calculate_quality_metrics(
        self, 
        market_data: pd.DataFrame, 
        sr_context: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate quality metrics for S/R analysis."""
        try:
            # Data quality
            data_quality = self._calculate_data_quality_score(market_data)
            
            # S/R confidence
            sr_confidence = self._calculate_sr_confidence_score(sr_context)
            
            # Overall quality
            overall_quality = (data_quality + sr_confidence) / 2
            
            return {
                "data_quality_score": data_quality,
                "sr_confidence_score": sr_confidence,
                "overall_quality_score": overall_quality
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating quality metrics: {e}")
            return {}
    
    def _calculate_data_quality_score(self, market_data: pd.DataFrame) -> float:
        """Calculate data quality score."""
        try:
            # Check for missing values
            missing_ratio = market_data.isnull().sum().sum() / market_data.size
            missing_score = 1 - missing_ratio
            
            # Check for zero volumes
            zero_volume_ratio = (market_data["volume"] == 0).sum() / len(market_data)
            volume_score = 1 - zero_volume_ratio
            
            # Check data consistency
            price_consistency = (
                market_data["high"] >= market_data["low"]
            ).all()
            consistency_score = 1.0 if price_consistency else 0.5
            
            # Combine scores
            quality_score = (missing_score + volume_score + consistency_score) / 3
            
            return float(quality_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating data quality score: {e}")
            return 0.5
    
    def _calculate_sr_confidence_score(self, sr_context: dict[str, Any]) -> float:
        """Calculate S/R confidence score."""
        try:
            support_levels = sr_context.get("support", [])
            resistance_levels = sr_context.get("resistance", [])
            
            # Check if we have levels
            if not support_levels and not resistance_levels:
                return 0.0
            
            # Average strength of levels
            all_strengths = (
                [level["strength"] for level in support_levels] +
                [level["strength"] for level in resistance_levels]
            )
            avg_strength = np.mean(all_strengths) if all_strengths else 0
            
            # Level count score (normalized)
            level_count = len(support_levels) + len(resistance_levels)
            count_score = min(level_count / 20, 1.0)  # Max score at 20 levels
            
            # Method diversity (different detection methods)
            methods = set()
            for level in support_levels + resistance_levels:
                methods.add(level.get("method", "unknown"))
            diversity_score = len(methods) / 3  # Assuming 3 methods max
            
            # Combine scores
            confidence_score = (avg_strength + count_score + diversity_score) / 3
            
            return float(confidence_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating S/R confidence score: {e}")
            return 0.5
    
    def _calculate_advanced_metrics(
        self, 
        market_data: pd.DataFrame, 
        sr_context: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate advanced S/R metrics."""
        try:
            current_price = float(market_data["close"].iloc[-1])
            
            # Multi-timeframe score
            mtf_score = self._calculate_multi_timeframe_score(market_data)
            
            # Clarity factor
            clarity_factor = self._calculate_clarity_factor(sr_context)
            
            # Directional pressure
            directional_pressure = self._calculate_directional_pressure(
                market_data, sr_context
            )
            
            # S/R score
            sr_score = self._calculate_sr_score(sr_context)
            
            # Delta S/R score
            delta_sr_score = self._calculate_delta_sr_score(market_data, sr_context)
            
            return {
                "multi_timeframe_score": mtf_score,
                "clarity_factor": clarity_factor,
                "directional_pressure": directional_pressure,
                "sr_score": sr_score,
                "delta_sr_score": delta_sr_score
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating advanced metrics: {e}")
            return {}
    
    def _calculate_multi_timeframe_score(self, market_data: pd.DataFrame) -> float:
        """Calculate multi-timeframe S/R score."""
        try:
            # Simple implementation - check multiple SMAs
            current_price = market_data["close"].iloc[-1]
            
            # Different timeframes
            sma_10 = market_data["close"].rolling(10).mean().iloc[-1]
            sma_20 = market_data["close"].rolling(20).mean().iloc[-1]
            sma_50 = market_data["close"].rolling(50).mean().iloc[-1]
            
            # Score based on alignment
            score = 0
            if current_price > sma_10:
                score += 0.33
            if current_price > sma_20:
                score += 0.33
            if current_price > sma_50:
                score += 0.34
                
            return float(score)
            
        except Exception as e:
            self.logger.error(f"Error calculating MTF score: {e}")
            return 0.5
    
    def _calculate_clarity_factor(self, sr_context: dict[str, Any]) -> float:
        """Calculate clarity factor for S/R levels."""
        try:
            support_levels = sr_context.get("support", [])
            resistance_levels = sr_context.get("resistance", [])
            
            if not support_levels and not resistance_levels:
                return 0.0
            
            # Check spacing between levels
            all_prices = sorted([
                level["price"] 
                for level in support_levels + resistance_levels
            ])
            
            if len(all_prices) < 2:
                return 1.0
            
            # Calculate spacing consistency
            spacings = []
            for i in range(1, len(all_prices)):
                spacing = (all_prices[i] - all_prices[i-1]) / all_prices[i-1]
                spacings.append(spacing)
            
            # Lower std deviation means more consistent spacing
            if spacings:
                spacing_std = np.std(spacings)
                clarity = 1 / (1 + spacing_std * 10)  # Normalize
            else:
                clarity = 0.5
                
            return float(clarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating clarity factor: {e}")
            return 0.5
    
    def _calculate_directional_pressure(
        self, 
        market_data: pd.DataFrame, 
        sr_context: dict[str, Any]
    ) -> float:
        """Calculate directional pressure based on S/R and momentum."""
        try:
            current_price = market_data["close"].iloc[-1]
            
            # Recent price movement
            price_change = market_data["close"].pct_change(5).iloc[-1]
            
            # Volume trend
            volume_trend = (
                market_data["volume"].iloc[-5:].mean() / 
                market_data["volume"].iloc[-20:].mean()
            )
            
            # S/R position
            support_levels = sr_context.get("support", [])
            resistance_levels = sr_context.get("resistance", [])
            
            # Check if breaking levels
            breaking_resistance = any(
                abs(current_price - r["price"]) / current_price < 0.001
                for r in resistance_levels
            )
            breaking_support = any(
                abs(current_price - s["price"]) / current_price < 0.001
                for s in support_levels
            )
            
            # Calculate pressure
            pressure = 0.5  # Neutral
            if price_change > 0 and volume_trend > 1:
                pressure += 0.25
            elif price_change < 0 and volume_trend > 1:
                pressure -= 0.25
                
            if breaking_resistance:
                pressure += 0.25
            elif breaking_support:
                pressure -= 0.25
                
            return float(max(0, min(1, pressure)))
            
        except Exception as e:
            self.logger.error(f"Error calculating directional pressure: {e}")
            return 0.5
    
    def _calculate_sr_score(self, sr_context: dict[str, Any]) -> float:
        """Calculate overall S/R score."""
        try:
            support_levels = sr_context.get("support", [])
            resistance_levels = sr_context.get("resistance", [])
            
            # Strength score
            all_strengths = (
                [level["strength"] for level in support_levels] +
                [level["strength"] for level in resistance_levels]
            )
            strength_score = np.mean(all_strengths) if all_strengths else 0
            
            # Balance score (similar number of S and R)
            total_levels = len(support_levels) + len(resistance_levels)
            if total_levels > 0:
                balance = abs(len(support_levels) - len(resistance_levels)) / total_levels
                balance_score = 1 - balance
            else:
                balance_score = 0
            
            # Combined score
            sr_score = (strength_score + balance_score) / 2
            
            return float(sr_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating S/R score: {e}")
            return 0.5
    
    def _calculate_delta_sr_score(
        self, 
        market_data: pd.DataFrame, 
        sr_context: dict[str, Any]
    ) -> float:
        """Calculate delta S/R score (rate of change)."""
        try:
            # Simple implementation - check recent level breaks
            current_price = market_data["close"].iloc[-1]
            recent_prices = market_data["close"].iloc[-10:]
            
            support_levels = sr_context.get("support", [])
            resistance_levels = sr_context.get("resistance", [])
            
            # Count recent breaks
            breaks = 0
            for level in support_levels + resistance_levels:
                level_price = level["price"]
                # Check if price crossed this level recently
                crosses = (
                    (recent_prices.shift(1) < level_price) & 
                    (recent_prices > level_price)
                ) | (
                    (recent_prices.shift(1) > level_price) & 
                    (recent_prices < level_price)
                )
                if crosses.any():
                    breaks += 1
            
            # Normalize
            delta_score = min(breaks / 5, 1.0)  # Max score at 5 breaks
            
            return float(delta_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating delta S/R score: {e}")
            return 0.5