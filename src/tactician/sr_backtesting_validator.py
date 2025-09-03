"""S/R Backtesting Validator Module.

This module provides comprehensive backtesting capabilities for S/R probability calculations,
ensuring that rule-based heuristics are validated against historical data.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.core.decorators import handles_errors, traced
from src.utils.logger import system_logger


class SRBacktestingValidator:
    """Validates S/R detection rules through comprehensive backtesting."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the S/R backtesting validator."""
        self.config = config
        self.logger = system_logger.getChild("SRBacktestingValidator")
        
        # Backtesting configuration
        self.backtest_config = config.get("sr_backtesting", {})
        self.lookback_periods = self.backtest_config.get("lookback_periods", 500)
        self.min_test_points = self.backtest_config.get("min_test_points", 20)
        self.confidence_threshold = self.backtest_config.get("confidence_threshold", 0.6)
        
        # Performance metrics storage
        self.backtest_results = {}
        self.validation_metrics = {}
        
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return={},
        context="validate SR levels"
    )
    @traced(span_name="SRBacktest.validate_levels")
    async def validate_sr_levels(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]],
        current_price: float
    ) -> Dict[str, Any]:
        """
        Validate S/R levels through historical backtesting.
        
        Args:
            market_data: Historical market data
            sr_levels: Detected S/R levels
            current_price: Current market price
            
        Returns:
            Validation results with performance metrics
        """
        try:
            self.logger.info(f"🔍 Validating {len(sr_levels)} S/R levels through backtesting")
            
            # Separate support and resistance levels
            support_levels = [l for l in sr_levels if l.get("type") == "support"]
            resistance_levels = [l for l in sr_levels if l.get("type") == "resistance"]
            
            # Validate each aspect
            price_action_metrics = await self._validate_price_action_strength(
                market_data, support_levels, resistance_levels
            )
            
            volatility_metrics = await self._validate_volatility_impact(
                market_data, sr_levels
            )
            
            volume_metrics = await self._validate_volume_patterns(
                market_data, sr_levels
            )
            
            sr_strength_metrics = await self._validate_sr_strength(
                market_data, sr_levels
            )
            
            age_metrics = await self._validate_level_age_impact(
                market_data, sr_levels
            )
            
            # Calculate probability adjustments based on validation
            probability_adjustments = self._calculate_probability_adjustments(
                price_action_metrics,
                volatility_metrics,
                volume_metrics,
                sr_strength_metrics,
                age_metrics
            )
            
            # Generate comprehensive validation report
            validation_result = {
                "validated": True,
                "timestamp": datetime.now().isoformat(),
                "levels_tested": len(sr_levels),
                "metrics": {
                    "price_action": price_action_metrics,
                    "volatility": volatility_metrics,
                    "volume": volume_metrics,
                    "sr_strength": sr_strength_metrics,
                    "age": age_metrics
                },
                "probability_adjustments": probability_adjustments,
                "overall_confidence": self._calculate_overall_confidence(
                    price_action_metrics,
                    volatility_metrics,
                    volume_metrics,
                    sr_strength_metrics,
                    age_metrics
                )
            }
            
            self.logger.info(f"✅ Validation completed with confidence: {validation_result['overall_confidence']:.2%}")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Error validating S/R levels: {e}")
            return {}
    
    async def _validate_price_action_strength(
        self,
        market_data: pd.DataFrame,
        support_levels: List[Dict[str, Any]],
        resistance_levels: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Validate price action strength around S/R levels."""
        try:
            # Analyze price behavior near levels
            support_bounces = 0
            support_breaks = 0
            resistance_rejections = 0
            resistance_breaks = 0
            
            close_prices = market_data["close"].values
            high_prices = market_data["high"].values
            low_prices = market_data["low"].values
            
            # Test support levels
            for level in support_levels:
                level_price = level["price"]
                touches = 0
                bounces = 0
                breaks = 0
                
                for i in range(1, len(close_prices) - 1):
                    # Check if price touched support
                    if low_prices[i] <= level_price * 1.002:  # Within 0.2% of level
                        touches += 1
                        
                        # Check if it bounced (next candle closed higher)
                        if close_prices[i+1] > close_prices[i]:
                            bounces += 1
                            support_bounces += 1
                        # Check if it broke (next candle closed below)
                        elif close_prices[i+1] < level_price * 0.998:
                            breaks += 1
                            support_breaks += 1
                
                # Update level metrics
                if touches > 0:
                    level["bounce_rate"] = bounces / touches
                    level["break_rate"] = breaks / touches
            
            # Test resistance levels
            for level in resistance_levels:
                level_price = level["price"]
                touches = 0
                rejections = 0
                breaks = 0
                
                for i in range(1, len(close_prices) - 1):
                    # Check if price touched resistance
                    if high_prices[i] >= level_price * 0.998:  # Within 0.2% of level
                        touches += 1
                        
                        # Check if it was rejected (next candle closed lower)
                        if close_prices[i+1] < close_prices[i]:
                            rejections += 1
                            resistance_rejections += 1
                        # Check if it broke (next candle closed above)
                        elif close_prices[i+1] > level_price * 1.002:
                            breaks += 1
                            resistance_breaks += 1
                
                # Update level metrics
                if touches > 0:
                    level["rejection_rate"] = rejections / touches
                    level["break_rate"] = breaks / touches
            
            # Calculate overall metrics
            total_support_tests = support_bounces + support_breaks
            total_resistance_tests = resistance_rejections + resistance_breaks
            
            return {
                "support_bounce_rate": support_bounces / max(total_support_tests, 1),
                "support_break_rate": support_breaks / max(total_support_tests, 1),
                "resistance_rejection_rate": resistance_rejections / max(total_resistance_tests, 1),
                "resistance_break_rate": resistance_breaks / max(total_resistance_tests, 1),
                "total_tests": total_support_tests + total_resistance_tests
            }
            
        except Exception as e:
            self.logger.error(f"Error validating price action strength: {e}")
            return {}
    
    async def _validate_volatility_impact(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Validate how volatility affects S/R level behavior."""
        try:
            # Calculate rolling volatility
            returns = market_data["close"].pct_change()
            volatility = returns.rolling(window=20).std()
            
            # Categorize volatility periods
            low_vol_threshold = volatility.quantile(0.33)
            high_vol_threshold = volatility.quantile(0.67)
            
            # Test S/R behavior in different volatility regimes
            low_vol_accuracy = 0
            med_vol_accuracy = 0
            high_vol_accuracy = 0
            
            for level in sr_levels:
                level_price = level["price"]
                level_type = level["type"]
                
                # Find periods when price was near this level
                proximity_mask = np.abs(market_data["close"] - level_price) / level_price < 0.005
                
                if proximity_mask.sum() > 0:
                    # Test in low volatility
                    low_vol_mask = proximity_mask & (volatility <= low_vol_threshold)
                    if low_vol_mask.sum() > 0:
                        low_vol_accuracy += self._calculate_level_accuracy(
                            market_data[low_vol_mask], level_price, level_type
                        )
                    
                    # Test in medium volatility
                    med_vol_mask = proximity_mask & (volatility > low_vol_threshold) & (volatility <= high_vol_threshold)
                    if med_vol_mask.sum() > 0:
                        med_vol_accuracy += self._calculate_level_accuracy(
                            market_data[med_vol_mask], level_price, level_type
                        )
                    
                    # Test in high volatility
                    high_vol_mask = proximity_mask & (volatility > high_vol_threshold)
                    if high_vol_mask.sum() > 0:
                        high_vol_accuracy += self._calculate_level_accuracy(
                            market_data[high_vol_mask], level_price, level_type
                        )
            
            # Normalize by number of levels
            n_levels = max(len(sr_levels), 1)
            
            return {
                "low_volatility_accuracy": low_vol_accuracy / n_levels,
                "medium_volatility_accuracy": med_vol_accuracy / n_levels,
                "high_volatility_accuracy": high_vol_accuracy / n_levels,
                "volatility_adaptability": np.std([low_vol_accuracy, med_vol_accuracy, high_vol_accuracy]) / n_levels
            }
            
        except Exception as e:
            self.logger.error(f"Error validating volatility impact: {e}")
            return {}
    
    async def _validate_volume_patterns(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Validate volume patterns around S/R levels."""
        try:
            volume = market_data["volume"].values
            avg_volume = np.mean(volume)
            
            # Analyze volume at S/R interactions
            high_volume_confirmations = 0
            low_volume_failures = 0
            volume_surge_breakouts = 0
            
            for level in sr_levels:
                level_price = level["price"]
                level_type = level["type"]
                
                # Find S/R interactions
                if level_type == "support":
                    interactions = np.where(
                        (market_data["low"] <= level_price * 1.002) &
                        (market_data["close"] > level_price)
                    )[0]
                else:  # resistance
                    interactions = np.where(
                        (market_data["high"] >= level_price * 0.998) &
                        (market_data["close"] < level_price)
                    )[0]
                
                for idx in interactions:
                    if idx < len(volume) - 1:
                        interaction_volume = volume[idx]
                        next_volume = volume[idx + 1]
                        
                        # High volume confirmation
                        if interaction_volume > avg_volume * 1.5:
                            high_volume_confirmations += 1
                        
                        # Low volume failure
                        if interaction_volume < avg_volume * 0.7:
                            # Check if level failed in next periods
                            if level_type == "support" and market_data["close"].iloc[idx+1] < level_price * 0.995:
                                low_volume_failures += 1
                            elif level_type == "resistance" and market_data["close"].iloc[idx+1] > level_price * 1.005:
                                low_volume_failures += 1
                        
                        # Volume surge breakout
                        if next_volume > interaction_volume * 2:
                            volume_surge_breakouts += 1
            
            total_interactions = high_volume_confirmations + low_volume_failures
            
            return {
                "high_volume_confirmation_rate": high_volume_confirmations / max(total_interactions, 1),
                "low_volume_failure_rate": low_volume_failures / max(total_interactions, 1),
                "volume_surge_breakout_rate": volume_surge_breakouts / max(total_interactions, 1),
                "volume_reliability": high_volume_confirmations / max(low_volume_failures, 1)
            }
            
        except Exception as e:
            self.logger.error(f"Error validating volume patterns: {e}")
            return {}
    
    async def _validate_sr_strength(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Validate S/R strength calculations."""
        try:
            # Group levels by strength
            weak_levels = [l for l in sr_levels if l.get("strength", 0) < 0.3]
            medium_levels = [l for l in sr_levels if 0.3 <= l.get("strength", 0) < 0.7]
            strong_levels = [l for l in sr_levels if l.get("strength", 0) >= 0.7]
            
            # Test accuracy by strength category
            weak_accuracy = await self._test_level_category_accuracy(market_data, weak_levels)
            medium_accuracy = await self._test_level_category_accuracy(market_data, medium_levels)
            strong_accuracy = await self._test_level_category_accuracy(market_data, strong_levels)
            
            # Calculate strength correlation
            if sr_levels:
                strengths = [l.get("strength", 0) for l in sr_levels]
                accuracies = []
                
                for level in sr_levels:
                    accuracy = self._calculate_level_accuracy(
                        market_data,
                        level["price"],
                        level["type"]
                    )
                    accuracies.append(accuracy)
                
                strength_correlation = np.corrcoef(strengths, accuracies)[0, 1]
            else:
                strength_correlation = 0
            
            return {
                "weak_level_accuracy": weak_accuracy,
                "medium_level_accuracy": medium_accuracy,
                "strong_level_accuracy": strong_accuracy,
                "strength_correlation": strength_correlation,
                "strength_reliability": (strong_accuracy - weak_accuracy) if strong_accuracy > weak_accuracy else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error validating S/R strength: {e}")
            return {}
    
    async def _validate_level_age_impact(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Validate how level age affects reliability."""
        try:
            current_time = len(market_data)
            
            # Categorize levels by age
            new_levels = []
            established_levels = []
            old_levels = []
            
            for level in sr_levels:
                # Calculate age (bars since detection)
                detection_time = level.get("detection_time", 0)
                age = current_time - detection_time
                
                if age < 50:
                    new_levels.append(level)
                elif age < 200:
                    established_levels.append(level)
                else:
                    old_levels.append(level)
            
            # Test accuracy by age category
            new_accuracy = await self._test_level_category_accuracy(market_data, new_levels)
            established_accuracy = await self._test_level_category_accuracy(market_data, established_levels)
            old_accuracy = await self._test_level_category_accuracy(market_data, old_levels)
            
            # Calculate age decay factor
            age_decay_factor = 0
            if old_accuracy < established_accuracy:
                age_decay_factor = (established_accuracy - old_accuracy) / established_accuracy
            
            return {
                "new_level_accuracy": new_accuracy,
                "established_level_accuracy": established_accuracy,
                "old_level_accuracy": old_accuracy,
                "age_decay_factor": age_decay_factor,
                "optimal_age_range": "50-200 bars" if established_accuracy > max(new_accuracy, old_accuracy) else "varies"
            }
            
        except Exception as e:
            self.logger.error(f"Error validating level age impact: {e}")
            return {}
    
    def _calculate_level_accuracy(
        self,
        data: pd.DataFrame,
        level_price: float,
        level_type: str
    ) -> float:
        """Calculate accuracy of a single S/R level."""
        try:
            if len(data) == 0:
                return 0.0
            
            correct_reactions = 0
            total_tests = 0
            
            for i in range(len(data) - 1):
                if level_type == "support":
                    # Test support: if low touches level, should bounce
                    if data["low"].iloc[i] <= level_price * 1.002:
                        total_tests += 1
                        if data["close"].iloc[i+1] > data["close"].iloc[i]:
                            correct_reactions += 1
                else:  # resistance
                    # Test resistance: if high touches level, should reject
                    if data["high"].iloc[i] >= level_price * 0.998:
                        total_tests += 1
                        if data["close"].iloc[i+1] < data["close"].iloc[i]:
                            correct_reactions += 1
            
            return correct_reactions / max(total_tests, 1)
            
        except Exception:
            return 0.0
    
    async def _test_level_category_accuracy(
        self,
        market_data: pd.DataFrame,
        levels: List[Dict[str, Any]]
    ) -> float:
        """Test accuracy of a category of levels."""
        if not levels:
            return 0.0
        
        total_accuracy = 0
        for level in levels:
            accuracy = self._calculate_level_accuracy(
                market_data,
                level["price"],
                level["type"]
            )
            total_accuracy += accuracy
        
        return total_accuracy / len(levels)
    
    def _calculate_probability_adjustments(
        self,
        price_action_metrics: Dict[str, float],
        volatility_metrics: Dict[str, float],
        volume_metrics: Dict[str, float],
        sr_strength_metrics: Dict[str, float],
        age_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate probability adjustments based on validation results."""
        
        # Base adjustments
        breakout_adjustment = 0.0
        rebounce_adjustment = 0.0
        consolidation_adjustment = 0.0
        
        # Price action adjustments
        if price_action_metrics:
            # High bounce rate increases rebounce probability
            rebounce_adjustment += (price_action_metrics.get("support_bounce_rate", 0.5) - 0.5) * 0.2
            rebounce_adjustment += (price_action_metrics.get("resistance_rejection_rate", 0.5) - 0.5) * 0.2
            
            # High break rate increases breakout probability
            breakout_adjustment += (price_action_metrics.get("support_break_rate", 0.5) - 0.5) * 0.2
            breakout_adjustment += (price_action_metrics.get("resistance_break_rate", 0.5) - 0.5) * 0.2
        
        # Volatility adjustments
        if volatility_metrics:
            # High volatility increases breakout probability
            high_vol_accuracy = volatility_metrics.get("high_volatility_accuracy", 0.5)
            if high_vol_accuracy > 0.6:
                breakout_adjustment += 0.1
            
            # Low volatility increases consolidation probability
            low_vol_accuracy = volatility_metrics.get("low_volatility_accuracy", 0.5)
            if low_vol_accuracy > 0.6:
                consolidation_adjustment += 0.1
        
        # Volume adjustments
        if volume_metrics:
            # Volume surge increases breakout probability
            surge_rate = volume_metrics.get("volume_surge_breakout_rate", 0)
            breakout_adjustment += surge_rate * 0.15
            
            # High volume confirmation increases reliability
            confirmation_rate = volume_metrics.get("high_volume_confirmation_rate", 0.5)
            if confirmation_rate > 0.7:
                # Boost the dominant probability
                rebounce_adjustment *= 1.1
                breakout_adjustment *= 1.1
        
        # Strength adjustments
        if sr_strength_metrics:
            # Strong levels have higher rebounce probability
            strength_reliability = sr_strength_metrics.get("strength_reliability", 0)
            rebounce_adjustment += strength_reliability * 0.1
        
        # Age adjustments
        if age_metrics:
            # Established levels are more reliable
            established_accuracy = age_metrics.get("established_level_accuracy", 0.5)
            if established_accuracy > 0.6:
                rebounce_adjustment += 0.05
        
        # Normalize adjustments
        total_adjustment = abs(breakout_adjustment) + abs(rebounce_adjustment) + abs(consolidation_adjustment)
        if total_adjustment > 0:
            breakout_adjustment /= total_adjustment
            rebounce_adjustment /= total_adjustment
            consolidation_adjustment /= total_adjustment
        
        return {
            "breakout_adjustment": np.clip(breakout_adjustment, -0.3, 0.3),
            "rebounce_adjustment": np.clip(rebounce_adjustment, -0.3, 0.3),
            "consolidation_adjustment": np.clip(consolidation_adjustment, -0.3, 0.3)
        }
    
    def _calculate_overall_confidence(
        self,
        price_action_metrics: Dict[str, float],
        volatility_metrics: Dict[str, float],
        volume_metrics: Dict[str, float],
        sr_strength_metrics: Dict[str, float],
        age_metrics: Dict[str, float]
    ) -> float:
        """Calculate overall confidence in S/R validation."""
        
        confidence_scores = []
        
        # Price action confidence
        if price_action_metrics:
            pa_confidence = (
                price_action_metrics.get("support_bounce_rate", 0) * 0.25 +
                price_action_metrics.get("resistance_rejection_rate", 0) * 0.25 +
                (1 - price_action_metrics.get("support_break_rate", 1)) * 0.25 +
                (1 - price_action_metrics.get("resistance_break_rate", 1)) * 0.25
            )
            confidence_scores.append(pa_confidence)
        
        # Volatility confidence
        if volatility_metrics:
            vol_confidence = np.mean([
                volatility_metrics.get("low_volatility_accuracy", 0),
                volatility_metrics.get("medium_volatility_accuracy", 0),
                volatility_metrics.get("high_volatility_accuracy", 0)
            ])
            confidence_scores.append(vol_confidence)
        
        # Volume confidence
        if volume_metrics:
            vol_confidence = (
                volume_metrics.get("high_volume_confirmation_rate", 0) * 0.5 +
                (1 - volume_metrics.get("low_volume_failure_rate", 1)) * 0.5
            )
            confidence_scores.append(vol_confidence)
        
        # Strength confidence
        if sr_strength_metrics:
            strength_confidence = sr_strength_metrics.get("strength_correlation", 0) * 0.5 + 0.5
            confidence_scores.append(strength_confidence)
        
        # Age confidence
        if age_metrics:
            age_confidence = age_metrics.get("established_level_accuracy", 0)
            confidence_scores.append(age_confidence)
        
        # Calculate weighted average
        if confidence_scores:
            return np.mean(confidence_scores)
        else:
            return 0.5


async def setup_sr_backtesting_validator(config: Dict[str, Any]) -> Optional[SRBacktestingValidator]:
    """Factory function to create and initialize SR backtesting validator."""
    try:
        validator = SRBacktestingValidator(config)
        return validator
    except Exception as e:
        system_logger.error(f"Failed to setup SR backtesting validator: {e}")
        return None