"""
Trading Target Meta Labels System

Comprehensive meta labeling system for specific trading targets:
- Breakout patterns
- Mean reversion setups
- Trend following signals
- Rejection patterns
- Support/Resistance levels
- Consolidation patterns
- And more...

This system integrates with NAS/TAS systems to generate enhanced meta labels.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from src.utils.logger import system_logger
from src.core.decorators import handles_errors


class TradingTarget(Enum):
    """Enumeration of trading targets."""
    BREAKOUT = "breakout"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    REJECTION = "rejection"
    SUPPORT_RESISTANCE = "support_resistance"
    CONSOLIDATION = "consolidation"
    MOMENTUM = "momentum"
    VOLATILITY_EXPANSION = "volatility_expansion"
    VOLATILITY_COMPRESSION = "volatility_compression"
    REVERSAL = "reversal"


@dataclass
class TargetMetaLabel:
    """Meta label for a specific trading target."""
    target_type: TradingTarget
    signal_strength: float  # 0-1 scale
    confidence: float  # 0-1 scale
    probability: float  # 0-1 scale
    time_horizon: str  # "short", "medium", "long"
    risk_level: str  # "low", "medium", "high"
    setup_quality: float  # 0-1 scale
    entry_conditions: Dict[str, Any]
    exit_conditions: Dict[str, Any]
    metadata: Dict[str, Any]


class TradingTargetMetaLabeler:
    """
    Comprehensive meta labeling system for trading targets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("TradingTargetMetaLabeler")
        
        # Configuration parameters
        self.breakout_config = config.get("breakout", {})
        self.mean_reversion_config = config.get("mean_reversion", {})
        self.trend_following_config = config.get("trend_following", {})
        self.rejection_config = config.get("rejection", {})
        self.sr_config = config.get("support_resistance", {})
        self.consolidation_config = config.get("consolidation", {})
        
        self.logger.info("🎯 Trading Target Meta Labeler initialized")
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={},
        context="generate_all_target_labels"
    )
    async def generate_all_target_labels(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        additional_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, TargetMetaLabel]:
        """Generate meta labels for all trading targets."""
        try:
            self.logger.info("🚀 Generating comprehensive target meta labels")
            
            # Calculate base features
            features = await self._calculate_base_features(price_data, volume_data)
            if additional_features:
                features.update(additional_features)
            
            target_labels = {}
            
            # Generate labels for each target type
            target_labels["breakout"] = await self._generate_breakout_labels(price_data, features)
            target_labels["mean_reversion"] = await self._generate_mean_reversion_labels(price_data, features)
            target_labels["trend_following"] = await self._generate_trend_following_labels(price_data, features)
            target_labels["rejection"] = await self._generate_rejection_labels(price_data, features)
            target_labels["support_resistance"] = await self._generate_sr_labels(price_data, features)
            target_labels["consolidation"] = await self._generate_consolidation_labels(price_data, features)
            target_labels["momentum"] = await self._generate_momentum_labels(price_data, features)
            target_labels["volatility_expansion"] = await self._generate_volatility_expansion_labels(price_data, features)
            target_labels["volatility_compression"] = await self._generate_volatility_compression_labels(price_data, features)
            target_labels["reversal"] = await self._generate_reversal_labels(price_data, features)
            
            self.logger.info(f"✅ Generated {len(target_labels)} target meta labels")
            return target_labels
            
        except Exception as e:
            self.logger.error(f"Error generating target labels: {e}")
            return {}
    
    async def _calculate_base_features(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate base features for target detection."""
        try:
            features = {}
            
            # Price features
            features["close"] = price_data["close"]
            features["high"] = price_data["high"]
            features["low"] = price_data["low"]
            features["open"] = price_data["open"]
            
            # Technical indicators
            features["sma_20"] = price_data["close"].rolling(20).mean()
            features["sma_50"] = price_data["close"].rolling(50).mean()
            features["ema_12"] = price_data["close"].ewm(span=12).mean()
            features["ema_26"] = price_data["close"].ewm(span=26).mean()
            
            # Bollinger Bands
            bb_middle = features["sma_20"]
            bb_std = price_data["close"].rolling(20).std()
            features["bb_upper"] = bb_middle + (bb_std * 2)
            features["bb_lower"] = bb_middle - (bb_std * 2)
            features["bb_position"] = (price_data["close"] - features["bb_lower"]) / (features["bb_upper"] - features["bb_lower"])
            
            # RSI
            delta = price_data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features["rsi"] = 100 - (100 / (1 + rs))
            
            # MACD
            macd = features["ema_12"] - features["ema_26"]
            signal = macd.ewm(span=9).mean()
            features["macd"] = macd
            features["macd_signal"] = signal
            features["macd_histogram"] = macd - signal
            
            # Volume features
            if "volume" in volume_data.columns:
                features["volume"] = volume_data["volume"]
                features["volume_sma"] = volume_data["volume"].rolling(20).mean()
                features["volume_ratio"] = volume_data["volume"] / features["volume_sma"]
            
            # Volatility
            returns = price_data["close"].pct_change()
            features["volatility_20"] = returns.rolling(20).std()
            features["volatility_10"] = returns.rolling(10).std()
            
            # Momentum
            features["momentum_5"] = price_data["close"].pct_change(5)
            features["momentum_10"] = price_data["close"].pct_change(10)
            features["momentum_20"] = price_data["close"].pct_change(20)
            
            # Support and Resistance
            features["recent_high_20"] = price_data["high"].rolling(20).max()
            features["recent_low_20"] = price_data["low"].rolling(20).min()
            features["recent_high_50"] = price_data["high"].rolling(50).max()
            features["recent_low_50"] = price_data["low"].rolling(50).min()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating base features: {e}")
            return {}
    
    async def _generate_breakout_labels(self, price_data: pd.DataFrame, features: Dict[str, Any]) -> TargetMetaLabel:
        """Generate breakout meta labels."""
        try:
            current_price = price_data["close"].iloc[-1]
            recent_high = features["recent_high_20"].iloc[-1]
            recent_low = features["recent_low_20"].iloc[-1]
            volume_ratio = features.get("volume_ratio", pd.Series(1.0)).iloc[-1]
            bb_position = features["bb_position"].iloc[-1]
            momentum = features["momentum_5"].iloc[-1]
            
            # Breakout conditions
            is_breakout_up = current_price > recent_high * 1.001  # 0.1% above recent high
            is_breakout_down = current_price < recent_low * 0.999  # 0.1% below recent low
            is_high_volume = volume_ratio > 1.5
            is_strong_momentum = abs(momentum) > 0.01
            
            # Calculate signal strength
            signal_strength = 0.0
            if is_breakout_up or is_breakout_down:
                signal_strength = min(volume_ratio / 2, 1.0) * min(abs(momentum) * 50, 1.0)
            
            # Calculate confidence
            confidence = 0.0
            if is_breakout_up or is_breakout_down:
                confidence = min(volume_ratio / 3, 1.0) * (1.0 - abs(bb_position - 0.5) * 2)
            
            # Calculate probability
            probability = signal_strength * confidence
            
            # Determine time horizon
            time_horizon = "short" if abs(momentum) > 0.02 else "medium"
            
            # Determine risk level
            risk_level = "high" if volume_ratio > 3 else "medium" if volume_ratio > 1.5 else "low"
            
            # Setup quality
            setup_quality = (signal_strength + confidence) / 2
            
            return TargetMetaLabel(
                target_type=TradingTarget.BREAKOUT,
                signal_strength=signal_strength,
                confidence=confidence,
                probability=probability,
                time_horizon=time_horizon,
                risk_level=risk_level,
                setup_quality=setup_quality,
                entry_conditions={
                    "price_break": is_breakout_up or is_breakout_down,
                    "volume_confirmation": is_high_volume,
                    "momentum_confirmation": is_strong_momentum
                },
                exit_conditions={
                    "stop_loss": recent_low * 0.98 if is_breakout_up else recent_high * 1.02,
                    "take_profit": recent_high * 1.05 if is_breakout_up else recent_low * 0.95
                },
                metadata={
                    "recent_high": recent_high,
                    "recent_low": recent_low,
                    "volume_ratio": volume_ratio,
                    "bb_position": bb_position,
                    "momentum": momentum
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating breakout labels: {e}")
            return self._create_empty_target_label(TradingTarget.BREAKOUT)
    
    async def _generate_mean_reversion_labels(self, price_data: pd.DataFrame, features: Dict[str, Any]) -> TargetMetaLabel:
        """Generate mean reversion meta labels."""
        try:
            bb_position = features["bb_position"].iloc[-1]
            rsi = features["rsi"].iloc[-1]
            volatility = features["volatility_20"].iloc[-1]
            momentum = features["momentum_5"].iloc[-1]
            
            # Mean reversion conditions
            is_at_edge = bb_position < 0.2 or bb_position > 0.8
            is_oversold = rsi < 30
            is_overbought = rsi > 70
            is_low_volatility = volatility < 0.02
            is_sideways = abs(momentum) < 0.01
            
            # Calculate signal strength
            signal_strength = 0.0
            if is_at_edge:
                signal_strength = abs(bb_position - 0.5) * 2  # Higher when further from center
                if is_oversold or is_overbought:
                    signal_strength *= 1.2
            
            # Calculate confidence
            confidence = 0.0
            if is_at_edge and is_low_volatility and is_sideways:
                confidence = 0.8
                if is_oversold or is_overbought:
                    confidence = 0.9
            
            # Calculate probability
            probability = signal_strength * confidence
            
            # Determine time horizon
            time_horizon = "short" if is_oversold or is_overbought else "medium"
            
            # Determine risk level
            risk_level = "low" if is_low_volatility else "medium"
            
            # Setup quality
            setup_quality = (signal_strength + confidence) / 2
            
            return TargetMetaLabel(
                target_type=TradingTarget.MEAN_REVERSION,
                signal_strength=signal_strength,
                confidence=confidence,
                probability=probability,
                time_horizon=time_horizon,
                risk_level=risk_level,
                setup_quality=setup_quality,
                entry_conditions={
                    "at_edge": is_at_edge,
                    "oversold": is_oversold,
                    "overbought": is_overbought,
                    "low_volatility": is_low_volatility
                },
                exit_conditions={
                    "mean_reversion": True,
                    "target_price": features["sma_20"].iloc[-1]
                },
                metadata={
                    "bb_position": bb_position,
                    "rsi": rsi,
                    "volatility": volatility,
                    "momentum": momentum
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating mean reversion labels: {e}")
            return self._create_empty_target_label(TradingTarget.MEAN_REVERSION)
    
    async def _generate_trend_following_labels(self, price_data: pd.DataFrame, features: Dict[str, Any]) -> TargetMetaLabel:
        """Generate trend following meta labels."""
        try:
            sma_20 = features["sma_20"].iloc[-1]
            sma_50 = features["sma_50"].iloc[-1]
            current_price = price_data["close"].iloc[-1]
            momentum = features["momentum_5"].iloc[-1]
            rsi = features["rsi"].iloc[-1]
            
            # Trend following conditions
            is_uptrend = current_price > sma_20 > sma_50
            is_downtrend = current_price < sma_20 < sma_50
            is_strong_momentum = abs(momentum) > 0.01
            is_healthy_rsi = 40 < rsi < 70
            
            # Calculate signal strength
            signal_strength = 0.0
            if is_uptrend or is_downtrend:
                signal_strength = abs(momentum) * 20  # Scale momentum
                if is_strong_momentum:
                    signal_strength *= 1.5
            
            # Calculate confidence
            confidence = 0.0
            if (is_uptrend or is_downtrend) and is_healthy_rsi:
                confidence = 0.7
                if is_strong_momentum:
                    confidence = 0.9
            
            # Calculate probability
            probability = signal_strength * confidence
            
            # Determine time horizon
            time_horizon = "long" if abs(momentum) > 0.02 else "medium"
            
            # Determine risk level
            risk_level = "medium" if is_healthy_rsi else "high"
            
            # Setup quality
            setup_quality = (signal_strength + confidence) / 2
            
            return TargetMetaLabel(
                target_type=TradingTarget.TREND_FOLLOWING,
                signal_strength=signal_strength,
                confidence=confidence,
                probability=probability,
                time_horizon=time_horizon,
                risk_level=risk_level,
                setup_quality=setup_quality,
                entry_conditions={
                    "uptrend": is_uptrend,
                    "downtrend": is_downtrend,
                    "strong_momentum": is_strong_momentum,
                    "healthy_rsi": is_healthy_rsi
                },
                exit_conditions={
                    "trend_reversal": True,
                    "stop_loss": sma_20
                },
                metadata={
                    "sma_20": sma_20,
                    "sma_50": sma_50,
                    "momentum": momentum,
                    "rsi": rsi
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating trend following labels: {e}")
            return self._create_empty_target_label(TradingTarget.TREND_FOLLOWING)
    
    async def _generate_rejection_labels(self, price_data: pd.DataFrame, features: Dict[str, Any]) -> TargetMetaLabel:
        """Generate rejection meta labels."""
        try:
            current_price = price_data["close"].iloc[-1]
            high = price_data["high"].iloc[-1]
            low = price_data["low"].iloc[-1]
            open_price = price_data["open"].iloc[-1]
            rsi = features["rsi"].iloc[-1]
            bb_position = features["bb_position"].iloc[-1]
            
            # Rejection conditions
            is_upper_rejection = (high - max(open_price, current_price)) / current_price > 0.01
            is_lower_rejection = (min(open_price, current_price) - low) / current_price > 0.01
            is_wick_rejection = is_upper_rejection or is_lower_rejection
            is_at_sr_level = bb_position < 0.2 or bb_position > 0.8
            is_extreme_rsi = rsi < 30 or rsi > 70
            
            # Calculate signal strength
            signal_strength = 0.0
            if is_wick_rejection:
                wick_size = max(
                    (high - max(open_price, current_price)) / current_price,
                    (min(open_price, current_price) - low) / current_price
                )
                signal_strength = min(wick_size * 50, 1.0)
            
            # Calculate confidence
            confidence = 0.0
            if is_wick_rejection and is_at_sr_level:
                confidence = 0.8
                if is_extreme_rsi:
                    confidence = 0.9
            
            # Calculate probability
            probability = signal_strength * confidence
            
            # Determine time horizon
            time_horizon = "short"
            
            # Determine risk level
            risk_level = "low" if is_at_sr_level else "medium"
            
            # Setup quality
            setup_quality = (signal_strength + confidence) / 2
            
            return TargetMetaLabel(
                target_type=TradingTarget.REJECTION,
                signal_strength=signal_strength,
                confidence=confidence,
                probability=probability,
                time_horizon=time_horizon,
                risk_level=risk_level,
                setup_quality=setup_quality,
                entry_conditions={
                    "upper_rejection": is_upper_rejection,
                    "lower_rejection": is_lower_rejection,
                    "at_sr_level": is_at_sr_level,
                    "extreme_rsi": is_extreme_rsi
                },
                exit_conditions={
                    "rejection_confirmed": True,
                    "target_price": current_price * 0.98 if is_upper_rejection else current_price * 1.02
                },
                metadata={
                    "wick_size": max(
                        (high - max(open_price, current_price)) / current_price,
                        (min(open_price, current_price) - low) / current_price
                    ),
                    "bb_position": bb_position,
                    "rsi": rsi
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating rejection labels: {e}")
            return self._create_empty_target_label(TradingTarget.REJECTION)
    
    async def _generate_sr_labels(self, price_data: pd.DataFrame, features: Dict[str, Any]) -> TargetMetaLabel:
        """Generate support/resistance meta labels."""
        try:
            current_price = price_data["close"].iloc[-1]
            recent_high = features["recent_high_20"].iloc[-1]
            recent_low = features["recent_low_20"].iloc[-1]
            bb_position = features["bb_position"].iloc[-1]
            volume_ratio = features.get("volume_ratio", pd.Series(1.0)).iloc[-1]
            
            # Support/Resistance conditions
            is_near_resistance = abs(current_price - recent_high) / current_price < 0.01
            is_near_support = abs(current_price - recent_low) / current_price < 0.01
            is_at_sr_level = is_near_resistance or is_near_support
            is_high_volume = volume_ratio > 1.5
            is_at_edge = bb_position < 0.2 or bb_position > 0.8
            
            # Calculate signal strength
            signal_strength = 0.0
            if is_at_sr_level:
                distance_to_level = min(
                    abs(current_price - recent_high) / current_price,
                    abs(current_price - recent_low) / current_price
                )
                signal_strength = 1.0 - (distance_to_level * 100)
                if is_high_volume:
                    signal_strength *= 1.2
            
            # Calculate confidence
            confidence = 0.0
            if is_at_sr_level and is_at_edge:
                confidence = 0.8
                if is_high_volume:
                    confidence = 0.9
            
            # Calculate probability
            probability = signal_strength * confidence
            
            # Determine time horizon
            time_horizon = "short" if is_high_volume else "medium"
            
            # Determine risk level
            risk_level = "low" if is_at_edge else "medium"
            
            # Setup quality
            setup_quality = (signal_strength + confidence) / 2
            
            return TargetMetaLabel(
                target_type=TradingTarget.SUPPORT_RESISTANCE,
                signal_strength=signal_strength,
                confidence=confidence,
                probability=probability,
                time_horizon=time_horizon,
                risk_level=risk_level,
                setup_quality=setup_quality,
                entry_conditions={
                    "near_resistance": is_near_resistance,
                    "near_support": is_near_support,
                    "high_volume": is_high_volume,
                    "at_edge": is_at_edge
                },
                exit_conditions={
                    "sr_break": True,
                    "target_price": recent_high * 1.02 if is_near_resistance else recent_low * 0.98
                },
                metadata={
                    "recent_high": recent_high,
                    "recent_low": recent_low,
                    "distance_to_level": min(
                        abs(current_price - recent_high) / current_price,
                        abs(current_price - recent_low) / current_price
                    ),
                    "volume_ratio": volume_ratio,
                    "bb_position": bb_position
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating SR labels: {e}")
            return self._create_empty_target_label(TradingTarget.SUPPORT_RESISTANCE)
    
    async def _generate_consolidation_labels(self, price_data: pd.DataFrame, features: Dict[str, Any]) -> TargetMetaLabel:
        """Generate consolidation meta labels."""
        try:
            bb_width = (features["bb_upper"].iloc[-1] - features["bb_lower"].iloc[-1]) / features["sma_20"].iloc[-1]
            volatility = features["volatility_20"].iloc[-1]
            momentum = features["momentum_5"].iloc[-1]
            bb_position = features["bb_position"].iloc[-1]
            
            # Consolidation conditions
            is_narrow_range = bb_width < 0.05
            is_low_volatility = volatility < 0.02
            is_sideways = abs(momentum) < 0.01
            is_in_middle = 0.3 < bb_position < 0.7
            
            # Calculate signal strength
            signal_strength = 0.0
            if is_narrow_range and is_low_volatility and is_sideways:
                signal_strength = 0.8
                if is_in_middle:
                    signal_strength = 0.9
            
            # Calculate confidence
            confidence = 0.0
            if is_narrow_range and is_low_volatility:
                confidence = 0.7
                if is_sideways and is_in_middle:
                    confidence = 0.9
            
            # Calculate probability
            probability = signal_strength * confidence
            
            # Determine time horizon
            time_horizon = "medium" if is_narrow_range else "long"
            
            # Determine risk level
            risk_level = "low" if is_low_volatility else "medium"
            
            # Setup quality
            setup_quality = (signal_strength + confidence) / 2
            
            return TargetMetaLabel(
                target_type=TradingTarget.CONSOLIDATION,
                signal_strength=signal_strength,
                confidence=confidence,
                probability=probability,
                time_horizon=time_horizon,
                risk_level=risk_level,
                setup_quality=setup_quality,
                entry_conditions={
                    "narrow_range": is_narrow_range,
                    "low_volatility": is_low_volatility,
                    "sideways": is_sideways,
                    "in_middle": is_in_middle
                },
                exit_conditions={
                    "breakout": True,
                    "target_price": features["bb_upper"].iloc[-1] if bb_position > 0.5 else features["bb_lower"].iloc[-1]
                },
                metadata={
                    "bb_width": bb_width,
                    "volatility": volatility,
                    "momentum": momentum,
                    "bb_position": bb_position
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating consolidation labels: {e}")
            return self._create_empty_target_label(TradingTarget.CONSOLIDATION)
    
    async def _generate_momentum_labels(self, price_data: pd.DataFrame, features: Dict[str, Any]) -> TargetMetaLabel:
        """Generate momentum meta labels."""
        try:
            momentum_5 = features["momentum_5"].iloc[-1]
            momentum_10 = features["momentum_10"].iloc[-1]
            rsi = features["rsi"].iloc[-1]
            macd = features["macd"].iloc[-1]
            macd_signal = features["macd_signal"].iloc[-1]
            
            # Momentum conditions
            is_strong_momentum = abs(momentum_5) > 0.02
            is_momentum_accelerating = abs(momentum_5) > abs(momentum_10)
            is_macd_bullish = macd > macd_signal
            is_macd_bearish = macd < macd_signal
            is_rsi_momentum = 40 < rsi < 70
            
            # Calculate signal strength
            signal_strength = 0.0
            if is_strong_momentum:
                signal_strength = min(abs(momentum_5) * 25, 1.0)
                if is_momentum_accelerating:
                    signal_strength *= 1.2
            
            # Calculate confidence
            confidence = 0.0
            if is_strong_momentum and is_rsi_momentum:
                confidence = 0.7
                if is_momentum_accelerating:
                    confidence = 0.9
            
            # Calculate probability
            probability = signal_strength * confidence
            
            # Determine time horizon
            time_horizon = "short" if abs(momentum_5) > 0.03 else "medium"
            
            # Determine risk level
            risk_level = "high" if abs(momentum_5) > 0.03 else "medium"
            
            # Setup quality
            setup_quality = (signal_strength + confidence) / 2
            
            return TargetMetaLabel(
                target_type=TradingTarget.MOMENTUM,
                signal_strength=signal_strength,
                confidence=confidence,
                probability=probability,
                time_horizon=time_horizon,
                risk_level=risk_level,
                setup_quality=setup_quality,
                entry_conditions={
                    "strong_momentum": is_strong_momentum,
                    "momentum_accelerating": is_momentum_accelerating,
                    "macd_bullish": is_macd_bullish,
                    "macd_bearish": is_macd_bearish,
                    "rsi_momentum": is_rsi_momentum
                },
                exit_conditions={
                    "momentum_fade": True,
                    "target_price": price_data["close"].iloc[-1] * (1 + momentum_5 * 2)
                },
                metadata={
                    "momentum_5": momentum_5,
                    "momentum_10": momentum_10,
                    "rsi": rsi,
                    "macd": macd,
                    "macd_signal": macd_signal
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating momentum labels: {e}")
            return self._create_empty_target_label(TradingTarget.MOMENTUM)
    
    async def _generate_volatility_expansion_labels(self, price_data: pd.DataFrame, features: Dict[str, Any]) -> TargetMetaLabel:
        """Generate volatility expansion meta labels."""
        try:
            volatility_20 = features["volatility_20"].iloc[-1]
            volatility_10 = features["volatility_10"].iloc[-1]
            volume_ratio = features.get("volume_ratio", pd.Series(1.0)).iloc[-1]
            bb_width = (features["bb_upper"].iloc[-1] - features["bb_lower"].iloc[-1]) / features["sma_20"].iloc[-1]
            
            # Volatility expansion conditions
            is_volatility_increasing = volatility_10 > volatility_20 * 1.2
            is_high_volume = volume_ratio > 1.5
            is_bb_expanding = bb_width > 0.05
            is_breakout_volatility = volatility_10 > 0.03
            
            # Calculate signal strength
            signal_strength = 0.0
            if is_volatility_increasing:
                signal_strength = min((volatility_10 / volatility_20 - 1) * 2, 1.0)
                if is_high_volume:
                    signal_strength *= 1.2
            
            # Calculate confidence
            confidence = 0.0
            if is_volatility_increasing and is_high_volume:
                confidence = 0.8
                if is_bb_expanding:
                    confidence = 0.9
            
            # Calculate probability
            probability = signal_strength * confidence
            
            # Determine time horizon
            time_horizon = "short"
            
            # Determine risk level
            risk_level = "high" if is_breakout_volatility else "medium"
            
            # Setup quality
            setup_quality = (signal_strength + confidence) / 2
            
            return TargetMetaLabel(
                target_type=TradingTarget.VOLATILITY_EXPANSION,
                signal_strength=signal_strength,
                confidence=confidence,
                probability=probability,
                time_horizon=time_horizon,
                risk_level=risk_level,
                setup_quality=setup_quality,
                entry_conditions={
                    "volatility_increasing": is_volatility_increasing,
                    "high_volume": is_high_volume,
                    "bb_expanding": is_bb_expanding,
                    "breakout_volatility": is_breakout_volatility
                },
                exit_conditions={
                    "volatility_peak": True,
                    "target_price": price_data["close"].iloc[-1] * (1 + volatility_10)
                },
                metadata={
                    "volatility_20": volatility_20,
                    "volatility_10": volatility_10,
                    "volume_ratio": volume_ratio,
                    "bb_width": bb_width
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating volatility expansion labels: {e}")
            return self._create_empty_target_label(TradingTarget.VOLATILITY_EXPANSION)
    
    async def _generate_volatility_compression_labels(self, price_data: pd.DataFrame, features: Dict[str, Any]) -> TargetMetaLabel:
        """Generate volatility compression meta labels."""
        try:
            volatility_20 = features["volatility_20"].iloc[-1]
            volatility_10 = features["volatility_10"].iloc[-1]
            bb_width = (features["bb_upper"].iloc[-1] - features["bb_lower"].iloc[-1]) / features["sma_20"].iloc[-1]
            momentum = features["momentum_5"].iloc[-1]
            
            # Volatility compression conditions
            is_volatility_decreasing = volatility_10 < volatility_20 * 0.8
            is_narrow_bb = bb_width < 0.03
            is_low_volatility = volatility_10 < 0.02
            is_sideways = abs(momentum) < 0.01
            
            # Calculate signal strength
            signal_strength = 0.0
            if is_volatility_decreasing and is_narrow_bb:
                signal_strength = 0.8
                if is_low_volatility and is_sideways:
                    signal_strength = 0.9
            
            # Calculate confidence
            confidence = 0.0
            if is_volatility_decreasing and is_narrow_bb:
                confidence = 0.7
                if is_low_volatility:
                    confidence = 0.9
            
            # Calculate probability
            probability = signal_strength * confidence
            
            # Determine time horizon
            time_horizon = "medium"
            
            # Determine risk level
            risk_level = "low" if is_low_volatility else "medium"
            
            # Setup quality
            setup_quality = (signal_strength + confidence) / 2
            
            return TargetMetaLabel(
                target_type=TradingTarget.VOLATILITY_COMPRESSION,
                signal_strength=signal_strength,
                confidence=confidence,
                probability=probability,
                time_horizon=time_horizon,
                risk_level=risk_level,
                setup_quality=setup_quality,
                entry_conditions={
                    "volatility_decreasing": is_volatility_decreasing,
                    "narrow_bb": is_narrow_bb,
                    "low_volatility": is_low_volatility,
                    "sideways": is_sideways
                },
                exit_conditions={
                    "volatility_expansion": True,
                    "target_price": features["bb_upper"].iloc[-1] if momentum > 0 else features["bb_lower"].iloc[-1]
                },
                metadata={
                    "volatility_20": volatility_20,
                    "volatility_10": volatility_10,
                    "bb_width": bb_width,
                    "momentum": momentum
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating volatility compression labels: {e}")
            return self._create_empty_target_label(TradingTarget.VOLATILITY_COMPRESSION)
    
    async def _generate_reversal_labels(self, price_data: pd.DataFrame, features: Dict[str, Any]) -> TargetMetaLabel:
        """Generate reversal meta labels."""
        try:
            rsi = features["rsi"].iloc[-1]
            macd = features["macd"].iloc[-1]
            macd_signal = features["macd_signal"].iloc[-1]
            momentum = features["momentum_5"].iloc[-1]
            bb_position = features["bb_position"].iloc[-1]
            
            # Reversal conditions
            is_rsi_divergence = (rsi > 70 and momentum < 0) or (rsi < 30 and momentum > 0)
            is_macd_divergence = (macd > macd_signal and momentum < 0) or (macd < macd_signal and momentum > 0)
            is_at_extreme = bb_position < 0.1 or bb_position > 0.9
            is_momentum_reversal = abs(momentum) < 0.005
            
            # Calculate signal strength
            signal_strength = 0.0
            if is_rsi_divergence or is_macd_divergence:
                signal_strength = 0.7
                if is_at_extreme:
                    signal_strength = 0.9
            
            # Calculate confidence
            confidence = 0.0
            if is_rsi_divergence and is_macd_divergence:
                confidence = 0.8
                if is_at_extreme:
                    confidence = 0.9
            
            # Calculate probability
            probability = signal_strength * confidence
            
            # Determine time horizon
            time_horizon = "short"
            
            # Determine risk level
            risk_level = "high" if is_at_extreme else "medium"
            
            # Setup quality
            setup_quality = (signal_strength + confidence) / 2
            
            return TargetMetaLabel(
                target_type=TradingTarget.REVERSAL,
                signal_strength=signal_strength,
                confidence=confidence,
                probability=probability,
                time_horizon=time_horizon,
                risk_level=risk_level,
                setup_quality=setup_quality,
                entry_conditions={
                    "rsi_divergence": is_rsi_divergence,
                    "macd_divergence": is_macd_divergence,
                    "at_extreme": is_at_extreme,
                    "momentum_reversal": is_momentum_reversal
                },
                exit_conditions={
                    "reversal_confirmed": True,
                    "target_price": price_data["close"].iloc[-1] * 0.98 if rsi > 70 else price_data["close"].iloc[-1] * 1.02
                },
                metadata={
                    "rsi": rsi,
                    "macd": macd,
                    "macd_signal": macd_signal,
                    "momentum": momentum,
                    "bb_position": bb_position
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating reversal labels: {e}")
            return self._create_empty_target_label(TradingTarget.REVERSAL)
    
    def _create_empty_target_label(self, target_type: TradingTarget) -> TargetMetaLabel:
        """Create empty target label for error cases."""
        return TargetMetaLabel(
            target_type=target_type,
            signal_strength=0.0,
            confidence=0.0,
            probability=0.0,
            time_horizon="short",
            risk_level="high",
            setup_quality=0.0,
            entry_conditions={},
            exit_conditions={},
            metadata={}
        )