#!/usr/bin/env python3
"""Enhanced S/R Breakout Predictor.

This module provides advanced breakout prediction capabilities with ML integration,
real-time monitoring, and comprehensive validation.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

from src.core.decorators import handles_errors
from src.utils.logger import system_logger
from src.core.sr_error_handlers import sr_error_handler, SROptimizationError, SRDataError


class BreakoutType(Enum):
    """Types of breakouts."""
    SUPPORT_BREAKDOWN = "support_breakdown"
    RESISTANCE_BREAKOUT = "resistance_breakout"
    FALSE_BREAKOUT = "false_breakout"
    CONSOLIDATION = "consolidation"


class BreakoutConfidence(Enum):
    """Breakout confidence levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class BreakoutSignal:
    """Breakout signal with detailed information."""
    level_id: str
    breakout_type: BreakoutType
    confidence: BreakoutConfidence
    probability: float
    expected_direction: str
    target_price: Optional[float]
    stop_loss: Optional[float]
    time_to_breakout: Optional[int]  # bars
    volume_confirmation: bool
    momentum_confirmation: bool
    features: Dict[str, float]
    timestamp: datetime
    validation_score: float


@dataclass
class BreakoutValidation:
    """Breakout validation result."""
    is_valid: bool
    false_breakout_probability: float
    confirmation_required: bool
    validation_metrics: Dict[str, float]
    recommended_action: str


@dataclass
class BreakoutPerformance:
    """Breakout prediction performance metrics."""
    total_predictions: int
    correct_predictions: int
    false_breakouts: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    profit_factor: float
    average_hold_time: float
    max_drawdown: float


class EnhancedSRBreakoutPredictor:
    """Enhanced S/R breakout predictor with ML integration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced breakout predictor."""
        self.config = config
        self.logger = system_logger.getChild("EnhancedSRBreakoutPredictor")
        self.breakout_config = config.get("breakout_prediction", {})
        
        # Prediction parameters
        self.proximity_threshold = self.breakout_config.get("breakout_detection", {}).get("proximity_threshold", 0.02)
        self.volume_spike_threshold = self.breakout_config.get("breakout_detection", {}).get("volume_spike_threshold", 1.5)
        self.momentum_threshold = self.breakout_config.get("breakout_detection", {}).get("momentum_threshold", 0.01)
        self.confirmation_bars = self.breakout_config.get("breakout_detection", {}).get("confirmation_bars", 2)
        
        # Validation parameters
        self.false_breakout_threshold = self.breakout_config.get("breakout_validation", {}).get("false_breakout_threshold", 0.03)
        self.confirmation_timeframe = self.breakout_config.get("breakout_validation", {}).get("confirmation_timeframe", "5m")
        self.min_breakout_duration = self.breakout_config.get("breakout_validation", {}).get("min_breakout_duration", 5)
        
        # Performance tracking
        self.prediction_history: List[BreakoutSignal] = []
        self.validation_results: List[BreakoutValidation] = []
        self.performance_metrics = BreakoutPerformance(
            total_predictions=0,
            correct_predictions=0,
            false_breakouts=0,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            profit_factor=0.0,
            average_hold_time=0.0,
            max_drawdown=0.0
        )
        
        # Real-time monitoring
        self.active_signals: Dict[str, BreakoutSignal] = {}
        self.monitoring_enabled = True
        
        # ML integration
        self.ml_model = None
        self.feature_importance = {}
    
    @sr_error_handler(
        exceptions=(SROptimizationError, SRDataError),
        default_return=[],
        context="breakout prediction",
        max_retries=2
    )
    async def predict_breakouts(
        self,
        market_data: pd.DataFrame,
        sr_levels: List[Dict[str, Any]],
        current_price: Optional[float] = None
    ) -> List[BreakoutSignal]:
        """Predict potential breakouts from S/R levels."""
        try:
            if not sr_levels:
                return []
            
            self.logger.info(f"🔮 Predicting breakouts for {len(sr_levels)} levels")
            
            current_price = current_price or market_data['close'].iloc[-1]
            breakout_signals = []
            
            # Analyze each level for breakout potential
            for level in sr_levels:
                signal = await self._analyze_level_for_breakout(
                    market_data, level, current_price
                )
                
                if signal and signal.probability > 0.3:  # Minimum threshold
                    breakout_signals.append(signal)
                    self.active_signals[signal.level_id] = signal
            
            # Sort by probability and confidence
            breakout_signals.sort(key=lambda x: (x.probability, x.confidence.value), reverse=True)
            
            # Update performance tracking
            self.performance_metrics.total_predictions += len(breakout_signals)
            self.prediction_history.extend(breakout_signals)
            
            self.logger.info(f"✅ Generated {len(breakout_signals)} breakout signals")
            return breakout_signals
            
        except Exception as e:
            self.logger.error(f"Breakout prediction failed: {e}")
            return []
    
    async def _analyze_level_for_breakout(
        self,
        market_data: pd.DataFrame,
        level: Dict[str, Any],
        current_price: float
    ) -> Optional[BreakoutSignal]:
        """Analyze a specific level for breakout potential."""
        try:
            level_price = level.get('price', 0)
            level_type = level.get('type', 'unknown')
            level_id = level.get('id', f"level_{level_price}")
            
            if level_price <= 0:
                return None
            
            # Calculate proximity
            proximity = abs(current_price - level_price) / level_price
            
            if proximity > self.proximity_threshold:
                return None  # Too far from level
            
            # Extract features
            features = await self._extract_breakout_features(
                market_data, level, current_price
            )
            
            # Calculate breakout probability
            probability = await self._calculate_breakout_probability(features)
            
            # Determine breakout type and direction
            breakout_type, direction = self._determine_breakout_type_and_direction(
                level_type, current_price, level_price
            )
            
            # Calculate confidence
            confidence = self._calculate_breakout_confidence(features, probability)
            
            # Calculate target and stop loss
            target_price, stop_loss = self._calculate_target_and_stop_loss(
                level_price, level_type, current_price, features
            )
            
            # Estimate time to breakout
            time_to_breakout = self._estimate_time_to_breakout(features)
            
            # Check confirmations
            volume_confirmation = features.get('volume_spike', 0) > self.volume_spike_threshold
            momentum_confirmation = features.get('momentum', 0) > self.momentum_threshold
            
            # Calculate validation score
            validation_score = self._calculate_validation_score(features)
            
            return BreakoutSignal(
                level_id=level_id,
                breakout_type=breakout_type,
                confidence=confidence,
                probability=probability,
                expected_direction=direction,
                target_price=target_price,
                stop_loss=stop_loss,
                time_to_breakout=time_to_breakout,
                volume_confirmation=volume_confirmation,
                momentum_confirmation=momentum_confirmation,
                features=features,
                timestamp=datetime.now(),
                validation_score=validation_score
            )
            
        except Exception as e:
            self.logger.error(f"Level analysis failed: {e}")
            return None
    
    async def _extract_breakout_features(
        self,
        market_data: pd.DataFrame,
        level: Dict[str, Any],
        current_price: float
    ) -> Dict[str, float]:
        """Extract features for breakout prediction (12 specific factors)."""
        try:
            features = {}
            
            # Factor 1: Proximity to Level (0-1, closer = higher breakout probability)
            level_price = level.get('price', 0)
            features['proximity_to_level'] = abs(current_price - level_price) / level_price
            
            # Factor 2: Volume Spike (1.0+ = normal, >1.5 = spike)
            features['volume_spike'] = self._calculate_volume_spike(market_data)
            
            # Factor 3: Price Momentum (-1 to +1, positive = upward momentum)
            features['momentum'] = self._calculate_momentum(market_data)
            
            # Factor 4: Volatility (0-1, higher = more likely to break)
            features['volatility'] = self._calculate_volatility(market_data)
            
            # Factor 5: Time at Level (bars, longer = more likely to break)
            features['time_at_level'] = self._calculate_time_at_level(market_data, level_price)
            
            # Factor 6: Level Strength (0-1, weaker = more likely to break)
            features['level_strength'] = level.get('strength', 0.5)
            
            # Factor 7: Touch Count (number of previous touches)
            features['touch_count'] = level.get('touch_count', 0)
            
            # Factor 8: RSI Position (0-100, extremes = more likely to break)
            features['rsi'] = self._calculate_rsi(market_data['close'])
            
            # Factor 9: MACD Signal (momentum confirmation)
            features['macd_signal'] = self._calculate_macd_signal(market_data['close'])
            
            # Factor 10: Bollinger Band Position (0-1, extremes = more likely to break)
            features['bollinger_position'] = self._calculate_bollinger_position(market_data)
            
            # Factor 11: Order Flow Imbalance (-1 to +1, imbalance = more likely to break)
            features['order_flow_imbalance'] = self._calculate_order_flow_imbalance(market_data)
            
            # Factor 12: Market Sentiment (0-1, extreme sentiment = more likely to break)
            features['market_sentiment'] = self._calculate_market_sentiment(market_data)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return {}
    
    async def _calculate_breakout_probability(self, features: Dict[str, float]) -> float:
        """Calculate breakout probability using features."""
        try:
            # Base probability from proximity
            proximity = features.get('proximity_to_level', 1.0)
            base_prob = max(0.0, 1.0 - proximity / self.proximity_threshold)
            
            # Adjust based on level strength
            strength = features.get('level_strength', 0.5)
            strength_factor = 1.0 - (strength * 0.3)  # Stronger levels less likely to break
            
            # Volume confirmation
            volume_spike = features.get('volume_spike', 1.0)
            volume_factor = min(volume_spike / self.volume_spike_threshold, 1.5)
            
            # Momentum confirmation
            momentum = features.get('momentum', 0.0)
            momentum_factor = 1.0 + (momentum * 2.0)  # Positive momentum increases probability
            
            # Volatility factor
            volatility = features.get('volatility', 0.0)
            volatility_factor = 1.0 + (volatility * 0.5)  # Higher volatility increases probability
            
            # Technical indicators
            rsi = features.get('rsi', 50.0)
            rsi_factor = 1.0
            if rsi > 70:  # Overbought - more likely to break resistance
                rsi_factor = 1.2
            elif rsi < 30:  # Oversold - more likely to break support
                rsi_factor = 1.2
            
            # Combine factors
            probability = base_prob * strength_factor * volume_factor * momentum_factor * volatility_factor * rsi_factor
            
            # Apply ML model if available
            if self.ml_model:
                ml_probability = await self._get_ml_prediction(features)
                probability = (probability * 0.6) + (ml_probability * 0.4)  # Weighted combination
            
            return min(max(probability, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Probability calculation failed: {e}")
            return 0.0
    
    def _determine_breakout_type_and_direction(
        self,
        level_type: str,
        current_price: float,
        level_price: float
    ) -> Tuple[BreakoutType, str]:
        """Determine breakout type and direction."""
        try:
            if level_type == "resistance":
                if current_price > level_price:
                    return BreakoutType.RESISTANCE_BREAKOUT, "up"
                else:
                    return BreakoutType.CONSOLIDATION, "sideways"
            elif level_type == "support":
                if current_price < level_price:
                    return BreakoutType.SUPPORT_BREAKDOWN, "down"
                else:
                    return BreakoutType.CONSOLIDATION, "sideways"
            else:
                return BreakoutType.CONSOLIDATION, "sideways"
                
        except Exception as e:
            self.logger.error(f"Breakout type determination failed: {e}")
            return BreakoutType.CONSOLIDATION, "sideways"
    
    def _calculate_breakout_confidence(
        self,
        features: Dict[str, float],
        probability: float
    ) -> BreakoutConfidence:
        """Calculate breakout confidence level."""
        try:
            confidence_score = 0.0
            
            # Base confidence from probability
            confidence_score += probability * 0.4
            
            # Volume confirmation
            if features.get('volume_spike', 0) > self.volume_spike_threshold:
                confidence_score += 0.2
            
            # Momentum confirmation
            if features.get('momentum', 0) > self.momentum_threshold:
                confidence_score += 0.2
            
            # Level strength (weaker levels more likely to break)
            strength = features.get('level_strength', 0.5)
            confidence_score += (1.0 - strength) * 0.1
            
            # Technical indicators
            rsi = features.get('rsi', 50.0)
            if rsi > 70 or rsi < 30:  # Extreme RSI values
                confidence_score += 0.1
            
            # Determine confidence level
            if confidence_score >= 0.8:
                return BreakoutConfidence.VERY_HIGH
            elif confidence_score >= 0.6:
                return BreakoutConfidence.HIGH
            elif confidence_score >= 0.4:
                return BreakoutConfidence.MEDIUM
            else:
                return BreakoutConfidence.LOW
                
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return BreakoutConfidence.LOW
    
    def _calculate_target_and_stop_loss(
        self,
        level_price: float,
        level_type: str,
        current_price: float,
        features: Dict[str, float]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate target price and stop loss."""
        try:
            volatility = features.get('volatility', 0.01)
            atr_multiplier = 2.0  # 2x ATR for target
            
            if level_type == "resistance" and current_price > level_price:
                # Resistance breakout
                target_price = level_price * (1 + volatility * atr_multiplier)
                stop_loss = level_price * (1 - volatility * 0.5)
            elif level_type == "support" and current_price < level_price:
                # Support breakdown
                target_price = level_price * (1 - volatility * atr_multiplier)
                stop_loss = level_price * (1 + volatility * 0.5)
            else:
                return None, None
            
            return target_price, stop_loss
            
        except Exception as e:
            self.logger.error(f"Target/stop loss calculation failed: {e}")
            return None, None
    
    def _estimate_time_to_breakout(self, features: Dict[str, float]) -> Optional[int]:
        """Estimate time to breakout in bars."""
        try:
            # Base estimation from proximity and momentum
            proximity = features.get('proximity_to_level', 1.0)
            momentum = features.get('momentum', 0.0)
            
            # Closer to level and higher momentum = faster breakout
            base_time = int(proximity * 100)  # Base time in bars
            momentum_adjustment = int(momentum * 50)  # Momentum adjustment
            
            estimated_time = max(1, base_time - momentum_adjustment)
            return min(estimated_time, 100)  # Cap at 100 bars
            
        except Exception as e:
            self.logger.error(f"Time estimation failed: {e}")
            return None
    
    def _calculate_validation_score(self, features: Dict[str, float]) -> float:
        """Calculate validation score for the breakout signal."""
        try:
            score = 0.0
            
            # Volume confirmation
            if features.get('volume_spike', 0) > self.volume_spike_threshold:
                score += 0.3
            
            # Momentum confirmation
            if features.get('momentum', 0) > self.momentum_threshold:
                score += 0.3
            
            # Technical indicators alignment
            rsi = features.get('rsi', 50.0)
            if 30 < rsi < 70:  # Not overbought/oversold
                score += 0.2
            
            # Level strength (weaker levels more likely to break)
            strength = features.get('level_strength', 0.5)
            score += (1.0 - strength) * 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Validation score calculation failed: {e}")
            return 0.0
    
    async def validate_breakout(
        self,
        signal: BreakoutSignal,
        market_data: pd.DataFrame
    ) -> BreakoutValidation:
        """Validate a breakout signal."""
        try:
            # Check for false breakout patterns
            false_breakout_prob = self._calculate_false_breakout_probability(signal, market_data)
            
            # Check confirmation requirements
            confirmation_required = signal.confidence in [BreakoutConfidence.LOW, BreakoutConfidence.MEDIUM]
            
            # Calculate validation metrics
            validation_metrics = {
                "false_breakout_probability": false_breakout_prob,
                "volume_confirmation": signal.volume_confirmation,
                "momentum_confirmation": signal.momentum_confirmation,
                "validation_score": signal.validation_score
            }
            
            # Determine if signal is valid
            is_valid = (
                false_breakout_prob < 0.3 and
                signal.validation_score > 0.5 and
                (signal.volume_confirmation or signal.momentum_confirmation)
            )
            
            # Recommend action
            if is_valid and signal.confidence in [BreakoutConfidence.HIGH, BreakoutConfidence.VERY_HIGH]:
                recommended_action = "enter_position"
            elif is_valid:
                recommended_action = "wait_for_confirmation"
            else:
                recommended_action = "avoid"
            
            validation = BreakoutValidation(
                is_valid=is_valid,
                false_breakout_probability=false_breakout_prob,
                confirmation_required=confirmation_required,
                validation_metrics=validation_metrics,
                recommended_action=recommended_action
            )
            
            self.validation_results.append(validation)
            return validation
            
        except Exception as e:
            self.logger.error(f"Breakout validation failed: {e}")
            return BreakoutValidation(
                is_valid=False,
                false_breakout_probability=1.0,
                confirmation_required=True,
                validation_metrics={},
                recommended_action="avoid"
            )
    
    def _calculate_false_breakout_probability(
        self,
        signal: BreakoutSignal,
        market_data: pd.DataFrame
    ) -> float:
        """Calculate probability of false breakout."""
        try:
            false_prob = 0.0
            
            # High volatility increases false breakout probability
            volatility = signal.features.get('volatility', 0.0)
            false_prob += min(volatility * 2.0, 0.3)
            
            # Weak volume confirmation increases false breakout probability
            if not signal.volume_confirmation:
                false_prob += 0.2
            
            # Weak momentum increases false breakout probability
            if not signal.momentum_confirmation:
                false_prob += 0.2
            
            # Strong level increases false breakout probability
            strength = signal.features.get('level_strength', 0.5)
            false_prob += strength * 0.3
            
            return min(false_prob, 1.0)
            
        except Exception as e:
            self.logger.error(f"False breakout probability calculation failed: {e}")
            return 0.5
    
    async def monitor_active_signals(
        self,
        market_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Monitor active breakout signals for updates."""
        try:
            if not self.monitoring_enabled:
                return []
            
            updates = []
            current_price = market_data['close'].iloc[-1]
            
            for signal_id, signal in list(self.active_signals.items()):
                # Check if signal is still valid
                time_elapsed = (datetime.now() - signal.timestamp).total_seconds() / 60  # minutes
                
                if time_elapsed > 60:  # Remove signals older than 1 hour
                    del self.active_signals[signal_id]
                    continue
                
                # Check for breakout confirmation
                level_price = signal.features.get('proximity_to_level', 0) * current_price + current_price
                
                if signal.expected_direction == "up" and current_price > level_price * 1.01:
                    # Breakout confirmed
                    updates.append({
                        "signal_id": signal_id,
                        "status": "confirmed",
                        "current_price": current_price,
                        "breakout_price": level_price
                    })
                    del self.active_signals[signal_id]
                elif signal.expected_direction == "down" and current_price < level_price * 0.99:
                    # Breakdown confirmed
                    updates.append({
                        "signal_id": signal_id,
                        "status": "confirmed",
                        "current_price": current_price,
                        "breakout_price": level_price
                    })
                    del self.active_signals[signal_id]
                elif time_elapsed > 30:  # Check for false breakout after 30 minutes
                    if abs(current_price - level_price) / level_price > 0.02:
                        # False breakout
                        updates.append({
                            "signal_id": signal_id,
                            "status": "false_breakout",
                            "current_price": current_price,
                            "breakout_price": level_price
                        })
                        del self.active_signals[signal_id]
            
            return updates
            
        except Exception as e:
            self.logger.error(f"Signal monitoring failed: {e}")
            return []
    
    def update_performance_metrics(self, validation_result: BreakoutValidation) -> None:
        """Update performance metrics based on validation results."""
        try:
            if validation_result.is_valid:
                self.performance_metrics.correct_predictions += 1
            else:
                self.performance_metrics.false_breakouts += 1
            
            # Recalculate metrics
            total = self.performance_metrics.total_predictions
            if total > 0:
                self.performance_metrics.accuracy = self.performance_metrics.correct_predictions / total
                self.performance_metrics.precision = self.performance_metrics.correct_predictions / (self.performance_metrics.correct_predictions + self.performance_metrics.false_breakouts)
            
        except Exception as e:
            self.logger.error(f"Performance metrics update failed: {e}")
    
    # Technical indicator calculation methods
    def _calculate_volume_spike(self, market_data: pd.DataFrame) -> float:
        """Calculate volume spike ratio."""
        try:
            if len(market_data) < 20:
                return 1.0
            
            current_volume = market_data['volume'].iloc[-1]
            avg_volume = market_data['volume'].rolling(window=20).mean().iloc[-1]
            
            return current_volume / avg_volume if avg_volume > 0 else 1.0
            
        except Exception:
            return 1.0
    
    def _calculate_momentum(self, market_data: pd.DataFrame) -> float:
        """Calculate price momentum."""
        try:
            if len(market_data) < 10:
                return 0.0
            
            current_price = market_data['close'].iloc[-1]
            past_price = market_data['close'].iloc[-10]
            
            return (current_price - past_price) / past_price
            
        except Exception:
            return 0.0
    
    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate price volatility."""
        try:
            if len(market_data) < 20:
                return 0.01
            
            returns = market_data['close'].pct_change().dropna()
            return returns.std() if len(returns) > 0 else 0.01
            
        except Exception:
            return 0.01
    
    def _calculate_time_at_level(self, market_data: pd.DataFrame, level_price: float) -> int:
        """Calculate time spent at level."""
        try:
            proximity_threshold = 0.005  # 0.5%
            time_at_level = 0
            
            for i in range(len(market_data) - 1, -1, -1):
                price = market_data['close'].iloc[i]
                proximity = abs(price - level_price) / level_price
                
                if proximity <= proximity_threshold:
                    time_at_level += 1
                else:
                    break
            
            return time_at_level
            
        except Exception:
            return 0
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not rsi.empty else 50.0
            
        except Exception:
            return 50.0
    
    def _calculate_macd_signal(self, prices: pd.Series) -> float:
        """Calculate MACD signal."""
        try:
            if len(prices) < 26:
                return 0.0
            
            ema_12 = prices.ewm(span=12).mean()
            ema_26 = prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            
            return (macd.iloc[-1] - signal.iloc[-1]) if not macd.empty else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_bollinger_position(self, market_data: pd.DataFrame) -> float:
        """Calculate Bollinger Band position."""
        try:
            if len(market_data) < 20:
                return 0.5
            
            prices = market_data['close']
            sma = prices.rolling(window=20).mean()
            std = prices.rolling(window=20).std()
            upper = sma + (std * 2)
            lower = sma - (std * 2)
            
            current_price = prices.iloc[-1]
            position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
            
            return position if not np.isnan(position) else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_order_flow_imbalance(self, market_data: pd.DataFrame) -> float:
        """Calculate order flow imbalance (simplified)."""
        try:
            # Simplified calculation based on volume and price movement
            if len(market_data) < 5:
                return 0.0
            
            recent_data = market_data.tail(5)
            volume_up = recent_data[recent_data['close'] > recent_data['open']]['volume'].sum()
            volume_down = recent_data[recent_data['close'] < recent_data['open']]['volume'].sum()
            
            total_volume = volume_up + volume_down
            if total_volume == 0:
                return 0.0
            
            return (volume_up - volume_down) / total_volume
            
        except Exception:
            return 0.0
    
    def _calculate_market_sentiment(self, market_data: pd.DataFrame) -> float:
        """Calculate market sentiment (simplified)."""
        try:
            # Simplified sentiment based on recent price action
            if len(market_data) < 10:
                return 0.0
            
            recent_returns = market_data['close'].pct_change().tail(10)
            positive_returns = (recent_returns > 0).sum()
            
            return positive_returns / len(recent_returns)
            
        except Exception:
            return 0.5
    
    def _get_previous_breakout_history(self, level: Dict[str, Any]) -> float:
        """Get previous breakout history for level."""
        try:
            # Simplified - would need historical data
            return 0.5  # Neutral history
            
        except Exception:
            return 0.5
    
    async def _get_ml_prediction(self, features: Dict[str, float]) -> float:
        """Get ML model prediction."""
        try:
            if not self.ml_model:
                return 0.5
            
            # Convert features to array and predict
            feature_array = np.array([list(features.values())])
            prediction = self.ml_model.predict_proba(feature_array)[0][1]  # Probability of breakout
            
            return float(prediction)
            
        except Exception as e:
            self.logger.error(f"ML prediction failed: {e}")
            return 0.5
    
    def get_performance_metrics(self) -> BreakoutPerformance:
        """Get current performance metrics."""
        return self.performance_metrics
    
    def get_active_signals(self) -> Dict[str, BreakoutSignal]:
        """Get currently active signals."""
        return self.active_signals.copy()
    
    def enable_monitoring(self) -> None:
        """Enable real-time monitoring."""
        self.monitoring_enabled = True
        self.logger.info("✅ Breakout monitoring enabled")
    
    def disable_monitoring(self) -> None:
        """Disable real-time monitoring."""
        self.monitoring_enabled = False
        self.logger.info("✅ Breakout monitoring disabled")