"""
Improved Trading Strategies for Backtesting

This module provides real trading strategies to replace the mock/random implementations
found in the backtesting codebase. These strategies implement proper market analysis,
risk management, and position sizing.

Key Features:
- Real market analysis instead of random signals
- Proper risk management with stop-losses and take-profits
- Dynamic position sizing based on volatility
- Multiple strategy types with different market regime adaptations
- Comprehensive signal validation and filtering
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

# Import memory optimization
try:
    from .memory_optimizer import get_backtesting_memory_optimizer
    MEMORY_OPTIMIZATION_AVAILABLE = True
except ImportError:
    MEMORY_OPTIMIZATION_AVAILABLE = False

# Import math validation
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, validate_finite, validate_positive
)

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of trading strategies."""
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    MOMENTUM = "momentum"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    MULTI_TIMEFRAME = "multi_timeframe"
    ADAPTIVE = "adaptive"


class MarketRegime(Enum):
    """Market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class TradingSignal:
    """Trading signal with comprehensive information."""
    action: str  # 'buy', 'sell', 'hold'
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: float = 0.1  # Fraction of portfolio
    reasoning: str = ""
    timestamp: Optional[datetime] = None
    market_regime: Optional[MarketRegime] = None
    risk_reward_ratio: Optional[float] = None


@dataclass
class StrategyConfig:
    """Configuration for trading strategies."""
    strategy_type: StrategyType = StrategyType.ADAPTIVE
    
    # Risk management
    max_position_size: float = 0.2
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.04  # 4%
    max_daily_risk: float = 0.05  # 5%
    
    # Technical indicators
    short_ma_period: int = 10
    long_ma_period: int = 50
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    atr_period: int = 14
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    
    # Volatility parameters
    volatility_lookback: int = 20
    volatility_threshold_high: float = 0.03  # 3% daily
    volatility_threshold_low: float = 0.01   # 1% daily
    
    # Signal filtering
    min_signal_confidence: float = 0.6
    min_risk_reward_ratio: float = 1.5
    max_correlation_threshold: float = 0.8
    
    # Position sizing
    enable_dynamic_sizing: bool = True
    base_position_size: float = 0.1
    volatility_scaling: bool = True
    kelly_criterion: bool = False


class TechnicalIndicators:
    """Technical indicators for trading strategies."""
    
    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period).mean()
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram


class MarketRegimeDetector:
    """Detect market regimes for strategy adaptation."""
    
    def __init__(self, config: StrategyConfig):
        """Initialize market regime detector."""
        self.config = config
        self.logger = logger.getChild('MarketRegimeDetector')
    
    def detect_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime."""
        if len(data) < self.config.volatility_lookback:
            return MarketRegime.SIDEWAYS
        
        # Calculate recent volatility
        returns = data['close'].pct_change().dropna()
        recent_volatility = returns.tail(self.config.volatility_lookback).std()
        
        # Calculate trend strength
        short_ma = TechnicalIndicators.calculate_sma(data['close'], self.config.short_ma_period)
        long_ma = TechnicalIndicators.calculate_sma(data['close'], self.config.long_ma_period)
        
        if len(short_ma) == 0 or len(long_ma) == 0:
            return MarketRegime.SIDEWAYS
        
        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        
        # Determine regime
        if recent_volatility > self.config.volatility_threshold_high:
            return MarketRegime.HIGH_VOLATILITY
        elif recent_volatility < self.config.volatility_threshold_low:
            return MarketRegime.LOW_VOLATILITY
        elif current_short > current_long * 1.02:  # 2% above
            return MarketRegime.TRENDING_UP
        elif current_short < current_long * 0.98:  # 2% below
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.SIDEWAYS


class RiskManager:
    """Risk management for trading strategies."""
    
    def __init__(self, config: StrategyConfig):
        """Initialize risk manager."""
        self.config = config
        self.logger = logger.getChild('RiskManager')
        self.daily_risk_used = 0.0
        self.open_positions = []
    
    def calculate_position_size(
        self, 
        signal: TradingSignal, 
        portfolio_value: float, 
        volatility: float
    ) -> float:
        """Calculate optimal position size."""
        base_size = self.config.base_position_size
        
        if not self.config.enable_dynamic_sizing:
            return min(base_size, self.config.max_position_size)
        
        # Adjust for volatility
        if self.config.volatility_scaling:
            volatility_adjustment = 1.0 / (1.0 + volatility * 10)  # Reduce size in high volatility
            base_size *= volatility_adjustment
        
        # Adjust for signal confidence
        confidence_adjustment = signal.confidence
        base_size *= confidence_adjustment
        
        # Adjust for risk-reward ratio
        if signal.risk_reward_ratio and signal.risk_reward_ratio > 0:
            rr_adjustment = min(2.0, signal.risk_reward_ratio / self.config.min_risk_reward_ratio)
            base_size *= rr_adjustment
        
        # Apply maximum position size limit
        final_size = min(base_size, self.config.max_position_size)
        
        # Check daily risk limit
        position_risk = final_size * self.config.stop_loss_pct
        if self.daily_risk_used + position_risk > self.config.max_daily_risk:
            # Reduce position size to stay within daily risk limit
            available_risk = self.config.max_daily_risk - self.daily_risk_used
            final_size = available_risk / self.config.stop_loss_pct
            final_size = max(0.0, final_size)
        
        return final_size
    
    def calculate_stop_loss_take_profit(
        self, 
        entry_price: float, 
        action: str, 
        atr: float,
        volatility: float
    ) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit levels."""
        # Base percentages
        sl_pct = self.config.stop_loss_pct
        tp_pct = self.config.take_profit_pct
        
        # Adjust based on volatility
        volatility_multiplier = 1.0 + (volatility * 2)  # Increase SL/TP in volatile markets
        sl_pct *= volatility_multiplier
        tp_pct *= volatility_multiplier
        
        # Adjust based on ATR
        if atr > 0:
            atr_pct = atr / entry_price
            sl_pct = max(sl_pct, atr_pct * 1.5)  # At least 1.5x ATR for stop loss
            tp_pct = max(tp_pct, atr_pct * 2.5)  # At least 2.5x ATR for take profit
        
        # Calculate levels
        if action == 'buy':
            stop_loss = entry_price * (1 - sl_pct)
            take_profit = entry_price * (1 + tp_pct)
        else:  # sell
            stop_loss = entry_price * (1 + sl_pct)
            take_profit = entry_price * (1 - tp_pct)
        
        return stop_loss, take_profit


class ImprovedTradingStrategy:
    """Improved trading strategy with real market analysis."""
    
    def __init__(self, config: StrategyConfig):
        """Initialize improved trading strategy."""
        self.config = config
        self.logger = logger.getChild('ImprovedTradingStrategy')
        
        # Initialize components
        self.regime_detector = MarketRegimeDetector(config)
        self.risk_manager = RiskManager(config)
        self.indicators = TechnicalIndicators()
        
        # Initialize memory optimizer if available
        self.memory_optimizer = None
        if MEMORY_OPTIMIZATION_AVAILABLE:
            try:
                self.memory_optimizer = get_backtesting_memory_optimizer()
            except Exception as e:
                self.logger.warning(f"Memory optimizer not available: {e}")
        
        self.logger.info(f"✅ ImprovedTradingStrategy initialized ({config.strategy_type.value})")
    
    def generate_signal(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> TradingSignal:
        """Generate trading signal based on market analysis."""
        try:
            # Get current market data
            current_idx = data.index.get_loc(timestamp)
            current_data = data.iloc[:current_idx + 1]
            
            if len(current_data) < max(self.config.long_ma_period, self.config.rsi_period):
                return self._create_hold_signal(data.loc[timestamp, 'close'], "Insufficient data")
            
            # Detect market regime
            regime = self.regime_detector.detect_regime(current_data)
            
            # Generate signal based on strategy type and regime
            if self.config.strategy_type == StrategyType.TREND_FOLLOWING:
                signal = self._trend_following_signal(current_data, regime)
            elif self.config.strategy_type == StrategyType.MEAN_REVERSION:
                signal = self._mean_reversion_signal(current_data, regime)
            elif self.config.strategy_type == StrategyType.MOMENTUM:
                signal = self._momentum_signal(current_data, regime)
            elif self.config.strategy_type == StrategyType.VOLATILITY_BREAKOUT:
                signal = self._volatility_breakout_signal(current_data, regime)
            elif self.config.strategy_type == StrategyType.ADAPTIVE:
                signal = self._adaptive_signal(current_data, regime)
            else:
                signal = self._trend_following_signal(current_data, regime)
            
            # Add timestamp and regime to signal
            signal.timestamp = timestamp
            signal.market_regime = regime
            
            # Validate and filter signal
            signal = self._validate_signal(signal, current_data)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Error generating signal: {e}")
            return self._create_hold_signal(data.loc[timestamp, 'close'], f"Error: {e}")
    
    def _trend_following_signal(self, data: pd.DataFrame, regime: MarketRegime) -> TradingSignal:
        """Generate trend following signal."""
        current_price = data['close'].iloc[-1]
        
        # Calculate moving averages
        short_ma = self.indicators.calculate_sma(data['close'], self.config.short_ma_period)
        long_ma = self.indicators.calculate_sma(data['close'], self.config.long_ma_period)
        
        if len(short_ma) == 0 or len(long_ma) == 0:
            return self._create_hold_signal(current_price, "Insufficient MA data")
        
        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2] if len(short_ma) > 1 else current_short
        prev_long = long_ma.iloc[-2] if len(long_ma) > 1 else current_long
        
        # Calculate ATR for stop loss/take profit
        atr = self.indicators.calculate_atr(data['high'], data['low'], data['close'], self.config.atr_period)
        current_atr = atr.iloc[-1] if len(atr) > 0 else current_price * 0.02
        
        # Calculate volatility
        returns = data['close'].pct_change().dropna()
        volatility = returns.tail(self.config.volatility_lookback).std()
        
        # Generate signal
        action = "hold"
        confidence = 0.5
        reasoning = "No clear trend"
        
        # Check for golden cross (bullish)
        if (current_short > current_long and prev_short <= prev_long and 
            regime in [MarketRegime.TRENDING_UP, MarketRegime.LOW_VOLATILITY]):
            action = "buy"
            confidence = 0.8
            reasoning = "Golden cross with favorable regime"
        
        # Check for death cross (bearish)
        elif (current_short < current_long and prev_short >= prev_long and 
              regime in [MarketRegime.TRENDING_DOWN, MarketRegime.HIGH_VOLATILITY]):
            action = "sell"
            confidence = 0.7
            reasoning = "Death cross with bearish regime"
        
        # Adjust confidence based on regime
        if regime == MarketRegime.HIGH_VOLATILITY:
            confidence *= 0.8  # Reduce confidence in volatile markets
        elif regime == MarketRegime.SIDEWAYS:
            confidence *= 0.6  # Reduce confidence in sideways markets
        
        # Calculate stop loss and take profit
        stop_loss, take_profit = self.risk_manager.calculate_stop_loss_take_profit(
            current_price, action, current_atr, volatility
        )
        
        # Calculate risk-reward ratio
        if action == "buy":
            risk = current_price - stop_loss
            reward = take_profit - current_price
        elif action == "sell":
            risk = stop_loss - current_price
            reward = current_price - take_profit
        else:
            risk = reward = 0
        
        risk_reward_ratio = safe_divide(reward, risk, 0.0) if risk > 0 else 0.0
        
        return TradingSignal(
            action=action,
            strength=self._determine_signal_strength(confidence),
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss if action != "hold" else None,
            take_profit=take_profit if action != "hold" else None,
            reasoning=reasoning,
            risk_reward_ratio=risk_reward_ratio
        )
    
    def _mean_reversion_signal(self, data: pd.DataFrame, regime: MarketRegime) -> TradingSignal:
        """Generate mean reversion signal."""
        current_price = data['close'].iloc[-1]
        
        # Calculate Bollinger Bands
        upper_bb, middle_bb, lower_bb = self.indicators.calculate_bollinger_bands(
            data['close'], self.config.bollinger_period, self.config.bollinger_std
        )
        
        if len(upper_bb) == 0:
            return self._create_hold_signal(current_price, "Insufficient BB data")
        
        current_upper = upper_bb.iloc[-1]
        current_middle = middle_bb.iloc[-1]
        current_lower = lower_bb.iloc[-1]
        
        # Calculate RSI
        rsi = self.indicators.calculate_rsi(data['close'], self.config.rsi_period)
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # Generate signal
        action = "hold"
        confidence = 0.5
        reasoning = "No mean reversion opportunity"
        
        # Oversold condition (buy signal)
        if (current_price <= current_lower and current_rsi <= self.config.rsi_oversold and
            regime not in [MarketRegime.TRENDING_DOWN]):
            action = "buy"
            confidence = 0.75
            reasoning = "Oversold mean reversion opportunity"
        
        # Overbought condition (sell signal)
        elif (current_price >= current_upper and current_rsi >= self.config.rsi_overbought and
              regime not in [MarketRegime.TRENDING_UP]):
            action = "sell"
            confidence = 0.75
            reasoning = "Overbought mean reversion opportunity"
        
        # Adjust confidence based on regime
        if regime == MarketRegime.SIDEWAYS:
            confidence *= 1.2  # Mean reversion works better in sideways markets
        elif regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            confidence *= 0.7  # Mean reversion is riskier in trending markets
        
        confidence = min(1.0, confidence)
        
        # Calculate stop loss and take profit
        volatility = data['close'].pct_change().tail(self.config.volatility_lookback).std()
        atr = self.indicators.calculate_atr(data['high'], data['low'], data['close'], self.config.atr_period)
        current_atr = atr.iloc[-1] if len(atr) > 0 else current_price * 0.02
        
        stop_loss, take_profit = self.risk_manager.calculate_stop_loss_take_profit(
            current_price, action, current_atr, volatility
        )
        
        # For mean reversion, use tighter targets
        if action == "buy":
            take_profit = min(take_profit, current_middle)  # Target middle band
        elif action == "sell":
            take_profit = max(take_profit, current_middle)  # Target middle band
        
        # Calculate risk-reward ratio
        if action == "buy":
            risk = current_price - stop_loss
            reward = take_profit - current_price
        elif action == "sell":
            risk = stop_loss - current_price
            reward = current_price - take_profit
        else:
            risk = reward = 0
        
        risk_reward_ratio = safe_divide(reward, risk, 0.0) if risk > 0 else 0.0
        
        return TradingSignal(
            action=action,
            strength=self._determine_signal_strength(confidence),
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss if action != "hold" else None,
            take_profit=take_profit if action != "hold" else None,
            reasoning=reasoning,
            risk_reward_ratio=risk_reward_ratio
        )
    
    def _momentum_signal(self, data: pd.DataFrame, regime: MarketRegime) -> TradingSignal:
        """Generate momentum signal."""
        current_price = data['close'].iloc[-1]
        
        # Calculate MACD
        macd_line, signal_line, histogram = self.indicators.calculate_macd(data['close'])
        
        if len(macd_line) == 0:
            return self._create_hold_signal(current_price, "Insufficient MACD data")
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        prev_histogram = histogram.iloc[-2] if len(histogram) > 1 else current_histogram
        
        # Calculate RSI for confirmation
        rsi = self.indicators.calculate_rsi(data['close'], self.config.rsi_period)
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # Generate signal
        action = "hold"
        confidence = 0.5
        reasoning = "No momentum signal"
        
        # Bullish momentum
        if (current_macd > current_signal and current_histogram > prev_histogram and 
            current_rsi > 50 and regime in [MarketRegime.TRENDING_UP, MarketRegime.LOW_VOLATILITY]):
            action = "buy"
            confidence = 0.8
            reasoning = "Strong bullish momentum"
        
        # Bearish momentum
        elif (current_macd < current_signal and current_histogram < prev_histogram and 
              current_rsi < 50 and regime in [MarketRegime.TRENDING_DOWN, MarketRegime.HIGH_VOLATILITY]):
            action = "sell"
            confidence = 0.8
            reasoning = "Strong bearish momentum"
        
        # Calculate stop loss and take profit
        volatility = data['close'].pct_change().tail(self.config.volatility_lookback).std()
        atr = self.indicators.calculate_atr(data['high'], data['low'], data['close'], self.config.atr_period)
        current_atr = atr.iloc[-1] if len(atr) > 0 else current_price * 0.02
        
        stop_loss, take_profit = self.risk_manager.calculate_stop_loss_take_profit(
            current_price, action, current_atr, volatility
        )
        
        # Calculate risk-reward ratio
        if action == "buy":
            risk = current_price - stop_loss
            reward = take_profit - current_price
        elif action == "sell":
            risk = stop_loss - current_price
            reward = current_price - take_profit
        else:
            risk = reward = 0
        
        risk_reward_ratio = safe_divide(reward, risk, 0.0) if risk > 0 else 0.0
        
        return TradingSignal(
            action=action,
            strength=self._determine_signal_strength(confidence),
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss if action != "hold" else None,
            take_profit=take_profit if action != "hold" else None,
            reasoning=reasoning,
            risk_reward_ratio=risk_reward_ratio
        )
    
    def _volatility_breakout_signal(self, data: pd.DataFrame, regime: MarketRegime) -> TradingSignal:
        """Generate volatility breakout signal."""
        current_price = data['close'].iloc[-1]
        
        # Calculate volatility
        returns = data['close'].pct_change().dropna()
        current_volatility = returns.tail(self.config.volatility_lookback).std()
        avg_volatility = returns.tail(50).std()  # Longer-term average
        
        # Calculate price momentum
        price_change_5 = (current_price / data['close'].iloc[-6] - 1) if len(data) > 5 else 0
        price_change_10 = (current_price / data['close'].iloc[-11] - 1) if len(data) > 10 else 0
        
        # Generate signal
        action = "hold"
        confidence = 0.5
        reasoning = "No volatility breakout"
        
        # Volatility expansion with upward momentum
        if (current_volatility > avg_volatility * 1.5 and price_change_5 > 0.02 and 
            price_change_10 > 0.01):
            action = "buy"
            confidence = 0.7
            reasoning = "Volatility breakout with upward momentum"
        
        # Volatility expansion with downward momentum
        elif (current_volatility > avg_volatility * 1.5 and price_change_5 < -0.02 and 
              price_change_10 < -0.01):
            action = "sell"
            confidence = 0.7
            reasoning = "Volatility breakout with downward momentum"
        
        # Calculate stop loss and take profit
        atr = self.indicators.calculate_atr(data['high'], data['low'], data['close'], self.config.atr_period)
        current_atr = atr.iloc[-1] if len(atr) > 0 else current_price * 0.02
        
        stop_loss, take_profit = self.risk_manager.calculate_stop_loss_take_profit(
            current_price, action, current_atr, current_volatility
        )
        
        # Calculate risk-reward ratio
        if action == "buy":
            risk = current_price - stop_loss
            reward = take_profit - current_price
        elif action == "sell":
            risk = stop_loss - current_price
            reward = current_price - take_profit
        else:
            risk = reward = 0
        
        risk_reward_ratio = safe_divide(reward, risk, 0.0) if risk > 0 else 0.0
        
        return TradingSignal(
            action=action,
            strength=self._determine_signal_strength(confidence),
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss if action != "hold" else None,
            take_profit=take_profit if action != "hold" else None,
            reasoning=reasoning,
            risk_reward_ratio=risk_reward_ratio
        )
    
    def _adaptive_signal(self, data: pd.DataFrame, regime: MarketRegime) -> TradingSignal:
        """Generate adaptive signal based on market regime."""
        # Choose strategy based on market regime
        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            return self._trend_following_signal(data, regime)
        elif regime == MarketRegime.SIDEWAYS:
            return self._mean_reversion_signal(data, regime)
        elif regime == MarketRegime.HIGH_VOLATILITY:
            return self._volatility_breakout_signal(data, regime)
        else:
            return self._momentum_signal(data, regime)
    
    def _validate_signal(self, signal: TradingSignal, data: pd.DataFrame) -> TradingSignal:
        """Validate and filter trading signal."""
        # Check minimum confidence
        if signal.confidence < self.config.min_signal_confidence:
            return self._create_hold_signal(signal.entry_price, "Confidence too low")
        
        # Check minimum risk-reward ratio
        if (signal.risk_reward_ratio is not None and 
            signal.risk_reward_ratio < self.config.min_risk_reward_ratio):
            return self._create_hold_signal(signal.entry_price, "Risk-reward ratio too low")
        
        # Validate prices
        if signal.stop_loss is not None and not validate_finite(signal.stop_loss):
            signal.stop_loss = None
        
        if signal.take_profit is not None and not validate_finite(signal.take_profit):
            signal.take_profit = None
        
        return signal
    
    def _create_hold_signal(self, price: float, reason: str) -> TradingSignal:
        """Create a hold signal."""
        return TradingSignal(
            action="hold",
            strength=SignalStrength.WEAK,
            confidence=0.5,
            entry_price=price,
            reasoning=reason
        )
    
    def _determine_signal_strength(self, confidence: float) -> SignalStrength:
        """Determine signal strength based on confidence."""
        if confidence >= 0.9:
            return SignalStrength.VERY_STRONG
        elif confidence >= 0.75:
            return SignalStrength.STRONG
        elif confidence >= 0.6:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK


class StrategyFactory:
    """Factory for creating trading strategies."""
    
    @staticmethod
    def create_strategy(strategy_type: StrategyType, **kwargs) -> ImprovedTradingStrategy:
        """Create a trading strategy of the specified type."""
        config = StrategyConfig(strategy_type=strategy_type, **kwargs)
        return ImprovedTradingStrategy(config)
    
    @staticmethod
    def create_trend_following_strategy(**kwargs) -> ImprovedTradingStrategy:
        """Create a trend following strategy."""
        return StrategyFactory.create_strategy(StrategyType.TREND_FOLLOWING, **kwargs)
    
    @staticmethod
    def create_mean_reversion_strategy(**kwargs) -> ImprovedTradingStrategy:
        """Create a mean reversion strategy."""
        return StrategyFactory.create_strategy(StrategyType.MEAN_REVERSION, **kwargs)
    
    @staticmethod
    def create_momentum_strategy(**kwargs) -> ImprovedTradingStrategy:
        """Create a momentum strategy."""
        return StrategyFactory.create_strategy(StrategyType.MOMENTUM, **kwargs)
    
    @staticmethod
    def create_adaptive_strategy(**kwargs) -> ImprovedTradingStrategy:
        """Create an adaptive strategy."""
        return StrategyFactory.create_strategy(StrategyType.ADAPTIVE, **kwargs)


# Convenience functions for backward compatibility
def create_baseline_strategy() -> ImprovedTradingStrategy:
    """Create a baseline strategy for comparison."""
    config = StrategyConfig(
        strategy_type=StrategyType.TREND_FOLLOWING,
        max_position_size=0.1,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        min_signal_confidence=0.6
    )
    return ImprovedTradingStrategy(config)


def create_optimized_strategy() -> ImprovedTradingStrategy:
    """Create an optimized strategy."""
    config = StrategyConfig(
        strategy_type=StrategyType.ADAPTIVE,
        max_position_size=0.15,
        stop_loss_pct=0.015,
        take_profit_pct=0.045,
        min_signal_confidence=0.7,
        enable_dynamic_sizing=True,
        volatility_scaling=True
    )
    return ImprovedTradingStrategy(config)


async def generate_strategy_signals(
    strategy: ImprovedTradingStrategy,
    data: pd.DataFrame,
    start_idx: int = 0,
    end_idx: Optional[int] = None
) -> List[TradingSignal]:
    """Generate signals for a strategy over a data range."""
    signals = []
    end_idx = end_idx or len(data)
    
    for i in range(start_idx, min(end_idx, len(data))):
        timestamp = data.index[i]
        signal = strategy.generate_signal(data, timestamp)
        signals.append(signal)
    
    return signals