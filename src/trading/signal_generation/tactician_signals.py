"""
Tactician Signal Generation

This module integrates the Tactician component to generate timing signals
for trade execution based on position sizing, scenario predictions, and risk management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success


logger = system_logger.getChild('TacticianSignals')

class TimingSignal(Enum):
    """Types of timing signals."""
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"
    CLOSE_ALL = "close_all"

class TimingConfidence(Enum):
    """Timing confidence levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class PositionSizing:
    """Position sizing information."""
    recommended_size: float
    max_size: float
    leverage: float
    risk_per_trade: float
    kelly_fraction: float
    confidence_multiplier: float

@dataclass
class TacticianSignal:
    """Tactician-generated timing signal."""
    timestamp: datetime
    symbol: str
    timing_signal: TimingSignal
    confidence: TimingConfidence
    confidence_score: float
    position_sizing: PositionSizing
    scenario_predictions: Dict[str, Any] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    timing_indicators: Dict[str, float] = field(default_factory=dict)
    signal_type: Optional[int] = None  # Analyst signal type (long/short)
    metadata: Dict[str, Any] = field(default_factory=dict)

class TacticianSignalGenerator:
    """
    Tactician Signal Generator that integrates with the Tactician component
    for timing signal generation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tactician signal generator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger.getChild('TacticianSignalGenerator')

        # Tactician component (will be injected)
        self.tactician = None

        # Signal generation parameters
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% risk per trade
        self.max_leverage = config.get('max_leverage', 3.0)
        self.kelly_fraction = config.get('kelly_fraction', 0.25)  # Conservative Kelly

        # Confidence thresholds
        self.confidence_thresholds = {
            TimingConfidence.LOW: 0.5,
            TimingConfidence.MEDIUM: 0.65,
            TimingConfidence.HIGH: 0.8,
            TimingConfidence.VERY_HIGH: 0.9
        }

        # Signal history (using deque for efficient O(1) operations)
        self.max_history = config.get('max_history', 1000)
        self.signal_history: deque = deque(maxlen=self.max_history)

        # Performance tracking
        self.signal_count = 0
        self.successful_signals = 0
        self.failed_signals = 0

    async def initialize(self, tactician_component) -> bool:
        """
        Initialize the signal generator with tactician component.

        Args:
            tactician_component: Initialized Tactician component

        Returns:
            bool: True if initialization successful
        """
        try:
            self.tactician = tactician_component
            self.logger.info("✅ Tactician Signal Generator initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Tactician Signal Generator: {e}")
            return False

    @handles_errors
    @traced(span_name="tactician_signal_generation")
    @log_execution_time()
    async def generate_timing_signal(
        self,
        symbol: str,
        analyst_signal: Dict[str, Any],
        market_data: pd.DataFrame,
        current_position: Optional[Dict[str, Any]] = None,
        account_balance: float = 10000.0,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Optional[TacticianSignal]:
        """
        Generate timing signal using Tactician component.

        Args:
            symbol: Trading symbol
            analyst_signal: Signal from Analyst component
            market_data: Market data DataFrame
            current_position: Current position information
            account_balance: Account balance for position sizing
            additional_context: Additional context for signal generation

        Returns:
            TacticianSignal or None if no signal generated
        """
        try:
            if not self.tactician:
                self.logger.error("❌ Tactician component not initialized")
                return None

            tprint_info(f"🔄 Generating tactician timing signal for {symbol}")

            # Perform timing analysis using Tactician
            timing_analysis = await self._perform_timing_analysis(
                symbol, analyst_signal, market_data, current_position, additional_context
            )

            if not timing_analysis:
                tprint_warning(f"⚠️ No timing analysis result for {symbol}")
                return None

            # Calculate position sizing
            position_sizing = await self._calculate_position_sizing(
                symbol, analyst_signal, timing_analysis, account_balance
            )

            # Generate timing signal
            signal = await self._generate_timing_signal_from_analysis(
                symbol, timing_analysis, position_sizing, current_position
            )

            if signal:
                # Store signal in history
                self._store_signal(signal)
                self.signal_count += 1

                tprint_success(f"✅ Generated {signal.timing_signal.value} signal for {symbol} "
                             f"(confidence: {signal.confidence_score:.3f})")

            return signal

        except Exception as e:
            self.logger.error(f"❌ Timing signal generation failed for {symbol}: {e}")
            return None

    async def _perform_timing_analysis(
        self,
        symbol: str,
        analyst_signal: Dict[str, Any],
        market_data: pd.DataFrame,
        current_position: Optional[Dict[str, Any]],
        additional_context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Perform timing analysis using Tactician component."""
        try:
            # Prepare timing context
            timing_context = {
                'symbol': symbol,
                'analyst_signal': analyst_signal,
                'market_data': market_data,
                'current_position': current_position,
                'additional_context': additional_context or {}
            }

            # Call Tactician's timing analysis method
            if hasattr(self.tactician, 'analyze_timing'):
                timing_result = await self.tactician.analyze_timing(timing_context)
            elif hasattr(self.tactician, 'run_timing_analysis'):
                timing_result = await self.tactician.run_timing_analysis(timing_context)
            else:
                # Fallback to basic timing analysis
                timing_result = await self._fallback_timing_analysis(timing_context)

            return timing_result

        except Exception as e:
            self.logger.error(f"❌ Timing analysis failed: {e}")
            return None

    async def _fallback_timing_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback timing analysis when Tactician methods are not available."""
        try:
            market_data = context['market_data']
            analyst_signal = context['analyst_signal']
            current_position = context.get('current_position')

            if len(market_data) < 20:
                return None

            # Basic timing indicators
            close_prices = market_data['close'].values
            returns = np.diff(close_prices) / close_prices[:-1]

            # Calculate timing metrics
            recent_volatility = np.std(returns[-10:])
            price_momentum = returns[-5:].mean()
            # Avoid division by zero
            historical_volume_mean = market_data['volume'].iloc[-20:-5].mean()
            if historical_volume_mean > 0:
                volume_trend = market_data['volume'].iloc[-5:].mean() / historical_volume_mean
            else:
                volume_trend = 1.0  # Default to neutral if no historical volume

            # Determine timing signal based on analyst signal and market conditions
            analyst_direction = analyst_signal.get('signal_type', 'hold')
            confidence_score = analyst_signal.get('confidence_score', 0.5)

            # Timing logic
            if current_position:
                # We have a position, consider exit timing
                if confidence_score < 0.4 or recent_volatility > 0.05:
                    timing_signal = 'exit'
                    timing_confidence = min(confidence_score + 0.2, 1.0)
                else:
                    timing_signal = 'hold'
                    timing_confidence = confidence_score
            else:
                # No position, consider entry timing
                if analyst_direction in ['buy', 'sell'] and confidence_score > 0.6:
                    timing_signal = 'enter'
                    timing_confidence = confidence_score
                else:
                    timing_signal = 'hold'
                    timing_confidence = 0.5

            # Generate timing analysis result
            timing_result = {
                'timing_signal': timing_signal,
                'confidence_score': timing_confidence,
                'scenario_predictions': {
                    'bullish_probability': 0.6 if analyst_direction == 'buy' else 0.4,
                    'bearish_probability': 0.6 if analyst_direction == 'sell' else 0.4,
                    'sideways_probability': 0.3
                },
                'risk_metrics': {
                    'volatility': recent_volatility,
                    'momentum': price_momentum,
                    'volume_trend': volume_trend
                },
                'timing_indicators': {
                    'rsi': 50.0,  # Default value
                    'macd': 0.0,  # Default value
                    'bollinger_position': 0.5  # Default value
                },
                'analysis_metadata': {
                    'method': 'fallback',
                    'timestamp': datetime.now().isoformat()
                }
            }

            return timing_result

        except Exception as e:
            self.logger.error(f"❌ Fallback timing analysis failed: {e}")
            return None

    async def _calculate_position_sizing(
        self,
        symbol: str,
        analyst_signal: Dict[str, Any],
        timing_analysis: Dict[str, Any],
        account_balance: float
    ) -> PositionSizing:
        """Calculate position sizing based on confidence and risk parameters."""
        try:
            # Get confidence scores
            analyst_confidence = analyst_signal.get('confidence_score', 0.5)
            timing_confidence = timing_analysis.get('confidence_score', 0.5)

            # Combined confidence
            combined_confidence = (analyst_confidence + timing_confidence) / 2

            # Kelly criterion calculation for position sizing
            # Note: These values represent actual trading outcomes, not model training targets
            win_probability = combined_confidence
            avg_win = 0.005   # 0.5% average win (realistic trading outcome expectation)
            avg_loss = 0.003  # 0.3% average loss (realistic risk management)

            kelly_fraction = (win_probability * avg_win - (1 - win_probability) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, self.kelly_fraction))  # Cap at configured fraction

            # Position size calculation
            risk_amount = account_balance * self.risk_per_trade
            confidence_multiplier = combined_confidence

            # Base position size
            base_size = risk_amount * confidence_multiplier

            # Apply Kelly fraction
            recommended_size = base_size * kelly_fraction

            # Maximum position size (10% of account)
            max_size = account_balance * 0.1

            # Leverage calculation
            leverage = min(combined_confidence * self.max_leverage, self.max_leverage)

            return PositionSizing(
                recommended_size=min(recommended_size, max_size),
                max_size=max_size,
                leverage=leverage,
                risk_per_trade=self.risk_per_trade,
                kelly_fraction=kelly_fraction,
                confidence_multiplier=confidence_multiplier
            )

        except Exception as e:
            self.logger.error(f"❌ Position sizing calculation failed: {e}")
            # Return conservative defaults
            return PositionSizing(
                recommended_size=account_balance * 0.01,  # 1% of account
                max_size=account_balance * 0.05,  # 5% max
                leverage=1.0,
                risk_per_trade=self.risk_per_trade,
                kelly_fraction=0.1,
                confidence_multiplier=0.5
            )


    async def _generate_timing_signal_from_analysis(
        self,
        symbol: str,
        timing_analysis: Dict[str, Any],
        position_sizing: PositionSizing,
        current_position: Optional[Dict[str, Any]]
    ) -> Optional[TacticianSignal]:
        """Generate timing signal from analysis result."""
        try:
            # Extract timing information
            timing_signal_str = timing_analysis.get('timing_signal', 'hold')
            confidence_score = timing_analysis.get('confidence_score', 0.0)

            # Check confidence threshold
            if confidence_score < self.confidence_threshold:
                return None

            # Map timing signal
            timing_signal = self._map_timing_signal(timing_signal_str, current_position)

            # Determine confidence level
            confidence = self._determine_confidence_level(confidence_score)

            # Create signal
            signal = TacticianSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                timing_signal=timing_signal,
                confidence=confidence,
                confidence_score=confidence_score,
                position_sizing=position_sizing,
                scenario_predictions=timing_analysis.get('scenario_predictions', {}),
                risk_metrics=timing_analysis.get('risk_metrics', {}),
                timing_indicators=timing_analysis.get('timing_indicators', {}),
                metadata=timing_analysis.get('analysis_metadata', {})
            )

            return signal

        except Exception as e:
            self.logger.error(f"❌ Timing signal generation from analysis failed: {e}")
            return None

    def _map_timing_signal(self, signal_str: str, current_position: Optional[Dict[str, Any]]) -> TimingSignal:
        """Map timing signal string to TimingSignal enum."""
        if current_position:
            # We have a position
            if signal_str == 'exit':
                position_type = current_position.get('type', 'long')
                return TimingSignal.EXIT_LONG if position_type == 'long' else TimingSignal.EXIT_SHORT
            else:
                return TimingSignal.HOLD
        else:
            # No position
            if signal_str == 'enter':
                # This would need to be determined by analyst signal direction
                # For now, default to long entry
                return TimingSignal.ENTER_LONG
            else:
                return TimingSignal.HOLD

    def _determine_confidence_level(self, confidence_score: float) -> TimingConfidence:
        """Determine confidence level based on score."""
        if confidence_score >= self.confidence_thresholds[TimingConfidence.VERY_HIGH]:
            return TimingConfidence.VERY_HIGH
        elif confidence_score >= self.confidence_thresholds[TimingConfidence.HIGH]:
            return TimingConfidence.HIGH
        elif confidence_score >= self.confidence_thresholds[TimingConfidence.MEDIUM]:
            return TimingConfidence.MEDIUM
        else:
            return TimingConfidence.LOW

    def _store_signal(self, signal: TacticianSignal):
        """Store signal in history (deque automatically handles maxlen)."""
        self.signal_history.append(signal)

    def get_signal_history(self, n: int = 100) -> List[TacticianSignal]:
        """Get recent signal history."""
        # Convert deque to list for return
        signal_list = list(self.signal_history)
        return signal_list[-n:] if len(signal_list) >= n else signal_list

    def get_signal_stats(self) -> Dict[str, Any]:
        """Get signal generation statistics."""
        if self.signal_count == 0:
            return {
                'total_signals': 0,
                'success_rate': 0.0,
                'signal_distribution': {},
                'avg_confidence': 0.0,
                'avg_position_size': 0.0
            }

        # Calculate signal distribution
        signal_distribution = {}
        for signal in self.signal_history:
            signal_type = signal.timing_signal.value
            signal_distribution[signal_type] = signal_distribution.get(signal_type, 0) + 1

        # Calculate averages
        avg_confidence = np.mean([s.confidence_score for s in self.signal_history])
        avg_position_size = np.mean([s.position_sizing.recommended_size for s in self.signal_history])

        return {
            'total_signals': self.signal_count,
            'success_rate': self.successful_signals / self.signal_count if self.signal_count > 0 else 0.0,
            'signal_distribution': signal_distribution,
            'avg_confidence': avg_confidence,
            'avg_position_size': avg_position_size,
            'recent_signals': len(self.signal_history)
        }

    def update_signal_performance(self, signal: TacticianSignal, was_successful: bool):
        """Update signal performance tracking."""
        if was_successful:
            self.successful_signals += 1
        else:
            self.failed_signals += 1

# Convenience functions

def create_tactician_signal_generator(config: Dict[str, Any]) -> TacticianSignalGenerator:
    """Create a configured tactician signal generator."""
    return TacticianSignalGenerator(config)

async def generate_tactician_signal(
    signal_generator: TacticianSignalGenerator,
    symbol: str,
    analyst_signal: Dict[str, Any],
    market_data: pd.DataFrame,
    tactician_component,
    current_position: Optional[Dict[str, Any]] = None,
    account_balance: float = 10000.0
) -> Optional[TacticianSignal]:
    """Generate tactician signal with convenience function."""
    if not signal_generator.tactician:
        await signal_generator.initialize(tactician_component)

    return await signal_generator.generate_timing_signal(
        symbol=symbol,
        analyst_signal=analyst_signal,
        market_data=market_data,
        current_position=current_position,
        account_balance=account_balance
    )
