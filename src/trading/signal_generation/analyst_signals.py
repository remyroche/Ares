"""
Analyst Signal Generation

This module integrates the Analyst component to generate trading signals
based on market analysis, feature engineering, and ML predictions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.printing import tprint

logger = system_logger.getChild('AnalystSignals')

class SignalType(Enum):
    """Types of trading signals."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"

class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class AnalystSignal:
    """Analyst-generated trading signal."""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    signal_strength: SignalStrength
    confidence_score: float
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    market_health_score: float = 0.0
    volatility_score: float = 0.0
    liquidation_risk_score: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    ml_predictions: Dict[str, Any] = field(default_factory=dict)
    regime_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class AnalystSignalGenerator:
    """
    Analyst Signal Generator that integrates with the Analyst component
    for trading signal generation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the analyst signal generator.

        Args:
            config: Configuration dictionary
        """
        tprint(f"[ANALYST_SIG_GEN] __init__: Initializing with config")
        self.config = config
        self.logger = logger.getChild('AnalystSignalGenerator')

        # Analyst component (will be injected)
        self.analyst: Optional[Any] = None

        # Signal generation parameters
        self.confidence_threshold: float = config.get('confidence_threshold', 0.6)
        self.signal_strength_thresholds = {
            SignalStrength.WEAK: 0.5,
            SignalStrength.MODERATE: 0.65,
            SignalStrength.STRONG: 0.8,
            SignalStrength.VERY_STRONG: 0.9
        }

        self.regime_timeframe: str = config.get('regime_timeframe', '15m')

        # Signal history (using deque for efficient O(1) operations)
        self.max_history: int = config.get('max_history', 1000)
        self.signal_history: deque[AnalystSignal] = deque(maxlen=self.max_history)

        # Performance tracking
        self.signal_count: int = 0
        self.successful_signals: int = 0
        self.failed_signals: int = 0
        tprint(f"[ANALYST_SIG_GEN] __init__ -> initialized (confidence_threshold={self.confidence_threshold}, max_history={self.max_history})")

    async def initialize(self, analyst_component: Any) -> bool:
        """
        Initialize the signal generator with analyst component.

        Args:
            analyst_component: Initialized Analyst component

        Returns:
            bool: True if initialization successful
        """
        tprint(f"[ANALYST_SIG_GEN] initialize: Initializing with analyst component")
        try:
            tprint_info("🔄 Initializing Analyst Signal Generator...")
            self.analyst = analyst_component

            tprint_success("✅ Analyst Signal Generator initialized")
            self.logger.info("✅ Analyst Signal Generator initialized")
            tprint(f"[ANALYST_SIG_GEN] initialize -> True")
            return True
        except Exception as e:
            error_msg = f"❌ Failed to initialize Analyst Signal Generator: {e}"
            tprint_error(error_msg)
            self.logger.error(error_msg)
            tprint(f"[ANALYST_SIG_GEN] initialize -> False (error: {e})", color="red")
            return False

    async def generate_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        regime_data: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Optional[AnalystSignal]:
        """
        Generate trading signal using Analyst component.

        Args:
            symbol: Trading symbol
            market_data: Market data DataFrame
            regime_data: Current regime information
            additional_context: Additional context for signal generation

        Returns:
            AnalystSignal or None if no signal generated
        """
        tprint(f"[ANALYST_SIG_GEN] generate_signal: symbol={symbol}, data_rows={len(market_data)}, has_regime={regime_data is not None}")
        try:
            if not self.analyst:
                self.logger.error("❌ Analyst component not initialized")
                tprint(f"[ANALYST_SIG_GEN] generate_signal -> None (analyst not initialized)", color="red")
                return None

            tprint_info(f"🔄 Generating analyst signal for {symbol}")

            # Perform market analysis using Analyst
            analysis_result = await self._perform_market_analysis(
                symbol, market_data, regime_data, additional_context
            )

            if not analysis_result:
                tprint_warning(f"⚠️ No analysis result for {symbol}")
                tprint(f"[ANALYST_SIG_GEN] generate_signal -> None (no analysis result)")
                return None

            # Generate signal based on analysis
            signal = await self._generate_signal_from_analysis(
                symbol, analysis_result, market_data
            )

            if signal:
                # Store signal in history
                self._store_signal(signal)
                self.signal_count += 1

                tprint_success(f"✅ Generated {signal.signal_type.value} signal for {symbol} "
                             f"(confidence: {signal.confidence_score:.3f})")
                tprint(f"[ANALYST_SIG_GEN] generate_signal -> {signal.signal_type.value} (confidence={signal.confidence_score:.3f}, strength={signal.signal_strength.value})")

            return signal

        except Exception as e:
            self.logger.error(f"❌ Signal generation failed for {symbol}: {e}")
            tprint(f"[ANALYST_SIG_GEN] generate_signal -> ERROR: {e}", color="red")
            return None

    async def _perform_market_analysis(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        regime_data: Optional[Dict[str, Any]],
        additional_context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Perform market analysis using Analyst component."""
        tprint(f"[ANALYST_SIG_GEN] _perform_market_analysis: symbol={symbol}")
        try:
            # Prepare analysis context
            analysis_context = {
                'symbol': symbol,
                'market_data': market_data,
                'regime_data': regime_data,
                'additional_context': additional_context or {}
            }

            # Call Analyst's analyze method
            if hasattr(self.analyst, 'analyze'):
                tprint(f"[ANALYST_SIG_GEN] _perform_market_analysis: calling analyst.analyze()")
                analysis_result = await self.analyst.analyze(analysis_context)
            elif hasattr(self.analyst, 'run_analysis'):
                tprint(f"[ANALYST_SIG_GEN] _perform_market_analysis: calling analyst.run_analysis()")
                analysis_result = await self.analyst.run_analysis(analysis_context)
            else:
                # Fallback to basic analysis
                tprint(f"[ANALYST_SIG_GEN] _perform_market_analysis: using fallback analysis")
                analysis_result = await self._fallback_analysis(analysis_context)

            tprint(f"[ANALYST_SIG_GEN] _perform_market_analysis -> result={analysis_result is not None}")
            return analysis_result

        except Exception as e:
            self.logger.error(f"❌ Market analysis failed: {e}")
            tprint(f"[ANALYST_SIG_GEN] _perform_market_analysis -> ERROR: {e}", color="red")
            return None

    async def _fallback_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when Analyst methods are not available."""
        tprint(f"[ANALYST_SIG_GEN] _fallback_analysis: using fallback analysis")
        try:
            market_data = context['market_data']

            # Basic technical analysis
            if len(market_data) < 20:
                tprint(f"[ANALYST_SIG_GEN] _fallback_analysis -> None (insufficient data: {len(market_data)} < 20)")
                return None

            # Calculate basic indicators
            close_prices = market_data['close'].values
            returns = np.diff(close_prices) / close_prices[:-1]

            # Simple signal based on price momentum
            recent_returns = returns[-5:].mean()
            volatility = np.std(returns[-20:])

            # Generate basic analysis result
            signal_direction = 'buy' if recent_returns > 0.001 else 'sell' if recent_returns < -0.001 else 'hold'
            confidence_score = min(abs(recent_returns) / volatility, 1.0) if volatility > 0 else 0.5

            analysis_result = {
                'signal_direction': signal_direction,
                'confidence_score': confidence_score,
                'market_health_score': 0.7,  # Default value
                'volatility_score': volatility,
                'liquidation_risk_score': 0.3,  # Default value
                'feature_importance': {},
                'ml_predictions': {},
                'analysis_metadata': {
                    'method': 'fallback',
                    'timestamp': datetime.now().isoformat()
                }
            }

            tprint(f"[ANALYST_SIG_GEN] _fallback_analysis -> {signal_direction} (confidence={confidence_score:.3f}, volatility={volatility:.4f})")
            return analysis_result

        except Exception as e:
            self.logger.error(f"❌ Fallback analysis failed: {e}")
            tprint(f"[ANALYST_SIG_GEN] _fallback_analysis -> ERROR: {e}", color="red")
            return None


    async def _generate_signal_from_analysis(
        self,
        symbol: str,
        analysis_result: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> Optional[AnalystSignal]:
        """Generate signal from analysis result."""
        tprint(f"[ANALYST_SIG_GEN] _generate_signal_from_analysis: symbol={symbol}")
        try:
            # Extract signal information
            signal_direction = analysis_result.get('signal_direction', 'hold')
            confidence_score = analysis_result.get('confidence_score', 0.0)

            tprint(f"[ANALYST_SIG_GEN] _generate_signal_from_analysis: direction={signal_direction}, confidence={confidence_score:.3f}, threshold={self.confidence_threshold:.3f}")

            # Check confidence threshold
            if confidence_score < self.confidence_threshold:
                tprint_warning(f"⚠️ Signal confidence {confidence_score:.3f} below threshold {self.confidence_threshold:.3f}")
                tprint(f"[ANALYST_SIG_GEN] _generate_signal_from_analysis -> None (below confidence threshold)")
                return None

            # Determine signal type
            signal_type = self._map_signal_direction(signal_direction)

            # Determine signal strength
            signal_strength = self._determine_signal_strength(confidence_score)

            # Calculate price targets
            current_price = market_data['close'].iloc[-1]
            price_target, stop_loss = self._calculate_price_targets(
                signal_type, current_price, analysis_result
            )

            # Create signal
            signal = AnalystSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                signal_strength=signal_strength,
                confidence_score=confidence_score,
                price_target=price_target,
                stop_loss=stop_loss,
                market_health_score=analysis_result.get('market_health_score', 0.0),
                volatility_score=analysis_result.get('volatility_score', 0.0),
                liquidation_risk_score=analysis_result.get('liquidation_risk_score', 0.0),
                feature_importance=analysis_result.get('feature_importance', {}),
                ml_predictions=analysis_result.get('ml_predictions', {}),
                regime_id=analysis_result.get('regime_id'),
                metadata=analysis_result.get('analysis_metadata', {})
            )
            tprint_success(f"✅ Signal generated from analysis: {signal_type.value} (confidence: {confidence_score:.3f})")
            tprint(f"[ANALYST_SIG_GEN] _generate_signal_from_analysis -> {signal_type.value} (strength={signal_strength.value}, price_target={price_target}, stop_loss={stop_loss})")
            return signal

        except Exception as e:
            self.logger.error(f"❌ Signal generation from analysis failed: {e}")
            tprint(f"[ANALYST_SIG_GEN] _generate_signal_from_analysis -> ERROR: {e}", color="red")
            return None

    def _map_signal_direction(self, direction: str) -> SignalType:
        """Map analysis direction to signal type."""
        tprint(f"[ANALYST_SIG_GEN] _map_signal_direction: direction={direction}")
        direction_map = {
            'buy': SignalType.BUY,
            'sell': SignalType.SELL,
            'hold': SignalType.HOLD,
            'close': SignalType.CLOSE
        }
        result = direction_map.get(direction.lower(), SignalType.HOLD)
        tprint(f"[ANALYST_SIG_GEN] _map_signal_direction -> {result.value}")
        return result

    def _determine_signal_strength(self, confidence_score: float) -> SignalStrength:
        """Determine signal strength based on confidence score."""
        tprint(f"[ANALYST_SIG_GEN] _determine_signal_strength: confidence={confidence_score:.3f}")
        if confidence_score >= self.signal_strength_thresholds[SignalStrength.VERY_STRONG]:
            result = SignalStrength.VERY_STRONG
        elif confidence_score >= self.signal_strength_thresholds[SignalStrength.STRONG]:
            result = SignalStrength.STRONG
        elif confidence_score >= self.signal_strength_thresholds[SignalStrength.MODERATE]:
            result = SignalStrength.MODERATE
        else:
            result = SignalStrength.WEAK
        tprint(f"[ANALYST_SIG_GEN] _determine_signal_strength -> {result.value}")
        return result

    def _calculate_price_targets(
        self,
        signal_type: SignalType,
        current_price: float,
        analysis_result: Dict[str, Any]
    ) -> tuple[Optional[float], Optional[float]]:
        """Calculate price targets and stop loss."""
        tprint(f"[ANALYST_SIG_GEN] _calculate_price_targets: signal_type={signal_type.value}, current_price={current_price}")
        try:
            volatility = analysis_result.get('volatility_score', 0.02)

            if signal_type == SignalType.BUY:
                # For buy signals, target 2x volatility, stop loss at 1x volatility
                price_target = current_price * (1 + 2 * volatility)
                stop_loss = current_price * (1 - volatility)
            elif signal_type == SignalType.SELL:
                # For sell signals, target 2x volatility, stop loss at 1x volatility
                price_target = current_price * (1 - 2 * volatility)
                stop_loss = current_price * (1 + volatility)
            else:
                # For hold/close signals, no price targets
                price_target = None
                stop_loss = None

            tprint(f"[ANALYST_SIG_GEN] _calculate_price_targets -> target={price_target}, stop={stop_loss} (volatility={volatility:.4f})")
            return price_target, stop_loss

        except Exception as e:
            self.logger.error(f"❌ Price target calculation failed: {e}")
            tprint(f"[ANALYST_SIG_GEN] _calculate_price_targets -> ERROR: {e}", color="red")
            return None, None

    def _store_signal(self, signal: AnalystSignal):
        """Store signal in history (deque automatically handles maxlen)."""
        tprint(f"[ANALYST_SIG_GEN] _store_signal: storing signal (history_size={len(self.signal_history)})")
        self.signal_history.append(signal)
        tprint(f"[ANALYST_SIG_GEN] _store_signal -> stored (new_size={len(self.signal_history)})")

    def get_signal_history(self, n: int = 100) -> List[AnalystSignal]:
        """Get recent signal history."""
        tprint(f"[ANALYST_SIG_GEN] get_signal_history: n={n}, total_signals={len(self.signal_history)}")
        # Convert deque to list for return
        signal_list = list(self.signal_history)
        result = signal_list[-n:] if len(signal_list) >= n else signal_list
        tprint(f"[ANALYST_SIG_GEN] get_signal_history -> {len(result)} signals")
        return result

    def get_signal_stats(self) -> Dict[str, Any]:
        """Get signal generation statistics."""
        tprint(f"[ANALYST_SIG_GEN] get_signal_stats: total_signals={self.signal_count}")
        if self.signal_count == 0:
            tprint(f"[ANALYST_SIG_GEN] get_signal_stats -> empty stats")
            return {
                'total_signals': 0,
                'success_rate': 0.0,
                'signal_distribution': {},
                'avg_confidence': 0.0
            }

        # Calculate signal distribution
        signal_distribution = {}
        for signal in self.signal_history:
            signal_type = signal.signal_type.value
            signal_distribution[signal_type] = signal_distribution.get(signal_type, 0) + 1

        # Calculate average confidence
        avg_confidence = np.mean([s.confidence_score for s in self.signal_history])

        stats = {
            'total_signals': self.signal_count,
            'success_rate': self.successful_signals / self.signal_count if self.signal_count > 0 else 0.0,
            'signal_distribution': signal_distribution,
            'avg_confidence': avg_confidence,
            'recent_signals': len(self.signal_history)
        }
        tprint(f"[ANALYST_SIG_GEN] get_signal_stats -> {stats}")
        return stats

    def update_signal_performance(self, signal: AnalystSignal, was_successful: bool):
        """Update signal performance tracking."""
        tprint(f"[ANALYST_SIG_GEN] update_signal_performance: was_successful={was_successful}")
        if was_successful:
            self.successful_signals += 1
        else:
            self.failed_signals += 1
        tprint(f"[ANALYST_SIG_GEN] update_signal_performance -> success={self.successful_signals}, failed={self.failed_signals}")

# Convenience functions

def create_analyst_signal_generator(config: Dict[str, Any]) -> AnalystSignalGenerator:
    """Create a configured analyst signal generator."""
    tprint(f"[ANALYST_SIG_GEN] create_analyst_signal_generator: Creating generator")
    generator = AnalystSignalGenerator(config)
    tprint(f"[ANALYST_SIG_GEN] create_analyst_signal_generator -> created")
    return generator

async def generate_analyst_signal(
    signal_generator: AnalystSignalGenerator,
    symbol: str,
    market_data: pd.DataFrame,
    analyst_component,
    regime_data: Optional[Dict[str, Any]] = None
) -> Optional[AnalystSignal]:
    """Generate analyst signal with convenience function."""
    tprint(f"[ANALYST_SIG_GEN] generate_analyst_signal: symbol={symbol}")
    if not signal_generator.analyst:
        tprint(f"[ANALYST_SIG_GEN] generate_analyst_signal: initializing analyst component")
        await signal_generator.initialize(analyst_component)

    signal = await signal_generator.generate_signal(
        symbol=symbol,
        market_data=market_data,
        regime_data=regime_data
    )
    tprint(f"[ANALYST_SIG_GEN] generate_analyst_signal -> {signal.signal_type.value if signal else None}")
    return signal
