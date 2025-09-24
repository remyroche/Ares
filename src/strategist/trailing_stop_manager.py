"""
Advanced Trailing Stop Management System

Intelligent trailing stop with adaptive distance calculation,
regime awareness, and dynamic adjustment capabilities.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging
import math

from ..interfaces.base_interfaces import MarketData, AnalysisResult
from .exit_strategy_manager import ExitType, ExitSignal, ExitLevel


class TrailingStopType(Enum):
    """Types of trailing stop strategies"""
    FIXED_DISTANCE = "fixed_distance"
    ATR_BASED = "atr_based"
    VOLATILITY_BASED = "volatility_based"
    REGIME_ADAPTIVE = "regime_adaptive"
    PRICE_ACTION_BASED = "price_action_based"
    TIME_BASED = "time_based"


class TrailingStopMode(Enum):
    """Trailing stop modes"""
    STATIC = "static"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"


@dataclass
class TrailingStopConfig:
    """Configuration for trailing stop strategies"""
    # Basic Settings
    enabled: bool = True
    activation_profit: float = 0.01  # Activate after 1% profit
    base_distance_pct: float = 0.015  # 1.5% base distance

    # Distance Calculation Settings
    distance_type: TrailingStopType = TrailingStopType.ATR_BASED
    atr_multiplier: float = 2.0  # ATR multiplier
    volatility_multiplier: float = 1.5  # Volatility multiplier
    min_distance_pct: float = 0.005  # 0.5% minimum distance
    max_distance_pct: float = 0.05   # 5% maximum distance

    # Adaptation Settings
    adaptive_mode: bool = True
    regime_awareness: bool = True
    price_action_awareness: bool = True
    time_based_adjustment: bool = True

    # Step Settings
    step_size_pct: float = 0.002  # 0.2% step size
    step_frequency: int = 60  # Step every 60 seconds
    minimum_step_pct: float = 0.001  # 0.1% minimum step

    # Regime-specific Settings
    regime_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "bull_trend": 0.8,      # Tighter stops in bull trends
        "bear_trend": 1.2,      # Wider stops in bear trends
        "sideways": 1.0,        # Normal stops in sideways
        "volatile": 1.5,        # Much wider stops in volatile markets
        "ranging": 0.9          # Slightly tighter in ranging markets
    })

    # Time-based Settings
    time_progression_enabled: bool = True
    initial_tightening_time: int = 300  # 5 minutes
    gradual_tightening_rate: float = 0.1  # 10% tightening per hour

    # Emergency Settings
    emergency_tightening: bool = True
    emergency_tightening_threshold: float = 0.05  # 5% profit threshold
    emergency_tightening_factor: float = 0.5  # Tighten to 50% of normal distance


@dataclass
class TrailingStopState:
    """Current state of trailing stop"""
    symbol: str
    entry_price: float
    current_price: float
    stop_price: float
    activation_price: float
    distance_pct: float
    last_update: datetime
    steps_taken: int = 0
    regime_history: List[str] = field(default_factory=list)
    price_history: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrailingStopResult:
    """Result of trailing stop evaluation"""
    should_exit: bool = False
    exit_level: Optional[ExitLevel] = None
    current_stop_price: float = 0.0
    current_distance_pct: float = 0.0
    confidence: float = 0.0
    reasoning: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrailingStopManager:
    """
    Advanced trailing stop management system that provides intelligent
    trailing stops with adaptive distance calculation and regime awareness.
    """

    def __init__(self, config: Optional[TrailingStopConfig] = None):
        self.config = config or TrailingStopConfig()
        self.logger = logging.getLogger(__name__)

        # State tracking
        self.active_trailing_stops: Dict[str, TrailingStopState] = {}
        self.stop_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}

    async def evaluate_trailing_stop(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        position_data: Optional[Dict[str, Any]] = None,
        market_data: Optional[MarketData] = None,
        analysis_result: Optional[AnalysisResult] = None
    ) -> TrailingStopResult:
        """
        Evaluate trailing stop condition

        Args:
            symbol: Trading symbol
            entry_price: Entry price of position
            current_price: Current market price
            position_data: Position information
            market_data: Current market data
            analysis_result: Latest analysis

        Returns:
            TrailingStopResult: Trailing stop evaluation
        """
        try:
            result = TrailingStopResult()

            # Calculate profit percentage
            profit_pct = (current_price - entry_price) / entry_price

            # Check if trailing stop should be activated
            if profit_pct < self.config.activation_profit:
                result.reasoning.append(f"Profit {profit_pct:.".2%"} below activation threshold {self.config.activation_profit:.".2%"}")
                return result

            # Get or create trailing stop state
            if symbol not in self.active_trailing_stops:
                await self._initialize_trailing_stop(symbol, entry_price, current_price, analysis_result)

            trailing_state = self.active_trailing_stops[symbol]

            # Update trailing stop
            updated_state = await self._update_trailing_stop(
                trailing_state, current_price, profit_pct, market_data, analysis_result
            )

            # Check if stop is triggered
            if current_price <= updated_state.stop_price:
                result.should_exit = True
                result.exit_level = ExitLevel(
                    exit_type=ExitType.TRAILING_STOP,
                    level=updated_state.stop_price,
                    quantity=1.0,
                    signal_strength=ExitSignal.STRONG,
                    triggered=True,
                    triggered_at=datetime.now(),
                    metadata={
                        "reason": "Trailing stop triggered",
                        "entry_price": entry_price,
                        "stop_price": updated_state.stop_price,
                        "current_price": current_price,
                        "profit_pct": profit_pct,
                        "distance_pct": updated_state.distance_pct
                    }
                )
                result.reasoning.append(f"Trailing stop triggered at {updated_state.stop_price".4f"}")
                result.confidence = 0.9
            else:
                result.current_stop_price = updated_state.stop_price
                result.current_distance_pct = updated_state.distance_pct
                result.reasoning.append(f"Trailing stop updated to {updated_state.stop_price".4f"}")

            # Store updated state
            self.active_trailing_stops[symbol] = updated_state

            return result

        except Exception as e:
            self.logger.error(f"Error evaluating trailing stop for {symbol}: {e}")
            return TrailingStopResult()

    async def _initialize_trailing_stop(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        analysis_result: Optional[AnalysisResult] = None
    ) -> None:
        """Initialize trailing stop state"""
        try:
            # Calculate initial distance
            distance_pct = await self._calculate_initial_distance(entry_price, current_price, analysis_result)

            # Set activation price (where trailing stop becomes active)
            activation_price = entry_price * (1 + self.config.activation_profit)

            # Set initial stop price (below activation price)
            stop_price = activation_price * (1 - distance_pct)

            state = TrailingStopState(
                symbol=symbol,
                entry_price=entry_price,
                current_price=current_price,
                stop_price=stop_price,
                activation_price=activation_price,
                distance_pct=distance_pct,
                last_update=datetime.now(),
                regime_history=[analysis_result.market_regime] if analysis_result else [],
                metadata={
                    "initial_distance": distance_pct,
                    "initialization_time": datetime.now()
                }
            )

            self.active_trailing_stops[symbol] = state
            self.logger.info(f"Initialized trailing stop for {symbol}: {stop_price".4f"} (distance: {distance_pct".3%"})")

        except Exception as e:
            self.logger.error(f"Error initializing trailing stop for {symbol}: {e}")

    async def _update_trailing_stop(
        self,
        state: TrailingStopState,
        current_price: float,
        profit_pct: float,
        market_data: Optional[MarketData] = None,
        analysis_result: Optional[AnalysisResult] = None
    ) -> TrailingStopState:
        """Update trailing stop based on current conditions"""
        try:
            # Calculate new distance
            new_distance = await self._calculate_adaptive_distance(state, current_price, profit_pct, analysis_result)

            # Update state
            updated_state = state.__class__(
                symbol=state.symbol,
                entry_price=state.entry_price,
                current_price=current_price,
                stop_price=state.stop_price,
                activation_price=state.activation_price,
                distance_pct=new_distance,
                last_update=datetime.now(),
                steps_taken=state.steps_taken,
                regime_history=state.regime_history.copy(),
                metadata=state.metadata.copy()
            )

            # Update regime history
            if analysis_result:
                updated_state.regime_history.append(analysis_result.market_regime)

            # Calculate new stop price based on distance type
            new_stop_price = await self._calculate_stop_price(
                updated_state, current_price, new_distance, profit_pct, analysis_result
            )

            # Apply step logic (only move stop up, never down)
            if new_stop_price > updated_state.stop_price:
                updated_state.stop_price = new_stop_price
                updated_state.steps_taken += 1
                updated_state.metadata["last_step"] = datetime.now()

            return updated_state

        except Exception as e:
            self.logger.error(f"Error updating trailing stop: {e}")
            return state

    async def _calculate_initial_distance(
        self,
        entry_price: float,
        current_price: float,
        analysis_result: Optional[AnalysisResult] = None
    ) -> float:
        """Calculate initial trailing stop distance"""
        try:
            base_distance = self.config.base_distance_pct

            # Adjust based on distance type
            if self.config.distance_type == TrailingStopType.ATR_BASED:
                if analysis_result:
                    atr = analysis_result.technical_indicators.get('ATR', 0)
                    if atr > 0:
                        base_distance = (atr / current_price) * self.config.atr_multiplier

            elif self.config.distance_type == TrailingStopType.VOLATILITY_BASED:
                if analysis_result:
                    volatility = analysis_result.technical_indicators.get('volatility', 0)
                    if volatility > 0:
                        base_distance = volatility * self.config.volatility_multiplier

            elif self.config.distance_type == TrailingStopType.REGIME_ADAPTIVE:
                if analysis_result:
                    regime = analysis_result.market_regime
                    multiplier = self.config.regime_multipliers.get(regime, 1.0)
                    base_distance *= multiplier

            # Apply bounds
            base_distance = max(self.config.min_distance_pct, min(base_distance, self.config.max_distance_pct))

            return base_distance

        except Exception as e:
            self.logger.error(f"Error calculating initial distance: {e}")
            return self.config.base_distance_pct

    async def _calculate_adaptive_distance(
        self,
        state: TrailingStopState,
        current_price: float,
        profit_pct: float,
        analysis_result: Optional[AnalysisResult] = None
    ) -> float:
        """Calculate adaptive trailing stop distance"""
        try:
            base_distance = self.config.base_distance_pct

            # Start with distance type calculation
            if self.config.distance_type == TrailingStopType.ATR_BASED:
                if analysis_result:
                    atr = analysis_result.technical_indicators.get('ATR', 0)
                    if atr > 0:
                        base_distance = (atr / current_price) * self.config.atr_multiplier

            elif self.config.distance_type == TrailingStopType.VOLATILITY_BASED:
                if analysis_result:
                    volatility = analysis_result.technical_indicators.get('volatility', 0)
                    if volatility > 0:
                        base_distance = volatility * self.config.volatility_multiplier

            # Apply regime awareness
            if self.config.regime_awareness and analysis_result:
                regime = analysis_result.market_regime
                regime_multiplier = self.config.regime_multipliers.get(regime, 1.0)
                base_distance *= regime_multiplier

            # Apply time-based adjustment
            if self.config.time_based_adjustment:
                position_age = (datetime.now() - state.metadata.get('initialization_time', datetime.now())).seconds
                if position_age > self.config.initial_tightening_time:
                    hours_held = position_age / 3600
                    tightening_factor = 1 - (self.config.gradual_tightening_rate * hours_held)
                    base_distance *= max(tightening_factor, 0.5)  # Don't tighten below 50%

            # Apply profit-based adjustment
            if profit_pct > self.config.emergency_tightening_threshold:
                emergency_factor = self.config.emergency_tightening_factor
                base_distance *= emergency_factor
                state.metadata["emergency_tightening"] = True

            # Apply bounds
            base_distance = max(self.config.min_distance_pct, min(base_distance, self.config.max_distance_pct))

            return base_distance

        except Exception as e:
            self.logger.error(f"Error calculating adaptive distance: {e}")
            return state.distance_pct

    async def _calculate_stop_price(
        self,
        state: TrailingStopState,
        current_price: float,
        distance_pct: float,
        profit_pct: float,
        analysis_result: Optional[AnalysisResult] = None
    ) -> float:
        """Calculate new trailing stop price"""
        try:
            # Basic calculation: current price minus distance
            new_stop_price = current_price * (1 - distance_pct)

            # Apply price action awareness
            if self.config.price_action_awareness and analysis_result:
                # Look for support levels to place stop above
                support_level = analysis_result.support_resistance.get('support', 0)
                if support_level > 0 and support_level > new_stop_price:
                    # Place stop just above support level
                    new_stop_price = support_level * 1.002  # 0.2% above support

            # Apply minimum step logic
            min_step = current_price * self.config.minimum_step_pct
            current_stop = state.stop_price

            if new_stop_price - current_stop < min_step:
                # Don't move stop unless it's at least one minimum step
                return current_stop

            return new_stop_price

        except Exception as e:
            self.logger.error(f"Error calculating stop price: {e}")
            return state.stop_price

    def get_trailing_stop_status(self, symbol: str) -> Dict[str, Any]:
        """Get current trailing stop status"""
        if symbol not in self.active_trailing_stops:
            return {"active": False, "message": "No trailing stop active"}

        state = self.active_trailing_stops[symbol]
        return {
            "active": True,
            "symbol": state.symbol,
            "entry_price": state.entry_price,
            "current_stop_price": state.stop_price,
            "activation_price": state.activation_price,
            "distance_pct": state.distance_pct,
            "steps_taken": state.steps_taken,
            "last_update": state.last_update,
            "config": self.config.__dict__
        }

    def update_config(self, new_config: TrailingStopConfig) -> None:
        """Update trailing stop configuration"""
        self.config = new_config
        self.logger.info("Trailing stop configuration updated")

    def get_performance_metrics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get trailing stop performance metrics"""
        if symbol:
            return self.performance_metrics.get(symbol, {})

        return {
            "total_active_stops": len(self.active_trailing_stops),
            "total_steps_taken": sum(state.steps_taken for state in self.active_trailing_stops.values()),
            "average_distance": np.mean([state.distance_pct for state in self.active_trailing_stops.values()]) if self.active_trailing_stops else 0,
            "config": self.config.__dict__
        }