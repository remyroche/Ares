"""
Advanced Exit Strategy Management System

Comprehensive exit strategy implementation with multiple exit types,
dynamic calculations, and market-adaptive features.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging
import math

from ..interfaces.base_interfaces import MarketData, AnalysisResult, StrategyResult, TradeDecision


class ExitType(Enum):
    """Types of exit strategies"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TIME_BASED = "time_based"
    VOLATILITY_BASED = "volatility_based"
    REGIME_BASED = "regime_based"
    EMERGENCY = "emergency"
    BREAK_EVEN = "break_even"


class ExitSignal(Enum):
    """Exit signal types"""
    NONE = "none"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    EMERGENCY = "emergency"


@dataclass
class ExitLevel:
    """Individual exit level configuration"""
    exit_type: ExitType
    level: float
    quantity: float  # Percentage of position to exit (0-1)
    signal_strength: ExitSignal = ExitSignal.NONE
    triggered: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    triggered_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExitStrategyConfig:
    """Configuration for exit strategies"""
    # Stop Loss Settings
    stop_loss_enabled: bool = True
    stop_loss_pct: float = 0.02  # 2% default
    dynamic_stop_loss: bool = True
    atr_stop_loss_multiplier: float = 2.0
    vol_stop_loss_multiplier: float = 1.5

    # Take Profit Settings
    take_profit_enabled: bool = True
    take_profit_pct: float = 0.04  # 4% default
    multi_target_tp: bool = True
    tp_levels: List[float] = field(default_factory=lambda: [0.02, 0.04, 0.08])  # Multiple targets

    # Trailing Stop Settings
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float = 0.015  # 1.5% default
    trailing_stop_activation: float = 0.01  # Activate after 1% profit
    adaptive_trailing: bool = True
    trailing_step: float = 0.005  # Step size for trailing

    # Time-based Settings
    time_based_enabled: bool = True
    max_hold_time: int = 7200  # 2 hours in seconds
    min_hold_time: int = 300   # 5 minutes minimum
    regime_aware_timing: bool = True

    # Volatility-based Settings
    volatility_enabled: bool = True
    volatility_threshold: float = 0.05  # 5%
    volatility_exit_multiplier: float = 2.0
    use_atr_for_volatility: bool = True

    # Regime-based Settings
    regime_enabled: bool = True
    regime_exit_thresholds: Dict[str, float] = field(default_factory=dict)

    # Emergency Settings
    emergency_enabled: bool = True
    emergency_drawdown_threshold: float = 0.10  # 10%
    emergency_volatility_spike: float = 0.15    # 15%

    # Break-even Settings
    break_even_enabled: bool = True
    break_even_trigger: float = 0.01  # Trigger at 1% profit
    break_even_buffer: float = 0.005   # 0.5% buffer above entry


@dataclass
class ExitStrategyResult:
    """Result of exit strategy evaluation"""
    should_exit: bool = False
    exit_signal: ExitSignal = ExitSignal.NONE
    exit_levels: List[ExitLevel] = field(default_factory=list)
    total_exit_quantity: float = 0.0  # Percentage of position to exit
    primary_exit_type: Optional[ExitType] = None
    confidence: float = 0.0
    reasoning: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExitStrategyManager:
    """
    Advanced exit strategy management system that coordinates multiple exit types
    and provides intelligent exit signals based on market conditions.
    """

    def __init__(self, config: Optional[ExitStrategyConfig] = None):
        self.config = config or ExitStrategyConfig()
        self.logger = logging.getLogger(__name__)

        # State tracking
        self.active_exits: Dict[str, List[ExitLevel]] = {}
        self.exit_history: List[Dict[str, Any]] = []
        self.market_data_cache: Dict[str, List[MarketData]] = {}

        # Performance tracking
        self.exit_performance: Dict[str, Dict[str, Any]] = {}

    async def evaluate_exit_strategies(
        self,
        symbol: str,
        market_data: MarketData,
        analysis_result: Optional[AnalysisResult] = None,
        strategy_result: Optional[StrategyResult] = None,
        position_data: Optional[Dict[str, Any]] = None
    ) -> ExitStrategyResult:
        """
        Evaluate all exit strategies for a given position

        Args:
            symbol: Trading symbol
            market_data: Current market data
            analysis_result: Latest analysis (optional)
            strategy_result: Strategy context (optional)
            position_data: Current position information (optional)

        Returns:
            ExitStrategyResult: Comprehensive exit evaluation
        """
        try:
            result = ExitStrategyResult()

            # Get position context
            entry_price = position_data.get('entry_price', market_data.close) if position_data else market_data.close
            current_price = market_data.close
            position_size = position_data.get('quantity', 1.0) if position_data else 1.0
            position_age = position_data.get('age_seconds', 0) if position_data else 0

            # Calculate profit/loss percentage
            pnl_pct = (current_price - entry_price) / entry_price

            # Evaluate each exit type
            exit_evaluations = []

            # Stop Loss Evaluation
            if self.config.stop_loss_enabled:
                sl_result = await self._evaluate_stop_loss(
                    symbol, current_price, entry_price, pnl_pct, market_data, analysis_result
                )
                if sl_result:
                    exit_evaluations.append(sl_result)

            # Take Profit Evaluation
            if self.config.take_profit_enabled:
                tp_result = await self._evaluate_take_profit(
                    symbol, current_price, entry_price, pnl_pct, position_data
                )
                if tp_result:
                    exit_evaluations.append(tp_result)

            # Trailing Stop Evaluation
            if self.config.trailing_stop_enabled:
                ts_result = await self._evaluate_trailing_stop(
                    symbol, current_price, entry_price, pnl_pct, market_data
                )
                if ts_result:
                    exit_evaluations.append(ts_result)

            # Time-based Evaluation
            if self.config.time_based_enabled:
                tb_result = await self._evaluate_time_based(
                    symbol, position_age, pnl_pct, market_data, analysis_result
                )
                if tb_result:
                    exit_evaluations.append(tb_result)

            # Volatility-based Evaluation
            if self.config.volatility_enabled:
                vol_result = await self._evaluate_volatility_based(
                    symbol, current_price, market_data, analysis_result
                )
                if vol_result:
                    exit_evaluations.append(vol_result)

            # Regime-based Evaluation
            if self.config.regime_enabled and analysis_result:
                reg_result = await self._evaluate_regime_based(
                    symbol, analysis_result, pnl_pct, position_data
                )
                if reg_result:
                    exit_evaluations.append(reg_result)

            # Emergency Evaluation
            if self.config.emergency_enabled:
                emer_result = await self._evaluate_emergency(
                    symbol, current_price, entry_price, market_data, analysis_result
                )
                if emer_result:
                    exit_evaluations.append(emer_result)

            # Break-even Evaluation
            if self.config.break_even_enabled:
                be_result = await self._evaluate_break_even(
                    symbol, current_price, entry_price, pnl_pct
                )
                if be_result:
                    exit_evaluations.append(be_result)

            # Process all exit evaluations
            result = await self._process_exit_evaluations(exit_evaluations, symbol, position_data)

            # Log result
            self.logger.info(f"Exit evaluation for {symbol}: {result.should_exit} "
                           f"({result.total_exit_quantity:.2%}) - {result.primary_exit_type}")

            return result

        except Exception as e:
            self.logger.error(f"Error evaluating exit strategies for {symbol}: {e}")
            return ExitStrategyResult(should_exit=False)

    async def _evaluate_stop_loss(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        pnl_pct: float,
        market_data: MarketData,
        analysis_result: Optional[AnalysisResult] = None
    ) -> Optional[ExitLevel]:
        """Evaluate stop loss condition"""
        try:
            # Basic percentage-based stop loss
            if self.config.dynamic_stop_loss and analysis_result:
                # Use ATR for dynamic stop loss
                atr = analysis_result.technical_indicators.get('ATR', 0)
                if atr > 0:
                    stop_distance = atr * self.config.atr_stop_loss_multiplier
                    stop_price = entry_price * (1 - stop_distance)
                else:
                    stop_price = entry_price * (1 - self.config.stop_loss_pct)
            else:
                stop_price = entry_price * (1 - self.config.stop_loss_pct)

            # Check if stop loss triggered
            if current_price <= stop_price:
                return ExitLevel(
                    exit_type=ExitType.STOP_LOSS,
                    level=stop_price,
                    quantity=1.0,  # Full position exit
                    signal_strength=ExitSignal.STRONG,
                    triggered=True,
                    triggered_at=datetime.now(),
                    metadata={
                        "reason": "Stop loss triggered",
                        "entry_price": entry_price,
                        "stop_price": stop_price,
                        "current_price": current_price,
                        "loss_pct": abs(pnl_pct)
                    }
                )

            return None

        except Exception as e:
            self.logger.error(f"Error evaluating stop loss for {symbol}: {e}")
            return None

    async def _evaluate_take_profit(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        pnl_pct: float,
        position_data: Optional[Dict[str, Any]] = None
    ) -> Optional[ExitLevel]:
        """Evaluate take profit condition"""
        try:
            # Multi-target take profit
            if self.config.multi_target_tp:
                # Get current position size for partial exits
                remaining_quantity = position_data.get('remaining_quantity', 1.0) if position_data else 1.0

                # Check each target level
                for i, tp_pct in enumerate(self.config.tp_levels):
                    if pnl_pct >= tp_pct:
                        # Calculate partial exit quantity based on target level
                        if i == 0:  # First target - 50% exit
                            exit_quantity = min(0.5, remaining_quantity)
                        elif i == 1:  # Second target - 30% exit
                            exit_quantity = min(0.3, remaining_quantity)
                        else:  # Final target - remaining
                            exit_quantity = remaining_quantity

                        if exit_quantity > 0:
                            return ExitLevel(
                                exit_type=ExitType.TAKE_PROFIT,
                                level=entry_price * (1 + tp_pct),
                                quantity=exit_quantity,
                                signal_strength=ExitSignal.MODERATE if i < 2 else ExitSignal.STRONG,
                                triggered=True,
                                triggered_at=datetime.now(),
                                metadata={
                                    "reason": f"Take profit target {i+1} reached",
                                    "target_level": tp_pct,
                                    "profit_pct": pnl_pct,
                                    "target_index": i
                                }
                            )
            else:
                # Single target take profit
                if pnl_pct >= self.config.take_profit_pct:
                    return ExitLevel(
                        exit_type=ExitType.TAKE_PROFIT,
                        level=entry_price * (1 + self.config.take_profit_pct),
                        quantity=1.0,
                        signal_strength=ExitSignal.STRONG,
                        triggered=True,
                        triggered_at=datetime.now(),
                        metadata={
                            "reason": "Take profit target reached",
                            "profit_pct": pnl_pct
                        }
                    )

            return None

        except Exception as e:
            self.logger.error(f"Error evaluating take profit for {symbol}: {e}")
            return None

    async def _evaluate_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        pnl_pct: float,
        market_data: MarketData
    ) -> Optional[ExitLevel]:
        """Evaluate trailing stop condition"""
        try:
            # Get trailing stop state
            trailing_state = self.active_exits.get(symbol, [])

            # Check if trailing stop should be activated
            if pnl_pct < self.config.trailing_stop_activation:
                return None  # Not yet profitable enough

            # Calculate trailing stop level
            if self.config.adaptive_trailing:
                # Use ATR for adaptive trailing stop
                atr = market_data.close * 0.02  # Simplified ATR calculation
                trailing_distance = atr * 0.5  # 0.5 * ATR
            else:
                trailing_distance = current_price * self.config.trailing_stop_pct

            # Calculate trailing stop price
            trailing_stop_price = current_price - trailing_distance

            # Check if trailing stop triggered
            if current_price <= trailing_stop_price:
                return ExitLevel(
                    exit_type=ExitType.TRAILING_STOP,
                    level=trailing_stop_price,
                    quantity=1.0,
                    signal_strength=ExitSignal.STRONG,
                    triggered=True,
                    triggered_at=datetime.now(),
                    metadata={
                        "reason": "Trailing stop triggered",
                        "current_price": current_price,
                        "trailing_stop_price": trailing_stop_price,
                        "profit_pct": pnl_pct
                    }
                )

            # Update trailing stop level for next evaluation
            self._update_trailing_stop(symbol, trailing_stop_price, current_price)

            return None

        except Exception as e:
            self.logger.error(f"Error evaluating trailing stop for {symbol}: {e}")
            return None

    async def _evaluate_time_based(
        self,
        symbol: str,
        position_age: float,
        pnl_pct: float,
        market_data: MarketData,
        analysis_result: Optional[AnalysisResult] = None
    ) -> Optional[ExitLevel]:
        """Evaluate time-based exit condition"""
        try:
            # Check minimum hold time
            if position_age < self.config.min_hold_time:
                return None

            # Check maximum hold time
            if position_age > self.config.max_hold_time:
                return ExitLevel(
                    exit_type=ExitType.TIME_BASED,
                    level=market_data.close,
                    quantity=1.0,
                    signal_strength=ExitSignal.MODERATE,
                    triggered=True,
                    triggered_at=datetime.now(),
                    metadata={
                        "reason": "Maximum hold time exceeded",
                        "hold_time": position_age,
                        "max_time": self.config.max_hold_time
                    }
                )

            return None

        except Exception as e:
            self.logger.error(f"Error evaluating time-based exit for {symbol}: {e}")
            return None

    async def _evaluate_volatility_based(
        self,
        symbol: str,
        current_price: float,
        market_data: MarketData,
        analysis_result: Optional[AnalysisResult] = None
    ) -> Optional[ExitLevel]:
        """Evaluate volatility-based exit condition"""
        try:
            # Get volatility measures
            if analysis_result:
                volatility = analysis_result.technical_indicators.get('ATR', 0) / current_price
                if volatility > self.config.volatility_threshold * self.config.volatility_exit_multiplier:
                    return ExitLevel(
                        exit_type=ExitType.VOLATILITY_BASED,
                        level=current_price,
                        quantity=1.0,
                        signal_strength=ExitSignal.MODERATE,
                        triggered=True,
                        triggered_at=datetime.now(),
                        metadata={
                            "reason": "High volatility detected",
                            "volatility": volatility,
                            "threshold": self.config.volatility_threshold
                        }
                    )

            return None

        except Exception as e:
            self.logger.error(f"Error evaluating volatility-based exit for {symbol}: {e}")
            return None

    async def _evaluate_regime_based(
        self,
        symbol: str,
        analysis_result: AnalysisResult,
        pnl_pct: float,
        position_data: Optional[Dict[str, Any]] = None
    ) -> Optional[ExitLevel]:
        """Evaluate regime-based exit condition"""
        try:
            current_regime = analysis_result.market_regime

            # Get regime-specific exit thresholds
            regime_threshold = self.config.regime_exit_thresholds.get(current_regime, 0.02)

            # Exit if current regime suggests unfavorable conditions
            if pnl_pct < regime_threshold:
                return ExitLevel(
                    exit_type=ExitType.REGIME_BASED,
                    level=analysis_result.support_resistance.get('current_price', 0),
                    quantity=1.0,
                    signal_strength=ExitSignal.MODERATE,
                    triggered=True,
                    triggered_at=datetime.now(),
                    metadata={
                        "reason": f"Unfavorable regime: {current_regime}",
                        "regime": current_regime,
                        "threshold": regime_threshold,
                        "current_pnl": pnl_pct
                    }
                )

            return None

        except Exception as e:
            self.logger.error(f"Error evaluating regime-based exit for {symbol}: {e}")
            return None

    async def _evaluate_emergency(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        market_data: MarketData,
        analysis_result: Optional[AnalysisResult] = None
    ) -> Optional[ExitLevel]:
        """Evaluate emergency exit condition"""
        try:
            # Check for extreme drawdown
            drawdown_pct = (current_price - entry_price) / entry_price
            if drawdown_pct < -self.config.emergency_drawdown_threshold:
                return ExitLevel(
                    exit_type=ExitType.EMERGENCY,
                    level=current_price,
                    quantity=1.0,
                    signal_strength=ExitSignal.EMERGENCY,
                    triggered=True,
                    triggered_at=datetime.now(),
                    metadata={
                        "reason": "Emergency drawdown threshold breached",
                        "drawdown_pct": drawdown_pct,
                        "threshold": -self.config.emergency_drawdown_threshold
                    }
                )

            return None

        except Exception as e:
            self.logger.error(f"Error evaluating emergency exit for {symbol}: {e}")
            return None

    async def _evaluate_break_even(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        pnl_pct: float
    ) -> Optional[ExitLevel]:
        """Evaluate break-even exit condition"""
        try:
            if pnl_pct >= self.config.break_even_trigger:
                break_even_price = entry_price * (1 + self.config.break_even_buffer)
                if current_price <= break_even_price:
                    return ExitLevel(
                        exit_type=ExitType.BREAK_EVEN,
                        level=break_even_price,
                        quantity=1.0,
                        signal_strength=ExitSignal.WEAK,
                        triggered=True,
                        triggered_at=datetime.now(),
                        metadata={
                            "reason": "Break-even level breached",
                            "entry_price": entry_price,
                            "break_even_price": break_even_price,
                            "current_price": current_price,
                            "profit_pct": pnl_pct
                        }
                    )

            return None

        except Exception as e:
            self.logger.error(f"Error evaluating break-even exit for {symbol}: {e}")
            return None

    async def _process_exit_evaluations(
        self,
        exit_evaluations: List[ExitLevel],
        symbol: str,
        position_data: Optional[Dict[str, Any]] = None
    ) -> ExitStrategyResult:
        """Process all exit evaluations and determine final result"""
        try:
            result = ExitStrategyResult()
            total_exit_quantity = 0.0
            max_signal_strength = ExitSignal.NONE
            reasoning = []

            # Process each exit evaluation
            for exit_level in exit_evaluations:
                if exit_level.triggered:
                    # Add to total exit quantity
                    remaining_quantity = position_data.get('remaining_quantity', 1.0) if position_data else 1.0
                    actual_quantity = min(exit_level.quantity, remaining_quantity - total_exit_quantity)
                    total_exit_quantity += actual_quantity

                    # Track maximum signal strength
                    if exit_level.signal_strength.value > max_signal_strength.value:
                        max_signal_strength = exit_level.signal_strength
                        result.primary_exit_type = exit_level.exit_type

                    # Add to exit levels
                    result.exit_levels.append(exit_level)

                    # Add reasoning
                    reasoning.append(f"{exit_level.exit_type.value}: {exit_level.metadata.get('reason', 'Exit triggered')}")

            # Determine if we should exit
            result.should_exit = total_exit_quantity > 0
            result.total_exit_quantity = total_exit_quantity
            result.exit_signal = max_signal_strength
            result.reasoning = reasoning

            # Calculate confidence based on signal strength and number of exit signals
            signal_multiplier = {
                ExitSignal.NONE: 0.0,
                ExitSignal.WEAK: 0.3,
                ExitSignal.MODERATE: 0.6,
                ExitSignal.STRONG: 0.9,
                ExitSignal.EMERGENCY: 1.0
            }

            base_confidence = signal_multiplier[max_signal_strength]
            exit_count_bonus = min(0.1 * len(exit_evaluations), 0.1)  # Up to 10% bonus for multiple exits
            result.confidence = min(base_confidence + exit_count_bonus, 1.0)

            return result

        except Exception as e:
            self.logger.error(f"Error processing exit evaluations for {symbol}: {e}")
            return ExitStrategyResult()

    def _update_trailing_stop(self, symbol: str, trailing_stop_price: float, current_price: float) -> None:
        """Update trailing stop level"""
        # This would typically store the trailing stop state for the next evaluation
        if symbol not in self.active_exits:
            self.active_exits[symbol] = []

        # Update or add trailing stop level
        for exit_level in self.active_exits[symbol]:
            if exit_level.exit_type == ExitType.TRAILING_STOP:
                exit_level.level = trailing_stop_price
                break
        else:
            # Add new trailing stop level
            self.active_exits[symbol].append(ExitLevel(
                exit_type=ExitType.TRAILING_STOP,
                level=trailing_stop_price,
                quantity=0.0,  # Not triggered yet
                metadata={"current_price": current_price}
            ))

    def get_exit_configuration(self, symbol: str) -> Dict[str, Any]:
        """Get current exit configuration for a symbol"""
        return {
            "config": self.config.__dict__,
            "active_exits": [exit_level.__dict__ for exit_level in self.active_exits.get(symbol, [])],
            "exit_history": self.exit_history[-10:]  # Last 10 exits
        }

    def update_config(self, new_config: ExitStrategyConfig) -> None:
        """Update exit strategy configuration"""
        self.config = new_config
        self.logger.info("Exit strategy configuration updated")