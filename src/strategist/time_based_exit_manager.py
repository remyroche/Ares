"""
Time-Based Exit Strategy Management System

Intelligent time-based exits with regime awareness, market condition
adaptation, and progressive exit strategies.
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


class TimeBasedStrategy(Enum):
    """Time-based exit strategies"""
    FIXED_TIME = "fixed_time"
    PROGRESSIVE_EXIT = "progressive_exit"
    REGIME_AWARE = "regime_aware"
    PROFIT_TIME_LOCK = "profit_time_lock"
    MARKET_HOURS_BASED = "market_hours_based"
    WEEKDAY_BASED = "weekday_based"


class MarketSession(Enum):
    """Market trading sessions"""
    ASIAN = "asian"
    EUROPEAN = "european"
    AMERICAN = "american"
    OVERNIGHT = "overnight"
    WEEKEND = "weekend"


@dataclass
class TimeBasedConfig:
    """Configuration for time-based exit strategies"""
    # Basic Settings
    enabled: bool = True
    min_hold_time: int = 300  # 5 minutes minimum
    max_hold_time: int = 7200  # 2 hours maximum

    # Strategy Settings
    primary_strategy: TimeBasedStrategy = TimeBasedStrategy.REGIME_AWARE
    progressive_exit_enabled: bool = True
    profit_time_lock_enabled: bool = True

    # Time-based Settings
    exit_by_market_hours: bool = True
    exit_before_weekend: bool = True
    exit_on_low_liquidity: bool = True

    # Regime-aware Settings
    regime_specific_times: Dict[str, int] = field(default_factory=lambda: {
        "bull_trend": 5400,      # 1.5 hours in bull trends
        "bear_trend": 3600,      # 1 hour in bear trends
        "sideways": 7200,        # 2 hours in sideways markets
        "volatile": 1800,        # 30 minutes in volatile markets
        "ranging": 9000          # 2.5 hours in ranging markets
    })

    # Progressive Exit Settings
    progressive_exit_times: List[int] = field(default_factory=lambda: [1800, 3600, 5400])  # 30min, 1hr, 1.5hr
    progressive_exit_quantities: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.4])  # 30%, 30%, 40%

    # Profit Time Lock Settings
    profit_lock_times: Dict[float, int] = field(default_factory=lambda: {
        0.02: 1800,  # 2% profit: lock for 30 minutes
        0.05: 3600,  # 5% profit: lock for 1 hour
        0.10: 7200   # 10% profit: lock for 2 hours
    })

    # Market Hours Settings
    market_sessions: Dict[MarketSession, Dict[str, Any]] = field(default_factory=lambda: {
        MarketSession.ASIAN: {"start_hour": 0, "end_hour": 8, "exit_preference": "low"},
        MarketSession.EUROPEAN: {"start_hour": 8, "end_hour": 16, "exit_preference": "medium"},
        MarketSession.AMERICAN: {"start_hour": 16, "end_hour": 24, "exit_preference": "high"},
        MarketSession.OVERNIGHT: {"start_hour": 0, "end_hour": 6, "exit_preference": "low"},
        MarketSession.WEEKEND: {"exit_immediately": True}
    })

    # Weekday Settings
    weekday_preferences: Dict[int, str] = field(default_factory=lambda: {
        0: "low",      # Sunday
        1: "medium",   # Monday
        2: "high",     # Tuesday
        3: "high",     # Wednesday
        4: "medium",   # Thursday
        5: "low",      # Friday
        6: "low"       # Saturday
    })


@dataclass
class PositionTimer:
    """Timer for tracking position duration"""
    symbol: str
    entry_time: datetime
    entry_price: float
    min_exit_time: Optional[datetime] = None
    max_exit_time: Optional[datetime] = None
    profit_lock_time: Optional[datetime] = None
    progressive_exits: List[datetime] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeBasedResult:
    """Result of time-based exit evaluation"""
    should_exit: bool = False
    exit_levels: List[ExitLevel] = field(default_factory=list)
    exit_reason: str = ""
    confidence: float = 0.0
    remaining_time: int = 0  # Seconds until next exit consideration
    metadata: Dict[str, Any] = field(default_factory=dict)


class TimeBasedExitManager:
    """
    Advanced time-based exit management system that provides intelligent
    timing strategies with regime awareness and market condition adaptation.
    """

    def __init__(self, config: Optional[TimeBasedConfig] = None):
        self.config = config or TimeBasedConfig()
        self.logger = logging.getLogger(__name__)

        # State tracking
        self.active_timers: Dict[str, PositionTimer] = {}
        self.exit_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}

    async def evaluate_time_based_exit(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        position_data: Optional[Dict[str, Any]] = None,
        market_data: Optional[MarketData] = None,
        analysis_result: Optional[AnalysisResult] = None
    ) -> TimeBasedResult:
        """
        Evaluate time-based exit conditions

        Args:
            symbol: Trading symbol
            entry_price: Entry price of position
            current_price: Current market price
            position_data: Position information
            market_data: Current market data
            analysis_result: Latest analysis

        Returns:
            TimeBasedResult: Time-based exit evaluation
        """
        try:
            result = TimeBasedResult()

            # Get position age
            position_age = self._get_position_age(symbol, position_data)
            if position_age < self.config.min_hold_time:
                result.remaining_time = self.config.min_hold_time - position_age
                result.metadata["reason"] = f"Minimum hold time not reached: {position_age}s / {self.config.min_hold_time}s"
                return result

            # Check if position timer exists
            if symbol not in self.active_timers:
                await self._initialize_position_timer(symbol, entry_price, current_price, analysis_result)

            timer = self.active_timers[symbol]

            # Calculate profit percentage
            profit_pct = (current_price - entry_price) / entry_price

            # Evaluate different time-based strategies
            exit_signals = []

            # Check market hours strategy
            if self.config.exit_by_market_hours:
                market_exit = await self._evaluate_market_hours_exit(symbol, timer, current_price, market_data)
                if market_exit:
                    exit_signals.append(market_exit)

            # Check regime-aware strategy
            if self.config.primary_strategy == TimeBasedStrategy.REGIME_AWARE and analysis_result:
                regime_exit = await self._evaluate_regime_aware_exit(symbol, timer, profit_pct, analysis_result)
                if regime_exit:
                    exit_signals.append(regime_exit)

            # Check profit time lock
            if self.config.profit_time_lock_enabled:
                lock_exit = await self._evaluate_profit_time_lock(symbol, timer, profit_pct, current_price)
                if lock_exit:
                    exit_signals.append(lock_exit)

            # Check progressive exit
            if self.config.progressive_exit_enabled:
                progressive_exit = await self._evaluate_progressive_exit(symbol, timer, position_age, profit_pct)
                if progressive_exit:
                    exit_signals.append(progressive_exit)

            # Check maximum hold time
            if position_age > self.config.max_hold_time:
                max_time_exit = ExitLevel(
                    exit_type=ExitType.TIME_BASED,
                    level=current_price,
                    quantity=1.0,
                    signal_strength=ExitSignal.MODERATE,
                    triggered=True,
                    triggered_at=datetime.now(),
                    metadata={
                        "reason": "Maximum hold time exceeded",
                        "max_time": self.config.max_hold_time,
                        "position_age": position_age
                    }
                )
                exit_signals.append(max_time_exit)

            # Process exit signals
            if exit_signals:
                result.should_exit = True
                result.exit_levels = exit_signals

                # Determine primary exit reason and confidence
                primary_signal = max(exit_signals, key=lambda x: x.signal_strength.value)
                result.exit_reason = primary_signal.metadata.get("reason", "Time-based exit triggered")
                result.confidence = self._calculate_confidence(exit_signals, profit_pct, analysis_result)

                # Update timer
                timer.metadata["last_exit_evaluation"] = datetime.now()

            return result

        except Exception as e:
            self.logger.error(f"Error evaluating time-based exit for {symbol}: {e}")
            return TimeBasedResult()

    async def _initialize_position_timer(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        analysis_result: Optional[AnalysisResult] = None
    ) -> None:
        """Initialize position timer"""
        try:
            now = datetime.now()

            # Set minimum and maximum exit times
            min_exit_time = now + timedelta(seconds=self.config.min_hold_time)
            max_exit_time = now + timedelta(seconds=self.config.max_hold_time)

            # Set progressive exit times
            progressive_exits = []
            for exit_time in self.config.progressive_exit_times:
                progressive_exits.append(now + timedelta(seconds=exit_time))

            timer = PositionTimer(
                symbol=symbol,
                entry_time=now,
                entry_price=entry_price,
                min_exit_time=min_exit_time,
                max_exit_time=max_exit_time,
                progressive_exits=progressive_exits,
                metadata={
                    "initialization_time": now,
                    "regime_at_entry": analysis_result.market_regime if analysis_result else None
                }
            )

            self.active_timers[symbol] = timer
            self.logger.info(f"Initialized timer for {symbol}: min={self.config.min_hold_time}s, max={self.config.max_hold_time}s")

        except Exception as e:
            self.logger.error(f"Error initializing position timer for {symbol}: {e}")

    async def _evaluate_market_hours_exit(
        self,
        symbol: str,
        timer: PositionTimer,
        current_price: float,
        market_data: Optional[MarketData] = None
    ) -> Optional[ExitLevel]:
        """Evaluate market hours based exit"""
        try:
            now = datetime.now()
            current_hour = now.hour
            current_weekday = now.weekday()

            # Check weekend exit
            if self.config.exit_before_weekend and current_weekday >= 5:  # Saturday/Sunday
                return ExitLevel(
                    exit_type=ExitType.TIME_BASED,
                    level=current_price,
                    quantity=1.0,
                    signal_strength=ExitSignal.MODERATE,
                    triggered=True,
                    triggered_at=now,
                    metadata={
                        "reason": "Weekend exit - reduced liquidity",
                        "weekday": current_weekday,
                        "hour": current_hour
                    }
                )

            # Check market session preferences
            current_session = self._get_current_market_session(current_hour)

            if current_session:
                session_config = self.config.market_sessions[current_session]
                exit_preference = session_config.get("exit_preference", "medium")

                if exit_preference == "low":
                    # Consider partial exit in low liquidity sessions
                    return ExitLevel(
                        exit_type=ExitType.TIME_BASED,
                        level=current_price,
                        quantity=0.5,  # Partial exit
                        signal_strength=ExitSignal.WEAK,
                        triggered=True,
                        triggered_at=now,
                        metadata={
                            "reason": f"Low liquidity session: {current_session.value}",
                            "session": current_session.value,
                            "preference": exit_preference
                        }
                    )

            return None

        except Exception as e:
            self.logger.error(f"Error evaluating market hours exit: {e}")
            return None

    async def _evaluate_regime_aware_exit(
        self,
        symbol: str,
        timer: PositionTimer,
        profit_pct: float,
        analysis_result: AnalysisResult
    ) -> Optional[ExitLevel]:
        """Evaluate regime-aware exit"""
        try:
            current_regime = analysis_result.market_regime
            position_age = (datetime.now() - timer.entry_time).seconds

            # Get regime-specific hold time
            regime_hold_time = self.config.regime_specific_times.get(current_regime, self.config.max_hold_time)

            if position_age > regime_hold_time:
                return ExitLevel(
                    exit_type=ExitType.TIME_BASED,
                    level=analysis_result.support_resistance.get('current_price', timer.entry_price * (1 + profit_pct)),
                    quantity=1.0,
                    signal_strength=ExitSignal.MODERATE,
                    triggered=True,
                    triggered_at=datetime.now(),
                    metadata={
                        "reason": f"Regime-specific hold time exceeded: {current_regime}",
                        "regime": current_regime,
                        "regime_hold_time": regime_hold_time,
                        "position_age": position_age
                    }
                )

            return None

        except Exception as e:
            self.logger.error(f"Error evaluating regime-aware exit: {e}")
            return None

    async def _evaluate_profit_time_lock(
        self,
        symbol: str,
        timer: PositionTimer,
        profit_pct: float,
        current_price: float
    ) -> Optional[ExitLevel]:
        """Evaluate profit time lock exit"""
        try:
            # Find applicable profit lock
            applicable_lock = None
            lock_duration = 0

            for profit_threshold, duration in self.config.profit_lock_times.items():
                if profit_pct >= profit_threshold:
                    if applicable_lock is None or profit_threshold > applicable_lock:
                        applicable_lock = profit_threshold
                        lock_duration = duration

            if applicable_lock is not None:
                # Check if lock time has expired
                position_age = (datetime.now() - timer.entry_time).seconds

                if position_age < lock_duration:
                    return None  # Still locked

                # Lock has expired - consider exit
                return ExitLevel(
                    exit_type=ExitType.TIME_BASED,
                    level=current_price,
                    quantity=0.5 if profit_pct > applicable_lock * 1.5 else 1.0,
                    signal_strength=ExitSignal.MODERATE,
                    triggered=True,
                    triggered_at=datetime.now(),
                    metadata={
                        "reason": f"Profit time lock expired after {lock_duration}s",
                        "profit_threshold": applicable_lock,
                        "lock_duration": lock_duration,
                        "position_age": position_age
                    }
                )

            return None

        except Exception as e:
            self.logger.error(f"Error evaluating profit time lock: {e}")
            return None

    async def _evaluate_progressive_exit(
        self,
        symbol: str,
        timer: PositionTimer,
        position_age: int,
        profit_pct: float
    ) -> Optional[ExitLevel]:
        """Evaluate progressive exit"""
        try:
            # Check each progressive exit time
            for i, (exit_time, exit_quantity) in enumerate(zip(self.config.progressive_exit_times, self.config.progressive_exit_quantities)):
                if position_age >= exit_time:
                    # Check if this progressive exit hasn't been triggered yet
                    if len(timer.progressive_exits) <= i:
                        return ExitLevel(
                            exit_type=ExitType.TIME_BASED,
                            level=timer.entry_price * (1 + profit_pct),
                            quantity=exit_quantity,
                            signal_strength=ExitSignal.WEAK,
                            triggered=True,
                            triggered_at=datetime.now(),
                            metadata={
                                "reason": f"Progressive exit {i+1}: {exit_time}s reached",
                                "progressive_exit_index": i,
                                "exit_time": exit_time,
                                "exit_quantity": exit_quantity
                            }
                        )

            return None

        except Exception as e:
            self.logger.error(f"Error evaluating progressive exit: {e}")
            return None

    def _get_position_age(self, symbol: str, position_data: Optional[Dict[str, Any]] = None) -> int:
        """Get position age in seconds"""
        if position_data and 'entry_time' in position_data:
            entry_time = position_data['entry_time']
            if isinstance(entry_time, datetime):
                return int((datetime.now() - entry_time).total_seconds())

        # Fallback: use timer if available
        if symbol in self.active_timers:
            return int((datetime.now() - self.active_timers[symbol].entry_time).total_seconds())

        return 0

    def _get_current_market_session(self, hour: int) -> Optional[MarketSession]:
        """Get current market session based on hour"""
        if 0 <= hour < 8:
            return MarketSession.ASIAN
        elif 8 <= hour < 16:
            return MarketSession.EUROPEAN
        elif 16 <= hour < 24:
            return MarketSession.AMERICAN
        else:
            return None

    def _calculate_confidence(
        self,
        exit_signals: List[ExitLevel],
        profit_pct: float,
        analysis_result: Optional[AnalysisResult] = None
    ) -> float:
        """Calculate confidence for time-based exit signal"""
        try:
            if not exit_signals:
                return 0.0

            # Base confidence from signal strength
            max_strength = max(signal.signal_strength.value for signal in exit_signals)
            base_confidence = max_strength * 0.3  # Max 30% from strength

            # Profit bonus
            if profit_pct > 0:
                profit_bonus = min(profit_pct * 0.1, 0.3)  # Up to 30% bonus
                base_confidence += profit_bonus

            # Regime bonus
            if analysis_result:
                regime_confidence = analysis_result.confidence * 0.2  # 20% from analysis
                base_confidence += regime_confidence

            # Multiple signals bonus
            if len(exit_signals) > 1:
                signal_bonus = min(len(exit_signals) * 0.1, 0.2)  # Up to 20% for multiple signals
                base_confidence += signal_bonus

            return min(base_confidence, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def get_timer_status(self, symbol: str) -> Dict[str, Any]:
        """Get current timer status for a symbol"""
        if symbol not in self.active_timers:
            return {"active": False, "message": "No timer active"}

        timer = self.active_timers[symbol]
        now = datetime.now()
        age = (now - timer.entry_time).total_seconds()

        return {
            "active": True,
            "symbol": timer.symbol,
            "entry_time": timer.entry_time,
            "age_seconds": age,
            "min_exit_time": timer.min_exit_time,
            "max_exit_time": timer.max_exit_time,
            "progressive_exits": timer.progressive_exits,
            "config": self.config.__dict__
        }

    def update_config(self, new_config: TimeBasedConfig) -> None:
        """Update time-based exit configuration"""
        self.config = new_config
        self.logger.info("Time-based exit configuration updated")