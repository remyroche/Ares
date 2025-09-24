"""
Advanced Take Profit Management System

Sophisticated take profit strategies with multiple targets,
Fibonacci-based levels, and market-adaptive profit taking.
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


class TakeProfitType(Enum):
    """Types of take profit strategies"""
    FIXED_PERCENTAGE = "fixed_percentage"
    FIBONACCI_LEVELS = "fibonacci_levels"
    VOLATILITY_BASED = "volatility_based"
    REGIME_ADAPTIVE = "regime_adaptive"
    DYNAMIC_SCALING = "dynamic_scaling"
    PARTIAL_PROFIT = "partial_profit"


class TakeProfitStrategy(Enum):
    """Take profit strategy approaches"""
    SINGLE_TARGET = "single_target"
    MULTI_TARGET = "multi_target"
    SCALED_EXIT = "scaled_exit"
    PYRAMID_EXIT = "pyramid_exit"
    REGIME_BASED = "regime_based"


@dataclass
class TakeProfitLevel:
    """Individual take profit level"""
    level: float  # Price level
    quantity: float  # Percentage of position (0-1)
    strategy: TakeProfitStrategy
    confidence: float = 0.0
    priority: int = 1  # 1 = highest priority
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TakeProfitConfig:
    """Configuration for take profit strategies"""
    # Basic Settings
    enabled: bool = True
    base_profit_target: float = 0.04  # 4% default

    # Multi-target Settings
    multi_target_enabled: bool = True
    target_levels: List[float] = field(default_factory=lambda: [0.02, 0.04, 0.06, 0.10])  # Multiple targets
    target_quantities: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.2, 0.2])  # Corresponding quantities

    # Fibonacci Settings
    fibonacci_enabled: bool = True
    fib_levels: List[float] = field(default_factory=lambda: [0.236, 0.382, 0.618, 1.0, 1.618])

    # Volatility-based Settings
    volatility_adjustment: bool = True
    volatility_multiplier: float = 1.5  # Increase targets in high volatility
    min_volatility_threshold: float = 0.02  # 2%
    max_volatility_threshold: float = 0.08  # 8%

    # Regime-based Settings
    regime_adaptive: bool = True
    regime_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "bull_trend": 1.2,
        "bear_trend": 0.8,
        "sideways": 0.9,
        "volatile": 1.0
    })

    # Dynamic Scaling Settings
    dynamic_scaling: bool = True
    scale_with_confidence: bool = True
    scale_with_time: bool = True
    max_scale_factor: float = 2.0  # Maximum scaling

    # Risk-based Settings
    risk_adjusted_targets: bool = True
    risk_multiplier: float = 1.0  # Higher = more aggressive targets


@dataclass
class TakeProfitResult:
    """Result of take profit evaluation"""
    should_exit: bool = False
    exit_levels: List[TakeProfitLevel] = field(default_factory=list)
    total_exit_quantity: float = 0.0
    hit_targets: List[int] = field(default_factory=list)  # Which targets were hit
    confidence: float = 0.0
    reasoning: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TakeProfitManager:
    """
    Advanced take profit management system that provides sophisticated
    profit-taking strategies with multiple targets and adaptive scaling.
    """

    def __init__(self, config: Optional[TakeProfitConfig] = None):
        self.config = config or TakeProfitConfig()
        self.logger = logging.getLogger(__name__)

        # State tracking
        self.active_targets: Dict[str, List[TakeProfitLevel]] = {}
        self.target_history: List[Dict[str, Any]] = []
        self.hit_targets_cache: Dict[str, List[int]] = {}

        # Performance tracking
        self.target_performance: Dict[str, Dict[str, Any]] = {}

    async def evaluate_take_profit(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        position_data: Optional[Dict[str, Any]] = None,
        market_data: Optional[MarketData] = None,
        analysis_result: Optional[AnalysisResult] = None
    ) -> TakeProfitResult:
        """
        Evaluate take profit opportunities

        Args:
            symbol: Trading symbol
            entry_price: Entry price of position
            current_price: Current market price
            position_data: Position information
            market_data: Current market data
            analysis_result: Latest analysis

        Returns:
            TakeProfitResult: Take profit evaluation
        """
        try:
            result = TakeProfitResult()

            # Calculate profit percentage
            profit_pct = (current_price - entry_price) / entry_price

            # Check if any targets are hit
            if profit_pct <= 0:
                return result  # No profit yet

            # Get position context
            remaining_quantity = position_data.get('remaining_quantity', 1.0) if position_data else 1.0
            position_age = position_data.get('age_seconds', 0) if position_data else 0

            # Generate take profit levels
            target_levels = await self._generate_take_profit_levels(
                symbol, entry_price, current_price, profit_pct, position_data, market_data, analysis_result
            )

            # Check which targets are hit
            hit_targets = []
            exit_levels = []

            for i, target in enumerate(target_levels):
                if current_price >= target.level and target.level > entry_price:
                    hit_targets.append(i)
                    exit_levels.append(target)

            # Process hit targets
            if hit_targets:
                result.should_exit = True
                result.hit_targets = hit_targets
                result.exit_levels = exit_levels

                # Calculate total exit quantity
                total_exit_qty = 0.0
                for target in exit_levels:
                    actual_qty = min(target.quantity, remaining_quantity - total_exit_qty)
                    total_exit_qty += actual_qty

                result.total_exit_quantity = total_exit_qty

                # Generate reasoning
                for i, target in enumerate(exit_levels):
                    reasoning = f"Target {i+1} hit at {target.level".4f"} ({(target.level/entry_price-1)*100".2f"}%)"
                    result.reasoning.append(reasoning)

                # Calculate confidence
                result.confidence = self._calculate_confidence(hit_targets, target_levels, analysis_result)

            # Store target levels for next evaluation
            self.active_targets[symbol] = target_levels

            return result

        except Exception as e:
            self.logger.error(f"Error evaluating take profit for {symbol}: {e}")
            return TakeProfitResult()

    async def _generate_take_profit_levels(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        profit_pct: float,
        position_data: Optional[Dict[str, Any]] = None,
        market_data: Optional[MarketData] = None,
        analysis_result: Optional[AnalysisResult] = None
    ) -> List[TakeProfitLevel]:
        """Generate take profit levels based on various strategies"""
        try:
            targets = []

            # Multi-target strategy
            if self.config.multi_target_enabled:
                multi_targets = self._generate_multi_target_levels(entry_price, position_data, analysis_result)
                targets.extend(multi_targets)

            # Fibonacci levels
            if self.config.fibonacci_enabled:
                fib_targets = self._generate_fibonacci_levels(entry_price, current_price, analysis_result)
                targets.extend(fib_targets)

            # Volatility-based targets
            if self.config.volatility_adjustment:
                vol_targets = self._generate_volatility_based_levels(entry_price, analysis_result)
                targets.extend(vol_targets)

            # Regime-adaptive targets
            if self.config.regime_adaptive and analysis_result:
                regime_targets = self._generate_regime_based_levels(entry_price, analysis_result)
                targets.extend(regime_targets)

            # Dynamic scaling
            if self.config.dynamic_scaling:
                targets = self._apply_dynamic_scaling(targets, position_data, analysis_result)

            # Sort by priority and level
            targets.sort(key=lambda x: (x.priority, x.level))

            # Remove duplicates and consolidate
            consolidated_targets = self._consolidate_targets(targets, entry_price)

            return consolidated_targets

        except Exception as e:
            self.logger.error(f"Error generating take profit levels for {symbol}: {e}")
            return []

    def _generate_multi_target_levels(
        self,
        entry_price: float,
        position_data: Optional[Dict[str, Any]] = None,
        analysis_result: Optional[AnalysisResult] = None
    ) -> List[TakeProfitLevel]:
        """Generate multi-target levels"""
        try:
            targets = []
            base_strategy = TakeProfitStrategy.MULTI_TARGET

            for i, (target_pct, quantity) in enumerate(zip(self.config.target_levels, self.config.target_quantities)):
                level = entry_price * (1 + target_pct)
                confidence = self._calculate_target_confidence(target_pct, i, analysis_result)

                target = TakeProfitLevel(
                    level=level,
                    quantity=quantity,
                    strategy=base_strategy,
                    confidence=confidence,
                    priority=i + 1,
                    metadata={
                        "target_index": i,
                        "target_pct": target_pct,
                        "base_target": True
                    }
                )
                targets.append(target)

            return targets

        except Exception as e:
            self.logger.error(f"Error generating multi-target levels: {e}")
            return []

    def _generate_fibonacci_levels(
        self,
        entry_price: float,
        current_price: float,
        analysis_result: Optional[AnalysisResult] = None
    ) -> List[TakeProfitLevel]:
        """Generate Fibonacci-based levels"""
        try:
            targets = []
            base_strategy = TakeProfitStrategy.MULTI_TARGET

            # Calculate Fibonacci levels based on recent swing
            if analysis_result and 'support_resistance' in analysis_result.features:
                swing_high = analysis_result.features.get('swing_high', current_price)
                swing_low = analysis_result.features.get('swing_low', entry_price)

                # Use recent price action for Fibonacci calculation
                fib_range = swing_high - swing_low
                if fib_range > 0:
                    for i, fib_level in enumerate(self.config.fib_levels):
                        level = swing_low + (fib_range * fib_level)
                        if level > entry_price:  # Only consider levels above entry
                            quantity = 0.25 if i < 2 else 0.15  # Higher quantity for early targets

                            target = TakeProfitLevel(
                                level=level,
                                quantity=quantity,
                                strategy=TakeProfitStrategy.MULTI_TARGET,
                                confidence=0.7,
                                priority=i + 5,  # Lower priority than base targets
                                metadata={
                                    "fibonacci_level": fib_level,
                                    "swing_high": swing_high,
                                    "swing_low": swing_low,
                                    "fibonacci_based": True
                                }
                            )
                            targets.append(target)

            return targets

        except Exception as e:
            self.logger.error(f"Error generating Fibonacci levels: {e}")
            return []

    def _generate_volatility_based_levels(
        self,
        entry_price: float,
        analysis_result: Optional[AnalysisResult] = None
    ) -> List[TakeProfitLevel]:
        """Generate volatility-based levels"""
        try:
            targets = []

            if not analysis_result:
                return targets

            # Get volatility measures
            volatility = analysis_result.technical_indicators.get('ATR', 0) / entry_price
            volume = analysis_result.technical_indicators.get('volume_ratio', 1.0)

            # Adjust targets based on volatility
            if volatility > self.config.max_volatility_threshold:
                # Lower targets in high volatility
                adjusted_targets = [t * 0.7 for t in self.config.target_levels]
            elif volatility > self.config.min_volatility_threshold:
                # Scale targets with volatility
                scale_factor = min(volatility * self.config.volatility_multiplier, 2.0)
                adjusted_targets = [t * scale_factor for t in self.config.target_levels]
            else:
                adjusted_targets = self.config.target_levels

            # Create targets
            for i, target_pct in enumerate(adjusted_targets):
                level = entry_price * (1 + target_pct)
                quantity = self.config.target_quantities[i] if i < len(self.config.target_quantities) else 0.2

                target = TakeProfitLevel(
                    level=level,
                    quantity=quantity,
                    strategy=TakeProfitStrategy.VOLATILITY_BASED,
                    confidence=0.6,
                    priority=i + 10,
                    metadata={
                        "volatility_adjusted": True,
                        "volatility": volatility,
                        "original_target": self.config.target_levels[i]
                    }
                )
                targets.append(target)

            return targets

        except Exception as e:
            self.logger.error(f"Error generating volatility-based levels: {e}")
            return []

    def _generate_regime_based_levels(
        self,
        entry_price: float,
        analysis_result: AnalysisResult
    ) -> List[TakeProfitLevel]:
        """Generate regime-adaptive levels"""
        try:
            targets = []
            current_regime = analysis_result.market_regime

            # Get regime multiplier
            regime_multiplier = self.config.regime_multipliers.get(current_regime, 1.0)

            # Adjust targets based on regime
            adjusted_targets = [t * regime_multiplier for t in self.config.target_levels]

            for i, target_pct in enumerate(adjusted_targets):
                level = entry_price * (1 + target_pct)
                quantity = self.config.target_quantities[i] if i < len(self.config.target_quantities) else 0.2

                target = TakeProfitLevel(
                    level=level,
                    quantity=quantity,
                    strategy=TakeProfitStrategy.REGIME_BASED,
                    confidence=0.8,
                    priority=i + 15,
                    metadata={
                        "regime": current_regime,
                        "regime_multiplier": regime_multiplier,
                        "regime_adjusted": True
                    }
                )
                targets.append(target)

            return targets

        except Exception as e:
            self.logger.error(f"Error generating regime-based levels: {e}")
            return []

    def _apply_dynamic_scaling(
        self,
        targets: List[TakeProfitLevel],
        position_data: Optional[Dict[str, Any]] = None,
        analysis_result: Optional[AnalysisResult] = None
    ) -> List[TakeProfitLevel]:
        """Apply dynamic scaling to targets"""
        try:
            scaled_targets = []

            for target in targets:
                scaled_target = target.__class__(
                    level=target.level,
                    quantity=target.quantity,
                    strategy=target.strategy,
                    confidence=target.confidence,
                    priority=target.priority,
                    metadata=target.metadata.copy()
                )

                # Scale with confidence
                if self.config.scale_with_confidence and analysis_result:
                    confidence_score = analysis_result.confidence
                    scale_factor = 1.0 + (confidence_score * 0.5)  # Up to 50% scaling
                    scaled_target.level *= scale_factor
                    scaled_target.metadata["confidence_scaled"] = True
                    scaled_target.metadata["confidence_score"] = confidence_score

                # Scale with time (longer positions get higher targets)
                if self.config.scale_with_time and position_data:
                    position_age = position_data.get('age_seconds', 0)
                    if position_age > 3600:  # After 1 hour
                        time_scale = min(position_age / 7200, 0.5)  # Max 50% scaling after 2 hours
                        scaled_target.level *= (1 + time_scale)
                        scaled_target.metadata["time_scaled"] = True
                        scaled_target.metadata["position_age"] = position_age

                scaled_targets.append(scaled_target)

            return scaled_targets

        except Exception as e:
            self.logger.error(f"Error applying dynamic scaling: {e}")
            return targets

    def _consolidate_targets(
        self,
        targets: List[TakeProfitLevel],
        entry_price: float
    ) -> List[TakeProfitLevel]:
        """Consolidate duplicate targets and remove unrealistic ones"""
        try:
            consolidated = []

            # Sort by level
            targets.sort(key=lambda x: x.level)

            # Remove targets too close to entry price
            min_target_distance = entry_price * 0.005  # 0.5% minimum distance
            targets = [t for t in targets if t.level >= entry_price + min_target_distance]

            # Consolidate nearby targets
            tolerance = entry_price * 0.01  # 1% tolerance

            for target in targets:
                # Check if this target is close to existing ones
                close_existing = None
                for existing in consolidated:
                    if abs(target.level - existing.level) / entry_price < tolerance:
                        close_existing = existing
                        break

                if close_existing:
                    # Merge with existing target
                    merged_quantity = (close_existing.quantity + target.quantity) / 2
                    close_existing.quantity = merged_quantity
                    close_existing.confidence = max(close_existing.confidence, target.confidence)
                    close_existing.metadata.update(target.metadata)
                else:
                    # Add new target
                    consolidated.append(target)

            return consolidated

        except Exception as e:
            self.logger.error(f"Error consolidating targets: {e}")
            return targets

    def _calculate_target_confidence(
        self,
        target_pct: float,
        target_index: int,
        analysis_result: Optional[AnalysisResult] = None
    ) -> float:
        """Calculate confidence for a target level"""
        try:
            base_confidence = 0.5  # Base confidence

            # Higher confidence for earlier targets
            index_bonus = max(0, 0.3 - (target_index * 0.05))
            base_confidence += index_bonus

            # Adjust based on analysis confidence
            if analysis_result:
                analysis_confidence = analysis_result.confidence
                base_confidence = (base_confidence + analysis_confidence) / 2

            return min(base_confidence, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating target confidence: {e}")
            return 0.5

    def _calculate_confidence(
        self,
        hit_targets: List[int],
        all_targets: List[TakeProfitLevel],
        analysis_result: Optional[AnalysisResult] = None
    ) -> float:
        """Calculate overall confidence for take profit signal"""
        try:
            if not hit_targets:
                return 0.0

            # Base confidence from number of targets hit
            base_confidence = len(hit_targets) / max(len(all_targets), 1)

            # Bonus for earlier targets
            for target_index in hit_targets:
                if target_index < len(all_targets):
                    target = all_targets[target_index]
                    if target.priority <= 3:  # Early targets
                        base_confidence += 0.1

            # Analysis confidence bonus
            if analysis_result:
                base_confidence = (base_confidence + analysis_result.confidence) / 2

            return min(base_confidence, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.0

    def get_target_status(self, symbol: str) -> Dict[str, Any]:
        """Get current target status for a symbol"""
        return {
            "active_targets": [target.__dict__ for target in self.active_targets.get(symbol, [])],
            "hit_targets": self.hit_targets_cache.get(symbol, []),
            "config": self.config.__dict__
        }

    def update_config(self, new_config: TakeProfitConfig) -> None:
        """Update take profit configuration"""
        self.config = new_config
        self.logger.info("Take profit configuration updated")