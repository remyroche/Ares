"""
Exit Strategy Orchestrator

Main orchestrator that coordinates all exit strategies and provides
unified exit signals with intelligent prioritization and risk management.
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
from .exit_strategy_manager import ExitStrategyManager, ExitStrategyConfig, ExitStrategyResult
from .take_profit_manager import TakeProfitManager, TakeProfitConfig, TakeProfitResult
from .trailing_stop_manager import TrailingStopManager, TrailingStopConfig, TrailingStopResult
from .time_based_exit_manager import TimeBasedExitManager, TimeBasedConfig, TimeBasedResult


class ExitPriority(Enum):
    """Exit strategy priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class ExitOrchestratorConfig:
    """Configuration for exit strategy orchestrator"""
    def __init__(self):
        # Strategy enablement
        self.enable_exit_strategies = True
        self.enable_take_profit = True
        self.enable_trailing_stop = True
        self.enable_time_based = True

        # Priority settings
        self.emergency_priority_threshold = 0.8  # 80% confidence for emergency
        self.critical_priority_threshold = 0.6   # 60% confidence for critical
        self.default_priority_threshold = 0.3    # 30% confidence for default

        # Exit coordination
        self.allow_simultaneous_exits = True
        self.max_concurrent_exits = 3
        self.exit_conflict_resolution = "highest_priority"

        # Risk management
        self.max_daily_exits = 10
        self.max_position_exits = 3
        self.cooldown_period = 300  # 5 minutes between exits

        # Performance tracking
        self.track_exit_performance = True
        self.adaptive_learning = True
        self.performance_window = 100  # Last 100 exits for analysis


@dataclass
class UnifiedExitResult:
    """Unified exit strategy result"""
    should_exit: bool = False
    exit_levels: List[Dict[str, Any]] = field(default_factory=list)
    total_exit_quantity: float = 0.0
    primary_strategy: str = ""
    priority: ExitPriority = ExitPriority.LOW
    confidence: float = 0.0
    reasoning: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    performance_impact: Dict[str, Any] = field(default_factory=dict)


class ExitStrategyOrchestrator:
    """
    Main orchestrator for all exit strategies. Coordinates multiple exit types,
    resolves conflicts, and provides unified exit signals with intelligent
    prioritization and risk management.
    """

    def __init__(self, config: Optional[ExitOrchestratorConfig] = None):
        self.config = config or ExitOrchestratorConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize strategy managers
        self.exit_strategy_manager = ExitStrategyManager()
        self.take_profit_manager = TakeProfitManager()
        self.trailing_stop_manager = TrailingStopManager()
        self.time_based_manager = TimeBasedExitManager()

        # State tracking
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.exit_history: List[Dict[str, Any]] = []
        self.daily_exit_counts: Dict[str, int] = {}
        self.performance_metrics: Dict[str, Any] = {}

        # Learning and adaptation
        self.strategy_performance: Dict[str, List[float]] = {}
        self.adaptive_weights: Dict[str, float] = {
            "exit_strategy": 1.0,
            "take_profit": 1.0,
            "trailing_stop": 1.0,
            "time_based": 1.0
        }

    async def evaluate_all_exits(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        position_data: Optional[Dict[str, Any]] = None,
        market_data: Optional[MarketData] = None,
        analysis_result: Optional[AnalysisResult] = None,
        strategy_result: Optional[StrategyResult] = None
    ) -> UnifiedExitResult:
        """
        Evaluate all exit strategies and provide unified result

        Args:
            symbol: Trading symbol
            entry_price: Entry price of position
            current_price: Current market price
            position_data: Position information
            market_data: Current market data
            analysis_result: Latest analysis
            strategy_result: Strategy context

        Returns:
            UnifiedExitResult: Comprehensive exit evaluation
        """
        try:
            result = UnifiedExitResult()

            # Check daily exit limits
            if not self._check_exit_limits(symbol):
                result.reasoning.append("Daily exit limit reached")
                return result

            # Evaluate individual strategies
            strategy_results = await self._evaluate_strategies(
                symbol, entry_price, current_price, position_data, market_data, analysis_result, strategy_result
            )

            # Coordinate and prioritize exits
            coordinated_result = await self._coordinate_exits(strategy_results, symbol, position_data)

            # Apply risk assessment
            risk_assessment = await self._assess_exit_risk(coordinated_result, position_data, analysis_result)

            # Calculate performance impact
            performance_impact = await self._calculate_performance_impact(coordinated_result, position_data)

            # Make final decision
            final_result = await self._make_exit_decision(coordinated_result, risk_assessment, performance_impact)

            # Update performance tracking
            if self.config.track_exit_performance:
                await self._update_performance_tracking(final_result, strategy_results)

            return final_result

        except Exception as e:
            self.logger.error(f"Error in exit orchestration for {symbol}: {e}")
            return UnifiedExitResult(should_exit=False)

    async def _evaluate_strategies(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        position_data: Optional[Dict[str, Any]] = None,
        market_data: Optional[MarketData] = None,
        analysis_result: Optional[AnalysisResult] = None,
        strategy_result: Optional[StrategyResult] = None
    ) -> Dict[str, Any]:
        """Evaluate all individual exit strategies"""
        try:
            results = {}

            # Exit strategy evaluation
            if self.config.enable_exit_strategies:
                exit_result = await self.exit_strategy_manager.evaluate_exit_strategies(
                    symbol, market_data, analysis_result, strategy_result, position_data
                )
                results["exit_strategy"] = {
                    "result": exit_result,
                    "weight": self.adaptive_weights["exit_strategy"],
                    "priority": self._get_strategy_priority("exit_strategy", exit_result)
                }

            # Take profit evaluation
            if self.config.enable_take_profit:
                tp_result = await self.take_profit_manager.evaluate_take_profit(
                    symbol, entry_price, current_price, position_data, market_data, analysis_result
                )
                results["take_profit"] = {
                    "result": tp_result,
                    "weight": self.adaptive_weights["take_profit"],
                    "priority": self._get_strategy_priority("take_profit", tp_result)
                }

            # Trailing stop evaluation
            if self.config.enable_trailing_stop:
                ts_result = await self.trailing_stop_manager.evaluate_trailing_stop(
                    symbol, entry_price, current_price, position_data, market_data, analysis_result
                )
                results["trailing_stop"] = {
                    "result": ts_result,
                    "weight": self.adaptive_weights["trailing_stop"],
                    "priority": self._get_strategy_priority("trailing_stop", ts_result)
                }

            # Time-based evaluation
            if self.config.enable_time_based:
                tb_result = await self.time_based_manager.evaluate_time_based_exit(
                    symbol, entry_price, current_price, position_data, market_data, analysis_result
                )
                results["time_based"] = {
                    "result": tb_result,
                    "weight": self.adaptive_weights["time_based"],
                    "priority": self._get_strategy_priority("time_based", tb_result)
                }

            return results

        except Exception as e:
            self.logger.error(f"Error evaluating strategies for {symbol}: {e}")
            return {}

    async def _coordinate_exits(
        self,
        strategy_results: Dict[str, Any],
        symbol: str,
        position_data: Optional[Dict[str, Any]] = None
    ) -> UnifiedExitResult:
        """Coordinate multiple exit strategies and resolve conflicts"""
        try:
            result = UnifiedExitResult()
            exit_levels = []
            total_quantity = 0.0
            max_priority = ExitPriority.LOW
            reasoning = []

            # Sort strategies by priority
            sorted_strategies = sorted(
                strategy_results.items(),
                key=lambda x: x[1]["priority"].value,
                reverse=True
            )

            # Process strategies in priority order
            for strategy_name, strategy_info in sorted_strategies:
                strategy_result = strategy_info["result"]
                weight = strategy_info["weight"]

                if hasattr(strategy_result, 'should_exit') and strategy_result.should_exit:
                    # Convert strategy-specific results to unified format
                    levels = await self._convert_strategy_result(strategy_name, strategy_result, position_data)
                    exit_levels.extend(levels)

                    # Track maximum priority
                    if strategy_info["priority"].value > max_priority.value:
                        max_priority = strategy_info["priority"]
                        result.primary_strategy = strategy_name

                    # Add reasoning
                    if hasattr(strategy_result, 'reasoning'):
                        reasoning.extend(strategy_result.reasoning)

            # Apply conflict resolution
            if len(exit_levels) > 1:
                exit_levels = await self._resolve_exit_conflicts(exit_levels, position_data)

            # Calculate total exit quantity
            remaining_quantity = position_data.get('remaining_quantity', 1.0) if position_data else 1.0
            for level in exit_levels:
                actual_quantity = min(level["quantity"], remaining_quantity - total_quantity)
                total_quantity += actual_quantity

            # Determine if we should exit
            result.should_exit = total_quantity > 0
            result.exit_levels = exit_levels
            result.total_exit_quantity = total_quantity
            result.priority = max_priority
            result.reasoning = reasoning
            result.confidence = self._calculate_unified_confidence(strategy_results)

            return result

        except Exception as e:
            self.logger.error(f"Error coordinating exits for {symbol}: {e}")
            return UnifiedExitResult()

    async def _convert_strategy_result(
        self,
        strategy_name: str,
        strategy_result: Any,
        position_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Convert strategy-specific results to unified format"""
        try:
            levels = []

            if strategy_name == "exit_strategy" and hasattr(strategy_result, 'exit_levels'):
                for level in strategy_result.exit_levels:
                    levels.append({
                        "strategy": strategy_name,
                        "exit_type": level.exit_type.value,
                        "level": level.level,
                        "quantity": level.quantity,
                        "signal_strength": level.signal_strength.value,
                        "metadata": level.metadata
                    })

            elif strategy_name == "take_profit" and hasattr(strategy_result, 'exit_levels'):
                for level in strategy_result.exit_levels:
                    levels.append({
                        "strategy": strategy_name,
                        "exit_type": "take_profit",
                        "level": level.level,
                        "quantity": level.quantity,
                        "signal_strength": 0.7,  # Default signal strength
                        "metadata": level.metadata
                    })

            elif strategy_name == "trailing_stop" and hasattr(strategy_result, 'exit_level'):
                if strategy_result.exit_level:
                    levels.append({
                        "strategy": strategy_name,
                        "exit_type": strategy_result.exit_level.exit_type.value,
                        "level": strategy_result.exit_level.level,
                        "quantity": strategy_result.exit_level.quantity,
                        "signal_strength": strategy_result.exit_level.signal_strength.value,
                        "metadata": strategy_result.exit_level.metadata
                    })

            elif strategy_name == "time_based" and hasattr(strategy_result, 'exit_levels'):
                for level in strategy_result.exit_levels:
                    levels.append({
                        "strategy": strategy_name,
                        "exit_type": level.exit_type.value,
                        "level": level.level,
                        "quantity": level.quantity,
                        "signal_strength": level.signal_strength.value,
                        "metadata": level.metadata
                    })

            return levels

        except Exception as e:
            self.logger.error(f"Error converting strategy result for {strategy_name}: {e}")
            return []

    async def _resolve_exit_conflicts(
        self,
        exit_levels: List[Dict[str, Any]],
        position_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Resolve conflicts between multiple exit levels"""
        try:
            if len(exit_levels) <= 1:
                return exit_levels

            # Apply conflict resolution strategy
            if self.config.exit_conflict_resolution == "highest_priority":
                # Keep only highest priority exits
                max_priority = max(level["signal_strength"] for level in exit_levels)
                exit_levels = [level for level in exit_levels if level["signal_strength"] >= max_priority * 0.8]

            elif self.config.exit_conflict_resolution == "consolidate":
                # Consolidate similar exit levels
                consolidated = []
                tolerance = 0.01  # 1% price tolerance

                for level in exit_levels:
                    # Check if this level is close to existing ones
                    close_existing = None
                    for existing in consolidated:
                        if abs(level["level"] - existing["level"]) / level["level"] < tolerance:
                            close_existing = existing
                            break

                    if close_existing:
                        # Merge quantities
                        close_existing["quantity"] = min(1.0, close_existing["quantity"] + level["quantity"] * 0.5)
                    else:
                        consolidated.append(level)

                exit_levels = consolidated

            # Limit concurrent exits
            if len(exit_levels) > self.config.max_concurrent_exits:
                # Sort by signal strength and keep top ones
                exit_levels.sort(key=lambda x: x["signal_strength"], reverse=True)
                exit_levels = exit_levels[:self.config.max_concurrent_exits]

            return exit_levels

        except Exception as e:
            self.logger.error(f"Error resolving exit conflicts: {e}")
            return exit_levels

    async def _assess_exit_risk(
        self,
        exit_result: UnifiedExitResult,
        position_data: Optional[Dict[str, Any]] = None,
        analysis_result: Optional[AnalysisResult] = None
    ) -> Dict[str, Any]:
        """Assess risk of proposed exit"""
        try:
            risk_assessment = {
                "overall_risk": "low",
                "market_risk": 0.0,
                "liquidity_risk": 0.0,
                "timing_risk": 0.0,
                "volatility_risk": 0.0,
                "recommendations": []
            }

            if not exit_result.should_exit:
                return risk_assessment

            # Market risk assessment
            if analysis_result:
                market_risk = 1.0 - analysis_result.confidence
                risk_assessment["market_risk"] = market_risk

                # Volatility risk
                volatility = analysis_result.technical_indicators.get('ATR', 0)
                if volatility > 0:
                    risk_assessment["volatility_risk"] = min(volatility * 100, 1.0)

            # Liquidity risk (simplified)
            position_size = position_data.get('quantity', 1.0) if position_data else 1.0
            exit_quantity = exit_result.total_exit_quantity
            if position_size > 0:
                size_ratio = exit_quantity / position_size
                risk_assessment["liquidity_risk"] = min(size_ratio, 1.0)

            # Timing risk
            if position_data:
                position_age = position_data.get('age_seconds', 0)
                if position_age < 300:  # Less than 5 minutes
                    risk_assessment["timing_risk"] = 0.8  # High risk for early exits
                elif position_age > 7200:  # More than 2 hours
                    risk_assessment["timing_risk"] = 0.2  # Low risk for mature positions

            # Overall risk calculation
            total_risk = (
                risk_assessment["market_risk"] * 0.4 +
                risk_assessment["liquidity_risk"] * 0.3 +
                risk_assessment["timing_risk"] * 0.2 +
                risk_assessment["volatility_risk"] * 0.1
            )

            if total_risk > 0.7:
                risk_assessment["overall_risk"] = "high"
                risk_assessment["recommendations"].append("Consider reducing exit quantity")
            elif total_risk > 0.4:
                risk_assessment["overall_risk"] = "medium"
            else:
                risk_assessment["overall_risk"] = "low"

            return risk_assessment

        except Exception as e:
            self.logger.error(f"Error assessing exit risk: {e}")
            return {"overall_risk": "unknown", "error": str(e)}

    async def _calculate_performance_impact(
        self,
        exit_result: UnifiedExitResult,
        position_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calculate expected performance impact of exit"""
        try:
            impact = {
                "expected_pnl": 0.0,
                "performance_score": 0.0,
                "opportunity_cost": 0.0,
                "risk_adjusted_return": 0.0
            }

            if not exit_result.should_exit or not position_data:
                return impact

            entry_price = position_data.get('entry_price', 0)
            current_price = position_data.get('current_price', 0)
            position_size = position_data.get('quantity', 1.0)

            if entry_price > 0 and current_price > 0:
                # Calculate expected PnL
                exit_quantity = exit_result.total_exit_quantity * position_size
                expected_pnl = exit_quantity * (current_price - entry_price)
                impact["expected_pnl"] = expected_pnl

                # Performance score (simplified)
                profit_pct = (current_price - entry_price) / entry_price
                impact["performance_score"] = min(profit_pct * 100, 100)  # Cap at 100%

                # Opportunity cost (estimated future potential)
                potential_future_gain = position_size * (current_price * 0.02)  # Assume 2% more potential
                opportunity_cost = potential_future_gain * (1 - exit_result.total_exit_quantity)
                impact["opportunity_cost"] = opportunity_cost

                # Risk-adjusted return
                risk_score = position_data.get('risk_score', 0.5)
                impact["risk_adjusted_return"] = expected_pnl * (1 - risk_score)

            return impact

        except Exception as e:
            self.logger.error(f"Error calculating performance impact: {e}")
            return {"error": str(e)}

    async def _make_exit_decision(
        self,
        exit_result: UnifiedExitResult,
        risk_assessment: Dict[str, Any],
        performance_impact: Dict[str, Any]
    ) -> UnifiedExitResult:
        """Make final exit decision based on all evaluations"""
        try:
            # Check if we should modify the exit based on risk assessment
            if risk_assessment["overall_risk"] == "high":
                # Reduce exit quantity for high-risk exits
                reduction_factor = 0.5
                exit_result.total_exit_quantity *= reduction_factor
                exit_result.reasoning.append(f"Reduced exit quantity due to high risk: {risk_assessment['overall_risk']}")
                exit_result.metadata["risk_reduction"] = True

            # Check performance impact
            if performance_impact.get("expected_pnl", 0) < 0:
                # Negative expected PnL - consider not exiting
                if exit_result.confidence < 0.7:
                    exit_result.should_exit = False
                    exit_result.reasoning.append("Cancelled exit due to negative expected PnL and low confidence")
                    exit_result.metadata["cancelled_negative_pnl"] = True

            # Final confidence adjustment
            if exit_result.should_exit:
                # Boost confidence for high-impact exits
                expected_pnl = performance_impact.get("expected_pnl", 0)
                if expected_pnl > 0:
                    pnl_boost = min(expected_pnl / 100, 0.2)  # Up to 20% boost
                    exit_result.confidence = min(exit_result.confidence + pnl_boost, 1.0)

                exit_result.risk_assessment = risk_assessment
                exit_result.performance_impact = performance_impact

            return exit_result

        except Exception as e:
            self.logger.error(f"Error making exit decision: {e}")
            return exit_result

    def _get_strategy_priority(self, strategy_name: str, strategy_result: Any) -> ExitPriority:
        """Get priority level for a strategy"""
        try:
            if hasattr(strategy_result, 'exit_signal'):
                signal = strategy_result.exit_signal
                if signal.value >= 0.8:
                    return ExitPriority.EMERGENCY
                elif signal.value >= 0.6:
                    return ExitPriority.CRITICAL
                elif signal.value >= 0.3:
                    return ExitPriority.HIGH
                else:
                    return ExitPriority.MEDIUM
            else:
                return ExitPriority.MEDIUM

        except Exception as e:
            self.logger.error(f"Error getting strategy priority: {e}")
            return ExitPriority.LOW

    def _calculate_unified_confidence(self, strategy_results: Dict[str, Any]) -> float:
        """Calculate unified confidence from all strategy results"""
        try:
            if not strategy_results:
                return 0.0

            total_weight = 0.0
            weighted_confidence = 0.0

            for strategy_name, strategy_info in strategy_results.items():
                weight = strategy_info["weight"]
                result = strategy_info["result"]

                # Get confidence from strategy result
                if hasattr(result, 'confidence'):
                    confidence = result.confidence
                elif hasattr(result, 'exit_signal'):
                    confidence = result.exit_signal.value
                else:
                    confidence = 0.5  # Default confidence

                weighted_confidence += confidence * weight
                total_weight += weight

            if total_weight > 0:
                return weighted_confidence / total_weight
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating unified confidence: {e}")
            return 0.0

    def _check_exit_limits(self, symbol: str) -> bool:
        """Check if exit limits are exceeded"""
        try:
            # Daily exit limit
            today = datetime.now().strftime("%Y-%m-%d")
            daily_count = self.daily_exit_counts.get(today, 0)

            if daily_count >= self.config.max_daily_exits:
                return False

            # Position exit limit
            position_exits = self.active_positions.get(symbol, {}).get("exit_count", 0)
            if position_exits >= self.config.max_position_exits:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking exit limits: {e}")
            return True  # Allow exit if we can't determine limits

    async def _update_performance_tracking(
        self,
        final_result: UnifiedExitResult,
        strategy_results: Dict[str, Any]
    ) -> None:
        """Update performance tracking with exit results"""
        try:
            # Record exit in history
            exit_record = {
                "timestamp": datetime.now(),
                "primary_strategy": final_result.primary_strategy,
                "should_exit": final_result.should_exit,
                "total_exit_quantity": final_result.total_exit_quantity,
                "confidence": final_result.confidence,
                "priority": final_result.priority.value,
                "strategy_results": {k: v["result"].__dict__ if hasattr(v["result"], '__dict__') else str(v["result"])
                                   for k, v in strategy_results.items()}
            }
            self.exit_history.append(exit_record)

            # Update daily counts
            today = datetime.now().strftime("%Y-%m-%d")
            self.daily_exit_counts[today] = self.daily_exit_counts.get(today, 0) + (1 if final_result.should_exit else 0)

            # Keep history limited
            if len(self.exit_history) > 1000:
                self.exit_history = self.exit_history[-1000:]

        except Exception as e:
            self.logger.error(f"Error updating performance tracking: {e}")

    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get current orchestrator status"""
        return {
            "active_positions": len(self.active_positions),
            "exit_history_count": len(self.exit_history),
            "daily_exit_counts": self.daily_exit_counts,
            "adaptive_weights": self.adaptive_weights,
            "config": self.config.__dict__
        }

    def update_config(self, new_config: ExitOrchestratorConfig) -> None:
        """Update orchestrator configuration"""
        self.config = new_config
        self.logger.info("Exit orchestrator configuration updated")