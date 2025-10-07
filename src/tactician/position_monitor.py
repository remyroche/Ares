# src/tactician/position_monitor.py
from ...utils.logger import system_logger
from ...core.decorators import handles_errors

"""
Position Monitor for real-time position monitoring and confidence assessment.

This module provides continuous monitoring of open positions with confidence score
re-assessment and position decision logic every 10 seconds, using the existing
PositionDivisionStrategy for consistency.
"""
import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .enhanced_order_manager import EnhancedOrderManager
from .position_division_strategy import PositionDivisionStrategy
from ...utils.confidence import normalize_dual_confidence
from ...core.exceptions import (
    error,
    failed,
    initialization_error,
    invalid,
    missing,
    warning,
)
import json
import logging
import time

class PositionAction(Enum):
    """Enum for position actions."""

    STAY = "stay"
    EXIT = "exit"
    SCALE_UP = "scale_up"  # Restored for enhanced exit strategy
    SCALE_DOWN = "scale_down"  # Restored for enhanced exit strategy
    HEDGE = "hedge"
    TAKE_PROFIT = "take_profit"  # Restored for enhanced exit strategy
    STOP_LOSS = "stop_loss"  # Restored for enhanced exit strategy
    FULL_CLOSE = "full_close"
    PARTIAL_PROFIT = "partial_profit"  # New: Take partial profit
    TRAILING_STOP = "trailing_stop"  # New: Update trailing stop

@dataclass
class PositionAssessment:
    """Position assessment data structure."""

    position_id: str
    symbol: str
    side: str  # "LONG" or "SHORT"
    current_quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    analyst_confidence: float
    tactician_confidence: float
    combined_confidence: float
    position_action: PositionAction
    action_reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PositionAlert:
    """Position alert data structure."""

    alert_id: str
    position_id: str
    alert_type: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class PositionMonitor:
    """
    Real-time position monitor with confidence assessment and decision logic.

    Features:
    - Continuous position monitoring
    - Confidence score re-assessment
    - Position action recommendations
    - Alert generation for critical conditions
    - Integration with PositionDivisionStrategy
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the position monitor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("PositionMonitor")

        # Configuration
        self.monitor_config = config.get("position_monitor", {})
        self.monitoring_interval = self.monitor_config.get("monitoring_interval", 10)  # seconds
        
        # Enhanced exit strategy configuration with optimization support
        self.max_position_age = 10800  # 3 hours (will be optimized)
        
        # Load optimized parameters if available
        self.optimized_parameters = self._load_optimized_parameters()
        
        # Confidence-based exit thresholds (optimized)
        self.confidence_thresholds = self._get_optimized_confidence_thresholds()
        
        # PnL-based exit thresholds (optimized)
        self.pnl_thresholds = self._get_optimized_pnl_thresholds()
        
        # Profit-taking configuration (optimized)
        self.profit_taking_config = self._get_optimized_profit_taking_config()
        
        # Additional optimized parameters
        self.stop_loss_config = self._get_optimized_stop_loss_config()
        self.time_based_config = self._get_optimized_time_based_config()
        self.trailing_stop_config = self._get_optimized_trailing_stop_config()
        self.regime_aware_config = self._get_optimized_regime_aware_config()

        # Trailing-stop contextual cache configuration
        self.context_cache_size = max(
            1,
            int(self.trailing_stop_config.get("context_window", 5))
        )

        # Component managers
        self.order_manager: Optional[EnhancedOrderManager] = None
        self.position_strategy: Optional[PositionDivisionStrategy] = None

        # Monitoring state
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.position_assessments: List[PositionAssessment] = []
        self.position_alerts: List[PositionAlert] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="position monitor initialization"
    )
    async def initialize(self) -> bool:
        """
        Initialize the position monitor.

        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing Position Monitor...")

            # Initialize order manager
            self.order_manager = EnhancedOrderManager(self.config)
            await self.order_manager.initialize()

            # Initialize position strategy
            self.position_strategy = PositionDivisionStrategy(self.config)
            await self.position_strategy.initialize()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(invalid("Invalid position monitor configuration"))
                return False

            self.logger.info("✅ Position Monitor initialized successfully")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Position Monitor initialization failed: {e}"))
            return False

    def _validate_configuration(self) -> bool:
        """
        Validate position monitor configuration.

        Returns:
            bool: True if configuration is valid
        """
        try:
            if self.monitoring_interval <= 0:
                self.logger.error(invalid("Monitoring interval must be positive"))
                return False

            if self.max_position_age <= 0:
                self.logger.error(invalid("Max position age must be positive"))
                return False

            # Validate confidence thresholds
            for threshold_name, threshold_value in self.confidence_thresholds.items():
                if not 0 <= threshold_value <= 1:
                    self.logger.error(invalid(f"Confidence threshold '{threshold_name}' must be between 0 and 1"))
                    return False

            # Validate PnL thresholds
            if self.pnl_thresholds["stop_loss"] >= 0:
                self.logger.error(invalid("Stop loss threshold must be negative"))
                return False

            if self.pnl_thresholds["profit_target"] <= 0:
                self.logger.error(invalid("Profit target must be positive"))
                return False

            # Validate profit taking configuration
            if not 0 <= self.profit_taking_config["min_confidence_for_profit"] <= 1:
                self.logger.error(invalid("Minimum confidence for profit must be between 0 and 1"))
                return False

            if not 0 <= self.profit_taking_config["confidence_profit_multiplier"] <= 1:
                self.logger.error(invalid("Confidence profit multiplier must be between 0 and 1"))
                return False

            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Configuration validation failed: {e}"))
            return False

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="position monitoring start"
    )
    async def start_monitoring(self) -> bool:
        """
        Start continuous position monitoring.

        Returns:
            bool: True if monitoring started successfully
        """
        try:
            if self.is_monitoring:
                self.logger.warning(warning("Position monitoring already active"))
                return True

            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

            self.logger.info("✅ Position monitoring started")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Failed to start position monitoring: {e}"))
            return False

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="position monitoring stop"
    )
    async def stop_monitoring(self) -> bool:
        """
        Stop continuous position monitoring.

        Returns:
            bool: True if monitoring stopped successfully
        """
        try:
            if not self.is_monitoring:
                self.logger.warning(warning("Position monitoring not active"))
                return True

            self.is_monitoring = False

            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass

            self.logger.info("✅ Position monitoring stopped")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Failed to stop position monitoring: {e}"))
            return False

    async def _monitoring_loop(self) -> None:
        """
        Main monitoring loop that runs continuously.
        """
        try:
            while self.is_monitoring:
                # Monitor all active positions
                await self._monitor_positions()

                # Auto-refresh step12 configuration if enabled
                await self._auto_refresh_step12_config()

                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)

        except asyncio.CancelledError:
            self.logger.info("Position monitoring loop cancelled")
        except Exception as e:
            self.logger.error(failed(f"❌ Error in monitoring loop: {e}"))

    async def _monitor_positions(self) -> None:
        """
        Monitor all active positions and generate assessments.
        """
        try:
            for position_id, position_data in self.active_positions.items():
                # Get current market data
                current_price = await self._get_current_price(position_data["symbol"])
                if current_price is None:
                    continue

                # Update position data
                position_data["current_price"] = current_price
                position_data["unrealized_pnl"] = self._calculate_unrealized_pnl(position_data)

                # Assess position
                assessment = await self._assess_position(position_id, position_data)
                if assessment:
                    self.position_assessments.append(assessment)

                    # Check for alerts
                    await self._check_position_alerts(assessment)

                    # Log assessment
                    self.logger.info(
                        f"Position {position_id} assessment: {assessment.position_action.value} "
                        f"(confidence: {assessment.combined_confidence:.3f}, PnL: {assessment.unrealized_pnl:.4f})"
                    )

            # Clean up old positions
            await self._cleanup_old_positions()

        except Exception as e:
            self.logger.error(failed(f"❌ Error monitoring positions: {e}"))

    async def _assess_position(self, position_id: str, position_data: Dict[str, Any]) -> Optional[PositionAssessment]:
        """
        Assess a single position and determine recommended action.

        Args:
            position_id: Position ID
            position_data: Position data

        Returns:
            PositionAssessment: Assessment result or None if failed
        """
        try:
            # Get confidence scores from position strategy
            analyst_confidence = position_data.get("analyst_confidence", 0.5)
            tactician_confidence = position_data.get("tactician_confidence", 0.5)

            # Normalize combined confidence
            combined_confidence = normalize_dual_confidence(analyst_confidence, tactician_confidence)

            # Determine position action
            position_action, action_reason = self._determine_position_action(
                position_data, combined_confidence
            )

            return PositionAssessment(
                position_id=position_id,
                symbol=position_data["symbol"],
                side=position_data["side"],
                current_quantity=position_data["quantity"],
                entry_price=position_data["entry_price"],
                current_price=position_data["current_price"],
                unrealized_pnl=position_data["unrealized_pnl"],
                analyst_confidence=analyst_confidence,
                tactician_confidence=tactician_confidence,
                combined_confidence=combined_confidence,
                position_action=position_action,
                action_reason=action_reason
            )

        except Exception as e:
            self.logger.error(failed(f"❌ Error assessing position {position_id}: {e}"))
            return None

    def _determine_position_action(
        self,
        position_data: Dict[str, Any],
        combined_confidence: float
    ) -> tuple[PositionAction, str]:
        """
        Determine recommended position action based on enhanced exit strategy.

        Args:
            position_data: Position data
            combined_confidence: Combined confidence score

        Returns:
            tuple: (PositionAction, reason)
        """
        try:
            unrealized_pnl = position_data["unrealized_pnl"]
            entry_time = position_data.get("entry_time")
            current_time = datetime.now()
            position_id = position_data.get("position_id", "unknown")

            # 1. CRITICAL CONDITIONS - Check first (highest priority)
            if unrealized_pnl <= self.pnl_thresholds["stop_loss"]:
                return PositionAction.STOP_LOSS, f"Critical stop loss: {unrealized_pnl:.4f} <= {self.pnl_thresholds['stop_loss']:.4f}"

            # 2. TIME-BASED EXITS
            if entry_time:
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                position_age = (current_time - entry_time).total_seconds()
                if position_age > self.max_position_age:
                    return PositionAction.FULL_CLOSE, f"Maximum hold time exceeded: {position_age:.0f}s > {self.max_position_age}s"

            # 3. CONFIDENCE-BASED EXITS
            if combined_confidence < self.confidence_thresholds["very_low"]:
                return PositionAction.FULL_CLOSE, f"Very low confidence: {combined_confidence:.3f} < {self.confidence_thresholds['very_low']:.3f}"
            elif combined_confidence < self.confidence_thresholds["low"]:
                return PositionAction.SCALE_DOWN, f"Low confidence: {combined_confidence:.3f} < {self.confidence_thresholds['low']:.3f}"

            # 4. PROFIT-TAKING LOGIC (confidence-based scaling)
            if unrealized_pnl > 0:
                profit_action, profit_reason = self._evaluate_profit_taking(
                    unrealized_pnl, combined_confidence, position_data
                )
                if profit_action != PositionAction.STAY:
                    return profit_action, profit_reason

            # 5. TRAILING STOP LOGIC
            if self.profit_taking_config["trailing_stop_enabled"]:
                trailing_action, trailing_reason = self._evaluate_trailing_stop(
                    position_data, combined_confidence
                )
                if trailing_action != PositionAction.STAY:
                    return trailing_action, trailing_reason

            # 6. CONFIDENCE-BASED POSITION MANAGEMENT
            if combined_confidence >= self.confidence_thresholds["high"]:
                return PositionAction.STAY, f"High confidence: {combined_confidence:.3f} >= {self.confidence_thresholds['high']:.3f}"
            elif combined_confidence >= self.confidence_thresholds["medium"]:
                return PositionAction.STAY, f"Medium confidence: {combined_confidence:.3f} (within acceptable range)"

            # 7. DEFAULT ACTION
            return PositionAction.STAY, f"Position maintained: confidence={combined_confidence:.3f}, pnl={unrealized_pnl:.4f}"

        except Exception as e:
            self.logger.error(failed(f"❌ Error determining position action: {e}"))
            return PositionAction.STAY, f"Error in position assessment: {e}"

    def _evaluate_profit_taking(
        self, 
        unrealized_pnl: float, 
        combined_confidence: float, 
        position_data: Dict[str, Any]
    ) -> tuple[PositionAction, str]:
        """
        Evaluate profit-taking opportunities with confidence-based scaling.

        Args:
            unrealized_pnl: Current unrealized PnL
            combined_confidence: Combined confidence score
            position_data: Position data

        Returns:
            tuple: (PositionAction, reason)
        """
        try:
            # Check if confidence is high enough for profit taking
            if combined_confidence < self.profit_taking_config["min_confidence_for_profit"]:
                return PositionAction.STAY, f"Confidence too low for profit taking: {combined_confidence:.3f} < {self.profit_taking_config['min_confidence_for_profit']:.3f}"

            # Calculate confidence-scaled profit targets
            base_profit_target = self.pnl_thresholds["profit_target"]
            
            if self.profit_taking_config["confidence_scaling"]:
                # Higher confidence = lower profit taking (hold longer for bigger gains)
                confidence_factor = 1.0 - (combined_confidence - 0.5) * self.profit_taking_config["confidence_profit_multiplier"]
                scaled_profit_target = base_profit_target * confidence_factor
            else:
                scaled_profit_target = base_profit_target

            # Check for full profit target
            if unrealized_pnl >= scaled_profit_target:
                return PositionAction.TAKE_PROFIT, f"Profit target reached: {unrealized_pnl:.4f} >= {scaled_profit_target:.4f} (confidence-scaled)"

            # Check for tiered profit taking
            if self.profit_taking_config["tiered_profit_taking"]:
                for i, level in enumerate(self.pnl_thresholds["scaling_levels"]):
                    tier_profit = scaled_profit_target * level
                    if unrealized_pnl >= tier_profit:
                        # Check if we haven't already taken profit at this tier
                        position_id = position_data.get("position_id", "unknown")
                        if not self._has_taken_profit_at_tier(position_id, i):
                            return PositionAction.PARTIAL_PROFIT, f"Tier {i+1} profit: {unrealized_pnl:.4f} >= {tier_profit:.4f} (confidence-scaled)"

            return PositionAction.STAY, f"Profit taking not triggered: {unrealized_pnl:.4f} < {scaled_profit_target:.4f}"

        except Exception as e:
            self.logger.error(failed(f"❌ Error evaluating profit taking: {e}"))
            return PositionAction.STAY, f"Error in profit evaluation: {e}"

    def _evaluate_trailing_stop(
        self,
        position_data: Dict[str, Any],
        combined_confidence: float
    ) -> tuple[PositionAction, str]:
        """
        Evaluate trailing stop conditions.

        Args:
            position_data: Position data
            combined_confidence: Combined confidence score

        Returns:
            tuple: (PositionAction, reason)
        """
        try:
            if not self.trailing_stop_config.get("enabled", True):
                return PositionAction.STAY, "Trailing stop disabled"

            trailing_state = self._ensure_trailing_state(position_data)
            context_cache = self._ensure_position_context_cache(position_data)

            current_price = position_data.get("current_price")
            entry_price = position_data.get("entry_price")
            side = (position_data.get("side") or "").upper()
            if current_price is None or entry_price is None or side not in {"LONG", "SHORT"}:
                return PositionAction.STAY, "Insufficient price data for trailing stop"

            try:
                entry_price = float(entry_price)
                current_price = float(current_price)
            except (TypeError, ValueError):
                return PositionAction.STAY, "Invalid price data for trailing stop"

            def _to_float(value: Any) -> Optional[float]:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            atr_value = None
            for candidate in (position_data.get("atr"), position_data.get("atr_value"), position_data.get("average_true_range")):
                atr_value = _to_float(candidate)
                if atr_value is not None:
                    break

            min_distance_pct = float(self.trailing_stop_config.get("min_distance", 0.01))
            atr_multiplier = float(self.trailing_stop_config.get("atr_multiplier", 1.5))
            base_distance = max(entry_price, current_price) * min_distance_pct
            if atr_value is not None:
                base_distance = max(base_distance, atr_value * atr_multiplier)
            base_distance = max(base_distance, 1e-6 * max(entry_price, current_price))

            momentum_score = self._aggregate_momentum_score(context_cache)
            regime_label = self._latest_regime_label(context_cache)
            last_regime = trailing_state.get("last_regime")

            if (
                trailing_state.get("is_active")
                and regime_label
                and last_regime
                and regime_label != last_regime
            ):
                reset_strength = float(self.trailing_stop_config.get("regime_reset_strength", 0.5))
                trailing_state["peak_price"] = self._apply_regime_reset(
                    trailing_state.get("peak_price"),
                    current_price,
                    side,
                    reset_strength,
                )

            trailing_state["last_regime"] = regime_label

            activation_threshold = float(self.trailing_stop_config.get("confidence_activation", 0.7))
            momentum_threshold = float(self.trailing_stop_config.get("momentum_activation_threshold", 0.0))
            unrealized_pnl = _to_float(position_data.get("unrealized_pnl")) or 0.0

            activated_this_cycle = False
            activation_reason = None
            if not trailing_state.get("is_active"):
                momentum_triggered = (
                    momentum_score is not None
                    and momentum_score < momentum_threshold
                    and unrealized_pnl > 0
                )
                confidence_triggered = unrealized_pnl > 0 and combined_confidence >= activation_threshold

                if momentum_triggered:
                    activation_reason = "negative_momentum"
                elif confidence_triggered:
                    activation_reason = "confidence_activation"

                if momentum_triggered or confidence_triggered:
                    trailing_state["is_active"] = True
                    activated_this_cycle = True
                    trailing_state["activation_reason"] = activation_reason
                    if side == "LONG":
                        trailing_state["peak_price"] = max(entry_price, current_price)
                    else:
                        trailing_state["peak_price"] = min(entry_price, current_price)
                    trailing_state["last_confidence"] = combined_confidence

            if not trailing_state.get("is_active"):
                reason = "Trailing stop inactive: insufficient profit or confidence"
                if momentum_score is not None and momentum_score >= momentum_threshold:
                    reason += f" (momentum={momentum_score:.3f})"
                return PositionAction.STAY, reason

            peak_price = trailing_state.get("peak_price")
            if peak_price is None:
                peak_price = current_price

            if side == "LONG":
                peak_price = max(peak_price, current_price)
            else:
                peak_price = min(peak_price, current_price)
            trailing_state["peak_price"] = peak_price

            last_conf = trailing_state.get("last_confidence")
            drop = 0.0
            if last_conf is not None:
                drop = max(0.0, last_conf - combined_confidence)
            tightening_factor = float(self.trailing_stop_config.get("confidence_tightening_factor", 0.0))
            tightening_floor = float(self.trailing_stop_config.get("confidence_tightening_floor", 0.2))
            if tightening_factor > 0 and drop > 0:
                scale = max(tightening_floor, 1.0 - drop * tightening_factor)
                base_distance *= scale
            trailing_state["last_confidence"] = combined_confidence

            if side == "LONG":
                trailing_level = peak_price - base_distance
                triggered = current_price <= trailing_level
            else:
                trailing_level = peak_price + base_distance
                triggered = current_price >= trailing_level

            trailing_state["trailing_stop"] = trailing_level
            trailing_state["last_update"] = datetime.now().isoformat()

            if triggered:
                trailing_state["is_active"] = False
                trailing_state["activation_reason"] = None
                action_reason = (
                    f"Trailing stop hit at {trailing_level:.4f} "
                    f"(current {current_price:.4f})"
                )
                return PositionAction.FULL_CLOSE, action_reason

            reason_parts = [
                "Trailing stop activated" if activated_this_cycle else "Trailing stop updated",
                f"level={trailing_level:.4f}",
                f"peak={peak_price:.4f}",
            ]
            if momentum_score is not None:
                reason_parts.append(f"momentum={momentum_score:.3f}")
            if regime_label:
                reason_parts.append(f"regime={regime_label}")
            if activation_reason:
                reason_parts.append(f"trigger={activation_reason}")

            return PositionAction.TRAILING_STOP, "; ".join(reason_parts)

        except Exception as e:
            self.logger.error(failed(f"❌ Error evaluating trailing stop: {e}"))
            return PositionAction.STAY, f"Error in trailing stop evaluation: {e}"

    def _has_taken_profit_at_tier(self, position_id: str, tier: int) -> bool:
        """
        Check if profit has already been taken at a specific tier.

        Args:
            position_id: Position ID
            tier: Tier level (0-based)

        Returns:
            bool: True if profit already taken at this tier
        """
        try:
            # This would check position history for previous profit taking at this tier
            # For now, return False (no previous profit taking)
            return False

        except Exception as e:
            self.logger.error(failed(f"❌ Error checking profit tier: {e}"))
            return False

    def _load_optimized_parameters(self) -> Optional[Dict[str, Any]]:
        """
        Load optimized parameters from backtesting optimization results.
        
        Returns:
            Dict: Optimized parameters or None if not available
        """
        try:
            # Try to load from existing optimization framework results
            optimization_paths = [
                "results/final_parameters_optimization.json",
                "src/training/steps/backtesting/results/final_parameters_optimization.json",
                "results/exit_strategy_optimization.json",
                "config/optimized_exit_strategy.json"
            ]
            
            for path in optimization_paths:
                if Path(path).exists():
                    with open(path, 'r') as f:
                        optimization_results = json.load(f)
                    
                    # Check for exit_strategy parameters in the results
                    if "exit_strategy" in optimization_results:
                        self.logger.info(f"✅ Loaded exit strategy parameters from: {path}")
                        return self._convert_optimization_results(optimization_results["exit_strategy"])
                    elif "best_parameters" in optimization_results and "exit_strategy" in optimization_results["best_parameters"]:
                        self.logger.info(f"✅ Loaded exit strategy parameters from: {path}")
                        return self._convert_optimization_results(optimization_results["best_parameters"]["exit_strategy"])
            
            # Fallback to default parameters
            self.logger.info("📝 Using default exit strategy parameters (no optimization found)")
            return None
            
        except Exception as e:
            self.logger.error(failed(f"❌ Error loading optimized parameters: {e}"))
            return None

    def _convert_optimization_results(self, exit_strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert optimization results to position monitor format.
        
        Args:
            exit_strategy_params: Raw optimization parameters
            
        Returns:
            Dict: Converted parameters for position monitor
        """
        try:
            converted = {
                "confidence_thresholds": {
                    "very_low": exit_strategy_params.get("confidence_very_low", 0.2),
                    "low": exit_strategy_params.get("confidence_low", 0.4),
                    "medium": exit_strategy_params.get("confidence_medium", 0.6),
                    "high": exit_strategy_params.get("confidence_high", 0.8)
                },
                "profit_taking": {
                    "base_profit_target": exit_strategy_params.get("base_profit_target", 0.04),
                    "min_confidence_for_profit": exit_strategy_params.get("min_confidence_for_profit", 0.6),
                    "confidence_profit_multiplier": exit_strategy_params.get("confidence_profit_multiplier", 0.5),
                    "scaling_levels": [
                        exit_strategy_params.get("profit_tier_1", 0.25),
                        exit_strategy_params.get("profit_tier_2", 0.5),
                        exit_strategy_params.get("profit_tier_3", 0.75)
                    ]
                },
                "stop_loss": {
                    "base_stop_loss": exit_strategy_params.get("base_stop_loss", -0.05),
                    "atr_multiplier": exit_strategy_params.get("atr_multiplier", 1.5),
                    "volatility_adjustment_factor": exit_strategy_params.get("volatility_adjustment_factor", 1.0)
                },
                "time_based": {
                    "max_hold_time": exit_strategy_params.get("max_hold_time", 10800),
                    "min_hold_time": exit_strategy_params.get("min_hold_time", 300),
                    "confidence_time_scaling_factor": exit_strategy_params.get("confidence_time_scaling_factor", 1.0)
                },
                "trailing_stop": {
                    "enabled": exit_strategy_params.get("trailing_enabled", True),
                    "atr_multiplier": exit_strategy_params.get("trailing_atr_multiplier", 1.5),
                    "min_distance": exit_strategy_params.get("trailing_min_distance", 0.01),
                    "confidence_activation": exit_strategy_params.get("trailing_confidence_activation", 0.7),
                    "momentum_activation_threshold": exit_strategy_params.get(
                        "trailing_momentum_activation_threshold", 0.0
                    ),
                    "confidence_tightening_factor": exit_strategy_params.get(
                        "trailing_confidence_tightening", 0.5
                    ),
                    "confidence_tightening_floor": exit_strategy_params.get(
                        "trailing_confidence_tightening_floor", 0.2
                    ),
                    "regime_reset_strength": exit_strategy_params.get(
                        "trailing_regime_reset_strength", 0.5
                    ),
                    "context_window": exit_strategy_params.get("trailing_context_window", 5)
                },
                "regime_aware": {
                    "transition_penalty": exit_strategy_params.get("regime_transition_penalty", 0.1),
                    "regime_specific_scaling": exit_strategy_params.get("regime_specific_scaling", 1.0)
                }
            }
            
            return converted
            
        except Exception as e:
            self.logger.error(failed(f"❌ Error converting optimization results: {e}"))
            return {}

    def _get_optimized_confidence_thresholds(self) -> Dict[str, float]:
        """Get optimized confidence thresholds."""
        if self.optimized_parameters and "confidence_thresholds" in self.optimized_parameters:
            return self.optimized_parameters["confidence_thresholds"]
        
        # Fallback to config or defaults
        return self.monitor_config.get("confidence_thresholds", {
            "very_low": 0.2,
            "low": 0.4,
            "medium": 0.6,
            "high": 0.8
        })

    def _get_optimized_pnl_thresholds(self) -> Dict[str, Any]:
        """Get optimized PnL thresholds."""
        if self.optimized_parameters and "profit_taking" in self.optimized_parameters:
            profit_config = self.optimized_parameters["profit_taking"]
            stop_loss_config = self.optimized_parameters.get("stop_loss", {})
            
            return {
                "stop_loss": stop_loss_config.get("base_stop_loss", -0.05),
                "profit_target": profit_config.get("base_profit_target", 0.04),
                "scaling_levels": profit_config.get("scaling_levels", [0.25, 0.5, 0.75])
            }
        
        # Fallback to config or defaults
        return self.monitor_config.get("pnl_thresholds", {
            "stop_loss": -0.05,
            "profit_target": 0.04,
            "scaling_levels": [0.25, 0.5, 0.75]
        })

    def _get_optimized_profit_taking_config(self) -> Dict[str, Any]:
        """Get optimized profit-taking configuration."""
        if self.optimized_parameters and "profit_taking" in self.optimized_parameters:
            return self.optimized_parameters["profit_taking"]
        
        # Fallback to config or defaults
        return self.monitor_config.get("profit_taking", {
            "confidence_scaling": True,
            "min_confidence_for_profit": 0.6,
            "confidence_profit_multiplier": 0.5,
            "tiered_profit_taking": True,
            "trailing_stop_enabled": True,
            "trailing_stop_atr_multiplier": 1.5
        })

    def _get_optimized_stop_loss_config(self) -> Dict[str, Any]:
        """Get optimized stop-loss configuration."""
        if self.optimized_parameters and "stop_loss" in self.optimized_parameters:
            return self.optimized_parameters["stop_loss"]
        
        # Fallback to config or defaults
        return self.monitor_config.get("stop_loss", {
            "base_stop_loss": -0.05,
            "atr_multiplier": 1.5,
            "volatility_adjustment": True,
            "regime_adjustment": True
        })

    def _get_optimized_time_based_config(self) -> Dict[str, Any]:
        """Get optimized time-based configuration."""
        if self.optimized_parameters and "time_based" in self.optimized_parameters:
            time_config = self.optimized_parameters["time_based"]
            # Update max position age with optimized value
            self.max_position_age = time_config.get("max_hold_time", 10800)
            return time_config
        
        # Fallback to config or defaults
        return self.monitor_config.get("time_based", {
            "max_hold_time": 10800,
            "min_hold_time": 300,
            "confidence_time_scaling": True
        })

    def _get_optimized_trailing_stop_config(self) -> Dict[str, Any]:
        """Get optimized trailing stop configuration."""
        defaults = {
            "enabled": True,
            "atr_multiplier": 1.5,
            "min_distance": 0.01,
            "confidence_activation": 0.7,
            "momentum_activation_threshold": 0.0,
            "confidence_tightening_factor": 0.5,
            "confidence_tightening_floor": 0.2,
            "regime_reset_strength": 0.5,
            "context_window": 5,
        }

        if self.optimized_parameters and "trailing_stop" in self.optimized_parameters:
            return {**defaults, **self.optimized_parameters["trailing_stop"]}

        # Fallback to config or defaults
        config = self.monitor_config.get("trailing_stop", {})
        return {**defaults, **config}

    def _get_optimized_regime_aware_config(self) -> Dict[str, Any]:
        """Get optimized regime-aware configuration."""
        if self.optimized_parameters and "regime_aware" in self.optimized_parameters:
            return self.optimized_parameters["regime_aware"]
        
        # Fallback to config or defaults
        return self.monitor_config.get("regime_aware", {
            "enabled": True,
            "regime_specific_params": True,
            "transition_penalty": 0.1
        })

    def refresh_optimized_parameters(self, optimization_results: Dict[str, Any]) -> None:
        """
        Refresh optimized parameters from new optimization results.
        
        Args:
            optimization_results: New optimization results
        """
        try:
            if "best_parameters" in optimization_results:
                self.optimized_parameters = optimization_results["best_parameters"]
                
                # Update all configurations
                self.confidence_thresholds = self._get_optimized_confidence_thresholds()
                self.pnl_thresholds = self._get_optimized_pnl_thresholds()
                self.profit_taking_config = self._get_optimized_profit_taking_config()
                self.stop_loss_config = self._get_optimized_stop_loss_config()
                self.time_based_config = self._get_optimized_time_based_config()
                self.trailing_stop_config = self._get_optimized_trailing_stop_config()
                self.regime_aware_config = self._get_optimized_regime_aware_config()
                
                self.logger.info("✅ Optimized parameters refreshed successfully")
                self.logger.info(f"📊 New confidence thresholds: {self.confidence_thresholds}")
                self.logger.info(f"💰 New profit targets: {self.pnl_thresholds['profit_target']:.1%}")
                self.logger.info(f"🛡️ New stop loss: {self.pnl_thresholds['stop_loss']:.1%}")
                
            else:
                self.logger.warning("⚠️ No 'best_parameters' found in optimization results")
                
        except Exception as e:
            self.logger.error(failed(f"❌ Error refreshing optimized parameters: {e}"))

    def get_optimization_status(self) -> Dict[str, Any]:
        """
        Get current optimization status and parameter information.
        
        Returns:
            Dict: Optimization status information
        """
        try:
            status = {
                "optimization_loaded": self.optimized_parameters is not None,
                "confidence_thresholds": self.confidence_thresholds,
                "pnl_thresholds": self.pnl_thresholds,
                "profit_taking_config": self.profit_taking_config,
                "stop_loss_config": self.stop_loss_config,
                "time_based_config": self.time_based_config,
                "trailing_stop_config": self.trailing_stop_config,
                "regime_aware_config": self.regime_aware_config,
                "max_position_age": self.max_position_age
            }
            
            return status
            
        except Exception as e:
            self.logger.error(failed(f"❌ Error getting optimization status: {e}"))
            return {"error": str(e)}

    def _calculate_unrealized_pnl(self, position_data: Dict[str, Any]) -> float:
        """
        Calculate unrealized PnL for a position.

        Args:
            position_data: Position data

        Returns:
            float: Unrealized PnL
        """
        try:
            entry_price = position_data["entry_price"]
            current_price = position_data["current_price"]
            quantity = position_data["quantity"]
            side = position_data["side"]

            if side.upper() == "LONG":
                return (current_price - entry_price) * quantity
            elif side.upper() == "SHORT":
                return (entry_price - current_price) * quantity
            else:
                return 0.0

        except Exception as e:
            self.logger.error(failed(f"❌ Error calculating unrealized PnL: {e}"))
            return 0.0

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            float: Current price or None if failed
        """
        try:
            # Attempt to use an injected price provider callable from config
            price_provider = self.monitor_config.get("price_provider")
            if callable(price_provider):
                return float(await price_provider(symbol)) if asyncio.iscoroutinefunction(price_provider) else float(price_provider(symbol))

            # Attempt exchange client from config
            exchange_client = self.monitor_config.get("exchange_client")
            if exchange_client is not None:
                # Expecting a method get_current_price(symbol) possibly async
                if hasattr(exchange_client, "get_current_price"):
                    func = getattr(exchange_client, "get_current_price")
                    return float(await func(symbol)) if asyncio.iscoroutinefunction(func) else float(func(symbol))

            self.logger.error(missing("No price provider configured for PositionMonitor"))
            return None

        except Exception as e:
            self.logger.error(failed(f"❌ Error getting current price for {symbol}: {e}"))
            return None

    async def _check_position_alerts(self, assessment: PositionAssessment) -> None:
        """
        Check for conditions that require alerts.

        Args:
            assessment: Position assessment
        """
        try:
            # Check for critical PnL
            if assessment.unrealized_pnl <= -0.1:  # -10%
                await self._create_alert(
                    assessment.position_id,
                    "critical_pnl",
                    "critical",
                    f"Critical PnL: {assessment.unrealized_pnl:.4f}"
                )

            # Check for very low confidence
            if assessment.combined_confidence < 0.2:
                await self._create_alert(
                    assessment.position_id,
                    "low_confidence",
                    "high",
                    f"Very low confidence: {assessment.combined_confidence:.3f}"
                )

            # Check for position action changes
            if assessment.position_action in [PositionAction.STOP_LOSS, PositionAction.FULL_CLOSE]:
                await self._create_alert(
                    assessment.position_id,
                    "position_action",
                    "medium",
                    f"Position action: {assessment.position_action.value} - {assessment.action_reason}"
                )

        except Exception as e:
            self.logger.error(failed(f"❌ Error checking position alerts: {e}"))

    async def _create_alert(
        self,
        position_id: str,
        alert_type: str,
        severity: str,
        message: str
    ) -> None:
        """
        Create a position alert.

        Args:
            position_id: Position ID
            alert_type: Type of alert
            severity: Alert severity
            message: Alert message
        """
        try:
            alert = PositionAlert(
                alert_id=f"alert_{len(self.position_alerts) + 1}",
                position_id=position_id,
                alert_type=alert_type,
                severity=severity,
                message=message
            )

            self.position_alerts.append(alert)
            self.logger.warning(f"Position Alert [{severity.upper()}]: {message}")

        except Exception as e:
            self.logger.error(failed(f"❌ Error creating alert: {e}"))

    async def _cleanup_old_positions(self) -> None:
        """
        Clean up old positions that are no longer active.
        """
        try:
            current_time = datetime.now()
            positions_to_remove = []

            for position_id, position_data in self.active_positions.items():
                entry_time = position_data.get("entry_time")
                if entry_time:
                    if isinstance(entry_time, str):
                        entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                    position_age = (current_time - entry_time).total_seconds()

                    if position_age > self.max_position_age * 2:  # 2x max age
                        positions_to_remove.append(position_id)

            for position_id in positions_to_remove:
                del self.active_positions[position_id]
                self.logger.info(f"Removed old position: {position_id}")

        except Exception as e:
            self.logger.error(failed(f"❌ Error cleaning up old positions: {e}"))

    async def _auto_refresh_step12_config(self) -> None:
        """
        Automatically refresh step12 configuration and confidence thresholds.
        This method is called periodically to check for new step12 results.
        """
        try:
            # Check if auto-refresh is enabled
            step12_config = self.config.get("step12_confidence_optimization", {})
            auto_refresh = step12_config.get("auto_refresh", True)
            
            if not auto_refresh:
                return
            
            # Check if we need to refresh (based on interval)
            current_time = datetime.now()
            if hasattr(self, '_last_step12_refresh'):
                time_since_refresh = (current_time - self._last_step12_refresh).total_seconds()
                refresh_interval = step12_config.get("refresh_interval", 300)  # 5 minutes default
                
                if time_since_refresh < refresh_interval:
                    return
            
            # Try to load updated step12 configuration
            updated_config = self._load_updated_step12_config()
            if updated_config:
                # Update confidence thresholds
                position_monitor_config = updated_config.get("position_monitor", {})
                
                self.high_confidence_threshold = position_monitor_config.get("high_confidence_threshold", self.high_confidence_threshold)
                self.low_confidence_threshold = position_monitor_config.get("low_confidence_threshold", self.low_confidence_threshold)
                self.very_low_confidence_threshold = position_monitor_config.get("very_low_confidence_threshold", self.very_low_confidence_threshold)
                
                self._last_step12_refresh = current_time
                self.logger.info("✅ Refreshed step12 confidence thresholds automatically")
                
        except Exception as e:
            self.logger.error(failed(f"❌ Error in step12 auto-refresh: {e}"))

    def _load_updated_step12_config(self) -> Optional[Dict[str, Any]]:
        """
        Load updated step12 configuration from results files.

        Returns:
            Dict: Updated configuration or None if no updates found
        """
        try:
            step12_config = self.config.get("step12_confidence_optimization", {})
            result_paths = step12_config.get("step12_results_paths", [])
            
            for path in result_paths:
                if Path(path).exists():
                    try:
                        with open(path, 'r') as f:
                            import yaml
                    except Exception as e:
                        pass  # TODO: Handle exception
                    except Exception as e:
                        pass  # TODO: Handle exception properly
                           
                        # Check if this is newer than our current config
                        updated_config = yaml.safe_load(f)
                        if "timestamp" in updated_config:
                            config_time = datetime.fromisoformat(updated_config["timestamp"])
                            if hasattr(self, '_last_step12_refresh'):
                                if config_time > self._last_step12_refresh:
                                    return updated_config
                            else:
                                return updated_config
                        else:
                            return updated_config
                            
                    except Exception as e:
                        self.logger.warning(f"Could not load step12 config from {path}: {e}")
                        continue
            
            return None
            
        except Exception as e:
            self.logger.error(failed(f"❌ Error loading updated step12 config: {e}"))
            return None

    def _ensure_position_context_cache(self, position_data: Dict[str, Any]) -> Dict[str, deque]:
        """Ensure the contextual cache structure exists for a position."""
        cache = position_data.get("context_cache")
        if not isinstance(cache, dict):
            cache = {}

        def _normalize_existing(values: Any) -> deque:
            if isinstance(values, deque):
                return deque(list(values)[-self.context_cache_size:], maxlen=self.context_cache_size)
            if isinstance(values, list):
                return deque(values[-self.context_cache_size:], maxlen=self.context_cache_size)
            return deque(maxlen=self.context_cache_size)

        for key in ("analyst_momentum", "tactician_momentum", "analyst_regime", "tactician_regime"):
            cache[key] = _normalize_existing(cache.get(key))

        position_data["context_cache"] = cache
        return cache

    def _append_context_entry(self, cache: Dict[str, deque], key: str, entry: Any) -> None:
        """Append a normalized context entry to the cache."""
        if entry is None:
            return

        normalized: Dict[str, Any] = {}
        if isinstance(entry, dict):
            label = entry.get("label") or entry.get("prediction") or entry.get("state")
            if label is not None:
                normalized["label"] = label
            value = entry.get("value")
            if value is None:
                value = entry.get("score")
            if value is None and label is not None:
                converted = self._momentum_label_to_score(str(label))
                if converted is not None:
                    value = converted
            if value is not None:
                try:
                    normalized["value"] = float(value)
                except (TypeError, ValueError):
                    pass
            confidence = entry.get("confidence") or entry.get("probability")
            if confidence is None and isinstance(entry.get("probabilities"), dict) and label in entry["probabilities"]:
                confidence = entry["probabilities"][label]
            if confidence is not None:
                try:
                    normalized["confidence"] = float(confidence)
                except (TypeError, ValueError):
                    pass
        elif isinstance(entry, (int, float)):
            normalized["value"] = float(entry)
        else:
            normalized["label"] = str(entry)

        if not normalized:
            return

        cache.setdefault(key, deque(maxlen=self.context_cache_size)).append(normalized)

    def _ensure_trailing_state(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure trailing state storage is present for a position."""
        state = position_data.get("trailing_state")
        if not isinstance(state, dict):
            state = {}

        if "is_active" not in state:
            state["is_active"] = False
        if "peak_price" not in state:
            state["peak_price"] = position_data.get("entry_price")
        if "trailing_stop" not in state:
            state["trailing_stop"] = None
        if "last_confidence" not in state:
            analyst_conf = position_data.get("analyst_confidence")
            tactician_conf = position_data.get("tactician_confidence")
            if analyst_conf is not None and tactician_conf is not None:
                state["last_confidence"] = normalize_dual_confidence(analyst_conf, tactician_conf)
            else:
                state["last_confidence"] = None
        if "last_regime" not in state:
            state["last_regime"] = None
        if "activation_reason" not in state:
            state["activation_reason"] = None

        position_data["trailing_state"] = state
        return state

    def _aggregate_momentum_score(self, context_cache: Dict[str, deque]) -> Optional[float]:
        """Aggregate the most recent momentum estimate from caches."""
        scores: List[float] = []
        for key in ("analyst_momentum", "tactician_momentum"):
            series = context_cache.get(key)
            if series and len(series) > 0:
                latest = series[-1]
                if isinstance(latest, dict):
                    value = latest.get("value")
                    if value is None and latest.get("label") is not None:
                        value = self._momentum_label_to_score(str(latest["label"]))
                    if value is not None:
                        try:
                            scores.append(float(value))
                        except (TypeError, ValueError):
                            continue
        if not scores:
            return None
        return sum(scores) / len(scores)

    @staticmethod
    def _momentum_label_to_score(label: str) -> Optional[float]:
        """Convert textual momentum labels to numeric scores."""
        mapping = {
            "bullish": 1.0,
            "bearish": -1.0,
            "positive": 1.0,
            "negative": -1.0,
            "up": 1.0,
            "down": -1.0,
            "neutral": 0.0,
        }
        normalized = label.lower()
        if normalized in mapping:
            return mapping[normalized]
        return None

    def _latest_regime_label(self, context_cache: Dict[str, deque]) -> Optional[str]:
        """Return the latest observed regime label, prioritizing tactician data."""
        for key in ("tactician_regime", "analyst_regime"):
            series = context_cache.get(key)
            if series and len(series) > 0:
                latest = series[-1]
                if isinstance(latest, dict):
                    label = latest.get("label")
                else:
                    label = latest
                if label is not None:
                    return str(label)
        return None

    @staticmethod
    def _apply_regime_reset(
        peak_price: Optional[float],
        current_price: Optional[float],
        side: str,
        strength: float,
    ) -> Optional[float]:
        """Blend the peak price toward the current price on regime change."""
        if peak_price is None or current_price is None:
            return peak_price
        strength = max(0.0, min(1.0, strength))
        if strength == 0:
            return peak_price

        side_upper = (side or "").upper()
        if side_upper == "LONG":
            return current_price + (peak_price - current_price) * (1 - strength)
        if side_upper == "SHORT":
            return current_price - (current_price - peak_price) * (1 - strength)
        return peak_price

    def update_position_context(self, position_id: str, context_update: Dict[str, Any]) -> None:
        """Update cached contextual information for an active position."""
        try:
            position = self.active_positions.get(position_id)
            if not position:
                return

            cache = self._ensure_position_context_cache(position)

            if "analyst_momentum" in context_update:
                self._append_context_entry(cache, "analyst_momentum", context_update.get("analyst_momentum"))
            if "tactician_momentum" in context_update:
                self._append_context_entry(cache, "tactician_momentum", context_update.get("tactician_momentum"))
            if "analyst_regime" in context_update:
                self._append_context_entry(cache, "analyst_regime", context_update.get("analyst_regime"))
            if "tactician_regime" in context_update:
                self._append_context_entry(cache, "tactician_regime", context_update.get("tactician_regime"))

            if context_update.get("analyst_confidence") is not None:
                try:
                    position["analyst_confidence"] = float(context_update["analyst_confidence"])
                except (TypeError, ValueError):
                    pass
            if context_update.get("tactician_confidence") is not None:
                try:
                    position["tactician_confidence"] = float(context_update["tactician_confidence"])
                except (TypeError, ValueError):
                    pass

            self._ensure_trailing_state(position)

        except Exception as e:
            self.logger.error(failed(f"❌ Error updating position context: {e}"))

    def add_position(self, position_data: Dict[str, Any]) -> None:
        """
        Add a position to monitoring.

        Args:
            position_data: Position data
        """
        try:
            position_id = position_data.get("position_id")
            if not position_id:
                self.logger.error(missing("Position ID is required"))
                return

            enriched_position = {**position_data, "position_id": position_id}
            if "entry_time" not in enriched_position:
                enriched_position["entry_time"] = datetime.now().isoformat()

            self._ensure_position_context_cache(enriched_position)
            self._ensure_trailing_state(enriched_position)

            self.active_positions[position_id] = enriched_position
            self.logger.info(f"Added position to monitoring: {position_id}")

        except Exception as e:
            self.logger.error(failed(f"❌ Error adding position: {e}"))

    def remove_position(self, position_id: str) -> None:
        """
        Remove a position from monitoring.

        Args:
            position_id: Position ID to remove
        """
        try:
            if position_id in self.active_positions:
                del self.active_positions[position_id]
                self.logger.info(f"Removed position from monitoring: {position_id}")
            else:
                self.logger.warning(warning(f"Position not found: {position_id}"))

        except Exception as e:
            self.logger.error(failed(f"❌ Error removing position: {e}"))

    def get_active_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active positions.

        Returns:
            Dict[str, Dict[str, Any]]: Active positions
        """
        return self.active_positions.copy()

    def get_position_assessments(self, limit: Optional[int] = None) -> List[PositionAssessment]:
        """
        Get position assessments.

        Args:
            limit: Maximum number of assessments to return

        Returns:
            List[PositionAssessment]: Position assessments
        """
        if limit:
            return self.position_assessments[-limit:]
        return self.position_assessments.copy()

    def get_position_alerts(self, unresolved_only: bool = True) -> List[PositionAlert]:
        """
        Get position alerts.

        Args:
            unresolved_only: Return only unresolved alerts

        Returns:
            List[PositionAlert]: Position alerts
        """
        if unresolved_only:
            return [alert for alert in self.position_alerts if not alert.resolved]
        return self.position_alerts.copy()

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Mark an alert as resolved.

        Args:
            alert_id: Alert ID to resolve

        Returns:
            bool: True if alert was resolved
        """
        try:
            for alert in self.position_alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    self.logger.info(f"Resolved alert: {alert_id}")
                    return True

            self.logger.warning(warning(f"Alert not found: {alert_id}"))
            return False

        except Exception as e:
            self.logger.error(failed(f"❌ Error resolving alert: {e}"))
            return False

    async def cleanup(self) -> None:
        """
        Cleanup resources.
        """
        try:
            self.logger.info("Cleaning up Position Monitor...")

            # Stop monitoring
            await self.stop_monitoring()

            # Cleanup component managers
            if self.order_manager:
                await self.order_manager.cleanup()

            if self.position_strategy:
                await self.position_strategy.cleanup()

            self.logger.info("✅ Position Monitor cleanup completed")

        except Exception as e:
            self.logger.error(failed(f"❌ Position Monitor cleanup failed: {e}"))
