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
import math
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

            force_exit = self.confidence_thresholds.get("force_exit", 0.3)
            force_hold = self.confidence_thresholds.get("force_hold", 0.8)
            if force_exit >= force_hold:
                self.logger.error(invalid("Force exit threshold must be below force hold threshold"))
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

            for pct_name, pct_value in [
                ("tp_exit_percentage", self.profit_taking_config.get("exit_percentage", 1.0)),
                ("sl_exit_percentage", self.stop_loss_config.get("exit_percentage", 1.0))
            ]:
                if not 0 < pct_value <= 1:
                    self.logger.error(invalid(f"{pct_name} must be between 0 and 1"))
                    return False

            for conf_name, conf_value in [
                ("tp_confidence_threshold", self.profit_taking_config.get("confidence_exit_threshold", 0.55)),
                ("sl_confidence_threshold", self.stop_loss_config.get("confidence_exit_threshold", 0.4))
            ]:
                if not 0 <= conf_value <= 1:
                    self.logger.error(invalid(f"{conf_name} must be between 0 and 1"))
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
            position_action, action_reason, action_metadata = self._determine_position_action(
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
                action_reason=action_reason,
                metadata=action_metadata
            )

        except Exception as e:
            self.logger.error(failed(f"❌ Error assessing position {position_id}: {e}"))
            return None

    def _determine_position_action(
        self,
        position_data: Dict[str, Any],
        combined_confidence: float
    ) -> tuple[PositionAction, str, Dict[str, Any]]:
        """
        Determine recommended position action based on enhanced exit strategy.

        Args:
            position_data: Position data
            combined_confidence: Combined confidence score

        Returns:
            tuple: (PositionAction, reason, metadata)
        """
        try:
            unrealized_pnl = position_data["unrealized_pnl"]
            entry_time = position_data.get("entry_time")
            current_time = datetime.now()
            position_id = position_data.get("position_id", "unknown")
            metadata: Dict[str, Any] = {
                "confidence": combined_confidence,
                "position_id": position_id,
            }

            force_exit_threshold = self.confidence_thresholds.get(
                "force_exit",
                self.confidence_thresholds.get("very_low", 0.2)
            )
            force_hold_threshold = self.confidence_thresholds.get(
                "force_hold",
                self.confidence_thresholds.get("high", 0.8)
            )

            dynamic_stop_loss = self._calculate_stop_loss_threshold(position_data)
            metadata.setdefault("dynamic_thresholds", {})["stop_loss"] = dynamic_stop_loss

            # 0. Force exit if confidence is critically low
            if combined_confidence <= force_exit_threshold:
                metadata.update({
                    "exit_percentage": 1.0,
                    "exit_type": "force_exit"
                })
                reason = (
                    f"Confidence {combined_confidence:.3f} below force exit threshold "
                    f"{force_exit_threshold:.3f}"
                )
                return PositionAction.FULL_CLOSE, reason, metadata

            # 1. CRITICAL CONDITIONS - Check with dynamic stop loss
            if unrealized_pnl <= dynamic_stop_loss:
                return self._handle_stop_loss_exit(
                    unrealized_pnl,
                    combined_confidence,
                    dynamic_stop_loss,
                    force_hold_threshold,
                    metadata
                )

            # 2. TIME-BASED EXITS
            if entry_time:
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                position_age = (current_time - entry_time).total_seconds()
                if position_age > self.max_position_age:
                    metadata.update({
                        "exit_percentage": 1.0,
                        "exit_type": "time_limit",
                        "position_age": position_age
                    })
                    return (
                        PositionAction.FULL_CLOSE,
                        f"Maximum hold time exceeded: {position_age:.0f}s > {self.max_position_age}s",
                        metadata
                    )

            # 3. CONFIDENCE-BASED EXITS
            if combined_confidence < self.confidence_thresholds["very_low"]:
                metadata.update({
                    "exit_percentage": 1.0,
                    "exit_type": "confidence_floor"
                })
                return (
                    PositionAction.FULL_CLOSE,
                    f"Very low confidence: {combined_confidence:.3f} < {self.confidence_thresholds['very_low']:.3f}",
                    metadata
                )
            elif combined_confidence < self.confidence_thresholds["low"]:
                metadata.update({
                    "exit_percentage": min(
                        0.5,
                        self.stop_loss_config.get("exit_percentage", 1.0)
                    ),
                    "exit_type": "confidence_reduction"
                })
                return (
                    PositionAction.SCALE_DOWN,
                    f"Low confidence: {combined_confidence:.3f} < {self.confidence_thresholds['low']:.3f}",
                    metadata
                )

            # 4. PROFIT-TAKING LOGIC (confidence-based scaling)
            if unrealized_pnl > 0:
                profit_action, profit_reason, profit_metadata = self._evaluate_profit_taking(
                    unrealized_pnl, combined_confidence, position_data
                )
                if profit_action != PositionAction.STAY:
                    profit_metadata.setdefault("confidence", combined_confidence)
                    profit_metadata.setdefault("position_id", position_id)
                    return profit_action, profit_reason, profit_metadata

            # 5. TRAILING STOP LOGIC
            if self.profit_taking_config["trailing_stop_enabled"]:
                trailing_action, trailing_reason, trailing_metadata = self._evaluate_trailing_stop(
                    position_data, combined_confidence
                )
                if trailing_action != PositionAction.STAY:
                    trailing_metadata.setdefault("confidence", combined_confidence)
                    trailing_metadata.setdefault("position_id", position_id)
                    return trailing_action, trailing_reason, trailing_metadata

            # 6. CONFIDENCE-BASED POSITION MANAGEMENT
            if combined_confidence >= self.confidence_thresholds["high"]:
                metadata.update({
                    "exit_type": "confidence_hold",
                    "force_hold_threshold": force_hold_threshold
                })
                return (
                    PositionAction.STAY,
                    f"High confidence: {combined_confidence:.3f} >= {self.confidence_thresholds['high']:.3f}",
                    metadata
                )
            elif combined_confidence >= self.confidence_thresholds["medium"]:
                metadata.update({
                    "exit_type": "confidence_neutral"
                })
                return (
                    PositionAction.STAY,
                    f"Medium confidence: {combined_confidence:.3f} (within acceptable range)",
                    metadata
                )

            # 7. DEFAULT ACTION
            metadata.update({
                "exit_type": "default",
                "dynamic_thresholds": metadata.get("dynamic_thresholds", {})
            })
            return (
                PositionAction.STAY,
                f"Position maintained: confidence={combined_confidence:.3f}, pnl={unrealized_pnl:.4f}",
                metadata
            )

        except Exception as e:
            self.logger.error(failed(f"❌ Error determining position action: {e}"))
            return PositionAction.STAY, f"Error in position assessment: {e}", {}

    def _extract_atr_value(self, position_data: Dict[str, Any]) -> float:
        """Extract ATR value from position data if available."""
        atr_keys = [
            "atr",
            "atr_14",
            "atr_value",
            "atr_percent",
            "average_true_range"
        ]

        for key in atr_keys:
            atr_value = position_data.get(key)
            if atr_value is None:
                continue
            try:
                return abs(float(atr_value))
            except (TypeError, ValueError):
                continue

        return 0.0

    def _calculate_stop_loss_threshold(self, position_data: Dict[str, Any]) -> float:
        """Calculate dynamic stop loss threshold with optional ATR scaling."""
        base_stop_loss = self.pnl_thresholds.get("stop_loss", -0.05)
        threshold = base_stop_loss

        if self.stop_loss_config.get("use_log_atr_multiplier", False):
            atr_value = self._extract_atr_value(position_data)
            atr_multiplier = max(self.stop_loss_config.get("atr_log_multiplier", 0.0), 0.0)

            if atr_value > 0 and atr_multiplier > 0:
                atr_factor = math.log1p(atr_value * atr_multiplier)
                atr_factor = max(0.5, min(2.5, atr_factor))
                threshold = base_stop_loss * atr_factor

        return threshold

    def _calculate_profit_target_threshold(
        self,
        combined_confidence: float,
        position_data: Dict[str, Any]
    ) -> float:
        """Calculate dynamic profit target with confidence and ATR scaling."""
        base_profit_target = self.pnl_thresholds.get("profit_target", 0.04)
        target = base_profit_target

        if self.profit_taking_config.get("confidence_scaling", True):
            confidence_multiplier = self.profit_taking_config.get("confidence_profit_multiplier", 0.5)
            confidence_factor = 1.0 - (combined_confidence - 0.5) * confidence_multiplier
            confidence_factor = max(0.2, min(2.0, confidence_factor))
            target *= confidence_factor

        if self.profit_taking_config.get("use_log_atr_multiplier", False):
            atr_value = self._extract_atr_value(position_data)
            atr_multiplier = max(self.profit_taking_config.get("atr_log_multiplier", 0.0), 0.0)

            if atr_value > 0 and atr_multiplier > 0:
                atr_factor = math.log1p(atr_value * atr_multiplier)
                atr_factor = max(0.5, min(2.5, atr_factor))
                target *= atr_factor

        return target

    def _handle_stop_loss_exit(
        self,
        unrealized_pnl: float,
        combined_confidence: float,
        dynamic_threshold: float,
        force_hold_threshold: float,
        metadata: Dict[str, Any]
    ) -> tuple[PositionAction, str, Dict[str, Any]]:
        """Handle stop loss exits with confidence-aware scaling."""

        exit_percentage = max(0.0, min(1.0, self.stop_loss_config.get("exit_percentage", 1.0)))
        confidence_threshold = self.stop_loss_config.get("confidence_exit_threshold", 0.4)

        metadata = dict(metadata)
        metadata.update({
            "exit_type": "stop_loss",
            "exit_percentage": exit_percentage,
            "threshold": dynamic_threshold
        })

        if combined_confidence <= confidence_threshold:
            action = PositionAction.FULL_CLOSE if exit_percentage >= 0.99 else PositionAction.STOP_LOSS
            reason = (
                f"Stop loss triggered at {unrealized_pnl:.4f} (threshold {dynamic_threshold:.4f}) "
                f"with low confidence {combined_confidence:.3f}"
            )
            return action, reason, metadata

        if combined_confidence >= force_hold_threshold:
            # High confidence despite drawdown – reduce exit size significantly
            reduced_percentage = max(0.1, exit_percentage * 0.25)
            metadata.update({
                "exit_percentage": reduced_percentage,
                "exit_type": "confidence_buffered_stop"
            })
            reason = (
                f"Stop loss reached but confidence {combined_confidence:.3f} >= force hold "
                f"{force_hold_threshold:.3f}; scaling down position"
            )
            return PositionAction.SCALE_DOWN, reason, metadata

        # Moderate confidence – partial scale down
        partial_percentage = max(0.1, exit_percentage * 0.5)
        metadata.update({
            "exit_percentage": partial_percentage,
            "exit_type": "partial_stop"
        })
        reason = (
            f"Stop loss reached with moderate confidence {combined_confidence:.3f}; "
            "scaling down position"
        )
        return PositionAction.SCALE_DOWN, reason, metadata

    def _evaluate_profit_taking(
        self,
        unrealized_pnl: float,
        combined_confidence: float,
        position_data: Dict[str, Any]
    ) -> tuple[PositionAction, str, Dict[str, Any]]:
        """
        Evaluate profit-taking opportunities with confidence-based scaling.

        Args:
            unrealized_pnl: Current unrealized PnL
            combined_confidence: Combined confidence score
            position_data: Position data

        Returns:
            tuple: (PositionAction, reason, metadata)
        """
        try:
            metadata: Dict[str, Any] = {
                "exit_type": "take_profit",
                "unrealized_pnl": unrealized_pnl
            }

            force_hold_threshold = self.confidence_thresholds.get(
                "force_hold",
                self.confidence_thresholds.get("high", 0.8)
            )

            # Check if confidence is high enough for profit taking
            if combined_confidence < self.profit_taking_config["min_confidence_for_profit"]:
                metadata["exit_type"] = "insufficient_confidence"
                return (
                    PositionAction.STAY,
                    f"Confidence too low for profit taking: {combined_confidence:.3f} < {self.profit_taking_config['min_confidence_for_profit']:.3f}",
                    metadata
                )

            if combined_confidence >= force_hold_threshold:
                metadata["exit_type"] = "force_hold_override"
                return (
                    PositionAction.STAY,
                    f"Confidence {combined_confidence:.3f} >= force hold {force_hold_threshold:.3f}; deferring profit taking",
                    metadata
                )

            # Calculate dynamic profit target
            scaled_profit_target = self._calculate_profit_target_threshold(
                combined_confidence,
                position_data
            )
            metadata["target"] = scaled_profit_target

            confidence_exit_threshold = self.profit_taking_config.get("confidence_exit_threshold", 0.55)
            exit_percentage = max(0.0, min(1.0, self.profit_taking_config.get("exit_percentage", 1.0)))
            metadata["exit_percentage"] = exit_percentage

            # Check for full profit target
            if unrealized_pnl >= scaled_profit_target:
                if combined_confidence <= confidence_exit_threshold:
                    action = PositionAction.TAKE_PROFIT if exit_percentage >= 0.99 else PositionAction.PARTIAL_PROFIT
                    reason = (
                        f"Profit target reached: {unrealized_pnl:.4f} >= {scaled_profit_target:.4f} "
                        f"(confidence threshold {confidence_exit_threshold:.3f})"
                    )
                    return action, reason, metadata

                partial_percentage = max(0.1, exit_percentage * 0.5)
                metadata.update({
                    "exit_percentage": partial_percentage,
                    "exit_type": "confidence_buffered_profit"
                })
                reason = (
                    f"Profit target hit but confidence {combined_confidence:.3f} > {confidence_exit_threshold:.3f}; "
                    "taking partial profit"
                )
                return PositionAction.PARTIAL_PROFIT, reason, metadata

            # Check for tiered profit taking
            if self.profit_taking_config["tiered_profit_taking"]:
                for i, level in enumerate(self.pnl_thresholds["scaling_levels"]):
                    tier_profit = scaled_profit_target * level
                    if unrealized_pnl >= tier_profit:
                        # Check if we haven't already taken profit at this tier
                        position_id = position_data.get("position_id", "unknown")
                        if not self._has_taken_profit_at_tier(position_id, i):
                            tier_percentage = max(0.05, exit_percentage / (i + 1))
                            metadata.update({
                                "exit_percentage": tier_percentage,
                                "exit_type": f"tier_{i+1}_profit",
                                "tier_target": tier_profit
                            })
                            return (
                                PositionAction.PARTIAL_PROFIT,
                                f"Tier {i+1} profit: {unrealized_pnl:.4f} >= {tier_profit:.4f} (confidence-scaled)",
                                metadata
                            )

            metadata["exit_type"] = "target_not_met"
            return (
                PositionAction.STAY,
                f"Profit taking not triggered: {unrealized_pnl:.4f} < {scaled_profit_target:.4f}",
                metadata
            )

        except Exception as e:
            self.logger.error(failed(f"❌ Error evaluating profit taking: {e}"))
            return PositionAction.STAY, f"Error in profit evaluation: {e}", {}

    def _evaluate_trailing_stop(
        self,
        position_data: Dict[str, Any],
        combined_confidence: float
    ) -> tuple[PositionAction, str, Dict[str, Any]]:
        """
        Evaluate trailing stop conditions.

        Args:
            position_data: Position data
            combined_confidence: Combined confidence score

        Returns:
            tuple: (PositionAction, reason)
        """
        try:
            metadata: Dict[str, Any] = {
                "exit_type": "trailing_stop",
                "confidence": combined_confidence
            }

            trailing_config = dict(self.trailing_stop_config or {})
            profit_config = dict(self.profit_taking_config or {})
            metadata["trailing_config"] = trailing_config

            # Ensure trailing stop is enabled in the configurations
            if not profit_config.get("trailing_stop_enabled", False):
                metadata["status"] = "disabled"
                return PositionAction.STAY, "Trailing stop disabled", metadata

            if not trailing_config.get("enabled", True):
                metadata["status"] = "disabled"
                return PositionAction.STAY, "Trailing stop disabled in configuration", metadata

            high_conf_threshold = self.confidence_thresholds.get("high", 0.8)
            confidence_activation = trailing_config.get("confidence_activation", 0.7)

            # Trailing stop is only active when confidence is below the high threshold
            if combined_confidence >= high_conf_threshold:
                metadata["status"] = "confidence_above_high_threshold"
                metadata["high_threshold"] = high_conf_threshold
                return (
                    PositionAction.STAY,
                    "Confidence above high threshold; trailing stop inactive",
                    metadata
                )

            # Require a minimum confidence to avoid activating during panic exits
            if combined_confidence < confidence_activation:
                metadata["status"] = "confidence_below_activation"
                metadata["activation_threshold"] = confidence_activation
                return (
                    PositionAction.STAY,
                    "Confidence below trailing activation threshold",
                    metadata
                )

            entry_price = float(position_data.get("entry_price", 0.0) or 0.0)
            current_price = float(position_data.get("current_price", 0.0) or 0.0)
            quantity = abs(float(position_data.get("quantity", 0.0) or 0.0))
            side = str(position_data.get("side", "LONG")).upper()

            if entry_price <= 0 or current_price <= 0 or quantity <= 0:
                metadata["status"] = "insufficient_market_data"
                return (
                    PositionAction.STAY,
                    "Insufficient data for trailing stop evaluation",
                    metadata
                )

            # Determine the dynamically scaled profit target used for activation
            dynamic_profit_target = self._calculate_profit_target_threshold(
                combined_confidence,
                position_data
            )
            profit_per_unit = dynamic_profit_target / max(quantity, 1e-8)

            if side == "LONG":
                target_price = entry_price + profit_per_unit
                price_above_target = current_price >= target_price
            elif side == "SHORT":
                target_price = entry_price - profit_per_unit
                price_above_target = current_price <= target_price
            else:
                metadata["status"] = "unsupported_side"
                metadata["side"] = side
                return PositionAction.STAY, "Unsupported position side for trailing stop", metadata

            trailing_state = position_data.setdefault(
                "trailing_state",
                {
                    "activated": False,
                    "target_price": target_price,
                    "extreme_price": None,
                    "atr_at_activation": None,
                    "activation_price": None,
                    "activation_time": None
                }
            )

            atr_value = self._extract_atr_value(position_data)

            stored_target_price = trailing_state.get("target_price")
            if stored_target_price is None:
                activation_target_price = target_price
            elif side == "LONG":
                activation_target_price = max(stored_target_price, target_price)
            else:
                activation_target_price = min(stored_target_price, target_price)
            trailing_state["target_price"] = activation_target_price

            if side == "LONG":
                price_above_target = current_price >= activation_target_price
            else:
                price_above_target = current_price <= activation_target_price

            if not trailing_state.get("activated", False):
                if not price_above_target:
                    trailing_state.update({
                        "activated": False,
                        "extreme_price": None,
                        "atr_at_activation": None,
                        "activation_price": None,
                        "activation_time": None
                    })
                    metadata.update({
                        "status": "target_not_reached",
                        "target_price": activation_target_price,
                        "unrealized_target": dynamic_profit_target
                    })
                    return (
                        PositionAction.STAY,
                        "Trailing stop inactive until dynamic target price is exceeded",
                        metadata
                    )

                trailing_state.update({
                    "activated": True,
                    "activation_price": current_price,
                    "activation_time": time.time(),
                    "extreme_price": current_price
                })
                if atr_value is not None:
                    trailing_state["atr_at_activation"] = float(atr_value)

            # Update the best favorable price since the target was reached
            extreme_price = trailing_state.get("extreme_price")
            if extreme_price is None:
                extreme_price = current_price
            if side == "LONG":
                extreme_price = max(extreme_price, current_price)
            else:  # SHORT
                extreme_price = min(extreme_price, current_price)
            trailing_state["extreme_price"] = extreme_price

            if trailing_state.get("atr_at_activation") is None and atr_value is not None:
                trailing_state["atr_at_activation"] = float(atr_value)

            atr_for_calculation = trailing_state.get("atr_at_activation") or atr_value or 0.0

            metadata.update({
                "target_price": trailing_state.get("target_price", target_price),
                "unrealized_target": dynamic_profit_target,
                "activation_price": trailing_state.get("activation_price"),
                "activated": trailing_state.get("activated", False)
            })

            # Configure trailing reversal percentage and volatility modulation
            reversal_pct = max(trailing_config.get("reversal_percentage", 0.02), 0.0)
            min_distance = max(trailing_config.get("min_distance", 0.0), 0.0)

            if trailing_config.get("use_atr_log_scaling", False):
                atr_log_multiplier = max(trailing_config.get("atr_log_multiplier", 0.0), 0.0)
                if atr_for_calculation > 0 and atr_log_multiplier > 0:
                    reversal_pct *= math.log1p(atr_for_calculation * atr_log_multiplier)

            atr_multiplier = max(trailing_config.get("atr_multiplier", 0.0), 0.0)
            atr_pct = 0.0
            if atr_for_calculation > 0 and atr_multiplier > 0 and current_price > 0:
                atr_pct = (atr_for_calculation / current_price) * atr_multiplier

            trailing_pct = max(reversal_pct, atr_pct, min_distance)
            trailing_pct = min(max(trailing_pct, 0.0005), 0.5)

            trailing_state.update({
                "activated": True,
                "trailing_pct": trailing_pct,
                "last_update": time.time()
            })

            if side == "LONG":
                trigger_price = extreme_price * (1.0 - trailing_pct)
                reversal_amount = (extreme_price - current_price) / max(extreme_price, 1e-8)
                triggered = current_price <= trigger_price
            else:  # SHORT
                trigger_price = extreme_price * (1.0 + trailing_pct)
                reversal_amount = (current_price - extreme_price) / max(abs(extreme_price), 1e-8)
                triggered = current_price >= trigger_price

            metadata.update({
                "extreme_price": extreme_price,
                "trigger_price": trigger_price,
                "trailing_pct": trailing_pct,
                "reversal_observed": reversal_amount
            })

            if not triggered:
                metadata["status"] = "armed"
                return (
                    PositionAction.STAY,
                    "Trailing stop armed and monitoring price reversal",
                    metadata
                )

            exit_percentage = max(0.0, min(1.0, profit_config.get("exit_percentage", 1.0)))
            metadata.update({
                "status": "triggered",
                "exit_percentage": exit_percentage
            })

            if side == "LONG":
                reason = (
                    f"Trailing stop hit: price {current_price:.4f} fell "
                    f"{trailing_pct:.2%} from peak {extreme_price:.4f}"
                )
            else:
                reason = (
                    f"Trailing stop hit: price {current_price:.4f} rebounded "
                    f"{trailing_pct:.2%} from trough {extreme_price:.4f}"
                )

            return PositionAction.TRAILING_STOP, reason, metadata

        except Exception as e:
            self.logger.error(failed(f"❌ Error evaluating trailing stop: {e}"))
            return PositionAction.STAY, f"Error in trailing stop evaluation: {e}", {}

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
                    "high": exit_strategy_params.get("confidence_high", 0.8),
                    "force_exit": exit_strategy_params.get("confidence_force_exit_threshold", 0.35),
                    "force_hold": exit_strategy_params.get("confidence_force_hold_threshold", 0.85)
                },
                "profit_taking": {
                    "base_profit_target": exit_strategy_params.get("base_profit_target", 0.04),
                    "min_confidence_for_profit": exit_strategy_params.get("min_confidence_for_profit", 0.6),
                    "confidence_profit_multiplier": exit_strategy_params.get("confidence_profit_multiplier", 0.5),
                    "scaling_levels": [
                        exit_strategy_params.get("profit_tier_1", 0.25),
                        exit_strategy_params.get("profit_tier_2", 0.5),
                        exit_strategy_params.get("profit_tier_3", 0.75)
                    ],
                    "confidence_exit_threshold": exit_strategy_params.get("tp_confidence_threshold", 0.55),
                    "exit_percentage": exit_strategy_params.get("tp_exit_percentage", 1.0),
                    "trailing_stop_enabled": exit_strategy_params.get("trailing_stop_enabled", True),
                    "use_log_atr_multiplier": exit_strategy_params.get("use_tp_atr_log_scaling", False),
                    "atr_log_multiplier": exit_strategy_params.get("tp_atr_log_multiplier", 0.0)
                },
                "stop_loss": {
                    "base_stop_loss": exit_strategy_params.get("base_stop_loss", -0.05),
                    "atr_multiplier": exit_strategy_params.get("atr_multiplier", 1.5),
                    "volatility_adjustment_factor": exit_strategy_params.get("volatility_adjustment_factor", 1.0),
                    "confidence_exit_threshold": exit_strategy_params.get("sl_confidence_threshold", 0.4),
                    "exit_percentage": exit_strategy_params.get("sl_exit_percentage", 1.0),
                    "use_log_atr_multiplier": exit_strategy_params.get("use_sl_atr_log_scaling", False),
                    "atr_log_multiplier": exit_strategy_params.get("sl_atr_log_multiplier", 0.0)
                },
                "time_based": {
                    "max_hold_time": exit_strategy_params.get("max_hold_time", 10800),
                    "min_hold_time": exit_strategy_params.get("min_hold_time", 300),
                    "confidence_time_scaling_factor": exit_strategy_params.get("confidence_time_scaling_factor", 1.0)
                },
                "trailing_stop": {
                    "atr_multiplier": exit_strategy_params.get("trailing_atr_multiplier", 1.5),
                    "min_distance": exit_strategy_params.get("trailing_min_distance", 0.01),
                    "confidence_activation": exit_strategy_params.get("trailing_confidence_activation", 0.7),
                    "reversal_percentage": exit_strategy_params.get("trailing_reversal_pct", 0.02),
                    "use_atr_log_scaling": exit_strategy_params.get("trailing_use_atr_log_scaling", False),
                    "atr_log_multiplier": exit_strategy_params.get("trailing_atr_log_multiplier", 0.0)
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
            "high": 0.8,
            "force_exit": 0.35,
            "force_hold": 0.85
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
            return dict(self.optimized_parameters["profit_taking"])

        # Fallback to config or defaults
        config = dict(self.monitor_config.get("profit_taking", {
            "confidence_scaling": True,
            "min_confidence_for_profit": 0.6,
            "confidence_profit_multiplier": 0.5,
            "tiered_profit_taking": True,
            "trailing_stop_enabled": True,
            "trailing_stop_atr_multiplier": 1.5,
            "confidence_exit_threshold": 0.55,
            "exit_percentage": 1.0,
            "use_log_atr_multiplier": False,
            "atr_log_multiplier": 0.0
        }))
        return config

    def _get_optimized_stop_loss_config(self) -> Dict[str, Any]:
        """Get optimized stop-loss configuration."""
        if self.optimized_parameters and "stop_loss" in self.optimized_parameters:
            return self.optimized_parameters["stop_loss"]
        
        # Fallback to config or defaults
        return self.monitor_config.get("stop_loss", {
            "base_stop_loss": -0.05,
            "atr_multiplier": 1.5,
            "volatility_adjustment": True,
            "regime_adjustment": True,
            "confidence_exit_threshold": 0.4,
            "exit_percentage": 1.0,
            "use_log_atr_multiplier": False,
            "atr_log_multiplier": 0.0
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
        if self.optimized_parameters and "trailing_stop" in self.optimized_parameters:
            return dict(self.optimized_parameters["trailing_stop"])

        # Fallback to config or defaults
        config = dict(self.monitor_config.get("trailing_stop", {
            "enabled": True,
            "atr_multiplier": 1.5,
            "min_distance": 0.01,
            "confidence_activation": 0.7,
            "reversal_percentage": 0.02,
            "use_atr_log_scaling": False,
            "atr_log_multiplier": 0.0
        }))
        return config

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

            self.active_positions[position_id] = position_data
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
