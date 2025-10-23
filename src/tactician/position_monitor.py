# src/tactician/position_monitor.py
from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.core.error_classes import initialization_error
from src.utils.warning_symbols import error, failed, invalid, missing, warning

"""
Position Monitor for real-time position monitoring and confidence assessment.

This module provides continuous monitoring of open positions with confidence score
re-assessment and position decision logic every 10 seconds, using the existing
PositionDivisionStrategy for consistency.
"""
import asyncio
from collections import deque
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from .enhanced_order_manager import EnhancedOrderManager
from .position_division_strategy import PositionDivisionStrategy
from src.utils.confidence import normalize_dual_confidence
import json
import logging
import time

from src.trading.utils.helpers import (
    calculate_atr14,
    calculate_realized_volatility,
    calculate_three_bar_momentum,
    calculate_three_bar_rsi,
    calculate_volatility_slope,
)
from src.trading.utils.ohlcv import ensure_ohlcv_dataframe

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
        self._missing_price_provider_logged = False
        self._latest_market_snapshots: Dict[str, Dict[str, Any]] = {}

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
        self.trailing_state: Dict[str, Dict[str, Any]] = {}
        self.position_assessments: List[PositionAssessment] = []
        self.position_alerts: List[PositionAlert] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False

    def _cache_market_snapshot(self, symbol: str, snapshot: Optional[Dict[str, Any]]) -> None:
        """Store the latest market snapshot and reset missing-provider logging."""

        if snapshot is None:
            return

        self._latest_market_snapshots[symbol] = snapshot
        self._missing_price_provider_logged = False

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
            if self.position_strategy:
                trailing_update = self._build_trailing_strategy_update()
                if trailing_update:
                    self.position_strategy.update_trailing_configuration(trailing_update)

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
                # Get current market snapshot including multi-timeframe context
                market_snapshot = await self._get_market_snapshot(position_data["symbol"])
                if not market_snapshot:
                    continue

                current_price = market_snapshot.get("latest_price")
                if current_price is None:
                    continue

                # Update position data with latest price and context bundle
                position_data["current_price"] = float(current_price)
                position_data["market_snapshot"] = market_snapshot
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
            direction = 1.0 if side == "LONG" else -1.0
            if not trailing_state.get("is_active"):
                directional_score = None
                if momentum_score is not None:
                    directional_score = direction * (momentum_score - momentum_threshold)
                adverse_momentum = directional_score is not None and directional_score < 0
                momentum_triggered = adverse_momentum and unrealized_pnl > 0
                confidence_triggered = unrealized_pnl > 0 and combined_confidence >= activation_threshold

                if momentum_triggered:
                    activation_reason = "adverse_momentum"
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
                if (
                    momentum_score is not None
                    and not (
                        direction * (momentum_score - momentum_threshold) < 0
                    )
                ):
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
            market_snapshot = position_data.get("market_snapshot")
            timeframe_config = self.trailing_stop_config.get("metrics_timeframes", {})
            metrics = self._compute_trailing_metrics(market_snapshot, timeframe_config)
            if not metrics:
                return PositionAction.STAY, "Trailing stop skipped: insufficient market data"

            current_price = position_data.get("current_price")
            if current_price is None:
                return PositionAction.STAY, "Trailing stop skipped: missing current price"

            atr_value = metrics.get("atr14")
            if atr_value is None or pd.isna(atr_value) or atr_value <= 0:
                return PositionAction.STAY, "Trailing stop skipped: ATR unavailable"

            activation_confidence = self.trailing_stop_config.get("confidence_activation", 0.7)
            if combined_confidence < activation_confidence:
                return (
                    PositionAction.STAY,
                    f"Trailing stop inactive: confidence {combined_confidence:.3f} < {activation_confidence:.3f}",
                )

            side = position_data.get("side", "").upper()
            side_multiplier = 1 if side == "LONG" else -1 if side == "SHORT" else None
            if side_multiplier is None:
                return PositionAction.STAY, "Trailing stop skipped: unknown position side"

            trailing_state = position_data.setdefault("trailing_state", {})
            trailing_state["metrics"] = metrics
            trailing_state["last_update"] = datetime.now()

            atr_multiplier = self.trailing_stop_config.get("atr_multiplier", 1.5)
            min_distance_pct = self.trailing_stop_config.get("min_distance", 0.01)

            trailing_distance = max(atr_value * atr_multiplier, float(current_price) * min_distance_pct)

            volatility = metrics.get("realized_volatility20")
            volatility_cfg = self.trailing_stop_config.get("volatility_adjustment", {})
            if (
                volatility_cfg.get("enabled", True)
                and volatility is not None
                and not pd.isna(volatility)
            ):
                offset = volatility_cfg.get("offset", 1.0)
                scale = volatility_cfg.get("scale", 1.0)
                min_mult = volatility_cfg.get("min_multiplier", 0.5)
                max_mult = volatility_cfg.get("max_multiplier", 2.0)
                volatility_multiplier = offset + scale * float(volatility)
                volatility_multiplier = max(min_mult, min(max_mult, volatility_multiplier))
                trailing_distance *= volatility_multiplier

            slope = metrics.get("volatility_slope")
            slope_cfg = self.trailing_stop_config.get("slope_adjustment", {})
            if slope_cfg.get("enabled", True) and slope is not None and not pd.isna(slope):
                offset = slope_cfg.get("offset", 1.0)
                scale = slope_cfg.get("scale", 1.0)
                min_mult = slope_cfg.get("min_multiplier", 0.7)
                max_mult = slope_cfg.get("max_multiplier", 1.3)
                slope_multiplier = offset + scale * float(slope)
                slope_multiplier = max(min_mult, min(max_mult, slope_multiplier))
                trailing_distance *= slope_multiplier

            momentum = metrics.get("momentum3")
            momentum_cfg = self.trailing_stop_config.get("momentum_adjustment", {})
            if (
                momentum_cfg.get("enabled", True)
                and momentum is not None
                and not pd.isna(momentum)
            ):
                if side_multiplier > 0:
                    threshold = momentum_cfg.get("long_threshold")
                    adjustment = momentum_cfg.get("long_multiplier", 1.0)
                    if threshold is not None and float(momentum) < float(threshold):
                        trailing_distance *= adjustment
                else:
                    threshold = momentum_cfg.get("short_threshold")
                    adjustment = momentum_cfg.get("short_multiplier", 1.0)
                    if threshold is not None and float(momentum) > float(threshold):
                        trailing_distance *= adjustment

            rsi = metrics.get("rsi3")
            rsi_cfg = self.trailing_stop_config.get("rsi_adjustment", {})
            if rsi_cfg.get("enabled", True) and rsi is not None and not pd.isna(rsi):
                if side_multiplier > 0:
                    threshold = rsi_cfg.get("long_threshold")
                    adjustment = rsi_cfg.get("long_multiplier", 1.0)
                    if threshold is not None and float(rsi) < float(threshold):
                        trailing_distance *= adjustment
                else:
                    threshold = rsi_cfg.get("short_threshold")
                    adjustment = rsi_cfg.get("short_multiplier", 1.0)
                    if threshold is not None and float(rsi) > float(threshold):
                        trailing_distance *= adjustment

            current_price_float = float(current_price)
            entry_price = position_data.get("entry_price", current_price_float)
            try:
                entry_price = float(entry_price)
            except (TypeError, ValueError):
                entry_price = current_price_float

            default_extreme = max(entry_price, current_price_float) if side_multiplier > 0 else min(entry_price, current_price_float)
            cached_extreme = trailing_state.get("extreme_price")
            if cached_extreme is not None:
                try:
                    cached_extreme = float(cached_extreme)
                except (TypeError, ValueError):
                    cached_extreme = default_extreme
            else:
                cached_extreme = default_extreme

            extreme_price = max(cached_extreme, current_price_float) if side_multiplier > 0 else min(cached_extreme, current_price_float)
            trailing_price = extreme_price - side_multiplier * trailing_distance

            trailing_state["extreme_price"] = extreme_price
            trailing_state["trailing_price"] = trailing_price

            price_difference = side_multiplier * (current_price_float - trailing_price)
            if price_difference <= 0:
                return PositionAction.FULL_CLOSE, f"Trailing stop hit at {trailing_price:.4f}"

            return PositionAction.STAY, f"Trailing stop active at {trailing_price:.4f}"
            trailing_config = self.trailing_stop_config or {}
            if not trailing_config.get("enabled", True):
                return PositionAction.STAY, "Trailing stop disabled"

            activation_threshold = trailing_config.get("confidence_activation", 0.7)
            if combined_confidence < activation_threshold:
                return PositionAction.STAY, (
                    f"Confidence {combined_confidence:.3f} below trailing activation "
                    f"threshold {activation_threshold:.3f}"
                )

            side = position_data.get("side", "LONG").upper()
            entry_price = float(position_data.get("entry_price", 0) or 0)
            current_price = float(position_data.get("current_price", entry_price) or entry_price)

            if entry_price <= 0:
                return PositionAction.STAY, "Invalid entry price for trailing stop"

            def _latest_value(series: Any) -> Optional[float]:
                if series is None:
                    return None
                try:
                    if hasattr(series, "iloc"):
                        return float(series.iloc[-1])
                    if hasattr(series, "__getitem__") and not isinstance(series, str):
                        return float(series[-1])
                except (IndexError, TypeError, ValueError):
                    pass
                try:
                    seq = list(series)
                    if not seq:
                        return None
                    return float(seq[-1])
                except (TypeError, ValueError):
                    return None

            latest_atr = _latest_value(position_data.get("atr_series"))
            if latest_atr is None:
                latest_atr = position_data.get("atr")
            sigma_source = position_data.get("sigma_series")
            if sigma_source is None:
                sigma_source = position_data.get("volatility_series")
            latest_sigma = _latest_value(sigma_source)

            atr_multiplier = float(trailing_config.get("atr_multiplier", 1.5))
            min_distance = float(trailing_config.get("min_distance", 0.01))

            distance_components = []
            if latest_atr is not None:
                try:
                    distance_components.append(float(latest_atr) * atr_multiplier)
                except (ValueError, TypeError):
                    pass
            if latest_sigma is not None:
                try:
                    distance_components.append(float(latest_sigma))
                except (ValueError, TypeError):
                    pass

            if distance_components:
                trailing_distance = sum(distance_components) / len(distance_components)
            else:
                trailing_distance = min_distance

            trailing_distance = max(trailing_distance, min_distance)

            if side == "SHORT":
                profit_pct = max(0.0, (entry_price - current_price) / entry_price)
            else:
                profit_pct = max(0.0, (current_price - entry_price) / entry_price)

            tightening_factor = 1.0 / (1.0 + profit_pct * 5.0)
            trailing_distance = max(min_distance, trailing_distance * tightening_factor)

            trailing_state = position_data.setdefault("trailing_stop_state", {})

            action_on_update = PositionAction.TRAILING_STOP
            update_reason = ""

            if side == "SHORT":
                trough_price = min(
                    float(trailing_state.get("trough_price", entry_price) or entry_price),
                    current_price
                )
                trailing_state["trough_price"] = trough_price
                new_level = trough_price + trailing_distance
                previous_level = trailing_state.get("level")
                if previous_level is not None:
                    new_level = min(float(previous_level), new_level)
                trailing_state["level"] = new_level
                trailing_state["distance"] = trailing_distance
                update_reason = (
                    f"Short trailing stop updated to {new_level:.4f} "
                    f"(distance {trailing_distance:.4f})"
                )
                if current_price >= new_level:
                    trailing_state["triggered"] = True
                    if profit_pct > 0:
                        return (
                            PositionAction.TAKE_PROFIT,
                            f"Short trailing stop triggered at {new_level:.4f}"
                        )
                    return (
                        PositionAction.STOP_LOSS,
                        f"Short trailing stop hit at {new_level:.4f}"
                    )

            else:  # Default to LONG behaviour
                peak_price = max(
                    float(trailing_state.get("peak_price", entry_price) or entry_price),
                    current_price
                )
                trailing_state["peak_price"] = peak_price
                new_level = peak_price - trailing_distance
                previous_level = trailing_state.get("level")
                if previous_level is not None:
                    new_level = max(float(previous_level), new_level)
                trailing_state["level"] = new_level
                trailing_state["distance"] = trailing_distance
                update_reason = (
                    f"Long trailing stop updated to {new_level:.4f} "
                    f"(distance {trailing_distance:.4f})"
                )
                if current_price <= new_level:
                    trailing_state["triggered"] = True
                    if profit_pct > 0:
                        return (
                            PositionAction.TAKE_PROFIT,
                            f"Long trailing stop triggered at {new_level:.4f}"
                        )
                    return (
                        PositionAction.STOP_LOSS,
                        f"Long trailing stop hit at {new_level:.4f}"
                    )

            previous_level = position_data["trailing_stop_state"].get("previous_reported_level")
            current_level = position_data["trailing_stop_state"].get("level")
            tolerance = 1e-6
            position_data["trailing_stop_state"]["previous_reported_level"] = current_level

            if previous_level is None or abs(float(previous_level) - float(current_level)) > tolerance:
                return action_on_update, update_reason

            return PositionAction.STAY, "Trailing stop unchanged"
            if not self.trailing_stop_config.get("enabled", True):
                return PositionAction.STAY, "Trailing stop disabled"

            position_id = position_data.get("position_id")
            if not position_id:
                return PositionAction.STAY, "Position missing identifier for trailing management"

            state = self.trailing_state.get(position_id)
            if state is None:
                state = self._initialize_trailing_state(position_id, position_data)
                if state is None:
                    return PositionAction.STAY, "Unable to initialize trailing state"

            side = str(state.get("side", position_data.get("side", "LONG"))).upper()
            state["side"] = side

            current_price = self._safe_float(position_data.get("current_price"))
            if current_price is None:
                return PositionAction.STAY, "Current price unavailable for trailing stop evaluation"

            entry_price = self._safe_float(state.get("entry_price"))
            if entry_price is None:
                entry_price = self._safe_float(position_data.get("entry_price")) or current_price
                state["entry_price"] = entry_price

            reference_extreme = self._update_trailing_extreme(state, current_price, entry_price)

            atr = self._extract_atr(position_data)
            if atr is not None:
                state["last_atr"] = atr
                if self._safe_float(state.get("entry_atr")) is None:
                    state["entry_atr"] = atr
            else:
                atr = self._safe_float(state.get("last_atr")) or self._safe_float(state.get("entry_atr"))

            sigma = self._extract_sigma(position_data)
            if sigma is not None:
                state["last_sigma"] = sigma
                if self._safe_float(state.get("entry_sigma")) is None:
                    state["entry_sigma"] = sigma
            else:
                sigma = self._safe_float(state.get("last_sigma"))

            profit_buffer = state.get("profit_buffer")
            if profit_buffer is None:
                profit_buffer = self._compute_profit_buffer(entry_price, position_data)
                state["profit_buffer"] = profit_buffer

            breakeven_buffer = state.get("breakeven_buffer")
            if breakeven_buffer is None:
                breakeven_buffer = self._compute_breakeven_buffer(entry_price, position_data)
                state["breakeven_buffer"] = breakeven_buffer

            trailing_metadata = position_data.setdefault("metadata", {}).setdefault("trailing", {})
            trailing_metadata.update(
                {
                    "side": side,
                    "peak_price": state.get("peak_price"),
                    "trough_price": state.get("trough_price"),
                    "entry_price": entry_price,
                    "last_atr": atr,
                    "last_sigma": sigma,
                    "profit_buffer": profit_buffer,
                    "breakeven_buffer": breakeven_buffer,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            activation_threshold = self.trailing_stop_config.get("confidence_activation", 0.0)
            trailing_metadata["activation_threshold"] = activation_threshold
            trailing_metadata["combined_confidence"] = combined_confidence

            if combined_confidence < activation_threshold:
                trailing_metadata["active"] = False
                self.trailing_state[position_id] = state
                return (
                    PositionAction.STAY,
                    f"Confidence {combined_confidence:.3f} below trailing activation {activation_threshold:.3f}",
                )

            trailing_metadata["active"] = True

            min_distance = self._safe_float(self.trailing_stop_config.get("min_distance")) or 0.0
            atr_distance_multiplier = self.trailing_stop_config.get("atr_distance_multiplier", 0.8)
            atr_component = (atr_distance_multiplier * atr) if atr is not None else 0.0
            distance = max(atr_component, profit_buffer or 0.0, min_distance)

            volatility_scaler = self._calculate_volatility_scaler(position_data, state, sigma)
            distance *= volatility_scaler
            trailing_metadata["volatility_scaler"] = volatility_scaler

            profit = (current_price - entry_price) if side == "LONG" else (entry_price - current_price)
            atr_reference = self._determine_atr_reference(entry_price, atr, state)

            distance = self._apply_tightening_tiers(distance, profit, atr_reference, state)
            trailing_metadata["trail_distance"] = distance

            prev_stop = self._safe_float(state.get("trailing_stop"))
            if prev_stop is not None:
                stop_hit = (side == "LONG" and current_price <= prev_stop) or (
                    side == "SHORT" and current_price >= prev_stop
                )
                if stop_hit:
                    trailing_metadata.update(
                        {
                            "trailing_stop": prev_stop,
                            "stop_hit": True,
                            "trail_distance": state.get("last_distance", distance),
                        }
                    )
                    state["trailing_stop"] = prev_stop
                    self.trailing_state[position_id] = state
                    return PositionAction.STOP_LOSS, f"Trailing stop hit at {prev_stop:.4f}"

            new_stop = self._calculate_trailing_stop(side, reference_extreme, distance)

            breakeven_applied = self._maybe_apply_breakeven(
                side,
                profit,
                atr_reference,
                entry_price,
                breakeven_buffer,
                state,
            )
            if breakeven_applied or state.get("breakeven_applied"):
                state["breakeven_applied"] = True
                if side == "LONG":
                    new_stop = max(new_stop, entry_price + (breakeven_buffer or 0.0))
                else:
                    new_stop = min(new_stop, entry_price - (breakeven_buffer or 0.0))

            state["last_distance"] = distance
            stop_improved = (
                prev_stop is None
                or (side == "LONG" and new_stop > prev_stop)
                or (side == "SHORT" and new_stop < prev_stop)
            )

            if stop_improved:
                state["trailing_stop"] = new_stop
                trailing_metadata["trailing_stop"] = new_stop
            else:
                trailing_metadata["trailing_stop"] = prev_stop if prev_stop is not None else new_stop

            trailing_metadata.update(
                {
                    "trail_distance": distance,
                    "breakeven_applied": state.get("breakeven_applied", False),
                    "tightening_triggers": sorted(state.get("tightening_triggered", set())),
                }
            )

            partial_signal = self._extract_trailing_tp_signal(position_data)
            if partial_signal:
                partial_action = self._handle_trailing_partial(
                    position_id, partial_signal, trailing_metadata, state
                )
                if partial_action is not None:
                    self.trailing_state[position_id] = state
                    return partial_action

            if stop_improved:
                self.trailing_state[position_id] = state
                return PositionAction.TRAILING_STOP, f"Updated trailing stop to {new_stop:.4f}"

            self.trailing_state[position_id] = state
            effective_stop = state.get("trailing_stop") or prev_stop or new_stop
            return PositionAction.STAY, f"Trailing stop unchanged at {effective_stop:.4f}"

        except Exception as e:
            self.logger.error(failed(f"❌ Error evaluating trailing stop: {e}"))
            return PositionAction.STAY, f"Error in trailing stop evaluation: {e}"

    def _initialize_trailing_state(
        self, position_id: str, position_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create initial trailing stop state for a position."""

        try:
            side = str(position_data.get("side", "LONG")).upper()
            entry_price = self._safe_float(position_data.get("entry_price"))
            current_price = self._safe_float(position_data.get("current_price"))

            if entry_price is None and current_price is None:
                self.logger.warning(
                    warning(f"Unable to initialize trailing state for {position_id}: missing price information")
                )
                return None

            if entry_price is None:
                entry_price = current_price

            atr = self._extract_atr(position_data)
            sigma = self._extract_sigma(position_data)
            profit_buffer = self._compute_profit_buffer(entry_price, position_data)
            breakeven_buffer = self._compute_breakeven_buffer(entry_price, position_data)

            state: Dict[str, Any] = {
                "position_id": position_id,
                "side": side,
                "entry_price": entry_price,
                "entry_atr": atr,
                "entry_sigma": sigma,
                "profit_buffer": profit_buffer,
                "breakeven_buffer": breakeven_buffer,
                "peak_price": current_price if side == "LONG" else None,
                "trough_price": current_price if side == "SHORT" else None,
                "trailing_stop": None,
                "breakeven_applied": False,
                "tightening_triggered": set(),
                "tp_triggers": set(),
            }

            self.trailing_state[position_id] = state

            position_data.setdefault("metadata", {}).setdefault("trailing", {}).update(
                {
                    "initialized": True,
                    "trail_distance": None,
                    "trailing_stop": None,
                    "profit_buffer": profit_buffer,
                    "breakeven_buffer": breakeven_buffer,
                }
            )

            return state

        except Exception as exc:
            self.logger.error(failed(f"❌ Error initializing trailing state for {position_id}: {exc}"))
            return None

    def _update_trailing_extreme(
        self, state: Dict[str, Any], current_price: float, entry_price: float
    ) -> float:
        """Update peak/trough price for trailing logic and return the reference extreme."""

        side = state.get("side", "LONG")
        if str(side).upper() == "LONG":
            peak = self._safe_float(state.get("peak_price")) or entry_price
            peak = max(peak, current_price)
            state["peak_price"] = peak
            return peak

        trough = self._safe_float(state.get("trough_price")) or entry_price
        trough = min(trough, current_price)
        state["trough_price"] = trough
        return trough

    def _extract_atr(self, position_data: Dict[str, Any]) -> Optional[float]:
        """Extract ATR value from position data if available."""

        candidates = [
            position_data.get("atr"),
            position_data.get("ATR"),
            position_data.get("entry_atr"),
            position_data.get("initial_atr"),
        ]

        for key in ("volatility", "volatility_metrics", "indicators", "analytics"):
            section = position_data.get(key)
            if isinstance(section, dict):
                candidates.append(section.get("atr"))

        for value in candidates:
            atr_value = self._safe_float(value)
            if atr_value is not None:
                return atr_value

        return None

    def _extract_sigma(self, position_data: Dict[str, Any]) -> Optional[float]:
        """Extract volatility sigma from position data."""

        candidates = [
            position_data.get("sigma"),
            position_data.get("volatility_sigma"),
            position_data.get("entry_sigma"),
            position_data.get("std"),
            position_data.get("stddev"),
        ]

        for key in ("volatility", "volatility_metrics", "analytics"):
            section = position_data.get(key)
            if isinstance(section, dict):
                for sigma_key in ("sigma", "std", "stddev"):
                    if sigma_key in section:
                        candidates.append(section.get(sigma_key))

        for value in candidates:
            sigma_value = self._safe_float(value)
            if sigma_value is not None:
                return sigma_value

        return None

    def _compute_profit_buffer(self, entry_price: Optional[float], position_data: Dict[str, Any]) -> float:
        """Determine the profit buffer used for trailing distance floors."""

        explicit = self._safe_float(position_data.get("profit_buffer"))
        if explicit is None:
            explicit = self._safe_float(position_data.get("trail_profit_buffer"))
        if explicit is not None:
            return max(0.0, explicit)

        buffer_pct = self._safe_float(position_data.get("profit_buffer_pct"))
        if buffer_pct is None:
            buffer_pct = self._safe_float(self.trailing_stop_config.get("profit_buffer_pct"))

        if buffer_pct is not None and entry_price is not None:
            return max(0.0, abs(entry_price) * buffer_pct)

        min_distance = self._safe_float(self.trailing_stop_config.get("min_distance"))
        return max(0.0, min_distance or 0.0)

    def _compute_breakeven_buffer(self, entry_price: Optional[float], position_data: Dict[str, Any]) -> float:
        """Determine the breakeven buffer to apply once activated."""

        explicit = self._safe_float(position_data.get("breakeven_buffer"))
        if explicit is None:
            explicit = self._safe_float(position_data.get("breakeven_offset"))
        if explicit is not None:
            return max(0.0, explicit)

        buffer_pct = self._safe_float(position_data.get("breakeven_buffer_pct"))
        if buffer_pct is None:
            buffer_pct = self._safe_float(self.trailing_stop_config.get("breakeven_buffer_pct"))

        if buffer_pct is not None and entry_price is not None:
            return max(0.0, abs(entry_price) * buffer_pct)

        return 0.0

    def _calculate_volatility_scaler(
        self, position_data: Dict[str, Any], state: Dict[str, Any], sigma: Optional[float]
    ) -> float:
        """Compute scaling factor for trailing distance based on volatility regime."""

        scaler = 1.0
        regime = (
            position_data.get("volatility_regime")
            or position_data.get("regime")
            or position_data.get("market_regime")
            or state.get("volatility_regime")
            or "normal"
        )
        regime = str(regime).lower()
        state["volatility_regime"] = regime

        scaling = self.trailing_stop_config.get("volatility_regime_scaling", {})
        if isinstance(scaling, dict):
            regime_scale = self._safe_float(scaling.get(regime))
            if regime_scale is None:
                regime_scale = self._safe_float(scaling.get("default"))
            if regime_scale is not None:
                scaler *= regime_scale

        if sigma is None:
            sigma = self._safe_float(state.get("last_sigma")) or self._safe_float(state.get("entry_sigma"))

        sigma_tiers = self.trailing_stop_config.get("sigma_tiers", [])
        if isinstance(sigma_tiers, list) and sigma is not None:
            for tier in sigma_tiers:
                if not isinstance(tier, dict):
                    continue
                threshold = self._safe_float(tier.get("threshold"))
                multiplier = self._safe_float(tier.get("multiplier"))
                if threshold is None or multiplier is None:
                    continue
                if sigma >= threshold:
                    scaler *= multiplier

        return scaler

    def _determine_atr_reference(
        self, entry_price: Optional[float], atr: Optional[float], state: Dict[str, Any]
    ) -> float:
        """Determine the ATR-like reference used for profit multiples."""

        for candidate in (atr, state.get("entry_atr"), self.trailing_stop_config.get("min_distance")):
            value = self._safe_float(candidate)
            if value is not None and value > 0:
                return value

        if entry_price is not None:
            return max(abs(entry_price) * 0.001, 1e-6)

        return 1e-6

    def _apply_tightening_tiers(
        self,
        distance: float,
        profit: float,
        atr_reference: float,
        state: Dict[str, Any],
    ) -> float:
        """Apply configured tightening tiers to trailing distance."""

        tiers = self.trailing_stop_config.get("tightening_tiers", [])
        if not isinstance(tiers, list) or atr_reference <= 0:
            return distance

        triggered: Set[int] = state.setdefault("tightening_triggered", set())
        min_distance = self._safe_float(self.trailing_stop_config.get("min_distance")) or 0.0

        for index, tier in enumerate(tiers):
            if not isinstance(tier, dict) or index in triggered:
                continue

            trigger_multiple = self._safe_float(tier.get("profit_multiple", tier.get("trigger_multiple")))
            if trigger_multiple is None:
                continue

            if profit < trigger_multiple * atr_reference:
                continue

            tighten_value = tier.get("distance_multiplier")
            if tighten_value is None:
                tighten_value = tier.get("trail_factor")
            if tighten_value is None:
                tighten_value = tier.get("tighten_to")

            tighten_factor = self._safe_float(tighten_value)
            if tighten_factor is None or tighten_factor <= 0:
                continue

            if tighten_factor < 1:
                distance = max(distance * tighten_factor, min_distance)
            else:
                distance = max(tighten_factor, min_distance)

            triggered.add(index)

        state["tightening_triggered"] = triggered
        return distance

    def _calculate_trailing_stop(self, side: str, reference_extreme: float, distance: float) -> float:
        """Calculate the trailing stop level based on side and distance."""

        if str(side).upper() == "LONG":
            return reference_extreme - distance
        return reference_extreme + distance

    def _maybe_apply_breakeven(
        self,
        side: str,
        profit: float,
        atr_reference: float,
        entry_price: float,
        breakeven_buffer: Optional[float],
        state: Dict[str, Any],
    ) -> bool:
        """Determine whether breakeven protection should activate."""

        if breakeven_buffer is None or breakeven_buffer <= 0:
            return False

        if state.get("breakeven_applied"):
            return True

        threshold_multiple = self._safe_float(self.trailing_stop_config.get("breakeven_activation_multiple"))
        if threshold_multiple is None:
            threshold_multiple = 1.0

        if atr_reference <= 0:
            atr_reference = 1e-6

        if profit >= threshold_multiple * atr_reference:
            state["breakeven_applied"] = True
            return True

        return False

    def _extract_trailing_tp_signal(self, position_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract a trailing take-profit signal if available."""

        signal_sources: List[Dict[str, Any]] = []
        for key in ("signals", "tactician_signals", "trailing_signals"):
            source = position_data.get(key)
            if isinstance(source, dict):
                signal_sources.append(source)

        metadata = position_data.get("metadata")
        if isinstance(metadata, dict):
            for key in ("signals", "trailing_signals"):
                source = metadata.get(key)
                if isinstance(source, dict):
                    signal_sources.append(source)

        for source in signal_sources:
            for signal_name in ("TP_trail", "tp_trail", "tpTrail"):
                if signal_name not in source:
                    continue
                raw_signal = source[signal_name]
                if isinstance(raw_signal, dict):
                    signal = dict(raw_signal)
                    signal.setdefault("id", signal_name)
                    signal.setdefault("active", True)
                    return signal
                if raw_signal:
                    return {"id": signal_name, "active": True}

        fallback = position_data.get("TP_trail") or position_data.get("tp_trail")
        if isinstance(fallback, dict):
            signal = dict(fallback)
            signal.setdefault("id", "tp_trail")
            signal.setdefault("active", True)
            return signal
        if fallback:
            return {"id": "tp_trail", "active": True}

        return None

    def _handle_trailing_partial(
        self,
        position_id: str,
        signal: Dict[str, Any],
        trailing_metadata: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Optional[tuple[PositionAction, str]]:
        """Handle trailing take-profit signals and return action if triggered."""

        if not signal.get("active", True):
            return None

        triggers: Set[str] = state.setdefault("tp_triggers", set())
        signal_identifier = str(
            signal.get("id")
            or signal.get("name")
            or signal.get("tier")
            or f"tp_trail_{len(triggers) + 1}"
        )

        if signal_identifier in triggers:
            return None

        partial_size = self._safe_float(signal.get("size"))
        if partial_size is None:
            default_size = self._safe_float(self.trailing_stop_config.get("default_partial_size"))
            if default_size is None:
                default_size = 0.25
            partial_size = default_size

        partial_size = max(0.0, min(1.0, partial_size))

        triggers.add(signal_identifier)
        state["tp_triggers"] = triggers

        trailing_metadata.update(
            {
                "tp_signal_id": signal_identifier,
                "partial_size": partial_size,
                "partial_reason": signal.get("reason", "TP_trail signal"),
            }
        )

        if "target" in signal:
            trailing_metadata["partial_target"] = signal.get("target")

        reason = signal.get("reason") or f"Trailing TP signal triggered ({partial_size:.2f})"
        return PositionAction.PARTIAL_PROFIT, reason

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Convert value to float if possible."""

        try:
            if value is None:
                return None
            result = float(value)
            if math.isnan(result) or math.isinf(result):
                return None
            return result
        except (TypeError, ValueError):
            return None

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

                    if "position_monitor_exit_strategy" in optimization_results:
                        self.logger.info(f"✅ Loaded pre-formatted exit strategy from: {path}")
                        return optimization_results["position_monitor_exit_strategy"]

                    if "exit_strategy" in optimization_results and isinstance(optimization_results["exit_strategy"], dict):
                        exit_strategy_payload = optimization_results["exit_strategy"]
                        # Handle already formatted schema
                        if "confidence_thresholds" in exit_strategy_payload and "profit_taking" in exit_strategy_payload:
                            self.logger.info(f"✅ Loaded exit strategy parameters from: {path}")
                            return exit_strategy_payload

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
            def _ensure_dict(value: Any) -> Dict[str, Any]:
                return value if isinstance(value, dict) else {}

            def _parse_schedule(schedule_value: Any, default: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                if isinstance(schedule_value, str):
                    try:
                        schedule_value = json.loads(schedule_value)
                    except json.JSONDecodeError:
                        return default
                if isinstance(schedule_value, list):
                    if not schedule_value or not isinstance(schedule_value[0], list):
                        return schedule_value  # Assume correct format or empty
                    # Convert list of lists [[mins, mod]] to list of dicts
                    return [{"minutes": item[0], "modifier": item[1]} for item in schedule_value]
                return default

            def _merge_dict(defaults: Dict[str, Any], *candidates: Dict[str, Any]) -> Dict[str, Any]:
                merged = dict(defaults)
                for candidate in candidates:
                    if isinstance(candidate, dict):
                        for key, value in candidate.items():
                            merged[key] = value
                return merged

            def _resolve_scalar(
                nested_key: str,
                fallback_keys: List[str],
                default: Any,
                *sources: Dict[str, Any],
            ) -> Any:
                for source in sources:
                    if isinstance(source, dict) and nested_key in source and source[nested_key] is not None:
                        return source[nested_key]
                for key in fallback_keys:
                    if key in exit_strategy_params and exit_strategy_params[key] is not None:
                        return exit_strategy_params[key]
                return default

            # Prepare defaults for new advanced exit parameters
            stop_loss_defaults = {
                "profit_buffer": 0.001,
                "trail_activation_thresholds": {
                    "activate_at_0_5_atr": 0.5,
                    "trail_tighten_at_0_8_atr": 0.8,
                    "hard_stop_at_1_2_atr": 1.2,
                },
                "regime_bands": {
                    "bull_trend": 0.9,
                    "bear_trend": 1.1,
                    "sideways_range": 1.0,
                },
                "time_decay_modifiers": {
                    "enabled": True,
                    "time_decay_schedule": [
                        {"minutes": 0, "modifier": 1.0},
                        {"minutes": 45, "modifier": 0.95},
                        {"minutes": 90, "modifier": 0.9},
                    ],
                },
            }

            take_profit_defaults = {
                "profit_buffer": 0.0025,
                "trail_activation_thresholds": {
                    "activate_at_0_5_atr": 0.55,
                    "trail_tighten_at_0_8_atr": 0.95,
                    "hard_stop_at_1_2_atr": 1.4,
                },
                "regime_bands": {
                    "bull_trend": 1.1,
                    "bear_trend": 0.9,
                    "sideways_range": 1.0,
                },
                "time_decay_modifiers": {
                    "enabled": True,
                    "time_decay_schedule": [
                        {"minutes": 0, "modifier": 1.0},
                        {"minutes": 90, "modifier": 0.9},
                        {"minutes": 180, "modifier": 0.85},
                    ],
                },
            }

            stop_loss_section = _ensure_dict(exit_strategy_params.get("stop_loss"))
            trailing_stop_section = _ensure_dict(stop_loss_section.get("trailing_stop_config"))
            take_profit_section = _ensure_dict(exit_strategy_params.get("take_profit"))
            dynamic_take_profit_section = _ensure_dict(take_profit_section.get("dynamic_take_profit_config"))
            profit_taking_section = _ensure_dict(exit_strategy_params.get("profit_taking"))

            stop_loss_trail_activation_dict = _ensure_dict(
                trailing_stop_section.get("trail_activation_thresholds")
            )
            take_profit_trail_activation_dict = _ensure_dict(
                dynamic_take_profit_section.get("trail_activation_thresholds")
            )

            stop_loss_profit_buffer = _resolve_scalar(
                "profit_buffer",
                ["stop_loss_profit_buffer", "profit_buffer"],
                stop_loss_defaults["profit_buffer"],
                trailing_stop_section,
                stop_loss_section,
            )

            take_profit_profit_buffer = _resolve_scalar(
                "profit_buffer",
                ["take_profit_profit_buffer", "profit_buffer"],
                take_profit_defaults["profit_buffer"],
                dynamic_take_profit_section,
                take_profit_section,
                profit_taking_section,
            )

            stop_loss_trail_activation = {
                "activate_at_0_5_atr": _resolve_scalar(
                    "activate_at_0_5_atr",
                    [
                        "stop_loss_trail_activate_at_0_5_atr",
                        "trail_activate_at_0_5_atr",
                    ],
                    stop_loss_defaults["trail_activation_thresholds"]["activate_at_0_5_atr"],
                    stop_loss_trail_activation_dict,
                    trailing_stop_section,
                ),
                "trail_tighten_at_0_8_atr": _resolve_scalar(
                    "trail_tighten_at_0_8_atr",
                    ["stop_loss_trail_tighten_at_0_8_atr", "trail_tighten_at_0_8_atr"],
                    stop_loss_defaults["trail_activation_thresholds"]["trail_tighten_at_0_8_atr"],
                    stop_loss_trail_activation_dict,
                    trailing_stop_section,
                ),
                "hard_stop_at_1_2_atr": _resolve_scalar(
                    "hard_stop_at_1_2_atr",
                    ["stop_loss_hard_stop_at_1_2_atr", "hard_stop_at_1_2_atr"],
                    stop_loss_defaults["trail_activation_thresholds"]["hard_stop_at_1_2_atr"],
                    stop_loss_trail_activation_dict,
                    trailing_stop_section,
                ),
            }

            take_profit_trail_activation = {
                "activate_at_0_5_atr": _resolve_scalar(
                    "activate_at_0_5_atr",
                    [
                        "take_profit_trail_activate_at_0_5_atr",
                        "trail_activate_at_0_5_atr",
                    ],
                    take_profit_defaults["trail_activation_thresholds"]["activate_at_0_5_atr"],
                    take_profit_trail_activation_dict,
                    dynamic_take_profit_section,
                    profit_taking_section,
                ),
                "trail_tighten_at_0_8_atr": _resolve_scalar(
                    "trail_tighten_at_0_8_atr",
                    ["take_profit_trail_tighten_at_0_8_atr", "trail_tighten_at_0_8_atr"],
                    take_profit_defaults["trail_activation_thresholds"]["trail_tighten_at_0_8_atr"],
                    take_profit_trail_activation_dict,
                    dynamic_take_profit_section,
                    profit_taking_section,
                ),
                "hard_stop_at_1_2_atr": _resolve_scalar(
                    "hard_stop_at_1_2_atr",
                    ["take_profit_hard_stop_at_1_2_atr", "hard_stop_at_1_2_atr"],
                    take_profit_defaults["trail_activation_thresholds"]["hard_stop_at_1_2_atr"],
                    take_profit_trail_activation_dict,
                    dynamic_take_profit_section,
                    profit_taking_section,
                ),
            }

            stop_loss_regime_bands = _merge_dict(
                stop_loss_defaults["regime_bands"],
                _ensure_dict(trailing_stop_section.get("regime_bands")),
                _ensure_dict(stop_loss_section.get("regime_bands")),
                _ensure_dict(exit_strategy_params.get("stop_loss_regime_bands")),
            )

            take_profit_regime_bands = _merge_dict(
                take_profit_defaults["regime_bands"],
                _ensure_dict(dynamic_take_profit_section.get("regime_bands")),
                _ensure_dict(take_profit_section.get("regime_bands")),
                _ensure_dict(profit_taking_section.get("regime_bands")),
                _ensure_dict(exit_strategy_params.get("take_profit_regime_bands")),
            )

            stop_loss_time_decay_dict = _ensure_dict(
                trailing_stop_section.get("time_decay_modifiers")
            )
            take_profit_time_decay_dict = _ensure_dict(
                dynamic_take_profit_section.get("time_decay_modifiers")
            )

            stop_loss_time_decay_enabled = _resolve_scalar(
                "enabled",
                ["stop_loss_time_decay_enabled", "time_decay_enabled"],
                stop_loss_defaults["time_decay_modifiers"]["enabled"],
                stop_loss_time_decay_dict,
            )
            take_profit_time_decay_enabled = _resolve_scalar(
                "enabled",
                ["take_profit_time_decay_enabled", "time_decay_enabled"],
                take_profit_defaults["time_decay_modifiers"]["enabled"],
                take_profit_time_decay_dict,
                profit_taking_section,
            )

            raw_stop_loss_schedule = _resolve_scalar(
                "time_decay_schedule",
                ["stop_loss_time_decay_schedule", "time_decay_schedule"],
                stop_loss_defaults["time_decay_modifiers"]["time_decay_schedule"],
                stop_loss_time_decay_dict,
            )
            stop_loss_time_decay_schedule = _parse_schedule(
                raw_stop_loss_schedule,
                stop_loss_defaults["time_decay_modifiers"]["time_decay_schedule"],
            )

            raw_take_profit_schedule = _resolve_scalar(
                "time_decay_schedule",
                ["take_profit_time_decay_schedule", "time_decay_schedule"],
                take_profit_defaults["time_decay_modifiers"]["time_decay_schedule"],
                take_profit_time_decay_dict,
                profit_taking_section,
            )
            take_profit_time_decay_schedule = _parse_schedule(
                raw_take_profit_schedule,
                take_profit_defaults["time_decay_modifiers"]["time_decay_schedule"],
            )

            stop_loss_time_decay_modifiers = {
                "enabled": stop_loss_time_decay_enabled,
                "time_decay_schedule": stop_loss_time_decay_schedule,
            }

            take_profit_time_decay_modifiers = {
                "enabled": take_profit_time_decay_enabled,
                "time_decay_schedule": take_profit_time_decay_schedule,
            regime_bands = {
                "trending": exit_strategy_params.get("regime_trending_profit_band", 0.75),
                "ranging": exit_strategy_params.get("regime_ranging_profit_band", 0.55),
                "high_volatility": exit_strategy_params.get("regime_high_volatility_profit_band", 0.65)
            }

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
                    "profit_buffer": exit_strategy_params.get("profit_buffer_ratio", 0.01),
                    "time_decay_half_life": exit_strategy_params.get("profit_time_decay_half_life", 3600),
                    "ml_adjustment_weight": exit_strategy_params.get("profit_ml_adjustment_weight", 0.3),
                    "ml_trigger_multiplier": exit_strategy_params.get("ml_trigger_confidence_multiplier", 1.0),
                    "scaling_levels": [
                        exit_strategy_params.get("profit_tier_1", 0.25),
                        exit_strategy_params.get("profit_tier_2", 0.5),
                        exit_strategy_params.get("profit_tier_3", 0.75)
                    ],
                    "profit_buffer": take_profit_profit_buffer,
                    "trail_activation_thresholds": take_profit_trail_activation,
                    "regime_bands": take_profit_regime_bands,
                    "time_decay_modifiers": take_profit_time_decay_modifiers,
                    "regime_bands": regime_bands
                },
                "stop_loss": {
                    "base_stop_loss": exit_strategy_params.get("base_stop_loss", -0.05),
                    "atr_multiplier": exit_strategy_params.get("atr_multiplier", 1.5),
                    "volatility_adjustment_factor": exit_strategy_params.get("volatility_adjustment_factor", 1.0),
                    "profit_buffer": stop_loss_profit_buffer,
                    "trail_activation_thresholds": stop_loss_trail_activation,
                    "regime_bands": stop_loss_regime_bands,
                    "time_decay_modifiers": stop_loss_time_decay_modifiers,
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
                    "tightening_threshold": exit_strategy_params.get("trailing_tightening_threshold", 0.02),
                    "time_decay": exit_strategy_params.get("trailing_time_decay", 0.95),
                    "ml_adjustment_weight": exit_strategy_params.get("trailing_ml_adjustment_weight", 0.3),
                    "ml_trigger_multiplier": exit_strategy_params.get("ml_trigger_trailing_multiplier", 1.0)
                },
                "regime_aware": {
                    "transition_penalty": exit_strategy_params.get("regime_transition_penalty", 0.1),
                    "regime_specific_scaling": exit_strategy_params.get("regime_specific_scaling", 1.0),
                    "profit_bands": regime_bands,
                    "trailing_sensitivity": exit_strategy_params.get("regime_trailing_sensitivity", 1.0)
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
        base_config: Dict[str, Any] = {
        if self.optimized_parameters and "trailing_stop" in self.optimized_parameters:
            return self.optimized_parameters["trailing_stop"]

        # Fallback to config or defaults
        return self.monitor_config.get("trailing_stop", {
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
            "metrics_timeframes": {
                "primary": "15m",
                "volatility": "1h",
            },
            "volatility_adjustment": {
                "enabled": True,
                "offset": 1.0,
                "scale": 1.0,
                "min_multiplier": 0.5,
                "max_multiplier": 2.0,
            },
            "slope_adjustment": {
                "enabled": True,
                "offset": 1.0,
                "scale": 1.0,
                "min_multiplier": 0.7,
                "max_multiplier": 1.3,
            },
            "momentum_adjustment": {
                "enabled": True,
                "long_threshold": 0.0,
                "short_threshold": 0.0,
                "long_multiplier": 0.9,
                "short_multiplier": 0.9,
            },
            "rsi_adjustment": {
                "enabled": True,
                "long_threshold": 35.0,
                "short_threshold": 65.0,
                "long_multiplier": 0.9,
                "short_multiplier": 0.9,
            },
        })
            "profit_buffer_pct": 0.002,
            "breakeven_buffer_pct": 0.0005,
            "breakeven_activation_multiple": 1.0,
            "volatility_regime_scaling": {
                "low": 0.9,
                "normal": 1.0,
                "high": 1.2,
                "extreme": 1.35,
                "default": 1.0,
            },
            "sigma_tiers": [
                {"threshold": 0.02, "multiplier": 1.05},
                {"threshold": 0.04, "multiplier": 1.15},
            ],
            "tightening_tiers": [
                {"profit_multiple": 1.5, "distance_multiplier": 0.75},
                {"profit_multiple": 2.5, "distance_multiplier": 0.55},
                {"profit_multiple": 3.5, "distance_multiplier": 0.4},
            ],
            "default_partial_size": 0.25,
        }

        if self.optimized_parameters and "trailing_stop" in self.optimized_parameters:
            optimized_config = self.optimized_parameters["trailing_stop"] or {}
            return self._merge_trailing_configs(base_config, optimized_config)

        user_config = self.monitor_config.get("trailing_stop", {})
        return self._merge_trailing_configs(base_config, user_config)

    def _merge_trailing_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge trailing stop configuration dictionaries with nested support."""

        merged = {**base}
        if not override:
            return merged

        for key, value in override.items():
            if key == "volatility_regime_scaling" and isinstance(value, dict):
                base_scaling = dict(base.get("volatility_regime_scaling", {}))
                base_scaling.update(value)
                merged[key] = base_scaling
            elif key in {"sigma_tiers", "tightening_tiers"} and isinstance(value, list):
                merged[key] = value
            else:
                merged[key] = value

        return merged

    def _build_trailing_strategy_update(self) -> Dict[str, Any]:
        """Translate monitor trailing configuration to strategy-friendly values."""

        trailing_config = self.trailing_stop_config or {}
        update: Dict[str, Any] = {}

        if trailing_config:
            update["enabled"] = trailing_config.get("enabled", True)
            update["atr_multiplier"] = trailing_config.get(
                "atr_multiplier",
                trailing_config.get("trailing_stop_atr_multiplier", 1.5),
            )
            min_distance = trailing_config.get("min_distance")
            if min_distance is None:
                min_distance = trailing_config.get("min_trailing_distance_pct")
            if min_distance is not None:
                update["min_trailing_distance_pct"] = min_distance
            if "time_decay_bars" in trailing_config:
                update["time_decay_bars"] = trailing_config["time_decay_bars"]
            if "time_decay_floor_atr" in trailing_config:
                update["time_decay_floor_atr"] = trailing_config["time_decay_floor_atr"]

        profit_buffer = self.pnl_thresholds.get("profit_target")
        if profit_buffer is not None:
            update["profit_buffer_pct"] = profit_buffer

        return update

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
                if self.position_strategy:
                    trailing_update = self._build_trailing_strategy_update()
                    if trailing_update:
                        self.position_strategy.update_trailing_configuration(trailing_update)

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

    async def _get_market_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve a normalized market data snapshot for the requested symbol."""

        try:
            price_provider = self.monitor_config.get("price_provider")
            if callable(price_provider):
                raw_snapshot = (
                    await price_provider(symbol)
                    if asyncio.iscoroutinefunction(price_provider)
                    else price_provider(symbol)
                )
                normalized = self._normalize_market_snapshot(symbol, raw_snapshot)
                if normalized:
                    self._cache_market_snapshot(symbol, normalized)
                    return normalized

            exchange_client = self.monitor_config.get("exchange_client")
            if exchange_client is not None:
                if hasattr(exchange_client, "get_market_snapshot"):
                    func = getattr(exchange_client, "get_market_snapshot")
                    raw_snapshot = (
                        await func(symbol)
                        if asyncio.iscoroutinefunction(func)
                        else func(symbol)
                    )
                    normalized = self._normalize_market_snapshot(symbol, raw_snapshot)
                    if normalized:
                        self._cache_market_snapshot(symbol, normalized)
                        return normalized

                if hasattr(exchange_client, "get_current_price"):
                    func = getattr(exchange_client, "get_current_price")
                    price_value = (
                        await func(symbol)
                        if asyncio.iscoroutinefunction(func)
                        else func(symbol)
                    )
                    normalized = self._normalize_market_snapshot(symbol, price_value)
                    if normalized:
                        self._cache_market_snapshot(symbol, normalized)
                        return normalized

            if not self._missing_price_provider_logged:
                self.logger.error(missing("No price provider configured for PositionMonitor"))
                self._missing_price_provider_logged = True

            # Fall back to the last known snapshot if available
            if symbol in self._latest_market_snapshots:
                return self._latest_market_snapshots[symbol]

            return None

        except Exception as e:
            self.logger.error(failed(f"❌ Error getting market snapshot for {symbol}: {e}"))
            return self._latest_market_snapshots.get(symbol)

    def _normalize_market_snapshot(
        self,
        symbol: str,
        raw_snapshot: Any,
    ) -> Optional[Dict[str, Any]]:
        """Normalize various snapshot formats into a consistent structure."""

        if raw_snapshot is None:
            return None

        snapshot: Dict[str, Any] = {
            "symbol": symbol,
            "latest_price": None,
            "latest_timestamp": None,
            "timeframes": {},
            "metadata": {},
        }

        if isinstance(raw_snapshot, (int, float)):
            snapshot["latest_price"] = float(raw_snapshot)
            snapshot["latest_timestamp"] = datetime.now()
            return snapshot

        if isinstance(raw_snapshot, dict):
            snapshot["symbol"] = raw_snapshot.get("symbol", symbol)
            if raw_snapshot.get("latest_price") is not None:
                snapshot["latest_price"] = float(raw_snapshot["latest_price"])
            snapshot["latest_timestamp"] = raw_snapshot.get("latest_timestamp")

            metadata = raw_snapshot.get("metadata")
            if isinstance(metadata, dict):
                snapshot["metadata"] = metadata

            timeframe_data = raw_snapshot.get("timeframes")
            if isinstance(timeframe_data, dict):
                for interval, frame in timeframe_data.items():
                    normalized_df = ensure_ohlcv_dataframe(frame)
                    if normalized_df is not None:
                        snapshot["timeframes"][interval] = normalized_df
            else:
                for key, value in raw_snapshot.items():
                    if isinstance(value, (pd.DataFrame, list, dict)):
                        normalized_df = ensure_ohlcv_dataframe(value)
                        if normalized_df is not None:
                            snapshot["timeframes"][key] = normalized_df

        elif isinstance(raw_snapshot, pd.DataFrame):
            normalized_df = ensure_ohlcv_dataframe(raw_snapshot)
            if normalized_df is not None:
                snapshot["timeframes"]["default"] = normalized_df

        elif hasattr(raw_snapshot, "close") and hasattr(raw_snapshot, "timestamp"):
            try:
                snapshot["latest_price"] = float(getattr(raw_snapshot, "close"))
                snapshot["latest_timestamp"] = getattr(raw_snapshot, "timestamp")
            except Exception:
                return None
            return snapshot

        if snapshot["latest_price"] is None:
            for df in snapshot["timeframes"].values():
                if "close" in df.columns and not df.empty:
                    snapshot["latest_price"] = float(df["close"].iloc[-1])
                    snapshot["latest_timestamp"] = df.index[-1]
                    break

        if not snapshot["timeframes"] and snapshot["latest_price"] is None:
            return None

        return snapshot

    def _compute_trailing_metrics(
        self,
        market_snapshot: Dict[str, Any],
        timeframe_overrides: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, float]]:
        """Compute technical metrics required for trailing stop evaluation."""

        if not market_snapshot:
            return None

        timeframes = market_snapshot.get("timeframes", {})
        if not timeframes:
            return None

        overrides = timeframe_overrides or {}
        primary_name = overrides.get("primary")
        data_primary = timeframes.get(primary_name) if primary_name else None

        if data_primary is None or data_primary.empty:
            data_primary = next(
                (df for df in timeframes.values() if isinstance(df, pd.DataFrame) and not df.empty),
                None,
            )
            if data_primary is None:
                return None

        volatility_name = overrides.get("volatility", primary_name)
        volatility_source = timeframes.get(volatility_name) if volatility_name else None
        if volatility_source is None or volatility_source.empty:
            volatility_source = data_primary

        metrics: Dict[str, float] = {}

        atr_series = calculate_atr14(data_primary)
        metrics["atr14"] = self._get_latest_metric_value(atr_series)

        realized_vol = calculate_realized_volatility(data_primary)
        metrics["realized_volatility20"] = self._get_latest_metric_value(realized_vol)

        momentum_series = calculate_three_bar_momentum(data_primary)
        metrics["momentum3"] = self._get_latest_metric_value(momentum_series)

        rsi_series = calculate_three_bar_rsi(data_primary)
        metrics["rsi3"] = self._get_latest_metric_value(rsi_series)

        slope_series = calculate_volatility_slope(volatility_source)
        metrics["volatility_slope"] = self._get_latest_metric_value(slope_series)

        return metrics

    @staticmethod
    def _get_latest_metric_value(series: Optional[pd.Series]) -> float:
        """Return the latest value from a metric series, preserving NaN where appropriate."""

        if series is None or series.empty:
            return float(np.nan)

        value = series.iloc[-1]
        return float(value) if not pd.isna(value) else float(np.nan)

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Compatibility wrapper around _get_market_snapshot for legacy callers."""

        snapshot = await self._get_market_snapshot(symbol)
        if snapshot and snapshot.get("latest_price") is not None:
            return float(snapshot["latest_price"])
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
                self.trailing_state.pop(position_id, None)
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
                    except yaml.YAMLError as e:
                        self.logger.warning(f"YAML parsing error in {path}: {e}")
                        continue
                    except (FileNotFoundError, PermissionError) as e:
                        self.logger.warning(f"File access error for {path}: {e}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Unexpected error processing {path}: {e}")
                        continue
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
            stored_data = dict(position_data)
            stored_data["position_id"] = position_id
            stored_data.setdefault("metadata", {})
            if "current_price" not in stored_data and "entry_price" in stored_data:
                stored_data["current_price"] = stored_data.get("entry_price")

            self.active_positions[position_id] = stored_data
            self._initialize_trailing_state(position_id, stored_data)
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
                self.trailing_state.pop(position_id, None)
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

            self.trailing_state.clear()

            self.logger.info("✅ Position Monitor cleanup completed")

        except Exception as e:
            self.logger.error(failed(f"❌ Position Monitor cleanup failed: {e}"))
