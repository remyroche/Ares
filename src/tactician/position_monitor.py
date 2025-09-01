# src/tactician/position_monitor.py
"""
Position Monitor for real-time position monitoring and confidence assessment.

This module provides continuous monitoring of open positions with confidence score
re-assessment and position decision logic every 10 seconds, using the existing
PositionDivisionStrategy for consistency.
"""

import asyncio
import yaml
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tactician.enhanced_order_manager import EnhancedOrderManager
from src.tactician.position_division_strategy import PositionDivisionStrategy
from src.utils.confidence import normalize_dual_confidence
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    failed,
    invalid,
    missing,
    warning,
)

class PositionAction(Enum):
    """Enum for position actions."""

    STAY = "stay"
    EXIT = "exit"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    HEDGE = "hedge"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    FULL_CLOSE = "full_close"

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

        # Load step12 confidence optimization config
        step12_config = config.get("step12_confidence_optimization", {})
        position_monitor_config = step12_config.get("position_monitor", {})

        # Confidence thresholds for step12 optimization
        self.confidence_threshold = position_monitor_config.get("confidence_threshold", 0.6)
        self.high_confidence_threshold = position_monitor_config.get("high_confidence_threshold", 0.6)
        self.low_confidence_threshold = position_monitor_config.get("low_confidence_threshold", 0.3)
        self.very_low_confidence_threshold = position_monitor_config.get("very_low_confidence_threshold", 0.3)

        self.pnl_threshold = position_monitor_config.get("pnl_threshold", -0.05)  # -5%
        self.max_position_age = position_monitor_config.get("max_position_age", 3600)  # 1 hour

        # Component managers
        self.order_manager: Optional[EnhancedOrderManager] = None
        self.position_strategy: Optional[PositionDivisionStrategy] = None

        # Monitoring state
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.position_assessments: List[PositionAssessment] = []
        self.position_alerts: List[PositionAlert] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="position monitor initialization"
    )
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

            if not 0 <= self.confidence_threshold <= 1:
                self.logger.error(invalid("Confidence threshold must be between 0 and 1"))
                return False

            if self.max_position_age <= 0:
                self.logger.error(invalid("Max position age must be positive"))
                return False

            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Configuration validation failed: {e}"))
            return False

    @handle_errors(
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

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="position monitoring stop"
    )
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
        Determine recommended position action based on current conditions.

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

            # Check for critical conditions first
            if unrealized_pnl <= self.pnl_threshold:
                return PositionAction.STOP_LOSS, f"PnL below threshold: {unrealized_pnl:.4f}"

            # Check position age
            if entry_time:
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                position_age = (current_time - entry_time).total_seconds()
                if position_age > self.max_position_age:
                    return PositionAction.FULL_CLOSE, f"Position age exceeded: {position_age:.0f}s"

            # Check confidence-based actions using step12 optimized thresholds
            if combined_confidence < self.very_low_confidence_threshold:
                return PositionAction.FULL_CLOSE, f"Very low confidence: {combined_confidence:.3f} < {self.very_low_confidence_threshold}"
            elif combined_confidence < self.low_confidence_threshold:
                return PositionAction.SCALE_DOWN, f"Low confidence: {combined_confidence:.3f} < {self.low_confidence_threshold}"
            elif combined_confidence >= self.high_confidence_threshold:
                # Check for take profit (high confidence and positive PnL)
                if unrealized_pnl > 0.02:  # 2% profit
                    return PositionAction.TAKE_PROFIT, f"High confidence and profit: {combined_confidence:.3f}, {unrealized_pnl:.4f}"
                else:
                    return PositionAction.STAY, f"High confidence: {combined_confidence:.3f} >= {self.high_confidence_threshold}"

            # Default action for medium confidence
            return PositionAction.STAY, f"Medium confidence: {combined_confidence:.3f} (within thresholds)"

        except Exception as e:
            self.logger.error(failed(f"❌ Error determining position action: {e}"))
            return PositionAction.STAY, f"Error: {e}"

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
                            updated_config = yaml.safe_load(f)

                        # Check if this is newer than our current config
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
