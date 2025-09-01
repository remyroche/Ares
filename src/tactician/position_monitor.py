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
    warning,
)


@dataclass
class PositionAssessment:
    """Position assessment with simplified action logic."""
    
    position_id: str
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    current_price: float
    current_quantity: float
    entry_time: datetime
    current_time: datetime
    
    # ML-based confidence scores
    tactician_confidence: float = 0.0
    analyst_confidence: float = 0.0
    combined_confidence: float = 0.0
    
    # Position metrics
    unrealized_pnl: float = 0.0
    pnl_percentage: float = 0.0
    position_age_hours: float = 0.0
    
    # Decision
    should_exit: bool = False
    should_scale_down: bool = False
    should_take_profit: bool = False
    should_stop_loss: bool = False
    action_reason: str = ""
    
    # Alert information
    alert_severity: str = "info"  # "info", "warning", "critical"
    alert_message: str = ""


class PositionMonitor:
    """
    Position Monitor with fixed 10-second monitoring interval and ML-based confidence.
    
    Features:
    - Fixed 10-second monitoring interval for all positions
    - ML-based confidence assessment (tactician/analyst output)
    - Simplified action logic (mutually exclusive actions)
    - Immediate alert system
    - Fixed -5% PnL threshold
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Position Monitor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("PositionMonitor")

        # Configuration
        self.monitor_config = config.get("position_monitor", {})
        self.monitoring_interval = 10  # Fixed 10 seconds for all positions
        self.pnl_threshold = -0.05  # Fixed -5% PnL threshold
        self.confidence_threshold = self.monitor_config.get("confidence_threshold", 0.6)
        self.high_confidence_threshold = self.monitor_config.get("high_confidence_threshold", 0.8)
        self.low_confidence_threshold = self.monitor_config.get("low_confidence_threshold", 0.4)
        self.very_low_confidence_threshold = self.monitor_config.get("very_low_confidence_threshold", 0.2)

        # Component managers
        self.order_manager: Optional[EnhancedOrderManager] = None
        self.position_strategy: Optional[PositionDivisionStrategy] = None

        # Monitoring state
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        self.assessment_history: List[PositionAssessment] = []

    @handle_errors(
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
            # Validate confidence thresholds
            if not 0 <= self.confidence_threshold <= 1:
                self.logger.error(invalid("Confidence threshold must be between 0 and 1"))
                return False

            if not 0 <= self.high_confidence_threshold <= 1:
                self.logger.error(invalid("High confidence threshold must be between 0 and 1"))
                return False

            if not 0 <= self.low_confidence_threshold <= 1:
                self.logger.error(invalid("Low confidence threshold must be between 0 and 1"))
                return False

            if not 0 <= self.very_low_confidence_threshold <= 1:
                self.logger.error(invalid("Very low confidence threshold must be between 0 and 1"))
                return False

            # Validate PnL threshold
            if self.pnl_threshold >= 0:
                self.logger.error(invalid("PnL threshold should be negative"))
                return False

            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Configuration validation failed: {e}"))
            return False

    async def start_monitoring(self) -> bool:
        """
        Start position monitoring with fixed 10-second interval.

        Returns:
            bool: True if monitoring started successfully
        """
        try:
            if self.is_monitoring:
                self.logger.warning(warning("Position monitoring already active"))
                return True

            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

            self.logger.info(f"✅ Position monitoring started (interval: {self.monitoring_interval}s)")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Failed to start position monitoring: {e}"))
            return False

    async def stop_monitoring(self) -> bool:
        """
        Stop position monitoring.

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
        """Main monitoring loop with fixed 10-second interval."""
        try:
            while self.is_monitoring:
                # Monitor all active positions
                await self._monitor_positions()

                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)

        except asyncio.CancelledError:
            self.logger.info("Position monitoring loop cancelled")
        except Exception as e:
            self.logger.error(failed(f"❌ Error in monitoring loop: {e}"))

    async def _monitor_positions(self) -> None:
        """Monitor all active positions."""
        try:
            if not self.active_positions:
                return

            for position_id, position_data in self.active_positions.items():
                assessment = await self._assess_position(position_id, position_data)
                
                if assessment:
                    # Store assessment
                    self.assessment_history.append(assessment)
                    
                    # Take action based on assessment
                    await self._handle_position_action(assessment)
                    
                    # Send immediate alert if needed
                    if assessment.alert_severity != "info":
                        await self._send_alert(assessment)

        except Exception as e:
            self.logger.error(failed(f"❌ Error monitoring positions: {e}"))

    async def _assess_position(self, position_id: str, position_data: Dict[str, Any]) -> Optional[PositionAssessment]:
        """
        Assess a single position.

        Args:
            position_id: Position ID
            position_data: Position data

        Returns:
            PositionAssessment: Position assessment or None
        """
        try:
            # Get current market data
            current_price = await self._get_current_price(position_data["symbol"])
            if current_price is None:
                return None

            # Calculate position metrics
            entry_price = position_data["entry_price"]
            current_quantity = position_data["current_quantity"]
            side = position_data["side"]
            entry_time = position_data["entry_time"]

            # Calculate PnL
            if side == "long":
                unrealized_pnl = (current_price - entry_price) * current_quantity
            else:
                unrealized_pnl = (entry_price - current_price) * current_quantity

            pnl_percentage = (unrealized_pnl / (entry_price * current_quantity)) * 100

            # Calculate position age
            current_time = datetime.now()
            position_age_hours = (current_time - entry_time).total_seconds() / 3600

            # Get ML confidence scores (from tactician/analyst output)
            tactician_confidence = position_data.get("tactician_confidence", 0.5)
            analyst_confidence = position_data.get("analyst_confidence", 0.5)
            combined_confidence = normalize_dual_confidence(tactician_confidence, analyst_confidence)

            # Determine actions based on ML confidence and PnL
            should_exit = False
            should_scale_down = False
            should_take_profit = False
            should_stop_loss = False
            action_reason = ""
            alert_severity = "info"
            alert_message = ""

            # Check PnL threshold (fixed -5%)
            if pnl_percentage <= self.pnl_threshold * 100:
                should_stop_loss = True
                action_reason = f"PnL threshold reached: {pnl_percentage:.2f}%"
                alert_severity = "critical"
                alert_message = f"Stop loss triggered: {pnl_percentage:.2f}% loss"

            # Check confidence thresholds
            elif combined_confidence <= self.very_low_confidence_threshold:
                should_exit = True
                action_reason = f"Very low confidence: {combined_confidence:.3f}"
                alert_severity = "critical"
                alert_message = f"Exit position due to very low confidence: {combined_confidence:.3f}"

            elif combined_confidence <= self.low_confidence_threshold:
                should_scale_down = True
                action_reason = f"Low confidence: {combined_confidence:.3f}"
                alert_severity = "warning"
                alert_message = f"Scale down position due to low confidence: {combined_confidence:.3f}"

            elif combined_confidence >= self.high_confidence_threshold and pnl_percentage > 0:
                should_take_profit = True
                action_reason = f"High confidence and positive PnL: {combined_confidence:.3f}, {pnl_percentage:.2f}%"
                alert_severity = "info"
                alert_message = f"Take profit opportunity: {pnl_percentage:.2f}% gain"

            # Create assessment
            assessment = PositionAssessment(
                position_id=position_id,
                symbol=position_data["symbol"],
                side=side,
                entry_price=entry_price,
                current_price=current_price,
                current_quantity=current_quantity,
                entry_time=entry_time,
                current_time=current_time,
                tactician_confidence=tactician_confidence,
                analyst_confidence=analyst_confidence,
                combined_confidence=combined_confidence,
                unrealized_pnl=unrealized_pnl,
                pnl_percentage=pnl_percentage,
                position_age_hours=position_age_hours,
                should_exit=should_exit,
                should_scale_down=should_scale_down,
                should_take_profit=should_take_profit,
                should_stop_loss=should_stop_loss,
                action_reason=action_reason,
                alert_severity=alert_severity,
                alert_message=alert_message
            )

            return assessment

        except Exception as e:
            self.logger.error(failed(f"❌ Error assessing position {position_id}: {e}"))
            return None

    async def _handle_position_action(self, assessment: PositionAssessment) -> None:
        """
        Handle position action based on assessment.

        Args:
            assessment: Position assessment
        """
        try:
            if not self.order_manager:
                return

            # Mutually exclusive actions (priority order)
            if assessment.should_stop_loss:
                await self._execute_stop_loss(assessment)
            elif assessment.should_exit:
                await self._execute_exit(assessment)
            elif assessment.should_scale_down:
                await self._execute_scale_down(assessment)
            elif assessment.should_take_profit:
                await self._execute_take_profit(assessment)

        except Exception as e:
            self.logger.error(failed(f"❌ Error handling position action: {e}"))

    async def _execute_stop_loss(self, assessment: PositionAssessment) -> None:
        """Execute stop loss action."""
        try:
            self.logger.warning(f"🛑 Executing stop loss for position {assessment.position_id}")
            
            # Close entire position
            success = await self.order_manager.close_position(
                position_id=assessment.position_id,
                quantity=assessment.current_quantity,
                reason="stop_loss"
            )

            if success:
                self.logger.info(f"✅ Stop loss executed for position {assessment.position_id}")
                # Remove from active positions
                self.active_positions.pop(assessment.position_id, None)
            else:
                self.logger.error(f"❌ Failed to execute stop loss for position {assessment.position_id}")

        except Exception as e:
            self.logger.error(failed(f"❌ Error executing stop loss: {e}"))

    async def _execute_exit(self, assessment: PositionAssessment) -> None:
        """Execute exit action."""
        try:
            self.logger.warning(f"🚪 Executing exit for position {assessment.position_id}")
            
            # Close entire position
            success = await self.order_manager.close_position(
                position_id=assessment.position_id,
                quantity=assessment.current_quantity,
                reason="low_confidence_exit"
            )

            if success:
                self.logger.info(f"✅ Exit executed for position {assessment.position_id}")
                # Remove from active positions
                self.active_positions.pop(assessment.position_id, None)
            else:
                self.logger.error(f"❌ Failed to execute exit for position {assessment.position_id}")

        except Exception as e:
            self.logger.error(failed(f"❌ Error executing exit: {e}"))

    async def _execute_scale_down(self, assessment: PositionAssessment) -> None:
        """Execute scale down action."""
        try:
            self.logger.warning(f"📉 Executing scale down for position {assessment.position_id}")
            
            # Scale down by 50%
            scale_quantity = assessment.current_quantity * 0.5
            
            success = await self.order_manager.close_position(
                position_id=assessment.position_id,
                quantity=scale_quantity,
                reason="low_confidence_scale_down"
            )

            if success:
                self.logger.info(f"✅ Scale down executed for position {assessment.position_id}")
                # Update position quantity
                self.active_positions[assessment.position_id]["current_quantity"] -= scale_quantity
            else:
                self.logger.error(f"❌ Failed to execute scale down for position {assessment.position_id}")

        except Exception as e:
            self.logger.error(failed(f"❌ Error executing scale down: {e}"))

    async def _execute_take_profit(self, assessment: PositionAssessment) -> None:
        """Execute take profit action."""
        try:
            self.logger.info(f"💰 Executing take profit for position {assessment.position_id}")
            
            # Take profit on 50% of position
            profit_quantity = assessment.current_quantity * 0.5
            
            success = await self.order_manager.close_position(
                position_id=assessment.position_id,
                quantity=profit_quantity,
                reason="take_profit"
            )

            if success:
                self.logger.info(f"✅ Take profit executed for position {assessment.position_id}")
                # Update position quantity
                self.active_positions[assessment.position_id]["current_quantity"] -= profit_quantity
            else:
                self.logger.error(f"❌ Failed to execute take profit for position {assessment.position_id}")

        except Exception as e:
            self.logger.error(failed(f"❌ Error executing take profit: {e}"))

    async def _send_alert(self, assessment: PositionAssessment) -> None:
        """
        Send immediate alert based on assessment.

        Args:
            assessment: Position assessment
        """
        try:
            # Immediate alert system
            alert_data = {
                "position_id": assessment.position_id,
                "symbol": assessment.symbol,
                "severity": assessment.alert_severity,
                "message": assessment.alert_message,
                "confidence": assessment.combined_confidence,
                "pnl_percentage": assessment.pnl_percentage,
                "timestamp": assessment.current_time.isoformat()
            }

            # Log alert
            if assessment.alert_severity == "critical":
                self.logger.critical(f"🚨 CRITICAL ALERT: {assessment.alert_message}")
            elif assessment.alert_severity == "warning":
                self.logger.warning(f"⚠️ WARNING: {assessment.alert_message}")
            else:
                self.logger.info(f"ℹ️ INFO: {assessment.alert_message}")

            # TODO: Send to external alert system (email, Slack, etc.)
            # await self._send_external_alert(alert_data)

        except Exception as e:
            self.logger.error(failed(f"❌ Error sending alert: {e}"))

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            float: Current price or None
        """
        try:
            # TODO: Implement actual price fetching
            # For now, return a placeholder
            return 50000.0  # Placeholder price

        except Exception as e:
            self.logger.error(failed(f"❌ Error getting current price: {e}"))
            return None

    def add_position(self, position_data: Dict[str, Any]) -> None:
        """
        Add position to monitoring.

        Args:
            position_data: Position data
        """
        try:
            position_id = position_data["position_id"]
            self.active_positions[position_id] = position_data
            self.logger.info(f"✅ Added position {position_id} to monitoring")

        except Exception as e:
            self.logger.error(failed(f"❌ Error adding position: {e}"))

    def remove_position(self, position_id: str) -> None:
        """
        Remove position from monitoring.

        Args:
            position_id: Position ID
        """
        try:
            if position_id in self.active_positions:
                self.active_positions.pop(position_id)
                self.logger.info(f"✅ Removed position {position_id} from monitoring")

        except Exception as e:
            self.logger.error(failed(f"❌ Error removing position: {e}"))

    def get_position_assessments(self, limit: Optional[int] = None) -> List[PositionAssessment]:
        """
        Get recent position assessments.

        Args:
            limit: Maximum number of assessments to return

        Returns:
            List[PositionAssessment]: Recent assessments
        """
        try:
            if limit:
                return self.assessment_history[-limit:]
            return self.assessment_history.copy()

        except Exception as e:
            self.logger.error(failed(f"❌ Error getting position assessments: {e}"))
            return []

    def get_active_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active positions.

        Returns:
            Dict: Active positions
        """
        return self.active_positions.copy()

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
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


# Setup function for easy integration
async def setup_position_monitor(config: Dict[str, Any]) -> Optional[PositionMonitor]:
    """Setup position monitor."""
    try:
        monitor = PositionMonitor(config)
        if await monitor.initialize():
            return monitor
        return None
    except Exception as e:
        system_logger.error(f"Failed to setup position monitor: {e}")
        return None
