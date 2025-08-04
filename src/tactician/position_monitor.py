# src/tactician/position_monitor.py
"""
Position Monitor for real-time position monitoring and confidence assessment.

This module provides continuous monitoring of open positions with confidence score
re-assessment and position decision logic every 10 seconds, using the existing
PositionDivisionStrategy for consistency.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.tactician.position_division_strategy import PositionDivisionStrategy


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
    """Data class for position assessment results."""
    position_id: str
    current_confidence: float
    entry_confidence: float
    confidence_change: float
    market_conditions: str
    risk_level: str
    recommended_action: PositionAction
    action_reason: str
    assessment_timestamp: datetime
    next_assessment: datetime
    division_analysis: Optional[Dict[str, Any]] = None


class PositionMonitor:
    """
    Real-time position monitor that assesses confidence scores and position decisions.
    
    This monitor runs every 10 seconds when positions are open to continuously
    evaluate whether to stay in the position, exit, or make other adjustments
    based on changing confidence scores and market conditions.
    
    Uses the existing PositionDivisionStrategy for consistent position logic.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("PositionMonitor")
        self.monitoring_interval = config.get("position_monitoring_interval", 10)  # 10 seconds
        self.is_running = False
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.assessment_history: List[PositionAssessment] = []
        self.max_history = config.get("max_assessment_history", 1000)
        
        # Initialize position division strategy
        self.position_division_strategy = PositionDivisionStrategy(config)
        
        # Risk thresholds (for additional risk assessment)
        self.risk_thresholds = {
            "high_risk": config.get("high_risk_threshold", 0.8),
            "medium_risk": config.get("medium_risk_threshold", 0.6),
            "low_risk": config.get("low_risk_threshold", 0.3),
        }
    
    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid position monitor configuration"),
            AttributeError: (False, "Missing required position monitor parameters"),
        },
        default_return=False,
        context="position monitor initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the position monitor."""
        try:
            self.logger.info("🔍 Initializing Position Monitor...")
            
            # Validate configuration
            if self.monitoring_interval < 1:
                self.logger.error("Monitoring interval must be at least 1 second")
                return False
            
            if self.max_history < 1:
                self.logger.error("Max history must be at least 1")
                return False
            
            # Initialize position division strategy
            division_success = await self.position_division_strategy.initialize()
            if not division_success:
                self.logger.error("Failed to initialize position division strategy")
                return False
            
            self.logger.info(f"✅ Position Monitor initialized with {self.monitoring_interval}s interval")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Position Monitor initialization failed: {e}")
            return False
    
    @handle_specific_errors(
        error_handlers={
            Exception: (False, "Position monitor start failed"),
        },
        default_return=False,
        context="position monitor start",
    )
    async def start_monitoring(self) -> bool:
        """Start the position monitoring loop."""
        try:
            self.is_running = True
            self.logger.info("🚦 Position Monitor started")
            
            while self.is_running:
                await self._monitor_positions()
                await asyncio.sleep(self.monitoring_interval)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Position Monitor failed: {e}")
            self.is_running = False
            return False
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="position monitoring",
    )
    async def _monitor_positions(self) -> None:
        """Monitor all active positions and assess confidence scores."""
        try:
            if not self.active_positions:
                return
            
            self.logger.debug(f"🔍 Monitoring {len(self.active_positions)} active positions")
            
            for position_id, position_data in self.active_positions.items():
                assessment = await self._assess_position(position_id, position_data)
                
                if assessment:
                    self.assessment_history.append(assessment)
                    
                    # Keep history size manageable
                    if len(self.assessment_history) > self.max_history:
                        self.assessment_history.pop(0)
                    
                    # Log assessment results
                    self.logger.info(
                        f"📊 Position {position_id}: Confidence {assessment.current_confidence:.3f} "
                        f"({assessment.confidence_change:+.3f}) -> {assessment.recommended_action.value}"
                    )
                    
                    # Execute recommended action
                    await self._execute_position_action(assessment)
            
        except Exception as e:
            self.logger.error(f"Error in position monitoring: {e}")
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="position assessment",
    )
    async def _assess_position(self, position_id: str, position_data: Dict[str, Any]) -> Optional[PositionAssessment]:
        """Assess a single position and determine recommended action."""
        try:
            # Get current market data and confidence scores
            current_confidence = await self._get_current_confidence(position_data)
            entry_confidence = position_data.get("entry_confidence", 0.5)
            confidence_change = current_confidence - entry_confidence
            
            # Assess market conditions
            market_conditions = await self._assess_market_conditions(position_data)
            
            # Assess risk level
            risk_level = await self._assess_risk_level(position_data, current_confidence)
            
            # Use position division strategy for analysis
            division_analysis = await self._analyze_position_with_division_strategy(
                position_id, position_data, current_confidence
            )
            
            # Determine recommended action based on division strategy
            recommended_action, action_reason = self._determine_position_action_from_division(
                division_analysis, current_confidence, confidence_change, market_conditions, risk_level
            )
            
            now = datetime.now()
            next_assessment = now + timedelta(seconds=self.monitoring_interval)
            
            return PositionAssessment(
                position_id=position_id,
                current_confidence=current_confidence,
                entry_confidence=entry_confidence,
                confidence_change=confidence_change,
                market_conditions=market_conditions,
                risk_level=risk_level,
                recommended_action=recommended_action,
                action_reason=action_reason,
                assessment_timestamp=now,
                next_assessment=next_assessment,
                division_analysis=division_analysis
            )
            
        except Exception as e:
            self.logger.error(f"Error assessing position {position_id}: {e}")
            return None
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="position division analysis",
    )
    async def _analyze_position_with_division_strategy(
        self, position_id: str, position_data: Dict[str, Any], current_confidence: float
    ) -> Optional[Dict[str, Any]]:
        """Analyze position using the position division strategy."""
        try:
            # Prepare ML predictions for the division strategy
            ml_predictions = {
                "movement_confidence_scores": {
                    "0.5": current_confidence,
                    "1.0": current_confidence * 0.9,
                    "1.5": current_confidence * 0.8,
                    "2.0": current_confidence * 0.7,
                },
                "current_price": position_data.get("current_price", 0.0),
            }
            
            # Prepare current positions list for analysis
            current_positions = [{
                "position_id": position_id,
                "entry_price": position_data.get("entry_price", 0.0),
                "position_size": position_data.get("position_size", 0.0),
                "entry_confidence": position_data.get("entry_confidence", 0.5),
                "current_price": position_data.get("current_price", 0.0),
            }]
            
            # Get short-term analysis if available
            short_term_analysis = position_data.get("short_term_analysis")
            
            # Use the position division strategy
            division_result = await self.position_division_strategy.analyze_position_division(
                ml_predictions=ml_predictions,
                current_positions=current_positions,
                current_price=position_data.get("current_price", 0.0),
                short_term_analysis=short_term_analysis
            )
            
            return division_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing position with division strategy: {e}")
            return None
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=0.5,
        context="confidence score calculation",
    )
    async def _get_current_confidence(self, position_data: Dict[str, Any]) -> float:
        """Get current confidence score for the position."""
        try:
            # This would integrate with the ML models to get current confidence
            # For now, using a simulated confidence score
            base_confidence = position_data.get("base_confidence", 0.5)
            
            # Simulate confidence changes based on market conditions
            market_volatility = position_data.get("market_volatility", 0.1)
            time_in_position = position_data.get("time_in_position_hours", 1.0)
            
            # Confidence tends to decrease over time and with volatility
            confidence_adjustment = -0.01 * time_in_position - 0.05 * market_volatility
            current_confidence = max(0.0, min(1.0, base_confidence + confidence_adjustment))
            
            return current_confidence
            
        except Exception as e:
            self.logger.error(f"Error getting current confidence: {e}")
            return 0.5
    
    @handle_errors(
        exceptions=(Exception,),
        default_return="neutral",
        context="market conditions assessment",
    )
    async def _assess_market_conditions(self, position_data: Dict[str, Any]) -> str:
        """Assess current market conditions."""
        try:
            # This would analyze current market data
            # For now, using simulated conditions
            volatility = position_data.get("market_volatility", 0.1)
            trend_strength = position_data.get("trend_strength", 0.5)
            
            if volatility > 0.2:
                return "high_volatility"
            elif trend_strength > 0.7:
                return "strong_trend"
            elif trend_strength < 0.3:
                return "weak_trend"
            else:
                return "neutral"
                
        except Exception as e:
            self.logger.error(f"Error assessing market conditions: {e}")
            return "neutral"
    
    @handle_errors(
        exceptions=(Exception,),
        default_return="medium",
        context="risk level assessment",
    )
    async def _assess_risk_level(self, position_data: Dict[str, Any], confidence: float) -> str:
        """Assess current risk level for the position."""
        try:
            # Calculate risk based on position size, leverage, and confidence
            position_size = position_data.get("position_size", 0.1)
            leverage = position_data.get("leverage", 1.0)
            volatility = position_data.get("market_volatility", 0.1)
            
            # Risk increases with position size, leverage, and volatility
            # Risk decreases with confidence
            risk_score = (position_size * leverage * volatility) / max(confidence, 0.1)
            
            if risk_score > self.risk_thresholds["high_risk"]:
                return "high"
            elif risk_score > self.risk_thresholds["medium_risk"]:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            self.logger.error(f"Error assessing risk level: {e}")
            return "medium"
    
    def _determine_position_action_from_division(
        self,
        division_analysis: Optional[Dict[str, Any]],
        current_confidence: float,
        confidence_change: float,
        market_conditions: str,
        risk_level: str
    ) -> Tuple[PositionAction, str]:
        """Determine the recommended position action based on division strategy analysis."""
        try:
            if not division_analysis:
                return PositionAction.STAY, "No division analysis available - default to stay"
            
            # Check for full close actions first (highest priority)
            full_close_actions = division_analysis.get("full_close_actions", [])
            for action in full_close_actions:
                if action.get("should_full_close", False):
                    return PositionAction.FULL_CLOSE, action.get("reason", "Full close recommended")
            
            # Check for stop loss actions
            stop_loss_actions = division_analysis.get("stop_loss_actions", [])
            for action in stop_loss_actions:
                if action.get("should_stop_loss", False):
                    return PositionAction.STOP_LOSS, action.get("reason", "Stop loss recommended")
            
            # Check for take profit actions
            take_profit_actions = division_analysis.get("take_profit_actions", [])
            for action in take_profit_actions:
                if action.get("should_take_profit", False):
                    return PositionAction.TAKE_PROFIT, action.get("reason", "Take profit recommended")
            
            # Check for entry actions (for scaling up)
            entry_recommendation = division_analysis.get("entry_recommendation", {})
            if entry_recommendation.get("should_enter", False):
                return PositionAction.SCALE_UP, entry_recommendation.get("reason", "Scale up recommended")
            
            # Additional risk-based actions
            if risk_level == "high" and current_confidence < 0.5:
                return PositionAction.HEDGE, "High risk with low confidence - hedge recommended"
            
            if market_conditions == "high_volatility":
                return PositionAction.SCALE_DOWN, "High volatility - reduce position size"
            
            # Default: stay
            return PositionAction.STAY, "Conditions acceptable - maintain position"
            
        except Exception as e:
            self.logger.error(f"Error determining position action from division: {e}")
            return PositionAction.STAY, "Error in assessment - default to stay"
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="position action execution",
    )
    async def _execute_position_action(self, assessment: PositionAssessment) -> None:
        """Execute the recommended position action."""
        try:
            action = assessment.recommended_action
            
            if action == PositionAction.EXIT or action == PositionAction.FULL_CLOSE:
                await self._execute_exit(assessment)
            elif action == PositionAction.SCALE_UP:
                await self._execute_scale_up(assessment)
            elif action == PositionAction.SCALE_DOWN:
                await self._execute_scale_down(assessment)
            elif action == PositionAction.HEDGE:
                await self._execute_hedge(assessment)
            elif action == PositionAction.TAKE_PROFIT:
                await self._execute_take_profit(assessment)
            elif action == PositionAction.STOP_LOSS:
                await self._execute_stop_loss(assessment)
            else:  # STAY
                await self._execute_stay(assessment)
                
        except Exception as e:
            self.logger.error(f"Error executing position action: {e}")
    
    async def _execute_exit(self, assessment: PositionAssessment) -> None:
        """Execute position exit."""
        self.logger.info(f"🚪 EXITING position {assessment.position_id}: {assessment.action_reason}")
        # This would integrate with the trading system to close the position
        # For now, just log the action
    
    async def _execute_scale_up(self, assessment: PositionAssessment) -> None:
        """Execute position scale up."""
        self.logger.info(f"📈 SCALING UP position {assessment.position_id}: {assessment.action_reason}")
        # This would integrate with the trading system to increase position size
    
    async def _execute_scale_down(self, assessment: PositionAssessment) -> None:
        """Execute position scale down."""
        self.logger.info(f"📉 SCALING DOWN position {assessment.position_id}: {assessment.action_reason}")
        # This would integrate with the trading system to reduce position size
    
    async def _execute_hedge(self, assessment: PositionAssessment) -> None:
        """Execute position hedge."""
        self.logger.info(f"🛡️ HEDGING position {assessment.position_id}: {assessment.action_reason}")
        # This would integrate with the trading system to add a hedge position
    
    async def _execute_take_profit(self, assessment: PositionAssessment) -> None:
        """Execute take profit."""
        self.logger.info(f"💰 TAKING PROFIT on position {assessment.position_id}: {assessment.action_reason}")
        # This would integrate with the trading system to take profit
    
    async def _execute_stop_loss(self, assessment: PositionAssessment) -> None:
        """Execute stop loss."""
        self.logger.info(f"🛑 STOP LOSS on position {assessment.position_id}: {assessment.action_reason}")
        # This would integrate with the trading system to stop loss
    
    async def _execute_stay(self, assessment: PositionAssessment) -> None:
        """Execute position stay (no action)."""
        self.logger.debug(f"⏸️ STAYING in position {assessment.position_id}: {assessment.action_reason}")
        # No action needed, just continue monitoring
    
    def add_position(self, position_id: str, position_data: Dict[str, Any]) -> None:
        """Add a position to be monitored."""
        try:
            self.active_positions[position_id] = position_data
            self.logger.info(f"➕ Added position {position_id} to monitoring")
        except Exception as e:
            self.logger.error(f"Error adding position {position_id}: {e}")
    
    def remove_position(self, position_id: str) -> None:
        """Remove a position from monitoring."""
        try:
            if position_id in self.active_positions:
                del self.active_positions[position_id]
                self.logger.info(f"➖ Removed position {position_id} from monitoring")
        except Exception as e:
            self.logger.error(f"Error removing position {position_id}: {e}")
    
    def get_active_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active positions being monitored."""
        return self.active_positions.copy()
    
    def get_assessment_history(self, limit: Optional[int] = None) -> List[PositionAssessment]:
        """Get assessment history."""
        history = self.assessment_history.copy()
        if limit:
            history = history[-limit:]
        return history
    
    def get_position_status(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific position."""
        try:
            if position_id not in self.active_positions:
                return None
            
            position_data = self.active_positions[position_id]
            
            # Find the latest assessment for this position
            latest_assessment = None
            for assessment in reversed(self.assessment_history):
                if assessment.position_id == position_id:
                    latest_assessment = assessment
                    break
            
            return {
                "position_data": position_data,
                "latest_assessment": latest_assessment,
                "is_monitored": True
            }
            
        except Exception as e:
            self.logger.error(f"Error getting position status for {position_id}: {e}")
            return None
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="position monitor stop",
    )
    async def stop_monitoring(self) -> None:
        """Stop the position monitoring."""
        self.logger.info("🛑 Stopping Position Monitor...")
        try:
            self.is_running = False
            await self.position_division_strategy.stop()
            self.logger.info("✅ Position Monitor stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping position monitor: {e}")


# Global position monitor instance
position_monitor: Optional[PositionMonitor] = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="position monitor setup",
)
async def setup_position_monitor(config: Optional[Dict[str, Any]] = None) -> Optional[PositionMonitor]:
    """Setup the global position monitor instance."""
    try:
        global position_monitor
        if config is None:
            config = {
                "position_monitoring_interval": 10,
                "max_assessment_history": 1000,
                "high_risk_threshold": 0.8,
                "medium_risk_threshold": 0.6,
                "low_risk_threshold": 0.3,
                # Position division strategy config
                "position_division": {
                    "entry_confidence_threshold": 0.7,
                    "additional_position_threshold": 0.8,
                    "max_positions": 3,
                    "take_profit_confidence_decrease": 0.1,
                    "take_profit_short_term_decrease": 0.08,
                    "stop_loss_confidence_threshold": 0.3,
                    "stop_loss_short_term_threshold": 0.24,
                    "stop_loss_price_threshold": -0.05,
                    "full_close_confidence_threshold": 0.2,
                    "full_close_short_term_threshold": 0.16,
                    "ml_confidence_weight": 0.6,
                    "price_action_weight": 0.2,
                    "volume_weight": 0.2,
                }
            }
        
        position_monitor = PositionMonitor(config)
        success = await position_monitor.initialize()
        if success:
            return position_monitor
        return None
    except Exception as e:
        print(f"Error setting up position monitor: {e}")
        return None 