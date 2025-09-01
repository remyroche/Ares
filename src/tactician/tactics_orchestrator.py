# src/tactician/tactics_orchestrator.py

"""
Tactics Orchestrator for coordinating all tactical components.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# Temporary TradeDecision definition to allow imports
from dataclasses import dataclass

@dataclass
class TradeDecision:
    action: str
    confidence: float
    position_size: float = 0.0
    leverage: float = 1.0
    price: float = None
    metadata: dict = None

from src.tactician.enhanced_order_manager import EnhancedOrderManager
from src.tactician.leverage_sizer import LeverageSizer
from src.tactician.ml_tactics_manager import MLTacticsManager
from src.tactician.position_closing import PositionCloser
from src.tactician.position_division_strategy import PositionDivisionStrategy
from src.tactician.position_monitor import PositionAction, PositionAssessment, PositionMonitor
from src.tactician.position_sizer import PositionSizer
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    failed,
    invalid,
)


class DecisionPolicy:
    """
    Aggregates sizing, leverage, SR breakout, and ML signals into a unified TradeDecision.
    Provides audit-friendly metadata and metrics.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the decision policy.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("DecisionPolicy")

        # Configuration
        self.policy_config = config.get("decision_policy", {})
        self.confidence_threshold = self.policy_config.get("confidence_threshold", 0.6)
        self.risk_threshold = self.policy_config.get("risk_threshold", 0.1)

        # Component managers
        self.position_sizer: Optional[PositionSizer] = None
        self.leverage_sizer: Optional[LeverageSizer] = None
        self.sr_predictor: Optional[SRBreakoutPredictor] = None
        self.ml_tactics: Optional[MLTacticsManager] = None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="decision policy initialization"
    )
    async def initialize(self) -> bool:
        """
        Initialize the decision policy.

        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing Decision Policy...")

            # Initialize component managers
            await self._initialize_components()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(invalid("Invalid decision policy configuration"))
                return False

            self.logger.info("✅ Decision Policy initialized successfully")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Decision Policy initialization failed: {e}"))
            return False

    def refresh_step17_configuration(self, step17_results: dict[str, Any]) -> None:
        """
        Refresh configuration from step17 optimization results.
        This method is called automatically when step17 completes.

        Args:
            step17_results: Step17 optimization results
        """
        try:
            # Update decision policy configuration
            if "decision_policy" in step17_results:
                policy_optimization = step17_results["decision_policy"]
                self.confidence_threshold = policy_optimization.get("confidence_threshold", self.confidence_threshold)
                self.risk_threshold = policy_optimization.get("risk_threshold", self.risk_threshold)

            # Refresh all component managers
            if self.position_sizer:
                self.position_sizer.refresh_step17_configuration(step17_results)

            if self.leverage_sizer:
                self.leverage_sizer.refresh_step17_configuration(step17_results)

            if self.ml_tactics:
                self.ml_tactics.refresh_step17_configuration(step17_results)

            self.logger.info("✅ Decision policy configuration refreshed from step17 results")

        except Exception as e:
            self.logger.error(f"Error refreshing step17 configuration: {e}")

    async def _initialize_components(self) -> None:
        """Initialize all component managers."""
        try:
            # Initialize position sizer
            self.position_sizer = PositionSizer(self.config)
            await self.position_sizer.initialize()

            # Initialize leverage sizer
            self.leverage_sizer = LeverageSizer(self.config)
            await self.leverage_sizer.initialize()

            # Initialize SR breakout predictor with optimized parameters
            sr_config = self.config.copy()
            sr_config["sr_breakout_predictor"] = sr_config.get("sr_breakout_predictor", {})
            sr_config["sr_breakout_predictor"]["use_optimized_params"] = True
            self.sr_predictor = SRBreakoutPredictor(sr_config)
            await self.sr_predictor.initialize()

            # Initialize ML tactics manager
            self.ml_tactics = MLTacticsManager(self.config)
            await self.ml_tactics.initialize()

        except Exception as e:
            self.logger.error(failed(f"❌ Component initialization failed: {e}"))

    def _validate_configuration(self) -> bool:
        """
        Validate decision policy configuration.

        Returns:
            bool: True if configuration is valid
        """
        try:
            # Basic validation
            if not isinstance(self.confidence_threshold, (int, float)):
                return False
            if not 0 <= self.confidence_threshold <= 1:
                return False
            if not isinstance(self.risk_threshold, (int, float)):
                return False
            if not 0 <= self.risk_threshold <= 1:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False

    async def make_decision(self, market_data: Dict[str, Any]) -> TradeDecision:
        """
        Make a trading decision based on aggregated signals.

        Args:
            market_data: Current market data

        Returns:
            TradeDecision: Trading decision with confidence and metadata
        """
        try:
            # Get signals from all components
            position_signal = await self.position_sizer.get_signal(market_data)
            leverage_signal = await self.leverage_sizer.get_signal(market_data)
            sr_signal = await self.sr_predictor.get_signal(market_data)
            ml_signal = await self.ml_tactics.get_signal(market_data)

            # Aggregate signals (simplified for now)
            confidence = (position_signal.get("confidence", 0) + 
                         leverage_signal.get("confidence", 0) + 
                         sr_signal.get("confidence", 0) + 
                         ml_signal.get("confidence", 0)) / 4

            # Determine action based on confidence
            if confidence >= self.confidence_threshold:
                action = "BUY" if sr_signal.get("direction", "NEUTRAL") == "BULLISH" else "SELL"
            else:
                action = "HOLD"

            return TradeDecision(
                action=action,
                confidence=confidence,
                position_size=position_signal.get("size", 0.0),
                leverage=leverage_signal.get("leverage", 1.0),
                price=market_data.get("price", 0.0),
                metadata={
                    "position_signal": position_signal,
                    "leverage_signal": leverage_signal,
                    "sr_signal": sr_signal,
                    "ml_signal": ml_signal
                }
            )

        except Exception as e:
            self.logger.error(f"Error making decision: {e}")
            return TradeDecision(
                action="HOLD",
                confidence=0.0,
                metadata={"error": str(e)}
            )


class TacticsOrchestrator:
    """
    Main orchestrator for coordinating all tactical components.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tactics orchestrator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("TacticsOrchestrator")

        # Core components
        self.decision_policy: Optional[DecisionPolicy] = None
        self.position_monitor: Optional[PositionMonitor] = None
        self.order_manager: Optional[EnhancedOrderManager] = None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="tactics orchestrator initialization"
    )
    async def initialize(self) -> bool:
        """
        Initialize the tactics orchestrator.

        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing Tactics Orchestrator...")

            # Initialize decision policy
            self.decision_policy = DecisionPolicy(self.config)
            await self.decision_policy.initialize()

            # Initialize position monitor
            self.position_monitor = PositionMonitor(self.config)
            await self.position_monitor.initialize()

            # Initialize order manager
            self.order_manager = EnhancedOrderManager(self.config)
            await self.order_manager.initialize()

            self.logger.info("✅ Tactics Orchestrator initialized successfully")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Tactics Orchestrator initialization failed: {e}"))
            return False

    async def process_market_data(self, market_data: Dict[str, Any]) -> TradeDecision:
        """
        Process market data and return trading decision.

        Args:
            market_data: Current market data

        Returns:
            TradeDecision: Trading decision
        """
        try:
            # Get decision from policy
            decision = await self.decision_policy.make_decision(market_data)

            # Monitor existing positions
            if self.position_monitor:
                await self.position_monitor.monitor_positions(market_data)

            return decision

        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            return TradeDecision(
                action="HOLD",
                confidence=0.0,
                metadata={"error": str(e)}
            )

    async def execute_decision(self, decision: TradeDecision) -> bool:
        """
        Execute a trading decision.

        Args:
            decision: Trading decision to execute

        Returns:
            bool: True if execution successful
        """
        try:
            if decision.action == "HOLD":
                return True

            # Execute order through order manager
            if self.order_manager:
                success = await self.order_manager.execute_order(decision)
                return success

            return False

        except Exception as e:
            self.logger.error(f"Error executing decision: {e}")
            return False
