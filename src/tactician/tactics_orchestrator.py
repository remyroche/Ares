# src/tactician/tactics_orchestrator.py

"""
Tactics Orchestrator for coordinating all tactical components.
"""

import asyncio

# Temporary TradeDecision definition to allow imports
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# from exchange.factory import ExchangeFactory  # Temporarily commented due to syntax errors in binance.py
# from src.config.environment import get_exchange_name  # Temporarily commented due to missing function
# from src.interfaces.base_interfaces import TradeDecision  # Temporarily commented due to syntax errors
# from src.interfaces.event_bus import EventType  # Temporarily commented due to syntax errors


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
from src.tactician.position_monitor import (
    PositionAction,
    PositionAssessment,
    PositionMonitor,
)
from src.tactician.position_sizer import PositionSizer
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import copy, failed, import, invalid


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
            if not 0 <= self.confidence_threshold <= 1:
                self.logger.error(invalid("Confidence threshold must be between 0 and 1"))
                return False

            if not 0 <= self.risk_threshold <= 1:
                self.logger.error(invalid("Risk threshold must be between 0 and 1"))
                return False

            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Configuration validation failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="trade decision generation"
    )
    async def generate_decision(
        self,
        market_data: pd.DataFrame,
        analyst_confidence: float,
        tactician_confidence: float,
        position_data: Optional[Dict[str, Any]] = None
    ) -> Optional[TradeDecision]:
        """
        Generate a trade decision based on all available signals.

        Args:
            market_data: Market data
            analyst_confidence: Analyst confidence score
            tactician_confidence: Tactician confidence score
            position_data: Current position data (if any)

        Returns:
            TradeDecision: Generated trade decision or None if no decision
        """
        try:
            self.logger.info("Generating trade decision...")

            # Get component decisions
            sizing_decision = await self._get_sizing_decision(analyst_confidence, tactician_confidence)
            leverage_decision = await self._get_leverage_decision(analyst_confidence, tactician_confidence)
            sr_decision = await self._get_sr_decision(market_data)
            ml_decision = await self._get_ml_decision(market_data, analyst_confidence, tactician_confidence)

            # Aggregate decisions
            decision = self._aggregate_decisions(
                sizing_decision,
                leverage_decision,
                sr_decision,
                ml_decision,
                analyst_confidence,
                tactician_confidence
            )

            if decision:
                self.logger.info(f"✅ Trade decision generated: {decision.action}")

            return decision

        except Exception as e:
            self.logger.error(failed(f"❌ Trade decision generation failed: {e}"))
            return None

    async def _get_sizing_decision(
        self,
        analyst_confidence: float,
        tactician_confidence: float
    ) -> Optional[Dict[str, Any]]:
        """
        Get position sizing decision.

        Args:
            analyst_confidence: Analyst confidence score
            tactician_confidence: Tactician confidence score

        Returns:
            Dict: Sizing decision or None
        """
        try:
            if not self.position_sizer:
                return None

            # Calculate combined confidence
            combined_confidence = (analyst_confidence + tactician_confidence) / 2

            # Get position size
            position_size = await self.position_sizer.calculate_position_size(
                ml_predictions={},  # Placeholder for ML predictions
                analyst_confidence=analyst_confidence,
                tactician_confidence=tactician_confidence
            )

            return {
                "position_size": position_size,
                "confidence": combined_confidence,
                "source": "position_sizer"
            }

        except Exception as e:
            self.logger.error(failed(f"❌ Sizing decision failed: {e}"))
            return None

    async def _get_leverage_decision(
        self,
        analyst_confidence: float,
        tactician_confidence: float
    ) -> Optional[Dict[str, Any]]:
        """
        Get leverage decision.

        Args:
            analyst_confidence: Analyst confidence score
            tactician_confidence: Tactician confidence score

        Returns:
            Dict: Leverage decision or None
        """
        try:
            if not self.leverage_sizer:
                return None

            # Calculate combined confidence
            combined_confidence = (analyst_confidence + tactician_confidence) / 2

            # Get leverage
            leverage = await self.leverage_sizer.calculate_leverage(
                ml_predictions={},  # Placeholder for ML predictions
                analyst_confidence=analyst_confidence,
                tactician_confidence=tactician_confidence
            )

            return {
                "leverage": leverage,
                "confidence": combined_confidence,
                "source": "leverage_sizer"
            }

        except Exception as e:
            self.logger.error(failed(f"❌ Leverage decision failed: {e}"))
            return None

    async def _get_sr_decision(self, market_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Get SR breakout decision using centralized logic.

        Args:
            market_data: Market data

        Returns:
            Dict: SR decision or None
        """
        try:
            if not self.sr_predictor:
                return None

            # Get SR breakout prediction using centralized logic
            prediction = await self.sr_predictor.predict_breakout(market_data)

            if not prediction:
                return None

            return {
                "breakout_direction": prediction.get("direction"),
                "breakout_confidence": prediction.get("confidence", 0.0),
                "breakout_price": prediction.get("price"),
                "outcome": prediction.get("outcome", "consolidation"),
                "sr_context": prediction.get("sr_context", {}),
                "source": "sr_predictor"
            }

        except Exception as e:
            self.logger.error(failed(f"❌ SR decision failed: {e}"))
            return None

    async def _get_ml_decision(
        self,
        market_data: pd.DataFrame,
        analyst_confidence: float,
        tactician_confidence: float
    ) -> Optional[Dict[str, Any]]:
        """
        Get ML tactics decision.

        Args:
            market_data: Market data
            analyst_confidence: Analyst confidence score
            tactician_confidence: Tactician confidence score

        Returns:
            Dict: ML decision or None
        """
        try:
            if not self.ml_tactics:
                return None

            # Get ML tactics decision
            decision = await self.ml_tactics.get_tactics_decision(
                market_data,
                analyst_confidence,
                tactician_confidence
            )

            return decision

        except Exception as e:
            self.logger.error(failed(f"❌ ML decision failed: {e}"))
            return None

    def _aggregate_decisions(
        self,
        sizing_decision: Optional[Dict[str, Any]],
        leverage_decision: Optional[Dict[str, Any]],
        sr_decision: Optional[Dict[str, Any]],
        ml_decision: Optional[Dict[str, Any]],
        analyst_confidence: float,
        tactician_confidence: float
    ) -> Optional[TradeDecision]:
        """
        Aggregate all component decisions into a unified trade decision.

        Args:
            sizing_decision: Position sizing decision
            leverage_decision: Leverage decision
            sr_decision: SR breakout decision
            ml_decision: ML tactics decision
            analyst_confidence: Analyst confidence score
            tactician_confidence: Tactician confidence score

        Returns:
            TradeDecision: Aggregated trade decision or None
        """
        try:
            # Calculate overall confidence
            combined_confidence = (analyst_confidence + tactician_confidence) / 2

            # Check confidence threshold
            if combined_confidence < self.confidence_threshold:
                self.logger.info(f"Confidence {combined_confidence:.3f} below threshold {self.confidence_threshold}")
                return None

            # Determine action based on decisions
            action = self._determine_action(sizing_decision, leverage_decision, sr_decision, ml_decision)

            if not action:
                return None

            # Create trade decision
            decision = TradeDecision(
                action=action,
                confidence=combined_confidence,
                position_size=sizing_decision.get("position_size", 0.0) if sizing_decision else 0.0,
                leverage=leverage_decision.get("leverage", 1.0) if leverage_decision else 1.0,
                price=sr_decision.get("breakout_price") if sr_decision else None,
                metadata={
                    "analyst_confidence": analyst_confidence,
                    "tactician_confidence": tactician_confidence,
                    "sizing_decision": sizing_decision,
                    "leverage_decision": leverage_decision,
                    "sr_decision": sr_decision,
                    "ml_decision": ml_decision,
                    "timestamp": datetime.now().isoformat()
                }
            )

            return decision

        except Exception as e:
            self.logger.error(failed(f"❌ Decision aggregation failed: {e}"))
            return None

    def _determine_action(
        self,
        sizing_decision: Optional[Dict[str, Any]],
        leverage_decision: Optional[Dict[str, Any]],
        sr_decision: Optional[Dict[str, Any]],
        ml_decision: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Determine the final action based on all decisions.

        Args:
            sizing_decision: Position sizing decision
            leverage_decision: Leverage decision
            sr_decision: SR breakout decision
            ml_decision: ML tactics decision

        Returns:
            str: Action to take or None
        """
        try:
            # Check if we have enough information
            if not sizing_decision or not leverage_decision:
                return None

            # Check if position size is significant
            position_size = sizing_decision.get("position_size", 0.0)
            if position_size <= 0:
                return None

            # Check SR breakout direction
            breakout_direction = sr_decision.get("breakout_direction") if sr_decision else None

            # Determine action
            if breakout_direction == "up":
                return "BUY"
            elif breakout_direction == "down":
                return "SELL"
            else:
                # Use ML decision if available
                if ml_decision:
                    return ml_decision.get("action")

                # Default to no action
                return None

        except Exception as e:
            self.logger.error(failed(f"❌ Action determination failed: {e}"))
            return None

    async def cleanup(self) -> None:
        """
        Cleanup resources.
        """
        try:
            self.logger.info("Cleaning up Decision Policy...")

            # Cleanup component managers
            if self.position_sizer:
                await self.position_sizer.cleanup()

            if self.leverage_sizer:
                await self.leverage_sizer.cleanup()

            if self.sr_predictor:
                await self.sr_predictor.cleanup()

            if self.ml_tactics:
                await self.ml_tactics.cleanup()

            self.logger.info("✅ Decision Policy cleanup completed")

        except Exception as e:
            self.logger.error(failed(f"❌ Decision Policy cleanup failed: {e}"))

class TacticsOrchestrator:
    """
    Main tactics orchestrator that coordinates all tactical components.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tactics orchestrator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("TacticsOrchestrator")

        # Configuration
        self.orchestrator_config = config.get("tactics_orchestrator", {})
        self.decision_interval = self.orchestrator_config.get("decision_interval", 30)

        # Component managers
        self.decision_policy: Optional[DecisionPolicy] = None
        self.position_monitor: Optional[PositionMonitor] = None
        self.position_closer: Optional[PositionCloser] = None
        self.order_manager: Optional[EnhancedOrderManager] = None
        self.position_strategy: Optional[PositionDivisionStrategy] = None

        # State tracking
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.decision_history: List[TradeDecision] = []
        self.orchestrator_task: Optional[asyncio.Task] = None
        self.is_running = False

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

            # Initialize position closer
            self.position_closer = PositionCloser(self.config)
            await self.position_closer.initialize()

            # Initialize order manager
            self.order_manager = EnhancedOrderManager(self.config)
            await self.order_manager.initialize()

            # Initialize position strategy
            self.position_strategy = PositionDivisionStrategy(self.config)
            await self.position_strategy.initialize()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(invalid("Invalid tactics orchestrator configuration"))
                return False

            self.logger.info("✅ Tactics Orchestrator initialized successfully")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Tactics Orchestrator initialization failed: {e}"))
            return False

    def refresh_step17_configuration(self, step17_results: dict[str, Any]) -> None:
        """
        Refresh configuration from step17 optimization results.
        This method is called automatically when step17 completes.
        
        Args:
            step17_results: Step17 optimization results
        """
        try:
            self.logger.info("🔄 Refreshing tactics orchestrator configuration from step17 results...")
            
            # Refresh decision policy
            if self.decision_policy:
                self.decision_policy.refresh_step17_configuration(step17_results)
            
            # Refresh position monitor (already has auto-refresh)
            if self.position_monitor:
                # Position monitor auto-refreshes from step12 results
                pass
            
            # Refresh position closer
            if self.position_closer:
                self.position_closer.refresh_step17_configuration(step17_results)
            
            # Refresh order manager if it has step17 refresh method
            if hasattr(self.order_manager, 'refresh_step17_configuration'):
                self.order_manager.refresh_step17_configuration(step17_results)
            
            # Refresh position strategy if it has step17 refresh method
            if hasattr(self.position_strategy, 'refresh_step17_configuration'):
                self.position_strategy.refresh_step17_configuration(step17_results)
            
            self.logger.info("✅ Tactics orchestrator configuration refreshed from step17 results")
            
        except Exception as e:
            self.logger.error(f"Error refreshing step17 configuration: {e}")

    def _validate_configuration(self) -> bool:
        """
        Validate tactics orchestrator configuration.

        Returns:
            bool: True if configuration is valid
        """
        try:
            if self.decision_interval <= 0:
                self.logger.error(invalid("Decision interval must be positive"))
                return False

            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Configuration validation failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactics orchestration start"
    )
    async def start_orchestration(self) -> bool:
        """
        Start tactics orchestration.

        Returns:
            bool: True if orchestration started successfully
        """
        try:
            if self.is_running:
                self.logger.warning(warning("Tactics orchestration already active"))
                return True

            self.is_running = True
            self.orchestrator_task = asyncio.create_task(self._orchestration_loop())

            self.logger.info("✅ Tactics orchestration started")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Failed to start tactics orchestration: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactics orchestration stop"
    )
    async def stop_orchestration(self) -> bool:
        """
        Stop tactics orchestration.

        Returns:
            bool: True if orchestration stopped successfully
        """
        try:
            if not self.is_running:
                self.logger.warning(warning("Tactics orchestration not active"))
                return True

            self.is_running = False

            if self.orchestrator_task:
                self.orchestrator_task.cancel()
                try:
                    await self.orchestrator_task
                except asyncio.CancelledError:
                    pass

            self.logger.info("✅ Tactics orchestration stopped")
            return True

        except Exception as e:
            self.logger.error(failed(f"❌ Failed to stop tactics orchestration: {e}"))
            return False

    async def _orchestration_loop(self) -> None:
        """
        Main orchestration loop that runs continuously.
        """
        try:
            while self.is_running:
                # Monitor positions
                await self._monitor_positions()

                # Generate decisions
                await self._generate_decisions()

                # Execute decisions
                await self._execute_decisions()

                # Wait for next cycle
                await asyncio.sleep(self.decision_interval)

        except asyncio.CancelledError:
            self.logger.info("Tactics orchestration loop cancelled")
        except Exception as e:
            self.logger.error(failed(f"❌ Error in orchestration loop: {e}"))

    async def _monitor_positions(self) -> None:
        """
        Monitor all active positions.
        """
        try:
            if not self.position_monitor:
                return

            # Get position assessments
            assessments = self.position_monitor.get_position_assessments()

            for assessment in assessments:
                # Check if position should be closed
                if assessment.position_action in [PositionAction.STOP_LOSS, PositionAction.FULL_CLOSE]:
                    await self._close_position(assessment)

        except Exception as e:
            self.logger.error(failed(f"❌ Error monitoring positions: {e}"))

    async def _generate_decisions(self) -> None:
        """
        Generate new trade decisions using multi-output predictions.
        """
        try:
            # Get market data and analyst predictions
            market_data = await self._get_market_data()
            analyst_predictions = await self._get_analyst_predictions()
            
            if not market_data or not analyst_predictions:
                return
            
            # Generate Tactician multi-output predictions
            tactician_predictions = await self._generate_tactician_predictions(
                market_data, analyst_predictions
            )
            
            if not tactician_predictions:
                return
            
            # Evaluate green light signal
            green_light_signal = tactician_predictions.get("green_light_signal", {})
            
            if green_light_signal.get("signal") == "GREEN_LIGHT":
                # Generate trade decision
                decision = await self._create_trade_decision(
                    market_data, analyst_predictions, tactician_predictions
                )
                
                if decision:
                    self.decision_history.append(decision)
                    self.logger.info(f"Generated trade decision: {decision.action} (confidence: {decision.confidence:.3f})")
            
            # Check for exit signals on existing positions
            await self._check_exit_signals(tactician_predictions)

        except Exception as e:
            self.logger.error(failed(f"❌ Error generating decisions: {e}"))

    async def _get_market_data(self) -> Optional[pd.DataFrame]:
        """
        Get current market data.
        
        Returns:
            pd.DataFrame: Market data or None
        """
        try:
            # This would get actual market data
            # For now, return None to indicate no data available
            return None
            
        except Exception as e:
            self.logger.error(failed(f"❌ Error getting market data: {e}"))
            return None

    async def _get_analyst_predictions(self) -> Optional[Dict[str, Any]]:
        """
        Get Analyst predictions.
        
        Returns:
            Dict: Analyst predictions or None
        """
        try:
            # This would get actual Analyst predictions
            # For now, return None to indicate no predictions available
            return None
            
        except Exception as e:
            self.logger.error(failed(f"❌ Error getting Analyst predictions: {e}"))
            return None

    async def _generate_tactician_predictions(
        self,
        market_data: pd.DataFrame,
        analyst_predictions: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate Tactician multi-output predictions.
        
        Args:
            market_data: Market data
            analyst_predictions: Analyst predictions
            
        Returns:
            Dict: Tactician predictions or None
        """
        try:
            if not self.ml_tactics:
                return None
            
            # Extract Analyst barriers
            analyst_barriers = self._extract_analyst_barriers(analyst_predictions)
            
            # Extract analyst confidence
            analyst_confidence = analyst_predictions.get("confidence", 0.5)
            
            # Generate multi-output predictions
            tactician_predictions = await self.ml_tactics.generate_multi_output_predictions(
                market_data=market_data,
                analyst_barriers=analyst_barriers,
                symbol="BTCUSDT",  # This would come from context
                timeframe="1m",
                analyst_confidence=analyst_confidence
            )
            
            return tactician_predictions
            
        except Exception as e:
            self.logger.error(failed(f"❌ Error generating Tactician predictions: {e}"))
            return None

    def _extract_analyst_barriers(self, analyst_predictions: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract barrier values from Analyst predictions.
        
        Args:
            analyst_predictions: Analyst predictions
            
        Returns:
            Dict: Barrier values
        """
        try:
            # Extract barriers from Analyst predictions
            # This is a simplified extraction - adjust based on actual Analyst output structure
            barriers = {
                "upper_barrier": analyst_predictions.get("upper_barrier", 0.02),
                "lower_barrier": analyst_predictions.get("lower_barrier", -0.01)
            }
            
            return barriers
            
        except Exception as e:
            self.logger.error(failed(f"❌ Error extracting Analyst barriers: {e}"))
            return {"upper_barrier": 0.02, "lower_barrier": -0.01}

    async def _create_trade_decision(
        self,
        market_data: pd.DataFrame,
        analyst_predictions: Dict[str, Any],
        tactician_predictions: Dict[str, Any]
    ) -> Optional[TradeDecision]:
        """
        Create trade decision based on predictions.
        
        Args:
            market_data: Market data
            analyst_predictions: Analyst predictions
            tactician_predictions: Tactician predictions
            
        Returns:
            TradeDecision: Trade decision or None
        """
        try:
            # Get combined confidence
            combined_confidence = tactician_predictions.get("combined_confidence", 0.5)
            
            # Determine action based on direction
            action = self._determine_action_from_predictions(tactician_predictions)
            
            if not action:
                return None
            
            # Calculate position size and leverage
            position_size = await self._calculate_position_size(tactician_predictions)
            leverage = await self._calculate_leverage(tactician_predictions)
            
            # Create decision
            decision = TradeDecision(
                action=action,
                confidence=combined_confidence,
                position_size=position_size,
                leverage=leverage,
                price=None,  # Would be set based on current market price
                metadata={
                    "analyst_predictions": analyst_predictions,
                    "tactician_predictions": tactician_predictions,
                    "green_light_signal": tactician_predictions.get("green_light_signal", {}),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(failed(f"❌ Error creating trade decision: {e}"))
            return None

    def _determine_action_from_predictions(self, tactician_predictions: Dict[str, Any]) -> Optional[str]:
        """
        Determine action from Tactician predictions.
        
        Args:
            tactician_predictions: Tactician predictions
            
        Returns:
            str: Action or None
        """
        try:
            # Check direction from 50% barrier prediction (more reliable)
            fifty_percent_pred = tactician_predictions.get("fifty_percent", {})
            direction = fifty_percent_pred.get("direction", "UP")
            
            if direction == "UP":
                return "BUY"
            elif direction == "DOWN":
                return "SELL"
            else:
                return None
                
        except Exception as e:
            self.logger.error(failed(f"❌ Error determining action: {e}"))
            return None

    async def _calculate_position_size(self, tactician_predictions: Dict[str, Any]) -> float:
        """
        Calculate position size based on Tactician predictions.
        
        Args:
            tactician_predictions: Tactician predictions
            
        Returns:
            float: Position size
        """
        try:
            if not self.position_sizer:
                return 0.0
            
            # Use combined confidence for position sizing
            combined_confidence = tactician_predictions.get("combined_confidence", 0.5)
            
            # Calculate position size using position sizer
            position_size = await self.position_sizer.calculate_position_size(
                ml_predictions=tactician_predictions,
                analyst_confidence=combined_confidence,
                tactician_confidence=combined_confidence
            )
            
            return position_size
            
        except Exception as e:
            self.logger.error(failed(f"❌ Error calculating position size: {e}"))
            return 0.0

    async def _calculate_leverage(self, tactician_predictions: Dict[str, Any]) -> float:
        """
        Calculate leverage based on Tactician predictions.
        
        Args:
            tactician_predictions: Tactician predictions
            
        Returns:
            float: Leverage
        """
        try:
            if not self.leverage_sizer:
                return 1.0
            
            # Use combined confidence for leverage calculation
            combined_confidence = tactician_predictions.get("combined_confidence", 0.5)
            
            # Calculate leverage using leverage sizer
            leverage = await self.leverage_sizer.calculate_leverage(
                ml_predictions=tactician_predictions,
                analyst_confidence=combined_confidence,
                tactician_confidence=combined_confidence
            )
            
            return leverage
            
        except Exception as e:
            self.logger.error(failed(f"❌ Error calculating leverage: {e}"))
            return 1.0

    async def _check_exit_signals(self, tactician_predictions: Dict[str, Any]) -> None:
        """
        Check for exit signals on existing positions.
        
        Args:
            tactician_predictions: Tactician predictions
        """
        try:
            if not self.ml_tactics:
                return
            
            # Get current positions
            active_positions = self.get_active_positions()
            
            for position_id, position in active_positions.items():
                # Evaluate exit signal for this position
                exit_signal = await self.ml_tactics.evaluate_exit_signal(
                    tactician_predictions,
                    position
                )
                
                if exit_signal.get("exit_signal") in ["EXIT", "PARTIAL_EXIT"]:
                    self.logger.info(f"Exit signal for position {position_id}: {exit_signal['exit_signal']}")
                    # This would trigger position closing logic
                    
        except Exception as e:
            self.logger.error(failed(f"❌ Error checking exit signals: {e}"))

    async def _execute_decisions(self) -> None:
        """
        Execute pending trade decisions.
        """
        try:
            # This would typically involve executing orders based on decisions
            # For now, this is a placeholder
            pass

        except Exception as e:
            self.logger.error(failed(f"❌ Error executing decisions: {e}"))

    async def _close_position(self, assessment: PositionAssessment) -> None:
        """
        Close a position based on assessment.

        Args:
            assessment: Position assessment
        """
        try:
            if not self.position_closer or not self.order_manager:
                return

            # Close position
            result = await self.position_closer.close_position(
                {
                    "position_id": assessment.position_id,
                    "symbol": assessment.symbol,
                    "side": assessment.side,
                    "entry_price": assessment.entry_price,
                    "current_price": assessment.current_price,
                    "quantity": assessment.current_quantity
                },
                assessment.action_reason
            )

            if result:
                self.logger.info(f"Closed position {assessment.position_id}: {result.get('pnl', 0):.4f} PnL")

        except Exception as e:
            self.logger.error(failed(f"❌ Error closing position: {e}"))

    def get_active_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active positions.

        Returns:
            Dict[str, Dict[str, Any]]: Active positions
        """
        return self.active_positions.copy()

    def get_decision_history(self, limit: Optional[int] = None) -> List[TradeDecision]:
        """
        Get decision history.

        Args:
            limit: Maximum number of decisions to return

        Returns:
            List[TradeDecision]: Decision history
        """
        try:
            if limit:
                return self.decision_history[-limit:]
            return self.decision_history.copy()

        except Exception as e:
            self.logger.error(failed(f"❌ Error getting decision history: {e}"))
            return []

    async def cleanup(self) -> None:
        """
        Cleanup resources.
        """
        try:
            self.logger.info("Cleaning up Tactics Orchestrator...")

            # Stop orchestration
            await self.stop_orchestration()

            # Cleanup component managers
            if self.decision_policy:
                await self.decision_policy.cleanup()

            if self.position_monitor:
                await self.position_monitor.cleanup()

            if self.position_closer:
                await self.position_closer.cleanup()

            if self.order_manager:
                await self.order_manager.cleanup()

            if self.position_strategy:
                await self.position_strategy.cleanup()

            self.logger.info("✅ Tactics Orchestrator cleanup completed")

        except Exception as e:
            self.logger.error(failed(f"❌ Tactics Orchestrator cleanup failed: {e}"))
