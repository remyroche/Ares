from __future__ import annotations
'\nTactics Orchestrator for coordinating all tactical components.\n'
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any
import pandas as pd
from copy import copy

@dataclass
class TradeDecision:
    action: str
    confidence: float
    position_size: float = 0.0
    leverage: float = 1.0
    price: float = None
    metadata: dict = None
import contextlib
from src.tactician.enhanced_order_manager import EnhancedOrderManager
from src.tactician.leverage_sizer import LeverageSizer
from src.tactician.position_closing import PositionCloser
from src.tactician.position_division_strategy import PositionDivisionStrategy
from src.tactician.position_monitor import PositionAction, PositionAssessment, PositionMonitor
from src.tactician.position_sizer import PositionSizer
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.utils.logger import system_logger
from src.utils.warning_symbols import failed, invalid

class DecisionPolicy:
    """
    Aggregates sizing, leverage, SR breakout, and ML signals into a unified TradeDecision.
    Provides audit-friendly metadata and metrics.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the decision policy.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('DecisionPolicy')
        self.policy_config = config.get('decision_policy', {})
        self.confidence_threshold = self.policy_config.get('confidence_threshold', 0.6)
        self.risk_threshold = self.policy_config.get('risk_threshold', 0.1)
        self.position_sizer: PositionSizer | None = None
        self.leverage_sizer: LeverageSizer | None = None
        self.sr_predictor: SRBreakoutPredictor | None = None
        self.ml_tactics: MLTacticsManager | None = None

    @handles_errors(exceptions=(ValueError, AttributeError), default_return=False, context='decision policy initialization')
    async def initialize(self) -> bool:
        """
        Initialize the decision policy.

        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info('Initializing Decision Policy...')
            await self._initialize_components()
            if not self._validate_configuration():
                self.logger.error(invalid('Invalid decision policy configuration'))
                return False
            self.logger.info('✅ Decision Policy initialized successfully')
            return True
        except Exception as e:
            self.logger.exception(failed(f'❌ Decision Policy initialization failed: {e}'))
            return False

    def refresh_step17_configuration(self, step17_results: dict[str, Any]) -> None:
        """
        Refresh configuration from step17 optimization results.
        This method is called automatically when step17 completes.

        Args:
            step17_results: Step17 optimization results
        """
        try:
            if 'decision_policy' in step17_results:
                policy_optimization = step17_results['decision_policy']
                self.confidence_threshold = policy_optimization.get('confidence_threshold', self.confidence_threshold)
                self.risk_threshold = policy_optimization.get('risk_threshold', self.risk_threshold)
            if self.position_sizer:
                self.position_sizer.refresh_step17_configuration(step17_results)
            if self.leverage_sizer:
                self.leverage_sizer.refresh_step17_configuration(step17_results)
            if self.ml_tactics:
                self.ml_tactics.refresh_step17_configuration(step17_results)
            self.logger.info('✅ Decision policy configuration refreshed from step17 results')
        except Exception as e:
            self.logger.exception(f'Error refreshing step17 configuration: {e}')

    async def _initialize_components(self) -> None:
        """Initialize all component managers."""
        try:
            self.position_sizer = PositionSizer(self.config)
            await self.position_sizer.initialize()
            self.leverage_sizer = LeverageSizer(self.config)
            await self.leverage_sizer.initialize()
            sr_config = self.config.copy()
            sr_config['sr_breakout_predictor'] = sr_config.get('sr_breakout_predictor', {})
            sr_config['sr_breakout_predictor']['use_optimized_params'] = True
            self.sr_predictor = SRBreakoutPredictor(sr_config)
            await self.sr_predictor.initialize()
            self.ml_tactics = MLTacticsManager(self.config)
            await self.ml_tactics.initialize()
        except Exception as e:
            self.logger.exception(failed(f'❌ Component initialization failed: {e}'))

    def _validate_configuration(self) -> bool:
        """
        Validate decision policy configuration.

        Returns:
            bool: True if configuration is valid
        """
        try:
            if not 0 <= self.confidence_threshold <= 1:
                self.logger.error(invalid('Confidence threshold must be between 0 and 1'))
                return False
            if not 0 <= self.risk_threshold <= 1:
                self.logger.error(invalid('Risk threshold must be between 0 and 1'))
                return False
            return True
        except Exception as e:
            self.logger.exception(failed(f'❌ Configuration validation failed: {e}'))
            return False

    @handles_errors(exceptions=(ValueError, AttributeError), default_return=None, context='trade decision generation')
    async def generate_decision(self, market_data: pd.DataFrame, analyst_confidence: float, tactician_confidence: float, position_data: dict[str, Any] | None=None) -> TradeDecision | None:
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
            self.logger.info('Generating trade decision...')
            sizing_decision = await self._get_sizing_decision(analyst_confidence, tactician_confidence)
            leverage_decision = await self._get_leverage_decision(analyst_confidence, tactician_confidence)
            sr_decision = await self._get_sr_decision(market_data)
            ml_decision = await self._get_ml_decision(market_data, analyst_confidence, tactician_confidence)
            decision = self._aggregate_decisions(sizing_decision, leverage_decision, sr_decision, ml_decision, analyst_confidence, tactician_confidence)
            if decision:
                self.logger.info(f'✅ Trade decision generated: {decision.action}')
            return decision
        except Exception as e:
            self.logger.exception(failed(f'❌ Trade decision generation failed: {e}'))
            return None

    async def _get_sizing_decision(self, analyst_confidence: float, tactician_confidence: float) -> dict[str, Any] | None:
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
            combined_confidence = (analyst_confidence + tactician_confidence) / 2
            position_size = await self.position_sizer.calculate_position_size(ml_predictions={}, analyst_confidence=analyst_confidence, tactician_confidence=tactician_confidence)
            return {'position_size': position_size, 'confidence': combined_confidence, 'source': 'position_sizer'}
        except Exception as e:
            self.logger.exception(failed(f'❌ Sizing decision failed: {e}'))
            return None

    async def _get_leverage_decision(self, analyst_confidence: float, tactician_confidence: float) -> dict[str, Any] | None:
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
            combined_confidence = (analyst_confidence + tactician_confidence) / 2
            leverage = await self.leverage_sizer.calculate_leverage(ml_predictions={}, analyst_confidence=analyst_confidence, tactician_confidence=tactician_confidence)
            return {'leverage': leverage, 'confidence': combined_confidence, 'source': 'leverage_sizer'}
        except Exception as e:
            self.logger.exception(failed(f'❌ Leverage decision failed: {e}'))
            return None

    async def _get_sr_decision(self, market_data: pd.DataFrame) -> dict[str, Any] | None:
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
            prediction = await self.sr_predictor.predict_breakout(market_data)
            if not prediction:
                return None
            return {'breakout_direction': prediction.get('direction'), 'breakout_confidence': prediction.get('confidence', 0.0), 'breakout_price': prediction.get('price'), 'outcome': prediction.get('outcome', 'consolidation'), 'sr_context': prediction.get('sr_context', {}), 'source': 'sr_predictor'}
        except Exception as e:
            self.logger.exception(failed(f'❌ SR decision failed: {e}'))
            return None

    async def _get_ml_decision(self, market_data: pd.DataFrame, analyst_confidence: float, tactician_confidence: float) -> dict[str, Any] | None:
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
            return await self.ml_tactics.get_tactics_decision(market_data, analyst_confidence, tactician_confidence)
        except Exception as e:
            self.logger.exception(failed(f'❌ ML decision failed: {e}'))
            return None

    def _aggregate_decisions(self, sizing_decision: dict[str, Any] | None, leverage_decision: dict[str, Any] | None, sr_decision: dict[str, Any] | None, ml_decision: dict[str, Any] | None, analyst_confidence: float, tactician_confidence: float) -> TradeDecision | None:
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
            combined_confidence = (analyst_confidence + tactician_confidence) / 2
            if combined_confidence < self.confidence_threshold:
                self.logger.info(f'Confidence {combined_confidence:.3f} below threshold {self.confidence_threshold}')
                return None
            action = self._determine_action(sizing_decision, leverage_decision, sr_decision, ml_decision)
            if not action:
                return None

            # Create trade decision
            return TradeDecision(
                action=action,
                confidence=combined_confidence,
                position_size=(
                    sizing_decision.get("position_size", 0.0)
                    if sizing_decision
                    else 0.0
                ),
                leverage=(
                    leverage_decision.get("leverage", 1.0) if leverage_decision else 1.0
                ),
                price=sr_decision.get("breakout_price") if sr_decision else None,
                metadata={
                    "analyst_confidence": analyst_confidence,
                    "tactician_confidence": tactician_confidence,
                    "sizing_decision": sizing_decision,
                    "leverage_decision": leverage_decision,
                    "sr_decision": sr_decision,
                    "ml_decision": ml_decision,
                    # Clarify active tactician path in production
                    # If FullyMigratedTactician is used upstream, this flag can be set in config
                    "active_tactician_path": self.config.get(
                        "active_tactician_path",
                        "ml_tactics_manager",  # default to specialist/ML tactics path
                    ),
                    "timestamp": datetime.now().isoformat(),
                },
            )

        except Exception as e:
            self.logger.exception(failed(f'❌ Decision aggregation failed: {e}'))
            return None

    def _determine_action(self, sizing_decision: dict[str, Any] | None, leverage_decision: dict[str, Any] | None, sr_decision: dict[str, Any] | None, ml_decision: dict[str, Any] | None) -> str | None:
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
            if not sizing_decision or not leverage_decision:
                return None
            position_size = sizing_decision.get('position_size', 0.0)
            if position_size <= 0:
                return None
            breakout_direction = sr_decision.get('breakout_direction') if sr_decision else None
            if breakout_direction == 'up':
                return 'BUY'
            if breakout_direction == 'down':
                return 'SELL'
            if ml_decision:
                return ml_decision.get('action')
            return None
        except Exception as e:
            self.logger.exception(failed(f'❌ Action determination failed: {e}'))
            return None

    async def cleanup(self) -> None:
        """
        Cleanup resources.
        """
        try:
            self.logger.info('Cleaning up Decision Policy...')
            if self.position_sizer:
                await self.position_sizer.cleanup()
            if self.leverage_sizer:
                await self.leverage_sizer.cleanup()
            if self.sr_predictor:
                await self.sr_predictor.cleanup()
            if self.ml_tactics:
                await self.ml_tactics.cleanup()
            self.logger.info('✅ Decision Policy cleanup completed')
        except Exception as e:
            self.logger.exception(failed(f'❌ Decision Policy cleanup failed: {e}'))

class TacticsOrchestrator:
    """
    Main tactics orchestrator that coordinates all tactical components.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the tactics orchestrator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('TacticsOrchestrator')
        self.orchestrator_config = config.get('tactics_orchestrator', {})
        self.decision_interval = self.orchestrator_config.get('decision_interval', 30)
        self.decision_policy: DecisionPolicy | None = None
        self.position_monitor: PositionMonitor | None = None
        self.position_closer: PositionCloser | None = None
        self.order_manager: EnhancedOrderManager | None = None
        self.position_strategy: PositionDivisionStrategy | None = None
        self.active_positions: dict[str, dict[str, Any]] = {}
        self.decision_history: list[TradeDecision] = []
        self.orchestrator_task: asyncio.Task | None = None
        self.is_running = False

    @handles_errors(exceptions=(ValueError, AttributeError), default_return=False, context='tactics orchestrator initialization')
    async def initialize(self) -> bool:
        """
        Initialize the tactics orchestrator.

        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info('Initializing Tactics Orchestrator...')
            self.decision_policy = DecisionPolicy(self.config)
            await self.decision_policy.initialize()
            self.position_monitor = PositionMonitor(self.config)
            await self.position_monitor.initialize()
            self.position_closer = PositionCloser(self.config)
            await self.position_closer.initialize()
            self.order_manager = EnhancedOrderManager(self.config)
            await self.order_manager.initialize()
            self.position_strategy = PositionDivisionStrategy(self.config)
            await self.position_strategy.initialize()
            if not self._validate_configuration():
                self.logger.error(invalid('Invalid tactics orchestrator configuration'))
                return False
            self.logger.info('✅ Tactics Orchestrator initialized successfully')
            return True
        except Exception as e:
            self.logger.exception(failed(f'❌ Tactics Orchestrator initialization failed: {e}'))
            return False

    def refresh_step17_configuration(self, step17_results: dict[str, Any]) -> None:
        """
        Refresh configuration from step17 optimization results.
        This method is called automatically when step17 completes.

        Args:
            step17_results: Step17 optimization results
        """
        try:
            self.logger.info('🔄 Refreshing tactics orchestrator configuration from step17 results...')
            if self.decision_policy:
                self.decision_policy.refresh_step17_configuration(step17_results)
            if self.position_monitor:
                pass
            if self.position_closer:
                self.position_closer.refresh_step17_configuration(step17_results)
            if hasattr(self.order_manager, 'refresh_step17_configuration'):
                self.order_manager.refresh_step17_configuration(step17_results)
            if hasattr(self.position_strategy, 'refresh_step17_configuration'):
                self.position_strategy.refresh_step17_configuration(step17_results)
            self.logger.info('✅ Tactics orchestrator configuration refreshed from step17 results')
        except Exception as e:
            self.logger.exception(f'Error refreshing step17 configuration: {e}')

    def _validate_configuration(self) -> bool:
        """
        Validate tactics orchestrator configuration.

        Returns:
            bool: True if configuration is valid
        """
        try:
            if self.decision_interval <= 0:
                self.logger.error(invalid('Decision interval must be positive'))
                return False
            return True
        except Exception as e:
            self.logger.exception(failed(f'❌ Configuration validation failed: {e}'))
            return False

    @handles_errors(exceptions=(ValueError, AttributeError), default_return=None, context='tactics orchestration start')
    async def start_orchestration(self) -> bool:
        """
        Start tactics orchestration.

        Returns:
            bool: True if orchestration started successfully
        """
        try:
            if self.is_running:
                self.logger.warning(warning('Tactics orchestration already active'))
                return True
            self.is_running = True
            self.orchestrator_task = asyncio.create_task(self._orchestration_loop())
            self.logger.info('✅ Tactics orchestration started')
            return True
        except Exception as e:
            self.logger.exception(failed(f'❌ Failed to start tactics orchestration: {e}'))
            return False

    @handles_errors(exceptions=(ValueError, AttributeError), default_return=None, context='tactics orchestration stop')
    async def stop_orchestration(self) -> bool:
        """
        Stop tactics orchestration.

        Returns:
            bool: True if orchestration stopped successfully
        """
        try:
            if not self.is_running:
                self.logger.warning(warning('Tactics orchestration not active'))
                return True
            self.is_running = False
            if self.orchestrator_task:
                self.orchestrator_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.orchestrator_task
            self.logger.info('✅ Tactics orchestration stopped')
            return True
        except Exception as e:
            self.logger.exception(failed(f'❌ Failed to stop tactics orchestration: {e}'))
            return False

    async def _orchestration_loop(self) -> None:
        """
        Main orchestration loop that runs continuously.
        """
        try:
            while self.is_running:
                await self._monitor_positions()
                await self._generate_decisions()
                await self._execute_decisions()
                await asyncio.sleep(self.decision_interval)
        except asyncio.CancelledError:
            self.logger.info('Tactics orchestration loop cancelled')
        except Exception as e:
            self.logger.exception(failed(f'❌ Error in orchestration loop: {e}'))

    async def _monitor_positions(self) -> None:
        """
        Monitor all active positions.
        """
        try:
            if not self.position_monitor:
                return
            assessments = self.position_monitor.get_position_assessments()
            for assessment in assessments:
                if assessment.position_action in [PositionAction.STOP_LOSS, PositionAction.FULL_CLOSE]:
                    await self._close_position(assessment)
        except Exception as e:
            self.logger.exception(failed(f'❌ Error monitoring positions: {e}'))

    async def _generate_decisions(self) -> None:
        """
        Generate new trade decisions using multi-output predictions.
        """
        try:
            market_data = await self._get_market_data()
            analyst_predictions = await self._get_analyst_predictions()
            if not market_data or not analyst_predictions:
                return
            tactician_predictions = await self._generate_tactician_predictions(market_data, analyst_predictions)
            if not tactician_predictions:
                return
            green_light_signal = tactician_predictions.get('green_light_signal', {})
            if green_light_signal.get('signal') == 'GREEN_LIGHT':
                decision = await self._create_trade_decision(market_data, analyst_predictions, tactician_predictions)
                if decision:
                    self.decision_history.append(decision)
                    self.logger.info(f'Generated trade decision: {decision.action} (confidence: {decision.confidence:.3f})')
            await self._check_exit_signals(tactician_predictions)
        except Exception as e:
            self.logger.exception(failed(f'❌ Error generating decisions: {e}'))

    async def _get_market_data(self) -> pd.DataFrame | None:
        """
        Get current market data.

        Returns:
            pd.DataFrame: Market data or None
        """
        try:
            return None
        except Exception as e:
            self.logger.exception(failed(f'❌ Error getting market data: {e}'))
            return None

    async def _get_analyst_predictions(self) -> dict[str, Any] | None:
        """
        Get Analyst predictions.

        Returns:
            Dict: Analyst predictions or None
        """
        try:
            return None
        except Exception as e:
            self.logger.exception(failed(f'❌ Error getting Analyst predictions: {e}'))
            return None

    async def _generate_tactician_predictions(self, market_data: pd.DataFrame, analyst_predictions: dict[str, Any]) -> dict[str, Any] | None:
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
            analyst_barriers = self._extract_analyst_barriers(analyst_predictions)
            analyst_confidence = analyst_predictions.get('confidence', 0.5)
            return await self.ml_tactics.generate_multi_output_predictions(market_data=market_data, analyst_barriers=analyst_barriers, symbol='BTCUSDT', timeframe='1m', analyst_confidence=analyst_confidence)
        except Exception as e:
            self.logger.exception(failed(f'❌ Error generating Tactician predictions: {e}'))
            return None

    def _extract_analyst_barriers(self, analyst_predictions: dict[str, Any]) -> dict[str, float]:
        """
        Extract barrier values from Analyst predictions.

        Args:
            analyst_predictions: Analyst predictions

        Returns:
            Dict: Barrier values
        """
        try:
            return {'upper_barrier': analyst_predictions.get('upper_barrier', 0.02), 'lower_barrier': analyst_predictions.get('lower_barrier', -0.01)}
        except Exception as e:
            self.logger.exception(failed(f'❌ Error extracting Analyst barriers: {e}'))
            return {'upper_barrier': 0.02, 'lower_barrier': -0.01}

    async def _create_trade_decision(self, market_data: pd.DataFrame, analyst_predictions: dict[str, Any], tactician_predictions: dict[str, Any]) -> TradeDecision | None:
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
            combined_confidence = tactician_predictions.get('combined_confidence', 0.5)
            action = self._determine_action_from_predictions(tactician_predictions)
            if not action:
                return None
            position_size = await self._calculate_position_size(tactician_predictions)
            leverage = await self._calculate_leverage(tactician_predictions)
            return TradeDecision(action=action, confidence=combined_confidence, position_size=position_size, leverage=leverage, price=None, metadata={'analyst_predictions': analyst_predictions, 'tactician_predictions': tactician_predictions, 'green_light_signal': tactician_predictions.get('green_light_signal', {}), 'timestamp': datetime.now().isoformat()})
        except Exception as e:
            self.logger.exception(failed(f'❌ Error creating trade decision: {e}'))
            return None

    def _determine_action_from_predictions(self, tactician_predictions: dict[str, Any]) -> str | None:
        """
        Determine action from Tactician predictions.

        Args:
            tactician_predictions: Tactician predictions

        Returns:
            str: Action or None
        """
        try:
            fifty_percent_pred = tactician_predictions.get('fifty_percent', {})
            direction = fifty_percent_pred.get('direction', 'UP')
            if direction == 'UP':
                return 'BUY'
            if direction == 'DOWN':
                return 'SELL'
            return None
        except Exception as e:
            self.logger.exception(failed(f'❌ Error determining action: {e}'))
            return None

    async def _calculate_position_size(self, tactician_predictions: dict[str, Any]) -> float:
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
            combined_confidence = tactician_predictions.get('combined_confidence', 0.5)
            return await self.position_sizer.calculate_position_size(ml_predictions=tactician_predictions, analyst_confidence=combined_confidence, tactician_confidence=combined_confidence)
        except Exception as e:
            self.logger.exception(failed(f'❌ Error calculating position size: {e}'))
            return 0.0

    async def _calculate_leverage(self, tactician_predictions: dict[str, Any]) -> float:
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
            combined_confidence = tactician_predictions.get('combined_confidence', 0.5)
            return await self.leverage_sizer.calculate_leverage(ml_predictions=tactician_predictions, analyst_confidence=combined_confidence, tactician_confidence=combined_confidence)
        except Exception as e:
            self.logger.exception(failed(f'❌ Error calculating leverage: {e}'))
            return 1.0

    async def _check_exit_signals(self, tactician_predictions: dict[str, Any]) -> None:
        """
        Check for exit signals on existing positions.

        Args:
            tactician_predictions: Tactician predictions
        """
        try:
            if not self.ml_tactics:
                return
            active_positions = self.get_active_positions()
            for position_id, position in active_positions.items():
                exit_signal = await self.ml_tactics.evaluate_exit_signal(tactician_predictions, position)
                if exit_signal.get('exit_signal') in ['EXIT', 'PARTIAL_EXIT']:
                    self.logger.info(f"Exit signal for position {position_id}: {exit_signal['exit_signal']}")
        except Exception as e:
            self.logger.exception(failed(f'❌ Error checking exit signals: {e}'))

    async def _execute_decisions(self) -> None:
        """
        Execute pending trade decisions.
        """
        try:
            pass
        except Exception as e:
            self.logger.exception(failed(f'❌ Error executing decisions: {e}'))

    async def _close_position(self, assessment: PositionAssessment) -> None:
        """
        Close a position based on assessment.

        Args:
            assessment: Position assessment
        """
        try:
            if not self.position_closer or not self.order_manager:
                return
            result = await self.position_closer.close_position({'position_id': assessment.position_id, 'symbol': assessment.symbol, 'side': assessment.side, 'entry_price': assessment.entry_price, 'current_price': assessment.current_price, 'quantity': assessment.current_quantity}, assessment.action_reason)
            if result:
                self.logger.info(f"Closed position {assessment.position_id}: {result.get('pnl', 0):.4f} PnL")
        except Exception as e:
            self.logger.exception(failed(f'❌ Error closing position: {e}'))

    def get_active_positions(self) -> dict[str, dict[str, Any]]:
        """
        Get all active positions.

        Returns:
            Dict[str, Dict[str, Any]]: Active positions
        """
        return self.active_positions.copy()

    def get_decision_history(self, limit: int | None=None) -> list[TradeDecision]:
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
            self.logger.exception(failed(f'❌ Error getting decision history: {e}'))
            return []

    async def cleanup(self) -> None:
        """
        Cleanup resources.
        """
        try:
            self.logger.info('Cleaning up Tactics Orchestrator...')
            await self.stop_orchestration()
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
            self.logger.info('✅ Tactics Orchestrator cleanup completed')
        except Exception as e:
            self.logger.exception(failed(f'❌ Tactics Orchestrator cleanup failed: {e}'))