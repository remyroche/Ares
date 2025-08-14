# src/tactician/tactics_orchestrator.py

from datetime import datetime
from typing import Any
import os
import pandas as pd

from exchange.factory import ExchangeFactory
from src.config.environment import get_exchange_name
from src.interfaces.base_interfaces import TradeDecision
from src.interfaces.event_bus import EventType
from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
    invalid,
    missing,
)


class DecisionPolicy:
    """
    Aggregates sizing, leverage, SR breakout, and ML signals into a unified TradeDecision.
    Provides audit-friendly metadata and metrics.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("DecisionPolicy")
        policy_cfg = config.get("decision_policy", {})

        # Threshold uses historical rule: tactician_confidence^2 * analyst_confidence > 0.216
        self.min_conf_product: float = float(policy_cfg.get("min_conf_product", 0.216))

        # Metrics
        self.metrics = {
            "decisions_total": 0,
            "decisions_approved": 0,
            "decisions_rejected": 0,
            "avg_decision_latency_ms": 0.0,
        }

    async def decide(
        self,
        context: dict[str, Any],
    ) -> tuple[TradeDecision | None, dict[str, Any]]:
        start_time = datetime.now()
        try:
            sizing = context.get("sizing_results", {})
            leverage = context.get("leverage_results", {})
            sr = context.get("sr_results", {})
            ml = context.get("ml_predictions", {})
            current_price = context.get("current_price", 0.0)
            symbol = context.get("symbol", "UNKNOWN")

            # Extract inputs
            final_size = float(sizing.get("final_position_size", 0.0) or 0.0)
            leverage_val = float(leverage.get("recommended_leverage", 1.0) or 1.0)
            stop_loss = float(leverage.get("stop_loss", 0.0) or 0.0)
            take_profit = float(leverage.get("take_profit", 0.0) or 0.0)
            directional_conf = float(
                ml.get("directional_confidence", {}).get("long", 0.5) or 0.5,
            )
            target_direction = context.get("target_direction", "long")

            # Confidence product gate (historical rule)
            analyst_confidence = float(context.get("analyst_confidence", 0.0) or 0.0)
            tactician_confidence = float(
                context.get("tactician_confidence", 0.0) or 0.0,
            )
            confidence_product = (tactician_confidence**2) * analyst_confidence

            # Optional extra signals for auditing (not gating)
            sr_score = float(sr.get("breakout_strength", 0.0) or 0.0)
            market_risk = float(
                context.get("market_health_analysis", {}).get("risk_score", 0.5) or 0.5,
            )
            strategist_risk = float(
                context.get("strategist_risk_parameters", {}).get("risk_score", 0.5)
                or 0.5,
            )
            risk_score = max(market_risk, strategist_risk)

            # Decision gates (only confidence product + positive size)
            # Standard SR/tactics gating layered on top of confidence product
            sr_reco = str(sr.get("recommendation", "")).upper()
            sr_strength = float(sr.get("confidence", 0.0) or 0.0)
            near_sr = bool(sr.get("sr_context", {}).get("is_near_level", False))
            min_sr_strength = float(self.config.get("tactics_orchestrator", {}).get("min_sr_strength", 0.6))

            approved = (confidence_product > self.min_conf_product) and (final_size > 0)
            if near_sr and sr_strength < min_sr_strength:
                approved = False

            # Require SR recommendation alignment when near SR
            if near_sr and approved and sr_reco not in ("BREAKOUT_LIKELY", "BOUNCE_LIKELY"):
                approved = False

            action = (
                ("OPEN_LONG" if target_direction == "long" else "OPEN_SHORT")
                if approved
                else "HOLD"
            )

            metadata = {
                "thresholds": {
                    "min_confidence_product": self.min_conf_product,
                    "formula": "tactician_confidence^2 * analyst_confidence",
                },
                "inputs": {
                    "tactician_confidence": tactician_confidence,
                    "analyst_confidence": analyst_confidence,
                    "confidence_product": confidence_product,
                    "directional_confidence": directional_conf,
                    "sr_breakout_strength": sr_score,
                    "sr_recommendation": sr_reco,
                    "sr_confidence": sr_strength,
                    "near_sr": near_sr,
                    "min_sr_strength": min_sr_strength,
                    "risk_score": risk_score,
                    "final_position_size": final_size,
                    "recommended_leverage": leverage_val,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "current_price": current_price,
                },
                "approved": approved,
            }

            decision: TradeDecision | None = None
            if approved:
                qty = max(final_size, 0.0)
                decision = TradeDecision(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    action=action,
                    quantity=qty,
                    price=current_price,
                    leverage=leverage_val,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence_product,
                    risk_score=risk_score,
                )

            # Metrics
            self.metrics["decisions_total"] += 1
            if approved:
                self.metrics["decisions_approved"] += 1
            else:
                self.metrics["decisions_rejected"] += 1
            latency_ms = max(
                (datetime.now() - start_time).total_seconds() * 1000.0,
                0.0,
            )
            prev_avg = self.metrics["avg_decision_latency_ms"]
            n = self.metrics["decisions_total"]
            self.metrics["avg_decision_latency_ms"] = (
                prev_avg + (latency_ms - prev_avg) / n
            )

            return decision, metadata
        except Exception as e:
            self.print(error("Decision error: {e}"))
            return None, {"error": str(e)}


class TacticsOrchestrator:
    """
    Tactics orchestrator responsible for coordinating all tactics modules.
    This module handles the high-level coordination between different tactics components.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize tactics orchestrator.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("TacticsOrchestrator")

        # Tactics state
        self.is_running: bool = False
        self.tactics_start_time: datetime | None = None
        self.tactics_results: dict[str, Any] = {}

        # Configuration
        self.tactics_config: dict[str, Any] = self.config.get(
            "tactics_orchestrator",
            {},
        )
        self.tactics_interval: int = self.tactics_config.get("tactics_interval", 30)
        self.max_history: int = self.tactics_config.get("max_history", 100)

        # Component managers (will be initialized)
        self.position_monitor = None
        self.sr_breakout_predictor = None
        self.position_sizer = None
        self.leverage_sizer = None
        self.position_division_strategy = None
        self.ml_tactics_manager = None
        self.decision_policy: DecisionPolicy | None = None
        self.event_bus = None
        self._rolling_infer = None

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid tactics orchestrator configuration"),
            AttributeError: (False, "Missing required tactics components"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="tactics orchestrator initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize tactics orchestrator and all component managers.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Tactics Orchestrator...")

            # Initialize component managers
            await self._initialize_component_managers()

            # Validate configuration
            if not self._validate_configuration():
                self.print(invalid("Invalid configuration for tactics orchestrator"))
                return False

            # Decision policy and event bus (if provided via container elsewhere)
            self.decision_policy = DecisionPolicy(self.config)
            # Initialize rolling inference if configured
            try:
                tm = (self.config or {}).get("TRANSITION_MODELING", {})
                if bool(tm.get("enabled", False)) and bool(tm.get("rolling_mode", False)):
                    from src.transition.rolling_inference import RollingMTInference
                    models_dir = str((tm.get("artifacts_dir", "checkpoints/transition_datasets")))
                    models_dir = os.path.join(models_dir, "models")
                    symbol = str(self.config.get("symbol", "UNKNOWN"))
                    timeframe = str(self.config.get("timeframe", "1m"))
                    self._rolling_infer = RollingMTInference(self.config, models_dir=models_dir, symbol=symbol, timeframe=timeframe)
                    _ = self._rolling_infer.load()
            except Exception:
                self._rolling_infer = None
            try:
                # Optional import to avoid hard DI coupling here
                from src.interfaces.event_bus import event_bus as global_event_bus

                self.event_bus = global_event_bus
            except Exception:
                self.event_bus = None

            self.logger.info("✅ Tactics Orchestrator initialized successfully")
            return True

        except Exception:
            self.print(failed("❌ Tactics Orchestrator initialization failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="component managers initialization",
    )
    async def _initialize_component_managers(self) -> None:
        """Initialize all component managers."""
        try:
            # Initialize position monitor
            from src.tactician.position_monitor import PositionMonitor

            self.position_monitor = PositionMonitor(self.config)
            await self.position_monitor.initialize()

            # Initialize SR breakout predictor
            from src.tactician.sr_breakout_predictor import SRBreakoutPredictor

            self.sr_breakout_predictor = SRBreakoutPredictor(self.config)
            await self.sr_breakout_predictor.initialize()

            # Initialize position sizer
            from src.tactician.position_sizer import PositionSizer

            self.position_sizer = PositionSizer(self.config)
            await self.position_sizer.initialize()

            # Initialize leverage sizer
            from src.tactician.leverage_sizer import LeverageSizer

            self.leverage_sizer = LeverageSizer(self.config)
            await self.leverage_sizer.initialize()

            # Initialize position division strategy
            from src.tactician.position_division_strategy import (
                PositionDivisionStrategy,
            )

            self.position_division_strategy = PositionDivisionStrategy(self.config)
            await self.position_division_strategy.initialize()

            # Initialize ML tactics manager
            from src.tactician.ml_tactics_manager import MLTacticsManager

            self.ml_tactics_manager = MLTacticsManager(self.config)
            await self.ml_tactics_manager.initialize()

            # Attach order manager to position monitor if available for trailing updates
            try:
                from src.tactician.enhanced_order_manager import EnhancedOrderManager

                self.order_manager = EnhancedOrderManager(self.config)
                await self.order_manager.initialize()
                # Wire a real exchange client when not paper trading
                try:
                    exchange_name = get_exchange_name().lower()
                    exchange_client = ExchangeFactory.get_exchange(exchange_name)
                    await self.order_manager.attach_exchange_client(exchange_client)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to attach exchange client to order manager: {e}",
                    )
                if hasattr(self.position_monitor, "order_manager"):
                    self.position_monitor.order_manager = self.order_manager
            except Exception as e:
                self.logger.warning(
                    f"Order manager initialization failed or unavailable: {e}",
                )

            self.logger.info("✅ All component managers initialized")

        except Exception:
            self.print(failed("❌ Failed to initialize component managers: {e}"))
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate tactics orchestrator configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate required configuration sections
            required_sections = [
                "tactics_orchestrator",
                "position_monitor",
                "sr_breakout_predictor",
            ]

            for section in required_sections:
                if section not in self.config:
                    self.logger.error(
                        f"Missing required configuration section: {section}",
                    )
                    return False

            # Validate tactics orchestrator specific settings
            if self.tactics_interval <= 0:
                self.print(invalid("Invalid tactics_interval configuration"))
                return False

            if self.max_history <= 0:
                self.print(invalid("Invalid max_history configuration"))
                return False

            return True

        except Exception:
            self.print(failed("Configuration validation failed: {e}"))
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid tactics parameters"),
            AttributeError: (False, "Missing tactics components"),
            KeyError: (False, "Missing required tactics data"),
        },
        default_return=False,
        context="tactics execution",
    )
    async def execute_tactics(
        self,
        tactics_input: dict[str, Any],
    ) -> bool:
        """
        Execute the complete tactics pipeline.

        Args:
            tactics_input: Tactics input parameters

        Returns:
            bool: True if tactics successful, False otherwise
        """
        try:
            self.logger.info("🚀 Starting tactics pipeline execution...")
            self.tactics_start_time = datetime.now()
            self.is_running = True

            # Validate tactics input
            if not self._validate_tactics_input(tactics_input):
                return False

            # Execute tactics pipeline
            success = await self._execute_tactics_pipeline(tactics_input)

            if success:
                self.logger.info("✅ Tactics pipeline completed successfully")
                await self._store_tactics_results(tactics_input)
            else:
                self.print(failed("❌ Tactics pipeline failed"))

            self.is_running = False
            return success

        except Exception:
            self.print(failed("❌ Tactics execution failed: {e}"))
            self.is_running = False
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="tactics input validation",
    )
    def _validate_tactics_input(self, tactics_input: dict[str, Any]) -> bool:
        """
        Validate tactics input parameters.

        Args:
            tactics_input: Tactics input parameters

        Returns:
            bool: True if input is valid, False otherwise
        """
        try:
            required_fields = ["symbol", "exchange", "timeframe", "current_price"]

            for field in required_fields:
                if field not in tactics_input:
                    self.print(missing("Missing required tactics input field: {field}"))
                    return False

            # Validate specific field values
            if tactics_input.get("current_price", 0) <= 0:
                self.print(invalid("Invalid current_price value"))
                return False

            return True

        except Exception:
            self.print(failed("Tactics input validation failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="tactics pipeline execution",
    )
    async def _execute_tactics_pipeline(
        self,
        tactics_input: dict[str, Any],
    ) -> bool:
        """
        Execute the main tactics pipeline.

        Args:
            tactics_input: Tactics input parameters

        Returns:
            bool: True if pipeline successful, False otherwise
        """
        try:
            self.logger.info("📊 Executing tactics pipeline...")

            # Step 1: Position Monitoring
            self.logger.info("🔧 Step 1: Position Monitoring")
            position_results = await self.position_monitor.monitor_positions(
                tactics_input,
            )
            if not position_results:
                self.print(failed("❌ Position monitoring failed"))
                return False

            # Gather context inputs
            analyst_market_health = tactics_input.get("market_health_analysis")
            strategist_risk_parameters = tactics_input.get("strategist_risk_parameters")
            ml_predictions = tactics_input.get("ml_predictions", {})
            current_price = tactics_input.get("current_price", 0.0)
            target_direction = tactics_input.get("target_direction", "long")
            analyst_confidence = tactics_input.get("analyst_confidence", 0.5)
            tactician_confidence = tactics_input.get("tactician_confidence", 0.5)

            # Step 2: SR Breakout Prediction
            self.logger.info("🔧 Step 2: SR Breakout Prediction")
            sr_results = await self.sr_breakout_predictor.predict_breakouts(
                tactics_input,
            )
            if not sr_results:
                self.print(failed("❌ SR breakout prediction failed"))
                return False

            # Step 3: Position Sizing
            self.logger.info("🔧 Step 3: Position Sizing")
            sizing_results = await self.position_sizer.calculate_position_size(
                ml_predictions=ml_predictions,
                current_price=current_price,
                analyst_confidence=analyst_confidence,
                tactician_confidence=tactician_confidence,
                market_health_analysis=analyst_market_health,
                strategist_risk_parameters=strategist_risk_parameters,
            )
            if not sizing_results:
                self.print(failed("❌ Position sizing failed"))
                return False

            # Step 4: Leverage Sizing
            self.logger.info("🔧 Step 4: Leverage Sizing")
            leverage_results = await self.leverage_sizer.calculate_leverage(
                ml_predictions=ml_predictions,
                liquidation_risk_analysis=(
                    tactics_input.get("liquidation_risk_analysis") or {}
                ),
                market_health_analysis=analyst_market_health,
                current_price=current_price,
                target_direction=target_direction,
                analyst_confidence=analyst_confidence,
                tactician_confidence=tactician_confidence,
            )
            if not leverage_results:
                self.print(failed("❌ Leverage sizing failed"))
                return False

            # Step 5: Position Division
            self.logger.info("🔧 Step 5: Position Division")
            if hasattr(self.position_division_strategy, "analyze_and_divide"):
                division_results = (
                    await self.position_division_strategy.analyze_and_divide(
                        tactics_input,
                        market_health_analysis=analyst_market_health,
                        strategist_risk_parameters=strategist_risk_parameters,
                        analyst_confidence=analyst_confidence,
                        tactician_confidence=tactician_confidence,
                    )
                )
            else:
                division_results = {"status": "skipped"}
            if not division_results:
                self.print(failed("❌ Position division failed"))
                return False

            # Step 6: ML Tactics
            self.logger.info("🔧 Step 6: ML Tactics")
            ml_results = await self.ml_tactics_manager.execute_ml_tactics(tactics_input)
            if not ml_results:
                self.print(failed("❌ ML tactics failed"))
                return False

            # Inject rolling inference predictions if available
            try:
                if self._rolling_infer is not None:
                    # Expect a combined_df in input; if absent, skip gracefully
                    combined_df = tactics_input.get("combined_features_frame")
                    if isinstance(combined_df, pd.DataFrame) and not combined_df.empty:
                        roll_pred = self._rolling_infer.predict_latest(combined_df)
                        ml_predictions.update({"rolling": roll_pred})
                        # Map to directional_confidence/target_direction hints
                        if roll_pred.get("ready"):
                            target_dir = "long" if roll_pred.get("side") == "long" else "short"
                            p_path = roll_pred.get("p_path_class", {})
                            fav = max(float(p_path.get("continuation", 0.0)), float(p_path.get("beginning_of_trend", 0.0)))
                            # Update tactician_confidence in tactics_input so it propagates to decision_context
                            tactician_confidence = max(float(tactician_confidence), float(fav))
                            tactics_input["tactician_confidence"] = tactician_confidence
                            tactics_input["target_direction"] = target_dir
                            ml_predictions["directional_confidence"] = {"long": float(roll_pred.get("p_direction_up_" + str(roll_pred.get("horizon", 0)), 0.5))}
                            # Expose exit flag to decision policy instead of overriding size
                            tactics_input["rolling_exit_flag"] = bool(roll_pred.get("exit_flag", False))
            except Exception:
                pass

            # Decision aggregation and event publishing
            decision_context = {
                **tactics_input,
                "sizing_results": sizing_results,
                "leverage_results": leverage_results,
                "sr_results": sr_results,
                "ml_predictions": ml_predictions,
            }
            decision, decision_meta = (
                await self.decision_policy.decide(decision_context)
                if self.decision_policy
                else (None, {})
            )
            if self.event_bus and decision is not None:
                await self.event_bus.publish(
                    EventType.TRADE_DECISION_MADE,
                    {
                        "decision": decision.__dict__,
                        "metadata": decision_meta,
                        "position_results": position_results,
                    },
                )

            # Store final results
            self.tactics_results = {
                "position_results": position_results,
                "sr_results": sr_results,
                "sizing_results": sizing_results,
                "leverage_results": leverage_results,
                "division_results": division_results,
                "ml_results": ml_results,
                "decision": decision.__dict__ if decision else None,
                "decision_metadata": decision_meta,
                "tactics_input": tactics_input,
                "execution_time": datetime.now() - self.tactics_start_time,
            }

            self.logger.info("✅ Tactics pipeline completed successfully")
            return True

        except Exception:
            self.print(failed("❌ Tactics pipeline execution failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactics results storage",
    )
    async def _store_tactics_results(self, tactics_input: dict[str, Any]) -> None:
        """
        Store tactics results for later retrieval.

        Args:
            tactics_input: Tactics input parameters
        """
        try:
            # Store results in a format that can be retrieved later
            results_key = f"{tactics_input['symbol']}_{tactics_input['exchange']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # This would typically store to database or file system
            self.logger.info(f"📁 Storing tactics results with key: {results_key}")

        except Exception:
            self.print(failed("❌ Failed to store tactics results: {e}"))

    def get_tactics_status(self) -> dict[str, Any]:
        """
        Get current tactics status.

        Returns:
            dict: Tactics status information
        """
        return {
            "is_running": self.is_running,
            "tactics_start_time": self.tactics_start_time,
            "tactics_duration": datetime.now() - self.tactics_start_time
            if self.tactics_start_time
            else None,
            "has_results": bool(self.tactics_results),
            "decision_metrics": (
                self.decision_policy.metrics if self.decision_policy else {}
            ),
        }

    def get_tactics_results(self) -> dict[str, Any]:
        """
        Get the latest tactics results.

        Returns:
            dict: Tactics results
        """
        return self.tactics_results.copy()

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="tactics orchestrator cleanup",
    )
    async def stop(self) -> None:
        """Stop the tactics orchestrator and cleanup resources."""
        try:
            self.logger.info("🛑 Stopping Tactics Orchestrator...")

            # Stop component managers
            if self.position_monitor:
                await self.position_monitor.stop()
            if self.sr_breakout_predictor:
                await self.sr_breakout_predictor.stop()
            if self.position_sizer:
                await self.position_sizer.stop()
            if self.leverage_sizer:
                await self.leverage_sizer.stop()
            if self.position_division_strategy:
                await self.position_division_strategy.stop()
            if self.ml_tactics_manager:
                await self.ml_tactics_manager.stop()

            self.is_running = False
            self.logger.info("✅ Tactics Orchestrator stopped successfully")

        except Exception:
            self.print(failed("❌ Failed to stop Tactics Orchestrator: {e}"))


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="tactics orchestrator setup",
)
async def setup_tactics_orchestrator(
    config: dict[str, Any] | None = None,
) -> TacticsOrchestrator | None:
    """
    Setup and return a configured TacticsOrchestrator instance.

    Args:
        config: Configuration dictionary

    Returns:
        TacticsOrchestrator: Configured tactics orchestrator instance
    """
    try:
        orchestrator = TacticsOrchestrator(config or {})
        if await orchestrator.initialize():
            return orchestrator
        return None
    except Exception:
        system_logger.exception(failed("Failed to setup Tactics Orchestrator: {e}"))
        return None
