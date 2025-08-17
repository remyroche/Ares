# src/tactician/tactics_orchestrator.py

from datetime import datetime
from typing import Any, Optional
import os
import pandas as pd
import asyncio

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
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
from src.tactician.position_monitor import PositionAction, PositionAssessment


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
            min_sr_strength = float(
                self.config.get("tactics_orchestrator", {}).get("min_sr_strength", 0.6)
            )

            approved = (confidence_product > self.min_conf_product) and (final_size > 0)
            if near_sr and sr_strength < min_sr_strength:
                approved = False

            # Require SR recommendation alignment when near SR
            if (
                near_sr
                and approved
                and sr_reco not in ("BREAKOUT_LIKELY", "BOUNCE_LIKELY")
            ):
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

        # Initialize SRBreakoutPredictor for S/R analysis
        self.sr_breakout_predictor = SRBreakoutPredictor(config)

        # Initialize Tactician specialized utilities
        from src.tactician.position_sizer import PositionSizer
        from src.tactician.leverage_sizer import LeverageSizer
        from src.tactician.position_monitor import PositionMonitor
        from src.tactician.position_closing import PositionCloser

        self.position_sizer = PositionSizer(config)
        self.leverage_sizer = LeverageSizer(config)
        self.position_monitor = PositionMonitor(config)
        self.position_closer = PositionCloser(config)

        # S/R opportunity handling
        self.sr_opportunity_config = config.get("sr_opportunity_handling", {})
        self.enable_sr_opportunities = self.sr_opportunity_config.get(
            "enable_sr_opportunities", True
        )
        self.sr_min_confidence = self.sr_opportunity_config.get(
            "sr_min_confidence", 0.7
        )

        # Dynamic risk management configuration
        self.dynamic_risk_config = config.get("dynamic_risk_management", {})
        self.enable_dynamic_sl_tp = self.dynamic_risk_config.get(
            "enable_dynamic_sl_tp", True
        )
        self.confidence_update_interval = self.dynamic_risk_config.get(
            "confidence_update_interval", 30
        )  # seconds
        self.sl_tp_adjustment_threshold = self.dynamic_risk_config.get(
            "sl_tp_adjustment_threshold", 0.05
        )  # 5% confidence change

        # S/R opportunity state
        self.active_sr_opportunities = {}
        self.sr_opportunity_history = []

        # Dynamic risk monitoring
        self.risk_monitoring_tasks = {}

    async def initialize(self) -> bool:
        """Initialize the Tactics Orchestrator."""
        try:
            self.logger.info("Initializing Tactics Orchestrator...")

            # Initialize SRBreakoutPredictor
            sr_init_success = await self.sr_breakout_predictor.initialize()
            if not sr_init_success:
                self.logger.warning("Failed to initialize SRBreakoutPredictor")

            # Initialize Tactician specialized utilities
            position_sizer_init = await self.position_sizer.initialize()
            leverage_sizer_init = await self.leverage_sizer.initialize()
            position_monitor_init = await self.position_monitor.initialize()
            position_closer_init = await self.position_closer.initialize()

            if not all(
                [
                    position_sizer_init,
                    leverage_sizer_init,
                    position_monitor_init,
                    position_closer_init,
                ]
            ):
                self.logger.error(
                    "Failed to initialize one or more Tactician utilities"
                )
                return False

            self.logger.info("✅ Tactics Orchestrator initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Tactics Orchestrator: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid tactics orchestrator configuration"),
            AttributeError: (False, "Missing required tactics components"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="tactics orchestrator initialization",
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
                            target_dir = (
                                "long" if roll_pred.get("side") == "long" else "short"
                            )
                            p_path = roll_pred.get("p_path_class", {})
                            fav = max(
                                float(p_path.get("continuation", 0.0)),
                                float(p_path.get("beginning_of_trend", 0.0)),
                            )
                            # Update tactician_confidence in tactics_input so it propagates to decision_context
                            tactician_confidence = max(
                                float(tactician_confidence), float(fav)
                            )
                            tactics_input["tactician_confidence"] = tactician_confidence
                            tactics_input["target_direction"] = target_dir
                            ml_predictions["directional_confidence"] = {
                                "long": float(
                                    roll_pred.get(
                                        "p_direction_up_"
                                        + str(roll_pred.get("horizon", 0)),
                                        0.5,
                                    )
                                )
                            }
                            # Expose exit flag to decision policy instead of overriding size
                            tactics_input["rolling_exit_flag"] = bool(
                                roll_pred.get("exit_flag", False)
                            )
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
        default_return=False,
        context="S/R opportunity alert processing",
    )
    async def process_sr_opportunity_alert(self, sr_alert: dict[str, Any]) -> bool:
        """
        Process S/R opportunity alert from the Analyst.

        Args:
            sr_alert: S/R opportunity alert with outcome prediction and recommendations

        Returns:
            bool: True if opportunity was processed successfully
        """
        try:
            if not self.enable_sr_opportunities:
                self.logger.debug("S/R opportunities disabled")
                return False

            if not sr_alert.get("opportunity_detected", False):
                return False

            # Extract alert information
            outcome = sr_alert.get("outcome", "consolidation")
            confidence = sr_alert.get("confidence", 0)
            current_price = sr_alert.get("current_price", 0)
            sr_context = sr_alert.get("sr_context", {})
            tactician_recommendations = sr_alert.get("tactician_recommendations", {})

            # Validate opportunity
            if confidence < self.sr_min_confidence:
                self.logger.debug(
                    f"S/R opportunity confidence too low: {confidence:.2f} < {self.sr_min_confidence}"
                )
                return False

            # Create opportunity ID
            opportunity_id = (
                f"sr_{outcome}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            )

            # Process opportunity based on outcome
            if outcome == "breakout":
                success = await self._process_breakout_opportunity(
                    opportunity_id, sr_alert, tactician_recommendations
                )
            elif outcome == "rebounce":
                success = await self._process_rebounce_opportunity(
                    opportunity_id, sr_alert, tactician_recommendations
                )
            elif outcome == "consolidation":
                success = await self._process_consolidation_opportunity(
                    opportunity_id, sr_alert, tactician_recommendations
                )
            else:
                self.logger.warning(f"Unknown S/R outcome: {outcome}")
                return False

            # Store opportunity in history
            self.sr_opportunity_history.append(
                {
                    "opportunity_id": opportunity_id,
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "outcome": outcome,
                    "confidence": confidence,
                    "current_price": current_price,
                    "processed": success,
                }
            )

            return success

        except Exception as e:
            self.logger.error(f"Error processing S/R opportunity alert: {e}")
            return False

    async def _process_breakout_opportunity(
        self,
        opportunity_id: str,
        sr_alert: dict[str, Any],
        recommendations: dict[str, Any],
    ) -> bool:
        """Process breakout opportunity."""
        try:
            self.logger.info(f"🚀 Processing BREAKOUT opportunity: {opportunity_id}")

            action = recommendations.get("action", "MONITOR")

            # Determine position direction based on S/R context
            sr_context = sr_alert.get("sr_context", {})
            current_price = sr_alert.get("current_price", 0)

            # Check if we're near resistance (breakout = short) or support (breakout = long)
            nearest_resistance = sr_context.get("nearest_resistance", current_price)
            nearest_support = sr_context.get("nearest_support", current_price)

            distance_to_resistance = (
                abs(current_price - nearest_resistance) / current_price
            )
            distance_to_support = abs(current_price - nearest_support) / current_price

            # Determine which level we're breaking out from
            if distance_to_resistance < distance_to_support:
                # Breaking out from resistance level = SHORT position
                position_direction = "SHORT"
                breakout_level = nearest_resistance
                self.logger.info(
                    f"   Breakout from RESISTANCE level: {breakout_level:.4f} -> SHORT position"
                )
            else:
                # Breaking out from support level = LONG position
                position_direction = "LONG"
                breakout_level = nearest_support
                self.logger.info(
                    f"   Breakout from SUPPORT level: {breakout_level:.4f} -> LONG position"
                )

            if action == "PREPARE":
                # Prepare for breakout entry using Tactician's risk management
                await self._prepare_breakout_entry(
                    opportunity_id, sr_alert, position_direction, breakout_level
                )
                return True
            elif action == "MONITOR":
                # Monitor for confirmation
                await self._monitor_breakout_confirmation(
                    opportunity_id, sr_alert, position_direction, breakout_level
                )
                return True
            else:
                self.logger.info(
                    f"Breakout opportunity action: {action} - no immediate action required"
                )
                return True

        except Exception as e:
            self.logger.error(f"Error processing breakout opportunity: {e}")
            return False

    async def _process_rebounce_opportunity(
        self,
        opportunity_id: str,
        sr_alert: dict[str, Any],
        recommendations: dict[str, Any],
    ) -> bool:
        """Process rebounce opportunity."""
        try:
            self.logger.info(f"📉 Processing REBOUNCE opportunity: {opportunity_id}")

            action = recommendations.get("action", "MONITOR")

            # Determine position direction based on S/R context
            sr_context = sr_alert.get("sr_context", {})
            current_price = sr_alert.get("current_price", 0)

            # Check if we're near resistance (rebounce = long) or support (rebounce = short)
            nearest_resistance = sr_context.get("nearest_resistance", current_price)
            nearest_support = sr_context.get("nearest_support", current_price)

            distance_to_resistance = (
                abs(current_price - nearest_resistance) / current_price
            )
            distance_to_support = abs(current_price - nearest_support) / current_price

            # Determine which level we're rebouncing from
            if distance_to_resistance < distance_to_support:
                # Rebouncing from resistance level = LONG position (price bounces down from resistance)
                position_direction = "LONG"
                rebounce_level = nearest_resistance
                self.logger.info(
                    f"   Rebounce from RESISTANCE level: {rebounce_level:.4f} -> LONG position"
                )
            else:
                # Rebouncing from support level = SHORT position (price bounces up from support)
                position_direction = "SHORT"
                rebounce_level = nearest_support
                self.logger.info(
                    f"   Rebounce from SUPPORT level: {rebounce_level:.4f} -> SHORT position"
                )

            if action == "PREPARE":
                # Prepare for rebounce entry using Tactician's risk management
                await self._prepare_rebounce_entry(
                    opportunity_id, sr_alert, position_direction, rebounce_level
                )
                return True
            elif action == "MONITOR":
                # Monitor for confirmation
                await self._monitor_rebounce_confirmation(
                    opportunity_id, sr_alert, position_direction, rebounce_level
                )
                return True
            else:
                self.logger.info(
                    f"Rebounce opportunity action: {action} - no immediate action required"
                )
                return True

        except Exception as e:
            self.logger.error(f"Error processing rebounce opportunity: {e}")
            return False

    async def _process_consolidation_opportunity(
        self,
        opportunity_id: str,
        sr_alert: dict[str, Any],
        recommendations: dict[str, Any],
    ) -> bool:
        """Process consolidation opportunity."""
        try:
            self.logger.info(
                f"📊 Processing CONSOLIDATION opportunity: {opportunity_id}"
            )

            action = recommendations.get("action", "MONITOR")

            if action == "MONITOR":
                # Monitor for range-bound trading opportunities
                await self._monitor_consolidation_range(opportunity_id, sr_alert)
                return True
            else:
                self.logger.info(
                    f"Consolidation opportunity action: {action} - no immediate action required"
                )
                return True

        except Exception as e:
            self.logger.error(f"Error processing consolidation opportunity: {e}")
            return False

    async def _prepare_breakout_entry(
        self,
        opportunity_id: str,
        sr_alert: dict[str, Any],
        position_direction: str,
        breakout_level: float,
    ) -> None:
        """Prepare for breakout entry using Tactician's specialized utilities."""
        try:
            current_price = sr_alert.get("current_price", 0)

            # Use Tactician's specialized utilities for risk management
            risk_params = await self._calculate_tactician_risk_parameters(
                sr_alert, position_direction
            )

            # Store opportunity details with comprehensive risk analysis
            self.active_sr_opportunities[opportunity_id] = {
                "type": "breakout",
                "entry_price": breakout_level,
                "stop_loss": risk_params["stop_loss"],
                "take_profit": risk_params["take_profit"],
                "position_size": risk_params["position_size"],
                "leverage": risk_params["leverage"],
                "confidence": risk_params["confidence"],
                "risk_score": risk_params["risk_score"],
                "position_direction": position_direction,
                "status": "preparing",
                "timestamp": pd.Timestamp.now().isoformat(),
                "sr_alert": sr_alert,
                "position_sizer_analysis": risk_params.get(
                    "position_sizer_analysis", {}
                ),
                "leverage_sizer_analysis": risk_params.get(
                    "leverage_sizer_analysis", {}
                ),
                "exit_levels_analysis": risk_params.get("exit_levels_analysis", {}),
            }

            # Start dynamic risk monitoring
            await self._start_dynamic_risk_monitoring(opportunity_id)

            self.logger.info(f"✅ Prepared breakout entry: {opportunity_id}")
            self.logger.info(
                f"   Entry: {breakout_level:.4f}, Stop: {risk_params['stop_loss']:.4f}, TP: {risk_params['take_profit']:.4f}"
            )
            self.logger.info(
                f"   Position Size: {risk_params['position_size']:.2%}, Leverage: {risk_params['leverage']:.1f}x"
            )
            self.logger.info(
                f"   Confidence: {risk_params['confidence']:.3f}, Risk Score: {risk_params['risk_score']:.3f}"
            )

        except Exception as e:
            self.logger.error(f"Error preparing breakout entry: {e}")

    async def _prepare_rebounce_entry(
        self,
        opportunity_id: str,
        sr_alert: dict[str, Any],
        position_direction: str,
        rebounce_level: float,
    ) -> None:
        """Prepare for rebounce entry using Tactician's specialized utilities."""
        try:
            current_price = sr_alert.get("current_price", 0)

            # Use Tactician's specialized utilities for risk management
            risk_params = await self._calculate_tactician_risk_parameters(
                sr_alert, position_direction
            )

            # Store opportunity details with comprehensive risk analysis
            self.active_sr_opportunities[opportunity_id] = {
                "type": "rebounce",
                "entry_price": rebounce_level,
                "stop_loss": risk_params["stop_loss"],
                "take_profit": risk_params["take_profit"],
                "position_size": risk_params["position_size"],
                "leverage": risk_params["leverage"],
                "confidence": risk_params["confidence"],
                "risk_score": risk_params["risk_score"],
                "position_direction": position_direction,
                "status": "preparing",
                "timestamp": pd.Timestamp.now().isoformat(),
                "sr_alert": sr_alert,
                "position_sizer_analysis": risk_params.get(
                    "position_sizer_analysis", {}
                ),
                "leverage_sizer_analysis": risk_params.get(
                    "leverage_sizer_analysis", {}
                ),
                "exit_levels_analysis": risk_params.get("exit_levels_analysis", {}),
            }

            # Start dynamic risk monitoring
            await self._start_dynamic_risk_monitoring(opportunity_id)

            self.logger.info(f"✅ Prepared rebounce entry: {opportunity_id}")
            self.logger.info(
                f"   Entry: {rebounce_level:.4f}, Stop: {risk_params['stop_loss']:.4f}, TP: {risk_params['take_profit']:.4f}"
            )
            self.logger.info(
                f"   Position Size: {risk_params['position_size']:.2%}, Leverage: {risk_params['leverage']:.1f}x"
            )
            self.logger.info(
                f"   Confidence: {risk_params['confidence']:.3f}, Risk Score: {risk_params['risk_score']:.3f}"
            )

        except Exception as e:
            self.logger.error(f"Error preparing rebounce entry: {e}")

    async def _monitor_breakout_confirmation(
        self,
        opportunity_id: str,
        sr_alert: dict[str, Any],
        position_direction: str,
        breakout_level: float,
    ) -> None:
        """Monitor for breakout confirmation."""
        try:
            self.logger.info(f"👀 Monitoring breakout confirmation: {opportunity_id}")

            # Store monitoring opportunity
            self.active_sr_opportunities[opportunity_id] = {
                "type": "breakout_monitor",
                "status": "monitoring",
                "timestamp": pd.Timestamp.now().isoformat(),
                "sr_alert": sr_alert,
                "position_direction": position_direction,
                "breakout_level": breakout_level,
            }

        except Exception as e:
            self.logger.error(f"Error monitoring breakout confirmation: {e}")

    async def _monitor_rebounce_confirmation(
        self,
        opportunity_id: str,
        sr_alert: dict[str, Any],
        position_direction: str,
        rebounce_level: float,
    ) -> None:
        """Monitor for rebounce confirmation."""
        try:
            self.logger.info(f"👀 Monitoring rebounce confirmation: {opportunity_id}")

            # Store monitoring opportunity
            self.active_sr_opportunities[opportunity_id] = {
                "type": "rebounce_monitor",
                "status": "monitoring",
                "timestamp": pd.Timestamp.now().isoformat(),
                "sr_alert": sr_alert,
                "position_direction": position_direction,
                "rebounce_level": rebounce_level,
            }

        except Exception as e:
            self.logger.error(f"Error monitoring rebounce confirmation: {e}")

    async def _monitor_consolidation_range(
        self, opportunity_id: str, sr_alert: dict[str, Any]
    ) -> None:
        """Monitor consolidation range for range-bound trading opportunities."""
        try:
            self.logger.info(f"📊 Monitoring consolidation range: {opportunity_id}")

            # Store monitoring opportunity
            self.active_sr_opportunities[opportunity_id] = {
                "type": "consolidation_monitor",
                "status": "monitoring",
                "timestamp": pd.Timestamp.now().isoformat(),
                "sr_alert": sr_alert,
            }

        except Exception as e:
            self.logger.error(f"Error monitoring consolidation range: {e}")

    async def execute_sr_opportunity(
        self, opportunity_id: str, current_price: float
    ) -> bool:
        """
        Execute S/R opportunity when conditions are met.

        Args:
            opportunity_id: ID of the opportunity to execute
            current_price: Current market price

        Returns:
            bool: True if execution was successful
        """
        try:
            if opportunity_id not in self.active_sr_opportunities:
                self.logger.warning(f"Opportunity {opportunity_id} not found")
                return False

            opportunity = self.active_sr_opportunities[opportunity_id]
            opportunity_type = opportunity.get("type", "")

            if opportunity_type == "breakout":
                return await self._execute_breakout_opportunity(
                    opportunity_id, current_price
                )
            elif opportunity_type == "rebounce":
                return await self._execute_rebounce_opportunity(
                    opportunity_id, current_price
                )
            else:
                self.logger.warning(f"Unknown opportunity type: {opportunity_type}")
                return False

        except Exception as e:
            self.logger.error(f"Error executing S/R opportunity: {e}")
            return False

    async def _execute_breakout_opportunity(
        self, opportunity_id: str, current_price: float
    ) -> bool:
        """Execute breakout opportunity."""
        try:
            opportunity = self.active_sr_opportunities[opportunity_id]
            entry_price = opportunity.get("entry_price", 0)
            position_direction = opportunity.get("position_direction", "LONG")

            # Check if price has reached entry level based on position direction
            if position_direction == "LONG":
                # For LONG positions (breakout from support), wait for price to go above entry
                if current_price >= entry_price:
                    self.logger.info(
                        f"🚀 Executing LONG breakout opportunity: {opportunity_id}"
                    )
                    success = await self._execute_trade(opportunity)

                    if success:
                        opportunity["status"] = "executed"
                        self.logger.info(
                            f"✅ LONG breakout opportunity executed: {opportunity_id}"
                        )
                    else:
                        self.logger.error(
                            f"❌ Failed to execute LONG breakout opportunity: {opportunity_id}"
                        )

                    return success
                else:
                    self.logger.debug(
                        f"LONG breakout opportunity {opportunity_id} waiting for entry price: {current_price:.4f} < {entry_price:.4f}"
                    )
                    return False
            else:  # SHORT position
                # For SHORT positions (breakout from resistance), wait for price to go below entry
                if current_price <= entry_price:
                    self.logger.info(
                        f"🚀 Executing SHORT breakout opportunity: {opportunity_id}"
                    )
                    success = await self._execute_trade(opportunity)

                    if success:
                        opportunity["status"] = "executed"
                        self.logger.info(
                            f"✅ SHORT breakout opportunity executed: {opportunity_id}"
                        )
                    else:
                        self.logger.error(
                            f"❌ Failed to execute SHORT breakout opportunity: {opportunity_id}"
                        )

                    return success
                else:
                    self.logger.debug(
                        f"SHORT breakout opportunity {opportunity_id} waiting for entry price: {current_price:.4f} > {entry_price:.4f}"
                    )
                    return False

        except Exception as e:
            self.logger.error(f"Error executing breakout opportunity: {e}")
            return False

    async def _execute_rebounce_opportunity(
        self, opportunity_id: str, current_price: float
    ) -> bool:
        """Execute rebounce opportunity."""
        try:
            opportunity = self.active_sr_opportunities[opportunity_id]
            entry_price = opportunity.get("entry_price", 0)
            position_direction = opportunity.get("position_direction", "SHORT")

            # Check if price has reached entry level based on position direction
            if position_direction == "LONG":
                # For LONG positions (rebounce from resistance), wait for price to go above entry
                if current_price >= entry_price:
                    self.logger.info(
                        f"📉 Executing LONG rebounce opportunity: {opportunity_id}"
                    )
                    success = await self._execute_trade(opportunity)

                    if success:
                        opportunity["status"] = "executed"
                        self.logger.info(
                            f"✅ LONG rebounce opportunity executed: {opportunity_id}"
                        )
                    else:
                        self.logger.error(
                            f"❌ Failed to execute LONG rebounce opportunity: {opportunity_id}"
                        )

                    return success
                else:
                    self.logger.debug(
                        f"LONG rebounce opportunity {opportunity_id} waiting for entry price: {current_price:.4f} < {entry_price:.4f}"
                    )
                    return False
            else:  # SHORT position
                # For SHORT positions (rebounce from support), wait for price to go below entry
                if current_price <= entry_price:
                    self.logger.info(
                        f"📉 Executing SHORT rebounce opportunity: {opportunity_id}"
                    )
                    success = await self._execute_trade(opportunity)

                    if success:
                        opportunity["status"] = "executed"
                        self.logger.info(
                            f"✅ SHORT rebounce opportunity executed: {opportunity_id}"
                        )
                    else:
                        self.logger.error(
                            f"❌ Failed to execute SHORT rebounce opportunity: {opportunity_id}"
                        )

                    return success
                else:
                    self.logger.debug(
                        f"SHORT rebounce opportunity {opportunity_id} waiting for entry price: {current_price:.4f} > {entry_price:.4f}"
                    )
                    return False

        except Exception as e:
            self.logger.error(f"Error executing rebounce opportunity: {e}")
            return False

    async def _execute_trade(self, opportunity: dict[str, Any]) -> bool:
        """
        Execute trade based on opportunity details.
        This would integrate with the actual order execution system.
        """
        try:
            # Extract trade parameters
            position_size = opportunity.get("position_size", 0.0)
            leverage = opportunity.get("leverage", 1.0)
            stop_loss = opportunity.get("stop_loss", 0)
            take_profit = opportunity.get("take_profit", 0)
            position_direction = opportunity.get("position_direction", "LONG")
            opportunity_type = opportunity.get("type", "unknown")

            # Log trade execution (placeholder for actual order execution)
            self.logger.info(
                f"📊 Executing {position_direction} {opportunity_type} trade:"
            )
            self.logger.info(f"   Position Direction: {position_direction}")
            self.logger.info(f"   Position Size: {position_size:.2%}")
            self.logger.info(f"   Leverage: {leverage}x")
            self.logger.info(f"   Stop Loss: {stop_loss:.4f}")
            self.logger.info(f"   Take Profit: {take_profit:.4f}")

            # TODO: Integrate with actual order execution system
            # await self.order_executor.execute_order({
            #     "direction": position_direction,
            #     "size": position_size,
            #     "leverage": leverage,
            #     "stop_loss": stop_loss,
            #     "take_profit": take_profit,
            #     "opportunity_type": opportunity_type
            # })

            return True

        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False

    async def _calculate_tactician_risk_parameters(
        self, sr_alert: dict[str, Any], position_direction: str
    ) -> dict[str, Any]:
        """
        Calculate risk parameters using Tactician's specialized utilities.
        This uses the existing PositionSizer, LeverageSizer, and PositionCloser.
        """
        try:
            confidence = sr_alert.get("confidence", 0)
            sr_context = sr_alert.get("sr_context", {})
            current_price = sr_alert.get("current_price", 0)

            # Use PositionSizer for position sizing
            position_size_result = await self.position_sizer.calculate_position_size(
                confidence=confidence,
                market_data=sr_context,
                position_direction=position_direction,
            )

            # Use LeverageSizer for leverage calculation
            leverage_result = await self.leverage_sizer.calculate_leverage(
                confidence=confidence,
                market_data=sr_context,
                position_direction=position_direction,
            )

            # Use PositionCloser for dynamic stop-loss and take-profit levels
            sl_tp_levels = await self.position_closer.calculate_dynamic_exit_levels(
                confidence=confidence,
                market_data=sr_context,
                current_price=current_price,
                position_direction=position_direction,
            )

            return {
                "position_size": position_size_result.get("position_size", 0.1),
                "leverage": leverage_result.get("leverage", 1.0),
                "stop_loss": sl_tp_levels.get("stop_loss", current_price * 0.99),
                "take_profit": sl_tp_levels.get("take_profit", current_price * 1.01),
                "confidence": confidence,
                "risk_score": sl_tp_levels.get("risk_score", 0.5),
                "position_sizer_analysis": position_size_result,
                "leverage_sizer_analysis": leverage_result,
                "exit_levels_analysis": sl_tp_levels,
            }

        except Exception as e:
            self.logger.error(f"Error calculating Tactician risk parameters: {e}")
            # Fallback to conservative values
            return {
                "position_size": 0.1,
                "leverage": 1.0,
                "stop_loss": current_price * 0.99,
                "take_profit": current_price * 1.01,
                "confidence": 0.5,
                "risk_score": 0.5,
            }

    async def _start_dynamic_risk_monitoring(self, opportunity_id: str) -> None:
        """
        Start dynamic risk monitoring for an opportunity using PositionMonitor.
        Continuously updates stop-loss and take-profit levels based on evolving confidence.
        """
        try:
            if opportunity_id in self.risk_monitoring_tasks:
                # Stop existing monitoring task
                self.risk_monitoring_tasks[opportunity_id].cancel()

            # Add opportunity to PositionMonitor for continuous monitoring
            opportunity = self.active_sr_opportunities[opportunity_id]

            # Register position with PositionMonitor
            await self.position_monitor.register_position(
                position_id=opportunity_id,
                position_data=opportunity,
                monitoring_config={
                    "monitoring_interval": self.confidence_update_interval,
                    "confidence_threshold": self.sl_tp_adjustment_threshold,
                    "enable_dynamic_sl_tp": self.enable_dynamic_sl_tp,
                },
            )

            # Create monitoring task that uses PositionMonitor
            monitoring_task = asyncio.create_task(
                self._monitor_dynamic_risk_with_position_monitor(opportunity_id)
            )
            self.risk_monitoring_tasks[opportunity_id] = monitoring_task

            self.logger.info(
                f"🔄 Started dynamic risk monitoring for opportunity: {opportunity_id}"
            )

        except Exception as e:
            self.logger.error(f"Error starting dynamic risk monitoring: {e}")

    async def _monitor_dynamic_risk_with_position_monitor(
        self, opportunity_id: str
    ) -> None:
        """
        Monitor and update dynamic risk parameters using PositionMonitor.
        """
        try:
            while True:
                await asyncio.sleep(self.confidence_update_interval)

                if opportunity_id not in self.active_sr_opportunities:
                    self.logger.info(
                        f"Opportunity {opportunity_id} no longer active, stopping monitoring"
                    )
                    break

                opportunity = self.active_sr_opportunities[opportunity_id]

                # Use PositionMonitor to assess the position
                assessment = await self.position_monitor.assess_position(opportunity_id)

                if assessment and assessment.recommended_action != PositionAction.STAY:
                    # Position needs adjustment based on PositionMonitor assessment
                    await self._handle_position_monitor_assessment(
                        opportunity_id, assessment
                    )

        except asyncio.CancelledError:
            self.logger.info(f"Dynamic risk monitoring cancelled for {opportunity_id}")
        except Exception as e:
            self.logger.error(
                f"Error in dynamic risk monitoring for {opportunity_id}: {e}"
            )

    async def _handle_position_monitor_assessment(
        self, opportunity_id: str, assessment: PositionAssessment
    ) -> None:
        """
        Handle PositionMonitor assessment and update opportunity accordingly.
        """
        try:
            opportunity = self.active_sr_opportunities[opportunity_id]

            # Update opportunity with new assessment
            opportunity.update(
                {
                    "confidence": assessment.current_confidence,
                    "confidence_change": assessment.confidence_change,
                    "risk_level": assessment.risk_level,
                    "last_assessment": assessment.assessment_timestamp.isoformat(),
                    "next_assessment": assessment.next_assessment.isoformat(),
                }
            )

            # Handle different actions
            if assessment.recommended_action == PositionAction.SCALE_DOWN:
                # Reduce position size
                await self._scale_down_position(opportunity_id, assessment)
            elif assessment.recommended_action == PositionAction.SCALE_UP:
                # Increase position size
                await self._scale_up_position(opportunity_id, assessment)
            elif assessment.recommended_action == PositionAction.TAKE_PROFIT:
                # Update take profit level
                await self._update_take_profit(opportunity_id, assessment)
            elif assessment.recommended_action == PositionAction.STOP_LOSS:
                # Update stop loss level
                await self._update_stop_loss(opportunity_id, assessment)
            elif assessment.recommended_action == PositionAction.EXIT:
                # Close position
                await self._close_position(opportunity_id, assessment)

            self.logger.info(
                f"🔄 PositionMonitor assessment for {opportunity_id}: {assessment.recommended_action.value}"
            )

        except Exception as e:
            self.logger.error(f"Error handling PositionMonitor assessment: {e}")

    async def _scale_down_position(
        self, opportunity_id: str, assessment: PositionAssessment
    ) -> None:
        """Scale down position based on PositionMonitor assessment."""
        try:
            # Use PositionCloser to calculate new position size
            new_size = await self.position_closer.calculate_scaled_position_size(
                current_size=assessment.current_confidence,
                confidence_change=assessment.confidence_change,
                action="scale_down",
            )

            # Update opportunity
            opportunity = self.active_sr_opportunities[opportunity_id]
            opportunity["position_size"] = new_size

            self.logger.info(
                f"📉 Scaled down position {opportunity_id} to {new_size:.2%}"
            )

        except Exception as e:
            self.logger.error(f"Error scaling down position: {e}")

    async def _scale_up_position(
        self, opportunity_id: str, assessment: PositionAssessment
    ) -> None:
        """Scale up position based on PositionMonitor assessment."""
        try:
            # Use PositionCloser to calculate new position size
            new_size = await self.position_closer.calculate_scaled_position_size(
                current_size=assessment.current_confidence,
                confidence_change=assessment.confidence_change,
                action="scale_up",
            )

            # Update opportunity
            opportunity = self.active_sr_opportunities[opportunity_id]
            opportunity["position_size"] = new_size

            self.logger.info(
                f"📈 Scaled up position {opportunity_id} to {new_size:.2%}"
            )

        except Exception as e:
            self.logger.error(f"Error scaling up position: {e}")

    async def _update_take_profit(
        self, opportunity_id: str, assessment: PositionAssessment
    ) -> None:
        """Update take profit level based on PositionMonitor assessment."""
        try:
            # Use PositionCloser to calculate new take profit level
            new_tp = await self.position_closer.calculate_dynamic_take_profit(
                current_price=assessment.current_confidence,
                confidence_change=assessment.confidence_change,
                market_conditions=assessment.market_conditions,
            )

            # Update opportunity
            opportunity = self.active_sr_opportunities[opportunity_id]
            opportunity["take_profit"] = new_tp

            self.logger.info(
                f"🎯 Updated take profit for {opportunity_id} to {new_tp:.4f}"
            )

        except Exception as e:
            self.logger.error(f"Error updating take profit: {e}")

    async def _update_stop_loss(
        self, opportunity_id: str, assessment: PositionAssessment
    ) -> None:
        """Update stop loss level based on PositionMonitor assessment."""
        try:
            # Use PositionCloser to calculate new stop loss level
            new_sl = await self.position_closer.calculate_dynamic_stop_loss(
                current_price=assessment.current_confidence,
                confidence_change=assessment.confidence_change,
                risk_level=assessment.risk_level,
            )

            # Update opportunity
            opportunity = self.active_sr_opportunities[opportunity_id]
            opportunity["stop_loss"] = new_sl

            self.logger.info(
                f"🛑 Updated stop loss for {opportunity_id} to {new_sl:.4f}"
            )

        except Exception as e:
            self.logger.error(f"Error updating stop loss: {e}")

    async def _close_position(
        self, opportunity_id: str, assessment: PositionAssessment
    ) -> None:
        """Close position based on PositionMonitor assessment."""
        try:
            # Use PositionCloser to close the position
            close_result = await self.position_closer.close_position(
                position_id=opportunity_id,
                reason=assessment.action_reason,
                assessment=assessment,
            )

            if close_result:
                # Remove from active opportunities
                if opportunity_id in self.active_sr_opportunities:
                    del self.active_sr_opportunities[opportunity_id]

                # Stop monitoring
                if opportunity_id in self.risk_monitoring_tasks:
                    self.risk_monitoring_tasks[opportunity_id].cancel()
                    del self.risk_monitoring_tasks[opportunity_id]

                self.logger.info(
                    f"✅ Closed position {opportunity_id} based on PositionMonitor assessment"
                )
            else:
                self.logger.error(f"❌ Failed to close position {opportunity_id}")

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")


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
