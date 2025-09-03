import asyncio
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict

import pandas as pd

from src.core.decorators import handles_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
import pandas as pd
    error,
    failed,
    initialization_error,
    invalid,
)
from src.utils.tracing import with_tracing_span
from src.utils.warning_symbols import error, failed, initialization_error, invalid

DEFAULT_SUPERVISOR_CONFIG = {
    "supervisor": {"supervision_interval": 60, "max_history": 100},
}

class CircuitBreaker:
    """Circuit breaker pattern for external services."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED = OPEN, HALF_OPEN

    @handles_errors(fallback=None)
    async def call(self, func: callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                msg = "Circuit breaker is OPEN"
                raise Exception(msg)

        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise

class OnlineLearningManager:
    """Manages online learning for model weighting based on performance."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("OnlineLearningManager")
        self.model_performances: dict[str, list[float]] = defaultdict(list)
        self.model_weights: dict[str, float] = {}
        self.learning_rate: float = config.get("learning_rate", 0.01)
        self.min_weight: float = config.get("min_weight", 0.1)
        self.max_weight: float = config.get("max_weight", 0.8)

    @handles_errors(fallback=None)
    async def update_model_performance(self, model_id: str, performance: float) -> None:
        """Update model performance and recalculate weights."""
        try:
            self.model_performances[model_id].append(performance)

            # Keep only recent performances (last 100)
            if len(self.model_performances[model_id]) > 100:
                self.model_performances[model_id] = self.model_performances[model_id][-100:]

            # Recalculate weights based on recent performance
            await self._recalculate_weights()

            self.logger.info(f"Updated performance for model {model_id}: {performance}")

        except Exception:
            self.print(error("Error updating model performance: {e}"))

    @handles_errors(fallback=None)
    async def _recalculate_weights(self) -> None:
        """Recalculate model weights based on performance."""
        try:
            if not self.model_performances:
                return

            # Calculate average performance for each model
            avg_performances = {}
            for model_id, performances in self.model_performances.items():
                if performances:
                    avg_performances[model_id] = sum(performances) / len(performances)

            if not avg_performances:
                return

            # Calculate total performance
            total_performance = sum(avg_performances.values())

            if total_performance == 0:
                # Equal weights if no performance
                equal_weight = 1.0 / len(avg_performances)
                self.model_weights = dict.fromkeys(avg_performances, equal_weight)
            else:
                # Weight based on performance
                for model_id, avg_perf in avg_performances.items():
                    weight = avg_perf / total_performance
                    # Apply min/max constraints
                    weight = max(self.min_weight, min(self.max_weight, weight))
                    self.model_weights[model_id] = weight

            self.logger.info(f"Recalculated weights: {self.model_weights}")

        except Exception:
            self.print(error("Error recalculating weights: {e}"))

    def get_model_weights(self) -> dict[str, float]:
        """Get current model weights."""
        return self.model_weights.copy()

    def get_model_performances(self) -> dict[str, list[float]]:
        """Get model performance history."""
        return {k: v.copy() for k, v in self.model_performances.items()}

class Supervisor:
    """"
    System-Level Supervisor component responsible for:
    - System Health Monitoring: Monitor all component health and performance
    - Circuit Breaker Management: Handle failures and recovery across all components
    - Component Coordination: Orchestrate communication between components
    - Portfolio-Level Risk Management: Global portfolio guards and kill-switches (excluding position sizing)
    - Performance Tracking: System-wide performance monitoring and reporting
    - Online Learning: Model weighting based on system performance
    - Recovery Management: Automatic recovery and fallback mechanisms
    """"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("Supervisor")
        self.is_running: bool = False
        self.status: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.supervisor_config: dict[str, Any] = self.config.get("supervisor", {})
        self.supervision_interval: int = self.supervisor_config.get(
            "supervision_interval",
            60,
        )
        self.max_history: int = self.supervisor_config.get("max_history", 100)
        self.supervision_results: dict[str, Any] = {}
        self.components: dict[str, Any] = {}

        # Advanced error handling and recovery
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.recovery_attempts: dict[str, int] = defaultdict(int)
        self.max_recovery_attempts: int = self.supervisor_config.get(
            "max_recovery_attempts",
            3,
        )
        self.recovery_cooldown: int = self.supervisor_config.get(
            "recovery_cooldown",
            300,
        )  # 5 minutes
        self.last_recovery_attempt: dict[str, float] = {}

        # Online learning for model weighting
        self.online_learning = OnlineLearningManager(
            self.supervisor_config.get("online_learning", {}),
        )

        # Enhanced prediction service for ML model integration
        self.enhanced_prediction_service = None
        self.is_initialized: bool = False
        self.enhanced_prediction_service_config = self.supervisor_config.get("enhanced_prediction_service", {})
        self.entry_threshold: float = self.enhanced_prediction_service_config.get("entry_threshold", 0.7)
        self.max_confidence_threshold: float = self.enhanced_prediction_service_config.get(
            "max_confidence_threshold", 0.9
        )

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid supervisor configuration"),
            AttributeError: (False, "Missing required supervisor parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="supervisor initialization",
    )
    async def initialize(self) -> bool:
        try:
            self.logger.info("Initializing Supervisor...")
            await self._load_supervisor_configuration()
            if not self._validate_configuration():
                self.print(invalid("Invalid configuration for supervisor"))
                return False
            await self._initialize_components()
            await self._setup_circuit_breakers()
            await self._setup_online_learning()
            await self._setup_component_monitors()
            self.logger.info("✅ Supervisor initialization completed successfully")
            self.is_initialized = True
            return True
        except Exception:
            self.print(failed("❌ Supervisor initialization failed: {e}"))
            return False

    @handles_errors(fallback=None)
    async def _load_supervisor_configuration(self) -> None:
        try:
            self.supervisor_config.setdefault("supervision_interval", 60)
            self.supervisor_config.setdefault("max_history", 100)
            self.supervisor_config.setdefault("max_recovery_attempts", 3)
            self.supervisor_config.setdefault("recovery_cooldown", 300)
            self.supervision_interval = self.supervisor_config["supervision_interval"]
            self.max_history = self.supervisor_config["max_history"]
            self.max_recovery_attempts = self.supervisor_config["max_recovery_attempts"]
            self.recovery_cooldown = self.supervisor_config["recovery_cooldown"]
            self.logger.info("Supervisor configuration loaded successfully")
        except Exception:
            self.print(error("Error loading supervisor configuration: {e}"))

    @handles_errors(fallback=False)
    def _validate_configuration(self) -> bool:
        try:
            if self.supervision_interval <= 0:
                self.print(invalid("Invalid supervision interval"))
                return False
            if self.max_history <= 0:
                self.print(invalid("Invalid max history"))
                return False
            if self.max_recovery_attempts <= 0:
                self.print(invalid("Invalid max recovery attempts"))
                return False
            if self.recovery_cooldown <= 0:
                self.print(invalid("Invalid recovery cooldown"))
                return False
            self.logger.info("Configuration validation successful")
            return True
        except Exception:
            self.print(error("Error validating configuration: {e}"))
            return False

    @handles_errors(fallback=None)
    async def _initialize_components(self) -> None:
        try:
            # Initialize critical components with updated structure
            self.components = {
                "database": None,
                "exchange": None,
                "analyst": None,
                "strategist": None,
                "tactician": None,
                "sentinel": None,
                "paper_trader": None,
                "performance_monitor": None,
                "enhanced_training_manager": None,
                "model_manager": None,
                "state_manager": None,
            }

            # Initialize enhanced prediction service
            await self._initialize_enhanced_prediction_service()

            self.logger.info("Components initialized successfully")
        except Exception:
            self.print(initialization_error("Error initializing components: {e}"))

    @handles_errors(fallback=None)
    async def _setup_circuit_breakers(self) -> None:
        """Setup circuit breakers for critical services."""
        try:
            # Setup circuit breakers for external services
            self.circuit_breakers = {
                "exchange": CircuitBreaker(failure_threshold=5, timeout=60),
                "database": CircuitBreaker(failure_threshold=3, timeout=30),
                "analyst": CircuitBreaker(failure_threshold=3, timeout=30),
                "strategist": CircuitBreaker(failure_threshold=3, timeout=30),
                "tactician": CircuitBreaker(failure_threshold=3, timeout=30),
                "enhanced_training_manager": CircuitBreaker(
                    failure_threshold=3,
                    timeout=60,
                ),
            }

            self.logger.info("Circuit breakers setup complete")
        except Exception:
            self.print(error("Error setting up circuit breakers: {e}"))

    @handles_errors(fallback=None)
    async def _setup_online_learning(self) -> None:
        """Setup online learning for model weighting."""
        try:
            # Initialize online learning with default configuration
            online_learning_config = self.supervisor_config.get("online_learning", {})
            self.online_learning = OnlineLearningManager(online_learning_config)

            self.logger.info("Online learning setup complete")
        except Exception:
            self.print(error("Error setting up online learning: {e}"))

    @handles_errors(fallback=None)
    async def _setup_component_monitors(self) -> None:
        """Setup component-specific monitoring."""
        try:
            # Initialize component monitors with default states
            for monitors in self.component_monitors.values():
                for monitor_name in monitors:
                    monitors[monitor_name] = False

            self.logger.info("Component monitors setup complete")
        except Exception:
            self.print(error("Error setting up component monitors: {e}"))

    @handles_errors(fallback=False)
    async def _initialize_enhanced_prediction_service(self) -> bool:
        """Initialize the enhanced prediction service."""
        try:
            from src.supervisor.enhanced_prediction_service import EnhancedPredictionService

            self.enhanced_prediction_service = EnhancedPredictionService(self.config)
            success = await self.enhanced_prediction_service.initialize()

            if success:
                self.logger.info("✅ Enhanced Prediction Service initialized successfully")
            else:
                self.logger.warning("⚠️ Enhanced Prediction Service initialization failed")

            return success

        except Exception as e:
            self.logger.error(f"❌ Error initializing Enhanced Prediction Service: {e}")
            return False

    @handles_errors
    @with_tracing_span("get_analyst_predictions")
    async def get_analyst_predictions(
        self,
        market_data: pd.DataFrame,
        regime_info: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str = "1h",
    ) -> Dict[str, Any]:
        """"
        Get Analyst predictions using calibrated confidence scores from ML models.

        The Analyst decides if we enter a position based on calibrated confidence scores.
        """"
        try:
            if not self.is_initialized:
                self.logger.error(error("❌ Supervisor not initialized"))
                return {}

            # Step 1: Get calibrated confidence scores from Enhanced Prediction Service
            calibrated_confidence = await self.enhanced_prediction_service.get_calibrated_confidence_scores(
                market_data, regime_info, symbol, exchange
            )

            # Step 2: Analyst decides if we enter a position using Analyst models
            analyst_decision = await self._analyst_decide_position_entry(
                market_data,
                regime_info,
                calibrated_confidence["analyst_models"],
                symbol,
                exchange,
            )

            return {
                "calibrated_confidence_scores": calibrated_confidence,
                "analyst_decision": analyst_decision,
                "timestamp": datetime.now().isoformat(),
            }

        except ValueError as e:
            # Enhanced Prediction Service failed - no calibrated confidence
            self.logger.error(error(f"❌ Enhanced Prediction Service failed: {e}"))
            return {
                "error": str(e),
                "analyst_decision": {
                    "should_enter_position": False,
                    "reason": "no_calibrated_confidence",
                },
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(error(f"❌ Error getting analyst predictions: {e}"))
            return {}

    @handles_errors
    @with_tracing_span("get_tactician_predictions")
    async def get_tactician_predictions(
        self,
        market_data: pd.DataFrame,
        regime_info: Dict[str, Any],
        analyst_signals: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str = "1m",
    ) -> Dict[str, Any]:
        """"
        Get Tactician predictions using calibrated confidence scores from ML models.

        The Tactician decides when, how much, and with what leverage based on calibrated confidence scores.
        Must agree with Analyst on trade direction.
        """"
        try:
            if not self.is_initialized:
                self.logger.error(error("❌ Supervisor not initialized"))
                return {}

            # Step 1: Get calibrated confidence scores from Enhanced Prediction Service
            calibrated_confidence = await self.enhanced_prediction_service.get_calibrated_confidence_scores(
                market_data, regime_info, symbol, exchange
            )

            # Step 2: Tactician decides execution parameters using Tactician models
            tactician_decision = await self._tactician_calculate_execution_parameters(
                market_data,
                analyst_signals,
                calibrated_confidence["tactician_models"],
                symbol,
                exchange,
            )

            return {
                "calibrated_confidence_scores": calibrated_confidence,
                "tactician_decision": tactician_decision,
                "timestamp": datetime.now().isoformat(),
            }

        except ValueError as e:
            # Enhanced Prediction Service failed - no calibrated confidence
            self.logger.error(error(f"❌ Enhanced Prediction Service failed: {e}"))
            return {
                "error": str(e),
                "tactician_decision": {
                    "should_execute": False,
                    "reason": "no_calibrated_confidence",
                },
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(error(f"❌ Error getting tactician predictions: {e}"))
            return {}

    @handles_errors
    @with_tracing_span("analyst_decide_position_entry")
    async def _analyst_decide_position_entry(
        self,
        market_data: pd.DataFrame,
        regime_info: Dict[str, Any],
        analyst_confidence_scores: Dict[str, float],
        symbol: str,
        exchange: str,
    ) -> Dict[str, Any]:
        """"
        Analyst decides if we enter a position and determines trade direction based on Analyst ML models.
        """"
        try:
            # Calculate aggregate Analyst confidence
            if not analyst_confidence_scores:
                return {
                    "should_enter_position": False,
                    "trade_direction": "neutral",
                    "entry_confidence": 0.0,
                    "max_confidence": 0.0,
                    "individual_confidences": {},
                    "entry_reason": "no_analyst_confidence",
                }

            avg_confidence = sum(analyst_confidence_scores.values()) / len(analyst_confidence_scores)
            max_confidence = max(analyst_confidence_scores.values())

            # Determine trade direction from Analyst models
            trade_direction = self._analyst_determine_trade_direction(analyst_confidence_scores, market_data)

            # Decision logic
            should_enter = (
                avg_confidence > self.enhanced_prediction_service.entry_threshold
                and max_confidence > self.enhanced_prediction_service.max_confidence_threshold
                and trade_direction != "neutral"
            )

            return {
                "should_enter_position": should_enter,
                "trade_direction": trade_direction,
                "entry_confidence": avg_confidence,
                "max_confidence": max_confidence,
                "individual_confidences": analyst_confidence_scores,
                "entry_reason": ("high_confidence" if should_enter else "low_confidence_or_neutral"),
            }

        except Exception as e:
            self.logger.error(error(f"❌ Error in analyst position decision: {e}"))
            return {
                "should_enter_position": False,
                "trade_direction": "neutral",
                "entry_confidence": 0.0,
                "max_confidence": 0.0,
                "individual_confidences": {},
                "entry_reason": "error",
                "error": str(e),
            }

    def _analyst_determine_trade_direction(self, confidence_scores: Dict[str, float], market_data: pd.DataFrame) -> str:
        """Determine trade direction based on Analyst model confidences."""
        try:
            # Logic to determine if models suggest long, short, or neutral
            # This would be based on the specific Analyst model outputs
            bullish_confidence = sum(
                conf for name, conf in confidence_scores.items() if "bullish" in name.lower() or "long" in name.lower()
            )
            bearish_confidence = sum(
                conf for name, conf in confidence_scores.items() if "bearish" in name.lower() or "short" in name.lower()
            )

            # If no directional models, use overall confidence pattern
            if bullish_confidence == 0 and bearish_confidence == 0:
                # Use price momentum as fallback
                if len(market_data) >= 2:
                    price_change = (market_data["close"].iloc[-1] - market_data["close"].iloc[-2]) / market_data[
                        "close"
                    ].iloc[-2]
                    if abs(price_change) > 0.001:  # 0.1% threshold
                        return "long" if price_change > 0 else "short"
                return "neutral"

            # Determine direction based on confidence
            if bullish_confidence > bearish_confidence and bullish_confidence > 0.6:
                return "long"
            elif bearish_confidence > bullish_confidence and bearish_confidence > 0.6:
                return "short"
            else:
                return "neutral"

        except Exception as e:
            self.logger.error(error(f"❌ Error determining trade direction: {e}"))
            return "neutral"

    @handles_errors
    @with_tracing_span("tactician_calculate_execution_parameters")
    async def _tactician_calculate_execution_parameters(
        self,
        market_data: pd.DataFrame,
        analyst_signals: Dict[str, Any],
        tactician_confidence_scores: Dict[str, float],
        symbol: str,
        exchange: str,
    ) -> Dict[str, Any]:
        """"
        Tactician decides when, how much, and what leverage based on Tactician ML models.
        Must agree with Analyst on trade direction.
        Enhanced with high precision triple barrier completion.
        """"
        try:
            # Import enhanced execution manager
            from src.tactician.enhanced_execution_manager import EnhancedExecutionManager
        except Exception as e:
            pass  # TODO: Handle exception properly
import copy
import numpy as np
# Initialize enhanced execution manager
            enhanced_manager = EnhancedExecutionManager(self.config)

            # Check if Analyst wants to enter
            analyst_decision = analyst_signals.get("analyst_decision", {})
            if not analyst_decision.get("should_enter_position", False):
                return {"should_execute": False, "reason": "analyst_no_entry"}

            # Calculate average tactician confidence
            if not tactician_confidence_scores:
                return {"should_execute": False, "reason": "no_tactician_confidence"}

            avg_tactician_confidence = sum(tactician_confidence_scores.values()) / len(tactician_confidence_scores)

            # Get current price for execution calculations
            current_price = market_data["close"].iloc[-1] if not market_data.empty else 0.0

            # Use enhanced execution manager for high precision parameters
            execution_params = enhanced_manager.calculate_execution_parameters(
                market_data=market_data,
                analyst_signal=analyst_decision,
                tactician_confidence=avg_tactician_confidence,
                current_price=current_price,
            )

            if not execution_params.get("should_execute", False):
                return execution_params

            # Add additional metadata
            execution_params.update(
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "execution_manager": "enhanced_precision",
                    "barrier_strategy": "fraction_based",
                    "barrier_types": ["upper_barrier", "lower_barrier"],
                    "timeframes": ["1m", "5m"],
                }
            )

            self.logger.info("🎯 Enhanced Tactician Execution Parameters:")
            self.logger.info(f"   Symbol: {symbol}")
            self.logger.info(f"   Direction: {execution_params.get('trade_direction', 'unknown')}")
            self.logger.info(f"   Precision Score: {execution_params.get('precision_score', 0.0):.3f}")
            self.logger.info(f"   Combined Confidence: {execution_params.get('combined_confidence', 0.0):.3f}")

            return execution_params

        except Exception as e:
            self.logger.error(error(f"❌ Error in enhanced tactician execution calculation: {e}"))
            return {"should_execute": False, "reason": "error", "error": str(e)}

    def _tactician_determine_direction(self, confidence_scores: Dict[str, float], market_data: pd.DataFrame) -> str:
        """Determine trade direction based on Tactician model confidences."""
        try:
            # Logic to determine if Tactician models suggest long, short, or neutral
            # This would be based on the specific Tactician model outputs (lower timeframe)
            bullish_confidence = sum(
                conf for name, conf in confidence_scores.items() if "bullish" in name.lower() or "long" in name.lower()
            )
            bearish_confidence = sum(
                conf for name, conf in confidence_scores.items() if "bearish" in name.lower() or "short" in name.lower()
            )

            # If no directional models, use overall confidence pattern
            if bullish_confidence == 0 and bearish_confidence == 0:
                # Use short-term price momentum as fallback
                if len(market_data) >= 3:
                    recent_change = (market_data["close"].iloc[-1] - market_data["close"].iloc[-3]) / market_data[
                        "close"
                    ].iloc[-3]
                    if abs(recent_change) > 0.0005:  # 0.05% threshold for short-term
                        return "long" if recent_change > 0 else "short"
                return "neutral"

            # Determine direction based on confidence
            if bullish_confidence > bearish_confidence and bullish_confidence > 0.6:
                return "long"
            elif bearish_confidence > bullish_confidence and bearish_confidence > 0.6:
                return "short"
            else:
                return "neutral"

        except Exception as e:
            self.logger.error(error(f"❌ Error determining tactician direction: {e}"))
            return "neutral"

    def _directions_agree(self, analyst_direction: str, tactician_direction: str) -> bool:
        """Check if Analyst and Tactician agree on trade direction."""
        if analyst_direction == "neutral" or tactician_direction == "neutral":
            return False
        return analyst_direction == tactician_direction

    def _tactician_calculate_leverage(self, confidence: float) -> float:
        """Calculate leverage based on confidence score."""
        if confidence > 0.9:
            return 3.0  # High leverage for very high confidence
        elif confidence > 0.8:
            return 2.5
        elif confidence > 0.7:
            return 2.0
        elif confidence > 0.6:
            return 1.5
        else:
            return 1.0  # No leverage for low confidence

    def _tactician_calculate_position_size(self, confidence: float, leverage: float) -> float:
        """Calculate position size based on confidence and leverage."""
        base_size = confidence * 100  # Base size as percentage
        adjusted_size = base_size * leverage
        return min(adjusted_size, 100.0)  # Cap at 100%

    def _tactician_calculate_entry_timing(self, market_data: pd.DataFrame, confidence: float) -> str:
        """Calculate optimal entry timing."""
        if confidence > 0.8:
            return "immediate"
        elif confidence > 0.7:
            return "within_5_minutes"
        else:
            return "wait_for_confirmation"

    @handles_errors
    async def _integrate_analyst_ml_profit_predictions(
        self,
        ml_profit_predictions: dict[str, Any],
        market_data: pd.DataFrame,
        regime_info: dict[str, Any],
        symbol: str,
        exchange: str,
    ) -> dict[str, Any]:
        """"
        Integrate ML profit predictions with existing Analyst components.

        This function enhances the Analyst's decision-making by incorporating:'
        1. ML profit predictions from steps 6-14
        2. Enhanced confidence scores with barrier analysis
        3. Risk-reward metrics
        4. Directional probability assessments
        """"
        try:
            integrated_predictions = {
                "ml_profit_integration": ml_profit_predictions,
                "enhanced_analyst_signals": {},
                "risk_metrics": {},
                "confidence_enhancement": {},
                "timestamp": datetime.now().isoformat(),
            }

            # Extract key components from ML profit predictions
            ml_profit_data = ml_profit_predictions.get("ml_profit_predictions", {})
            enhanced_confidence = ml_profit_predictions.get("enhanced_confidence_scores", {})
            barrier_analysis = ml_profit_predictions.get("barrier_analysis", {})
            regime_predictions = ml_profit_predictions.get("regime_predictions", {})

            # Generate enhanced analyst signals
            enhanced_signals = await self._generate_enhanced_analyst_signals(
                ml_profit_data,
                enhanced_confidence,
                barrier_analysis,
                regime_predictions,
            )
            integrated_predictions["enhanced_analyst_signals"] = enhanced_signals

            # Calculate risk metrics
            risk_metrics = await self._calculate_analyst_risk_metrics(ml_profit_data, barrier_analysis, market_data)
            integrated_predictions["risk_metrics"] = risk_metrics

            # Generate confidence enhancement
            confidence_enhancement = await self._generate_confidence_enhancement(
                enhanced_confidence, ml_profit_data, symbol, exchange
            )
            integrated_predictions["confidence_enhancement"] = confidence_enhancement

            return integrated_predictions

        except Exception as e:
            self.logger.error(error(f"❌ Error integrating analyst ML profit predictions: {e}"))
            return {}

    @handles_errors
    async def _integrate_tactician_ml_profit_predictions(
        self,
        ml_profit_predictions: dict[str, Any],
        market_data: pd.DataFrame,
        analyst_signals: dict[str, Any],
        symbol: str,
        exchange: str,
    ) -> dict[str, Any]:
        """"
        Integrate ML profit predictions with existing Tactician components.

        This function enhances the Tactician's execution by providing:'
        1. ML profit predictions with triple barrier probabilities
        2. Enhanced confidence scores for leverage decisions
        3. Barrier analysis for stop-loss placement
        4. Position decision signals (but NOT position sizing - that's Tactician's job)
        """"
        try:
            integrated_predictions = {
                "ml_profit_integration": ml_profit_predictions,
                "enhanced_tactician_signals": {},
                "position_decision_signals": {},
                "leverage_inputs": {},
                "timestamp": datetime.now().isoformat(),
            }

            # Extract key components from ML profit predictions
            ml_profit_data = ml_profit_predictions.get("ml_profit_predictions", {})
            enhanced_confidence = ml_profit_predictions.get("enhanced_confidence_scores", {})
            barrier_analysis = ml_profit_predictions.get("barrier_analysis", {})

            # Generate enhanced tactician signals
            enhanced_signals = await self._generate_enhanced_tactician_signals(
                ml_profit_data, enhanced_confidence, barrier_analysis, analyst_signals
            )
            integrated_predictions["enhanced_tactician_signals"] = enhanced_signals

            # Generate position decision signals (should we take a position?)
            position_decisions = await self._generate_position_decision_signals(
                ml_profit_data, enhanced_confidence, barrier_analysis
            )
            integrated_predictions["position_decision_signals"] = position_decisions

            # Generate leverage inputs for Tactician
            leverage_inputs = await self._generate_leverage_inputs(
                ml_profit_data, enhanced_confidence, barrier_analysis
            )
            integrated_predictions["leverage_inputs"] = leverage_inputs

            return integrated_predictions

        except Exception as e:
            self.logger.error(error(f"❌ Error integrating tactician ML profit predictions: {e}"))
            return {}

    @handles_errors
    async def _generate_enhanced_analyst_signals(
        self,
        ml_profit_data: dict[str, Any],
        enhanced_confidence: dict[str, Any],
        barrier_analysis: dict[str, Any],
        regime_predictions: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate enhanced analyst signals with ML profit integration."""
        try:
            enhanced_signals = {
                "directional_signals": {},
                "confidence_signals": {},
                "risk_signals": {},
                "regime_signals": {},
            }

            # Process directional signals from ML profit predictions
            for prediction_name, prediction_data in ml_profit_data.items():
                direction = prediction_data.get("direction", 0)
                magnitude = prediction_data.get("magnitude", 0.0)
                confidence = enhanced_confidence.get(prediction_name, {}).get("enhanced_confidence", 0.5)

                enhanced_signals["directional_signals"][prediction_name] = {
                    "direction": direction,
                    "magnitude": magnitude,
                    "confidence": confidence,
                    "signal_strength": abs(direction) * confidence,
                }

            # Process confidence signals
            for prediction_name, confidence_data in enhanced_confidence.items():
                enhanced_signals["confidence_signals"][prediction_name] = {
                    "enhanced_confidence": confidence_data.get("enhanced_confidence", 0.5),
                    "base_confidence": confidence_data.get("base_confidence", 0.5),
                    "confidence_improvement": confidence_data.get("enhanced_confidence", 0.5)
                    - confidence_data.get("base_confidence", 0.5),
                }

            # Process risk signals from barrier analysis
            for prediction_name, barrier_data in barrier_analysis.items():
                enhanced_signals["risk_signals"][prediction_name] = {
                    "risk_reward_ratio": barrier_data.get("risk_reward_ratio", 0.0),
                    "expected_value": barrier_data.get("expected_value", 0.0),
                    "barrier_distance": barrier_data.get("barrier_distance", 0.0),
                    "profit_distance": barrier_data.get("profit_distance", 0.0),
                }

            # Process regime signals
            for prediction_name, regime_data in regime_predictions.items():
                enhanced_signals["regime_signals"][prediction_name] = {
                    "regime": regime_data.get("regime", "unknown"),
                    "prediction": regime_data.get("prediction", 0.0),
                    "confidence": regime_data.get("confidence", 0.5),
                }

            return enhanced_signals

        except Exception as e:
            self.logger.error(error(f"❌ Error generating enhanced analyst signals: {e}"))
            return {}

    @handles_errors
    async def _generate_enhanced_tactician_signals(
        self,
        ml_profit_data: dict[str, Any],
        enhanced_confidence: dict[str, Any],
        barrier_analysis: dict[str, Any],
        analyst_signals: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate enhanced tactician signals with ML profit integration."""
        try:
            enhanced_signals = {
                "execution_signals": {},
                "timing_signals": {},
                "risk_signals": {},
            }

            # Process execution signals
            for prediction_name, prediction_data in ml_profit_data.items():
                direction = prediction_data.get("direction", 0)
                magnitude = prediction_data.get("magnitude", 0.0)
                confidence = enhanced_confidence.get(prediction_name, {}).get("enhanced_confidence", 0.5)

                # Determine execution urgency based on confidence and magnitude
                execution_urgency = confidence * magnitude

                enhanced_signals["execution_signals"][prediction_name] = {
                    "direction": direction,
                    "magnitude": magnitude,
                    "confidence": confidence,
                    "execution_urgency": execution_urgency,
                    "should_execute": confidence > self.enhanced_prediction_service.direction_confidence_threshold,
                }

            # Process position signals
            for prediction_name, prediction_data in ml_profit_data.items():
                confidence = enhanced_confidence.get(prediction_name, {}).get("enhanced_confidence", 0.5)
                magnitude = prediction_data.get("magnitude", 0.0)

                # Calculate position size based on confidence and magnitude
                position_size_factor = confidence * min(1.0, magnitude * 10)  # Scale magnitude

                enhanced_signals["position_signals"][prediction_name] = {
                    "position_size_factor": position_size_factor,
                    "confidence": confidence,
                    "magnitude": magnitude,
                    "recommended_size": (
                        "large" if position_size_factor > 0.7 else "medium" if position_size_factor > 0.4 else "small"
                    ),
                }

            # Process risk signals
            for prediction_name, barrier_data in barrier_analysis.items():
                enhanced_signals["risk_signals"][prediction_name] = {
                    "stop_loss_level": barrier_data.get("barrier_level", 0.0),
                    "take_profit_level": barrier_data.get("profit_target", 0.0),
                    "risk_reward_ratio": barrier_data.get("risk_reward_ratio", 0.0),
                    "expected_value": barrier_data.get("expected_value", 0.0),
                }

            # Process timing signals
            for prediction_name, prediction_data in ml_profit_data.items():
                confidence = enhanced_confidence.get(prediction_name, {}).get("enhanced_confidence", 0.5)

                # Determine timing based on confidence and volatility
                timing_urgency = "immediate" if confidence > 0.8 else "normal" if confidence > 0.6 else "cautious"

                enhanced_signals["timing_signals"][prediction_name] = {
                    "timing_urgency": timing_urgency,
                    "confidence": confidence,
                    "wait_for_confirmation": confidence < 0.6,
                }

            return enhanced_signals

        except Exception as e:
            self.logger.error(error(f"❌ Error generating enhanced tactician signals: {e}"))
            return {}

    @handles_errors
    async def _generate_position_decision_signals(
        self,
        ml_profit_data: dict[str, Any],
        enhanced_confidence: dict[str, Any],
        barrier_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """"
        Generate position decision signals (should we take a position?).

        This provides signals to the Tactician about whether to take positions,
        but does NOT calculate position sizing - that's the Tactician's responsibility.
        """"
        try:
            position_decisions = {
                "position_recommendations": {},
                "aggregate_position_signal": {},
            }

            # Generate position recommendations for each prediction
            for prediction_name, prediction_data in ml_profit_data.items():
                confidence_data = enhanced_confidence.get(prediction_name, {})
                optimized_confidence = confidence_data.get("optimized_confidence", 0.5)
                triple_barrier_probs = confidence_data.get("triple_barrier_details", {})

                # Determine if we should take a position based on confidence
                should_take_position = (
                    optimized_confidence > self.enhanced_prediction_service.direction_confidence_threshold
                )

                # Get the best triple barrier probability for decision making
                best_probability = 0.0
                best_scenario = None

                if triple_barrier_probs:
                    for scenario_name, scenario_data in triple_barrier_probs.items():
                        if scenario_data["probability"] > best_probability:
                            best_probability = scenario_data["probability"]
                            best_scenario = scenario_name

                position_decisions["position_recommendations"][prediction_name] = {
                    "should_take_position": should_take_position,
                    "confidence": optimized_confidence,
                    "best_triple_barrier_probability": best_probability,
                    "best_scenario": best_scenario,
                    "direction": prediction_data.get("direction", 0),
                    "magnitude": prediction_data.get("magnitude", 0.0),
                    "recommendation_strength": (
                        "strong" if optimized_confidence > 0.8 else "moderate" if optimized_confidence > 0.6 else "weak"
                    ),
                }

            # Calculate aggregate position signal
            total_recommendations = len(position_decisions["position_recommendations"])
            strong_recommendations = sum(
                1
                for rec in position_decisions["position_recommendations"].values()
                if rec["recommendation_strength"] == "strong"
            )
            moderate_recommendations = sum(
                1
                for rec in position_decisions["position_recommendations"].values()
                if rec["recommendation_strength"] == "moderate"
            )

            if total_recommendations > 0:
                strong_ratio = strong_recommendations / total_recommendations
                moderate_ratio = moderate_recommendations / total_recommendations

                if strong_ratio > 0.5:
                    aggregate_signal = "strong_buy"
                elif moderate_ratio > 0.5:
                    aggregate_signal = "moderate_buy"
                elif strong_ratio > 0.2:
                    aggregate_signal = "weak_buy"
                else:
                    aggregate_signal = "hold"
            else:
                aggregate_signal = "hold"

            position_decisions["aggregate_position_signal"] = {
                "signal": aggregate_signal,
                "total_recommendations": total_recommendations,
                "strong_recommendations": strong_recommendations,
                "moderate_recommendations": moderate_recommendations,
                "strong_ratio": strong_ratio if total_recommendations > 0 else 0.0,
                "moderate_ratio": moderate_ratio if total_recommendations > 0 else 0.0,
            }

            return position_decisions

        except Exception as e:
            self.logger.error(error(f"❌ Error generating position decision signals: {e}"))
            return {}

    @handles_errors
    async def _generate_leverage_inputs(
        self,
        ml_profit_data: dict[str, Any],
        enhanced_confidence: dict[str, Any],
        barrier_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """"
        Generate leverage inputs for the Tactician.

        This provides confidence and probability data to help the Tactician
        make leverage decisions, but does NOT calculate leverage itself.
        """"
        try:
            leverage_inputs = {
                "confidence_inputs": {},
                "probability_inputs": {},
                "risk_inputs": {},
            }

            # Generate confidence inputs for leverage decisions
            for prediction_name, prediction_data in ml_profit_data.items():
                confidence_data = enhanced_confidence.get(prediction_name, {})
                optimized_confidence = confidence_data.get("optimized_confidence", 0.5)
                triple_barrier_max_prob = confidence_data.get("triple_barrier_max_probability", 0.5)

                leverage_inputs["confidence_inputs"][prediction_name] = {
                    "model_confidence": prediction_data.get("model_confidence", 0.5),
                    "optimized_confidence": optimized_confidence,
                    "triple_barrier_max_probability": triple_barrier_max_prob,
                    "confidence_for_leverage": max(optimized_confidence, triple_barrier_max_prob),
                    "leverage_confidence_level": (
                        "high" if optimized_confidence > 0.8 else "medium" if optimized_confidence > 0.6 else "low"
                    ),
                }

            # Generate probability inputs
            for prediction_name, prediction_data in ml_profit_data.items():
                confidence_data = enhanced_confidence.get(prediction_name, {})
                triple_barrier_probs = confidence_data.get("triple_barrier_details", {})

                # Extract probability information for leverage decisions
                probabilities = []
                scenarios = []

                for scenario_name, scenario_data in triple_barrier_probs.items():
                    probabilities.append(scenario_data["probability"])
                    scenarios.append(
                        {
                            "name": scenario_name,
                            "probability": scenario_data["probability"],
                            "risk_reward_ratio": scenario_data["risk_reward_ratio"],
                        }
                    )

                leverage_inputs["probability_inputs"][prediction_name] = {
                    "all_probabilities": probabilities,
                    "max_probability": max(probabilities) if probabilities else 0.5,
                    "avg_probability": (sum(probabilities) / len(probabilities) if probabilities else 0.5),
                    "scenarios": scenarios,
                    "probability_consistency": (
                        1.0 - (max(probabilities) - min(probabilities)) if len(probabilities) > 1 else 1.0
                    ),
                }

            # Generate risk inputs
            for prediction_name, barrier_data in barrier_analysis.items():
                leverage_inputs["risk_inputs"][prediction_name] = {
                    "risk_reward_ratio": barrier_data.get("risk_reward_ratio", 1.0),
                    "expected_value": barrier_data.get("expected_value", 0.0),
                    "barrier_distance": barrier_data.get("barrier_distance", 0.0),
                    "profit_distance": barrier_data.get("profit_distance", 0.0),
                    "risk_level": (
                        "low"
                        if barrier_data.get("risk_reward_ratio", 1.0) > 2.0
                        else ("medium" if barrier_data.get("risk_reward_ratio", 1.0) > 1.5 else "high")
                    ),
                }

            return leverage_inputs

        except Exception as e:
            self.logger.error(error(f"❌ Error generating leverage inputs: {e}"))
            return {}

    @handles_errors
    async def _calculate_analyst_risk_metrics(
        self,
        ml_profit_data: dict[str, Any],
        barrier_analysis: dict[str, Any],
        market_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Calculate risk metrics for analyst decision making."""
        try:
            risk_metrics = {
                "aggregate_risk": {},
                "individual_risks": {},
                "portfolio_implications": {},
            }

            # Calculate aggregate risk metrics
            total_confidence = 0.0
            total_expected_value = 0.0
            total_risk_reward = 0.0
            prediction_count = 0

            for prediction_name, prediction_data in ml_profit_data.items():
                confidence = prediction_data.get("confidence", 0.5)
                barrier_data = barrier_analysis.get(prediction_name, {})

                total_confidence += confidence
                total_expected_value += barrier_data.get("expected_value", 0.0)
                total_risk_reward += barrier_data.get("risk_reward_ratio", 0.0)
                prediction_count += 1

            if prediction_count > 0:
                avg_confidence = total_confidence / prediction_count
                avg_expected_value = total_expected_value / prediction_count
                avg_risk_reward = total_risk_reward / prediction_count
            else:
                avg_confidence = 0.5
                avg_expected_value = 0.0
                avg_risk_reward = 0.0

            risk_metrics["aggregate_risk"] = {
                "average_confidence": avg_confidence,
                "average_expected_value": avg_expected_value,
                "average_risk_reward_ratio": avg_risk_reward,
                "prediction_count": prediction_count,
                "overall_risk_level": ("low" if avg_confidence > 0.7 else "medium" if avg_confidence > 0.5 else "high"),
            }

            # Calculate individual risk metrics
            for prediction_name, prediction_data in ml_profit_data.items():
                barrier_data = barrier_analysis.get(prediction_name, {})

                risk_metrics["individual_risks"][prediction_name] = {
                    "confidence": prediction_data.get("confidence", 0.5),
                    "expected_value": barrier_data.get("expected_value", 0.0),
                    "risk_reward_ratio": barrier_data.get("risk_reward_ratio", 0.0),
                    "risk_level": (
                        "low"
                        if prediction_data.get("confidence", 0.5) > 0.7
                        else ("medium" if prediction_data.get("confidence", 0.5) > 0.5 else "high")
                    ),
                }

            # Calculate portfolio implications
            current_volatility = market_data["close"].pct_change().std()

            risk_metrics["portfolio_implications"] = {
                "market_volatility": current_volatility,
                "recommended_position_size": (
                    "reduced" if current_volatility > 0.03 else "normal" if current_volatility > 0.02 else "increased"
                ),
                "risk_adjustment_factor": max(0.5, min(1.5, 1.0 / (1.0 + current_volatility * 10))),
            }

            return risk_metrics

        except Exception as e:
            self.logger.error(error(f"❌ Error calculating analyst risk metrics: {e}"))
            return {}

    @handles_errors(
        error_handlers={
            Exception: (False, "Supervisor run failed"),
        },
        default_return=False,
        context="supervisor run",
    )
    async def run(self) -> bool:
        self.is_running = True
        self.logger.info("🚦 Supervisor started.")
        while self.is_running:
            await self._perform_supervision()
            await asyncio.sleep(self.supervision_interval)
        return True

    @handles_errors(fallback=None)
    async def _perform_supervision(self) -> None:
        # Perform health checks
        await self._monitor_system_health()

        # Monitor component-specific features
        await self._monitor_component_features()

        # Coordinate components
        await self._coordinate_components()

        # Update online learning
        await self._update_online_learning()

        # Enforce portfolio risk guardrails (kill-switch)
        await self._enforce_portfolio_guards()

        # Update supervision results
        await self._update_supervision_results()

        # Check for recovery needs
        await self._check_recovery_needs()

    @handles_errors(fallback=None)
    async def _monitor_system_health(self) -> None:
        try:
            # Check critical components health
            for component in self.critical_components:
                health_status = await self._check_component_health(component)
                self.health_checks[component] = health_status

                if not health_status:
                    self.print(failed("⚠️ Component {component} health check failed"))
                    await self._trigger_recovery(component)

            # Log overall health status
            healthy_components = sum(self.health_checks.values())
            total_components = len(self.health_checks)
            health_percentage = (healthy_components / total_components) * 100 if total_components > 0 else 0

            self.logger.info(
                f"System health: {health_percentage:.1f}% ({healthy_components}/{total_components} components healthy)",
            )

        except Exception:
            self.print(error("Error monitoring system health: {e}"))

    def _monitor_analyst_features(self) -> None:
        """Monitor Analyst component features."""
        if "analyst" not in self.components or not self.components["analyst"]:
            return

        analyst = self.components["analyst"]
        analyst_monitors = self.component_monitors["analyst"]

        # Define analyst features to monitor
        analyst_features = {
            "dual_model_system": "dual_model_system",
            "liquidation_risk_model": "liquidation_risk_model",
            "feature_engineering_orchestrator": "feature_engineering_orchestrator",
            "ml_confidence_predictor": "ml_confidence_predictor",
            "regime_classifier": "regime_classifier",
        }

        # Monitor each feature
        for monitor_key, feature_name in analyst_features.items():
            analyst_monitors[monitor_key] = hasattr(analyst, feature_name) and getattr(analyst, feature_name) is not None

    def _monitor_strategist_features(self) -> None:
        """Monitor Strategist component features."""
        if "strategist" not in self.components or not self.components["strategist"]:
            return

        strategist = self.components["strategist"]
        strategist_monitors = self.component_monitors["strategist"]

        # Define strategist features to monitor (strategy generation and analysis integration)
        strategist_features = {
            "strategy_generator": "current_strategy",
            "market_analysis_integrator": "market_analysis",
            "strategy_history_manager": "strategy_history",
        }

        # Monitor each feature
        for monitor_key, feature_name in strategist_features.items():
            strategist_monitors[monitor_key] = (
                hasattr(strategist, feature_name) and getattr(strategist, feature_name) is not None
            )

    def _monitor_tactician_features(self) -> None:
        """Monitor Tactician component features."""
        if "tactician" not in self.components or not self.components["tactician"]:
            return

        tactician = self.components["tactician"]
        tactician_monitors = self.component_monitors["tactician"]

        # Define tactician features to monitor
        tactician_features = {
            "sr_breakout_predictor": "sr_breakout_predictor",
            "position_sizer": "position_sizer",
            "leverage_sizer": "leverage_sizer",
            "position_division_strategy": "position_division_strategy",
            "ml_predictions": "ml_predictions",
        }

        # Monitor each feature
        for monitor_key, feature_name in tactician_features.items():
            tactician_monitors[monitor_key] = (
                hasattr(tactician, feature_name) and getattr(tactician, feature_name) is not None
            )

    def _monitor_enhanced_training_manager_features(self) -> None:
        """Monitor Enhanced Training Manager component features."""
        if "enhanced_training_manager" not in self.components or not self.components["enhanced_training_manager"]:
            return

        training_manager = self.components["enhanced_training_manager"]
        training_monitors = self.component_monitors["enhanced_training_manager"]

        # Define training manager features to monitor
        training_features = {
            "advanced_model_training": "advanced_model_training",
            "ensemble_training": "ensemble_training",
            "multi_timeframe_training": "multi_timeframe_training",
            "adaptive_training": "adaptive_training",
            "multi_timeframe_manager": "multi_timeframe_manager",
            "ensemble_creator": "ensemble_creator",
        }

        # Monitor each feature
        for monitor_key, feature_name in training_features.items():
            training_monitors[monitor_key] = (
                hasattr(training_manager, feature_name) and getattr(training_manager, feature_name) is not None
            )

    def _log_component_feature_status(self) -> None:
        """Log the status of all component features."""
        for component, monitors in self.component_monitors.items():
            active_features = sum(monitors.values())
            total_features = len(monitors)
            if total_features > 0:
                feature_percentage = (active_features / total_features) * 100
                self.logger.info(
                    f"{component} features: {feature_percentage:.1f}% ({active_features}/{total_features} active)",
                )

    @handles_errors(fallback=None)
    async def _monitor_component_features(self) -> None:
        """Monitor component-specific features and sub-components."""
        try:
            # Monitor each component's features'
            self._monitor_analyst_features()
            self._monitor_strategist_features()
            self._monitor_tactician_features()
            self._monitor_enhanced_training_manager_features()

            # Log component feature status
            self._log_component_feature_status()

        except Exception:
            self.print(error("Error monitoring component features: {e}"))

    @handles_errors(fallback=False)
    async def _check_component_health(self, component: str) -> bool:
        """Check health of a specific component."""
        try:
            # Mock health check - replace with actual component health checks
            if component in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers[component]
                return circuit_breaker.state != "OPEN"

            # Default health check
            return True

        except Exception as e:
            self.logger.exception(
                f"Error checking health for component {component}: {e}",
            )
            return False

    @handles_errors(fallback=None)
    async def _coordinate_components(self) -> None:
        """"
        Coordinate components with clear separation of responsibilities:
        - Strategist: Provides trading strategies and market analysis
        - Tactician: Handles position sizing and execution tactics
        - Supervisor: Orchestrates communication and system-level coordination
        """"
        try:
            # Coordinate Analyst-Strategist
            await self._coordinate_analyst_strategist()

            # Coordinate Strategist-Tactician
            await self._coordinate_strategist_tactician()

            # Coordinate Training Manager
            await self._coordinate_training_manager()

        except Exception:
            self.print(error("Error coordinating components: {e}"))

    @handles_errors(fallback=None)
    async def _coordinate_analyst_strategist(self) -> None:
        """Coordinate Analyst and Strategist components."""
        try:
            analyst = self.components["analyst"]
            strategist = self.components["strategist"]

            # Share regime classification results
            if hasattr(analyst, "regime_classifier") and analyst.regime_classifier:
                regime_info = await analyst._perform_regime_classification({})
                if regime_info and hasattr(strategist, "current_regime"):
                    strategist.current_regime = regime_info.get("regime")
                    strategist.regime_confidence = regime_info.get("confidence", 0.0)

            # Share ML confidence predictions
            if hasattr(analyst, "ml_confidence_predictor") and analyst.ml_confidence_predictor:
                ml_predictions = await analyst._perform_ml_predictions({})
                if ml_predictions and hasattr(strategist, "ml_confidence_predictor"):
                    strategist.ml_confidence_predictor = ml_predictions

            self.logger.info("Analyst-Strategist coordination completed")

        except Exception:
            self.print(error("Error coordinating Analyst-Strategist: {e}"))

    @handles_errors(fallback=None)
    async def _coordinate_strategist_tactician(self) -> None:
        """"
        Coordinate Strategist and Tactician components.

        Strategy Coordination:
        - Strategist provides trading strategies and market analysis
        - Tactician handles position sizing and execution tactics
        - Supervisor orchestrates communication between the two
        """"
        try:
            strategist = self.components["strategist"]
            tactician = self.components["tactician"]

            # Share strategy information from Strategist to Tactician
            if hasattr(strategist, "current_strategy") and strategist.current_strategy:
                if hasattr(tactician, "strategy_input"):
                    tactician.strategy_input = strategist.current_strategy

            # Share market analysis results for tactical decisions
            if hasattr(strategist, "market_analysis") and strategist.market_analysis:
                if hasattr(tactician, "market_analysis_input"):
                    tactician.market_analysis_input = strategist.market_analysis

            # Share regime information for tactical decisions
            if hasattr(strategist, "current_regime") and strategist.current_regime:
                if hasattr(tactician, "current_regime"):
                    tactician.current_regime = strategist.current_regime

            self.logger.info("Strategist-Tactician coordination completed")

        except Exception:
            self.print(error("Error coordinating Strategist-Tactician: {e}"))

    @handles_errors(fallback=None)
    async def _coordinate_training_manager(self) -> None:
        """Coordinate Enhanced Training Manager with other components."""
        try:
            training_manager = self.components["enhanced_training_manager"]

            # Coordinate with Analyst for model updates
            if self.components.get("analyst"):
                analyst = self.components["analyst"]
                if hasattr(training_manager, "get_enhanced_training_results"):
                    training_results = training_manager.get_enhanced_training_results()
                    if training_results and hasattr(analyst, "update_models"):
                        await analyst.update_models(training_results)

            # Coordinate with Strategist for model updates
            if self.components.get("strategist"):
                strategist = self.components["strategist"]
                if hasattr(training_manager, "get_enhanced_training_results"):
                    training_results = training_manager.get_enhanced_training_results()
                    if training_results and hasattr(strategist, "update_models"):
                        await strategist.update_models(training_results)

            self.logger.info("Training Manager coordination completed")

        except Exception:
            self.print(error("Error coordinating Training Manager: {e}"))

    @handles_errors(fallback=None)
    async def _update_online_learning(self) -> None:
        """Update online learning with current performance data."""
        try:
            # Get current model performances from components
            model_performances = {}

            # Get performances from Analyst
            if self.components.get("analyst"):
                analyst = self.components["analyst"]
                if hasattr(analyst, "get_analysis_results"):
                    analysis_results = analyst.get_analysis_results()
                    if analysis_results:
                        model_performances["analyst"] = analysis_results.get(
                            "performance_score",
                            0.5,
                        )

            # Get performances from Strategist
            if self.components.get("strategist"):
                strategist = self.components["strategist"]
                if hasattr(strategist, "get_strategy_performance"):
                    strategy_performance = strategist.get_strategy_performance()
                    if strategy_performance:
                        model_performances["strategist"] = strategy_performance.get(
                            "win_rate",
                            0.5,
                        )

            # Get performances from Tactician
            if self.components.get("tactician"):
                tactician = self.components["tactician"]
                if hasattr(tactician, "get_tactics_results"):
                    tactics_results = tactician.get_tactics_results()
                    if tactics_results:
                        model_performances["tactician"] = tactics_results.get(
                            "performance_score",
                            0.5,
                        )

            # Update online learning with current performances
            for model_id, performance in model_performances.items():
                await self.online_learning.update_model_performance(
                    model_id,
                    performance,
                )

            # Get updated weights
            updated_weights = self.online_learning.get_model_weights()
            self.supervision_results["online_learning"] = {
                "timestamp": time.time(),  # Changed from datetime.now() to time.time()
                "model_weights": updated_weights,
                "model_performances": self.online_learning.get_model_performances(),
            }

            self.logger.info(f"Online learning updated: {updated_weights}")

        except Exception:
            self.print(error("Error updating online learning: {e}"))

    @handles_errors(fallback=None)
    async def _trigger_recovery(self, component: str) -> None:
        """Trigger recovery for a failed component."""
        try:
            current_time = time.time()
            last_attempt = self.last_recovery_attempt.get(component=0)

            # Check if we can attempt recovery
            if (
                current_time - last_attempt < self.recovery_cooldown
                or self.recovery_attempts[component] >= self.max_recovery_attempts
            ):
                return

            self.logger.info(f"🔄 Triggering recovery for component: {component}")

            # Attempt recovery
            recovery_success = await self._attempt_recovery(component)

            if recovery_success:
                self.logger.info(f"✅ Recovery successful for component: {component}")
                self.recovery_attempts[component] = 0
            else:
                self.recovery_attempts[component] += 1
                self.logger.warning(
                    f"⚠️ Recovery failed for component: {component} (attempt {self.recovery_attempts[component]}/{self.max_recovery_attempts})",
                )

            self.last_recovery_attempt[component] = current_time

        except Exception:
            self.print(error("Error triggering recovery for {component}: {e}"))

    @handles_errors(fallback=False)
    async def _attempt_recovery(self, component: str) -> bool:
        """Attempt to recover a failed component."""
        try:
            # Implement component-specific recovery logic
            if component == "database":
                return await self._recover_database()
            if component == "exchange":
                return await self._recover_exchange()
            if component == "analyst":
                return await self._recover_analyst()
            if component == "strategist":
                return await self._recover_strategist()
            if component == "tactician":
                return await self._recover_tactician()
            if component == "enhanced_training_manager":
                return await self._recover_enhanced_training_manager()
            # Generic recovery
            return await self._generic_recovery(component)

        except Exception:
            self.print(error("Error attempting recovery for {component}: {e}"))
            return False

    @handles_errors(fallback=False)
    async def _recover_database(self) -> bool:
        """Recover database connection."""
        try:
            # Implement database recovery logic
            self.logger.info("Attempting database recovery...")
            # Mock recovery - replace with actual database reconnection logic
            await asyncio.sleep(1)
            return True
        except Exception:
            self.print(failed("Database recovery failed: {e}"))
            return False

    @handles_errors(fallback=False)
    async def _recover_exchange(self) -> bool:
        """Recover exchange connection."""
        try:
            # Implement exchange recovery logic
            self.logger.info("Attempting exchange recovery...")
            # Mock recovery - replace with actual exchange reconnection logic
            await asyncio.sleep(1)
            return True
        except Exception:
            self.print(failed("Exchange recovery failed: {e}"))
            return False

    @handles_errors(fallback=False)
    async def _recover_analyst(self) -> bool:
        """Recover analyst component."""
        try:
            # Implement analyst recovery logic
            self.logger.info("Attempting analyst recovery...")
            # Mock recovery - replace with actual analyst restart logic
            await asyncio.sleep(1)
            return True
        except Exception:
            self.print(failed("Analyst recovery failed: {e}"))
            return False

    @handles_errors(fallback=False)
    async def _recover_strategist(self) -> bool:
        """Recover strategist component."""
        try:
            # Implement strategist recovery logic
            self.logger.info("Attempting strategist recovery...")
            # Mock recovery - replace with actual strategist restart logic
            await asyncio.sleep(1)
            return True
        except Exception:
            self.print(failed("Strategist recovery failed: {e}"))
            return False

    @handles_errors(fallback=False)
    async def _recover_tactician(self) -> bool:
        """Recover tactician component."""
        try:
            # Implement tactician recovery logic
            self.logger.info("Attempting tactician recovery...")
            # Mock recovery - replace with actual tactician restart logic
            await asyncio.sleep(1)
            return True
        except Exception:
            self.print(failed("Tactician recovery failed: {e}"))
            return False

    @handles_errors(fallback=False)
    async def _recover_enhanced_training_manager(self) -> bool:
        """Recover enhanced training manager component."""
        try:
            # Implement enhanced training manager recovery logic
            self.logger.info("Attempting enhanced training manager recovery...")
            # Mock recovery - replace with actual training manager restart logic
            await asyncio.sleep(1)
            return True
        except Exception:
            self.print(failed("Enhanced training manager recovery failed: {e}"))
            return False

    @handles_errors(fallback=False)
    async def _generic_recovery(self, component: str) -> bool:
        """Generic recovery for unspecified components."""
        try:
            self.logger.info(f"Attempting generic recovery for {component}...")
            # Mock recovery - replace with actual restart logic
            await asyncio.sleep(1)
            return True
        except Exception:
            self.print(failed("Generic recovery failed for {component}: {e}"))
            return False

    @handles_errors(fallback=None)
    async def _check_recovery_needs(self) -> None:
        """Check if any components need recovery."""
        try:
            for component, health_status in self.health_checks.items():
                if not health_status:
                    await self._trigger_recovery(component)

        except Exception:
            self.print(error("Error checking recovery needs: {e}"))

    @handles_errors(fallback=None)
    async def _update_supervision_results(self) -> None:
        try:
            # Add timestamp
            self.supervision_results["timestamp"] = time.time()  # Changed from datetime.now() to time.time()

            # Add health status
            self.supervision_results["health_status"] = self.health_checks.copy()

            # Add component monitors status
            self.supervision_results["component_monitors"] = self.component_monitors.copy()

            # Add recovery status
            self.supervision_results["recovery_status"] = {
                "recovery_attempts": dict(self.recovery_attempts),
                "last_recovery_attempts": self.last_recovery_attempt.copy(),
            }

            # Add to history
            self.history.append(self.supervision_results.copy())

            # Limit history size
            if len(self.history) > self.max_history:
                self.history.pop(0)

        except Exception:
            self.print(error("Error updating supervision results: {e}"))

    @handles_errors(fallback=None)
    async def stop(self) -> None:
        self.logger.info("🛑 Stopping Supervisor...")
        try:
            self.is_running = False
            self.logger.info("✅ Supervisor stopped successfully")
        except Exception:
            self.print(error("Error stopping supervisor: {e}"))

    def get_status(self) -> dict[str, Any]:
        return {
            "is_running": self.is_running,
            "supervision_interval": self.supervision_interval,
            "max_history": self.max_history,
            "health_checks": self.health_checks,
            "component_monitors": self.component_monitors,
            "recovery_attempts": dict(self.recovery_attempts),
            "online_learning_weights": self.online_learning.get_model_weights(),
        }

    def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        history = self.history.copy()
        if limit:
            history = history[-limit:]
        return history

    def get_supervision_results(self) -> dict[str, Any]:
        return self.supervision_results.copy()

    def get_components(self) -> dict[str, Any]:
        return self.components.copy()

    def get_online_learning_status(self) -> dict[str, Any]:
        """Get online learning status and statistics."""
        return {
            "model_weights": self.online_learning.get_model_weights(),
            "model_performances": self.online_learning.get_model_performances(),
            "learning_rate": self.online_learning.learning_rate,
            "min_weight": self.online_learning.min_weight,
            "max_weight": self.online_learning.max_weight,
        }

    def get_component_monitors(self) -> dict[str, Any]:
        """Get component monitors status."""
        return self.component_monitors.copy()

    @handles_errors(fallback=None)
    async def _enforce_portfolio_guards(self) -> None:
        """Pause tactician or reduce risk when daily loss / drawdown limits are breached."""
        try:
            perf_monitor = self.components.get("performance_monitor")
            tactician = self.components.get("tactician")
            if not perf_monitor or not tactician:
                return

            # Get performance metrics
            if hasattr(perf_monitor, "get_performance_metrics"):
                metrics = perf_monitor.get_performance_metrics()
            else:
                metrics = {}

            max_drawdown = float(
                metrics.get("max_drawdown", 0.0),
            )  # negative when losing
            total_return = float(metrics.get("total_return", 0.0))

            risk_cfg = self.supervisor_config.get("portfolio_guards", {})
            dd_limit = float(risk_cfg.get("max_drawdown_limit", -0.05))  # -5%
            daily_loss_limit = float(risk_cfg.get("max_daily_loss", -0.05))  # -5%

            # For daily loss, if available via metrics
            daily_return = float(metrics.get("daily_return", total_return))

            breach = (max_drawdown <= dd_limit) or (daily_return <= daily_loss_limit)
            if breach:
                # Pause tactician run loop or set is_running flag down
                if hasattr(tactician, "is_running"):
                    tactician.is_running = False
                self.logger.warning(
                    f"⛔ Portfolio guard triggered. MDD={max_drawdown:.2%}, Daily={daily_return:.2%}. Pausing Tactician.",
                )
                # Record in supervision results
                self.supervision_results.setdefault("guards", {})["paused"] = True
                self.supervision_results["guards"]["reason"] = {
                    "max_drawdown": max_drawdown,
                    "daily_return": daily_return,
                    "limits": {"dd_limit": dd_limit, "daily_limit": daily_loss_limit},
                }
        except Exception:
            self.print(error("Error enforcing portfolio guards: {e}"))
            return

supervisor: Supervisor | None = None

@handles_errors(fallback=None)
async def setup_supervisor(
    config: dict[str, Any] | None = None,
) -> Supervisor | None:
    global supervisor
    if config is None:
        config = DEFAULT_SUPERVISOR_CONFIG
    supervisor = Supervisor(config)
    success = await supervisor.initialize()
    if success:
        return supervisor
    return None
