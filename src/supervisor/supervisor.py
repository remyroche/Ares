import asyncio
import time
from collections import defaultdict
from typing import Any

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class CircuitBreaker:
    """Circuit breaker pattern for external services."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    @handle_errors(exceptions=(Exception,), default_return=None)
    async def call(self, func: callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise e


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

    @handle_errors(exceptions=(Exception,), default_return=None)
    async def update_model_performance(self, model_id: str, performance: float) -> None:
        """Update model performance and recalculate weights."""
        try:
            self.model_performances[model_id].append(performance)

            # Keep only recent performances (last 100)
            if len(self.model_performances[model_id]) > 100:
                self.model_performances[model_id] = self.model_performances[model_id][
                    -100:
                ]

            # Recalculate weights based on recent performance
            await self._recalculate_weights()

            self.logger.info(f"Updated performance for model {model_id}: {performance}")

        except Exception as e:
            self.logger.error(f"Error updating model performance: {e}")

    @handle_errors(exceptions=(Exception,), default_return=None)
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
                self.model_weights = {
                    model_id: equal_weight for model_id in avg_performances
                }
            else:
                # Weight based on performance
                for model_id, avg_perf in avg_performances.items():
                    weight = avg_perf / total_performance
                    # Apply min/max constraints
                    weight = max(self.min_weight, min(self.max_weight, weight))
                    self.model_weights[model_id] = weight

            self.logger.info(f"Recalculated weights: {self.model_weights}")

        except Exception as e:
            self.logger.error(f"Error recalculating weights: {e}")

    def get_model_weights(self) -> dict[str, float]:
        """Get current model weights."""
        return self.model_weights.copy()

    def get_model_performances(self) -> dict[str, list[float]]:
        """Get model performance history."""
        return {k: v.copy() for k, v in self.model_performances.items()}


class Supervisor:
    """
    Enhanced Supervisor component with DI, type hints, robust error handling,
    advanced error handling, automatic recovery, and online learning.
    Updated to accommodate recent changes to Tactician, Strategist, Trainer, and Analyst.
    """

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

        # Health monitoring - Updated to include new component features
        self.health_checks: dict[str, bool] = {}
        self.critical_components: list[str] = [
            "database",
            "exchange",
            "analyst",
            "strategist",
            "tactician",
            "enhanced_training_manager",
        ]

        # Component-specific monitoring
        self.component_monitors: dict[str, dict[str, Any]] = {
            "analyst": {
                "dual_model_system": False,
                "market_health_analyzer": False,
                "liquidation_risk_model": False,
                "feature_engineering_orchestrator": False,
                # Legacy S/R/Candle code removed,
                "ml_confidence_predictor": False,
                "regime_classifier": False,
            },
            "strategist": {
                "regime_classifier": False,
                "ml_confidence_predictor": False,
                "volatility_targeting": False,
            },
            "tactician": {
                # Legacy S/R/Candle code removed,
                "sr_breakout_predictor": False,
                "position_sizer": False,
                "leverage_sizer": False,
                "position_division_strategy": False,
                "ml_predictions": False,
                "position_monitor": False,
            },
            "enhanced_training_manager": {
                "advanced_model_training": False,
                "ensemble_training": False,
                "multi_timeframe_training": False,
                "adaptive_training": False,
                "multi_timeframe_manager": False,
                "ensemble_creator": False,
            },
        }

    @handle_specific_errors(
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
                self.logger.error("Invalid configuration for supervisor")
                return False
            await self._initialize_components()
            await self._setup_circuit_breakers()
            await self._setup_online_learning()
            await self._setup_component_monitors()
            self.logger.info("✅ Supervisor initialization completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Supervisor initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="supervisor configuration loading",
    )
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
        except Exception as e:
            self.logger.error(f"Error loading supervisor configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        try:
            if self.supervision_interval <= 0:
                self.logger.error("Invalid supervision interval")
                return False
            if self.max_history <= 0:
                self.logger.error("Invalid max history")
                return False
            if self.max_recovery_attempts <= 0:
                self.logger.error("Invalid max recovery attempts")
                return False
            if self.recovery_cooldown <= 0:
                self.logger.error("Invalid recovery cooldown")
                return False
            self.logger.info("Configuration validation successful")
            return True
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="component initialization",
    )
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

            self.logger.info("Components initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="circuit breakers setup",
    )
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
        except Exception as e:
            self.logger.error(f"Error setting up circuit breakers: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="online learning setup",
    )
    async def _setup_online_learning(self) -> None:
        """Setup online learning for model weighting."""
        try:
            # Initialize online learning with default configuration
            online_learning_config = self.supervisor_config.get("online_learning", {})
            self.online_learning = OnlineLearningManager(online_learning_config)

            self.logger.info("Online learning setup complete")
        except Exception as e:
            self.logger.error(f"Error setting up online learning: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="component monitors setup",
    )
    async def _setup_component_monitors(self) -> None:
        """Setup component-specific monitoring."""
        try:
            # Initialize component monitors with default states
            for component, monitors in self.component_monitors.items():
                for monitor_name in monitors:
                    monitors[monitor_name] = False

            self.logger.info("Component monitors setup complete")
        except Exception as e:
            self.logger.error(f"Error setting up component monitors: {e}")

    @handle_specific_errors(
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

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="supervision step",
    )
    async def _perform_supervision(self) -> None:
        # Perform health checks
        await self._monitor_system_health()

        # Monitor component-specific features
        await self._monitor_component_features()

        # Coordinate components
        await self._coordinate_components()

        # Update online learning
        await self._update_online_learning()

        # Update supervision results
        await self._update_supervision_results()

        # Check for recovery needs
        await self._check_recovery_needs()

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="system health monitoring",
    )
    async def _monitor_system_health(self) -> None:
        try:
            # Check critical components health
            for component in self.critical_components:
                health_status = await self._check_component_health(component)
                self.health_checks[component] = health_status

                if not health_status:
                    self.logger.warning(f"⚠️ Component {component} health check failed")
                    await self._trigger_recovery(component)

            # Log overall health status
            healthy_components = sum(self.health_checks.values())
            total_components = len(self.health_checks)
            health_percentage = (
                (healthy_components / total_components) * 100
                if total_components > 0
                else 0
            )

            self.logger.info(
                f"System health: {health_percentage:.1f}% ({healthy_components}/{total_components} components healthy)",
            )

        except Exception as e:
            self.logger.error(f"Error monitoring system health: {e}")

    def _monitor_analyst_features(self) -> None:
        """Monitor Analyst component features."""
        if "analyst" not in self.components or not self.components["analyst"]:
            return
        
        analyst = self.components["analyst"]
        analyst_monitors = self.component_monitors["analyst"]
        
        # Define analyst features to monitor
        analyst_features = {
            "dual_model_system": "dual_model_system",
            "market_health_analyzer": "market_health_analyzer",
            "liquidation_risk_model": "liquidation_risk_model",
            "feature_engineering_orchestrator": "feature_engineering_orchestrator",
            "ml_confidence_predictor": "ml_confidence_predictor",
            "regime_classifier": "regime_classifier"
        }
        
        # Monitor each feature
        for monitor_key, feature_name in analyst_features.items():
            analyst_monitors[monitor_key] = (
                hasattr(analyst, feature_name) and getattr(analyst, feature_name) is not None
            )


    def _monitor_strategist_features(self) -> None:
        """Monitor Strategist component features."""
        if "strategist" not in self.components or not self.components["strategist"]:
            return
        
        strategist = self.components["strategist"]
        strategist_monitors = self.component_monitors["strategist"]
        
        # Define strategist features to monitor
        strategist_features = {
            "regime_classifier": "regime_classifier",
            "ml_confidence_predictor": "ml_confidence_predictor",
            "volatility_targeting": "volatility_info"
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
            "ml_predictions": "ml_predictions"
        }
        
        # Monitor each feature
        for monitor_key, feature_name in tactician_features.items():
            tactician_monitors[monitor_key] = (
                hasattr(tactician, feature_name) and getattr(tactician, feature_name) is not None
            )


    def _monitor_enhanced_training_manager_features(self) -> None:
        """Monitor Enhanced Training Manager component features."""
        if ("enhanced_training_manager" not in self.components or 
            not self.components["enhanced_training_manager"]):
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
            "ensemble_creator": "ensemble_creator"
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
                    f"{component} features: {feature_percentage:.1f}% ({active_features}/{total_features} active)"
                )


    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="component features monitoring",
    )
    async def _monitor_component_features(self) -> None:
        """Monitor component-specific features and sub-components."""
        try:
            # Monitor each component's features
            self._monitor_analyst_features()
            self._monitor_strategist_features()
            self._monitor_tactician_features()
            self._monitor_enhanced_training_manager_features()
            
            # Log component feature status
            self._log_component_feature_status()

        except Exception as e:
            self.logger.error(f"Error monitoring component features: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="component health check",
    )
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
            self.logger.error(f"Error checking health for component {component}: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="component coordination",
    )
    async def _coordinate_components(self) -> None:
        try:
            # Enhanced coordination between components with new features
            coordination_results = {
                "timestamp": time.time(), # Changed from datetime.now() to time.time()
                "components_coordinated": len(self.components),
                "status": "coordinated",
                "component_features": self.component_monitors.copy(),
            }

            # Coordinate Analyst with Strategist
            if self.components.get("analyst") and self.components.get("strategist"):
                await self._coordinate_analyst_strategist()

            # Coordinate Strategist with Tactician
            if self.components.get("strategist") and self.components.get("tactician"):
                await self._coordinate_strategist_tactician()

            # Coordinate Training Manager with other components
            if self.components.get("enhanced_training_manager"):
                await self._coordinate_training_manager()

            self.supervision_results["coordination"] = coordination_results

        except Exception as e:
            self.logger.error(f"Error coordinating components: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="analyst strategist coordination",
    )
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
            if (
                hasattr(analyst, "ml_confidence_predictor")
                and analyst.ml_confidence_predictor
            ):
                ml_predictions = await analyst._perform_ml_predictions({})
                if ml_predictions and hasattr(strategist, "ml_confidence_predictor"):
                    strategist.ml_confidence_predictor = ml_predictions

            self.logger.info("Analyst-Strategist coordination completed")

        except Exception as e:
            self.logger.error(f"Error coordinating Analyst-Strategist: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="strategist tactician coordination",
    )
    async def _coordinate_strategist_tactician(self) -> None:
        """Coordinate Strategist and Tactician components."""
        try:
            strategist = self.components["strategist"]
            tactician = self.components["tactician"]

            # Share volatility targeting information
            if hasattr(strategist, "volatility_info") and strategist.volatility_info:
                if hasattr(tactician, "position_sizer") and tactician.position_sizer:
                    # Pass volatility info to position sizer
                    tactician.position_sizer.volatility_info = (
                        strategist.volatility_info
                    )

            # Share regime information for tactical decisions
            if hasattr(strategist, "current_regime") and strategist.current_regime:
                if hasattr(tactician, "current_regime"):
                    tactician.current_regime = strategist.current_regime

            self.logger.info("Strategist-Tactician coordination completed")

        except Exception as e:
            self.logger.error(f"Error coordinating Strategist-Tactician: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="training manager coordination",
    )
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

        except Exception as e:
            self.logger.error(f"Error coordinating Training Manager: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="online learning update",
    )
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
                "timestamp": time.time(), # Changed from datetime.now() to time.time()
                "model_weights": updated_weights,
                "model_performances": self.online_learning.get_model_performances(),
            }

            self.logger.info(f"Online learning updated: {updated_weights}")

        except Exception as e:
            self.logger.error(f"Error updating online learning: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="recovery trigger",
    )
    async def _trigger_recovery(self, component: str) -> None:
        """Trigger recovery for a failed component."""
        try:
            current_time = time.time()
            last_attempt = self.last_recovery_attempt.get(component, 0)

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

        except Exception as e:
            self.logger.error(f"Error triggering recovery for {component}: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="recovery attempt",
    )
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

        except Exception as e:
            self.logger.error(f"Error attempting recovery for {component}: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="database recovery",
    )
    async def _recover_database(self) -> bool:
        """Recover database connection."""
        try:
            # Implement database recovery logic
            self.logger.info("Attempting database recovery...")
            # Mock recovery - replace with actual database reconnection logic
            await asyncio.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"Database recovery failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="exchange recovery",
    )
    async def _recover_exchange(self) -> bool:
        """Recover exchange connection."""
        try:
            # Implement exchange recovery logic
            self.logger.info("Attempting exchange recovery...")
            # Mock recovery - replace with actual exchange reconnection logic
            await asyncio.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"Exchange recovery failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="analyst recovery",
    )
    async def _recover_analyst(self) -> bool:
        """Recover analyst component."""
        try:
            # Implement analyst recovery logic
            self.logger.info("Attempting analyst recovery...")
            # Mock recovery - replace with actual analyst restart logic
            await asyncio.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"Analyst recovery failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="strategist recovery",
    )
    async def _recover_strategist(self) -> bool:
        """Recover strategist component."""
        try:
            # Implement strategist recovery logic
            self.logger.info("Attempting strategist recovery...")
            # Mock recovery - replace with actual strategist restart logic
            await asyncio.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"Strategist recovery failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="tactician recovery",
    )
    async def _recover_tactician(self) -> bool:
        """Recover tactician component."""
        try:
            # Implement tactician recovery logic
            self.logger.info("Attempting tactician recovery...")
            # Mock recovery - replace with actual tactician restart logic
            await asyncio.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"Tactician recovery failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="enhanced training manager recovery",
    )
    async def _recover_enhanced_training_manager(self) -> bool:
        """Recover enhanced training manager component."""
        try:
            # Implement enhanced training manager recovery logic
            self.logger.info("Attempting enhanced training manager recovery...")
            # Mock recovery - replace with actual training manager restart logic
            await asyncio.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"Enhanced training manager recovery failed: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="generic recovery",
    )
    async def _generic_recovery(self, component: str) -> bool:
        """Generic recovery for unspecified components."""
        try:
            self.logger.info(f"Attempting generic recovery for {component}...")
            # Mock recovery - replace with actual restart logic
            await asyncio.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"Generic recovery failed for {component}: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="recovery needs check",
    )
    async def _check_recovery_needs(self) -> None:
        """Check if any components need recovery."""
        try:
            for component, health_status in self.health_checks.items():
                if not health_status:
                    await self._trigger_recovery(component)

        except Exception as e:
            self.logger.error(f"Error checking recovery needs: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="supervision results update",
    )
    async def _update_supervision_results(self) -> None:
        try:
            # Add timestamp
            self.supervision_results["timestamp"] = time.time() # Changed from datetime.now() to time.time()

            # Add health status
            self.supervision_results["health_status"] = self.health_checks.copy()

            # Add component monitors status
            self.supervision_results["component_monitors"] = (
                self.component_monitors.copy()
            )

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

        except Exception as e:
            self.logger.error(f"Error updating supervision results: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="supervisor stop",
    )
    async def stop(self) -> None:
        self.logger.info("🛑 Stopping Supervisor...")
        try:
            self.is_running = False
            self.logger.info("✅ Supervisor stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping supervisor: {e}")

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


supervisor: Supervisor | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="supervisor setup",
)
async def setup_supervisor(
    config: dict[str, Any] | None = None,
) -> Supervisor | None:
    try:
        global supervisor
        if config is None:
            config = {"supervisor": {"supervision_interval": 60, "max_history": 100}}
        supervisor = Supervisor(config)
        success = await supervisor.initialize()
        if success:
            return supervisor
        return None
    except Exception as e:
        print(f"Error setting up supervisor: {e}")
        return None
