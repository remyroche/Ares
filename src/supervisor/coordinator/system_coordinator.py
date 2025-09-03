"""
System Coordinator Module.

This is the main coordinator that orchestrates all the coordinator components
for system-level supervision.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.core.decorators import handles_errors
from src.core.domain import handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import error, failed, initialization_error, invalid

from .circuit_breaker import CircuitBreaker
from .component_monitor import ComponentMonitor
from .health_monitor import HealthMonitor
from .online_learning_manager import OnlineLearningManager
from .recovery_manager import RecoveryManager


class SystemCoordinator:
    """
    System-Level Coordinator component responsible for:
    - System Health Monitoring: Monitor all component health and performance
    - Circuit Breaker Management: Handle failures and recovery across all components
    - Component Coordination: Orchestrate communication between components
    - Portfolio-Level Risk Management: Global portfolio guards and kill-switches
    - Performance Tracking: System-wide performance monitoring and reporting
    - Online Learning: Model weighting based on system performance
    - Recovery Management: Automatic recovery and fallback mechanisms
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize System Coordinator.
        
        Args:
            config: Configuration dictionary
        """
        self.config: Dict[str, Any] = config
        self.logger = system_logger.getChild("SystemCoordinator")
        self.is_running: bool = False
        self.status: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.supervisor_config: Dict[str, Any] = self.config.get("supervisor", {})
        self.supervision_interval: int = self.supervisor_config.get(
            "supervision_interval",
            60,
        )
        self.max_history: int = self.supervisor_config.get("max_history", 100)
        self.supervision_results: Dict[str, Any] = {}
        self.components: Dict[str, Any] = {}

        # Initialize sub-components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.online_learning = OnlineLearningManager(
            self.supervisor_config.get("online_learning", {})
        )
        self.component_monitor = ComponentMonitor(self.supervisor_config)
        self.health_monitor = HealthMonitor(self.supervisor_config)
        self.recovery_manager = RecoveryManager(self.supervisor_config)

        # Enhanced prediction service configuration
        self.enhanced_prediction_service = None
        self.is_initialized: bool = False
        self.enhanced_prediction_service_config = self.supervisor_config.get(
            "enhanced_prediction_service", {}
        )
        self.entry_threshold: float = self.enhanced_prediction_service_config.get(
            "entry_threshold", 0.7
        )
        self.max_confidence_threshold: float = self.enhanced_prediction_service_config.get(
            "max_confidence_threshold", 0.9
        )

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
        """Initialize the System Coordinator."""
        try:
            self.logger.info("Initializing System Coordinator...")
            await self._load_supervisor_configuration()
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for supervisor")
                return False
            
            await self._initialize_components()
            await self._setup_circuit_breakers()
            await self._initialize_enhanced_prediction_service()
            
            self.is_initialized = True
            self.logger.info("✅ System Coordinator initialization completed successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ System Coordinator initialization failed: {e}")
            return False

    async def _load_supervisor_configuration(self) -> None:
        """Load supervisor configuration."""
        try:
            self.supervision_interval = self.supervisor_config.get("supervision_interval", 60)
            self.max_history = self.supervisor_config.get("max_history", 100)
            self.logger.info("Supervisor configuration loaded successfully")
        except Exception as e:
            self.logger.exception(f"Error loading supervisor configuration: {e}")

    def _validate_configuration(self) -> bool:
        """Validate configuration."""
        try:
            if self.supervision_interval <= 0:
                self.logger.error("Invalid supervision interval")
                return False
            if self.max_history <= 0:
                self.logger.error("Invalid max history")
                return False
            return True
        except Exception as e:
            self.logger.exception(f"Error validating configuration: {e}")
            return False

    async def _initialize_components(self) -> None:
        """Initialize coordinator sub-components."""
        # Components are already initialized in __init__
        self.logger.info("Coordinator components initialized")

    async def _setup_circuit_breakers(self) -> None:
        """Set up circuit breakers for each component."""
        component_names = ["analyst", "strategist", "tactician", "sentinel", 
                          "enhanced_training_manager", "risk_allocator"]
        
        for component in component_names:
            self.circuit_breakers[component] = CircuitBreaker(
                failure_threshold=self.supervisor_config.get("circuit_breaker_threshold", 5),
                timeout=self.supervisor_config.get("circuit_breaker_timeout", 60)
            )
        
        self.logger.info("Circuit breakers configured for all components")

    async def _initialize_enhanced_prediction_service(self) -> bool:
        """Initialize the enhanced prediction service."""
        try:
            if self.enhanced_prediction_service_config.get("enabled", True):
                from src.supervisor.enhanced_prediction_service import EnhancedPredictionService
                self.enhanced_prediction_service = EnhancedPredictionService(
                    self.enhanced_prediction_service_config
                )
                self.logger.info("Enhanced prediction service initialized")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced prediction service: {e}")
            return False

    @handles_errors(
        error_handlers={
            Exception: (False, "Supervisor run failed"),
        },
        default_return=False,
        context="supervisor run",
    )
    async def run(self) -> bool:
        """Run the supervisor main loop."""
        try:
            self.is_running = True
            self.logger.info("🚦 System Coordinator started.")
            
            while self.is_running:
                await self._perform_supervision()
                await asyncio.sleep(self.supervision_interval)
            
            return True
        except Exception as e:
            self.logger.exception(f"Error in supervisor run: {e}")
            self.is_running = False
            return False

    async def _perform_supervision(self) -> None:
        """Perform supervision tasks."""
        try:
            now = datetime.now()
            self.logger.info(f"Performing supervision at {now}")

            # Monitor system health
            health_status = await self.health_monitor.check_system_health()
            
            # Monitor components
            await self._monitor_all_components()
            
            # Coordinate components
            await self._coordinate_components()
            
            # Update online learning
            await self._update_online_learning()
            
            # Handle any failures
            await self._handle_system_failures()
            
            # Update status
            self.status = {
                "timestamp": now.isoformat(),
                "is_running": self.is_running,
                "health_status": health_status.get("overall_status", "unknown"),
                "active_components": len(self.components),
                "supervision_results": self.supervision_results,
            }
            
            # Update history
            self._update_history()
            
        except Exception as e:
            self.logger.error(error(f"Error in supervision: {e}"))

    async def _monitor_all_components(self) -> None:
        """Monitor all registered components."""
        for name, component in self.components.items():
            try:
                if name == "analyst":
                    self.component_monitor.monitor_analyst_features(component)
                elif name == "strategist":
                    self.component_monitor.monitor_strategist_features(component)
                elif name == "tactician":
                    self.component_monitor.monitor_tactician_features(component)
                elif name == "enhanced_training_manager":
                    self.component_monitor.monitor_training_manager_features(component)
            except Exception as e:
                self.logger.error(f"Error monitoring {name}: {e}")

    async def _coordinate_components(self) -> None:
        """Coordinate components with clear separation of responsibilities."""
        try:
            # Coordinate Analyst-Strategist
            await self._coordinate_analyst_strategist()
            
            # Coordinate Strategist-Tactician
            await self._coordinate_strategist_tactician()
            
            # Coordinate Training Manager
            await self._coordinate_training_manager()
            
        except Exception as e:
            self.logger.error(error(f"Error coordinating components: {e}"))

    async def _coordinate_analyst_strategist(self) -> None:
        """Coordinate Analyst and Strategist components."""
        if "analyst" not in self.components or "strategist" not in self.components:
            return
            
        try:
            analyst = self.components["analyst"]
            strategist = self.components["strategist"]
            
            # Share regime classification results
            if hasattr(analyst, "regime_classifier") and analyst.regime_classifier:
                regime_info = getattr(analyst, "regime_info", {})
                if regime_info and hasattr(strategist, "current_regime"):
                    strategist.current_regime = regime_info.get("regime")
                    strategist.regime_confidence = regime_info.get("confidence", 0.0)
            
            self.logger.debug("Analyst-Strategist coordination completed")
            
        except Exception as e:
            self.logger.error(error(f"Error coordinating Analyst-Strategist: {e}"))

    async def _coordinate_strategist_tactician(self) -> None:
        """Coordinate Strategist and Tactician components."""
        if "strategist" not in self.components or "tactician" not in self.components:
            return
            
        try:
            strategist = self.components["strategist"]
            tactician = self.components["tactician"]
            
            # Share strategy information
            if hasattr(strategist, "current_strategy") and strategist.current_strategy:
                if hasattr(tactician, "strategy_input"):
                    tactician.strategy_input = strategist.current_strategy
            
            self.logger.debug("Strategist-Tactician coordination completed")
            
        except Exception as e:
            self.logger.error(error(f"Error coordinating Strategist-Tactician: {e}"))

    async def _coordinate_training_manager(self) -> None:
        """Coordinate Enhanced Training Manager with other components."""
        if "enhanced_training_manager" not in self.components:
            return
            
        try:
            training_manager = self.components["enhanced_training_manager"]
            
            # Coordinate with Analyst for model updates
            if "analyst" in self.components and hasattr(training_manager, "get_enhanced_training_results"):
                training_results = training_manager.get_enhanced_training_results()
                if training_results and hasattr(self.components["analyst"], "update_models"):
                    await self.components["analyst"].update_models(training_results)
            
            self.logger.debug("Training Manager coordination completed")
            
        except Exception as e:
            self.logger.error(error(f"Error coordinating Training Manager: {e}"))

    async def _update_online_learning(self) -> None:
        """Update online learning based on component performance."""
        try:
            # Collect performance metrics from components
            for name, component in self.components.items():
                if hasattr(component, "get_performance_metrics"):
                    metrics = component.get_performance_metrics()
                    if metrics and "performance_score" in metrics:
                        await self.online_learning.update_model_performance(
                            name, metrics["performance_score"]
                        )
            
        except Exception as e:
            self.logger.error(f"Error updating online learning: {e}")

    async def _handle_system_failures(self) -> None:
        """Handle any system failures detected."""
        health_status = self.health_monitor.get_health_status()
        
        if health_status.get("errors"):
            for error_msg in health_status["errors"]:
                self.logger.error(failed(f"System error: {error_msg}"))
                
                # Attempt recovery for critical errors
                if "component" in error_msg:
                    # Extract component name from error message
                    component_name = error_msg.split("component")[1].split()[0]
                    await self.recovery_manager.handle_component_failure(
                        component_name,
                        {"error_type": "critical", "message": error_msg}
                    )

    def _update_history(self) -> None:
        """Update supervision history."""
        self.history.append(self.status.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def register_component(self, name: str, component: Any) -> None:
        """Register a component for supervision."""
        self.components[name] = component
        self.logger.info(f"Registered component: {name}")

    def get_status(self) -> Dict[str, Any]:
        """Get current supervisor status."""
        return self.status.copy()

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get supervision history."""
        if limit:
            return self.history[-limit:]
        return self.history.copy()

    async def stop(self) -> None:
        """Stop the System Coordinator."""
        self.logger.info("🛑 Stopping System Coordinator...")
        self.is_running = False
        self.status = {
            "timestamp": datetime.now().isoformat(),
            "is_running": False,
            "message": "System Coordinator stopped",
        }
        self.logger.info("✅ System Coordinator stopped successfully")