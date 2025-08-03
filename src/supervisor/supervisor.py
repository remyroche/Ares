import asyncio
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List
from collections import defaultdict

import pandas as pd

# Try to import AsyncIOScheduler, with fallback if not available
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    AsyncIOScheduler = None

from emails.ares_mailer import AresMailer  # Import AresMailer
from exchange.binance import BinanceExchange
from src.analyst.analyst import Analyst  # Import Analyst
from src.config import CONFIG, settings
from src.paper_trader import PaperTrader  # Import PaperTrader
from src.sentinel.sentinel import Sentinel  # Import Sentinel
from src.strategist.strategist import Strategist
from src.supervisor.performance_monitor import PerformanceMonitor
from src.tactician.tactician import Tactician  # Import Tactician
from src.training.enhanced_training_manager import EnhancedTrainingManager
from src.utils.error_handler import (
    get_logged_exceptions,
    handle_errors,
    handle_file_operations,
    handle_network_operations,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.model_manager import ModelManager
from src.utils.state_manager import StateManager
from src.core.config_service import ConfigurationService

from .dynamic_weighter import DynamicWeighter


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
                self.model_performances[model_id] = self.model_performances[model_id][-100:]
            
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
                self.model_weights = {model_id: equal_weight for model_id in avg_performances}
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
    """
    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("Supervisor")
        self.is_running: bool = False
        self.status: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.supervisor_config: dict[str, Any] = self.config.get("supervisor", {})
        self.supervision_interval: int = self.supervisor_config.get("supervision_interval", 60)
        self.max_history: int = self.supervisor_config.get("max_history", 100)
        self.supervision_results: dict[str, Any] = {}
        self.components: dict[str, Any] = {}
        
        # Advanced error handling and recovery
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.recovery_attempts: dict[str, int] = defaultdict(int)
        self.max_recovery_attempts: int = self.supervisor_config.get("max_recovery_attempts", 3)
        self.recovery_cooldown: int = self.supervisor_config.get("recovery_cooldown", 300)  # 5 minutes
        self.last_recovery_attempt: dict[str, float] = {}
        
        # Online learning for model weighting
        self.online_learning = OnlineLearningManager(self.supervisor_config.get("online_learning", {}))
        
        # Health monitoring
        self.health_checks: dict[str, bool] = {}
        self.critical_components: list[str] = ["database", "exchange", "analyst", "strategist"]

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
            # Initialize critical components
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

    @handle_specific_errors(
        error_handlers={
            Exception: (False, "Supervisor run failed"),
        },
        default_return=False,
        context="supervisor run",
    )
    async def run(self) -> bool:
        try:
            self.is_running = True
            self.logger.info("🚦 Supervisor started.")
            while self.is_running:
                await self._perform_supervision()
                await asyncio.sleep(self.supervision_interval)
            return True
        except Exception as e:
            self.logger.error(f"Error in supervisor run: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="supervision step",
    )
    async def _perform_supervision(self) -> None:
        try:
            # Perform health checks
            await self._monitor_system_health()
            
            # Coordinate components
            await self._coordinate_components()
            
            # Update online learning
            await self._update_online_learning()
            
            # Update supervision results
            await self._update_supervision_results()
            
            # Check for recovery needs
            await self._check_recovery_needs()
            
        except Exception as e:
            self.logger.error(f"Error in supervision step: {e}")

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
            health_percentage = (healthy_components / total_components) * 100 if total_components > 0 else 0
            
            self.logger.info(f"System health: {health_percentage:.1f}% ({healthy_components}/{total_components} components healthy)")
            
        except Exception as e:
            self.logger.error(f"Error monitoring system health: {e}")

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
            # Coordinate between components
            # This is where you would implement the actual coordination logic
            coordination_results = {
                "timestamp": datetime.now().isoformat(),
                "components_coordinated": len(self.components),
                "status": "coordinated"
            }
            
            self.supervision_results["coordination"] = coordination_results
            
        except Exception as e:
            self.logger.error(f"Error coordinating components: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="online learning update",
    )
    async def _update_online_learning(self) -> None:
        """Update online learning with current performance data."""
        try:
            # Get current model performances (mock data - replace with actual)
            model_performances = {
                "model_1": 0.75,
                "model_2": 0.82,
                "model_3": 0.68
            }
            
            # Update online learning with current performances
            for model_id, performance in model_performances.items():
                await self.online_learning.update_model_performance(model_id, performance)
            
            # Get updated weights
            updated_weights = self.online_learning.get_model_weights()
            self.supervision_results["online_learning"] = {
                "timestamp": datetime.now().isoformat(),
                "model_weights": updated_weights,
                "model_performances": self.online_learning.get_model_performances()
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
            if (current_time - last_attempt < self.recovery_cooldown or 
                self.recovery_attempts[component] >= self.max_recovery_attempts):
                return
            
            self.logger.info(f"🔄 Triggering recovery for component: {component}")
            
            # Attempt recovery
            recovery_success = await self._attempt_recovery(component)
            
            if recovery_success:
                self.logger.info(f"✅ Recovery successful for component: {component}")
                self.recovery_attempts[component] = 0
            else:
                self.recovery_attempts[component] += 1
                self.logger.warning(f"⚠️ Recovery failed for component: {component} (attempt {self.recovery_attempts[component]}/{self.max_recovery_attempts})")
            
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
            elif component == "exchange":
                return await self._recover_exchange()
            elif component == "analyst":
                return await self._recover_analyst()
            else:
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
            self.supervision_results["timestamp"] = datetime.now().isoformat()
            
            # Add health status
            self.supervision_results["health_status"] = self.health_checks.copy()
            
            # Add recovery status
            self.supervision_results["recovery_status"] = {
                "recovery_attempts": dict(self.recovery_attempts),
                "last_recovery_attempts": self.last_recovery_attempt.copy()
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
            "recovery_attempts": dict(self.recovery_attempts),
            "online_learning_weights": self.online_learning.get_model_weights()
        }

    def get_history(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
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
            "max_weight": self.online_learning.max_weight
        }


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
