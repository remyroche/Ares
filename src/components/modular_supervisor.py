# src/components/modular_supervisor.py

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import asyncio
import json
import traceback

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.warning_symbols import error, failed, initialization_error, invalid, missing


class ModularSupervisor:
    """
    Enhanced modular supervisor with comprehensive error handling and type safety.
    
    This class provides system supervision capabilities including performance monitoring,
    risk monitoring, system health checks, and coordination between components.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the ModularSupervisor with configuration.
        
        Args:
            config: Configuration dictionary containing supervisor settings
        """
        self.config: Dict[str, Any] = config
        self.logger = system_logger.getChild("ModularSupervisor")
        
        # Supervision state
        self.is_supervising: bool = False
        self.supervision_results: Dict[str, Any] = {}
        self.supervision_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.supervisor_config: Dict[str, Any] = self.config.get("modular_supervisor", {})
        self.supervision_interval: int = self.supervisor_config.get("supervision_interval", 60)
        self.max_supervision_history: int = self.supervisor_config.get("max_supervision_history", 100)
        self.enable_performance_monitoring: bool = self.supervisor_config.get("enable_performance_monitoring", True)
        self.enable_risk_monitoring: bool = self.supervisor_config.get("enable_risk_monitoring", True)
        self.enable_system_health_checks: bool = self.supervisor_config.get("enable_system_health_checks", True)
        self.enable_component_coordination: bool = self.supervisor_config.get("enable_component_coordination", True)
        
        # Supervision modules
        self.performance_monitor = None
        self.risk_monitor = None
        self.health_checker = None
        self.component_coordinator = None
        
        # Component references
        self.analyst = None
        self.strategist = None
        self.tactician = None
        
        self.logger.info("ModularSupervisor initialized with configuration")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid modular supervisor configuration"),
            AttributeError: (False, "Missing required supervisor parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="modular supervisor initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize the supervisor and all its modules.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Modular Supervisor...")
            
            # Load supervisor configuration
            await self._load_supervisor_configuration()
            
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(invalid("Invalid configuration for modular supervisor"))
                return False
            
            # Initialize supervision modules
            await self._initialize_supervision_modules()
            
            self.logger.info("✅ Modular Supervisor initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(failed(f"❌ Modular Supervisor initialization failed: {e}"))
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="supervisor configuration loading",
    )
    async def _load_supervisor_configuration(self) -> None:
        """
        Load and validate supervisor configuration.
        """
        try:
            # Set default supervisor parameters
            self.supervisor_config.setdefault("supervision_interval", 60)
            self.supervisor_config.setdefault("max_supervision_history", 100)
            self.supervisor_config.setdefault("enable_performance_monitoring", True)
            self.supervisor_config.setdefault("enable_risk_monitoring", True)
            self.supervisor_config.setdefault("enable_system_health_checks", True)
            self.supervisor_config.setdefault("enable_component_coordination", True)
            
            # Update configuration
            self.supervision_interval = self.supervisor_config["supervision_interval"]
            self.max_supervision_history = self.supervisor_config["max_supervision_history"]
            self.enable_performance_monitoring = self.supervisor_config["enable_performance_monitoring"]
            self.enable_risk_monitoring = self.supervisor_config["enable_risk_monitoring"]
            self.enable_system_health_checks = self.supervisor_config["enable_system_health_checks"]
            self.enable_component_coordination = self.supervisor_config["enable_component_coordination"]
            
            self.logger.info("Supervisor configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading supervisor configuration: {e}")
            raise

    def _validate_configuration(self) -> bool:
        """
        Validate the supervisor configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            required_keys = ["supervision_interval", "max_supervision_history"]
            for key in required_keys:
                if key not in self.supervisor_config:
                    self.logger.error(missing(f"Missing required configuration key: {key}"))
                    return False
            
            if self.supervision_interval <= 0:
                self.logger.error(invalid("Supervision interval must be positive"))
                return False
                
            if self.max_supervision_history <= 0:
                self.logger.error(invalid("Max supervision history must be positive"))
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    async def _initialize_supervision_modules(self) -> None:
        """
        Initialize all supervision modules based on configuration.
        """
        try:
            if self.enable_performance_monitoring:
                self.performance_monitor = PerformanceMonitor(self.supervisor_config)
                self.logger.info("Performance monitor initialized")
            
            if self.enable_risk_monitoring:
                self.risk_monitor = RiskMonitor(self.supervisor_config)
                self.logger.info("Risk monitor initialized")
            
            if self.enable_system_health_checks:
                self.health_checker = SystemHealthChecker(self.supervisor_config)
                self.logger.info("System health checker initialized")
            
            if self.enable_component_coordination:
                self.component_coordinator = ComponentCoordinator(self.supervisor_config)
                self.logger.info("Component coordinator initialized")
                
        except Exception as e:
            self.logger.error(f"Error initializing supervision modules: {e}")
            raise

    def register_components(self, analyst=None, strategist=None, tactician=None) -> None:
        """
        Register component references for supervision.
        
        Args:
            analyst: ModularAnalyst instance
            strategist: ModularStrategist instance
            tactician: ModularTactician instance
        """
        try:
            if analyst:
                self.analyst = analyst
                self.logger.info("Analyst component registered")
            
            if strategist:
                self.strategist = strategist
                self.logger.info("Strategist component registered")
            
            if tactician:
                self.tactician = tactician
                self.logger.info("Tactician component registered")
                
        except Exception as e:
            self.logger.error(f"Error registering components: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError, RuntimeError),
        default_return=None,
        context="system supervision",
    )
    async def supervise_system(self, system_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Perform comprehensive system supervision.
        
        Args:
            system_state: Current system state information
            
        Returns:
            Dict containing supervision results or None if supervision fails
        """
        try:
            if self.is_supervising:
                self.logger.warning("System supervision already in progress")
                return None
            
            self.is_supervising = True
            self.logger.info("Starting system supervision...")
            
            supervision_result = {
                "timestamp": datetime.now().isoformat(),
                "system_state": system_state,
                "performance_metrics": None,
                "risk_assessment": None,
                "system_health": None,
                "component_status": None,
                "coordination_status": None,
                "overall_supervision_score": 0.0,
                "recommendations": []
            }
            
            # Monitor performance
            if self.performance_monitor and self.enable_performance_monitoring:
                try:
                    supervision_result["performance_metrics"] = await self.performance_monitor.analyze_performance(
                        system_state
                    )
                except Exception as e:
                    self.logger.error(f"Performance monitoring failed: {e}")
            
            # Assess risk
            if self.risk_monitor and self.enable_risk_monitoring:
                try:
                    supervision_result["risk_assessment"] = await self.risk_monitor.assess_system_risk(
                        system_state
                    )
                except Exception as e:
                    self.logger.error(f"Risk assessment failed: {e}")
            
            # Check system health
            if self.health_checker and self.enable_system_health_checks:
                try:
                    supervision_result["system_health"] = await self.health_checker.check_system_health(
                        system_state
                    )
                except Exception as e:
                    self.logger.error(f"System health check failed: {e}")
            
            # Check component status
            supervision_result["component_status"] = await self._check_component_status()
            
            # Coordinate components
            if self.component_coordinator and self.enable_component_coordination:
                try:
                    supervision_result["coordination_status"] = await self.component_coordinator.coordinate_components(
                        self.analyst, self.strategist, self.tactician
                    )
                except Exception as e:
                    self.logger.error(f"Component coordination failed: {e}")
            
            # Calculate overall supervision score
            supervision_result["overall_supervision_score"] = self._calculate_supervision_score(supervision_result)
            
            # Generate recommendations
            supervision_result["recommendations"] = self._generate_recommendations(supervision_result)
            
            # Store results
            self.supervision_results = supervision_result
            self._add_to_history(supervision_result)
            
            self.logger.info(f"System supervision completed. Overall score: {supervision_result['overall_supervision_score']:.2f}")
            return supervision_result
            
        except Exception as e:
            self.logger.error(f"System supervision failed: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
            
        finally:
            self.is_supervising = False

    async def _check_component_status(self) -> Dict[str, Any]:
        """
        Check the status of all registered components.
        
        Returns:
            Dictionary containing component status information
        """
        component_status = {}
        
        try:
            if self.analyst:
                try:
                    component_status["analyst"] = self.analyst.get_status()
                except Exception as e:
                    component_status["analyst"] = {"error": str(e), "status": "unknown"}
            
            if self.strategist:
                try:
                    component_status["strategist"] = self.strategist.get_status()
                except Exception as e:
                    component_status["strategist"] = {"error": str(e), "status": "unknown"}
            
            if self.tactician:
                try:
                    component_status["tactician"] = self.tactician.get_status()
                except Exception as e:
                    component_status["tactician"] = {"error": str(e), "status": "unknown"}
            
            # Overall component health
            healthy_components = sum(1 for status in component_status.values() if "error" not in status)
            total_components = len(component_status)
            
            component_status["overall_health"] = {
                "healthy_components": healthy_components,
                "total_components": total_components,
                "health_percentage": (healthy_components / total_components * 100) if total_components > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error checking component status: {e}")
            component_status["error"] = str(e)
        
        return component_status

    def _calculate_supervision_score(self, supervision_result: Dict[str, Any]) -> float:
        """
        Calculate overall supervision score based on individual components.
        
        Args:
            supervision_result: Supervision results dictionary
            
        Returns:
            float: Overall score between 0.0 and 1.0
        """
        try:
            scores = []
            weights = []
            
            # Performance metrics score
            if supervision_result["performance_metrics"]:
                perf_score = supervision_result["performance_metrics"].get("score", 0.0)
                scores.append(perf_score)
                weights.append(0.25)
            
            # Risk assessment score (inverted - lower risk = higher score)
            if supervision_result["risk_assessment"]:
                risk_score = 1.0 - supervision_result["risk_assessment"].get("risk_level", 0.5)
                scores.append(risk_score)
                weights.append(0.25)
            
            # System health score
            if supervision_result["system_health"]:
                health_score = supervision_result["system_health"].get("health_score", 0.0)
                scores.append(health_score)
                weights.append(0.2)
            
            # Component status score
            if supervision_result["component_status"]:
                comp_health = supervision_result["component_status"].get("overall_health", {})
                comp_score = comp_health.get("health_percentage", 0.0) / 100.0
                scores.append(comp_score)
                weights.append(0.3)
            
            if not scores:
                return 0.0
            
            # Calculate weighted average
            total_weight = sum(weights)
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating supervision score: {e}")
            return 0.0

    def _generate_recommendations(self, supervision_result: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on supervision results.
        
        Args:
            supervision_result: Supervision results dictionary
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        try:
            overall_score = supervision_result.get("overall_supervision_score", 0.0)
            
            if overall_score >= 0.8:
                recommendations.append("System operating optimally - continue current operations")
            elif overall_score >= 0.6:
                recommendations.append("System operating well - minor optimizations recommended")
            elif overall_score >= 0.4:
                recommendations.append("System showing issues - review and address problems")
            else:
                recommendations.append("System experiencing significant issues - immediate attention required")
            
            # Add specific recommendations based on individual components
            if supervision_result.get("performance_metrics"):
                perf_metrics = supervision_result["performance_metrics"]
                if perf_metrics.get("score", 1.0) < 0.7:
                    recommendations.append("Performance below optimal levels - investigate bottlenecks")
            
            if supervision_result.get("risk_assessment"):
                risk_level = supervision_result["risk_assessment"].get("risk_level", 0.5)
                if risk_level > 0.7:
                    recommendations.append("High system risk detected - implement risk mitigation measures")
                elif risk_level > 0.5:
                    recommendations.append("Elevated system risk - monitor closely and prepare mitigation")
            
            if supervision_result.get("system_health"):
                health_status = supervision_result["system_health"].get("status", "unknown")
                if health_status != "healthy":
                    recommendations.append(f"System health issue detected: {health_status} - investigate and resolve")
            
            if supervision_result.get("component_status"):
                comp_health = supervision_result["component_status"].get("overall_health", {})
                health_percentage = comp_health.get("health_percentage", 100)
                if health_percentage < 80:
                    recommendations.append(f"Component health below threshold ({health_percentage}%) - review component status")
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations due to supervision errors")
        
        return recommendations

    def _add_to_history(self, supervision_result: Dict[str, Any]) -> None:
        """
        Add supervision result to history, maintaining maximum history size.
        
        Args:
            supervision_result: Supervision result to add
        """
        try:
            self.supervision_history.append(supervision_result)
            
            # Maintain maximum history size
            if len(self.supervision_history) > self.max_supervision_history:
                self.supervision_history.pop(0)
                
        except Exception as e:
            self.logger.error(f"Error adding to history: {e}")

    def get_supervision_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get supervision history.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of supervision results
        """
        try:
            if limit is None:
                return self.supervision_history.copy()
            else:
                return self.supervision_history[-limit:].copy()
        except Exception as e:
            self.logger.error(f"Error retrieving supervision history: {e}")
            return []

    def get_latest_supervision_result(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent supervision result.
        
        Returns:
            Latest supervision result or None if no supervision performed
        """
        try:
            if self.supervision_history:
                return self.supervision_history[-1].copy()
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving latest supervision result: {e}")
            return None

    def clear_history(self) -> None:
        """Clear supervision history."""
        try:
            self.supervision_history.clear()
            self.logger.info("Supervision history cleared")
        except Exception as e:
            self.logger.error(f"Error clearing history: {e}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current supervisor status.
        
        Returns:
            Dictionary containing current status information
        """
        try:
            return {
                "is_supervising": self.is_supervising,
                "supervision_interval": self.supervision_interval,
                "history_size": len(self.supervision_history),
                "max_history_size": self.max_supervision_history,
                "enabled_modules": {
                    "performance_monitoring": self.enable_performance_monitoring,
                    "risk_monitoring": self.enable_risk_monitoring,
                    "system_health_checks": self.enable_system_health_checks,
                    "component_coordination": self.enable_component_coordination
                },
                "registered_components": {
                    "analyst": self.analyst is not None,
                    "strategist": self.strategist is not None,
                    "tactician": self.tactician is not None
                },
                "last_supervision": self.supervision_history[-1]["timestamp"] if self.supervision_history else None
            }
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {}


# Placeholder classes for supervision modules
class PerformanceMonitor:
    """Placeholder for performance monitoring module."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def analyze_performance(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder performance analysis."""
        return {
            "score": 0.85,
            "cpu_usage": 0.45,
            "memory_usage": 0.62,
            "response_time": 0.12,
            "throughput": 1000
        }


class RiskMonitor:
    """Placeholder for risk monitoring module."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def assess_system_risk(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder risk assessment."""
        return {
            "risk_level": 0.25,
            "risk_factors": [],
            "mitigation_strategies": [],
            "confidence": 0.9
        }


class SystemHealthChecker:
    """Placeholder for system health checking module."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def check_system_health(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder system health check."""
        return {
            "health_score": 0.92,
            "status": "healthy",
            "issues": [],
            "last_check": datetime.now().isoformat()
        }


class ComponentCoordinator:
    """Placeholder for component coordination module."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def coordinate_components(self, analyst, strategist, tactician) -> Dict[str, Any]:
        """Placeholder component coordination."""
        return {
            "coordination_score": 0.88,
            "component_sync": True,
            "data_flow": "optimal",
            "last_coordination": datetime.now().isoformat()
        }
