# src/components/modular_supervisor.py

"""
Enhanced modular supervisor with comprehensive error handling and type safety.
Provides monitoring and supervision capabilities for trading system components.
"""

import asyncio
import json
import os
import psutil
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.warning_symbols import error, failed, initialization_error, invalid, missing


@dataclass
class SupervisionMetrics:
    """Data class for storing supervision metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    error_rate: float
    uptime: float
    performance_score: float
    risk_score: float
    system_score: float


@dataclass
class AlertThreshold:
    """Data class for alert thresholds."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    enabled: bool = True


class ModularSupervisor:
    """
    Enhanced modular supervisor with comprehensive error handling and type safety.
    Monitors system performance, risk metrics, and provides alerting capabilities.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the modular supervisor."""
        self.config: Dict[str, Any] = config
        self.logger = system_logger.getChild("ModularSupervisor")

        # Supervision state
        self.is_supervising: bool = False
        self.supervision_results: Dict[str, Any] = {}
        self.supervision_history: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None

        # Configuration
        self.supervisor_config: Dict[str, Any] = self.config.get("modular_supervisor", {})
        self.supervision_interval: int = self.supervisor_config.get("supervision_interval", 60)
        self.max_supervision_history: int = self.supervisor_config.get("max_supervision_history", 100)
        self.enable_performance_monitoring: bool = self.supervisor_config.get("enable_performance_monitoring", True)
        self.enable_risk_monitoring: bool = self.supervisor_config.get("enable_risk_monitoring", True)
        self.enable_system_monitoring: bool = self.supervisor_config.get("enable_system_monitoring", True)
        self.enable_alerting: bool = self.supervisor_config.get("enable_alerting", True)

        # Alert thresholds
        self.alert_thresholds: List[AlertThreshold] = self._initialize_alert_thresholds()

        # Performance tracking
        self.performance_history: List[float] = []
        self.risk_history: List[float] = []
        self.system_history: List[float] = []

    def _initialize_alert_thresholds(self) -> List[AlertThreshold]:
        """Initialize default alert thresholds."""
        default_thresholds = [
            AlertThreshold("cpu_usage", 80.0, 95.0),
            AlertThreshold("memory_usage", 85.0, 95.0),
            AlertThreshold("disk_usage", 90.0, 98.0),
            AlertThreshold("error_rate", 5.0, 15.0),
            AlertThreshold("performance_score", 70.0, 50.0),
            AlertThreshold("risk_score", 30.0, 50.0),
        ]
        
        # Override with config if provided
        config_thresholds = self.supervisor_config.get("alert_thresholds", {})
        for threshold in default_thresholds:
            if threshold.metric_name in config_thresholds:
                config_data = config_thresholds[threshold.metric_name]
                threshold.warning_threshold = config_data.get("warning", threshold.warning_threshold)
                threshold.critical_threshold = config_data.get("critical", threshold.critical_threshold)
                threshold.enabled = config_data.get("enabled", True)
        
        return default_thresholds

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
        """Initialize the modular supervisor."""
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
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="supervisor configuration loading",
    )
    async def _load_supervisor_configuration(self) -> None:
        """Load and validate supervisor configuration."""
        try:
            # Set default supervisor parameters
            self.supervisor_config.setdefault("supervision_interval", 60)
            self.supervisor_config.setdefault("max_supervision_history", 100)
            self.supervisor_config.setdefault("enable_performance_monitoring", True)
            self.supervisor_config.setdefault("enable_risk_monitoring", True)
            self.supervisor_config.setdefault("enable_system_monitoring", True)
            self.supervisor_config.setdefault("enable_alerting", True)

            # Update configuration
            self.supervision_interval = self.supervisor_config["supervision_interval"]
            self.max_supervision_history = self.supervisor_config["max_supervision_history"]
            self.enable_performance_monitoring = self.supervisor_config["enable_performance_monitoring"]
            self.enable_risk_monitoring = self.supervisor_config["enable_risk_monitoring"]
            self.enable_system_monitoring = self.supervisor_config["enable_system_monitoring"]
            self.enable_alerting = self.supervisor_config["enable_alerting"]

            self.logger.info("Supervisor configuration loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading supervisor configuration: {e}")
            raise

    def _validate_configuration(self) -> bool:
        """Validate supervisor configuration."""
        try:
            required_keys = ["supervision_interval", "max_supervision_history"]
            for key in required_keys:
                if key not in self.supervisor_config:
                    self.logger.error(f"Missing required configuration key: {key}")
                    return False

            if self.supervision_interval <= 0:
                self.logger.error("Supervision interval must be positive")
                return False

            if self.max_supervision_history <= 0:
                self.logger.error("Max supervision history must be positive")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="supervision modules initialization",
    )
    async def _initialize_supervision_modules(self) -> None:
        """Initialize all supervision modules."""
        try:
            if self.enable_performance_monitoring:
                await self._initialize_performance_monitoring()

            if self.enable_risk_monitoring:
                await self._initialize_risk_monitoring()

            if self.enable_system_monitoring:
                await self._initialize_system_monitoring()

            if self.enable_alerting:
                await self._initialize_alerting()

            self.logger.info("All supervision modules initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing supervision modules: {e}")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="performance monitoring initialization",
    )
    async def _initialize_performance_monitoring(self) -> None:
        """Initialize performance monitoring module."""
        try:
            self.logger.info("Initializing performance monitoring module")
            # Initialize performance tracking structures
            self.performance_history = []
            self.logger.info("Performance monitoring module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing performance monitoring: {e}")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="risk monitoring initialization",
    )
    async def _initialize_risk_monitoring(self) -> None:
        """Initialize risk monitoring module."""
        try:
            self.logger.info("Initializing risk monitoring module")
            # Initialize risk tracking structures
            self.risk_history = []
            self.logger.info("Risk monitoring module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing risk monitoring: {e}")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="system monitoring initialization",
    )
    async def _initialize_system_monitoring(self) -> None:
        """Initialize system monitoring module."""
        try:
            self.logger.info("Initializing system monitoring module")
            # Initialize system tracking structures
            self.system_history = []
            self.logger.info("System monitoring module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing system monitoring: {e}")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="alerting initialization",
    )
    async def _initialize_alerting(self) -> None:
        """Initialize alerting module."""
        try:
            self.logger.info("Initializing alerting module")
            # Initialize alerting structures
            self.logger.info("Alerting module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing alerting: {e}")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="supervision execution",
    )
    async def execute_supervision(self) -> bool:
        """Execute the main supervision cycle."""
        try:
            if not self.is_supervising:
                self.logger.warning("Supervisor is not active")
                return False

            self.logger.info("Starting supervision cycle...")

            # Validate supervision inputs
            if not self._validate_supervision_inputs():
                self.logger.error("Invalid supervision inputs")
                return False

            # Perform monitoring tasks
            if self.enable_performance_monitoring:
                await self._perform_performance_monitoring()

            if self.enable_risk_monitoring:
                await self._perform_risk_monitoring()

            if self.enable_system_monitoring:
                await self._perform_system_monitoring()

            if self.enable_alerting:
                await self._perform_alerting()

            # Store supervision results
            await self._store_supervision_results()

            self.logger.info("Supervision cycle completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error executing supervision: {e}")
            return False

    def _validate_supervision_inputs(self) -> bool:
        """Validate inputs for supervision execution."""
        try:
            if not hasattr(self, 'supervisor_config'):
                self.logger.error("Supervisor configuration not available")
                return False

            if not hasattr(self, 'alert_thresholds'):
                self.logger.error("Alert thresholds not available")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating supervision inputs: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="performance monitoring",
    )
    async def _perform_performance_monitoring(self) -> None:
        """Perform performance monitoring tasks."""
        try:
            self.logger.debug("Performing performance monitoring...")

            # Calculate performance metrics
            returns = self._calculate_returns()
            sharpe_ratio = self._calculate_sharpe_ratio()
            sortino_ratio = self._calculate_sortino_ratio()
            calmar_ratio = self._calculate_calmar_ratio()
            max_drawdown = self._calculate_max_drawdown()
            win_rate = self._calculate_win_rate()

            # Store performance metrics
            performance_score = self._calculate_performance_score(
                returns, sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown, win_rate
            )

            self.performance_history.append(performance_score)
            if len(self.performance_history) > self.max_supervision_history:
                self.performance_history.pop(0)

            self.logger.debug(f"Performance monitoring completed. Score: {performance_score:.2f}")

        except Exception as e:
            self.logger.error(f"Error performing performance monitoring: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="risk monitoring",
    )
    async def _perform_risk_monitoring(self) -> None:
        """Perform risk monitoring tasks."""
        try:
            self.logger.debug("Performing risk monitoring...")

            # Calculate risk metrics
            var_value = self._calculate_var()
            cvar_value = self._calculate_cvar()
            volatility = self._calculate_volatility()
            beta = self._calculate_beta()
            correlation = self._calculate_correlation()
            concentration = self._calculate_concentration()

            # Store risk metrics
            risk_score = self._calculate_risk_score(
                var_value, cvar_value, volatility, beta, correlation, concentration
            )

            self.risk_history.append(risk_score)
            if len(self.risk_history) > self.max_supervision_history:
                self.risk_history.pop(0)

            self.logger.debug(f"Risk monitoring completed. Score: {risk_score:.2f}")

        except Exception as e:
            self.logger.error(f"Error performing risk monitoring: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="system monitoring",
    )
    async def _perform_system_monitoring(self) -> None:
        """Perform system monitoring tasks."""
        try:
            self.logger.debug("Performing system monitoring...")

            # Monitor system resources
            cpu_usage = self._monitor_cpu_usage()
            memory_usage = self._monitor_memory_usage()
            disk_usage = self._monitor_disk_usage()
            network_latency = self._monitor_network_latency()
            error_rate = self._monitor_error_rate()
            uptime = self._monitor_uptime()

            # Store system metrics
            system_score = self._calculate_system_score(
                cpu_usage, memory_usage, disk_usage, network_latency, error_rate, uptime
            )

            self.system_history.append(system_score)
            if len(self.system_history) > self.max_supervision_history:
                self.system_history.pop(0)

            self.logger.debug(f"System monitoring completed. Score: {system_score:.2f}")

        except Exception as e:
            self.logger.error(f"Error performing system monitoring: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="alerting",
    )
    async def _perform_alerting(self) -> None:
        """Perform alerting tasks."""
        try:
            self.logger.debug("Performing alerting...")

            # Check various alert conditions
            performance_alerts = self._check_performance_alerts()
            risk_alerts = self._check_risk_alerts()
            system_alerts = self._check_system_alerts()
            threshold_alerts = self._check_threshold_alerts()

            # Process and log alerts
            all_alerts = performance_alerts + risk_alerts + system_alerts + threshold_alerts
            
            if all_alerts:
                self.logger.warning(f"Generated {len(all_alerts)} alerts")
                for alert in all_alerts:
                    self.logger.warning(f"Alert: {alert}")
            else:
                self.logger.debug("No alerts generated")

        except Exception as e:
            self.logger.error(f"Error performing alerting: {e}")

    # Performance monitoring calculation methods

    def _calculate_returns(self) -> Dict[str, float]:
        """Calculate return metrics."""
        try:
            # Placeholder implementation - replace with actual return calculation logic
            return {
                "daily_return": 0.0,
                "weekly_return": 0.0,
                "monthly_return": 0.0,
                "annual_return": 0.0
            }
        except Exception as e:
            self.logger.error(error(f"Error calculating returns: {e}"))
            return {}

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        try:
            # Placeholder implementation - replace with actual Sharpe ratio calculation
            return 0.0
        except Exception as e:
            self.logger.error(error(f"Error calculating Sharpe ratio: {e}"))
            return 0.0

    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio."""
        try:
            # Placeholder implementation - replace with actual Sortino ratio calculation
            return 0.0
        except Exception as e:
            self.logger.error(error(f"Error calculating Sortino ratio: {e}"))
            return 0.0

    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio."""
        try:
            # Placeholder implementation - replace with actual Calmar ratio calculation
            return 0.0
        except Exception as e:
            self.logger.error(error(f"Error calculating Calmar ratio: {e}"))
            return 0.0

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        try:
            # Placeholder implementation - replace with actual max drawdown calculation
            return 0.0
        except Exception as e:
            self.logger.error(error(f"Error calculating max drawdown: {e}"))
            return 0.0

    def _calculate_win_rate(self) -> float:
        """Calculate win rate."""
        try:
            # Placeholder implementation - replace with actual win rate calculation
            return 0.0
        except Exception as e:
            self.logger.error(error(f"Error calculating win rate: {e}"))
            return 0.0

    def _calculate_performance_score(self, returns: Dict[str, float], sharpe: float, 
                                   sortino: float, calmar: float, drawdown: float, 
                                   win_rate: float) -> float:
        """Calculate overall performance score."""
        try:
            # Simple scoring algorithm - replace with more sophisticated logic
            score = 0.0
            
            # Returns contribution (30%)
            if returns.get("annual_return", 0) > 0:
                score += 30.0
            
            # Risk-adjusted returns contribution (40%)
            if sharpe > 1.0:
                score += 20.0
            if sortino > 1.0:
                score += 20.0
            
            # Drawdown contribution (20%)
            if drawdown < 0.1:  # Less than 10% drawdown
                score += 20.0
            
            # Win rate contribution (10%)
            if win_rate > 0.5:  # More than 50% win rate
                score += 10.0
            
            return min(score, 100.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating performance score: {e}")
            return 0.0

    # Risk monitoring calculation methods

    def _calculate_var(self) -> float:
        """Calculate Value at Risk."""
        try:
            # Placeholder implementation - replace with actual VaR calculation
            return 0.0
        except Exception as e:
            self.logger.error(error(f"Error calculating VaR: {e}"))
            return 0.0

    def _calculate_cvar(self) -> float:
        """Calculate Conditional Value at Risk."""
        try:
            # Placeholder implementation - replace with actual CVaR calculation
            return 0.0
        except Exception as e:
            self.logger.error(error(f"Error calculating CVaR: {e}"))
            return 0.0

    def _calculate_volatility(self) -> float:
        """Calculate volatility."""
        try:
            # Placeholder implementation - replace with actual volatility calculation
            return 0.0
        except Exception as e:
            self.logger.error(error(f"Error calculating volatility: {e}"))
            return 0.0

    def _calculate_beta(self) -> float:
        """Calculate beta."""
        try:
            # Placeholder implementation - replace with actual beta calculation
            return 0.0
        except Exception as e:
            self.logger.error(error(f"Error calculating beta: {e}"))
            return 0.0

    def _calculate_correlation(self) -> float:
        """Calculate correlation."""
        try:
            # Placeholder implementation - replace with actual correlation calculation
            return 0.0
        except Exception as e:
            self.logger.error(error(f"Error calculating correlation: {e}"))
            return 0.0

    def _calculate_concentration(self) -> float:
        """Calculate concentration risk."""
        try:
            # Placeholder implementation - replace with actual concentration calculation
            return 0.0
        except Exception as e:
            self.logger.error(error(f"Error calculating concentration: {e}"))
            return 0.0

    def _calculate_risk_score(self, var: float, cvar: float, volatility: float, 
                             beta: float, correlation: float, concentration: float) -> float:
        """Calculate overall risk score."""
        try:
            # Simple risk scoring algorithm - replace with more sophisticated logic
            score = 100.0  # Start with perfect score
            
            # VaR contribution (25%)
            if var > 0.05:  # VaR > 5%
                score -= 25.0
            
            # Volatility contribution (25%)
            if volatility > 0.2:  # Volatility > 20%
                score -= 25.0
            
            # Beta contribution (20%)
            if abs(beta) > 1.5:  # High beta
                score -= 20.0
            
            # Correlation contribution (15%)
            if abs(correlation) > 0.8:  # High correlation
                score -= 15.0
            
            # Concentration contribution (15%)
            if concentration > 0.3:  # High concentration
                score -= 15.0
            
            return max(score, 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 100.0

    # System monitoring methods

    def _monitor_cpu_usage(self) -> float:
        """Monitor CPU usage."""
        try:
            return psutil.cpu_percent(interval=1)
        except Exception as e:
            self.logger.error(error(f"Error monitoring CPU usage: {e}"))
            return 0.0

    def _monitor_memory_usage(self) -> float:
        """Monitor memory usage."""
        try:
            memory = psutil.virtual_memory()
            return memory.percent
        except Exception as e:
            self.logger.error(error(f"Error monitoring memory usage: {e}"))
            return 0.0

    def _monitor_disk_usage(self) -> float:
        """Monitor disk usage."""
        try:
            disk = psutil.disk_usage('/')
            return (disk.used / disk.total) * 100
        except Exception as e:
            self.logger.error(error(f"Error monitoring disk usage: {e}"))
            return 0.0

    def _monitor_network_latency(self) -> float:
        """Monitor network latency."""
        try:
            # Placeholder implementation - replace with actual network latency monitoring
            return 0.0
        except Exception as e:
            self.logger.error(error(f"Error monitoring network latency: {e}"))
            return 0.0

    def _monitor_error_rate(self) -> float:
        """Monitor error rate."""
        try:
            # Placeholder implementation - replace with actual error rate monitoring
            return 0.0
        except Exception as e:
            self.logger.error(error(f"Error monitoring error rate: {e}"))
            return 0.0

    def _monitor_uptime(self) -> float:
        """Monitor system uptime."""
        try:
            uptime_seconds = time.time() - psutil.boot_time()
            return uptime_seconds / 3600  # Return in hours
        except Exception as e:
            self.logger.error(error(f"Error monitoring uptime: {e}"))
            return 0.0

    def _calculate_system_score(self, cpu: float, memory: float, disk: float, 
                               network: float, error_rate: float, uptime: float) -> float:
        """Calculate overall system score."""
        try:
            # Simple system scoring algorithm - replace with more sophisticated logic
            score = 100.0  # Start with perfect score
            
            # CPU contribution (25%)
            if cpu > 80:
                score -= 25.0
            elif cpu > 60:
                score -= 15.0
            
            # Memory contribution (25%)
            if memory > 85:
                score -= 25.0
            elif memory > 70:
                score -= 15.0
            
            # Disk contribution (20%)
            if disk > 90:
                score -= 20.0
            elif disk > 80:
                score -= 10.0
            
            # Error rate contribution (20%)
            if error_rate > 10:
                score -= 20.0
            elif error_rate > 5:
                score -= 10.0
            
            # Uptime contribution (10%)
            if uptime < 24:  # Less than 24 hours
                score -= 10.0
            
            return max(score, 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating system score: {e}")
            return 100.0

    # Alerting methods

    def _check_performance_alerts(self) -> List[str]:
        """Check performance alerts."""
        try:
            alerts = []
            
            if not self.performance_history:
                return alerts
            
            latest_score = self.performance_history[-1]
            
            if latest_score < 50:
                alerts.append(f"Critical: Performance score is critically low: {latest_score:.2f}")
            elif latest_score < 70:
                alerts.append(f"Warning: Performance score is below threshold: {latest_score:.2f}")
            
            return alerts
            
        except Exception as e:
            self.logger.error(error(f"Error checking performance alerts: {e}"))
            return []

    def _check_risk_alerts(self) -> List[str]:
        """Check risk alerts."""
        try:
            alerts = []
            
            if not self.risk_history:
                return alerts
            
            latest_score = self.risk_history[-1]
            
            if latest_score > 70:
                alerts.append(f"Critical: Risk score is critically high: {latest_score:.2f}")
            elif latest_score > 50:
                alerts.append(f"Warning: Risk score is above threshold: {latest_score:.2f}")
            
            return alerts
            
        except Exception as e:
            self.logger.error(error(f"Error checking risk alerts: {e}"))
            return []

    def _check_system_alerts(self) -> List[str]:
        """Check system alerts."""
        try:
            alerts = []
            
            if not self.system_history:
                return alerts
            
            latest_score = self.system_history[-1]
            
            if latest_score < 50:
                alerts.append(f"Critical: System score is critically low: {latest_score:.2f}")
            elif latest_score < 70:
                alerts.append(f"Warning: System score is below threshold: {latest_score:.2f}")
            
            return alerts
            
        except Exception as e:
            self.logger.error(error(f"Error checking system alerts: {e}"))
            return []

    def _check_threshold_alerts(self) -> List[str]:
        """Check threshold-based alerts."""
        try:
            alerts = []
            
            for threshold in self.alert_thresholds:
                if not threshold.enabled:
                    continue
                
                # Get current metric value (placeholder implementation)
                current_value = 0.0
                
                if current_value >= threshold.critical_threshold:
                    alerts.append(f"Critical: {threshold.metric_name} exceeds critical threshold: {current_value:.2f} >= {threshold.critical_threshold:.2f}")
                elif current_value >= threshold.warning_threshold:
                    alerts.append(f"Warning: {threshold.metric_name} exceeds warning threshold: {current_value:.2f} >= {threshold.warning_threshold:.2f}")
            
            return alerts
            
        except Exception as e:
            self.logger.error(error(f"Error checking threshold alerts: {e}"))
            return []

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="supervision results storage",
    )
    async def _store_supervision_results(self) -> None:
        """Store supervision results and history."""
        try:
            timestamp = datetime.now()
            
            # Create supervision result
            result = {
                "timestamp": timestamp.isoformat(),
                "performance_score": self.performance_history[-1] if self.performance_history else 0.0,
                "risk_score": self.risk_history[-1] if self.risk_history else 0.0,
                "system_score": self.system_history[-1] if self.system_history else 0.0,
                "is_supervising": self.is_supervising,
                "supervision_interval": self.supervision_interval,
            }
            
            # Store current result
            self.supervision_results = result
            
            # Add to history
            self.supervision_history.append(result)
            
            # Maintain history size
            if len(self.supervision_history) > self.max_supervision_history:
                self.supervision_history.pop(0)
            
            self.logger.debug("Supervision results stored successfully")
            
        except Exception as e:
            self.logger.error(f"Error storing supervision results: {e}")

    def get_supervision_results(self) -> Optional[Dict[str, Any]]:
        """Get current supervision results."""
        try:
            return self.supervision_results.copy() if self.supervision_results else None
        except Exception as e:
            self.logger.error(f"Error getting supervision results: {e}")
            return None

    def get_supervision_history(self) -> List[Dict[str, Any]]:
        """Get supervision history."""
        try:
            return self.supervision_history.copy()
        except Exception as e:
            self.logger.error(error(f"Error getting supervision history: {e}"))
            return []

    def get_supervisor_status(self) -> Dict[str, Any]:
        """Get supervisor status."""
        try:
            return {
                "is_supervising": self.is_supervising,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "supervision_interval": self.supervision_interval,
                "max_supervision_history": self.max_supervision_history,
                "enable_performance_monitoring": self.enable_performance_monitoring,
                "enable_risk_monitoring": self.enable_risk_monitoring,
                "enable_system_monitoring": self.enable_system_monitoring,
                "enable_alerting": self.enable_alerting,
                "performance_history_size": len(self.performance_history),
                "risk_history_size": len(self.risk_history),
                "system_history_size": len(self.system_history),
                "supervision_history_size": len(self.supervision_history),
            }
        except Exception as e:
            self.logger.error(f"Error getting supervisor status: {e}")
            return {}

    async def start(self) -> bool:
        """Start the supervisor."""
        try:
            if self.is_supervising:
                self.logger.warning("Supervisor is already running")
                return True
            
            self.logger.info("🚀 Starting Modular Supervisor...")
            self.is_supervising = True
            self.start_time = datetime.now()
            
            # Start supervision loop
            asyncio.create_task(self._supervision_loop())
            
            self.logger.info("✅ Modular Supervisor started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting supervisor: {e}")
            self.is_supervising = False
            return False

    async def _supervision_loop(self) -> None:
        """Main supervision loop."""
        try:
            while self.is_supervising:
                await self.execute_supervision()
                await asyncio.sleep(self.supervision_interval)
                
        except Exception as e:
            self.logger.error(f"Error in supervision loop: {e}")
            self.is_supervising = False

    async def stop(self) -> None:
        """Stop the supervisor."""
        try:
            self.logger.info("🛑 Stopping Modular Supervisor...")
            self.is_supervising = False
            
            # Wait for supervision loop to finish
            await asyncio.sleep(1)
            
            self.logger.info("✅ Modular Supervisor stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping supervisor: {e}")

    async def setup_modular_supervisor(self) -> None:
        """Setup function for modular supervisor."""
        try:
            # Initialize the supervisor
            if not await self.initialize():
                raise RuntimeError("Failed to initialize modular supervisor")
            
            # Start the supervisor
            if not await self.start():
                raise RuntimeError("Failed to start modular supervisor")
            
            self.logger.info("Modular supervisor setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in modular supervisor setup: {e}")
            raise


# Factory function for creating modular supervisor instances
async def create_modular_supervisor(config: Dict[str, Any]) -> ModularSupervisor:
    """Create and setup a modular supervisor instance."""
    try:
        supervisor = ModularSupervisor(config)
        await supervisor.setup_modular_supervisor()
        return supervisor
    except Exception as e:
        system_logger.error(f"Failed to create modular supervisor: {e}")
        raise
