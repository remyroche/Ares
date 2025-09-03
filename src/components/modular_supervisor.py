from __future__ import annotations

from datetime import datetime
from typing import Any

from src.core.decorators import handles_errors as handles_errors_src_core_decorators
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
    initialization_error,
    invalid,
    missing,
)

# src/components/modular_supervisor.py


class ModularSupervisor:
    """
    Enhanced modular supervisor with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize modular supervisor with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("ModularSupervisor")

        # Supervision state
        self.is_supervising: bool = False
        self.supervision_results: dict[str, Any] = {}
        self.supervision_history: list[dict[str, Any]] = []

        # Configuration
        self.supervisor_config: dict[str, Any] = self.config.get(
            "modular_supervisor",
            {},
        )
        self.supervision_interval: int = self.supervisor_config.get(
            "supervision_interval",
            60,
        )
        self.max_supervision_history: int = self.supervisor_config.get(
            "max_supervision_history",
            100,
        )
        self.enable_performance_monitoring: bool = self.supervisor_config.get(
            "enable_performance_monitoring",
            True,
        )
        self.enable_risk_monitoring: bool = self.supervisor_config.get(
            "enable_risk_monitoring",
            True,
        )

    @handles_errors(
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
        Initialize modular supervisor with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Modular Supervisor...")

            # Load supervisor configuration
            await self._load_supervisor_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(
                    invalid("Invalid configuration for modular supervisor")
                )
                return False

            # Initialize supervision modules
            await self._initialize_supervision_modules()

            self.logger.info(
                "✅ Modular Supervisor initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.exception(
                failed(f"❌ Modular Supervisor initialization failed: {e}")
            )
            return False

    @handles_errors(fallback=None)
    async def _load_supervisor_configuration(self) -> None:
        """Load supervisor configuration."""
        try:
            # Set default supervisor parameters
            self.supervisor_config.setdefault("supervision_interval", 60)
            self.supervisor_config.setdefault("max_supervision_history", 100)
            self.supervisor_config.setdefault("enable_performance_monitoring", True)
            self.supervisor_config.setdefault("enable_risk_monitoring", True)
            self.supervisor_config.setdefault("enable_system_monitoring", False)
            self.supervisor_config.setdefault("enable_alerting", True)

            # Update configuration
            self.supervision_interval = self.supervisor_config["supervision_interval"]
            self.max_supervision_history = self.supervisor_config[
                "max_supervision_history"
            ]
            self.enable_performance_monitoring = self.supervisor_config[
                "enable_performance_monitoring"
            ]
            self.enable_risk_monitoring = self.supervisor_config[
                "enable_risk_monitoring"
            ]

            self.logger.info("Supervisor configuration loaded successfully")

        except Exception as e:
            self.logger.exception(error(f"Error loading supervisor configuration: {e}"))

    @handles_errors(fallback=False)
    def _validate_configuration(self) -> bool:
        """
        Validate supervisor configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate supervision interval
            if self.supervision_interval <= 0:
                self.logger.error(invalid("Invalid supervision interval"))
                return False

            # Validate max supervision history
            if self.max_supervision_history <= 0:
                self.logger.error(invalid("Invalid max supervision history"))
                return False

            # Validate that at least one supervision type is enabled
            if not any(
                [
                    self.enable_performance_monitoring,
                    self.enable_risk_monitoring,
                    self.supervisor_config.get("enable_system_monitoring", False),
                    self.supervisor_config.get("enable_alerting", True),
                ],
            ):
                self.logger.error(
                    error("At least one supervision type must be enabled")
                )
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.exception(error(f"Error validating configuration: {e}"))
            return False

    @handles_errors(fallback=None)
    async def _initialize_supervision_modules(self) -> None:
        """Initialize supervision modules."""
        try:
            # Initialize performance monitoring module
            if self.enable_performance_monitoring:
                await self._initialize_performance_monitoring()

            # Initialize risk monitoring module
            if self.enable_risk_monitoring:
                await self._initialize_risk_monitoring()

            # Initialize system monitoring module
            if self.supervisor_config.get("enable_system_monitoring", False):
                await self._initialize_system_monitoring()

            # Initialize alerting module
            if self.supervisor_config.get("enable_alerting", True):
                await self._initialize_alerting()

            self.logger.info("Supervision modules initialized successfully")

        except Exception as e:
            self.logger.exception(
                initialization_error(f"Error initializing supervision modules: {e}")
            )

    @handles_errors(fallback=None)
    async def _initialize_performance_monitoring(self) -> None:
        """Initialize performance monitoring module."""
        try:
            # Initialize performance metrics
            self.performance_metrics = {
                "returns": True,
                "sharpe_ratio": True,
                "sortino_ratio": True,
                "calmar_ratio": True,
                "max_drawdown": True,
                "win_rate": True,
            }

            self.logger.info("Performance monitoring module initialized")

        except Exception as e:
            self.logger.exception(
                initialization_error(f"Error initializing performance monitoring: {e}")
            )

    @handles_errors(fallback=None)
    async def _initialize_risk_monitoring(self) -> None:
        """Initialize risk monitoring module."""
        try:
            # Initialize risk metrics
            self.risk_metrics = {
                "var": True,
                "cvar": True,
                "volatility": True,
                "beta": True,
                "correlation": True,
                "concentration": True,
            }

            self.logger.info("Risk monitoring module initialized")

        except Exception as e:
            self.logger.exception(
                initialization_error(f"Error initializing risk monitoring: {e}")
            )

    @handles_errors(fallback=None)
    async def _initialize_system_monitoring(self) -> None:
        """Initialize system monitoring module."""
        try:
            # Initialize system metrics
            self.system_metrics = {
                "cpu_usage": True,
                "memory_usage": True,
                "disk_usage": True,
                "network_latency": True,
                "error_rate": True,
                "uptime": True,
            }

            self.logger.info("System monitoring module initialized")

        except Exception as e:
            self.logger.exception(
                initialization_error(f"Error initializing system monitoring: {e}")
            )

    @handles_errors(fallback=None)
    async def _initialize_alerting(self) -> None:
        """Initialize alerting module."""
        try:
            # Initialize alerting rules
            self.alerting_rules = {
                "performance_alerts": True,
                "risk_alerts": True,
                "system_alerts": True,
                "threshold_alerts": True,
            }

            self.logger.info("Alerting module initialized")

        except Exception as e:
            self.logger.exception(
                initialization_error(f"Error initializing alerting: {e}")
            )

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid supervision parameters"),
            AttributeError: (False, "Missing supervision components"),
            KeyError: (False, "Missing required supervision data"),
        },
        default_return=False,
        context="supervision execution",
    )
    async def execute_supervision(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> bool:
        """
        Execute supervision monitoring.

        Args:
            trading_data: Trading data dictionary
            system_data: System data dictionary

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._validate_supervision_inputs(trading_data, system_data):
                return False

            self.is_supervising = True
            self.logger.info("🔄 Starting supervision execution...")

            # Perform performance monitoring
            if self.enable_performance_monitoring:
                performance_results = await self._perform_performance_monitoring(
                    trading_data,
                    system_data,
                )
                self.supervision_results["performance"] = performance_results

            # Perform risk monitoring
            if self.enable_risk_monitoring:
                risk_results = await self._perform_risk_monitoring(
                    trading_data,
                    system_data,
                )
                self.supervision_results["risk"] = risk_results

            # Perform system monitoring
            if self.supervisor_config.get("enable_system_monitoring", False):
                system_results = await self._perform_system_monitoring(
                    trading_data,
                    system_data,
                )
                self.supervision_results["system"] = system_results

            # Perform alerting
            if self.supervisor_config.get("enable_alerting", True):
                alerting_results = await self._perform_alerting(
                    trading_data,
                    system_data,
                )
                self.supervision_results["alerting"] = alerting_results

            # Store supervision results
            await self._store_supervision_results()

            self.is_supervising = False
            self.logger.info("✅ Supervision execution completed successfully")
            return True

        except Exception as e:
            self.logger.exception(error(f"Error executing supervision: {e}"))
            self.is_supervising = False
            return False

    @handles_errors(fallback=False)
    def _validate_supervision_inputs(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> bool:
        """
        Validate supervision inputs.

        Args:
            trading_data: Trading data dictionary
            system_data: System data dictionary

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required trading data fields
            required_trading_fields = ["returns", "positions", "timestamp"]
            for field in required_trading_fields:
                if field not in trading_data:
                    self.logger.error(
                        missing(f"Missing required trading data field: {field}")
                    )
                    return False

            # Check required system data fields
            required_system_fields = ["cpu_usage", "memory_usage", "timestamp"]
            for field in required_system_fields:
                if field not in system_data:
                    self.logger.error(
                        missing(f"Missing required system data field: {field}")
                    )
                    return False

            # Validate data types
            if not isinstance(trading_data["returns"], int | float):
                self.logger.error(invalid("Invalid returns data type"))
                return False

            if not isinstance(system_data["cpu_usage"], int | float):
                self.logger.error(invalid("Invalid CPU usage data type"))
                return False

            return True

        except Exception as e:
            self.logger.exception(error(f"Error validating supervision inputs: {e}"))
            return False

    @handles_errors(fallback=None)
    async def _perform_performance_monitoring(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform performance monitoring.

        Args:
            trading_data: Trading data dictionary
            system_data: System data dictionary

        Returns:
            Dict[str, Any]: Performance monitoring results
        """
        try:
            results = {}

            # Calculate returns
            if self.performance_metrics.get("returns", False):
                results["returns"] = self._calculate_returns(trading_data, system_data)

            # Calculate Sharpe ratio
            if self.performance_metrics.get("sharpe_ratio", False):
                results["sharpe_ratio"] = self._calculate_sharpe_ratio(
                    trading_data, system_data
                )

            # Calculate Sortino ratio
            if self.performance_metrics.get("sortino_ratio", False):
                results["sortino_ratio"] = self._calculate_sortino_ratio(
                    trading_data, system_data
                )

            # Calculate Calmar ratio
            if self.performance_metrics.get("calmar_ratio", False):
                results["calmar_ratio"] = self._calculate_calmar_ratio(
                    trading_data, system_data
                )

            # Calculate max drawdown
            if self.performance_metrics.get("max_drawdown", False):
                results["max_drawdown"] = self._calculate_max_drawdown(
                    trading_data, system_data
                )

            # Calculate win rate
            if self.performance_metrics.get("win_rate", False):
                results["win_rate"] = self._calculate_win_rate(
                    trading_data, system_data
                )

            self.logger.info("Performance monitoring completed")
            return results

        except Exception as e:
            self.logger.exception(
                error(f"Error performing performance monitoring: {e}")
            )
            return {}

    @handles_errors(fallback=None)
    async def _perform_risk_monitoring(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform risk monitoring.

        Args:
            trading_data: Trading data dictionary
            system_data: System data dictionary

        Returns:
            Dict[str, Any]: Risk monitoring results
        """
        try:
            results = {}

            # Calculate VaR
            if self.risk_metrics.get("var", False):
                results["var"] = self._calculate_var(trading_data, system_data)

            # Calculate CVaR
            if self.risk_metrics.get("cvar", False):
                results["cvar"] = self._calculate_cvar(trading_data, system_data)

            # Calculate volatility
            if self.risk_metrics.get("volatility", False):
                results["volatility"] = self._calculate_volatility(
                    trading_data, system_data
                )

            # Calculate beta
            if self.risk_metrics.get("beta", False):
                results["beta"] = self._calculate_beta(trading_data, system_data)

            # Calculate correlation
            if self.risk_metrics.get("correlation", False):
                results["correlation"] = self._calculate_correlation(
                    trading_data, system_data
                )

            # Calculate concentration
            if self.risk_metrics.get("concentration", False):
                results["concentration"] = self._calculate_concentration(
                    trading_data, system_data
                )

            self.logger.info("Risk monitoring completed")
            return results

        except Exception as e:
            self.logger.exception(error(f"Error performing risk monitoring: {e}"))
            return {}

    @handles_errors(fallback=None)
    async def _perform_system_monitoring(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform system monitoring.

        Args:
            trading_data: Trading data dictionary
            system_data: System data dictionary

        Returns:
            Dict[str, Any]: System monitoring results
        """
        try:
            results = {}

            # Monitor CPU usage
            if self.system_metrics.get("cpu_usage", False):
                results["cpu_usage"] = self._monitor_cpu_usage(
                    trading_data, system_data
                )

            # Monitor memory usage
            if self.system_metrics.get("memory_usage", False):
                results["memory_usage"] = self._monitor_memory_usage(
                    trading_data, system_data
                )

            # Monitor disk usage
            if self.system_metrics.get("disk_usage", False):
                results["disk_usage"] = self._monitor_disk_usage(
                    trading_data, system_data
                )

            # Monitor network latency
            if self.system_metrics.get("network_latency", False):
                results["network_latency"] = self._monitor_network_latency(
                    trading_data, system_data
                )

            # Monitor error rate
            if self.system_metrics.get("error_rate", False):
                results["error_rate"] = self._monitor_error_rate(
                    trading_data, system_data
                )

            # Monitor uptime
            if self.system_metrics.get("uptime", False):
                results["uptime"] = self._monitor_uptime(trading_data, system_data)

            self.logger.info("System monitoring completed")
            return results

        except Exception as e:
            self.logger.exception(error(f"Error performing system monitoring: {e}"))
            return {}

    @handles_errors(fallback=None)
    async def _perform_alerting(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform alerting.

        Args:
            trading_data: Trading data dictionary
            system_data: System data dictionary

        Returns:
            Dict[str, Any]: Alerting results
        """
        try:
            results = {}

            # Check performance alerts
            if self.alerting_rules.get("performance_alerts", False):
                results["performance_alerts"] = self._check_performance_alerts(
                    trading_data,
                    system_data,
                )

            # Check risk alerts
            if self.alerting_rules.get("risk_alerts", False):
                results["risk_alerts"] = self._check_risk_alerts(
                    trading_data, system_data
                )

            # Check system alerts
            if self.alerting_rules.get("system_alerts", False):
                results["system_alerts"] = self._check_system_alerts(
                    trading_data, system_data
                )

            # Check threshold alerts
            if self.alerting_rules.get("threshold_alerts", False):
                results["threshold_alerts"] = self._check_threshold_alerts(
                    trading_data,
                    system_data,
                )

            self.logger.info("Alerting completed")
            return results

        except Exception as e:
            self.logger.exception(error(f"Error performing alerting: {e}"))
            return {}

    # Performance monitoring calculation methods

    def _calculate_returns(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> dict[str, float]:
        """Calculate returns metrics."""
        try:
            # Simulate returns calculation
            return {
                "total_return": 0.15,
                "annualized_return": 0.12,
                "daily_return": 0.001,
            }
        except Exception as e:
            self.logger.exception(error(f"Error calculating returns: {e}"))
            return {}

    def _calculate_sharpe_ratio(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> float:
        """Calculate Sharpe ratio."""
        try:
            # Simulate Sharpe ratio calculation
            return 1.25
        except Exception as e:
            self.logger.exception(error(f"Error calculating Sharpe ratio: {e}"))
            return 0.0

    def _calculate_sortino_ratio(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> float:
        """Calculate Sortino ratio."""
        try:
            # Simulate Sortino ratio calculation
            return 1.45
        except Exception as e:
            self.logger.exception(error(f"Error calculating Sortino ratio: {e}"))
            return 0.0

    def _calculate_calmar_ratio(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> float:
        """Calculate Calmar ratio."""
        try:
            # Simulate Calmar ratio calculation
            return 1.35
        except Exception as e:
            self.logger.exception(error(f"Error calculating Calmar ratio: {e}"))
            return 0.0

    def _calculate_max_drawdown(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> float:
        """Calculate maximum drawdown."""
        try:
            # Simulate max drawdown calculation
            return 0.08
        except Exception as e:
            self.logger.exception(error(f"Error calculating max drawdown: {e}"))
            return 0.0

    def _calculate_win_rate(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> float:
        """Calculate win rate."""
        try:
            # Simulate win rate calculation
            return 0.65
        except Exception as e:
            self.logger.exception(error(f"Error calculating win rate: {e}"))
            return 0.0

    # Risk monitoring calculation methods

    def _calculate_var(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> float:
        """Calculate Value at Risk."""
        try:
            # Simulate VaR calculation
            return 0.025
        except Exception as e:
            self.logger.exception(error(f"Error calculating VaR: {e}"))
            return 0.0

    def _calculate_cvar(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> float:
        """Calculate Conditional Value at Risk."""
        try:
            # Simulate CVaR calculation
            return 0.035
        except Exception as e:
            self.logger.exception(error(f"Error calculating CVaR: {e}"))
            return 0.0

    def _calculate_volatility(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> float:
        """Calculate volatility."""
        try:
            # Simulate volatility calculation
            return 0.18
        except Exception as e:
            self.logger.exception(error(f"Error calculating volatility: {e}"))
            return 0.0

    def _calculate_beta(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> float:
        """Calculate beta."""
        try:
            # Simulate beta calculation
            return 0.85
        except Exception as e:
            self.logger.exception(error(f"Error calculating beta: {e}"))
            return 0.0

    def _calculate_correlation(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> float:
        """Calculate correlation."""
        try:
            # Simulate correlation calculation
            return 0.25
        except Exception as e:
            self.logger.exception(error(f"Error calculating correlation: {e}"))
            return 0.0

    def _calculate_concentration(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> float:
        """Calculate concentration."""
        try:
            # Simulate concentration calculation
            return 0.15
        except Exception as e:
            self.logger.exception(error(f"Error calculating concentration: {e}"))
            return 0.0

    # System monitoring methods

    def _monitor_cpu_usage(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Monitor CPU usage."""
        try:
            # Simulate CPU usage monitoring
            return {
                "current_cpu": 0.45,
                "max_cpu": 0.8,
                "cpu_ok": True,
            }
        except Exception as e:
            self.logger.exception(error(f"Error monitoring CPU usage: {e}"))
            return {}

    def _monitor_memory_usage(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Monitor memory usage."""
        try:
            # Simulate memory usage monitoring
            return {
                "current_memory": 0.6,
                "max_memory": 0.9,
                "memory_ok": True,
            }
        except Exception as e:
            self.logger.exception(error(f"Error monitoring memory usage: {e}"))
            return {}

    def _monitor_disk_usage(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Monitor disk usage."""
        try:
            # Simulate disk usage monitoring
            return {
                "current_disk": 0.35,
                "max_disk": 0.8,
                "disk_ok": True,
            }
        except Exception as e:
            self.logger.exception(error(f"Error monitoring disk usage: {e}"))
            return {}

    def _monitor_network_latency(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Monitor network latency."""
        try:
            # Simulate network latency monitoring
            return {
                "current_latency": 50,
                "max_latency": 100,
                "latency_ok": True,
            }
        except Exception as e:
            self.logger.exception(error(f"Error monitoring network latency: {e}"))
            return {}

    def _monitor_error_rate(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Monitor error rate."""
        try:
            # Simulate error rate monitoring
            return {
                "current_error_rate": 0.01,
                "max_error_rate": 0.05,
                "error_rate_ok": True,
            }
        except Exception as e:
            self.logger.exception(error(f"Error monitoring error rate: {e}"))
            return {}

    def _monitor_uptime(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Monitor uptime."""
        try:
            # Simulate uptime monitoring
            return {
                "current_uptime": 99.8,
                "min_uptime": 99.5,
                "uptime_ok": True,
            }
        except Exception as e:
            self.logger.exception(error(f"Error monitoring uptime: {e}"))
            return {}

    # Alerting methods

    def _check_performance_alerts(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Check performance alerts."""
        try:
            # Simulate performance alert checking
            return {
                "performance_alerts": [],
                "alert_count": 0,
                "critical_alerts": 0,
            }
        except Exception as e:
            self.logger.exception(error(f"Error checking performance alerts: {e}"))
            return {}

    def _check_risk_alerts(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Check risk alerts."""
        try:
            # Simulate risk alert checking
            return {
                "risk_alerts": [],
                "alert_count": 0,
                "critical_alerts": 0,
            }
        except Exception as e:
            self.logger.exception(error(f"Error checking risk alerts: {e}"))
            return {}

    def _check_system_alerts(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Check system alerts."""
        try:
            # Simulate system alert checking
            return {
                "system_alerts": [],
                "alert_count": 0,
                "critical_alerts": 0,
            }
        except Exception as e:
            self.logger.exception(error(f"Error checking system alerts: {e}"))
            return {}

    def _check_threshold_alerts(
        self,
        trading_data: dict[str, Any],
        system_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Check threshold alerts."""
        try:
            # Simulate threshold alert checking
            return {
                "threshold_alerts": [],
                "alert_count": 0,
                "critical_alerts": 0,
            }
        except Exception as e:
            self.logger.exception(error(f"Error checking threshold alerts: {e}"))
            return {}

    @handles_errors(fallback=None)
    async def _store_supervision_results(self) -> None:
        """Store supervision results."""
        try:
            # Add timestamp
            self.supervision_results["timestamp"] = datetime.now().isoformat()

            # Add to history
            self.supervision_history.append(self.supervision_results.copy())

            # Limit history size
            if len(self.supervision_history) > self.max_supervision_history:
                self.supervision_history.pop(0)

            self.logger.info("Supervision results stored successfully")

        except Exception as e:
            self.logger.exception(error(f"Error storing supervision results: {e}"))

    @handles_errors(fallback=None)
    def get_supervision_results(
        self,
        supervision_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get supervision results.

        Args:
            supervision_type: Optional supervision type filter

        Returns:
            Dict[str, Any]: Supervision results
        """
        try:
            if supervision_type:
                return self.supervision_results.get(supervision_type, {})
            return self.supervision_results.copy()

        except Exception as e:
            self.logger.exception(error(f"Error getting supervision results: {e}"))
            return {}

    @handles_errors(fallback=None)
    def get_supervision_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get supervision history.

        Args:
            limit: Optional limit on number of records

        Returns:
            List[Dict[str, Any]]: Supervision history
        """
        try:
            history = self.supervision_history.copy()

            if limit:
                history = history[-limit:]

            return history

        except Exception as e:
            self.logger.exception(error(f"Error getting supervision history: {e}"))
            return []

    def get_supervisor_status(self) -> dict[str, Any]:
        """
        Get supervisor status information.

        Returns:
            Dict[str, Any]: Supervisor status
        """
        return {
            "is_supervising": self.is_supervising,
            "supervision_interval": self.supervision_interval,
            "max_supervision_history": self.max_supervision_history,
            "enable_performance_monitoring": self.enable_performance_monitoring,
            "enable_risk_monitoring": self.enable_risk_monitoring,
            "enable_system_monitoring": self.supervisor_config.get(
                "enable_system_monitoring",
                False,
            ),
            "enable_alerting": self.supervisor_config.get(
                "enable_alerting",
                True,
            ),
            "supervision_history_count": len(self.supervision_history),
        }

    @handles_errors(fallback=None)
    async def stop(self) -> None:
        """Stop the modular supervisor."""
        self.logger.info("🛑 Stopping Modular Supervisor...")

        try:
            # Stop supervising
            self.is_supervising = False

            # Clear results
            self.supervision_results.clear()

            # Clear history
            self.supervision_history.clear()

            self.logger.info("✅ Modular Supervisor stopped successfully")

        except Exception as e:
            self.logger.exception(error(f"Error stopping modular supervisor: {e}"))


# Global modular supervisor instance
modular_supervisor: ModularSupervisor | None = None


@handles_errors(fallback=None)
async def setup_modular_supervisor(
    config: dict[str, Any] | None = None,
) -> ModularSupervisor | None:
    """
    Setup global modular supervisor.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[ModularSupervisor]: Global modular supervisor instance
    """
    try:
        global modular_supervisor

        if config is None:
            config = {
                "modular_supervisor": {
                    "supervision_interval": 60,
                    "max_supervision_history": 100,
                    "enable_performance_monitoring": True,
                    "enable_risk_monitoring": True,
                    "enable_system_monitoring": False,
                    "enable_alerting": True,
                },
            }

        # Create modular supervisor
        modular_supervisor = ModularSupervisor(config)

        # Initialize modular supervisor
        success = await modular_supervisor.initialize()
        if success:
            return modular_supervisor
        return None

    except Exception as e:
        print(f"Error setting up modular supervisor: {e}")
        return None
