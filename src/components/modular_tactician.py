# src/components/modular_tactician.py

import asyncio
import copy
from datetime import datetime
from typing import Any

from src.core.decorators import handles_errors
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import error, initialization_error, invalid, missing


class ModularTactician:
    """Enhanced modular tactician with comprehensive error handling and type safety."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize modular tactician with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("ModularTactician")

        # Tactician state
        self.is_tactician_active: bool = False
        self.tactician_results: dict[str, Any] = {}
        self.tactician_history: list[dict[str, Any]] = []

        # Configuration
        self.tactician_config: dict[str, Any] = self.config.get("modular_tactician", {})
        self.tactician_interval: int = self.tactician_config.get(
            "tactician_interval",
            5,
        )
        self.max_tactician_history: int = self.tactician_config.get(
            "max_tactician_history",
            100,
        )
        self.enable_entry_monitoring: bool = self.tactician_config.get(
            "enable_entry_monitoring",
            True,
        )
        self.enable_exit_monitoring: bool = self.tactician_config.get(
            "enable_exit_monitoring",
            True,
        )

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid modular tactician configuration"),
            AttributeError: (False, "Missing required tactician parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="modular tactician initialization",
    )
    async def initialize(self) -> bool:
        """Initialize modular tactician with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        self.logger.info("Initializing Modular Tactician...")

        # Load tactician configuration
        await self._load_tactician_configuration()

        # Validate configuration
        if not self._validate_configuration():
            self.logger.error(invalid("Invalid configuration for modular tactician"))
            return False

        # Initialize tactician modules
        await self._initialize_tactician_modules()

        self.logger.info(
            "✅ Modular Tactician initialization completed successfully",
        )
        return True

    @handles_errors(fallback=None)
    async def _load_tactician_configuration(self) -> None:
        """Load tactician configuration."""
        try:
            # Set default tactician parameters
            self.tactician_config.setdefault("tactician_interval", 5)
            self.tactician_config.setdefault("max_tactician_history", 100)
            self.tactician_config.setdefault("enable_entry_monitoring", True)
            self.tactician_config.setdefault("enable_exit_monitoring", True)
            self.tactician_config.setdefault("enable_position_monitoring", False)
            self.tactician_config.setdefault("enable_risk_monitoring", True)

            # Update configuration
            self.tactician_interval = self.tactician_config["tactician_interval"]
            self.max_tactician_history = self.tactician_config["max_tactician_history"]
            self.enable_entry_monitoring = self.tactician_config[
                "enable_entry_monitoring"
            ]
            self.enable_exit_monitoring = self.tactician_config[
                "enable_exit_monitoring"
            ]

            self.logger.info("Tactician configuration loaded successfully")

        except Exception as e:
            self.logger.error(error(f"Error loading tactician configuration: {e}"))

    @handles_errors(fallback=False)
    def _validate_configuration(self) -> bool:
        """Validate tactician configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate tactician interval
            if self.tactician_interval <= 0:
                self.logger.error(invalid("Invalid tactician interval"))
                return False

            # Validate max tactician history
            if self.max_tactician_history <= 0:
                self.logger.error(invalid("Invalid max tactician history"))
                return False

            # Validate that at least one tactician type is enabled
            if not any(
                [
                    self.enable_entry_monitoring,
                    self.enable_exit_monitoring,
                    self.tactician_config.get("enable_position_monitoring", False),
                    self.tactician_config.get("enable_risk_monitoring", True),
                ],
            ):
                self.logger.error(error("At least one tactician type must be enabled"))
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(error(f"Error validating configuration: {e}"))
            return False

    @handles_errors(fallback=None)
    async def _initialize_tactician_modules(self) -> None:
        """Initialize tactician modules."""
        try:
            # Initialize entry monitoring module
            if self.enable_entry_monitoring:
                await self._initialize_entry_monitoring()

            # Initialize exit monitoring module
            if self.enable_exit_monitoring:
                await self._initialize_exit_monitoring()

            # Initialize position monitoring module
            if self.tactician_config.get("enable_position_monitoring", False):
                await self._initialize_position_monitoring()

            # Initialize risk monitoring module
            if self.tactician_config.get("enable_risk_monitoring", True):
                await self._initialize_risk_monitoring()

            self.logger.info("Tactician modules initialized successfully")

        except Exception as e:
            self.logger.error(
                initialization_error(f"Error initializing tactician modules: {e}")
            )

    @handles_errors(fallback=None)
    async def _initialize_entry_monitoring(self) -> None:
        """Initialize entry monitoring module."""
        try:
            # Initialize entry monitoring strategies
            self.entry_monitoring_strategies = {
                "price_action": True,
                "volume_analysis": True,
                "momentum_indicators": True,
                "support_resistance": True,
            }

            self.logger.info("Entry monitoring module initialized")

        except Exception as e:
            self.logger.error(
                initialization_error(f"Error initializing entry monitoring: {e}")
            )

    @handles_errors(fallback=None)
    async def _initialize_exit_monitoring(self) -> None:
        """Initialize exit monitoring module."""
        try:
            # Initialize exit monitoring strategies
            self.exit_monitoring_strategies = {
                "stop_loss_tracking": True,
                "take_profit_tracking": True,
                "trailing_stop": True,
                "time_based_exit": True,
            }

            self.logger.info("Exit monitoring module initialized")

        except Exception as e:
            self.logger.error(
                initialization_error(f"Error initializing exit monitoring: {e}")
            )

    @handles_errors(fallback=None)
    async def _initialize_position_monitoring(self) -> None:
        """Initialize position monitoring module."""
        try:
            # Initialize position monitoring strategies
            self.position_monitoring_strategies = {
                "position_size_tracking": True,
                "exposure_limits": True,
                "correlation_monitoring": True,
                "concentration_limits": True,
            }

            self.logger.info("Position monitoring module initialized")

        except Exception as e:
            self.logger.error(
                initialization_error(f"Error initializing position monitoring: {e}")
            )

    @handles_errors(fallback=None)
    async def _initialize_risk_monitoring(self) -> None:
        """Initialize risk monitoring module."""
        try:
            # Initialize risk monitoring strategies
            self.risk_monitoring_strategies = {
                "var_monitoring": True,
                "drawdown_tracking": True,
                "volatility_monitoring": True,
                "stress_testing": True,
            }

            self.logger.info("Risk monitoring module initialized")

        except Exception as e:
            self.logger.error(
                initialization_error(f"Error initializing risk monitoring: {e}")
            )

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid tactician parameters"),
            AttributeError: (False, "Missing tactician components"),
            KeyError: (False, "Missing required tactician data"),
        },
        default_return=False,
        context="tactician execution",
    )
    async def execute_tactician(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> bool:
        """Execute tactician monitoring.

        Args:
            market_data: Market data dictionary
            strategy_data: Strategy data dictionary

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._validate_tactician_inputs(market_data, strategy_data):
                return False

            self.is_tactician_active = True
            self.logger.info("🔄 Starting tactician execution...")

            # Perform entry monitoring
            if self.enable_entry_monitoring:
                entry_results = await self._perform_entry_monitoring(
                    market_data,
                    strategy_data,
                )
                self.tactician_results["entry_monitoring"] = entry_results

            # Perform exit monitoring
            if self.enable_exit_monitoring:
                exit_results = await self._perform_exit_monitoring(
                    market_data,
                    strategy_data,
                )
                self.tactician_results["exit_monitoring"] = exit_results

            # Perform position monitoring
            if self.tactician_config.get("enable_position_monitoring", False):
                position_results = await self._perform_position_monitoring(
                    market_data,
                    strategy_data,
                )
                self.tactician_results["position_monitoring"] = position_results

            # Perform risk monitoring
            if self.tactician_config.get("enable_risk_monitoring", True):
                risk_results = await self._perform_risk_monitoring(
                    market_data,
                    strategy_data,
                )
                self.tactician_results["risk_monitoring"] = risk_results

            # Store tactician results
            await self._store_tactician_results()

            self.is_tactician_active = False
            self.logger.info("✅ Tactician execution completed successfully")
            return True

        except Exception as e:
            self.logger.error(error(f"Error executing tactician: {e}"))
            self.is_tactician_active = False
            return False

    @handles_errors(fallback=False)
    def _validate_tactician_inputs(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> bool:
        """Validate tactician inputs.

        Args:
            market_data: Market data dictionary
            strategy_data: Strategy data dictionary

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required market data fields
            required_market_fields = ["symbol", "price", "volume", "timestamp"]
            for field in required_market_fields:
                if field not in market_data:
                    self.logger.error(
                        missing(f"Missing required market data field: {field}")
                    )
                    return False

            # Check required strategy data fields
            required_strategy_fields = ["signal", "position_size"]
            for field in required_strategy_fields:
                if field not in strategy_data:
                    self.logger.error(
                        missing(f"Missing required strategy data field: {field}")
                    )
                    return False

            # Validate data types
            if not isinstance(market_data["price"], (int, float)):
                self.logger.error(invalid("Invalid price data type"))
                return False

            if not isinstance(strategy_data["position_size"], (int, float)):
                self.logger.error(invalid("Invalid position size data type"))
                return False

            return True

        except Exception as e:
            self.logger.error(error(f"Error validating tactician inputs: {e}"))
            return False

    @handles_errors(fallback=None)
    async def _perform_entry_monitoring(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform entry monitoring.

        Args:
            market_data: Market data dictionary
            strategy_data: Strategy data dictionary

        Returns:
            Dict[str, Any]: Entry monitoring results
        """
        try:
            results = {}

            # Analyze price action
            if self.entry_monitoring_strategies.get("price_action", False):
                results["price_action"] = self._analyze_price_action(
                    market_data,
                    strategy_data,
                )

            # Analyze volume
            if self.entry_monitoring_strategies.get("volume_analysis", False):
                results["volume_analysis"] = self._analyze_volume(
                    market_data,
                    strategy_data,
                )

            # Analyze momentum indicators
            if self.entry_monitoring_strategies.get("momentum_indicators", False):
                results["momentum_indicators"] = self._analyze_momentum_indicators(
                    market_data,
                    strategy_data,
                )

            # Analyze support resistance
            if self.entry_monitoring_strategies.get("support_resistance", False):
                results["support_resistance"] = self._analyze_support_resistance(
                    market_data,
                    strategy_data,
                )

            self.logger.info("Entry monitoring completed")
            return results

        except Exception as e:
            self.logger.error(error(f"Error performing entry monitoring: {e}"))
            return {}

    @handles_errors(fallback=None)
    async def _perform_exit_monitoring(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform exit monitoring.

        Args:
            market_data: Market data dictionary
            strategy_data: Strategy data dictionary

        Returns:
            Dict[str, Any]: Exit monitoring results
        """
        try:
            results = {}

            # Track stop loss
            if self.exit_monitoring_strategies.get("stop_loss_tracking", False):
                results["stop_loss_tracking"] = self._track_stop_loss(
                    market_data,
                    strategy_data,
                )

            # Track take profit
            if self.exit_monitoring_strategies.get("take_profit_tracking", False):
                results["take_profit_tracking"] = self._track_take_profit(
                    market_data,
                    strategy_data,
                )

            # Track trailing stop
            if self.exit_monitoring_strategies.get("trailing_stop", False):
                results["trailing_stop"] = self._track_trailing_stop(
                    market_data,
                    strategy_data,
                )

            # Track time based exit
            if self.exit_monitoring_strategies.get("time_based_exit", False):
                results["time_based_exit"] = self._track_time_based_exit(
                    market_data,
                    strategy_data,
                )

            self.logger.info("Exit monitoring completed")
            return results

        except Exception as e:
            self.logger.error(error(f"Error performing exit monitoring: {e}"))
            return {}

    @handles_errors(fallback=None)
    async def _perform_position_monitoring(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform position monitoring.

        Args:
            market_data: Market data dictionary
            strategy_data: Strategy data dictionary

        Returns:
            Dict[str, Any]: Position monitoring results
        """
        try:
            results = {}

            # Track position size
            if self.position_monitoring_strategies.get("position_size_tracking", False):
                results["position_size_tracking"] = self._track_position_size(
                    market_data,
                    strategy_data,
                )

            # Monitor exposure limits
            if self.position_monitoring_strategies.get("exposure_limits", False):
                results["exposure_limits"] = self._monitor_exposure_limits(
                    market_data,
                    strategy_data,
                )

            # Monitor correlation
            if self.position_monitoring_strategies.get("correlation_monitoring", False):
                results["correlation_monitoring"] = self._monitor_correlation(
                    market_data,
                    strategy_data,
                )

            # Monitor concentration limits
            if self.position_monitoring_strategies.get("concentration_limits", False):
                results["concentration_limits"] = self._monitor_concentration_limits(
                    market_data,
                    strategy_data,
                )

            self.logger.info("Position monitoring completed")
            return results

        except Exception as e:
            self.logger.error(error(f"Error performing position monitoring: {e}"))
            return {}

    @handles_errors(fallback=None)
    async def _perform_risk_monitoring(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform risk monitoring.

        Args:
            market_data: Market data dictionary
            strategy_data: Strategy data dictionary

        Returns:
            Dict[str, Any]: Risk monitoring results
        """
        try:
            results = {}

            # Monitor VaR
            if self.risk_monitoring_strategies.get("var_monitoring", False):
                results["var_monitoring"] = self._monitor_var(
                    market_data,
                    strategy_data,
                )

            # Track drawdown
            if self.risk_monitoring_strategies.get("drawdown_tracking", False):
                results["drawdown_tracking"] = self._track_drawdown(
                    market_data,
                    strategy_data,
                )

            # Monitor volatility
            if self.risk_monitoring_strategies.get("volatility_monitoring", False):
                results["volatility_monitoring"] = self._monitor_volatility(
                    market_data,
                    strategy_data,
                )

            # Perform stress testing
            if self.risk_monitoring_strategies.get("stress_testing", False):
                results["stress_testing"] = self._perform_stress_testing(
                    market_data,
                    strategy_data,
                )

            self.logger.info("Risk monitoring completed")
            return results

        except Exception as e:
            self.logger.error(error(f"Error performing risk monitoring: {e}"))
            return {}

    # Entry monitoring analysis methods

    def _analyze_price_action(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze price action for entry signals."""
        try:
            # Simulate price action analysis
            return {
                "trend_direction": "bullish",
                "support_level": 100.0,
                "resistance_level": 105.0,
                "entry_signal": True,
            }
        except Exception as e:
            self.logger.error(error(f"Error analyzing price action: {e}"))
            return {}

    def _analyze_volume(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze volume for entry signals."""
        try:
            # Simulate volume analysis
            return {
                "volume_trend": "increasing",
                "volume_ratio": 1.5,
                "volume_signal": True,
            }
        except Exception as e:
            self.logger.error(error(f"Error analyzing volume: {e}"))
            return {}

    def _analyze_momentum_indicators(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze momentum indicators for entry signals."""
        try:
            # Simulate momentum analysis
            return {
                "rsi_signal": "oversold",
                "macd_signal": "bullish",
                "momentum_signal": True,
            }
        except Exception as e:
            self.logger.error(error(f"Error analyzing momentum indicators: {e}"))
            return {}

    def _analyze_support_resistance(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze support and resistance levels."""
        try:
            # Simulate support resistance analysis
            return {
                "near_support": True,
                "support_strength": 0.8,
                "resistance_distance": 0.05,
            }
        except Exception as e:
            self.logger.error(error(f"Error analyzing support resistance: {e}"))
            return {}

    # Exit monitoring tracking methods

    def _track_stop_loss(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Track stop loss levels."""
        try:
            # Simulate stop loss tracking
            return {
                "stop_loss_triggered": False,
                "stop_loss_distance": 0.02,
                "stop_loss_level": 98.0,
            }
        except Exception as e:
            self.logger.error(error(f"Error tracking stop loss: {e}"))
            return {}

    def _track_take_profit(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Track take profit levels."""
        try:
            # Simulate take profit tracking
            return {
                "take_profit_triggered": False,
                "take_profit_distance": 0.04,
                "take_profit_level": 104.0,
            }
        except Exception as e:
            self.logger.error(error(f"Error tracking take profit: {e}"))
            return {}

    def _track_trailing_stop(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Track trailing stop levels."""
        try:
            # Simulate trailing stop tracking
            return {
                "trailing_stop_triggered": False,
                "trailing_stop_distance": 0.015,
                "trailing_stop_level": 98.5,
            }
        except Exception as e:
            self.logger.error(error(f"Error tracking trailing stop: {e}"))
            return {}

    def _track_time_based_exit(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Track time based exit conditions."""
        try:
            # Simulate time based exit tracking
            return {
                "time_exit_triggered": False,
                "time_in_position": 3600,  # seconds
                "max_time_limit": 7200,  # seconds
            }
        except Exception as e:
            self.logger.error(error(f"Error tracking time based exit: {e}"))
            return {}

    # Position monitoring methods

    def _track_position_size(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Track position size."""
        try:
            # Simulate position size tracking
            return {
                "current_position_size": 0.1,
                "max_position_size": 0.25,
                "position_size_ok": True,
            }
        except Exception as e:
            self.logger.error(error(f"Error tracking position size: {e}"))
            return {}

    def _monitor_exposure_limits(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Monitor exposure limits."""
        try:
            # Simulate exposure monitoring
            return {
                "total_exposure": 0.3,
                "max_exposure": 0.5,
                "exposure_ok": True,
            }
        except Exception as e:
            self.logger.error(error(f"Error monitoring exposure limits: {e}"))
            return {}

    def _monitor_correlation(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Monitor correlation between positions."""
        try:
            # Simulate correlation monitoring
            return {
                "avg_correlation": 0.2,
                "max_correlation": 0.7,
                "correlation_ok": True,
            }
        except Exception as e:
            self.logger.error(error(f"Error monitoring correlation: {e}"))
            return {}

    def _monitor_concentration_limits(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Monitor concentration limits."""
        try:
            # Simulate concentration monitoring
            return {
                "largest_position": 0.15,
                "max_concentration": 0.2,
                "concentration_ok": True,
            }
        except Exception as e:
            self.logger.error(error(f"Error monitoring concentration limits: {e}"))
            return {}

    # Risk monitoring methods

    def _monitor_var(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Monitor Value at Risk."""
        try:
            # Simulate VaR monitoring
            return {
                "current_var": 0.025,
                "max_var": 0.05,
                "var_ok": True,
            }
        except Exception as e:
            self.logger.error(error(f"Error monitoring VaR: {e}"))
            return {}

    def _track_drawdown(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Track drawdown."""
        try:
            # Simulate drawdown tracking
            return {
                "current_drawdown": 0.08,
                "max_drawdown": 0.15,
                "drawdown_ok": True,
            }
        except Exception as e:
            self.logger.error(error(f"Error tracking drawdown: {e}"))
            return {}

    def _monitor_volatility(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Monitor volatility."""
        try:
            # Simulate volatility monitoring
            return {
                "current_volatility": 0.18,
                "target_volatility": 0.15,
                "volatility_ok": True,
            }
        except Exception as e:
            self.logger.error(error(f"Error monitoring volatility: {e}"))
            return {}

    def _perform_stress_testing(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform stress testing."""
        try:
            # Simulate stress testing
            return {
                "stress_test_passed": True,
                "worst_case_loss": 0.12,
                "stress_test_score": 0.85,
            }
        except Exception as e:
            self.logger.error(error(f"Error performing stress testing: {e}"))
            return {}

    @handles_errors(fallback=None)
    async def _store_tactician_results(self) -> None:
        """Store tactician results."""
        try:
            # Add timestamp
            self.tactician_results["timestamp"] = datetime.now().isoformat()

            # Add to history
            self.tactician_history.append(self.tactician_results.copy())

            # Limit history size
            if len(self.tactician_history) > self.max_tactician_history:
                self.tactician_history.pop(0)

            self.logger.info("Tactician results stored successfully")

        except Exception as e:
            self.logger.error(error(f"Error storing tactician results: {e}"))

    @handles_errors(fallback=None)
    def get_tactician_results(
        self,
        tactician_type: str | None = None,
    ) -> dict[str, Any]:
        """Get tactician results.

        Args:
            tactician_type: Optional tactician type filter

        Returns:
            Dict[str, Any]: Tactician results
        """
        try:
            if tactician_type:
                return self.tactician_results.get(tactician_type, {})
            return self.tactician_results.copy()

        except Exception as e:
            self.logger.error(error(f"Error getting tactician results: {e}"))
            return {}

    @handles_errors(fallback=None)
    def get_tactician_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get tactician history.

        Args:
            limit: Optional limit on number of records

        Returns:
            List[Dict[str, Any]]: Tactician history
        """
        try:
            history = self.tactician_history.copy()

            if limit:
                history = history[-limit:]

            return history

        except Exception as e:
            self.logger.error(error(f"Error getting tactician history: {e}"))
            return []

    def get_tactician_status(self) -> dict[str, Any]:
        """Get tactician status information.

        Returns:
            Dict[str, Any]: Tactician status
        """
        return {
            "is_tactician_active": self.is_tactician_active,
            "tactician_interval": self.tactician_interval,
            "max_tactician_history": self.max_tactician_history,
            "enable_entry_monitoring": self.enable_entry_monitoring,
            "enable_exit_monitoring": self.enable_exit_monitoring,
            "enable_position_monitoring": self.tactician_config.get(
                "enable_position_monitoring",
                False,
            ),
            "enable_risk_monitoring": self.tactician_config.get(
                "enable_risk_monitoring",
                True,
            ),
            "tactician_history_count": len(self.tactician_history),
        }

    @handles_errors(fallback=None)
    async def stop(self) -> None:
        """Stop the modular tactician."""
        self.logger.info("🛑 Stopping Modular Tactician...")

        try:
            # Stop tactician
            self.is_tactician_active = False

            # Clear results
            self.tactician_results.clear()

            # Clear history
            self.tactician_history.clear()

            self.logger.info("✅ Modular Tactician stopped successfully")

        except Exception as e:
            self.logger.error(error(f"Error stopping modular tactician: {e}"))


# Global modular tactician instance
modular_tactician: ModularTactician | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="modular tactician setup",
)
async def setup_modular_tactician(
    config: dict[str, Any] | None = None,
) -> ModularTactician | None:
    """Setup global modular tactician.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[ModularTactician]: Global modular tactician instance
    """
    try:
        global modular_tactician

        if config is None:
            config = {
                "modular_tactician": {
                    "tactician_interval": 5,
                    "max_tactician_history": 100,
                    "enable_entry_monitoring": True,
                    "enable_exit_monitoring": True,
                    "enable_position_monitoring": False,
                    "enable_risk_monitoring": True,
                },
            }

        # Create modular tactician
        modular_tactician = ModularTactician(config)

        # Initialize modular tactician
        success = await modular_tactician.initialize()
        if success:
            return modular_tactician
        return None

    except Exception as e:
        print(f"Error setting up modular tactician: {e}")
        return None
